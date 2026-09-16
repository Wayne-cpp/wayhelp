"""/api/rag-eval 路由:report 空态/损坏、台账分页页签与两口径、PATCH 校验。"""

import json

import httpx

from app.main import AppRuntime, create_app
from app.models import FaithCase
from tests.conftest import FakeStreamModel, UserBoundMemoryStore, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401
from tests.test_rag_eval_service import _report, _seed
from app.tools.business import MOCK_TOOLS, RetrievalTrace, TurnToolset


def _app(db_session_factory, tmp_path, report=None, corrupt=False):
    runtime = AppRuntime(
        store=UserBoundMemoryStore(1000, 100, 8000),
        toolset_factory=lambda sid: TurnToolset(list(MOCK_TOOLS), RetrievalTrace()),
        session_factory=db_session_factory)
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=runtime)
    path = tmp_path / "rag_eval.json"
    if corrupt:
        path.write_text("{oops", encoding="utf-8")
    elif report is not None:
        path.write_text(json.dumps(report), encoding="utf-8")
    app.state.rag_eval_report_path = path   # 测试用临时报告,不碰仓库真实产物
    return app


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_report_empty_and_ok(db_session_factory, tmp_path):
    app = _app(db_session_factory, tmp_path)
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/report")
        assert resp.status_code == 200 and resp.json() == {"empty": True}
    app = _app(db_session_factory, tmp_path, report=_report())
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/report")
        assert resp.status_code == 200 and resp.json()["meta"]["run_id"] == "r1"


async def test_report_corrupt_502(db_session_factory, tmp_path):
    app = _app(db_session_factory, tmp_path, corrupt=True)
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/report")
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "rag_eval_report_corrupt"
        # 报告损坏不阻断台账:按「无当前报告」口径返回
        resp = await client.get("/api/rag-eval/faith-cases")
        assert resp.status_code == 200
        assert resp.json()["stats"]["judge_rate"] is None


async def test_faith_cases_list_and_stats(db_session_factory, tmp_path):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2", status="已解决")
        _seed(s, "B2", run_id="r1")
        s.commit()
    app = _app(sf, tmp_path,
               report=_report(run_id="r1", fabricated=2, judged=4,
                              case_ids=("A2", "B2")))
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/faith-cases")
        body = resp.json()
        assert resp.status_code == 200 and body["total"] == 1   # 缺省未解决页签
        assert body["items"][0]["case_id"] == "B2"
        st = body["stats"]
        assert st["judge_rate"] == 0.5 and st["confirmed_rate"] == 0.5
        assert st["current_run_missing"] == 0
        assert st["ledger"]["total"] == 2 and st["ledger"]["已解决"] == 1
        resp = await client.get("/api/rag-eval/faith-cases",
                                params={"status": "已解决", "page": 1, "size": 1})
        assert resp.json()["total"] == 1
        assert (await client.get("/api/rag-eval/faith-cases",
                                 params={"status": "无效"})).status_code == 422
        assert (await client.get("/api/rag-eval/faith-cases",
                                 params={"size": 51})).status_code == 422


async def test_patch(db_session_factory, tmp_path):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2")
        s.commit()
        rid = s.query(FaithCase).one().id
    app = _app(sf, tmp_path)
    async with await _client(app) as client:
        resp = await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                  json={"status": "已解决", "resolution": " 已补文档 "})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "已解决" and body["resolution"] == "已补文档"
        assert body["resolved_at"] is not None
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "无需解决",
                                         "resolution": "误报"})).status_code == 200
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "未解决",
                                         "resolution": "x"})).status_code == 422
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "已解决",
                                         "resolution": "   "})).status_code == 422
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "已解决",
                                         "resolution": "x" * 301})).status_code == 422
        assert (await client.patch("/api/rag-eval/faith-cases/999999",
                                   json={"status": "已解决",
                                         "resolution": "x"})).status_code == 404
