"""ch09 eval_runs 落表:run_id 幂等;metrics 契约;triggered_by 透传。"""

import asyncio
import json
import sys

import httpx

from app.jobs.runner import JobRunner
from app.main import AppRuntime, create_app
from app.models import EvalRun
from app.services import eval_runs
from tests.conftest import (FakeStreamModel, UserBoundMemoryStore, make_runtime,
                            make_settings)
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401
from app.tools.business import MOCK_TOOLS, RetrievalTrace, TurnToolset

# 照 rag_eval.json 真实契约(validate_rag_eval 钉的形状):generation 策略臂平铺,
# 不嵌 per_strategy;retrieval.hybrid_rerank.overall 带 recall_at_10(Task 10)
REPORT = {"meta": {"run_id": "20261009T030000Z-abc", "test_cases": 150,
                   "corpus_mode": "knowledge_docs_baseline",
                   "corpus_version": "c" * 64, "dataset_version": "d" * 64,
                   "evidence_confidence_version": "v1"},
          "retrieval": {"hybrid_rerank": {"overall": {
              "mrr": 0.71, "recall_at_10": 0.82, "evidence_coverage": 0.9,
              "sr10": 0.8}}},
          "generation": {"online_strategy": "hybrid_rerank",
                         "hybrid_rerank": {"faithful_rate": 0.9}},
          "gates": {"passed": True}}


def test_record_run_inserts_and_dedupes(db_session_factory):
    assert eval_runs.record_run(db_session_factory, REPORT, "手动") is True
    assert eval_runs.record_run(db_session_factory, REPORT, "手动") is False
    with db_session_factory() as s:
        rows = s.query(EvalRun).all()
        assert len(rows) == 1
        m = rows[0].metrics
        assert m["recall_at_10"] == 0.82 and m["mrr"] == 0.71
        assert m["faithfulness"] == 0.9 and m["quality_passed"] is True
        assert m["evidence_confidence_version"] == "v1"
        assert m["online_strategy"] == "hybrid_rerank"
        assert rows[0].triggered_by == "手动"
        assert rows[0].dataset_size == 150


def test_record_run_missing_metrics_null_not_zero(db_session_factory):
    bad = {"meta": {**REPORT["meta"], "run_id": "r2"},
           "retrieval": {"hybrid_rerank": {"overall": {}}},
           "generation": {"online_strategy": "hybrid_rerank",
                          "hybrid_rerank": {}},
           "gates": {}}
    assert eval_runs.record_run(db_session_factory, bad, "定时") is True
    with db_session_factory() as s:
        row = s.query(EvalRun).filter_by(run_id="r2").first()
        assert row.metrics["mrr"] is None and row.metrics["quality_passed"] is False
        assert row.metrics["recall_at_10"] is None
        assert row.metrics["faithfulness"] is None


def test_list_runs_ascending_shape(db_session_factory):
    eval_runs.record_run(db_session_factory, REPORT, "手动")
    eval_runs.record_run(db_session_factory,
                         {**REPORT, "meta": {**REPORT["meta"], "run_id": "r2"}},
                         "定时")
    out = eval_runs.list_runs(db_session_factory)
    assert [r["run_id"] for r in out] == ["20261009T030000Z-abc", "r2"]  # 升序
    assert set(out[0]) == {"run_id", "triggered_by", "dataset_size", "corpus_mode",
                           "corpus_version", "dataset_version", "metrics", "created_at"}
    assert out[0]["triggered_by"] == "手动" and out[1]["triggered_by"] == "定时"
    assert out[0]["metrics"]["recall_at_10"] == 0.82
    assert eval_runs.list_runs(db_session_factory, limit=1) == [out[1]]  # 最近 1 条


# --- JobRunner on_report_published 回调(spec §5.5:仅新 run_id 时回调一次) ---

async def _wait_terminal(runner, name, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        if runner.status(name)["status"] != "running":
            break
    return runner.status(name)


def _write_report_cmd(rp, report, exit_code=0):
    """模拟评估脚本的命令:写新报告 + 按 exit_code 退出(门槛退出 1 也算正常完成)。"""
    return [sys.executable, "-c",
            "import json,sys,pathlib;"
            f"pathlib.Path({str(rp)!r}).write_text(json.dumps({report!r}));"
            f"sys.exit({exit_code})"]


async def test_runner_callback_on_new_report(tmp_path):
    rp = tmp_path / "rag_eval.json"
    rp.write_text(json.dumps({**REPORT, "meta": {**REPORT["meta"], "run_id": "old"}}),
                  encoding="utf-8")
    loader = lambda: json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    calls = []
    r = JobRunner(tmp_path, report_loader=loader,
                  on_report_published=lambda report, by: calls.append((report, by)))
    r.register("t", _write_report_cmd(rp, REPORT, exit_code=1))
    assert await r.run("t", triggered_by="手动") is True
    st = await _wait_terminal(r, "t")
    assert st["status"] == "ok" and st["triggered_by"] == "手动"
    assert calls == [(REPORT, "手动")]   # 恰一次,报告与触发源透传


async def test_runner_no_callback_without_new_report(tmp_path):
    rp = tmp_path / "rag_eval.json"
    rp.write_text(json.dumps(REPORT), encoding="utf-8")
    loader = lambda: json.loads(rp.read_text(encoding="utf-8"))
    calls = []
    r = JobRunner(tmp_path, report_loader=loader,
                  on_report_published=lambda report, by: calls.append((report, by)))
    r.register("t", [sys.executable, "-c", "import sys; sys.exit(1)"])  # 不更新报告
    await r.run("t", triggered_by="定时")
    st = await _wait_terminal(r, "t")
    assert st["status"] == "failed" and st["triggered_by"] == "定时"
    assert calls == []


async def test_runner_callback_failure_keeps_ok(tmp_path):
    """回调抛错只记日志,不翻作业状态(观测不拦主路)。"""
    rp = tmp_path / "rag_eval.json"
    rp.write_text(json.dumps({**REPORT, "meta": {**REPORT["meta"], "run_id": "old"}}),
                  encoding="utf-8")
    loader = lambda: json.loads(rp.read_text(encoding="utf-8"))
    def boom(report, by):
        raise RuntimeError("观测不拦主路")
    r = JobRunner(tmp_path, report_loader=loader, on_report_published=boom)
    r.register("t", _write_report_cmd(rp, REPORT))
    await r.run("t")                    # 缺省 triggered_by=手动
    st = await _wait_terminal(r, "t")
    assert st["status"] == "ok" and st["report_run_id"] == REPORT["meta"]["run_id"]
    assert st["triggered_by"] == "手动"


# --- GET /api/eval-runs ---

def _app(db_session_factory):
    runtime = AppRuntime(
        store=UserBoundMemoryStore(1000, 100, 8000),
        toolset_factory=lambda sid: TurnToolset(list(MOCK_TOOLS), RetrievalTrace()),
        session_factory=db_session_factory)
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=runtime)


async def test_eval_runs_endpoint_ascending(db_session_factory):
    eval_runs.record_run(db_session_factory, REPORT, "手动")
    eval_runs.record_run(db_session_factory,
                         {**REPORT, "meta": {**REPORT["meta"], "run_id": "r2"}},
                         "定时")
    app = _app(db_session_factory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        resp = await client.get("/api/eval-runs")
        assert resp.status_code == 200
        body = resp.json()
        assert [r["run_id"] for r in body["runs"]] == ["20261009T030000Z-abc", "r2"]
        run = body["runs"][0]
        assert run["triggered_by"] == "手动" and run["dataset_size"] == 150
        assert run["metrics"]["recall_at_10"] == 0.82


async def test_eval_runs_endpoint_no_db_503(tmp_path):
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))   # 无 DB 装配
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        resp = await client.get("/api/eval-runs")
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "rag_eval_unavailable"
