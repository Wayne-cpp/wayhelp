"""ch09 审核读路径与飞轮 API。

服务级:list/detail/reject(仅待审)/retry(仅 failed,CAS 计数归零)/failures。
路由级:错误契约 {"error":{"code","message"}}(404/409/503)、202 通知与 worker 降级。
approve 写回整个在 Task 15,本文件不测不暴露。
"""

import httpx
import pytest

from app.main import AppRuntime, create_app
from app.models import LowConfidenceQuestion, ReviewQueue
from app.services import review_service
from app.services.review_service import ReviewConflictError, ReviewNotFoundError
from tests.conftest import FakeStreamModel, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _seed(sf):
    with sf() as s:
        rq = ReviewQueue(normalized_question="如何开发票?", occurrence_count=2,
                         ai_suggested_answer="示例")
        s.add(rq)
        s.flush()
        lcq = LowConfidenceQuestion(raw_question="发票咋开", source="user_feedback",
                                    resolved_question="如何开发票?",
                                    retrieved_chunks=[{"chunk_id": 1, "score": 0.3}],
                                    process_status="processed",
                                    matched_review_id=rq.id)
        s.add(lcq)
        bad = LowConfidenceQuestion(raw_question="坏问题", source="self_check",
                                    process_status="failed", attempt_count=5,
                                    last_error="ValueError: parse")
        s.add(bad)
        s.commit()
        return rq.id, lcq.id, bad.id


# ─── 服务级 ──────────────────────────────────────────────────────────────────


def test_list_and_detail(db_session_factory):
    rid, lcq_id, _ = _seed(db_session_factory)
    out = review_service.list_reviews(db_session_factory, "待审", 1)
    assert out["total"] == 1 and out["items"][0]["occurrence_count"] == 2
    detail = review_service.get_review_detail(db_session_factory, rid)
    assert detail["sources"][0]["raw_question"] == "发票咋开"
    assert detail["sources"][0]["retrieved_chunks"] == [{"chunk_id": 1, "score": 0.3}]


def test_reject_only_from_pending(db_session_factory):
    rid, _, _ = _seed(db_session_factory)
    out = review_service.reject(db_session_factory, rid)
    assert out["review_status"] == "驳回"
    with pytest.raises(ReviewConflictError):
        review_service.reject(db_session_factory, rid)


def test_retry_failed_question_cas(db_session_factory):
    _, _, bad_id = _seed(db_session_factory)
    review_service.retry_failed_question(db_session_factory, bad_id)
    with db_session_factory() as s:
        row = s.get(LowConfidenceQuestion, bad_id)
        assert row.process_status == "pending"
        assert row.attempt_count == 0 and row.next_attempt_at is None
        assert row.last_error is None
    with pytest.raises(ReviewConflictError):   # 非 failed 重复重试 409
        review_service.retry_failed_question(db_session_factory, bad_id)


def test_list_failed_questions(db_session_factory):
    _, _, bad_id = _seed(db_session_factory)
    out = review_service.list_failed_questions(db_session_factory, 1)
    assert out["total"] == 1
    item = out["items"][0]
    assert item["id"] == bad_id and item["raw_question"] == "坏问题"
    assert item["source"] == "self_check" and item["attempt_count"] == 5
    assert item["last_error"] == "ValueError: parse" and item["created_at"]


def test_not_found_raises(db_session_factory):
    _seed(db_session_factory)
    with pytest.raises(ReviewNotFoundError):
        review_service.get_review_detail(db_session_factory, 99999)
    with pytest.raises(ReviewNotFoundError):
        review_service.reject(db_session_factory, 99999)
    with pytest.raises(ReviewNotFoundError):
        review_service.retry_failed_question(db_session_factory, 99999)


# ─── 路由级 ──────────────────────────────────────────────────────────────────


def make_review_app(sf=None):
    rt = make_runtime(tools=[])
    runtime = AppRuntime(store=rt.store, toolset_factory=rt.toolset_factory,
                         session_factory=sf, knowledge_state=rt.knowledge_state)
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=runtime)


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_api_review_list_detail_reject(db_session_factory):
    rid, _, _ = _seed(db_session_factory)
    app = make_review_app(db_session_factory)
    async with _client(app) as client:
        resp = await client.get("/api/review", params={"status": "待审"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1 and body["items"][0]["id"] == rid
        assert body["items"][0]["normalized_question"] == "如何开发票?"
        assert body["items"][0]["ai_suggested_answer"] == "示例"
        assert body["items"][0]["review_status"] == "待审"
        assert body["items"][0]["last_write_error"] is None
        assert body["items"][0]["created_at"]

        resp = await client.get(f"/api/review/{rid}/detail")
        assert resp.status_code == 200
        src = resp.json()["sources"][0]
        assert src["raw_question"] == "发票咋开" and src["source"] == "user_feedback"
        assert src["retrieved_chunks"] == [{"chunk_id": 1, "score": 0.3}]

        resp = await client.post(f"/api/review/{rid}/reject")
        assert resp.status_code == 200 and resp.json() == {"review_status": "驳回"}
        resp = await client.post(f"/api/review/{rid}/reject")   # 驳回项再驳 409
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "review_conflict"
        resp = await client.get("/api/review", params={"status": "驳回"})
        assert resp.json()["total"] == 1


async def test_api_review_not_found(db_session_factory):
    _seed(db_session_factory)
    app = make_review_app(db_session_factory)
    async with _client(app) as client:
        resp = await client.get("/api/review/99999/detail")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "review_not_found"
        assert "message" in resp.json()["error"]


async def test_api_flywheel_run_202_and_degrades_without_worker(db_session_factory):
    app = make_review_app(db_session_factory)
    async with _client(app) as client:
        resp = await client.post("/api/flywheel/run")
        assert resp.status_code == 202 and resp.json() == {"notified": True}
        app.state.flywheel_worker = None   # worker 不存在时安全降级,仍 202
        resp = await client.post("/api/flywheel/run")
        assert resp.status_code == 202 and resp.json() == {"notified": False}


async def test_api_failures_and_retry(db_session_factory):
    _, _, bad_id = _seed(db_session_factory)
    app = make_review_app(db_session_factory)
    async with _client(app) as client:
        resp = await client.get("/api/flywheel/failures")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1 and body["items"][0]["id"] == bad_id
        assert body["items"][0]["attempt_count"] == 5
        assert body["items"][0]["last_error"] == "ValueError: parse"

        resp = await client.post(f"/api/flywheel/questions/{bad_id}/retry")
        assert resp.status_code == 202
        with db_session_factory() as s:
            row = s.get(LowConfidenceQuestion, bad_id)
            assert row.process_status == "pending" and row.attempt_count == 0
            assert row.next_attempt_at is None and row.last_error is None

        resp = await client.post(f"/api/flywheel/questions/{bad_id}/retry")
        assert resp.status_code == 409   # 非 failed 重复重试
        assert resp.json()["error"]["code"] == "review_conflict"
        resp = await client.get("/api/flywheel/failures")
        assert resp.json()["total"] == 0


async def test_api_unavailable_without_session_factory():
    app = make_review_app()   # runtime 不带 session_factory
    async with _client(app) as client:
        resp = await client.get("/api/review")
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "review_unavailable"
