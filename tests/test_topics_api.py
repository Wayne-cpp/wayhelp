"""topics API 契约:状态码与错误体(spec §7/§8)。"""

import httpx

from app.main import create_app
from app.services import topic_classify
from tests.conftest import FakeStreamModel, make_runtime, make_settings


def _app():
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=make_runtime(tools=[]))


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_classify_ok(monkeypatch):
    monkeypatch.setattr(topic_classify, "run_classification",
                        lambda *a, **k: {"classifier_version": "v",
                                         "classified_questions": 2,
                                         "skipped_questions": 0, "written_rows": 3})
    async with await _client(_app()) as c:
        resp = await c.post("/api/topics/classify")
    assert resp.status_code == 200 and resp.json()["written_rows"] == 3


async def test_classify_busy_409(monkeypatch):
    def busy(*a, **k):
        raise topic_classify.TopicJobBusyError("忙")
    monkeypatch.setattr(topic_classify, "run_classification", busy)
    async with await _client(_app()) as c:
        resp = await c.post("/api/topics/classify")
    assert resp.status_code == 409 and resp.json()["error"]["code"] == "job_busy"


async def test_classify_inference_down_503(monkeypatch):
    def down(*a, **k):
        raise topic_classify.TopicInferenceError("不可达")
    monkeypatch.setattr(topic_classify, "run_classification", down)
    async with await _client(_app()) as c:
        resp = await c.post("/api/topics/classify")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "inference_unavailable"


async def test_distribution_conflict_409(monkeypatch):
    def conflict(*a, **k):
        raise topic_classify.TopicVersionConflictError("混杂")
    monkeypatch.setattr(topic_classify, "get_distribution", conflict)
    async with await _client(_app()) as c:
        resp = await c.get("/api/topics/distribution")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "classification_version_conflict"


async def test_distribution_ok(monkeypatch):
    monkeypatch.setattr(topic_classify, "get_distribution",
                        lambda *a, **k: {"total": 0, "classified": 0,
                                         "unclassified": 0, "counts": [],
                                         "classifier_version": None,
                                         "classifier_metadata": None,
                                         "classified_at": None, "stale": False})
    async with await _client(_app()) as c:
        resp = await c.get("/api/topics/distribution")
    assert resp.status_code == 200 and resp.json()["stale"] is False
