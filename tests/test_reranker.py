import httpx
import pytest

from app.knowledge.reranker import SiliconFlowReranker
from tests.conftest import make_settings


def _transport(handler):
    return httpx.MockTransport(handler)


def test_ranking_parsed():
    def handler(request):
        body = __import__("json").loads(request.content)
        assert body["model"] == "BAAI/bge-reranker-v2-m3"
        assert body["query"] == "猫砂盆容量"
        assert len(body["documents"]) == 3 and body["top_n"] == 2
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]})
    r = SiliconFlowReranker(make_settings(rerank_api_key="k"),
                            client=httpx.Client(transport=_transport(handler)))
    out = r.rerank("猫砂盆容量", ["d0", "d1", "d2"], top_n=2)
    assert out.ok and out.ranking == [(2, 0.9), (0, 0.3)]


def test_retry_exhausted_degrades():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})
    r = SiliconFlowReranker(make_settings(rerank_api_key="k"),
                            client=httpx.Client(transport=_transport(handler)))
    out = r.rerank("q", ["d"], top_n=1)
    assert out.ok is False and out.ranking == [] and out.note
    assert calls["n"] == 3  # 1 + max_retries(2),退避 0.5s/1s


def test_auth_error_no_retry():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})
    r = SiliconFlowReranker(make_settings(rerank_api_key="k"),
                            client=httpx.Client(transport=_transport(handler)))
    out = r.rerank("q", ["d"], top_n=1)
    assert out.ok is False and calls["n"] == 1


def test_connect_error_retries_then_degrades(monkeypatch):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("refused")
    monkeypatch.setattr("time.sleep", lambda s: None)  # 退避不等真实时间
    r = SiliconFlowReranker(make_settings(rerank_api_key="k"),
                            client=httpx.Client(transport=_transport(handler)))
    out = r.rerank("q", ["d"], top_n=1)
    assert out.ok is False and calls["n"] == 3


def test_deadline_exhausted_short_circuits():
    import time as _t
    r = SiliconFlowReranker(make_settings(rerank_api_key="k"),
                            client=httpx.Client(transport=_transport(
                                lambda req: httpx.Response(200, json={"results": []}))))
    out = r.rerank("q", ["d"], top_n=1, deadline=_t.monotonic() - 1)
    assert out.ok is False and "budget" in (out.note or "")
