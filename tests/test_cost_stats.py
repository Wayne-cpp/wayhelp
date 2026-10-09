"""ch09 成本统计:按意图聚合/去重/缺失可见/未启用 503/不可达 502/上限 incomplete。"""
import httpx
import pytest
from httpx import MockTransport, Response

from app.services import cost_stats
from app.services.cost_stats import CostStatsError, cost_by_intent


def _settings(**kw):
    from app.config import Settings
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d",
                langfuse_enabled=True, langfuse_host="http://lf:3000",
                langfuse_public_key="pk", langfuse_secret_key="sk",
                model_input_price_per_mtok=1.0, model_output_price_per_mtok=2.0)
    base.update(kw)
    return Settings(**base)


TRACES = {"data": [
    {"id": "t1", "metadata": {"intent": "退款退货"}},
    {"id": "t2", "metadata": {"intent": "闲聊"}},
    {"id": "t3", "metadata": {}},
], "meta": {"totalPages": 1}}

OBS = {"data": [
    {"id": "o1", "type": "GENERATION",
     "usageDetails": {"input": 100, "output": 50},
     "costDetails": {"input": 0.0001, "output": 0.0001}},
    {"id": "o1", "type": "GENERATION",          # 重复行:按 observation_id 全局去重
     "usageDetails": {"input": 100, "output": 50},
     "costDetails": {"input": 0.0001, "output": 0.0001}},
    {"id": "o2", "type": "GENERATION", "usageDetails": {}, "costDetails": None},
], "meta": {"totalPages": 1}}


def _client(handler, settings):
    transport = MockTransport(handler)
    return lambda s: httpx.Client(base_url=s.langfuse_host, transport=transport,
                                  auth=(s.langfuse_public_key, s.langfuse_secret_key))


def test_aggregate_by_intent():
    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=TRACES)
        return Response(200, json=OBS)

    out = cost_by_intent(_settings(), days=7, client_factory=_client(handler, _settings()))
    by = {i["intent"]: i for i in out["intents"]}
    assert by["退款退货"]["requests"] == 1
    assert by["退款退货"]["total_tokens"] == 150     # 去重后 100+50
    assert by["退款退货"]["input_tokens"] == 100 and by["退款退货"]["output_tokens"] == 50
    assert by["退款退货"]["cost"] == pytest.approx(0.0002)   # 优先 costDetails
    assert out["missing_usage_observations"] == 1   # o2 usage 空:可见标记不静默当 0
    assert out["unknown"]["requests"] == 1          # t3 无意图聚 unknown
    assert out["unknown"]["total_tokens"] == 0      # o1/o2 已被 t1 计入,不重复算
    assert out["cost_available"] is True
    assert out["days"] == 7 and out["incomplete"] is False


def test_dedup_is_global_across_traces():
    """同一 observation 行出现在多个 trace 的响应里只计一次。"""
    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=TRACES)
        return Response(200, json=OBS)

    out = cost_by_intent(_settings(), days=7, client_factory=_client(handler, _settings()))
    total = sum(i["total_tokens"] for i in out["intents"]) + out["unknown"]["total_tokens"]
    assert total == 150   # 三个 trace 共享同一批 observation 行,去重后全局只留一份


def test_price_fallback_when_cost_details_missing():
    obs = {"data": [
        {"id": "o9", "type": "GENERATION",
         "usageDetails": {"input": 1000000, "output": 500000}, "costDetails": None},
    ], "meta": {"totalPages": 1}}
    traces = {"data": [{"id": "t9", "metadata": {"intent": "售后"}}],
              "meta": {"totalPages": 1}}

    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=traces)
        return Response(200, json=obs)

    out = cost_by_intent(_settings(), days=7, client_factory=_client(handler, _settings()))
    row = out["intents"][0]
    assert row["total_tokens"] == 1500000
    assert row["cost"] == pytest.approx(1.0 * 1 + 2.0 * 0.5)  # 单价/1e6 折算
    assert out["cost_available"] is True


def test_zero_prices_mean_cost_unavailable():
    obs = {"data": [
        {"id": "o9", "type": "GENERATION",
         "usageDetails": {"input": 10, "output": 5}, "costDetails": None},
    ], "meta": {"totalPages": 1}}
    traces = {"data": [{"id": "t9", "metadata": {"intent": "售后"}}],
              "meta": {"totalPages": 1}}

    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=traces)
        return Response(200, json=obs)

    s = _settings(model_input_price_per_mtok=0.0, model_output_price_per_mtok=0.0)
    out = cost_by_intent(s, days=7, client_factory=_client(handler, s))
    assert out["intents"][0]["cost"] == 0.0
    assert out["cost_available"] is False


def test_disabled_raises():
    s = _settings(langfuse_enabled=False)
    with pytest.raises(CostStatsError) as ei:
        cost_by_intent(s, days=7)
    assert ei.value.code == "langfuse_disabled" and ei.value.status == 503


def test_missing_keys_raise_disabled():
    s = _settings(langfuse_secret_key="")
    with pytest.raises(CostStatsError) as ei:
        cost_by_intent(s, days=7)
    assert ei.value.code == "langfuse_disabled"


def test_unreachable_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    s = _settings()
    with pytest.raises(CostStatsError) as ei:
        cost_by_intent(s, days=7, client_factory=_client(handler, s))
    assert ei.value.code == "langfuse_unavailable" and ei.value.status == 502


def test_langfuse_http_error_raises_unavailable():
    def handler(request):
        return Response(500, json={})

    s = _settings()
    with pytest.raises(CostStatsError) as ei:
        cost_by_intent(s, days=7, client_factory=_client(handler, s))
    assert ei.value.code == "langfuse_unavailable" and ei.value.status == 502


def test_trace_cap_marks_incomplete(monkeypatch):
    pages = {
        "1": {"data": [{"id": "t1", "metadata": {"intent": "售后"}}],
              "meta": {"totalPages": 2}},
        "2": {"data": [{"id": "t2", "metadata": {"intent": "售后"}}],
              "meta": {"totalPages": 2}},
    }

    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=pages[str(request.url.params["page"])])
        return Response(200, json={"data": [], "meta": {"totalPages": 1}})

    monkeypatch.setattr(cost_stats, "MAX_TRACES", 1)
    out = cost_by_intent(_settings(), days=7, client_factory=_client(handler, _settings()))
    assert out["incomplete"] is True
    assert sum(i["requests"] for i in out["intents"]) == 1   # 只统计前 1 个 trace


def test_obs_cap_marks_incomplete(monkeypatch):
    traces = {"data": [{"id": "t1", "metadata": {"intent": "售后"}}],
              "meta": {"totalPages": 1}}
    obs = {"data": [{"id": "o1", "type": "GENERATION",
                     "usageDetails": {"input": 1, "output": 1}, "costDetails": None},
                    {"id": "o2", "type": "GENERATION",
                     "usageDetails": {"input": 1, "output": 1}, "costDetails": None}],
           "meta": {"totalPages": 1}}

    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=traces)
        return Response(200, json=obs)

    monkeypatch.setattr(cost_stats, "MAX_OBS", 1)
    out = cost_by_intent(_settings(), days=7, client_factory=_client(handler, _settings()))
    assert out["incomplete"] is True
    assert out["intents"][0]["total_tokens"] == 2   # 只计第一个 observation
