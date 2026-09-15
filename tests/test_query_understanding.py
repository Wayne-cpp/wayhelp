from langchain_core.messages import AIMessage

from app.knowledge.query_understanding import QueryPlan, plan_query


class StubModel:
    def __init__(self, content): self._content = content
    def invoke(self, messages): return AIMessage(content=self._content)


class BoomModel:
    def invoke(self, messages): raise ConnectionError("down")


def test_parse_ok():
    m = StubModel('{"standard_query": "退货期限是多久", "synonyms": ["退款", "退钱"]}')
    plan = plan_query(m, "东西不想要了还能退不", enabled=True, timeout_seconds=5)
    assert plan.standard_query == "退货期限是多久"
    assert plan.synonyms == ("退款", "退钱") and plan.degraded is False


def test_parse_fail_degrades():
    plan = plan_query(StubModel("不是 JSON"), "原话", enabled=True, timeout_seconds=5)
    assert plan.degraded is True and plan.standard_query == "原话" and plan.synonyms == ()


def test_exception_degrades():
    plan = plan_query(BoomModel(), "原话", enabled=True, timeout_seconds=5)
    assert plan.degraded is True and plan.standard_query == "原话"


def test_disabled_passthrough():
    plan = plan_query(BoomModel(), "原话", enabled=False, timeout_seconds=5)
    assert plan.degraded is True and plan.standard_query == "原话"


def test_empty_standard_query_degrades():
    m = StubModel('{"standard_query": "", "synonyms": []}')
    assert plan_query(m, "原话", enabled=True, timeout_seconds=5).degraded is True
