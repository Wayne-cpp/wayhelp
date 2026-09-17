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


def test_parse_sub_queries():
    m = StubModel('{"standard_query": "投诉和退款的响应与处理时限", "synonyms": [], '
                  '"sub_queries": ["投诉多久响应", "退款多久处理"]}')
    plan = plan_query(m, "投诉和退款分别多久有人响应,处理要几天",
                      enabled=True, timeout_seconds=5)
    assert plan.sub_queries == ("投诉多久响应", "退款多久处理")
    assert plan.degraded is False


def test_parse_sub_queries_absent_or_dirty():
    # 缺字段(旧模型输出)→ 空,不降级
    plan = plan_query(StubModel('{"standard_query": "s", "synonyms": []}'),
                      "q", enabled=True, timeout_seconds=5)
    assert plan.sub_queries == () and plan.degraded is False
    # 非列表/脏条目/与主问同文/重复 → 过滤,封顶 3
    m = StubModel('{"standard_query": "s", "sub_queries": ["s", 1, " a ", "a", '
                  '"b", "c", "d"]}')
    plan = plan_query(m, "q", enabled=True, timeout_seconds=5)
    assert plan.sub_queries == ("a", "b", "c")


def test_passthrough_plan_has_empty_sub_queries():
    from app.knowledge.query_understanding import passthrough_plan
    assert passthrough_plan("q").sub_queries == ()
