# tests/test_graph_nodes.py(本任务先放分类与解析用例,后续任务继续追加)
# 注:classify 测试经编译图驱动——langgraph 1.2.11 裸调节点时 get_stream_writer()
# 抛 RuntimeError(计划 Task 9 注记预授权的兜底路线,裁决采纳)。
import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    GraphDeps, build_front_nodes, parse_expand_output, parse_intent_output,
    parse_refund_scope_output, parse_understand_output,
)
from app.graph.state import ChatGraphState, new_turn_state
from tests.conftest import TEST_USER_ID, make_settings


class _ClassifyModel:
    """只支持 ainvoke 的桩:返回固定文本。"""

    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._text)


def _graph(model):
    """前段两节点串成最小图:START → understand_query → classify_intent → END。"""
    nodes = build_front_nodes(GraphDeps(model=model, settings=make_settings(),
                                        retriever=None, store=None))
    g = StateGraph(ChatGraphState)
    g.add_node("understand_query", nodes["understand_query"])
    g.add_node("classify_intent", nodes["classify_intent"])
    g.add_edge(START, "understand_query")
    g.add_edge("understand_query", "classify_intent")
    g.add_edge("classify_intent", END)
    return g.compile()


@pytest.mark.parametrize("text,expected", [
    ('{"intent":"物流","confidence":0.9}', ("物流", 0.9)),
    ('前缀{"intent":"投诉","confidence":0}后缀', ("投诉", 0.0)),
    ('{"intent":"其他","confidence":"高"}', ("其他", None)),   # 非数字置 None 不拒判
    ('{"intent":"售后","confidence":1.5}', ("售后", None)),    # 越界置 None
    ('{"intent":"售后","confidence":true}', ("售后", None)),   # 布尔非法
    ('{"intent":"退票","confidence":0.9}', None),              # 非法枚举
    ('{"intent":"售后"}', ("售后", None)),                     # 缺 confidence 容忍
    ('不是 JSON', None),
    ('{"intent":"闲聊","confidence":0.9,"needs_knowledge":true}', ("闲聊", 0.9)),  # 容忍多余字段
])
def test_parse_intent_output(text, expected):
    assert parse_intent_output(text) == expected


@pytest.mark.parametrize("text,expected", [
    ('{"resolved_query":"保温杯能退吗"}', "保温杯能退吗"),
    ('{"resolved_query": ""}', ""),                    # 空串 = 透传标记
    ('前缀{"resolved_query":"到哪了"}后缀', "到哪了"),
    ('{"resolved_query": 3}', None),                   # 非字符串
    ('{"query":"x"}', None),                           # 缺字段
    ('不是 JSON', None),
])
def test_parse_understand_output(text, expected):
    assert parse_understand_output(text) == expected


@pytest.mark.parametrize("text,expected", [
    ('{"mode":"general"}', "general"),
    ('{"mode":"order_specific"}', "order_specific"),
    ('{"mode":"clarify"}', "clarify"),
    ('{"mode":"unknown"}', None),       # 枚举外 → None(调用方降级 clarify)
    ('{"mode": 1}', None),
    ('废话', None),
])
def test_parse_refund_scope_output(text, expected):
    assert parse_refund_scope_output(text) == expected


@pytest.mark.parametrize("text,expected", [
    ('{"queries":["a","b"]}', ["a", "b"]),
    ('{"queries":["a"," a ","","b","a"]}', ["a", "b"]),   # 去空去重
    ('{"queries":[]}', []),                                # 可空
    ('{"queries":[1,2]}', []),                             # 非字符串项被过滤,结构仍合法
    ('{"queries":"a"}', None),                             # 非列表结构非法
    ('nope', None),
])
def test_parse_expand_output(text, expected):
    assert parse_expand_output(text) == expected


async def test_classify_sets_route_from_table():
    g = _graph(_ClassifyModel('{"intent":"退款退货","confidence":0.8}'))
    out = await g.ainvoke(new_turn_state("我想退货"))
    assert out["intent"] == "退款退货" and out["intent_confidence"] == 0.8
    assert out["route"] == "refund"


async def test_classify_parse_failure_falls_back_to_other(caplog):
    g = _graph(_ClassifyModel("模型输出了一坨废话"))
    with caplog.at_level(logging.WARNING, logger="wayhelp.graph"):
        out = await g.ainvoke(new_turn_state("查订单"))
    assert out["intent"] == "其他" and out["intent_confidence"] is None
    assert out["route"] == "other"  # 解析失败兜底「其他」,不硬塞业务意图
    assert "解析失败" in caplog.text or "parse" in caplog.text


async def test_classify_model_exception_aborts_with_upstream_error():
    class _Boom:
        async def ainvoke(self, messages):
            raise ConnectionError("upstream down")

    from app.graph.errors import TurnAbortError
    with pytest.raises(TurnAbortError):
        await _graph(_Boom()).ainvoke(new_turn_state("q"))


# ── ch06 Task 3:understand_query 指代消解节点 ──

class _UnderstandModel:
    """按调用返回脚本文本的 ainvoke 桩;记录收到的 prompt。"""

    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self._texts.pop(0))


def _understand_graph(model):
    nodes = build_front_nodes(GraphDeps(model=model, settings=make_settings(),
                                        retriever=None, store=None))
    g = StateGraph(ChatGraphState)
    g.add_node("understand_query", nodes["understand_query"])
    g.add_edge(START, "understand_query")
    g.add_edge("understand_query", END)
    return g.compile()


def _cfg(user_id=TEST_USER_ID):
    return {"configurable": {"thread_id": "1", "user_id": user_id}}


async def test_understand_first_turn_skips_llm():
    model = _UnderstandModel([])
    out = await _understand_graph(model).ainvoke(new_turn_state("你好"), config=_cfg())
    assert out["resolved_query"] == "你好"
    assert model.prompts == []  # 首轮零成本


async def test_understand_resolves_coreference_with_history():
    model = _UnderstandModel(['{"resolved_query": "保温杯能退吗"}'])
    st = new_turn_state("它能退吗")
    st["messages"] = [HumanMessage(content="我想买保温杯"),
                      AIMessage(content="可以的,保温杯 89 元。")]
    out = await _understand_graph(model).ainvoke(st, config=_cfg())
    assert out["resolved_query"] == "保温杯能退吗"
    assert out["understanding_degraded"] is False


async def test_understand_passthrough_marker_and_blank():
    model = _UnderstandModel(['{"resolved_query": ""}'])
    st = new_turn_state("订单 1111-1001 到哪了")
    st["messages"] = [HumanMessage(content="你好"), AIMessage(content="您好!")]
    out = await _understand_graph(model).ainvoke(st, config=_cfg())
    assert out["resolved_query"] == "订单 1111-1001 到哪了"  # 空串 = 透传


async def test_understand_parse_failure_and_model_error_degrade():
    for bad in ("一坨废话", None):
        model = _UnderstandModel([bad] if bad else [])
        if bad is None:
            class _Boom:
                async def ainvoke(self, m):
                    raise ConnectionError("x")
            model = _Boom()
        st = new_turn_state("它呢")
        st["messages"] = [HumanMessage(content="前一句"), AIMessage(content="答")]
        out = await _understand_graph(model).ainvoke(st, config=_cfg())
        assert out["resolved_query"] == "它呢" and out["understanding_degraded"] is True


async def test_understand_rejects_hallucinated_order_id():
    model = _UnderstandModel(['{"resolved_query": "订单 1111-1002 能退吗"}'])
    st = new_turn_state("它能退吗")  # 原文与 active_order 都没有 1111-1002
    st["messages"] = [HumanMessage(content="看看保温杯"), AIMessage(content="好的")]
    out = await _understand_graph(model).ainvoke(st, config=_cfg())
    assert out["resolved_query"] == "它能退吗" and out["understanding_degraded"] is True


async def test_understand_uses_validated_active_order_and_clears_stale():
    model = _UnderstandModel(['{"resolved_query": "订单 1111-1001 能退吗"}'])
    st = new_turn_state("这单能退吗")
    st["messages"] = [HumanMessage(content="选了订单"), AIMessage(content="好的")]
    st["active_order"] = {"order_id": "1111-1001", "source_message_id": "3"}
    out = await _understand_graph(model).ainvoke(st, config=_cfg())
    assert out["resolved_query"] == "订单 1111-1001 能退吗"
    assert out["active_order"]["order_id"] == "1111-1001"
    # 失效 active_order(不属于该用户)被清空
    st2 = new_turn_state("这单能退吗")
    st2["active_order"] = {"order_id": "9999-1001", "source_message_id": "1"}
    out2 = await _understand_graph(model).ainvoke(st2, config=_cfg())
    assert out2["active_order"] is None


def test_history_turns_cuts_complete_turns():
    from app.graph.nodes import _history_turns
    msgs = [HumanMessage(content="u1"), AIMessage(content="a1"),
            HumanMessage(content="u2"), AIMessage(content="a2"),
            HumanMessage(content="u3")]
    turns = _history_turns(msgs, 6)
    assert [t[0].content for t in turns] == ["u1", "u2"]  # 末尾未成轮不切
    assert [t[0].content for t in _history_turns(msgs, 1)] == ["u2"]


from app.knowledge.retriever import (
    NOTE_NOT_BUILT, NOTE_REBUILDING, NOTE_UNCONFIGURED, RetrievalResult,
)
from app.graph.nodes import build_knowledge_nodes
from app.prompts.service import KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER


def _result(note=None, low=False, hits=None, score=0.9):
    return RetrievalResult(
        hits=hits or [], requested_strategy="hybrid_rerank",
        effective_strategy="hybrid_rerank", confidence_score=score,
        confidence_threshold=0.0553, low_confidence=low, note=note,
        query_plan=None, leg_counts={"dense": 3, "bm25": 2})


class _FakeRetriever:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc
        self.queries = []

    def search(self, query, **kw):
        self.queries.append(query)
        if self._exc:
            raise self._exc
        return self._result


def _knodes(result=None, exc=None):
    deps = GraphDeps(model=None, settings=make_settings(),
                     retriever=_FakeRetriever(result, exc), store=None)
    return build_knowledge_nodes(deps)


async def test_retrieve_ok_snapshot_serializable():
    import json as _json
    from app.knowledge.retriever import KnowledgeHit
    hit = KnowledgeHit(chunk_id=7, score=0.9, category="policy", questions="q",
                       answer="a", source_doc="d.md", chunk_index=0, section_path="退货")
    nodes = _knodes(result=_result(hits=[hit]))
    st = new_turn_state("退货政策")
    st["resolved_query"] = "退货政策"
    out = await nodes["retrieve"](st)
    assert out["retrieval_status"] == "ok"
    _json.dumps(out["retrieval_result"])  # 快照必须可序列化(进 checkpoint)
    assert out["retrieval_result"]["hits"][0]["chunk_id"] == 7


async def test_retrieve_unavailable_states_not_pooled():
    for note, code in ((NOTE_UNCONFIGURED, "kb_unconfigured"),
                       (NOTE_REBUILDING, "kb_rebuilding")):
        nodes = _knodes(result=_result(note=note, low=True, score=None))
        st = new_turn_state("q"); st["resolved_query"] = "q"
        out = await nodes["retrieve"](st)
        assert out["retrieval_status"] == "unavailable"
        assert out["retrieval_error_code"] == code  # 即使 low_confidence=True 也不算知识缺口


async def test_retrieve_exception_is_unavailable():
    nodes = _knodes(exc=TimeoutError("timeout"))
    st = new_turn_state("q"); st["resolved_query"] = "q"
    out = await nodes["retrieve"](st)
    assert out["retrieval_status"] == "unavailable"
    assert out["retrieval_error_code"] == "kb_unavailable"


async def test_gate_low_confidence_marks_pool_fields():
    nodes = _knodes(result=_result(note=NOTE_NOT_BUILT, low=True, score=None))
    st = new_turn_state("q"); st["resolved_query"] = "q"
    st.update(await nodes["retrieve"](st))
    out = await nodes["confidence_gate"](st)
    assert out["low_conf_source"] == "retrieval_low_conf"
    assert out["low_conf_reason"]["note"] == NOTE_NOT_BUILT
    assert nodes["route_after_gate"]({**st, **out}) == "gate_fallback"


def _gate_graph(nodes):
    """gate_fallback 首行取 writer,裸调在 langgraph 1.2.11 抛 RuntimeError——
    按 Task 8 裁决 A 经编译图驱动(nodes.py 实现不加防护),断言与 plan 一致。"""
    g = StateGraph(ChatGraphState)
    g.add_node("gate_fallback", nodes["gate_fallback"])
    g.add_edge(START, "gate_fallback")
    g.add_edge("gate_fallback", END)
    return g.compile()


async def test_gate_fallback_texts():
    g = _gate_graph(_knodes())
    out = await g.ainvoke({**new_turn_state("q"), "retrieval_status": "unavailable"})
    assert out["final_text"] == KB_UNAVAILABLE_ANSWER
    out2 = await g.ainvoke({**new_turn_state("q"), "retrieval_status": "low_confidence"})
    assert out2["final_text"] == REFUSAL_ANSWER
    assert out2["turn_messages"][-1].content == REFUSAL_ANSWER  # 进本轮消息


from app.graph.nodes import build_fixed_nodes
from app.prompts.service import CHITCHAT_REPLY, COMPLAINT_REPLY


def _fixed_graph(node_name):
    """固定回复节点首行取 writer,裸调在 langgraph 1.2.11 抛 RuntimeError——
    按 Task 8 裁决 A 经编译图驱动(nodes.py 实现不加防护),断言与 plan 一致。"""
    g = StateGraph(ChatGraphState)
    g.add_node(node_name, build_fixed_nodes()[node_name])
    g.add_edge(START, node_name)
    g.add_edge(node_name, END)
    return g.compile()


async def test_complaint_reply_with_two_independent_options():
    out = await _fixed_graph("complaint_reply").ainvoke(new_turn_state("我要投诉"))
    assert out["final_text"] == COMPLAINT_REPLY
    assert out["turn_messages"][-1].content == COMPLAINT_REPLY
    actions = out["suggested_actions"]
    assert [a["action"] for a in actions] == ["transfer_human", "create_ticket"]
    assert actions[0]["label"] == "转人工" and "ticket_type" not in actions[0]
    assert actions[1]["label"] == "建工单" and actions[1]["ticket_type"] == "投诉"


async def test_chitchat_reply_no_model_no_actions():
    out = await _fixed_graph("chitchat_reply").ainvoke(new_turn_state("你好"))
    assert out["final_text"] == CHITCHAT_REPLY
    assert out["suggested_actions"] == []


from app.graph.nodes import build_log_node
from app.prompts.service import AGENT_BUDGET_ANSWER
from app.sessions import InMemorySessionStore


def _log_deps(store):
    return GraphDeps(model=None, settings=make_settings(), retriever=None, store=store)


def _config(sid):
    return {"configurable": {"thread_id": sid}}


def _log_graph(store):
    """log 节点首行取 writer,裸调在 langgraph 1.2.11 抛 RuntimeError——
    按 Task 8 裁决 A 经编译图驱动(nodes.py 实现不加防护),断言语义与 plan 一致。"""
    g = StateGraph(ChatGraphState)
    g.add_node("log", build_log_node(_log_deps(store)))
    g.add_edge(START, "log")
    g.add_edge("log", END)
    return g.compile()


async def test_log_commits_and_returns_source_message_id():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    st = new_turn_state("你好")
    st.update({"final_text": "您好", "route": "chitchat"})
    st["turn_messages"].append(AIMessage(content="您好"))
    out = await _log_graph(store).ainvoke(st, config=_config(sid))
    assert out["source_message_id"]  # commit 成功后才返回
    assert out["messages"] == st["turn_messages"]  # 此刻才追加跨轮历史
    snap = await store.snapshot(sid)
    assert [m.role for m in snap] == ["user", "assistant"]


async def test_log_pools_retrieval_low_conf():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    st = new_turn_state("库外问题")
    st.update({"final_text": REFUSAL_ANSWER, "route": "knowledge",
               "retrieval_status": "low_confidence",
               "low_conf_source": "retrieval_low_conf",
               "low_conf_reason": {"note": "知识库尚未建立", "top1": None}})
    st["turn_messages"].append(AIMessage(content=REFUSAL_ANSWER))
    await _log_graph(store).ainvoke(st, config=_config(sid))
    assert len(store.low_confidence) == 1
    assert store.low_confidence[0].source == "retrieval_low_conf"
    assert store.low_confidence[0].raw_question == "库外问题"


async def test_log_pools_self_check_when_ok_but_refusal():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    st = new_turn_state("偏门问题")
    st.update({"final_text": REFUSAL_ANSWER, "route": "knowledge",
               "retrieval_status": "ok",
               "evidence": [{"ref_no": 1, "chunk_id": 7, "section_path": "s",
                             "question": "q", "answer": "a", "category": "c"}]})
    st["turn_messages"].append(AIMessage(content=REFUSAL_ANSWER))
    await _log_graph(store).ainvoke(st, config=_config(sid))
    assert store.low_confidence[0].source == "self_check"
    assert "chunk_id" in store.low_confidence[0].reason


async def test_log_does_not_pool_unavailable_or_budget():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    g = _log_graph(store)
    for status, text in (("unavailable", KB_UNAVAILABLE_ANSWER),
                         ("ok", AGENT_BUDGET_ANSWER)):
        st = new_turn_state("q")
        st.update({"final_text": text, "route": "knowledge",
                   "retrieval_status": status})
        st["turn_messages"].append(AIMessage(content=text))
        await g.ainvoke(st, config=_config(sid))
    assert store.low_confidence == []


def test_should_cite_rules():
    from app.graph.nodes import _should_cite
    base = {"route": "knowledge", "retrieval_status": "ok",
            "evidence": [{"ref_no": 1}], "final_text": "7 天无理由[1]"}
    assert _should_cite(base) is True
    assert _should_cite({**base, "route": "business"}) is False
    assert _should_cite({**base, "final_text": REFUSAL_ANSWER}) is False
    assert _should_cite({**base, "final_text": "没标角标的答复"}) is False
