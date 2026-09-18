# tests/test_graph_nodes.py(本任务先放分类与解析用例,后续任务继续追加)
# 注:classify 测试经编译图驱动——langgraph 1.2.11 裸调节点时 get_stream_writer()
# 抛 RuntimeError(计划 Task 9 注记预授权的兜底路线,裁决采纳)。
import json
import logging

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import GraphDeps, build_front_nodes, parse_intent_output
from app.graph.state import ROUTE_TABLE, ChatGraphState, new_turn_state
from tests.conftest import make_settings


class _ClassifyModel:
    """只支持 ainvoke 的桩:返回固定文本。"""

    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._text)


def _nodes(model_text):
    deps = GraphDeps(model=_ClassifyModel(model_text), settings=make_settings(),
                     retriever=None, store=None)
    return build_front_nodes(deps)


def _graph(model):
    """前段两节点串成最小图:START → resolve_reference → classify_intent → END。"""
    nodes = build_front_nodes(GraphDeps(model=model, settings=make_settings(),
                                        retriever=None, store=None))
    g = StateGraph(ChatGraphState)
    g.add_node("resolve_reference", nodes["resolve_reference"])
    g.add_node("classify_intent", nodes["classify_intent"])
    g.add_edge(START, "resolve_reference")
    g.add_edge("resolve_reference", "classify_intent")
    g.add_edge("classify_intent", END)
    return g.compile()


@pytest.mark.parametrize("text,expected", [
    ('{"intent":"物流","needs_knowledge":false}', ("物流", False)),
    ('{"intent":"售后","needs_knowledge":true}', ("售后", True)),
    ('前缀文本{"intent":"投诉","needs_knowledge":false}后缀', ("投诉", False)),
    ('{"intent":"售后","needs_knowledge":"false"}', None),   # 字符串 false 非法
    ('{"intent":"售后"}', None),                              # 缺字段
    ('{"intent":"退票","needs_knowledge":true}', None),       # 非法枚举
    ('不是 JSON', None),
    ('{"intent":"闲聊","needs_knowledge":false,"x":1}', ("闲聊", False)),  # 容忍多余字段
])
def test_parse_intent_output(text, expected):
    assert parse_intent_output(text) == expected


async def test_classify_sets_route_from_table():
    g = _graph(_ClassifyModel('{"intent":"退款退货","needs_knowledge":false}'))
    out = await g.ainvoke(new_turn_state("我想退货"))
    assert out["intent"] == "退款退货" and out["needs_knowledge"] is False
    assert out["route"] == "knowledge"  # 路由表保守覆盖


async def test_classify_parse_failure_falls_back_to_knowledge(caplog):
    g = _graph(_ClassifyModel("模型输出了一坨废话"))
    with caplog.at_level(logging.WARNING, logger="wayhelp.graph"):
        out = await g.ainvoke(new_turn_state("查订单"))
    assert (out["intent"], out["needs_knowledge"]) == ("售后", True)
    assert out["route"] == "knowledge"  # 解析失败不得退回 business
    assert "解析失败" in caplog.text or "parse" in caplog.text


async def test_classify_model_exception_aborts_with_upstream_error():
    class _Boom:
        async def ainvoke(self, messages):
            raise ConnectionError("upstream down")

    from app.graph.errors import TurnAbortError
    with pytest.raises(TurnAbortError):
        await _graph(_Boom()).ainvoke(new_turn_state("q"))


async def test_resolve_reference_passthrough():
    nodes = _nodes("{}")
    out = await nodes["resolve_reference"](new_turn_state("原样 透传"))
    assert out["resolved_query"] == "原样 透传"


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
