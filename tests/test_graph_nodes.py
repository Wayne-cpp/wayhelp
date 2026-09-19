# tests/test_graph_nodes.py(本任务先放分类与解析用例,后续任务继续追加)
# 注:classify 测试经编译图驱动——langgraph 1.2.11 裸调节点时 get_stream_writer()
# 抛 RuntimeError(计划 Task 9 注记预授权的兜底路线,裁决采纳)。
import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    GraphDeps, build_front_nodes, build_refund_nodes, parse_expand_output,
    parse_intent_output, parse_refund_scope_output, parse_understand_output,
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


# ── ch06 Task 7:refund 子流程三节点(refund_scope / refund_prepare / refund_policy)──
# 旧 retrieve/confidence_gate 用例按计划迁移为 refund_policy 用例(单查询路径状态映射等价):
# 维护 note→unavailable、低置信→low_conf_fields、ok→evidence。

def _kh(cid, score=0.9):
    from app.knowledge.retriever import KnowledgeHit
    return KnowledgeHit(chunk_id=cid, score=score, category="policy", questions="q",
                        answer="a", source_doc="d.md", chunk_index=0, section_path="退货")


def _scope_graph(model):
    nodes = build_front_nodes(GraphDeps(model=model, settings=make_settings(),
                                        retriever=None, store=None))
    g = StateGraph(ChatGraphState)
    g.add_node("refund_scope", nodes["refund_scope"])
    g.add_edge(START, "refund_scope")
    g.add_edge("refund_scope", END)
    return g.compile()


@pytest.mark.parametrize("text,mode", [
    ('{"mode":"general"}', "general"),
    ('{"mode":"order_specific"}', "order_specific"),
    ('{"mode":"clarify"}', "clarify"),
    ('{"mode":"unknown"}', "clarify"),   # 枚举外降级
    ("废话", "clarify"),                 # 解析失败降级
])
async def test_refund_scope_modes_and_degradation(text, mode):
    st = new_turn_state("如何申请退款")
    st["resolved_query"] = "如何申请退款"
    out = await _scope_graph(_ClassifyModel(text)).ainvoke(st)
    assert out["refund_mode"] == mode


async def test_refund_scope_model_error_degrades_to_clarify():
    class _Boom:
        async def ainvoke(self, m):
            raise ConnectionError("x")
    st = new_turn_state("q"); st["resolved_query"] = "q"
    out = await _scope_graph(_Boom()).ainvoke(st)
    assert out["refund_mode"] == "clarify"


def _prepare_graph():
    """refund_prepare 含 interrupt(),必须经带 checkpointer 的编译图驱动。"""
    from langgraph.checkpoint.memory import InMemorySaver
    nodes = build_refund_nodes(GraphDeps(model=None, settings=make_settings(),
                                         retriever=None, store=None))
    g = StateGraph(ChatGraphState)
    g.add_node("refund_prepare", nodes["refund_prepare"])
    g.add_edge(START, "refund_prepare")
    g.add_edge("refund_prepare", END)
    return g.compile(checkpointer=InMemorySaver())


async def test_refund_prepare_direct_with_valid_order():
    g = _prepare_graph()
    st = new_turn_state("订单 1111-1001 能退吗")
    st["resolved_query"] = "订单 1111-1001 能退吗"
    out = await g.ainvoke(st, config=_cfg())  # 直达 END,无 interrupt
    assert out["order_context"]["order_id"] == "1111-1001"
    assert out["order_context"]["product"] == "保温杯"


async def test_refund_prepare_interrupts_and_resumes():
    from langgraph.types import Command
    g = _prepare_graph()
    cfg = {"configurable": {"thread_id": "t1", "user_id": TEST_USER_ID}}
    st = new_turn_state("这个能退吗")
    st["resolved_query"] = "这个能退吗"
    await g.ainvoke(st, config=cfg)  # 挂起
    snap = await g.aget_state(cfg)
    intrs = [i for t in snap.tasks for i in t.interrupts]
    assert len(intrs) == 1 and intrs[0].value["type"] == "order_selector"
    assert [o["order_id"] for o in intrs[0].value["orders"]] == \
        ["1111-1001", "1111-1002", "1111-1003", "1111-1004"]
    out = await g.ainvoke(Command(resume={intrs[0].id: {"order_id": "1111-1002"}}),
                          config=cfg)
    assert out["order_context"]["order_id"] == "1111-1002"


async def test_refund_prepare_multi_candidates_and_cross_user_interrupt():
    g = _prepare_graph()
    cfg = {"configurable": {"thread_id": "t2", "user_id": TEST_USER_ID}}
    st = new_turn_state("1111-1001 和 1111-1002 哪个能退")
    st["resolved_query"] = "1111-1001 和 1111-1002 哪个能退"
    await g.ainvoke(st, config=cfg)  # 多候选 → 挂起
    snap = await g.aget_state(cfg)
    assert any(i for t in snap.tasks for i in t.interrupts)
    # 他人订单号:原文显式但无归属 → 不直通,挂起
    cfg2 = {"configurable": {"thread_id": "t3", "user_id": TEST_USER_ID}}
    st2 = new_turn_state("2222-1001 能退吗")
    st2["resolved_query"] = "2222-1001 能退吗"
    await g.ainvoke(st2, config=cfg2)
    snap2 = await g.aget_state(cfg2)
    assert any(i for t in snap2.tasks for i in t.interrupts)


def _rnodes(model=None, result=None, exc=None, retriever=None, **settings_over):
    r = retriever if retriever is not None else _FakeRetriever(result, exc)
    deps = GraphDeps(model=model, settings=make_settings(**settings_over),
                     retriever=r, store=None)
    return build_refund_nodes(deps)


class _ExpandModel:
    """按调用序返回脚本文本的 ainvoke 桩(扩写);记录 prompt。"""

    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self._texts.pop(0))


class _RankRetriever:
    """search 按调用序循环返回预设结果;rerank_candidates 记录调用,可返回统一结果/None/抛错。"""

    def __init__(self, results, rerank="unset", rerank_exc=None):
        self._results = list(results)
        self._n = 0
        self.search_queries = []
        self.rerank_calls = []
        self._rerank, self._rerank_exc = rerank, rerank_exc

    def search(self, query, **kw):
        self.search_queries.append(query)
        r = self._results[self._n % len(self._results)]
        self._n += 1
        return r

    def rerank_candidates(self, query, hits, deadline=None):
        self.rerank_calls.append((query, [h.chunk_id for h in hits]))
        if self._rerank_exc:
            raise self._rerank_exc
        if self._rerank is None:
            return None
        assert self._rerank != "unset"
        return self._rerank


def _st_order(query="这个能退吗", order_id="1111-1001"):
    from dataclasses import asdict
    from app.services.orders import get_order
    st = new_turn_state(query)
    st["resolved_query"] = query
    st["order_context"] = asdict(get_order(TEST_USER_ID, order_id))
    return st


async def test_refund_policy_ok_snapshot_serializable_with_evidence():
    import json as _json
    nodes = _rnodes(result=_result(hits=[_kh(7)]))
    st = new_turn_state("退货政策")
    st["resolved_query"] = "退货政策"
    out = await nodes["refund_policy"](st)
    assert out["retrieval_status"] == "ok"
    _json.dumps(out["retrieval_result"])  # 快照必须可序列化(进 checkpoint)
    assert out["retrieval_result"]["hits"][0]["chunk_id"] == 7
    assert out["evidence"][0]["chunk_id"] == 7          # ok → evidence(gate 合并语义)
    assert out["expanded_queries"] == ["退货政策"]      # 单查询也记录实际执行列表


async def test_refund_policy_unavailable_states_not_pooled():
    for note, code in ((NOTE_UNCONFIGURED, "kb_unconfigured"),
                       (NOTE_REBUILDING, "kb_rebuilding")):
        nodes = _rnodes(result=_result(note=note, low=True, score=None))
        st = new_turn_state("q"); st["resolved_query"] = "q"
        out = await nodes["refund_policy"](st)
        assert out["retrieval_status"] == "unavailable"
        assert out["retrieval_error_code"] == code  # 即使 low_confidence=True 也不算知识缺口
        assert "low_conf_source" not in out


async def test_refund_policy_exception_is_unavailable():
    nodes = _rnodes(exc=TimeoutError("timeout"))
    st = new_turn_state("q"); st["resolved_query"] = "q"
    out = await nodes["refund_policy"](st)
    assert out["retrieval_status"] == "unavailable"
    assert out["retrieval_error_code"] == "kb_unavailable"
    assert out["expanded_queries"] == ["q"]


async def test_refund_policy_retriever_none_unconfigured():
    deps = GraphDeps(model=None, settings=make_settings(), retriever=None, store=None)
    nodes = build_refund_nodes(deps)
    st = new_turn_state("q"); st["resolved_query"] = "q"
    out = await nodes["refund_policy"](st)
    assert out["retrieval_status"] == "unavailable"
    assert out["retrieval_error_code"] == "kb_unconfigured"


async def test_refund_policy_low_confidence_marks_pool_fields():
    from app.graph.nodes import route_after_gate
    nodes = _rnodes(result=_result(note=NOTE_NOT_BUILT, low=True, score=None))
    st = new_turn_state("q"); st["resolved_query"] = "q"
    out = await nodes["refund_policy"](st)
    assert out["low_conf_source"] == "retrieval_low_conf"
    assert out["low_conf_reason"]["note"] == NOTE_NOT_BUILT
    assert route_after_gate({**st, **out}) == "gate_fallback"


async def test_refund_policy_no_expand_without_order_context():
    model = _ExpandModel([])  # 若被调用 pop(0) 即 IndexError → 用例炸
    nodes = _rnodes(model=model, result=_result(hits=[_kh(7)]))
    st = new_turn_state("如何申请退款")
    st["resolved_query"] = "如何申请退款"
    out = await nodes["refund_policy"](st)
    assert model.prompts == []                       # general 不扩写(spec §6.4)
    assert out["expanded_queries"] == ["如何申请退款"]
    assert out["retrieval_status"] == "ok"


async def test_refund_policy_no_expand_when_disabled_or_not_rerank():
    for over in ({"refund_expand_enabled": False},
                 {"knowledge_strategy": "hybrid"}):
        model = _ExpandModel([])
        nodes = _rnodes(model=model, result=_result(hits=[_kh(7)]), **over)
        out = await nodes["refund_policy"](_st_order())
        assert model.prompts == []
        assert len(out["expanded_queries"]) == 1


async def test_refund_policy_expands_appends_order_facts_and_reranks_unified():
    from app.knowledge.retriever import RetrievalResult
    unified = RetrievalResult(
        hits=[_kh(8, 0.95), _kh(7, 0.9)], requested_strategy="hybrid_rerank",
        effective_strategy="hybrid_rerank", confidence_score=0.95,
        confidence_threshold=0.0553, low_confidence=False, note=None,
        query_plan=None, leg_counts={"rerank": 1})
    rt = _RankRetriever(results=[_result(hits=[_kh(7)]),
                                 _result(hits=[_kh(8), _kh(7)])], rerank=unified)
    model = _ExpandModel(['{"queries":["退货期限是多久","运费谁承担"]}'])
    nodes = _rnodes(model=model, retriever=rt)
    out = await nodes["refund_policy"](_st_order())
    # base 第一条;订单事实只附加进检索问句(expanded_queries 记实际执行列表)
    assert len(out["expanded_queries"]) == 3
    assert out["expanded_queries"][0] == rt.search_queries[0]
    assert len(rt.search_queries) == 3
    assert rt.search_queries[0].startswith("这个能退吗")
    assert "保温杯" in rt.search_queries[0] and "定制" not in rt.search_queries[0]
    assert rt.rerank_calls[0][1] == [7, 8]   # 只按 chunk_id 去重合并,不比较各查询分数(D10)
    assert out["retrieval_status"] == "ok"
    assert out["retrieval_result"]["confidence_score"] == 0.95   # 统一重排结果
    assert out["evidence"][0]["chunk_id"] == 8


async def test_refund_policy_rerank_failure_keeps_base_result():
    base = _result(hits=[_kh(7)])  # score 0.9 ≥ 阈值
    rt = _RankRetriever(results=[base], rerank=None)
    model = _ExpandModel(['{"queries":["退货期限是多久"]}'])
    nodes = _rnodes(model=model, retriever=rt)
    out = await nodes["refund_policy"](_st_order())
    assert out["retrieval_status"] == "ok"
    assert out["retrieval_result"]["confidence_score"] == 0.9   # base 完整策略/分数/阈值
    assert out["retrieval_result"]["note"] == "unified_rerank_failed"  # 可观测回退原因
    assert out["evidence"][0]["chunk_id"] == 7


async def test_refund_policy_zero_candidates_skips_rerank_and_note():
    # FIX B(ch06 评审):零候选 = rerank 从未运行,直接回退 base,不挂失败 note(spec §6.4)
    rt = _RankRetriever(results=[_result(hits=[])], rerank="unset")
    model = _ExpandModel(['{"queries":["退货期限是多久"]}'])
    nodes = _rnodes(model=model, retriever=rt)
    out = await nodes["refund_policy"](_st_order())
    assert rt.rerank_calls == []                       # 无候选可排,统一重排未运行
    assert out["retrieval_result"]["note"] is None     # 未失败不挂 unified_rerank_failed
    assert out["retrieval_status"] == "ok"             # 直接使用该回退结果(base)


async def test_refund_policy_expand_parse_failure_degrades_single_query():
    rt = _RankRetriever(results=[_result(hits=[_kh(7)])], rerank="unset")
    model = _ExpandModel(["废话"])  # 解析失败 → 单查询降级
    nodes = _rnodes(model=model, retriever=rt)
    out = await nodes["refund_policy"](_st_order())
    assert len(rt.search_queries) == 1                       # 只查 base_query
    assert rt.rerank_calls == []                             # 单查询无统一重排
    assert out["expanded_queries"] == [rt.search_queries[0]]
    assert out["retrieval_result"]["note"] == "expand_degraded:ValueError"
    assert out["retrieval_status"] == "ok"


async def test_refund_policy_maintenance_on_any_query_unavailable():
    ok = _result(hits=[_kh(7)])
    rebuilding = _result(note=NOTE_REBUILDING, low=True, hits=[], score=None)
    rt = _RankRetriever(results=[ok, rebuilding])
    model = _ExpandModel(['{"queries":["第二问"]}'])
    nodes = _rnodes(model=model, retriever=rt)
    out = await nodes["refund_policy"](_st_order())
    assert out["retrieval_status"] == "unavailable"          # 任一查询维护态 → 整轮不可用
    assert out["retrieval_error_code"] == "kb_rebuilding"
    assert len(out["expanded_queries"]) == 2


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


# ─── 检索 helpers 模块级提取(Task 4;供 Task 6/7 复用)───────────────────────


def test_state_fields_for_result_note_over_low_confidence():
    from app.graph.nodes import state_fields_for_result
    fields = state_fields_for_result(_result(note=NOTE_REBUILDING, low=True, score=None))
    assert fields["retrieval_status"] == "unavailable"   # 配置/维护态优先于低置信判定
    assert fields["retrieval_error_code"] == "kb_rebuilding"
    assert fields["retrieval_result"]["low_confidence"] is True
    assert state_fields_for_result(_result(low=False))["retrieval_status"] == "ok"
    assert state_fields_for_result(
        _result(note=NOTE_NOT_BUILT, low=True, score=None))["retrieval_status"] == "low_confidence"


def test_evidence_dicts_from_snapshot_same_budget_as_gate():
    from app.graph.nodes import evidence_dicts_from_snapshot, snapshot_retrieval
    from app.knowledge.retriever import KnowledgeHit
    hit = KnowledgeHit(chunk_id=7, score=0.9, category="policy", questions="q",
                       answer="a", source_doc="d.md", chunk_index=0, section_path="退货")
    ev = evidence_dicts_from_snapshot(snapshot_retrieval(_result(hits=[hit])),
                                      make_settings())
    assert [e["ref_no"] for e in ev] == [1]
    assert ev[0]["chunk_id"] == 7 and ev[0]["question"] == "q"
    assert evidence_dicts_from_snapshot(snapshot_retrieval(_result()),
                                        make_settings()) == []   # 空 hits 不组装


def test_low_conf_fields_reason_structure():
    from app.graph.nodes import low_conf_fields, snapshot_retrieval
    f = low_conf_fields(snapshot_retrieval(_result(note=NOTE_NOT_BUILT, low=True, score=None)))
    assert f["low_conf_source"] == "retrieval_low_conf"
    assert f["low_conf_reason"] == {"requested_strategy": "hybrid_rerank",
                                    "effective_strategy": "hybrid_rerank",
                                    "top1": None, "threshold": 0.0553,
                                    "note": NOTE_NOT_BUILT}


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


def test_should_cite_rules_ch06():
    from app.graph.nodes import _should_cite
    base = {"route": "business", "retrieval_status": "ok",
            "evidence": [{"ref_no": 1}], "final_text": "7 天无理由[1]"}
    assert _should_cite(base) is True                    # business 工具检索也可引用
    assert _should_cite({**base, "route": "refund"}) is True
    assert _should_cite({**base, "final_text": REFUSAL_ANSWER}) is False
    assert _should_cite({**base, "final_text": KB_UNAVAILABLE_ANSWER}) is False
    assert _should_cite({**base, "final_text": "没标角标"}) is False
    assert _should_cite({**base, "retrieval_status": "low_confidence"}) is False


async def test_log_updates_active_order_only_with_order_context():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    st = new_turn_state("这单能退吗")
    st.update({"final_text": "可以退。", "route": "refund",
               "order_context": {"order_id": "1111-1001"}})
    st["turn_messages"].append(AIMessage(content="可以退。"))
    out = await _log_graph(store).ainvoke(st, config=_config(sid))
    assert out["active_order"] == {"order_id": "1111-1001",
                                   "source_message_id": out["source_message_id"]}
    # 无 order_context 的轮次不建立焦点
    st2 = new_turn_state("你好")
    st2.update({"final_text": "您好", "route": "chitchat"})
    st2["turn_messages"].append(AIMessage(content="您好"))
    out2 = await _log_graph(store).ainvoke(st2, config=_config(sid))
    assert "active_order" not in out2 or out2.get("active_order") is None
