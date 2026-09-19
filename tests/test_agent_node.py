# tests/test_agent_node.py
# 注:main_agent 首行即取 writer,裸调在 langgraph 1.2.11 抛 RuntimeError——
# 按 Task 8 裁决经编译图驱动(与 test_graph_nodes._gate_graph 同式),断言与 plan 一致。
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.graph.agent_node import build_agent_node
from app.graph.nodes import GraphDeps
from app.graph.state import ChatGraphState, new_turn_state
from app.prompts.service import REFUSAL_ANSWER
from tests.conftest import TEST_USER_ID, FakeStreamModel, make_settings


def _tool_chunk(name, args, call_id):
    return ("tool", [{"index": 0, "name": name, "id": call_id,
                      "args": json.dumps(args)}])


def _agent_graph(deps):
    """main_agent 包进最小编译图返回(START → main_agent → END)。"""
    g = StateGraph(ChatGraphState)
    g.add_node("main_agent", build_agent_node(deps))
    g.add_edge(START, "main_agent")
    g.add_edge("main_agent", END)
    return g.compile()


def _cfg(user_id=""):
    return {"configurable": {"user_id": user_id}}


def _agent(script, **settings_over):
    return _agent_graph(GraphDeps(model=FakeStreamModel(script),
                                  settings=make_settings(**settings_over),
                                  retriever=None, store=None))


async def test_one_step_converge():
    node = _agent(["订单 1001 已发货。"])
    out = await node.ainvoke(new_turn_state("订单 1001 状态"))
    assert out["final_text"] == "订单 1001 已发货。"
    assert out["agent_steps"] == 1
    assert out["turn_messages"][-1].content == "订单 1001 已发货。"


async def test_multi_step_order_then_logistics():
    st = new_turn_state("订单 1111-1001 到哪了,先查订单再查物流")
    st.update({"resolved_query": "订单 1111-1001 到哪了,先查订单再查物流",
               "route": "business"})
    node = _agent([
        _tool_chunk("query_order", {"order_id": "1111-1001"}, "c1"), ("then", [
            _tool_chunk("query_logistics", {"order_id": "1111-1001"}, "c2"), ("then", [
                "订单已发货,物流派送中。"])]),
    ])
    out = await node.ainvoke(st, config=_cfg(TEST_USER_ID))
    assert out["agent_steps"] == 3
    tools = [m for m in out["turn_messages"] if m.type == "tool"]
    assert [t.name for t in tools] == ["query_order", "query_logistics"]
    assert "1111-1001" in tools[0].content  # 真实工具结果喂回
    assert out["final_text"] == "订单已发货,物流派送中。"


async def test_clarify_question_is_plain_convergence():
    node = _agent(["请问您要查询哪个订单号?"])
    out = await node.ainvoke(new_turn_state("帮我查下物流"))
    assert out["final_text"] == "请问您要查询哪个订单号?"
    assert out["agent_steps"] == 1  # 缺信息追问 = 无 tool_calls 收敛


async def test_suggest_options_signal_no_side_effect():
    node = _agent([
        _tool_chunk("suggest_options", {"options": ["转人工", "建工单"],
                                        "ticket_type": "售后"}, "s1"),
        ("then", ["好的,您可以选择下方按钮。"]),
    ])
    out = await node.ainvoke(new_turn_state("我要找人工"))
    assert [a["action"] for a in out["suggested_actions"]] == ["transfer_human", "create_ticket"]
    assert out["suggested_actions"][1]["ticket_type"] == "售后"
    tool_msgs = [m for m in out["turn_messages"] if m.type == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "s1"  # 配对完整


async def test_forged_create_ticket_gets_unknown_tool():
    node = _agent([
        _tool_chunk("create_ticket", {"description": "x", "ticket_type": "投诉"}, "c9"),
        ("then", ["我只能为您建议按钮。"]),
    ])
    out = await node.ainvoke(new_turn_state("帮我建个工单"))
    tool_msgs = [m for m in out["turn_messages"] if m.type == "tool"]
    assert tool_msgs[0].status == "error"  # 未注册工具,不写库(无 session_factory 可写)


async def test_suggest_options_invalid_args_no_suggestion():
    node = _agent([
        _tool_chunk("suggest_options", {"options": ["建工单"]}, "s1"),  # 缺 ticket_type
        ("then", ["抱歉。"]),
    ])
    out = await node.ainvoke(new_turn_state("投诉"))
    assert out["suggested_actions"] == []
    tool_msgs = [m for m in out["turn_messages"] if m.type == "tool"]
    assert tool_msgs[0].status == "error"


async def test_empty_text_falls_back():
    from app.prompts.service import FALLBACK_ANSWER
    node = _agent([""])
    out = await node.ainvoke(new_turn_state("q"))
    assert out["final_text"] == FALLBACK_ANSWER


from app.prompts.service import AGENT_BUDGET_ANSWER


async def test_step_budget_exhausted_after_tool_group_completed():
    node = _agent([
        _tool_chunk("query_order", {"order_id": "1"}, "c1"),
        ("then", [_tool_chunk("query_logistics", {"order_id": "1"}, "c2")]),
    ], max_agent_steps=2)
    out = await node.ainvoke(new_turn_state("一直查"))
    assert out["final_text"] == AGENT_BUDGET_ANSWER
    assert out["agent_steps"] == 2
    # 悬空 tool_calls 不留:第二组 ToolMessage 补齐后才兜底
    assert out["turn_messages"][-1].content == AGENT_BUDGET_ANSWER
    assert out["turn_messages"][-2].type == "tool" and out["turn_messages"][-2].tool_call_id == "c2"


async def test_token_budget_blocks_call_before_it_happens():
    node = _agent(["文本"], max_agent_tokens=10)  # 预算极小,首轮预留即超
    out = await node.ainvoke(new_turn_state("q"))
    assert out["final_text"] == AGENT_BUDGET_ANSWER
    assert out["agent_steps"] == 0  # 一次模型调用都没发起


async def test_usage_settled_once_when_present():
    script = [("tool", [{"index": 0, "name": "query_order", "id": "c1",
                         "args": '{"order_id":"1"}'}]),
              ("usage", {"input_tokens": 100, "output_tokens": 5}),
              ("then", ["好的", ("usage", {"input_tokens": 120, "output_tokens": 10})])]
    node = _agent(script)
    out = await node.ainvoke(new_turn_state("查订单"))
    assert out["agent_tokens"] == 100 + 5 + 120 + 10  # 两次调用各结算一次
    assert out["token_accounting"] == "usage"


async def test_suggest_options_ticket_type_on_create_ticket_regardless_of_order():
    """建工单不在末位时 ticket_type 也必须挂 create_ticket(spec §8:仅建工单选项携带)。"""
    node = _agent([
        _tool_chunk("suggest_options", {"options": ["建工单", "转人工"],
                                        "ticket_type": "售后"}, "s1"),
        ("then", ["好的。"]),
    ])
    out = await node.ainvoke(new_turn_state("你们这服务不行"))
    actions = out["suggested_actions"]
    assert actions[0]["action"] == "create_ticket" and actions[0]["ticket_type"] == "售后"
    assert actions[1]["action"] == "transfer_human" and "ticket_type" not in actions[1]


async def test_missing_usage_settles_as_estimated_reserve():
    """缺失 usage:按非零预留结算,token_accounting == estimated(spec §7.3)。"""
    node = _agent(["直接答复。"])
    out = await node.ainvoke(new_turn_state("你好"))
    assert out["token_accounting"] == "estimated"
    assert out["agent_tokens"] > 0


async def test_fixed_fallback_counts_toward_output_cap():
    """固定兜底计入可见输出长度:流满上限后追加 FALLBACK 即超限(spec §7.2)。"""
    from app.graph.errors import TurnAbortError
    node = _agent([
        _tool_chunk("query_order", {"order_id": "1"}, "c1"), "1234567890",
        ("then", [("finish", "stop")]),
    ], max_message_chars=10)
    with pytest.raises(TurnAbortError):
        await node.ainvoke(new_turn_state("q"))


# ---- ch06 Task 6:分支绑定 / query_faq 收尾 / order_context / 申请退款 ----

async def test_business_branch_binds_faq_and_refund_branch_not():
    # 经 main_agent 节点直驱:检查 model.received_tools / registry 行为
    model = FakeStreamModel(["答复。"])
    deps = GraphDeps(model=model, settings=make_settings(), retriever=None, store=None)
    g = _agent_graph(deps)
    st = new_turn_state("保温杯有库存吗")
    st.update({"resolved_query": "保温杯有库存吗", "route": "business",
               "intent": "商品咨询"})
    await g.ainvoke(st, config=_cfg())
    assert model.received_tools[0] == ["query_order", "query_product",
                                       "query_logistics", "query_faq", "suggest_options"]
    st2 = new_turn_state("这单能退吗")
    st2.update({"resolved_query": "订单 1111-1001 能退吗", "route": "refund",
                "intent": "退款退货", "refund_mode": "order_specific",
                "order_context": {"order_id": "1111-1001", "product": "保温杯",
                                  "status": "已完成", "amount": 89.0,
                                  "created_at": "2026-09-16T12:00:00",
                                  "delivered_at": "2026-09-17T12:00:00",
                                  "returnable_note": "普通商品,在 7 天无理由退货期内",
                                  "queried_at": "2026-09-19T12:00:00"}})
    model2 = FakeStreamModel(["可以退。"])
    await _agent_graph(GraphDeps(model=model2, settings=make_settings(),
                                 retriever=None, store=None)).ainvoke(st2, config=_cfg())
    assert model2.received_tools[0] == ["query_order", "query_product",
                                        "query_logistics", "suggest_options"]


async def test_clarify_branch_has_no_tools():
    model = FakeStreamModel(["请问您想咨询哪方面的问题呢?"])
    st = new_turn_state("那个事")
    st.update({"resolved_query": "那个事", "route": "refund", "refund_mode": "clarify"})
    await _agent_graph(GraphDeps(model=model, settings=make_settings(),
                                 retriever=None, store=None)).ainvoke(st, config=_cfg())
    assert model.received_tools[0] is None  # 不绑定工具


async def test_faq_low_confidence_terminal_refusal_without_second_model_call():
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    rt = _FakeRetriever(_result(low=True, hits=[], score=0.01))
    model = FakeStreamModel([
        ("tool", [{"index": 0, "name": "query_faq", "id": "f1", "args": "{}"}]),
        ("then", ["不该出现的第二次答复"]),
    ])
    st = new_turn_state("火星特产能退吗")
    st.update({"resolved_query": "火星特产能退吗", "route": "business"})
    out = await _agent_graph(GraphDeps(model=model, settings=make_settings(),
                                       retriever=rt, store=None)).ainvoke(st, config=_cfg())
    assert out["final_text"] == REFUSAL_ANSWER
    assert out["retrieval_status"] == "low_confidence"
    assert out["low_conf_source"] == "retrieval_low_conf"
    assert model._scripts == [["不该出现的第二次答复"]]  # 不再调用模型


async def test_faq_ok_writes_back_evidence_and_status():
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    rt = _FakeRetriever(_result())
    model = FakeStreamModel([
        ("tool", [{"index": 0, "name": "query_faq", "id": "f1", "args": "{}"}]),
        ("then", ["7 天无理由[1]。"]),
    ])
    st = new_turn_state("退货政策")
    st.update({"resolved_query": "退货政策是什么", "route": "business"})
    out = await _agent_graph(GraphDeps(model=model, settings=make_settings(),
                                       retriever=rt, store=None)).ainvoke(st, config=_cfg())
    assert out["retrieval_status"] == "ok" and out["evidence"]
    assert out["retrieval_result"]["hits"]


async def test_suggest_refund_requires_order_context():
    # 无 order_context:错误 ToolMessage,不出按钮
    model = FakeStreamModel([
        ("tool", [{"index": 0, "name": "suggest_options", "id": "s1",
                   "args": '{"options":["申请退款"]}'}]),
        ("then", ["好的。"]),
    ])
    st = new_turn_state("能退吗")
    st.update({"resolved_query": "能退吗", "route": "business"})
    out = await _agent_graph(GraphDeps(model=model, settings=make_settings(),
                                       retriever=None, store=None)).ainvoke(st, config=_cfg())
    assert out["suggested_actions"] == []
    # 有 order_context:按钮绑定服务端订单号(order_context 为 asdict(OrderInfo) 全字段)
    st2 = {**st, "order_context": {"order_id": "1111-1001", "product": "保温杯",
                                   "status": "已完成", "amount": 89.0,
                                   "created_at": "2026-09-16T12:00:00",
                                   "delivered_at": "2026-09-17T12:00:00",
                                   "returnable_note": "普通商品,在 7 天无理由退货期内",
                                   "queried_at": "2026-09-19T12:00:00"}}
    model2 = FakeStreamModel([
        ("tool", [{"index": 0, "name": "suggest_options", "id": "s1",
                   "args": '{"options":["申请退款"]}'}]),
        ("then", ["可以退,请点下方按钮。"]),
    ])
    out2 = await _agent_graph(GraphDeps(model=model2, settings=make_settings(),
                                        retriever=None, store=None)).ainvoke(st2, config=_cfg())
    assert out2["suggested_actions"] == [
        {"action": "refund_form", "label": "申请退款", "order_id": "1111-1001"}]


async def test_order_context_from_tool_only_when_user_mentioned():
    st = new_turn_state("帮我查订单 1111-1001")
    st.update({"resolved_query": "帮我查订单 1111-1001", "route": "business"})
    model = FakeStreamModel([
        ("tool", [{"index": 0, "name": "query_order", "id": "c1",
                   "args": '{"order_id":"1111-1001"}'}]),
        ("then", ["订单已完成。"]),
    ])
    out = await _agent_graph(GraphDeps(model=model, settings=make_settings(),
                                       retriever=None, store=None)).ainvoke(st, config=_cfg(TEST_USER_ID))
    assert out["order_context"]["order_id"] == "1111-1001"
    # 模型自猜的订单号(用户没提)不建焦点:工具照常执行成功,但不写 order_context
    st2 = new_turn_state("帮我查下订单")
    st2.update({"resolved_query": "帮我查下订单", "route": "business"})
    model2 = FakeStreamModel([
        ("tool", [{"index": 0, "name": "query_order", "id": "c1",
                   "args": '{"order_id":"1111-1001"}'}]),
        ("then", ["订单已完成。"]),
    ])
    out2 = await _agent_graph(GraphDeps(model=model2, settings=make_settings(),
                                        retriever=None, store=None)).ainvoke(st2, config=_cfg(TEST_USER_ID))
    tool_msgs = [m for m in out2["turn_messages"] if m.type == "tool"]
    assert [t.name for t in tool_msgs] == ["query_order"] and tool_msgs[0].status != "error"
    assert out2["order_context"] is None
