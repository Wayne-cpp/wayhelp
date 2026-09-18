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
from tests.conftest import FakeStreamModel, make_settings


def _tool_chunk(name, args, call_id):
    return ("tool", [{"index": 0, "name": name, "id": call_id,
                      "args": json.dumps(args)}])


def _agent(script, **settings_over):
    """main_agent 包进最小编译图返回(START → main_agent → END)。"""
    deps = GraphDeps(model=FakeStreamModel(script),
                     settings=make_settings(**settings_over),
                     retriever=None, store=None)
    g = StateGraph(ChatGraphState)
    g.add_node("main_agent", build_agent_node(deps))
    g.add_edge(START, "main_agent")
    g.add_edge("main_agent", END)
    return g.compile()


async def test_one_step_converge():
    node = _agent(["订单 1001 已发货。"])
    out = await node.ainvoke(new_turn_state("订单 1001 状态"))
    assert out["final_text"] == "订单 1001 已发货。"
    assert out["agent_steps"] == 1
    assert out["turn_messages"][-1].content == "订单 1001 已发货。"


async def test_multi_step_order_then_logistics():
    node = _agent([
        _tool_chunk("query_order", {"order_id": "1001"}, "c1"), ("then", [
            _tool_chunk("query_logistics", {"order_id": "1001"}, "c2"), ("then", [
                "订单已发货,物流派送中。"])]),
    ])
    out = await node.ainvoke(new_turn_state("订单 1001 到哪了,先查订单再查物流"))
    assert out["agent_steps"] == 3
    tools = [m for m in out["turn_messages"] if m.type == "tool"]
    assert [t.name for t in tools] == ["query_order", "query_logistics"]
    assert "1001" in tools[0].content  # 真实工具结果喂回
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
