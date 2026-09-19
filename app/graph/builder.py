"""ch05 图装配:确定性 Workflow 骨架,分流规则经写死的路由表生效。
ch06 Task 7:refund 出口切正式子流程(refund_scope → prepare/policy),旧 retrieve 链退役。"""

from langgraph.graph import END, START, StateGraph

from app.graph.agent_node import build_agent_node
from app.graph.nodes import (
    GraphDeps, build_fixed_nodes, build_front_nodes, build_knowledge_nodes,
    build_log_node, build_refund_nodes, route_after_gate, route_after_prepare,
    route_by_intent, route_refund_mode,
)
from app.graph.state import ChatGraphState


def build_chat_graph(deps: GraphDeps, checkpointer):
    g = StateGraph(ChatGraphState)
    front = build_front_nodes(deps)
    refund = build_refund_nodes(deps)
    knowledge = build_knowledge_nodes(deps)
    fixed = build_fixed_nodes()

    g.add_node("understand_query", front["understand_query"])
    g.add_node("classify_intent", front["classify_intent"])
    g.add_node("refund_scope", front["refund_scope"])
    g.add_node("refund_prepare", refund["refund_prepare"])
    g.add_node("refund_policy", refund["refund_policy"])
    g.add_node("gate_fallback", knowledge["gate_fallback"])
    g.add_node("complaint_reply", fixed["complaint_reply"])
    g.add_node("chitchat_reply", fixed["chitchat_reply"])
    g.add_node("other_fallback", fixed["other_fallback"])
    g.add_node("main_agent", build_agent_node(deps))
    g.add_node("log", build_log_node(deps))

    g.add_edge(START, "understand_query")
    g.add_edge("understand_query", "classify_intent")
    g.add_conditional_edges(
        "classify_intent", route_by_intent,
        {"business": "main_agent", "refund": "refund_scope",
         "complaint": "complaint_reply", "chitchat": "chitchat_reply",
         "other": "other_fallback"})
    g.add_conditional_edges(
        "refund_scope", route_refund_mode,
        {"general": "refund_policy", "order_specific": "refund_prepare",
         "clarify": "main_agent"})
    g.add_conditional_edges(
        "refund_prepare", route_after_prepare,
        {"refund_policy": "refund_policy", "other_fallback": "other_fallback"})
    g.add_conditional_edges(
        "refund_policy", route_after_gate,
        {"main_agent": "main_agent", "gate_fallback": "gate_fallback"})
    for node in ("main_agent", "gate_fallback", "complaint_reply", "chitchat_reply",
                 "other_fallback"):
        g.add_edge(node, "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=checkpointer)
