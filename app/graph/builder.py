"""ch05 图装配:确定性 Workflow 骨架,分流规则经写死的路由表生效。"""

from langgraph.graph import END, START, StateGraph

from app.graph.agent_node import build_agent_node
from app.graph.nodes import (
    GraphDeps, build_fixed_nodes, build_front_nodes, build_knowledge_nodes,
    build_log_node, route_by_intent,
)
from app.graph.state import ChatGraphState


def build_chat_graph(deps: GraphDeps, checkpointer):
    g = StateGraph(ChatGraphState)
    front = build_front_nodes(deps)
    knowledge = build_knowledge_nodes(deps)
    fixed = build_fixed_nodes()

    g.add_node("resolve_reference", front["resolve_reference"])
    g.add_node("classify_intent", front["classify_intent"])
    g.add_node("retrieve", knowledge["retrieve"])
    g.add_node("confidence_gate", knowledge["confidence_gate"])
    g.add_node("gate_fallback", knowledge["gate_fallback"])
    g.add_node("complaint_reply", fixed["complaint_reply"])
    g.add_node("chitchat_reply", fixed["chitchat_reply"])
    g.add_node("main_agent", build_agent_node(deps))
    g.add_node("log", build_log_node(deps))

    g.add_edge(START, "resolve_reference")
    g.add_edge("resolve_reference", "classify_intent")
    g.add_conditional_edges(
        "classify_intent", route_by_intent,
        {"knowledge": "retrieve", "business": "main_agent",
         "complaint": "complaint_reply", "chitchat": "chitchat_reply"})
    g.add_edge("retrieve", "confidence_gate")
    g.add_conditional_edges(
        "confidence_gate", knowledge["route_after_gate"],
        {"main_agent": "main_agent", "gate_fallback": "gate_fallback"})
    for node in ("main_agent", "gate_fallback", "complaint_reply", "chitchat_reply"):
        g.add_edge(node, "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=checkpointer)
