"""ch06 图 State:字段语义与每轮重置契约见 ch06 spec §10。"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")

# 纯 intent 映射(needs_knowledge 维度已移除,ch06 spec D2)
ROUTE_TABLE: dict[str, str] = {
    "物流": "business",
    "订单": "business",
    "商品咨询": "business",
    "退款退货": "refund",
    "售后": "refund",
    "投诉": "complaint",
    "闲聊": "chitchat",
    "其他": "other",
}


class ChatGraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # 跨轮,仅 log 提交成功后追加
    active_order: dict | None            # 跨轮,{order_id, source_message_id};不重置
    raw_query: str
    resolved_query: str
    understanding_degraded: bool
    intent: str | None
    intent_confidence: float | None      # 只记录不路由(D5)
    refund_mode: str | None              # general | order_specific | clarify(仅 refund 分支)
    route: str | None
    order_context: dict | None           # 本轮已校验订单快照(asdict(OrderInfo))
    retrieval_result: dict | None        # RetrievalResult 可序列化快照
    retrieval_status: str                # not_run | ok | low_confidence | unavailable
    retrieval_error_code: str | None     # kb_unconfigured/kb_rebuilding/kb_rebuild_required/kb_unavailable
    evidence: list[dict]
    expanded_queries: list[str]
    low_conf_source: str | None          # retrieval_low_conf | self_check
    low_conf_reason: dict | None
    suggested_actions: list[dict]
    agent_steps: int
    agent_tokens: int
    token_accounting: str                # none | usage | estimated | mixed
    final_text: str
    turn_messages: list[BaseMessage]
    source_message_id: str | None
    node_trace: list[dict]


def new_turn_state(raw_query: str) -> dict:
    """每轮图调用的显式输入:全部临时字段重置(messages/active_order 键刻意缺席:
    前者保留 checkpoint 历史,后者是跨轮订单焦点,ch06 spec §10)。"""
    return {
        "raw_query": raw_query,
        "resolved_query": "",
        "understanding_degraded": False,
        "intent": None,
        "intent_confidence": None,
        "refund_mode": None,
        "route": None,
        "order_context": None,
        "retrieval_result": None,
        "retrieval_status": "not_run",
        "retrieval_error_code": None,
        "evidence": [],
        "expanded_queries": [],
        "low_conf_source": None,
        "low_conf_reason": None,
        "suggested_actions": [],
        "agent_steps": 0,
        "agent_tokens": 0,
        "token_accounting": "none",
        "final_text": "",
        "turn_messages": [HumanMessage(content=raw_query)],
        "source_message_id": None,
        "node_trace": [],
    }
