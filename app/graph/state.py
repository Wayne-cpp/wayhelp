"""ch05 图 State:字段语义与每轮重置契约见 spec §10.1。"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")
ROUTES = ("knowledge", "business", "complaint", "chitchat")

# 分流规则写死:(intent, needs_knowledge) → route;needs_knowledge=True 优先于意图名称
ROUTE_TABLE: dict[tuple[str, bool], str] = {
    ("物流", False): "business",
    ("订单", False): "business",
    ("商品咨询", False): "business",
    ("售后", False): "business",
    ("退款退货", False): "knowledge",  # 保守覆盖:业务工具办不了退款退货,仍须先检索政策
    ("投诉", False): "complaint",
    ("闲聊", False): "chitchat",
    **{(intent, True): "knowledge" for intent in INTENTS},
}


class ChatGraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # 跨轮,仅 log 提交成功后追加
    raw_query: str
    resolved_query: str
    intent: str | None
    needs_knowledge: bool | None
    route: str | None
    retrieval_result: dict | None          # RetrievalResult 的可序列化快照
    retrieval_status: str                  # not_run | ok | low_confidence | unavailable
    retrieval_error_code: str | None       # kb_unconfigured/kb_rebuilding/kb_rebuild_required/kb_unavailable
    evidence: list[dict]
    low_conf_source: str | None            # retrieval_low_conf | self_check
    low_conf_reason: dict | None
    suggested_actions: list[dict]
    agent_steps: int
    agent_tokens: int
    token_accounting: str                  # none | usage | estimated | mixed
    final_text: str
    turn_messages: list[BaseMessage]       # 本轮消息(用户 + 工具往返 + 最终答复)
    source_message_id: str | None
    node_trace: list[dict]


def new_turn_state(raw_query: str) -> dict:
    """每轮图调用的显式输入:全部临时字段重置(messages 键刻意缺席,保留 checkpoint 历史)。
    同 thread 未覆盖字段会延续上一轮的值——所有分支都必须从同一个入口拿初值。"""
    return {
        "raw_query": raw_query,
        "resolved_query": "",
        "intent": None,
        "needs_knowledge": None,
        "route": None,
        "retrieval_result": None,
        "retrieval_status": "not_run",
        "retrieval_error_code": None,
        "evidence": [],
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
