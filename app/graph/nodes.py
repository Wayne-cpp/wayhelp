"""ch05 图节点。节点经闭包捕获 GraphDeps;日志统一 logger 'wayhelp.graph'。"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from app.config import Settings
from app.graph.errors import TurnAbortError
from app.graph.events import ev_error
from app.graph.state import INTENTS, ROUTE_TABLE
from app.prompts.intent import INTENT_PROMPT

logger = logging.getLogger("wayhelp.graph")


@dataclass(frozen=True)
class GraphDeps:
    model: Any
    settings: Settings
    retriever: Any   # KnowledgeRetriever | None(测试可注假检索器)
    store: Any       # SessionStore 协议(log 节点用;本任务可为 None)


def parse_intent_output(text: str) -> tuple[str, bool] | None:
    """从模型输出提取 {"intent","needs_knowledge"};任何不合法一律 None(调用方兜底)。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent = data.get("intent")
    nk = data.get("needs_knowledge")
    if intent not in INTENTS or type(nk) is not bool:  # 字符串 "false"/缺失都算非法
        return None
    return intent, nk


def route_by_intent(state) -> str:
    """条件边函数:只查写死的分流表。classify_intent 已保证 route 非 None。"""
    return state["route"]


def build_front_nodes(deps: GraphDeps) -> dict:
    async def resolve_reference(state):
        # 最简版:原样透传(正式指代消解是后续章节的事)
        logger.info("node=resolve_reference query=%r", state["raw_query"][:50])
        return {"resolved_query": state["raw_query"],
                "node_trace": [*state["node_trace"], {"node": "resolve_reference"}]}

    async def classify_intent(state):
        writer = get_stream_writer()
        prompt = INTENT_PROMPT.replace("{query}", state["resolved_query"])
        try:
            resp = await deps.model.ainvoke([HumanMessage(content=prompt)])
        except Exception as exc:  # 网络故障不是分类结果
            logger.warning("node=classify_intent upstream error: %s", type(exc).__name__)
            writer(ev_error("upstream_error", "上游模型暂时不可用"))
            raise TurnAbortError("upstream_error") from exc
        text = resp.content if isinstance(resp.content, str) else ""
        parsed = parse_intent_output(text)
        if parsed is None:
            logger.warning("node=classify_intent 解析失败,保守兜底 knowledge: %r", text[:120])
            intent, needs_knowledge = "售后", True
        else:
            intent, needs_knowledge = parsed
        route = ROUTE_TABLE[(intent, needs_knowledge)]
        logger.info("node=classify_intent intent=%s needs_knowledge=%s route=%s",
                    intent, needs_knowledge, route)
        return {"intent": intent, "needs_knowledge": needs_knowledge, "route": route,
                "node_trace": [*state["node_trace"],
                               {"node": "classify_intent", "intent": intent,
                                "needs_knowledge": needs_knowledge, "route": route}]}

    return {"resolve_reference": resolve_reference, "classify_intent": classify_intent}
