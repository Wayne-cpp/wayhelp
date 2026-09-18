"""ch05 图节点。节点经闭包捕获 GraphDeps;日志统一 logger 'wayhelp.graph'。"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
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


import asyncio
from dataclasses import asdict, is_dataclass

from app.knowledge.retriever import (
    NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED,
    KnowledgeHit, RetrievalResult, assemble_evidence,
)
from app.graph.events import ev_fixed_delta
from app.prompts.service import KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER

_STATE_NOTE_CODES = {
    NOTE_UNCONFIGURED: "kb_unconfigured",
    NOTE_REBUILDING: "kb_rebuilding",
    NOTE_REBUILD_REQUIRED: "kb_rebuild_required",
}


def snapshot_retrieval(r: RetrievalResult) -> dict:
    """RetrievalResult → 可进 checkpoint 的纯数据快照(不含 retriever/连接等运行时对象)。"""
    return {
        "requested_strategy": r.requested_strategy,
        "effective_strategy": r.effective_strategy,
        "confidence_score": r.confidence_score,
        "confidence_threshold": r.confidence_threshold,
        "low_confidence": r.low_confidence,
        "note": r.note,
        "leg_counts": dict(r.leg_counts),
        "hits": [asdict(h) for h in r.hits],
        "query_plan": asdict(r.query_plan) if is_dataclass(r.query_plan) else None,
    }


def build_knowledge_nodes(deps: GraphDeps) -> dict:
    async def retrieve(state):
        logger.info("node=retrieve query=%r", state["resolved_query"][:50])
        trace = [*state["node_trace"], {"node": "retrieve"}]
        if deps.retriever is None:
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unconfigured", "node_trace": trace}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(deps.retriever.search, state["resolved_query"]),
                timeout=deps.settings.knowledge_tool_timeout_seconds)  # 总等待上限,本层不重试
        except Exception as exc:  # 超时与检索异常同归 unavailable
            logger.warning("node=retrieve unavailable: %s", type(exc).__name__)
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unavailable", "node_trace": trace}
        snap = snapshot_retrieval(result)
        code = _STATE_NOTE_CODES.get(result.note)
        if code is not None:  # 配置/维护态优先于低置信判定
            return {"retrieval_result": snap, "retrieval_status": "unavailable",
                    "retrieval_error_code": code, "node_trace": trace}
        # NOTE_NOT_BUILT 等其余 note 参与低置信判定
        status = "low_confidence" if result.low_confidence else "ok"
        logger.info("node=retrieve done status=%s score=%s", status, result.confidence_score)
        return {"retrieval_result": snap, "retrieval_status": status, "node_trace": trace}

    async def confidence_gate(state):
        status = state["retrieval_status"]
        trace = [*state["node_trace"], {"node": "confidence_gate", "status": status}]
        if status == "low_confidence":
            r = state["retrieval_result"]
            logger.info("node=confidence_gate blocked low_confidence")
            return {"low_conf_source": "retrieval_low_conf",
                    "low_conf_reason": {
                        "requested_strategy": r["requested_strategy"],
                        "effective_strategy": r["effective_strategy"],
                        "top1": r["confidence_score"],
                        "threshold": r["confidence_threshold"],
                        "note": r["note"]},
                    "node_trace": trace}
        if status == "ok":
            hits = [KnowledgeHit(**h) for h in state["retrieval_result"]["hits"]]
            evidence = [e.to_dict() for e in assemble_evidence(
                hits, max_items=deps.settings.rerank_top_n,
                budget_chars=deps.settings.max_tool_result_chars,
                overhead_chars=200)] if hits else []
            return {"evidence": evidence, "node_trace": trace}
        return {"node_trace": trace}  # unavailable:直接落 fallback

    def route_after_gate(state) -> str:
        return "main_agent" if state["retrieval_status"] == "ok" else "gate_fallback"

    async def gate_fallback(state):
        writer = get_stream_writer()
        text = (KB_UNAVAILABLE_ANSWER if state["retrieval_status"] == "unavailable"
                else REFUSAL_ANSWER)
        writer(ev_fixed_delta(text))
        logger.info("node=gate_fallback status=%s", state["retrieval_status"])
        return {"final_text": text,
                "turn_messages": [*state["turn_messages"], AIMessage(content=text)],
                "node_trace": [*state["node_trace"], {"node": "gate_fallback"}]}

    return {"retrieve": retrieve, "confidence_gate": confidence_gate,
            "gate_fallback": gate_fallback, "route_after_gate": route_after_gate}
