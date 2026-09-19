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
from app.prompts.refund import REFUND_SCOPE_PROMPT
from app.prompts.understand import UNDERSTAND_PROMPT
from app.services import orders

logger = logging.getLogger("wayhelp.graph")


@dataclass(frozen=True)
class GraphDeps:
    model: Any
    settings: Settings
    retriever: Any   # KnowledgeRetriever | None(测试可注假检索器)
    store: Any       # SessionStore 协议(log 节点用;生产装配必连)
    system_prompt: str = ""  # Task 13 装配传 SERVICE_SYSTEM_PROMPT


def parse_intent_output(text: str) -> tuple[str, float | None] | None:
    """提取 {"intent","confidence"};intent 非法 → None;confidence 非法置 None 不拒判。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent = data.get("intent")
    if intent not in INTENTS:
        return None
    conf = data.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        conf = None
    elif not 0.0 <= float(conf) <= 1.0:
        conf = None
    return intent, (float(conf) if conf is not None else None)


def parse_understand_output(text: str) -> str | None:
    """提取 {"resolved_query": str};空串=透传标记(调用方回填 raw_query);非法 → None。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q = data.get("resolved_query")
    if not isinstance(q, str):
        return None
    return q


def parse_refund_scope_output(text: str) -> str | None:
    """提取 {"mode": ...};三枚举之外 → None(调用方降级 clarify)。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    mode = data.get("mode")
    return mode if mode in ("general", "order_specific", "clarify") else None


def parse_expand_output(text: str) -> list[str] | None:
    """提取 {"queries": [...]};结构非法 → None;合法则去空去重(可空 list)。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    queries = data.get("queries")
    if not isinstance(queries, list):
        return None
    out: list[str] = []
    for q in queries:
        if isinstance(q, str) and q.strip() and q.strip() not in out:
            out.append(q.strip())
    return out


def route_by_intent(state) -> str:
    """条件边函数:只查写死的分流表。classify_intent 已保证 route 非 None。"""
    return state["route"]


def _history_turns(messages, max_turns: int) -> list:
    """按 HumanMessage 切轮,取最近 max_turns 个完整轮(轮末须有后续消息;末尾未成轮不切)。"""
    turns: list[list] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            turns.append([m])
        elif turns:
            turns[-1].append(m)
    return [t for t in turns if len(t) >= 2][-max_turns:]


def _render_history(turns) -> str:
    lines: list[str] = []
    for t in turns:
        for m in t:
            if isinstance(m, HumanMessage):
                lines.append(f"用户:{m.content}")
            elif isinstance(m, AIMessage) and m.content:
                lines.append(f"助手:{m.content}")
    return "\n".join(lines)


def build_front_nodes(deps: GraphDeps) -> dict:
    async def understand_query(state, config):
        trace = [*state["node_trace"], {"node": "understand_query"}]
        user_id = (config.get("configurable") or {}).get("user_id", "")
        active = state.get("active_order")
        active_summary = None
        if active:
            o = orders.get_order(user_id, active["order_id"])
            if o is None:
                logger.info("node=understand_query active_order 失效清空: %s", active["order_id"])
                active = None
            else:
                active_summary = orders.order_summary(o)
        turns = _history_turns(state.get("messages") or [],
                               deps.settings.understand_history_turns)
        if not turns and active_summary is None:
            return {"resolved_query": state["raw_query"], "active_order": active,
                    "node_trace": trace}
        block_parts = []
        if turns:
            block_parts.append("对话历史:\n" + _render_history(turns))
        if active_summary:
            block_parts.append("已确认会话订单:" + active_summary)
        prompt = (UNDERSTAND_PROMPT
                  .replace("{history_block}", "\n".join(block_parts))
                  .replace("{query}", state["raw_query"]))
        try:
            resp = await deps.model.ainvoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else ""
            parsed = parse_understand_output(text)
            if parsed is None:
                raise ValueError("parse_understand_output")
            resolved = parsed.strip() or state["raw_query"]  # 空串 = 透传
            # 订单号来源校验:输出中的订单号必须可追溯到原文或已校验 active_order(spec §6.1)
            out_ids = set(orders.find_order_ids(resolved))
            src_ids = set(orders.find_order_ids(state["raw_query"]))
            if active:
                src_ids.add(active["order_id"])
            if not out_ids <= src_ids:
                raise ValueError("hallucinated order id")
        except Exception as exc:
            logger.warning("node=understand_query 降级透传: %s", type(exc).__name__)
            return {"resolved_query": state["raw_query"], "understanding_degraded": True,
                    "active_order": active,
                    "node_trace": [*trace[:-1], {"node": "understand_query", "degraded": True}]}
        logger.info("node=understand_query resolved=%r", resolved[:60])
        return {"resolved_query": resolved, "active_order": active, "node_trace": trace}

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
            logger.warning("node=classify_intent 解析失败,兜底「其他」: %r", text[:120])
            intent, confidence = "其他", None
        else:
            intent, confidence = parsed
        route = ROUTE_TABLE[intent]
        logger.info("node=classify_intent intent=%s confidence=%s route=%s",
                    intent, confidence, route)
        return {"intent": intent, "intent_confidence": confidence, "route": route,
                "node_trace": [*state["node_trace"],
                               {"node": "classify_intent", "intent": intent,
                                "confidence": confidence, "route": route}]}

    async def refund_scope(state):
        prompt = REFUND_SCOPE_PROMPT.replace("{query}", state["resolved_query"])
        trace = [*state["node_trace"], {"node": "refund_scope"}]
        try:
            resp = await deps.model.ainvoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else ""
            mode = parse_refund_scope_output(text)
            if mode is None:
                logger.warning("node=refund_scope 解析失败降级 clarify: %r", text[:120])
                mode = "clarify"
        except Exception as exc:
            logger.warning("node=refund_scope 模型异常降级 clarify: %s", type(exc).__name__)
            mode = "clarify"
        logger.info("node=refund_scope mode=%s", mode)
        return {"refund_mode": mode,
                "node_trace": [*trace[:-1], {"node": "refund_scope", "mode": mode}]}

    return {"understand_query": understand_query, "classify_intent": classify_intent,
            "refund_scope": refund_scope}


# ── ch06 Task 7:refund 子流程(refund_prepare 挂起 / refund_policy 扩写统一重排)──

import time

from langgraph.types import interrupt

from app.prompts.expand import EXPAND_PROMPT


def route_refund_mode(state) -> str:
    return state["refund_mode"]


def route_after_prepare(state) -> str:
    return "refund_policy" if state.get("order_context") else "other_fallback"


def build_refund_nodes(deps: GraphDeps) -> dict:
    async def refund_prepare(state, config):
        user_id = (config.get("configurable") or {}).get("user_id", "")
        trace = [*state["node_trace"], {"node": "refund_prepare"}]
        valid = []
        for oid in orders.find_order_ids(state["resolved_query"]):
            o = orders.get_order(user_id, oid)
            if o is not None:
                valid.append(o)
        if len(valid) == 1:  # 唯一候选且属于该用户 → 直通(spec §6.3)
            logger.info("node=refund_prepare direct order=%s", valid[0].order_id)
            return {"order_context": asdict(valid[0]), "node_trace": trace}
        candidates = orders.list_orders(user_id)
        if not candidates:
            logger.warning("node=refund_prepare 无可选订单,转 other_fallback")
            return {"order_context": None, "node_trace": trace}
        # 挂起:帧由驱动层在图结束后按 pending interrupt 发射,节点不发 SSE
        selected = interrupt({"type": "order_selector",
                              "orders": [orders.order_brief(o) for o in candidates]})
        oid = (selected or {}).get("order_id", "")
        o = orders.get_order(user_id, oid)  # resume 重跑后的防御性校验
        if o is None:
            logger.warning("node=refund_prepare resume 校验失败: %r", oid)
            return {"order_context": None, "node_trace": trace}
        logger.info("node=refund_prepare resumed order=%s", o.order_id)
        return {"order_context": asdict(o), "node_trace": trace}

    async def refund_policy(state):
        settings = deps.settings
        trace = [*state["node_trace"], {"node": "refund_policy"}]
        if deps.retriever is None:
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unconfigured", "node_trace": trace}
        oc = state.get("order_context")
        base_query = state["resolved_query"]
        if oc:  # 附加订单事实仅用于检索;raw_query 落库口径不变(spec §6.4)
            base_query = (f"{state['resolved_query']}"
                          f"(商品:{oc['product']},{oc['returnable_note']},状态:{oc['status']})")
        queries = [base_query]
        expand_note = None
        if (oc and settings.refund_expand_enabled
                and settings.knowledge_strategy == "hybrid_rerank"):
            prompt = (EXPAND_PROMPT
                      .replace("{order_summary}", orders.order_summary(oc))
                      .replace("{query}", state["resolved_query"]))
            try:
                resp = await deps.model.ainvoke([HumanMessage(content=prompt)])
                text = resp.content if isinstance(resp.content, str) else ""
                extra = parse_expand_output(text)
                if extra is None:
                    raise ValueError("parse_expand_output")
                for q in extra:
                    if q not in queries and len(queries) < 4:
                        queries.append(q)
            except Exception as exc:
                logger.warning("node=refund_policy 扩写降级单查询: %s", type(exc).__name__)
                expand_note = f"expand_degraded:{type(exc).__name__}"
        timeout = settings.knowledge_tool_timeout_seconds
        deadline = time.monotonic() + timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[asyncio.to_thread(deps.retriever.search, q,
                                                   scope=None, deadline=deadline)
                                 for q in queries]),
                timeout=timeout)
        except Exception as exc:
            logger.warning("node=refund_policy 检索不可用: %s", type(exc).__name__)
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unavailable",
                    "expanded_queries": queries, "node_trace": trace}
        for r in results:  # 维护态:任一查询命中维护 note 即整轮不可用(spec §6.4)
            if _STATE_NOTE_CODES.get(r.note) is not None:
                out = state_fields_for_result(r)
                out.update({"expanded_queries": queries, "node_trace": trace})
                return out
        base = results[0]
        result = base
        if len(queries) > 1 and settings.knowledge_strategy == "hybrid_rerank":
            candidates, seen = [], set()
            for r in results:  # 只按 chunk_id 合并候选,不比较/混合各查询分数(D10)
                for h in r.hits:
                    if h.chunk_id not in seen:
                        seen.add(h.chunk_id)
                        candidates.append(h)
            candidates = candidates[: 4 * settings.rerank_top_n]
            unified = None
            if candidates:
                try:
                    unified = await asyncio.wait_for(
                        asyncio.to_thread(deps.retriever.rerank_candidates,
                                          base_query, candidates, deadline),
                        timeout=max(deadline - time.monotonic(), 0.001))
                except Exception as exc:
                    logger.warning("node=refund_policy 统一重排失败: %s", type(exc).__name__)
                    unified = None
            if unified is not None:
                result = unified
            else:
                note = ";".join(x for x in
                                (base.note, expand_note or "unified_rerank_failed") if x)
                result = replace(base, note=note or None)  # 回退 base 完整策略/分数/阈值
        elif expand_note:
            result = replace(base, note=";".join(x for x in (base.note, expand_note) if x) or None)
        out = state_fields_for_result(result)
        out["expanded_queries"] = queries
        if out["retrieval_status"] == "low_confidence":
            out.update(low_conf_fields(out["retrieval_result"]))
        elif out["retrieval_status"] == "ok":
            out["evidence"] = evidence_dicts_from_snapshot(out["retrieval_result"], settings)
        out["node_trace"] = trace
        logger.info("node=refund_policy status=%s queries=%d", out["retrieval_status"], len(queries))
        return out

    return {"refund_prepare": refund_prepare, "refund_policy": refund_policy,
            "route_refund_mode": route_refund_mode, "route_after_prepare": route_after_prepare}


import asyncio
from dataclasses import asdict, is_dataclass, replace

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


def state_fields_for_result(result: RetrievalResult) -> dict:
    """RetrievalResult → retrieve 节点同款 state 字段(note→error_code 优先于低置信)。"""
    snap = snapshot_retrieval(result)
    code = _STATE_NOTE_CODES.get(result.note)
    if code is not None:
        return {"retrieval_result": snap, "retrieval_status": "unavailable",
                "retrieval_error_code": code}
    return {"retrieval_result": snap,
            "retrieval_status": "low_confidence" if result.low_confidence else "ok"}


def evidence_dicts_from_snapshot(snap: dict, settings) -> list[dict]:
    """快照 hits → assemble_evidence 分配引用号(confidence_gate 同款预算)。"""
    hits = [KnowledgeHit(**h) for h in snap["hits"]]
    if not hits:
        return []
    return [e.to_dict() for e in assemble_evidence(
        hits, max_items=settings.rerank_top_n,
        budget_chars=settings.max_tool_result_chars, overhead_chars=200)]


def low_conf_fields(snap: dict) -> dict:
    """低置信入池字段(confidence_gate 同款 reason 结构)。"""
    return {"low_conf_source": "retrieval_low_conf",
            "low_conf_reason": {
                "requested_strategy": snap["requested_strategy"],
                "effective_strategy": snap["effective_strategy"],
                "top1": snap["confidence_score"],
                "threshold": snap["confidence_threshold"],
                "note": snap["note"]}}


def route_after_gate(state) -> str:
    return "main_agent" if state["retrieval_status"] == "ok" else "gate_fallback"


def build_knowledge_nodes(deps: GraphDeps) -> dict:
    # ch06 Task 7:retrieve/confidence_gate 节点删除(语义并入 refund_policy);
    # gate_fallback 与 helpers 保留,route_after_gate 提为模块级供 builder import。

    async def gate_fallback(state):
        writer = get_stream_writer()
        text = (KB_UNAVAILABLE_ANSWER if state["retrieval_status"] == "unavailable"
                else REFUSAL_ANSWER)
        writer(ev_fixed_delta(text))
        logger.info("node=gate_fallback status=%s", state["retrieval_status"])
        return {"final_text": text,
                "turn_messages": [*state["turn_messages"], AIMessage(content=text)],
                "node_trace": [*state["node_trace"], {"node": "gate_fallback"}]}

    return {"gate_fallback": gate_fallback}


from app.prompts.service import CHITCHAT_REPLY, COMPLAINT_REPLY, FALLBACK_ANSWER

COMPLAINT_ACTIONS = [
    {"action": "transfer_human", "label": "转人工"},
    {"action": "create_ticket", "label": "建工单", "ticket_type": "投诉"},
]


def build_fixed_nodes() -> dict:
    async def complaint_reply(state):
        writer = get_stream_writer()
        writer(ev_fixed_delta(COMPLAINT_REPLY))
        logger.info("node=complaint_reply")
        return {"final_text": COMPLAINT_REPLY,
                "suggested_actions": [dict(a) for a in COMPLAINT_ACTIONS],
                "turn_messages": [*state["turn_messages"], AIMessage(content=COMPLAINT_REPLY)],
                "node_trace": [*state["node_trace"], {"node": "complaint_reply"}]}

    async def chitchat_reply(state):
        writer = get_stream_writer()
        writer(ev_fixed_delta(CHITCHAT_REPLY))
        logger.info("node=chitchat_reply")  # 零模型调用
        return {"final_text": CHITCHAT_REPLY,
                "turn_messages": [*state["turn_messages"], AIMessage(content=CHITCHAT_REPLY)],
                "node_trace": [*state["node_trace"], {"node": "chitchat_reply"}]}

    async def other_fallback(state):
        writer = get_stream_writer()
        writer(ev_fixed_delta(FALLBACK_ANSWER))
        logger.info("node=other_fallback")  # 零模型调用
        return {"final_text": FALLBACK_ANSWER,
                "turn_messages": [*state["turn_messages"], AIMessage(content=FALLBACK_ANSWER)],
                "node_trace": [*state["node_trace"], {"node": "other_fallback"}]}

    return {"complaint_reply": complaint_reply, "chitchat_reply": chitchat_reply,
            "other_fallback": other_fallback}


import asyncio
import contextlib

from langchain_core.messages import ToolMessage

from app.graph.events import ev_citations, ev_suggest_actions
from app.prompts.service import AGENT_BUDGET_ANSWER
from app.sessions import LowConfidenceRecord, StoredMessage
from app.tool_envelope import wrap


def _to_stored(turn_messages, settings) -> list[StoredMessage]:
    """本轮消息 → 落库行;临时 SystemMessage 不落库;ToolMessage 打 envelope。"""
    out: list[StoredMessage] = []
    for m in turn_messages:
        if isinstance(m, HumanMessage):
            out.append(StoredMessage("user", m.content))
        elif isinstance(m, AIMessage):
            out.append(StoredMessage("assistant", m.content or None,
                                     tool_calls=m.tool_calls or None))
        elif isinstance(m, ToolMessage):
            ok = m.status != "error"
            out.append(StoredMessage(
                "tool",
                wrap(m.content, ok,
                     None if ok else m.additional_kwargs.get("error_code", "tool_error"),
                     settings.max_tool_result_chars),
                tool_call_id=m.tool_call_id))
    return out


def _build_low_conf(state, cid: int | None) -> LowConfidenceRecord | None:
    if state["low_conf_source"] == "retrieval_low_conf":
        return LowConfidenceRecord(
            raw_question=state["raw_query"], source="retrieval_low_conf",
            reason=json.dumps(state["low_conf_reason"], ensure_ascii=False),
            conversation_id=cid)
    if (state["retrieval_status"] == "ok"
            and state["final_text"].strip() == REFUSAL_ANSWER):
        refs = [{"ref_no": e["ref_no"], "chunk_id": e["chunk_id"]}
                for e in state["evidence"]]
        return LowConfidenceRecord(
            raw_question=state["raw_query"], source="self_check",
            reason=json.dumps({"evidence_refs": refs}, ensure_ascii=False),
            conversation_id=cid)
    return None


def _should_cite(state) -> bool:
    if state["retrieval_status"] != "ok" or not state["evidence"]:
        return False
    final = state["final_text"].strip()
    if final in (REFUSAL_ANSWER, AGENT_BUDGET_ANSWER, KB_UNAVAILABLE_ANSWER):
        return False
    return re.search(r"\[\d{1,2}\]", final) is not None


def build_log_node(deps: GraphDeps):
    async def log_turn(state, config):
        writer = get_stream_writer()
        sid = config["configurable"]["thread_id"]
        stored = _to_stored(state["turn_messages"], deps.settings)
        cid = int(sid) if sid.isdecimal() else None
        low_conf = _build_low_conf(state, cid)
        commit_task = asyncio.ensure_future(
            deps.store.commit_turn(sid, stored, low_confidence=low_conf))
        try:
            result = await asyncio.shield(commit_task)  # 取消时等事务落地
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await commit_task
            raise
        # 提交成功后才发 citations / suggest_actions(失败不得发按钮或成功终帧)
        if _should_cite(state):
            writer(ev_citations(state["evidence"]))
        if state["suggested_actions"]:
            writer(ev_suggest_actions(result.source_message_id,
                                      state["suggested_actions"]))
        logger.info("node=log session=%s intent=%s route=%s gate=%s steps=%s tokens=%s accounting=%s",
                    sid, state.get("intent"),
                    state.get("route"), state.get("retrieval_status"),
                    state.get("agent_steps"), state.get("agent_tokens"),
                    state.get("token_accounting"))
        out = {"messages": state["turn_messages"],
               "source_message_id": result.source_message_id,
               "node_trace": [*state["node_trace"], {"node": "log"}]}
        oc = state.get("order_context")
        if oc:  # 仅已完成轮建立/替换同 thread 订单焦点(spec §10)
            out["active_order"] = {"order_id": oc["order_id"],
                                   "source_message_id": result.source_message_id}
        return out

    return log_turn
