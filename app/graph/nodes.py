"""ch05 图节点。节点经闭包捕获 GraphDeps;日志统一 logger 'wayhelp.graph'。"""

import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.config import Settings
from app.graph.errors import TurnAbortError
from app.graph.events import (
    ev_citations, ev_error, ev_fixed_delta, ev_suggest_actions,
)
from app.graph.state import INTENTS, ROUTE_TABLE
from app.knowledge.retriever import (
    NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED,
    KnowledgeHit, RetrievalResult, assemble_evidence,
)
from app.prompts.expand import EXPAND_PROMPT
from app.prompts.intent import INTENT_PROMPT
from app.prompts.refund import REFUND_SCOPE_PROMPT
from app.prompts.service import (
    AGENT_BUDGET_ANSWER, CHITCHAT_REPLY, COMPLAINT_REPLY, FALLBACK_ANSWER,
    KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER,
)
from app.prompts.understand import UNDERSTAND_PROMPT
from app.services import orders
from app.services.context_layers import (
    build_layered_view, evaluate_degrade, needs_reconcile, reconcile_checkpoint_ids,
    render_history_text,
)
from app.services.token_budget import compute_budget, measure_sys_tokens
from app.sessions import (
    ContextMeta, LowConfidenceRecord, PersistedTurn, StoredMessage,
)
from app.tools.catalog import TurnContext
from app.tools.executor import PendingWrite, ToolExecutor, ToolFace

logger = logging.getLogger("wayhelp.graph")


@dataclass(frozen=True)
class GraphDeps:
    model: Any
    settings: Settings
    retriever: Any   # KnowledgeRetriever | None(测试可注假检索器)
    store: Any       # SessionStore 协议(log 节点用;生产装配必连)
    system_prompt: str = ""  # Task 13 装配传 SERVICE_SYSTEM_PROMPT
    context_budget: Any = None  # ch07 ContextBudget(Task 9/10 消费;None 时按需现算)
    summary_runner: Any = None  # ch07 SummaryRunner(Task 10 消费;触发后台分段摘要)
    catalog: Any = None          # ch08 ToolCatalog(main_agent 工具面装配)
    mcp_gateway: Any = None      # ch08 McpGateway(每轮 MCP 发现;None = 未配置)
    session_factory: Any = None  # ch08 审计/写确认用


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
        sid = (config.get("configurable") or {}).get("thread_id", "")
        messages = list(state.get("messages") or [])
        stamped_back = None
        meta = ContextMeta(None, None, None)
        if messages:  # 有历史才读锚点/对齐(首轮零 store 访问)
            meta = await deps.store.get_context_meta(sid, user_id)
            if needs_reconcile(messages):
                records = await deps.store.list_checkpoint_records(sid, user_id) or []
                reconciled = reconcile_checkpoint_ids(messages, records)
                if reconciled is None:
                    logger.info("context_reconcile_failed session=%s reason=align", sid)
                    messages = []  # 对齐失败不猜 ID,历史整轮放弃(spec §2)
                else:
                    messages = stamped_back = reconciled
        view = build_layered_view(messages, meta, deps.settings)
        history_block = render_history_text(view)
        logger.info("history_ctx %s", json.dumps({
            "session": sid, "summary": view.summary,
            "window": [{"role": m.type,
                        "content": m.content if isinstance(m.content, str) else None,
                        "db_id": (m.additional_kwargs or {}).get("db_id")}
                       for m in [*view.layer2_rendered, *view.layer1]],
            "est": {"l1": view.l1_tokens, "l2": view.l2_tokens,
                    "summary": view.summary_tokens}}, ensure_ascii=False))
        base = {"history_block": history_block,
                "history_layer2": view.layer2_rendered,
                "history_layer1": view.layer1,
                "history_summary": view.summary}
        if stamped_back is not None:
            base["messages"] = stamped_back  # 同 .id 原地替换,把 db_id 写回 checkpoint
        if not history_block and active_summary is None:
            return {"resolved_query": state["raw_query"], "active_order": active,
                    "node_trace": trace, **base}
        block_parts = []
        if history_block:
            block_parts.append(history_block)
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
                    "node_trace": [*trace[:-1], {"node": "understand_query", "degraded": True}],
                    **base}
        logger.info("node=understand_query resolved=%r", resolved[:60])
        return {"resolved_query": resolved, "active_order": active,
                "node_trace": trace, **base}

    async def classify_intent(state):
        writer = get_stream_writer()
        prompt = (INTENT_PROMPT
                  .replace("{history_block}", state.get("history_block") or "(无对话历史)")
                  .replace("{query}", state["resolved_query"]))
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
        try:  # return_exceptions:仅扩写支路异常可丢弃;base 异常/总超时仍整轮不可用(spec §6.4)
            raw = await asyncio.wait_for(
                asyncio.gather(*[asyncio.to_thread(deps.retriever.search, q,
                                                   scope=None, deadline=deadline)
                                 for q in queries], return_exceptions=True),
                timeout=timeout)
        except Exception as exc:
            logger.warning("node=refund_policy 检索不可用: %s", type(exc).__name__)
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unavailable",
                    "expanded_queries": queries, "node_trace": trace}
        if isinstance(raw[0], BaseException):  # base_query 权威:它挂了就是不可用
            logger.warning("node=refund_policy base 检索异常: %s", type(raw[0]).__name__)
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unavailable",
                    "expanded_queries": queries, "node_trace": trace}
        results, dropped = [], []
        for q, r in zip(queries, raw):
            if isinstance(r, BaseException):  # 扩写支路临时异常:丢弃并记原因(spec §6.4)
                logger.warning("node=refund_policy 丢弃扩写支路: %s", type(r).__name__)
                dropped.append(f"expansion_branch_dropped:{type(r).__name__}")
            else:
                results.append(r)
        drop_note = ";".join(dropped) or None
        for r in results:  # 维护态:任一查询命中维护 note 即整轮不可用(spec §6.4)
            if _STATE_NOTE_CODES.get(r.note) is not None:
                out = state_fields_for_result(r)
                out.update({"expanded_queries": queries, "node_trace": trace})
                return out
        base = results[0]
        result = base
        if len(results) > 1 and settings.knowledge_strategy == "hybrid_rerank":
            candidates, seen = [], set()
            for r in results:  # 只按 chunk_id 合并候选,不比较/混合各查询分数(D10)
                for h in r.hits:
                    if h.chunk_id not in seen:
                        seen.add(h.chunk_id)
                        candidates.append(h)
            candidates = candidates[: 4 * settings.rerank_top_k]
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
            elif candidates:  # 重排真跑过且失败:回退 base 完整策略/分数/阈值+可观测 note
                note = ";".join(x for x in
                                (base.note, expand_note or "unified_rerank_failed") if x)
                result = replace(base, note=note or None)
            else:  # 无新增有效候选:rerank 未运行,直接使用该回退结果,不挂失败 note(spec §6.4)
                result = base
        elif expand_note:
            result = replace(base, note=";".join(x for x in (base.note, expand_note) if x) or None)
        if drop_note:  # 丢弃支路原因可观测:按既有 join 约定拼进最终结果 note
            result = replace(result, note=";".join(x for x in (result.note, drop_note) if x) or None)
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
        hits, max_items=settings.rerank_top_k,
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


def _to_stored(turn_messages) -> PersistedTurn:
    """本轮消息 → 落库行 + checkpoint 索引映射(spec §6);tool 结果不落库。"""
    stored: list[StoredMessage] = []
    indexes: list[int] = []
    for i, m in enumerate(turn_messages):
        if isinstance(m, HumanMessage):
            stored.append(StoredMessage("user", m.content))
            indexes.append(i)
        elif isinstance(m, AIMessage):
            stored.append(StoredMessage("assistant", m.content or None,
                                        tool_calls=m.tool_calls or None))
            indexes.append(i)
    return PersistedTurn(stored, indexes)


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
        stored_pack = _to_stored(state["turn_messages"])
        cid = int(sid) if sid.isdecimal() else None
        low_conf = _build_low_conf(state, cid)
        # ch07 Task 16:本轮用户行 prepare 已落库并盖章 → commit 幂等,只补 assistant 行
        # (resume 完成路径同样成立:盖章随图输入经 interrupt/checkpoint 存活)
        first = state["turn_messages"][0]
        prepared_user_id = None
        if isinstance(first, HumanMessage):
            v = (first.additional_kwargs or {}).get("db_id")
            prepared_user_id = int(v) if v is not None else None
        commit_task = asyncio.ensure_future(
            deps.store.commit_turn(sid, stored_pack.stored, low_confidence=low_conf,
                                   user_row_id=prepared_user_id))
        try:
            result = await asyncio.shield(commit_task)  # 取消时等事务落地
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await commit_task
            raise
        # 盖章:落库行 id 按索引映射写回 checkpoint 消息(additional_kwargs.db_id)
        stamped = list(state["turn_messages"])
        for row_id, msg_idx in zip(result.message_ids, stored_pack.checkpoint_indexes):
            m = stamped[msg_idx]
            stamped[msg_idx] = m.model_copy(update={
                "additional_kwargs": {**m.additional_kwargs, "db_id": int(row_id)}})
        user_id = (config.get("configurable") or {}).get("user_id", "")
        try:  # 上下文维护失败不阻塞本轮回复(锚点下轮再试)
            meta = await deps.store.get_context_meta(sid, user_id)
            budget = deps.context_budget or compute_budget(
                deps.settings, measure_sys_tokens(deps.system_prompt, []))
            merged = [*state["messages"], *stamped]
            new_l1 = evaluate_degrade(merged, meta.layer1_from, budget.layer1)
            if new_l1 is not None:
                moved = await deps.store.move_layer1_from(sid, user_id, new_l1)
                if moved:
                    logger.info("层1 降级 session=%s %s→%d", sid,
                                meta.layer1_from if meta.layer1_from is not None else "-",
                                new_l1)
                    meta = replace(meta, layer1_from=new_l1)
            view = build_layered_view(merged, meta, deps.settings)
            if view.l2_tokens > budget.layer2:
                logger.info("summary trigger session=%s 层2 约 %d token > 预算 %d",
                            sid, view.l2_tokens, budget.layer2)
                if deps.summary_runner is not None:
                    deps.summary_runner.maybe_trigger(sid, user_id)
        except Exception:
            logger.exception("context maintenance failed session=%s", sid)
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
        out = {"messages": stamped,
               "source_message_id": result.source_message_id,
               "node_trace": [*state["node_trace"], {"node": "log"}]}
        oc = state.get("order_context")
        if oc:  # 仅已完成轮建立/替换同 thread 订单焦点(spec §10)
            out["active_order"] = {"order_id": oc["order_id"],
                                   "source_message_id": result.source_message_id}
        return out

    return log_turn


# ── ch08 工单确认(spec §5):main_agent park 写调用 → 预览 interrupt → confirm/cancel ──

def route_after_agent(state) -> str:
    return "ticket_confirm" if state.get("pending_ticket") else "log"


def build_ticket_confirm_node(deps: GraphDeps):
    async def ticket_confirm(state, config):
        pending = state["pending_ticket"]
        trace = [*state["node_trace"], {"node": "ticket_confirm"}]
        # 挂起:帧由驱动层在图结束后按 pending interrupt 发射,节点不发 SSE
        selected = interrupt({"type": "ticket_preview",
                              "tool_call_id": pending["tool_call_id"],
                              "ticket_type": pending["args"].get("ticket_type"),
                              "description": pending["args"].get("description")})
        decision = (selected or {}).get("decision")
        sid = config["configurable"]["thread_id"]
        user_id = (config.get("configurable") or {}).get("user_id", "")
        ctx = TurnContext(user_id=user_id,
                          conversation_id=int(sid) if str(sid).isdecimal() else None,
                          resolved_query=state.get("resolved_query", ""),
                          retriever=None, settings=deps.settings,
                          session_factory=deps.session_factory)
        spec = deps.catalog.get(pending["name"]) if deps.catalog else None
        executor = ToolExecutor(ToolFace([spec] if spec else []), deps.settings,
                                session_factory=deps.session_factory,
                                mcp=deps.mcp_gateway)
        pending_write = PendingWrite(tool_call_id=pending["tool_call_id"],
                                     name=pending["name"], args=pending["args"],
                                     args_sha256=pending["args_sha256"])
        if decision == "confirm":
            # 只信 checkpoint 快照参数;客户端 resume 字段一律不采用(spec §5.3)
            outcome = await executor.execute_confirmed(pending_write, ctx)
        else:
            outcome = await executor.deny_write(pending_write, ctx, "用户取消")
        return {"turn_messages": [*state["turn_messages"], outcome.message],
                "pending_ticket": None, "node_trace": trace}

    return ticket_confirm
