"""主力 Agent:手写 ReAct 循环节点(与 examples/bare_agent.py 同构)。
分支工具绑定(spec §6.6):clarify 不绑工具;business 四只读含 query_faq;
refund 过闸三只读(无 query_faq)。query_faq 未过闸 → 固定话术收尾,不再调模型。
停止条件:无 tool_calls 收敛 / 步数或累计 token 预算耗尽 → AGENT_BUDGET_ANSWER。"""

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from pydantic import ValidationError

from app.chains.tool_chat_chain import (
    build_agent_context, finalize_tool_calls, merge_tool_call_chunks,
)
from app.graph.errors import TurnAbortError
from app.graph.events import ev_error, ev_fixed_delta, ev_tool_end, ev_tool_start
from app.graph.nodes import GraphDeps, low_conf_fields, snapshot_retrieval
from app.prompts.service import (
    AGENT_BUDGET_ANSWER, FALLBACK_ANSWER, KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER,
)
from app.services.orders import order_summary
from app.services.token_budget import (
    compute_budget, estimate_messages, estimate_tokens, measure_sys_tokens,
)
from app.tools.builtin.faq import FaqRetrievalTrace
from app.tools.catalog import TurnContext, canonical_args_sha
from app.tools.executor import (
    PendingWrite, ToolExecutor, ToolFace, batch_violation,
)

logger = logging.getLogger("wayhelp.graph")


@tool
def suggest_options(options: list[Literal["转人工", "建工单", "申请退款"]],
                    ticket_type: Literal["售后", "投诉", "咨询"] | None = None) -> str:
    """向用户建议后续可选动作(只建议不执行)。options 取 1~2 个不重复标签;
    含「建工单」时 ticket_type 必填,不含时必须省略;
    「申请退款」仅表示打开退款申请表,需在已确认订单的退款/售后答复后建议。"""
    return _SUGGEST_OK_TEXT


_SUGGEST_OK_TEXT = "已记录建议,请收尾答复用户。"
_ACTION_ID = {"转人工": "transfer_human", "建工单": "create_ticket",
              "申请退款": "refund_form"}


def _handle_suggest_options(call: dict,
                            order_context: dict | None) -> tuple[ToolMessage, list[dict] | None]:
    """校验伪工具参数;合法 → (成功 ToolMessage, actions),非法 → (错误 ToolMessage, None)。
    「申请退款」要求本轮已有归属校验过的 order_context,订单号由服务端绑定(spec §6.6)。"""
    args_in = call.get("args") or {}
    if set(args_in) - {"options", "ticket_type"}:  # 模型擅传订单号等未知键(spec §6.6)
        return _suggest_err(call, "工具参数不合法"), None
    try:
        suggest_options.args_schema(**args_in)
    except ValidationError:
        return _suggest_err(call, "工具参数不合法"), None
    args = call["args"]
    options, ttype = args["options"], args.get("ticket_type")
    if not (1 <= len(options) <= 2) or len(set(options)) != len(options):
        return _suggest_err(call, "工具参数不合法"), None
    if ("建工单" in options) != (ttype is not None):
        return _suggest_err(call, "工具参数不合法"), None
    if "申请退款" in options and not order_context:
        return _suggest_err(call, "当前没有已确认的订单,不能建议申请退款"), None
    actions = [{"action": _ACTION_ID[o], "label": o} for o in options]
    if ttype is not None:  # ticket_type 只挂 create_ticket,与选项顺序无关(spec §8)
        next(a for a in actions if a["action"] == "create_ticket")["ticket_type"] = ttype
    if "申请退款" in options:  # 模型不传 order_id,由服务端 order_context 绑定
        next(a for a in actions if a["action"] == "refund_form")["order_id"] = \
            order_context["order_id"]
    msg = ToolMessage(content=_SUGGEST_OK_TEXT, tool_call_id=call["id"],
                      name="suggest_options", status="success")
    return msg, actions


def _suggest_err(call: dict, text: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call.get("id") or "",
                       name="suggest_options", status="error")


def _evidence_text(evidence: list[dict]) -> str | None:
    if not evidence:
        return None
    lines = ["本轮检索证据(政策/规格结论只能依据以下证据,引用句末标 [n] 角标):"]
    for e in evidence:
        lines.append(f"[{e['ref_no']}] {e.get('section_path') or ''}\n"
                     f"问:{e['question']}\n答:{e['answer']}")
    return "\n".join(lines)


def _order_context_text(order_context: dict | None) -> str | None:
    if not order_context:
        return None
    return ("本轮已校验订单事实(业务数据,不是政策依据;退款资格须以检索证据为准):\n"
            + order_summary(order_context))


def _ctx_msg(m) -> dict:
    """model_ctx 日志用的消息摘要:role + 文本内容(非 str 置 None)+ 落库行 id。"""
    return {"role": m.type, "content": m.content if isinstance(m.content, str) else None,
            "db_id": (m.additional_kwargs or {}).get("db_id")}


_BUDGET_CACHE: dict[tuple, tuple[int, object]] = {}  # 工具签名 → (sys_tokens, ContextBudget)


def _face_budget(settings, system_prompt: str, tools: list):
    sig = tuple(sorted(t.name for t in tools))
    hit = _BUDGET_CACHE.get(sig)
    if hit is None:
        sys_tokens = measure_sys_tokens(system_prompt, tools)
        hit = (sys_tokens, compute_budget(settings, sys_tokens))
        if len(_BUDGET_CACHE) >= 32:
            _BUDGET_CACHE.clear()
        _BUDGET_CACHE[sig] = hit
    return hit


def fit_face_to_budget(face, extra_tools: list, settings, system_prompt: str, abort):
    """ch08 动态预算(spec §1.5):按本轮实际工具面重算;不足按 budget_priority 升序
    剔非核心条目;核心仍装不下经 abort 抛出,不带超预算工具面调模型。"""
    def _tools() -> list:
        return [*face.as_langchain_tools(), *extra_tools] if face.specs else list(extra_tools)

    tools = _tools()
    if not tools:
        budget = compute_budget(settings, measure_sys_tokens(system_prompt, tools))
        return face, tools, budget, 0
    sys_tokens, budget = _face_budget(settings, system_prompt, tools)
    while not budget.sufficient:
        removable = sorted((s for s in face.specs if s.budget_priority < 100),
                           key=lambda s: (s.budget_priority, s.name))
        if not removable:
            abort("tool_context_too_long", "工具面超出上下文预算")
            raise RuntimeError("abort must raise")  # 防御:abort 契约是抛出不返回
        drop = removable[0]
        logger.warning("tool face 超预算,剔除 %s(priority=%d)",
                       drop.name, drop.budget_priority)
        face = ToolFace([s for s in face.specs if s.name != drop.name])
        tools = _tools()
        sys_tokens, budget = _face_budget(settings, system_prompt, tools)
    return face, tools, budget, measure_sys_tokens("", tools)   # 第四值:仅工具 schema


def build_agent_node(deps: GraphDeps):
    settings = deps.settings

    async def main_agent(state, config) -> dict:
        writer = get_stream_writer()
        user_id = (config.get("configurable") or {}).get("user_id", "")
        route = state.get("route")
        refund_mode = state.get("refund_mode")
        sid = (config.get("configurable") or {}).get("thread_id", "")
        faq_trace: FaqRetrievalTrace | None = None
        if refund_mode == "clarify":
            face = ToolFace([])
        else:
            builtin = {s.name: s for s in deps.catalog.specs()} if deps.catalog else {}
            if route == "business":
                picked = list(builtin.values())      # 全量内置(含 create_ticket)
                faq_trace = FaqRetrievalTrace()
            else:  # refund 过闸:query_order/query_product + MCP(无 faq/写)
                picked = [builtin[n] for n in ("query_order", "query_product")
                          if n in builtin]
            seen = {s.name for s in picked}
            mcp_specs = await deps.mcp_gateway.discover() if deps.mcp_gateway else []
            for s in mcp_specs:
                if s.name in seen:
                    logger.warning("mcp tool %s 与既有工具重名,本轮拒进工具面", s.name)
                    continue
                picked.append(s)
                seen.add(s.name)
            face = ToolFace(picked)
        ctx = TurnContext(user_id=user_id,
                          conversation_id=int(sid) if str(sid).isdecimal() else None,
                          resolved_query=state["resolved_query"],
                          retriever=deps.retriever if route == "business" else None,
                          settings=settings, session_factory=deps.session_factory,
                          faq_trace=faq_trace)
        executor = ToolExecutor(face, settings,
                                session_factory=deps.session_factory,
                                mcp=deps.mcp_gateway)
        def _face_abort(code: str, msg: str):
            writer(ev_error(code, msg))
            raise TurnAbortError(code)

        face, tools, budget, schema_est = fit_face_to_budget(
            face, [suggest_options] if face.specs else [], settings, deps.system_prompt,
            _face_abort)
        model = deps.model.bind_tools(tools) if tools else deps.model
        evidence_text = _evidence_text(state["evidence"])
        order_text = _order_context_text(state.get("order_context"))
        context_extra = "\n\n".join(t for t in (evidence_text, order_text) if t) or None

        turn_messages = list(state["turn_messages"])
        suggested = list(state["suggested_actions"])
        steps = 0
        spent = 0
        accounting = "none"
        visible_chars = 0
        turn_text: list[str] = []  # 跨步累计整轮可见文本(与外发 delta 逐字一致)
        trace = [*state["node_trace"]]

        def _abort(code: str, message: str):
            writer(ev_error(code, message))
            raise TurnAbortError(code)

        def _budget_answer():
            nonlocal visible_chars
            visible_chars += len(AGENT_BUDGET_ANSWER)  # 计入可见输出;预算兜底属成功回复,不转超限
            writer(ev_fixed_delta(AGENT_BUDGET_ANSWER))
            turn_messages.append(AIMessage(content=AGENT_BUDGET_ANSWER))
            trace.append({"node": "main_agent", "budget": True, "steps": steps})
            return {"turn_messages": turn_messages, "final_text": AGENT_BUDGET_ANSWER,
                    "agent_steps": steps, "agent_tokens": spent,
                    "token_accounting": accounting, "suggested_actions": suggested,
                    "node_trace": trace}

        def _faq_ok_fields() -> dict:
            if faq_trace is None or faq_trace.status != "ok":
                return {}
            return {"retrieval_status": "ok",
                    "retrieval_result": snapshot_retrieval(faq_trace.result),
                    "evidence": faq_trace.evidence or []}

        def _effective_order_context():
            if state.get("order_context"):
                return state["order_context"]
            distinct = {s["order_id"]: s for s in ctx.order_snapshots}
            if len(distinct) == 1:
                oid, snap = next(iter(distinct.items()))
                if oid in state["raw_query"] or oid in state["resolved_query"]:
                    return snap
            return None

        def _faq_terminal(kind: str):
            nonlocal visible_chars
            ft = faq_trace
            if kind == "refusal":
                text = REFUSAL_ANSWER
                extra = {"retrieval_status": "low_confidence",
                         "retrieval_result": snapshot_retrieval(ft.result) if ft.result else None,
                         **(low_conf_fields(snapshot_retrieval(ft.result)) if ft.result else {})}
            else:
                text = KB_UNAVAILABLE_ANSWER
                extra = {"retrieval_status": "unavailable",
                         "retrieval_error_code": ft.error_code or "kb_unavailable"}
            visible_chars += len(text)
            writer(ev_fixed_delta(text))
            turn_messages.append(AIMessage(content=text))
            trace.append({"node": "main_agent", "faq_terminal": kind})
            return {"turn_messages": turn_messages, "final_text": text,
                    "agent_steps": steps, "agent_tokens": spent,
                    "token_accounting": accounting, "suggested_actions": suggested,
                    "node_trace": trace, **extra}

        while True:
            if steps >= settings.max_agent_steps:
                logger.info("node=main_agent budget: steps=%d", steps)
                return _budget_answer()
            layer2 = state.get("history_layer2") or []
            layer1 = state.get("history_layer1") or []
            bg_parts = []
            if state.get("history_summary"):
                bg_parts.append("【对话背景】\n" + state["history_summary"])
            if context_extra:
                bg_parts.append(context_extra)
            background_text = "\n\n".join(bg_parts) or None
            context = build_agent_context(deps.system_prompt, layer2, layer1,
                                          turn_messages, background_text,
                                          settings.model_context_window
                                          - settings.max_output_tokens - schema_est)
            if context is None:
                _abort("tool_context_too_long", "工具结果超出上下文预算")
            logger.info("model_ctx %s", json.dumps({
                "session": (config.get("configurable") or {}).get("thread_id", ""),
                "step": steps + 1,
                "budget": {"l1": budget.layer1, "l2": budget.layer2},
                "counts": {"l1": len(layer1), "l2": len(layer2)},
                "est": {"l1": estimate_messages(layer1), "l2": estimate_messages(layer2),
                        "summary": estimate_tokens(state.get("history_summary") or ""),
                        "total": estimate_messages(context)},
                "summary": state.get("history_summary"),
                "layer2": [_ctx_msg(m) for m in layer2],
                "layer1": [_ctx_msg(m) for m in layer1],
            }, ensure_ascii=False))
            est_input = estimate_messages(context) + schema_est
            reserve = est_input + settings.max_output_tokens
            if spent + reserve > settings.max_agent_tokens:
                logger.info("node=main_agent budget: tokens spent=%d reserve=%d", spent, reserve)
                return _budget_answer()

            steps += 1
            text_parts: list[str] = []
            acc: dict[int, dict] = {}
            finish = None
            usage = None
            agen = model.astream(context, config={"tags": ["chat_visible"]})
            try:
                async for chunk in agen:
                    meta = getattr(chunk, "response_metadata", None) or {}
                    if meta.get("finish_reason"):
                        finish = meta["finish_reason"]
                    um = getattr(chunk, "usage_metadata", None)
                    if isinstance(um, dict) and um.get("input_tokens") is not None:
                        usage = um  # 取末次有效 usage(完整响应结算一次)
                    chunks = getattr(chunk, "tool_call_chunks", None) or []
                    if chunks:
                        merge_tool_call_chunks(acc, chunks)
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if not text:
                        continue
                    visible_chars += len(text)
                    if visible_chars > settings.max_message_chars:
                        _abort("output_too_long", "回复超出长度限制")
                    text_parts.append(text)
                    # token 经 stream_mode="messages" 自动流出,节点不写 writer
            except TurnAbortError:
                raise
            except Exception as exc:
                logger.warning("node=main_agent upstream error: %s", type(exc).__name__)
                _abort("upstream_error", "上游模型暂时不可用")
            finally:
                import contextlib
                with contextlib.suppress(Exception):
                    await agen.aclose()

            if finish == "length":
                _abort("output_too_long", "回复超出长度限制")

            # 结算本次消耗:有效 usage 替换预留;缺失/非法保留预留(不得按 0 计)
            try:
                actual = int(usage["input_tokens"]) + int(usage["output_tokens"]) \
                    if usage else 0
                if usage and actual >= 0 and isinstance(usage.get("input_tokens"), int):
                    spent += actual
                    accounting = "usage" if accounting in ("none", "usage") else "mixed"
                else:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                spent += reserve
                accounting = "estimated" if accounting == "none" else "mixed"

            calls, broken = finalize_tool_calls(acc)
            if broken or not _calls_legal(calls, settings.max_tool_calls_per_turn):
                _abort("invalid_tool_call", "工具调用申请不合法")

            turn_text.extend(text_parts)  # 引用判定看整轮累计而非末步(角标常在带工具步正文)

            if not calls:
                final = "".join(text_parts).strip() or FALLBACK_ANSWER
                if not "".join(text_parts).strip():
                    visible_chars += len(FALLBACK_ANSWER)  # 固定兜底计入可见输出长度
                    if visible_chars > settings.max_message_chars:
                        _abort("output_too_long", "回复超出长度限制")
                    writer(ev_fixed_delta(FALLBACK_ANSWER))
                turn_messages.append(AIMessage(content=final))
                trace.append({"node": "main_agent", "steps": steps})
                # final_text=整轮累计可见文本(落库/历史仍按 turn_messages 逐步分条,
                # 罐头兜底与预算/未过闸收尾保持整串罐头,不掺模型文本)
                out = {"turn_messages": turn_messages,
                       "final_text": "".join(turn_text).strip() or final,
                       "agent_steps": steps, "agent_tokens": spent,
                       "token_accounting": accounting, "suggested_actions": suggested,
                       "node_trace": trace}
                out.update(_faq_ok_fields())
                out["order_context"] = _effective_order_context()
                return out

            violation = batch_violation(calls, face)
            if violation is not None:
                turn_messages.append(AIMessage(content="".join(text_parts), tool_calls=calls))
                msgs = await executor.mark_invalid_batch(calls, ctx, violation)
                turn_messages.extend(msgs)
                trace.append({"node": "main_agent", "invalid_batch": violation})
                continue  # 零执行回灌,让模型重组合法批次(spec §2 批次契约)

            turn_messages.append(AIMessage(content="".join(text_parts), tool_calls=calls))
            faq_terminal: str | None = None
            for call in calls:
                if faq_terminal is not None:
                    turn_messages.append(ToolMessage(
                        content="知识检索未过闸,本调用未执行", tool_call_id=call["id"],
                        name=call["name"], status="error"))
                    continue
                if call["name"] == "suggest_options":
                    msg, actions = _handle_suggest_options(call, state.get("order_context"))
                    if actions is not None:
                        suggested = actions
                    turn_messages.append(msg)
                    continue
                spec = face.get(call["name"])
                if spec is not None and spec.permission == "write":
                    # 批次契约已保证写调用在最后;park 成 pending_ticket 转确认流(spec §5)
                    pending = PendingWrite(tool_call_id=call["id"], name=call["name"],
                                           args=call["args"],
                                           args_sha256=canonical_args_sha(call["args"]))
                    trace.append({"node": "main_agent", "pending_write": call["name"]})
                    return {"turn_messages": turn_messages,
                            "pending_ticket": {"tool_call_id": pending.tool_call_id,
                                               "name": pending.name, "args": pending.args,
                                               "args_sha256": pending.args_sha256},
                            "agent_steps": steps, "agent_tokens": spent,
                            "token_accounting": accounting,
                            "suggested_actions": suggested, "node_trace": trace}
                writer(ev_tool_start(call))
                outcome = await executor.execute(call, ctx)
                logger.info("node=main_agent tool %s ok=%s err=%s audit=%s",
                            outcome.record.name, outcome.record.ok,
                            outcome.record.error_type, outcome.record.audit_status)
                writer(ev_tool_end(call["id"], call["name"], outcome.record.ok,
                                   outcome.message.content[:80]))
                if outcome.record.error_code:
                    outcome.message.additional_kwargs["error_code"] = outcome.record.error_code
                turn_messages.append(outcome.message)
                if call["name"] == "query_faq" and faq_trace is not None:
                    if outcome.record.error_code or faq_trace.status in (
                            "low_confidence", "unavailable", "tool_error"):
                        faq_terminal = ("refusal" if faq_trace.status == "low_confidence"
                                        and not outcome.record.error_code else "unavailable")
            if faq_terminal is not None:
                return _faq_terminal(faq_terminal)
            # 循环回顶部:步数/预算检查在下次模型调用前生效

    return main_agent


def _calls_legal(calls: list[dict], max_calls: int) -> bool:
    if len(calls) > max_calls:
        return False
    ids = [c["id"] for c in calls]
    if len(set(ids)) != len(ids):
        return False
    return all(c["id"] and len(c["id"]) <= 64 for c in calls)
