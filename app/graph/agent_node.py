"""主力 Agent:手写 ReAct 循环节点(与 examples/bare_agent.py 同构)。
停止条件:无 tool_calls 收敛 / 步数或累计 token 预算耗尽 → AGENT_BUDGET_ANSWER。"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from pydantic import ValidationError

from app.chains.tool_chat_chain import (
    build_agent_context, finalize_tool_calls, merge_tool_call_chunks,
)
from app.graph.errors import TurnAbortError
from app.graph.events import ev_error, ev_fixed_delta, ev_tool_end, ev_tool_start
from app.graph.nodes import GraphDeps
from app.prompts.service import AGENT_BUDGET_ANSWER, FALLBACK_ANSWER
from app.tools.business import MOCK_TOOLS
from app.tools.executor import ToolExecutor, ToolRegistry

logger = logging.getLogger("wayhelp.graph")


@tool
def suggest_options(options: list[Literal["转人工", "建工单"]],
                    ticket_type: Literal["售后", "投诉", "咨询"] | None = None) -> str:
    """向用户建议后续可选动作(只建议不执行)。options 取 1~2 个不重复标签;
    含「建工单」时 ticket_type 必填,不含时必须省略。"""
    return _SUGGEST_OK_TEXT


_SUGGEST_OK_TEXT = "已记录建议,请收尾答复用户。"
_ACTION_ID = {"转人工": "transfer_human", "建工单": "create_ticket"}


def _handle_suggest_options(call: dict) -> tuple[ToolMessage, list[dict] | None]:
    """校验伪工具参数;合法 → (成功 ToolMessage, actions),非法 → (错误 ToolMessage, None)。"""
    try:
        suggest_options.args_schema(**(call.get("args") or {}))
    except ValidationError:
        return _suggest_err(call, "工具参数不合法"), None
    args = call["args"]
    options, ttype = args["options"], args.get("ticket_type")
    if not (1 <= len(options) <= 2) or len(set(options)) != len(options):
        return _suggest_err(call, "工具参数不合法"), None
    if ("建工单" in options) != (ttype is not None):
        return _suggest_err(call, "工具参数不合法"), None
    actions = [{"action": _ACTION_ID[o], "label": o} for o in options]
    if ttype is not None:  # ticket_type 只挂 create_ticket,与选项顺序无关(spec §8)
        next(a for a in actions if a["action"] == "create_ticket")["ticket_type"] = ttype
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


def _tool_schema_tokens(tools) -> int:
    """绑定工具 schema 的估算开销(name+description+args schema)。"""
    import json as _json
    parts = []
    for t in tools:
        parts.append(t.name)
        parts.append(t.description or "")
        try:
            parts.append(_json.dumps(t.args_schema.schema(), ensure_ascii=False))
        except Exception:
            pass
    return count_tokens_approximately([AIMessage(content="\n".join(parts))])


def build_agent_node(deps: GraphDeps):
    settings = deps.settings

    async def main_agent(state) -> dict:
        writer = get_stream_writer()
        tools = [*MOCK_TOOLS, suggest_options]
        registry = ToolRegistry(tools)  # 注册表无 create_ticket/query_faq:伪造调用 → unknown_tool
        executor = ToolExecutor(registry, settings.tool_timeout_seconds,
                                settings.tool_max_retries, settings.max_tool_result_chars,
                                write_tools=set())  # 聊天图内无写工具
        model = deps.model.bind_tools(registry.tools)
        schema_tokens = _tool_schema_tokens(registry.tools)
        evidence_text = _evidence_text(state["evidence"])

        turn_messages = list(state["turn_messages"])
        suggested = list(state["suggested_actions"])
        steps = 0
        spent = 0
        accounting = "none"
        visible_chars = 0
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

        while True:
            if steps >= settings.max_agent_steps:
                logger.info("node=main_agent budget: steps=%d", steps)
                return _budget_answer()
            context = build_agent_context(deps.system_prompt, state["messages"],
                                          turn_messages, evidence_text,
                                          settings.max_input_tokens)
            if context is None:
                _abort("tool_context_too_long", "工具结果超出上下文预算")
            est_input = count_tokens_approximately(context) + schema_tokens
            reserve = est_input + settings.max_output_tokens
            if spent + reserve > settings.max_agent_tokens:
                logger.info("node=main_agent budget: tokens spent=%d reserve=%d", spent, reserve)
                return _budget_answer()

            steps += 1
            text_parts: list[str] = []
            acc: dict[int, dict] = {}
            finish = None
            usage = None
            agen = model.astream(context)
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

            if not calls:
                final = "".join(text_parts).strip() or FALLBACK_ANSWER
                if not "".join(text_parts).strip():
                    visible_chars += len(FALLBACK_ANSWER)  # 固定兜底计入可见输出长度
                    if visible_chars > settings.max_message_chars:
                        _abort("output_too_long", "回复超出长度限制")
                    writer(ev_fixed_delta(FALLBACK_ANSWER))
                turn_messages.append(AIMessage(content=final))
                trace.append({"node": "main_agent", "steps": steps})
                return {"turn_messages": turn_messages, "final_text": final,
                        "agent_steps": steps, "agent_tokens": spent,
                        "token_accounting": accounting, "suggested_actions": suggested,
                        "node_trace": trace}

            turn_messages.append(AIMessage(content="".join(text_parts), tool_calls=calls))
            for call in calls:
                if call["name"] == "suggest_options":
                    msg, actions = _handle_suggest_options(call)
                    if actions is not None:
                        suggested = actions  # 多次合法建议以最后一组替换
                    turn_messages.append(msg)
                    continue
                writer(ev_tool_start(call))
                outcome = await executor.execute(call)
                logger.info("node=main_agent tool %s ok=%s err=%s",
                            outcome.record.name, outcome.record.ok, outcome.record.error_type)
                writer(ev_tool_end(call["id"], call["name"], outcome.record.ok,
                                   outcome.message.content[:80]))
                if outcome.record.error_code:
                    outcome.message.additional_kwargs["error_code"] = outcome.record.error_code
                turn_messages.append(outcome.message)
            # 循环回顶部:步数/预算检查在下次模型调用前生效

    return main_agent


def _calls_legal(calls: list[dict], max_calls: int) -> bool:
    if len(calls) > max_calls:
        return False
    ids = [c["id"] for c in calls]
    if len(set(ids)) != len(ids):
        return False
    return all(c["id"] and len(c["id"]) <= 64 for c in calls)
