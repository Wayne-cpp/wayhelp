import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

from app.sessions import StoredMessage
from app.tool_envelope import unwrap


def rebuild_messages(stored: list[StoredMessage]) -> list[BaseMessage]:
    """StoredMessage -> LangChain 消息;tool 行解 envelope 恢复 status;损坏抛 ValueError。"""
    out: list[BaseMessage] = []
    pending_calls: dict[str, dict] = {}
    for m in stored:
        if m.role == "user":
            out.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            calls = m.tool_calls or []
            pending_calls = {c["id"]: c for c in calls}
            out.append(AIMessage(content=m.content or "", tool_calls=calls))
        elif m.role == "tool":
            if m.tool_call_id not in pending_calls:
                raise ValueError("tool message without matching tool call")
            call = pending_calls.pop(m.tool_call_id)
            content, ok = unwrap(m.content or "")
            out.append(ToolMessage(content=content, tool_call_id=m.tool_call_id,
                                   name=call["name"],
                                   status="success" if ok else "error"))
        else:
            raise ValueError(f"unknown stored role: {m.role}")
        if not pending_calls and out and isinstance(out[-1], AIMessage) and m.role == "assistant" and not m.tool_calls:
            pending_calls = {}
    if pending_calls:
        raise ValueError("tool calls without tool results")
    return out


def merge_tool_call_chunks(acc: dict[int, dict], chunks: list[dict]) -> None:
    for ch in chunks:
        idx = ch.get("index", 0)
        slot = acc.setdefault(idx, {"name": None, "args": "", "id": None})
        if ch.get("name"):
            slot["name"] = ch["name"]
        if ch.get("id"):
            slot["id"] = ch["id"]
        if ch.get("args"):
            slot["args"] += ch["args"]


def finalize_tool_calls(acc: dict[int, dict]) -> tuple[list[dict], bool]:
    calls: list[dict] = []
    for idx in sorted(acc):
        slot = acc[idx]
        try:
            args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            return [], True
        if not isinstance(args, dict) or not slot["name"] or not slot["id"]:
            return [], True
        calls.append({"name": slot["name"], "args": args, "id": slot["id"],
                      "type": "tool_call"})
    return calls, False


def _drop_broken_head(messages: list[BaseMessage]) -> list[BaseMessage]:
    """裁剪后修复:首条(非 system)必须是 human;tool 组不完整则从头部丢整个 turn。"""
    rest = list(messages)
    while rest:
        head = rest[0]
        if isinstance(head, HumanMessage) and _tool_groups_complete(rest):
            break
        # 丢弃一个完整 turn:从头部到下一条 HumanMessage 之前
        nxt = next((i for i in range(1, len(rest)) if isinstance(rest[i], HumanMessage)),
                   len(rest))
        rest = rest[nxt:]
    return rest


def _tool_groups_complete(messages: list[BaseMessage]) -> bool:
    i = 0
    while i < len(messages):
        m = messages[i]
        if isinstance(m, AIMessage) and m.tool_calls:
            ids = [c["id"] for c in m.tool_calls]
            j = i + 1
            seen: list[str] = []
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                seen.append(messages[j].tool_call_id)
                j += 1
            if sorted(seen) != sorted(ids):
                return False
            i = j
        else:
            i += 1
    return True


def build_messages_with_tools(system_prompt: str, history: list[StoredMessage],
                              current_input: str, max_input_tokens: int) -> list[BaseMessage]:
    from app.chains.chat_chain import check_input_budget

    check_input_budget(system_prompt, current_input, max_input_tokens)
    system = SystemMessage(content=system_prompt)
    history_msgs = rebuild_messages(history)
    current = HumanMessage(content=current_input)
    trimmed = trim_messages(
        [system, *history_msgs, current],
        max_tokens=max_input_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False,
    )
    fixed = [trimmed[0], *_drop_broken_head(trimmed[1:])]
    _validate(fixed, current_input, max_input_tokens)
    return fixed


def _validate(messages: list[BaseMessage], current_input: str, max_input_tokens: int) -> None:
    if not messages or not isinstance(messages[0], SystemMessage):
        raise RuntimeError("trimmed messages lost the system prompt")
    if not isinstance(messages[-1], HumanMessage) or messages[-1].content != current_input:
        raise RuntimeError("trimmed messages lost the current human message")
    if len(messages) > 1 and not isinstance(messages[1], HumanMessage):
        raise RuntimeError("trimmed messages must start with a human message after system")
    if not _tool_groups_complete(messages[1:]):
        raise RuntimeError("tool group incomplete after repair")
    if count_tokens_approximately(messages) > max_input_tokens:
        raise RuntimeError("trimmed messages still exceed input token budget")


def fit_tool_context(messages: list[BaseMessage], max_input_tokens: int,
                     protected_from: int) -> list[BaseMessage] | None:
    """第二次调用预算。messages = [system, ...历史..., 当前human, AIMessage(tool_calls), ToolMessage...];
    protected_from = 当前 human 的下标,该下标起(含)的尾部不可拆分/删除。
    先丢最旧完整历史 turn;仍超限返回 None(调用方发 tool_context_too_long)。"""
    result = list(messages)
    while count_tokens_approximately(result) > max_input_tokens:
        # 只允许在 [1, protected_from) 区间丢完整历史 turn(system 在 0,不动)
        drop_at = next((i for i in range(1, protected_from)
                        if isinstance(result[i], HumanMessage)), None)
        if drop_at is None:
            return None
        nxt = next((i for i in range(drop_at + 1, protected_from)
                    if isinstance(result[i], HumanMessage)), protected_from)
        del result[drop_at:nxt]
        protected_from -= nxt - drop_at
    return result


def build_agent_context(system_prompt: str, history: list[BaseMessage],
                        turn_messages: list[BaseMessage], evidence_text: str | None,
                        max_input_tokens: int) -> list[BaseMessage] | None:
    """组装并裁剪一次 ReAct 模型调用的上下文。
    布局:[system, ...历史完整 turn..., evidence?, ...本轮 turn_messages...];
    裁剪只允许丢最旧的完整历史 turn(fit_tool_context 的区间语义),
    system/evidence/当前 turn 受保护。返回 None = 历史丢光仍超限(调用方发 tool_context_too_long)。"""
    evidence = [SystemMessage(content=evidence_text)] if evidence_text else []
    base = [SystemMessage(content=system_prompt), *history, *evidence, *turn_messages]
    protected_from = len(base) - len(turn_messages) - len(evidence)
    if not history:
        # 无历史可丢:直接判定
        from langchain_core.messages.utils import count_tokens_approximately
        return base if count_tokens_approximately(base) <= max_input_tokens else None
    return fit_tool_context(base, max_input_tokens, protected_from=protected_from)
