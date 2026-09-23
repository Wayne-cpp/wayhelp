"""ch07:聊天主链 ReAct 上下文拼装与工具调用流式组装(tool_chat_chain 退役后仅存)。"""
import json

from langchain_core.messages import HumanMessage, SystemMessage


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


def build_agent_context(system_prompt: str, layer2: list, layer1: list,
                        turn_messages: list, background_text: str | None,
                        guard_tokens: int) -> list | None:
    """ch07 拼装(spec §8):[system, *层2, *层1, 当前用户句, 背景?, *ReAct步];
    总量超 guard_tokens → None(调用方发 tool_context_too_long)。"""
    from app.services.token_budget import estimate_messages
    if not turn_messages or not isinstance(turn_messages[0], HumanMessage):
        raise RuntimeError("turn_messages must start with the current human message")
    background = [HumanMessage(content=background_text)] if background_text else []
    base = [SystemMessage(content=system_prompt), *layer2, *layer1,
            turn_messages[0], *background, *turn_messages[1:]]
    return base if estimate_messages(base) <= guard_tokens else None
