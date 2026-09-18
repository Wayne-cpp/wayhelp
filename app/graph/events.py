"""节点经 get_stream_writer 发出的自定义事件 payload;驱动层按 kind 翻译成 SSE 帧。"""


def ev_tool_start(call: dict) -> dict:
    return {"kind": "tool_start", "tool_call_id": call["id"],
            "name": call["name"], "args": call["args"]}


def ev_tool_end(tool_call_id: str, name: str, ok: bool, summary: str) -> dict:
    return {"kind": "tool_end", "tool_call_id": tool_call_id,
            "name": name, "ok": ok, "summary": summary}


def ev_fixed_delta(content: str) -> dict:
    """固定话术/兜底文本的显式 delta(非模型 token,不经 messages 流)。"""
    return {"kind": "fixed_delta", "content": content}


def ev_citations(citations: list[dict]) -> dict:
    return {"kind": "citations", "citations": citations}


def ev_suggest_actions(source_message_id: str, options: list[dict]) -> dict:
    return {"kind": "suggest_actions", "source_message_id": source_message_id,
            "options": options}


def ev_error(code: str, message: str) -> dict:
    return {"kind": "error", "code": code, "message": message}
