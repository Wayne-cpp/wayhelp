import uuid
from dataclasses import dataclass
from typing import Protocol

from app.errors import SessionCapacityReachedError


@dataclass(frozen=True)
class StoredMessage:
    role: str  # "user" | "assistant" | "tool"
    content: str | None  # tool 行为 envelope JSON 文本;assistant 纯工具调用时可为 None
    tool_calls: list[dict] | None = None  # 仅 assistant 行: [{"name","args","id","type"}]
    tool_call_id: str | None = None       # 仅 tool 行


@dataclass(frozen=True)
class LowConfidenceRecord:
    raw_question: str
    source: str            # "retrieval_low_conf" | "self_check"(user_feedback 本章不写)
    reason: str | None
    conversation_id: int | None


@dataclass(frozen=True)
class CommitTurnResult:
    source_message_id: str  # 本轮用户消息 ID 的十进制字符串(内存实现为分配序号)


class SessionStore(Protocol):
    async def create(self, user_id: str) -> str: ...
    async def exists(self, session_id: str, user_id: str) -> bool: ...
    async def snapshot(self, session_id: str) -> list[StoredMessage]: ...
    async def commit_turn(self, session_id: str, messages: list[StoredMessage],
                          low_confidence: LowConfidenceRecord | None = None) -> CommitTurnResult: ...


def validate_turn(messages: list[StoredMessage], max_tool_calls: int) -> None:
    """spec §5.1 四条;非法抛 ValueError。"""
    if len(messages) < 2:
        raise ValueError("turn must contain at least user + assistant")
    if messages[0].role != "user" or not (messages[0].content or "").strip():
        raise ValueError("turn must start with a non-empty user message")
    last = messages[-1]
    if last.role != "assistant" or not (last.content or "").strip():
        raise ValueError("turn must end with a non-empty assistant message")
    middle = messages[1:-1]
    group_calls: list[dict] = []
    group_ids: list[str] = []

    def close_group() -> None:
        if group_calls:
            expected = [c["id"] for c in group_calls]
            if sorted(expected) != sorted(group_ids) or len(set(group_ids)) != len(group_ids):
                raise ValueError("tool call ids and tool messages must match one-to-one")
            group_calls.clear()
            group_ids.clear()

    for m in middle:
        if m.role == "assistant":
            close_group()  # 新 assistant 出现前先结清上一组
            if not m.tool_calls:
                raise ValueError("middle assistant message without tool calls")
            if len(m.tool_calls) > max_tool_calls:
                raise ValueError("too many tool calls in turn")
            for call in m.tool_calls:
                cid = call.get("id") if isinstance(call, dict) else None
                if not isinstance(cid, str) or not cid or len(cid) > 64:
                    raise ValueError("invalid tool call id")
            group_calls.extend(m.tool_calls)
        elif m.role == "tool":
            if not group_calls:
                raise ValueError("orphan tool message")
            if not m.tool_call_id or len(m.tool_call_id) > 64:
                raise ValueError("tool message missing tool_call_id")
            group_ids.append(m.tool_call_id)
        else:
            raise ValueError(f"unexpected role in turn middle: {m.role}")
    close_group()


def _turns(messages: list[StoredMessage]) -> list[list[StoredMessage]]:
    """按 user 起始切分完整 turn。"""
    turns: list[list[StoredMessage]] = []
    for m in messages:
        if m.role == "user":
            turns.append([m])
        elif turns:
            turns[-1].append(m)
    return [t for t in turns if len(t) >= 2]


class InMemorySessionStore:
    def __init__(self, max_sessions: int, max_messages_per_session: int,
                 max_message_chars: int, max_tool_calls_per_turn: int = 5):
        self._max_sessions = max_sessions
        self._max_messages = max_messages_per_session
        self._max_chars = max_message_chars
        self._max_tool_calls = max_tool_calls_per_turn
        self._sessions: dict[str, list[StoredMessage]] = {}
        self._msg_seq = 0
        self.low_confidence: list[LowConfidenceRecord] = []

    async def create(self, user_id: str) -> str:
        if len(self._sessions) >= self._max_sessions:
            raise SessionCapacityReachedError("session capacity reached")
        sid = str(uuid.uuid4())
        self._sessions[sid] = []
        return sid

    async def exists(self, session_id: str, user_id: str) -> bool:
        return session_id in self._sessions  # 内存实现不绑定 user_id(测试用)

    async def snapshot(self, session_id: str) -> list[StoredMessage]:
        return list(self._sessions[session_id])

    async def commit_turn(self, session_id: str, messages: list[StoredMessage],
                          low_confidence: LowConfidenceRecord | None = None) -> CommitTurnResult:
        validate_turn(messages, self._max_tool_calls)
        for m in messages:
            if m.content is not None and len(m.content) > self._max_chars and m.role != "tool":
                raise ValueError("message text exceeds MAX_MESSAGE_CHARS")
        if low_confidence is not None:  # 校验全过后才入池,与 DB 侧同成同败语义一致
            self.low_confidence.append(low_confidence)
        msgs = self._sessions[session_id]
        msgs.extend(messages)
        limit = max(self._max_messages, self._max_tool_calls + 3)
        while len(msgs) > limit:
            turns = _turns(msgs)
            if len(turns) <= 1:
                break  # 始终保留最新完整 turn
            del msgs[: len(turns[0])]
        self._msg_seq += 1  # 校验全过后才分配,与入池同点
        source_id = str(self._msg_seq)
        return CommitTurnResult(source_id)
