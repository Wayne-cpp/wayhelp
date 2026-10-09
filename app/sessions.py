import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
    source: str            # "retrieval_low_conf" | "self_check" | "user_feedback"
    reason: str | None
    conversation_id: int | None
    retrieved_chunks: list[dict] | None = None   # ch09:召回快照
    resolved_question: str | None = None         # ch09:指代消解后问题
    turn_message_id: int | None = None           # ch09:轮次锚点


@dataclass(frozen=True)
class CommitTurnResult:
    source_message_id: str  # 本轮用户消息 ID 的十进制字符串(内存实现为分配序号)
    message_ids: list[str] = field(default_factory=list)  # ch07:本轮逐行落库 ID;DB 实现在 Task 5 补全


@dataclass(frozen=True)
class ContextMeta:
    summary: str | None
    summary_upto: int | None
    layer1_from: int | None


@dataclass(frozen=True)
class ConversationMessage:
    id: str
    role: str
    content: str
    tool_calls: list[dict] | None
    created_at: datetime | None = None


@dataclass(frozen=True)
class PersistedMessageRecord:
    id: str
    role: str
    content: str | None
    tool_calls: list[dict] | None
    tool_call_id: str | None
    created_at: datetime | None = None


@dataclass(frozen=True)
class ConversationItem:
    id: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    preview: str | None
    summarized: bool


@dataclass(frozen=True)
class SummaryAppendResult:
    seq: int | None
    applied: bool
    reason: str | None


@dataclass(frozen=True)
class PersistedTurn:
    stored: list[StoredMessage]
    checkpoint_indexes: list[int]   # stored 行 ↔ turn_messages 原始索引


class SessionStore(Protocol):
    async def create(self, user_id: str) -> str: ...
    async def exists(self, session_id: str, user_id: str) -> bool: ...
    async def snapshot(self, session_id: str) -> list[StoredMessage]: ...
    async def append_user_message(self, session_id: str, content: str) -> str: ...
    async def commit_turn(self, session_id: str, messages: list[StoredMessage],
                          low_confidence: LowConfidenceRecord | None = None,
                          user_row_id: int | None = None) -> CommitTurnResult: ...
    async def get_context_meta(self, session_id: str, user_id: str) -> ContextMeta: ...
    async def list_conversations(self, user_id: str) -> list[ConversationItem]: ...
    async def list_messages(self, session_id: str, user_id: str) -> list[ConversationMessage] | None: ...
    async def list_checkpoint_records(self, session_id: str,
                                      user_id: str) -> list[PersistedMessageRecord] | None: ...
    async def move_layer1_from(self, session_id: str, user_id: str, new_id: int) -> bool: ...
    async def fetch_span_texts(self, session_id: str,
                               from_id: int, upto_id: int) -> list[tuple[int, str, str]]: ...
    async def append_summary(self, session_id: str, from_id: int, upto_id: int, content: str,
                             projection_tokens: int) -> SummaryAppendResult: ...


def validate_turn(messages: list[StoredMessage], max_tool_calls: int) -> None:
    """turn 结构校验;非法抛 ValueError(ch07:tool 行不落库,中间段只收带 tool_calls 的 assistant)。"""
    if len(messages) < 2:
        raise ValueError("turn must contain at least user + assistant")
    if messages[0].role != "user" or not (messages[0].content or "").strip():
        raise ValueError("turn must start with a non-empty user message")
    last = messages[-1]
    if last.role != "assistant" or not (last.content or "").strip():
        raise ValueError("turn must end with a non-empty assistant message")
    for m in messages[1:-1]:
        if m.role == "tool":
            raise ValueError("tool rows are not persisted (ch07)")
        if m.role != "assistant":
            raise ValueError(f"unexpected role in turn middle: {m.role}")
        if not m.tool_calls:
            raise ValueError("middle assistant message without tool calls")
        if len(m.tool_calls) > max_tool_calls:
            raise ValueError("too many tool calls in turn")
        for call in m.tool_calls:
            cid = call.get("id") if isinstance(call, dict) else None
            if not isinstance(cid, str) or not cid or len(cid) > 64:
                raise ValueError("invalid tool call id")


def _turns(messages: list[StoredMessage]) -> list[list[StoredMessage]]:
    """按 user 起始切分完整 turn。"""
    turns: list[list[StoredMessage]] = []
    for m in messages:
        if m.role == "user":
            turns.append([m])
        elif turns:
            turns[-1].append(m)
    return [t for t in turns if len(t) >= 2]


def _trim_cut(messages: list[StoredMessage]) -> int | None:
    """超限裁剪的前缀长度:第二个完整轮起始 user 的下标(含其前无回复的悬垂用户行,
    ch07 Task 16 prepare 先落库后悬垂行合法存在);不足两个完整轮 → None
    (始终保留最新完整轮)。"""
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    complete = [s for s, e in zip(starts, [*starts[1:], len(messages)]) if e - s >= 2]
    return complete[1] if len(complete) >= 2 else None


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
        self._owners: dict[str, str] = {}
        self._ids: dict[str, list[int]] = {}
        self._meta: dict[str, ContextMeta] = {}
        self._segments: dict[str, list[str]] = {}

    async def create(self, user_id: str) -> str:
        if len(self._sessions) >= self._max_sessions:
            raise SessionCapacityReachedError("session capacity reached")
        sid = str(uuid.uuid4())
        self._sessions[sid] = []
        self._owners[sid] = user_id
        self._ids[sid] = []
        return sid

    async def exists(self, session_id: str, user_id: str) -> bool:
        return self._owners.get(session_id) == user_id  # ch07:归属生效(与 DB 侧一致)

    async def snapshot(self, session_id: str) -> list[StoredMessage]:
        return list(self._sessions[session_id])

    async def append_user_message(self, session_id: str, content: str) -> str:
        """ch07 Task 16:prepare 阶段先落库本轮用户消息(中断轮不丢),返回行 id。"""
        if len(content) > self._max_chars:
            raise ValueError("message text exceeds MAX_MESSAGE_CHARS")
        self._sessions[session_id].append(StoredMessage("user", content))
        self._msg_seq += 1
        self._ids.setdefault(session_id, []).append(self._msg_seq)
        return str(self._msg_seq)

    async def commit_turn(self, session_id: str, messages: list[StoredMessage],
                          low_confidence: LowConfidenceRecord | None = None,
                          user_row_id: int | None = None) -> CommitTurnResult:
        validate_turn(messages, self._max_tool_calls)
        for m in messages:
            if m.content is not None and len(m.content) > self._max_chars and m.role != "tool":
                raise ValueError("message text exceeds MAX_MESSAGE_CHARS")
        if low_confidence is not None:  # 校验全过后才入池,与 DB 侧同成同败语义一致
            self.low_confidence.append(low_confidence)
        new_rows = list(messages[1:]) if user_row_id is not None else list(messages)
        msgs = self._sessions[session_id]
        msgs.extend(new_rows)
        limit = max(self._max_messages, self._max_tool_calls + 3)
        while len(msgs) > limit:
            cut = _trim_cut(msgs)
            if cut is None:
                break  # 始终保留最新完整 turn
            del msgs[:cut]
            del self._ids[session_id][:cut]  # 逐行 id 同步裁前段(含悬垂用户行)
        new_ids: list[int] = []
        for _ in new_rows:  # 校验全过后才分配,与入池同点
            self._msg_seq += 1
            new_ids.append(self._msg_seq)
        self._ids.setdefault(session_id, []).extend(new_ids)
        row_ids = [user_row_id, *new_ids] if user_row_id is not None else new_ids
        ids = [str(i) for i in row_ids]
        return CommitTurnResult(ids[0], ids)

    async def get_context_meta(self, session_id: str, user_id: str) -> ContextMeta:
        return self._meta.get(session_id, ContextMeta(None, None, None))

    async def list_conversations(self, user_id: str) -> list[ConversationItem]:
        out = []
        for sid, owner in self._owners.items():
            if owner != user_id:
                continue
            first_user = next((m for m in self._sessions[sid] if m.role == "user"), None)
            meta = self._meta.get(sid, ContextMeta(None, None, None))
            out.append(ConversationItem(sid, "进行中", None, None,
                                        (first_user.content or "")[:40] if first_user else None,
                                        meta.summary is not None))
        return out[::-1]  # 后建在前

    async def list_messages(self, session_id: str, user_id: str):
        if self._owners.get(session_id) != user_id:
            return None
        return [ConversationMessage(str(i), m.role, m.content, m.tool_calls)
                for i, m in zip(self._ids[session_id], self._sessions[session_id])
                if m.role in ("user", "assistant") and m.content]

    async def list_checkpoint_records(self, session_id: str, user_id: str):
        if self._owners.get(session_id) != user_id:
            return None
        return [PersistedMessageRecord(str(i), m.role, m.content, m.tool_calls, m.tool_call_id)
                for i, m in zip(self._ids[session_id], self._sessions[session_id])]

    def _turn_boundary_ids(self, session_id: str) -> set[int]:
        ids = self._ids.get(session_id, [])
        bounds = set()
        pos = 0
        for t in _turns(self._sessions.get(session_id, [])):  # 模块内直接引用
            pos += len(t)
            bounds.add(ids[pos - 1])
        return bounds

    async def move_layer1_from(self, session_id: str, user_id: str, new_id: int) -> bool:
        if self._owners.get(session_id) != user_id:
            return False
        meta = self._meta.get(session_id, ContextMeta(None, None, None))
        old = meta.layer1_from
        if old is not None and new_id <= old:
            return False
        if new_id not in self._ids.get(session_id, []):
            return False
        if new_id not in self._turn_boundary_ids(session_id):
            return False
        self._meta[session_id] = ContextMeta(meta.summary, meta.summary_upto, new_id)
        return True

    async def fetch_span_texts(self, session_id: str, from_id: int, upto_id: int):
        return [(i, m.role, m.content or "")
                for i, m in zip(self._ids.get(session_id, []), self._sessions.get(session_id, []))
                if from_id < i <= upto_id and m.role in ("user", "assistant")]

    async def append_summary(self, session_id: str, from_id: int, upto_id: int,
                             content: str, projection_tokens: int) -> SummaryAppendResult:
        from app.services.token_budget import build_summary_projection
        meta = self._meta.get(session_id, ContextMeta(None, None, None))
        if (meta.summary_upto or 0) != from_id:
            return SummaryAppendResult(None, False, "anchor-moved")
        if meta.layer1_from is None or upto_id > meta.layer1_from:
            return SummaryAppendResult(None, False, "beyond-layer1")
        segs = self._segments.setdefault(session_id, [])
        segs.append(content)
        seq = len(segs)
        projection = build_summary_projection(segs, projection_tokens)
        self._meta[session_id] = ContextMeta(projection, upto_id, meta.layer1_from)
        return SummaryAppendResult(seq, True, None)
