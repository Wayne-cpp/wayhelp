import asyncio

from sqlalchemy import func, select, update

from app.models import Conversation, ConversationSummary, LowConfidenceQuestion, Message
from app.sessions import (
    CommitTurnResult, ContextMeta, ConversationItem, ConversationMessage,
    LowConfidenceRecord, PersistedMessageRecord, StoredMessage, SummaryAppendResult,
    validate_turn,
)
from app.services.token_budget import build_summary_projection

_BIGINT_MAX = (1 << 63) - 1  # conversations.id 为 BIGINT 自增


def _as_db_id(session_id: str) -> int | None:
    """spec §4:仅正十进制且在 BIGINT 正范围内才是 DB 形态 id;
    另一种合法形态(规范 UUID)按不存在处理,返回 None 不查库。"""
    if not session_id.isdecimal():
        return None
    n = int(session_id)
    return n if 0 < n <= _BIGINT_MAX else None


class DbSessionStore:
    """SessionStore 的 MySQL 实现;所有方法 async,同步 Session 在线程内创建/关闭。"""

    def __init__(self, session_factory, max_message_chars: int):
        self._sf = session_factory
        self._max_chars = max_message_chars

    async def create(self, user_id: str) -> str:
        return await asyncio.to_thread(self._create_sync, user_id)

    def _create_sync(self, user_id: str) -> str:
        with self._sf() as s:
            conv = Conversation(user_id=user_id)
            s.add(conv)
            s.commit()
            return str(conv.id)

    async def exists(self, session_id: str, user_id: str) -> bool:
        return await asyncio.to_thread(self._exists_sync, session_id, user_id)

    def _exists_sync(self, session_id: str, user_id: str) -> bool:
        cid = _as_db_id(session_id)
        if cid is None:
            return False  # 外来合法形态(规范 UUID)/越界十进制:按不存在处理
        with self._sf() as s:
            row = s.execute(
                select(Conversation.id).where(
                    Conversation.id == cid,
                    Conversation.user_id == user_id,
                )
            ).first()
            return row is not None

    async def snapshot(self, session_id: str) -> list[StoredMessage]:
        return await asyncio.to_thread(self._snapshot_sync, session_id)

    def _snapshot_sync(self, session_id: str) -> list[StoredMessage]:
        cid = _as_db_id(session_id)
        if cid is None:
            return []  # 同 exists:外来合法形态视为无历史
        with self._sf() as s:
            rows = (
                s.query(Message)
                .filter(Message.conversation_id == cid)
                .order_by(Message.created_at, Message.id)
                .all()
            )
            return [
                StoredMessage(r.role, r.content, r.tool_calls, r.tool_call_id)
                for r in rows
            ]

    async def append_user_message(self, session_id: str, content: str) -> str:
        """ch07 Task 16:prepare 阶段先落库本轮用户消息(中断轮不丢),返回行 id。"""
        return await asyncio.to_thread(self._append_user_sync, session_id, content)

    def _append_user_sync(self, session_id: str, content: str) -> str:
        from app.errors import SessionNotFoundError
        if len(content) > self._max_chars:
            raise ValueError("message text exceeds MAX_MESSAGE_CHARS")
        cid = _as_db_id(session_id)
        if cid is None:
            raise SessionNotFoundError("session not found")
        with self._sf() as s:
            row = Message(conversation_id=cid, role="user", content=content)
            s.add(row)
            s.commit()
            return str(row.id)

    async def commit_turn(self, session_id: str, messages: list[StoredMessage],
                          low_confidence: LowConfidenceRecord | None = None,
                          user_row_id: int | None = None) -> CommitTurnResult:
        validate_turn(messages, max_tool_calls=64)  # DB 侧结构校验;数量上限由编排层把关
        ids = await asyncio.to_thread(self._commit_sync, session_id, messages,
                                      low_confidence, user_row_id)
        return CommitTurnResult(ids[0], ids)

    def _commit_sync(self, session_id: str, messages: list[StoredMessage],
                     low_confidence: LowConfidenceRecord | None = None,
                     user_row_id: int | None = None) -> list[str]:
        cid = int(session_id)
        with self._sf() as s:
            # ch07 Task 16:prepare 已落库并盖章的用户行不重复插,只补其余行
            rows = [Message(conversation_id=cid, role=m.role, content=m.content,
                            tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                    for m in (messages[1:] if user_row_id is not None else messages)]
            s.add_all(rows)
            s.flush()  # 同事务内统一取全部 messages.id;失败整体回滚,不暴露半成品
            ids = ([str(user_row_id)] if user_row_id is not None else []) \
                + [str(r.id) for r in rows]
            s.execute(update(Conversation).where(Conversation.id == cid)
                      # 一律走 DB 时钟:与 created_at 的 DEFAULT CURRENT_TIMESTAMP 同一
                      # 钟域,Python 侧写值在容器与主机时区不一致时会反超新建行,
                      # 令 list_conversations 的 updated_at DESC 序反转
                      .values(updated_at=func.now()))
            if low_confidence is not None:
                s.add(LowConfidenceQuestion(
                    conversation_id=low_confidence.conversation_id,
                    raw_question=low_confidence.raw_question,
                    source=low_confidence.source,
                    reason=low_confidence.reason,
                ))
            s.commit()  # 任一失败整体回滚(Session 上下文管理器);低置信度入池与消息同事务
            return ids

    async def get_context_meta(self, session_id: str, user_id: str) -> ContextMeta:
        return await asyncio.to_thread(self._get_context_meta_sync, session_id, user_id)

    def _get_context_meta_sync(self, session_id, user_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return ContextMeta(None, None, None)
        with self._sf() as s:
            row = (s.query(Conversation.summary, Conversation.summary_upto_msg_id,
                           Conversation.layer1_from_msg_id)
                   .filter(Conversation.id == cid, Conversation.user_id == user_id)
                   .first())
            if row is None:
                return ContextMeta(None, None, None)
            return ContextMeta(row[0], row[1], row[2])

    async def list_conversations(self, user_id: str) -> list[ConversationItem]:
        return await asyncio.to_thread(self._list_conversations_sync, user_id)

    def _list_conversations_sync(self, user_id):
        with self._sf() as s:
            first_q = (s.query(Message.content)
                       .filter(Message.conversation_id == Conversation.id,
                               Message.role == "user")
                       .order_by(Message.id).limit(1)
                       .correlate(Conversation).scalar_subquery())
            rows = (s.query(Conversation, first_q.label("first_q"))
                    .filter(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                    .all())
            return [ConversationItem(str(c.id), c.status, c.created_at, c.updated_at,
                                     ((fq or "")[:40] or None), c.summary is not None)
                    for c, fq in rows]

    async def list_messages(self, session_id: str, user_id: str):
        return await asyncio.to_thread(self._list_messages_sync, session_id, user_id)

    def _list_messages_sync(self, session_id, user_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return None
        with self._sf() as s:
            owner = (s.query(Conversation.id)
                     .filter(Conversation.id == cid, Conversation.user_id == user_id)
                     .first())
            if owner is None:
                return None
            rows = (s.query(Message)
                    .filter(Message.conversation_id == cid,
                            Message.role.in_(("user", "assistant")),
                            Message.content.isnot(None))
                    .order_by(Message.id).all())
            return [ConversationMessage(str(r.id), r.role, r.content, r.tool_calls,
                                        r.created_at) for r in rows]

    async def list_checkpoint_records(self, session_id: str, user_id: str):
        return await asyncio.to_thread(self._list_checkpoint_records_sync, session_id, user_id)

    def _list_checkpoint_records_sync(self, session_id, user_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return None
        with self._sf() as s:
            owner = (s.query(Conversation.id)
                     .filter(Conversation.id == cid, Conversation.user_id == user_id)
                     .first())
            if owner is None:
                return None
            rows = (s.query(Message).filter(Message.conversation_id == cid)
                    .order_by(Message.id).all())
            return [PersistedMessageRecord(str(r.id), r.role, r.content, r.tool_calls,
                                           r.tool_call_id, r.created_at) for r in rows]

    async def move_layer1_from(self, session_id: str, user_id: str, new_id: int) -> bool:
        return await asyncio.to_thread(self._move_layer1_from_sync, session_id, user_id, new_id)

    def _move_layer1_from_sync(self, session_id, user_id, new_id) -> bool:
        cid = _as_db_id(session_id)
        if cid is None:
            return False
        with self._sf() as s:
            conv = (s.query(Conversation)
                    .filter(Conversation.id == cid, Conversation.user_id == user_id)
                    .with_for_update().first())
            if conv is None:
                return False
            old = conv.layer1_from_msg_id
            if old is not None and new_id <= old:
                return False
            msg = s.get(Message, new_id)
            if msg is None or msg.conversation_id != cid:
                return False
            nxt = (s.query(Message.role)
                   .filter(Message.conversation_id == cid, Message.id > new_id)
                   .order_by(Message.id).limit(1).first())
            if nxt is not None and nxt[0] != "user":
                return False                       # 不在完整轮边界
            conv.layer1_from_msg_id = new_id
            s.commit()
            return True

    async def fetch_span_texts(self, session_id: str, from_id: int, upto_id: int):
        return await asyncio.to_thread(self._fetch_span_texts_sync, session_id, from_id, upto_id)

    def _fetch_span_texts_sync(self, session_id, from_id, upto_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return []
        with self._sf() as s:
            rows = (s.query(Message.id, Message.role, Message.content)
                    .filter(Message.conversation_id == cid,
                            Message.id > from_id, Message.id <= upto_id,
                            Message.role.in_(("user", "assistant")))
                    .order_by(Message.id).all())
            return [(int(r[0]), r[1], r[2] or "") for r in rows]

    async def append_summary(self, session_id: str, from_id: int, upto_id: int,
                             content: str, projection_tokens: int) -> SummaryAppendResult:
        return await asyncio.to_thread(self._append_summary_sync, session_id, from_id,
                                       upto_id, content, projection_tokens)

    def _append_summary_sync(self, session_id, from_id, upto_id, content,
                             projection_tokens) -> SummaryAppendResult:
        cid = _as_db_id(session_id)
        if cid is None:
            return SummaryAppendResult(None, False, "bad-session")
        with self._sf() as s:
            conv = (s.query(Conversation).filter(Conversation.id == cid)
                    .with_for_update().first())
            if conv is None:
                return SummaryAppendResult(None, False, "bad-session")
            if (conv.summary_upto_msg_id or 0) != from_id:
                return SummaryAppendResult(None, False, "anchor-moved")
            if conv.layer1_from_msg_id is None or upto_id > conv.layer1_from_msg_id:
                return SummaryAppendResult(None, False, "beyond-layer1")
            seq = (s.query(func.max(ConversationSummary.seq))
                   .filter(ConversationSummary.conversation_id == cid).scalar() or 0) + 1
            s.add(ConversationSummary(conversation_id=cid, seq=seq, from_msg_id=from_id,
                                      upto_msg_id=upto_id, content=content))
            s.flush()
            segs = [r[0] for r in s.query(ConversationSummary.content)
                    .filter(ConversationSummary.conversation_id == cid)
                    .order_by(ConversationSummary.seq).all()]
            conv.summary = build_summary_projection(segs, projection_tokens)
            conv.summary_upto_msg_id = upto_id
            s.commit()
            return SummaryAppendResult(seq, True, None)
