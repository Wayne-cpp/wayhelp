import asyncio
from datetime import datetime

from sqlalchemy import select, update

from app.models import Conversation, LowConfidenceQuestion, Message
from app.sessions import CommitTurnResult, LowConfidenceRecord, StoredMessage, validate_turn

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

    async def commit_turn(self, session_id: str, messages: list[StoredMessage],
                          low_confidence: LowConfidenceRecord | None = None) -> CommitTurnResult:
        validate_turn(messages, max_tool_calls=64)  # DB 侧结构校验;数量上限由编排层把关
        return CommitTurnResult(
            await asyncio.to_thread(self._commit_sync, session_id, messages, low_confidence))

    def _commit_sync(self, session_id: str, messages: list[StoredMessage],
                     low_confidence: LowConfidenceRecord | None = None) -> str:
        cid = int(session_id)
        with self._sf() as s:
            first = Message(conversation_id=cid, role=messages[0].role,
                            content=messages[0].content, tool_calls=messages[0].tool_calls,
                            tool_call_id=messages[0].tool_call_id)
            s.add(first)
            s.flush()  # 同事务内取 messages.id;失败整体回滚,不暴露半成品
            source_id = str(first.id)
            for m in messages[1:]:
                s.add(Message(conversation_id=cid, role=m.role, content=m.content,
                              tool_calls=m.tool_calls, tool_call_id=m.tool_call_id))
            s.execute(update(Conversation).where(Conversation.id == cid)
                      .values(updated_at=datetime.now()))
            if low_confidence is not None:
                s.add(LowConfidenceQuestion(
                    conversation_id=low_confidence.conversation_id,
                    raw_question=low_confidence.raw_question,
                    source=low_confidence.source,
                    reason=low_confidence.reason,
                ))
            s.commit()  # 任一失败整体回滚(Session 上下文管理器);低置信度入池与消息同事务
            return source_id
