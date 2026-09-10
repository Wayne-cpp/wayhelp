import asyncio
from datetime import datetime

from sqlalchemy import select, update

from app.models import Conversation, Message
from app.sessions import StoredMessage, validate_turn


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
        with self._sf() as s:
            row = s.execute(
                select(Conversation.id).where(
                    Conversation.id == int(session_id),
                    Conversation.user_id == user_id,
                )
            ).first()
            return row is not None

    async def snapshot(self, session_id: str) -> list[StoredMessage]:
        return await asyncio.to_thread(self._snapshot_sync, session_id)

    def _snapshot_sync(self, session_id: str) -> list[StoredMessage]:
        with self._sf() as s:
            rows = (
                s.query(Message)
                .filter(Message.conversation_id == int(session_id))
                .order_by(Message.created_at, Message.id)
                .all()
            )
            return [
                StoredMessage(r.role, r.content, r.tool_calls, r.tool_call_id)
                for r in rows
            ]

    async def commit_turn(self, session_id: str, messages: list[StoredMessage]) -> None:
        validate_turn(messages, max_tool_calls=64)  # DB 侧结构校验;数量上限由编排层把关
        await asyncio.to_thread(self._commit_sync, session_id, messages)

    def _commit_sync(self, session_id: str, messages: list[StoredMessage]) -> None:
        cid = int(session_id)
        with self._sf() as s:
            for m in messages:
                s.add(Message(
                    conversation_id=cid,
                    role=m.role,
                    content=m.content,
                    tool_calls=m.tool_calls,
                    tool_call_id=m.tool_call_id,
                ))
            s.execute(
                update(Conversation)
                .where(Conversation.id == cid)
                .values(updated_at=datetime.now())
            )
            s.commit()  # 任一失败整体回滚(Session 上下文管理器)
