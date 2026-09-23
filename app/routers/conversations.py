"""ch07 会话只读接口(spec §11):侧栏列表 + 历史回放;零写动作。"""

from fastapi import APIRouter, Request

from app.errors import SessionNotFoundError
from app.schemas import ConversationItemOut, ConversationMessageOut

router = APIRouter(prefix="/api", tags=["conversations"])


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


@router.get("/conversations", response_model=list[ConversationItemOut])
async def list_conversations(user_id: str, request: Request):
    items = await request.app.state.store.list_conversations(user_id)
    return [ConversationItemOut(id=i.id, status=i.status, created_at=_iso(i.created_at),
                                updated_at=_iso(i.updated_at), preview=i.preview,
                                summarized=i.summarized) for i in items]


@router.get("/conversations/{session_id}/messages",
            response_model=list[ConversationMessageOut])
async def list_messages(session_id: str, user_id: str, request: Request):
    rows = await request.app.state.store.list_messages(session_id, user_id)
    if rows is None:
        raise SessionNotFoundError("session not found")  # 404 不泄露
    return [ConversationMessageOut(id=r.id, role=r.role, content=r.content,
                                   created_at=_iso(r.created_at)) for r in rows]
