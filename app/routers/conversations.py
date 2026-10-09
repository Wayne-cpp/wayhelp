"""ch07 会话只读接口(spec §11):侧栏列表 + 历史回放;零写动作。"""

import asyncio

from fastapi import APIRouter, Request

from app.errors import SessionNotFoundError
from app.schemas import ConversationItemOut, ConversationMessageOut

router = APIRouter(prefix="/api", tags=["conversations"])


class AnnotatedMessageOut(ConversationMessageOut):
    """ch09:历史回放补反馈态;无反馈数据时按默认(eligible=False)降级。"""

    feedback_eligible: bool = False
    feedback_sentiment: str | None = None


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


@router.get("/conversations", response_model=list[ConversationItemOut])
async def list_conversations(user_id: str, request: Request):
    items = await request.app.state.store.list_conversations(user_id)
    return [ConversationItemOut(id=i.id, status=i.status, created_at=_iso(i.created_at),
                                updated_at=_iso(i.updated_at), preview=i.preview,
                                summarized=i.summarized) for i in items]


@router.get("/conversations/{session_id}/messages",
            response_model=list[AnnotatedMessageOut])
async def list_messages(session_id: str, user_id: str, request: Request):
    rows = await request.app.state.store.list_messages(session_id, user_id)
    if rows is None:
        raise SessionNotFoundError("session not found")  # 404 不泄露
    msgs = [{"id": r.id, "role": r.role, "content": r.content,
             "created_at": _iso(r.created_at)} for r in rows]
    sf = getattr(request.app.state, "session_factory", None)
    cid = int(session_id) if session_id.isdigit() else None
    if sf is not None and cid is not None:
        from app.services.feedback import annotate_messages
        msgs = await asyncio.to_thread(annotate_messages, sf, cid, msgs)
    return [AnnotatedMessageOut(**m) for m in msgs]
