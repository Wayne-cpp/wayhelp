import json
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatStreamRequest
from app.services.chat_service import (
    ChatService,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    SessionEvent,
)

router = APIRouter()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/v1/chat/stream")
async def chat_stream(body: ChatStreamRequest, request: Request) -> StreamingResponse:
    service: ChatService = request.app.state.chat_service
    prepared = await service.prepare(body.session_id, body.message)

    async def event_stream() -> AsyncIterator[str]:
        async for event in service.stream(prepared):
            if isinstance(event, SessionEvent):
                yield _sse({"type": "session", "session_id": event.session_id})
            elif isinstance(event, DeltaEvent):
                yield _sse({"type": "delta", "content": event.content})
            elif isinstance(event, DoneEvent):
                yield "data: [DONE]\n\n"
            elif isinstance(event, ErrorEvent):
                yield _sse({"type": "error", "code": event.code, "message": event.message})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
