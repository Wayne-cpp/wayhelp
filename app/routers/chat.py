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
    PreparedTurn,
    SessionEvent,
)

router = APIRouter()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class _EventStream:
    """包装 SSE async generator,保证从未迭代即被 aclose 时也释放 session 锁。

    Starlette 的 StreamingResponse 不会调用 body iterator 的 aclose,
    且 Python 3.12 下未启动的 async generator 执行 aclose() 不会跑其 body
    的 finally,因此锁释放必须由包装层的 aclose 兜底(幂等)。
    """

    def __init__(self, service: ChatService, prepared: PreparedTurn):
        self._service = service
        self._prepared = prepared
        self._gen = self._body()

    async def _body(self) -> AsyncIterator[str]:
        try:
            async for event in self._service.stream(self._prepared):
                if isinstance(event, SessionEvent):
                    yield _sse({"type": "session", "session_id": event.session_id})
                elif isinstance(event, DeltaEvent):
                    yield _sse({"type": "delta", "content": event.content})
                elif isinstance(event, DoneEvent):
                    yield "data: [DONE]\n\n"
                elif isinstance(event, ErrorEvent):
                    yield _sse({"type": "error", "code": event.code, "message": event.message})
        finally:
            self._service.release_turn(self._prepared)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._gen

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        try:
            await self._gen.aclose()
        finally:
            self._service.release_turn(self._prepared)


def event_stream(service: ChatService, prepared: PreparedTurn) -> _EventStream:
    return _EventStream(service, prepared)


@router.post("/v1/chat/stream")
async def chat_stream(body: ChatStreamRequest, request: Request) -> StreamingResponse:
    service: ChatService = request.app.state.chat_service
    prepared = await service.prepare(body.session_id, body.message)

    return StreamingResponse(
        event_stream(service, prepared),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
