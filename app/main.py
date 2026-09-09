from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.errors import (
    MessageTooLongError,
    SessionCapacityReachedError,
    SessionNotFoundError,
    UpstreamError,
)
from app.prompts.service import SERVICE_SYSTEM_PROMPT
from app.routers.chat import router as chat_router
from app.routers.extract import router as extract_router
from app.services.chat_service import ChatService
from app.sessions import InMemorySessionStore


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def create_app(settings: Settings | None = None, model: Any | None = None) -> FastAPI:
    settings = settings or Settings()
    if model is None:
        model = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_tokens=settings.max_output_tokens,
        )
    if count_tokens_approximately([SystemMessage(content=SERVICE_SYSTEM_PROMPT)]) >= settings.max_input_tokens:
        raise RuntimeError("system prompt alone exhausts the input token budget")

    from app.chains.extract_chain import build_extract_prompt

    if (
        count_tokens_approximately(build_extract_prompt().invoke({"text": ""}).to_messages())
        >= settings.max_input_tokens
    ):
        raise RuntimeError("extraction few-shot prompt alone exhausts the input token budget")

    store = InMemorySessionStore(
        settings.max_sessions, settings.max_messages_per_session, settings.max_message_chars
    )
    service = ChatService(store, model, settings, SERVICE_SYSTEM_PROMPT)

    app = FastAPI(title="mewhelp-ch01")
    app.state.settings = settings
    app.state.model = model
    app.state.store = store
    app.state.chat_service = service
    app.include_router(chat_router)
    app.include_router(extract_router)

    @app.exception_handler(MessageTooLongError)
    async def _(request: Request, exc: MessageTooLongError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_error_body(exc.code, "输入超出长度限制"))

    @app.exception_handler(SessionNotFoundError)
    async def _(request: Request, exc: SessionNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_error_body(exc.code, "会话不存在"))

    @app.exception_handler(SessionCapacityReachedError)
    async def _(request: Request, exc: SessionCapacityReachedError) -> JSONResponse:
        return JSONResponse(status_code=503, content=_error_body(exc.code, "会话容量已满,请稍后重试"))

    @app.exception_handler(UpstreamError)
    async def _(request: Request, exc: UpstreamError) -> JSONResponse:
        return JSONResponse(status_code=502, content=_error_body(exc.code, "上游模型暂时不可用"))

    return app


app = create_app()
