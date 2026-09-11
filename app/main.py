from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db import make_engine, make_session_factory, ping
from app.knowledge.embedding import build_embeddings
from app.knowledge.retriever import KnowledgeRetriever
from app.errors import (
    AppError,
    MessageTooLongError,
    SessionCapacityReachedError,
    SessionNotFoundError,
    UpstreamError,
)
from app.prompts.service import SERVICE_SYSTEM_PROMPT
from app.routers.chat import router as chat_router
from app.routers.extract import router as extract_router
from app.services.chat_service import ChatService
from app.store_db import DbSessionStore
from app.tools.business import build_tools


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


@dataclass(frozen=True)
class AppRuntime:
    store: Any  # SessionStore 协议
    toolset_factory: Callable[[str], list[BaseTool]]
    retriever: Any = None


def _build_production_runtime(settings: Settings) -> AppRuntime:
    try:
        engine = make_engine(settings.database_url)
        ping(engine)
    except Exception as exc:
        raise RuntimeError(f"database ping failed: {type(exc).__name__}") from exc
    session_factory = make_session_factory(engine)
    # 缺 Key 时构造禁用检索的实例,不构造需要 Key 的 embedding 客户端(spec §8)
    embed = build_embeddings(settings) if settings.has_embedding_key() else None
    retriever = KnowledgeRetriever(settings, embed=embed, session_factory=session_factory)

    def toolset_factory(session_id: str) -> list[BaseTool]:
        return build_tools(session_factory, int(session_id), retriever)

    return AppRuntime(store=DbSessionStore(session_factory, settings.max_message_chars),
                      toolset_factory=toolset_factory, retriever=retriever)


def create_app(settings: Settings | None = None, model: Any | None = None,
               runtime: AppRuntime | None = None) -> FastAPI:
    settings = settings or Settings()
    if model is None:
        model = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_tokens=settings.max_output_tokens,
        )
    if runtime is None:
        runtime = _build_production_runtime(settings)
    if count_tokens_approximately([SystemMessage(content=SERVICE_SYSTEM_PROMPT)]) >= settings.max_input_tokens:
        raise RuntimeError("system prompt alone exhausts the input token budget")

    from app.chains.extract_chain import build_extract_prompt

    if (
        count_tokens_approximately(build_extract_prompt().invoke({"text": ""}).to_messages())
        >= settings.max_input_tokens
    ):
        raise RuntimeError("extraction few-shot prompt alone exhausts the input token budget")

    service = ChatService(runtime.store, model, settings, SERVICE_SYSTEM_PROMPT,
                          runtime.toolset_factory)

    owns_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if owns_runtime and runtime.retriever is not None:
            runtime.retriever.close()

    app = FastAPI(title="wayhelp-ch03", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model
    app.state.store = runtime.store
    app.state.chat_service = service
    app.include_router(chat_router)
    app.include_router(extract_router)

    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    async def chat_ui() -> FileResponse:
        return FileResponse(static_dir / "chat.html")

    @app.get("/1784959384051.jpg", include_in_schema=False)
    async def brand_mark() -> FileResponse:
        return FileResponse(static_dir / "1784959384051.jpg")

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

    @app.exception_handler(AppError)
    async def _(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=500, content=_error_body(exc.code, "服务内部错误"))

    @app.exception_handler(Exception)
    async def _(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=_error_body("internal_error", "服务内部错误"))

    return app
