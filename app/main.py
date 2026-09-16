from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db import check_ch04_tables, make_engine, make_session_factory, ping
from app.jobs.runner import JobRunner
from app.knowledge.embedding import build_embeddings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.reranker import SiliconFlowReranker
from app.knowledge.retriever import KnowledgeRetriever
from app.knowledge.state import KnowledgeState, KnowledgeStateHolder
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
from app.routers.jobs import router as jobs_router
from app.routers.kb import router as kb_router
from app.routers.rag_eval import router as rag_eval_router
from app.services import rag_eval as rag_eval_service
from app.services.rag_eval import ReportCorruptError
from app.services.chat_service import ChatService
from app.services.kb_admin import DEFAULT_DOCS_DIR, KbAdminError
from app.store_db import DbSessionStore
from app.tools.business import build_tools


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


@dataclass(frozen=True)
class AppRuntime:
    store: Any  # SessionStore 协议
    toolset_factory: Callable[[str], list[BaseTool]]
    retriever: Any = None
    session_factory: Any = None   # /kb 管理动作复用
    embed: Any = None             # None = 未配置 EMBEDDING_API_KEY
    kb_store: Any = None          # MilvusKnowledgeStore,与 retriever 同一实例
    knowledge_state: Any = None   # KnowledgeStateHolder(T11)


def _build_production_runtime(settings: Settings, model) -> AppRuntime:
    try:
        engine = make_engine(settings.database_url)
        ping(engine)
        check_ch04_tables(engine)  # 缺表启动失败,提示升级命令
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"database ping failed: {type(exc).__name__}") from exc
    session_factory = make_session_factory(engine)
    # 缺 Key 时构造禁用检索的实例,不构造需要 Key 的 embedding 客户端(spec §8)
    embed = build_embeddings(settings) if settings.has_embedding_key() else None
    kb_store = MilvusKnowledgeStore(settings.milvus_uri, settings.embedding_dim)
    knowledge_state = KnowledgeStateHolder()  # 初值 ready;旧 schema 探针后置 rebuild_required
    if kb_store.file_exists():
        try:
            if kb_store.contract_error() is not None:
                knowledge_state.set(KnowledgeState.REBUILD_REQUIRED)
        except Exception:
            knowledge_state.set(KnowledgeState.REBUILD_REQUIRED)
    reranker = SiliconFlowReranker(settings) if settings.has_rerank_key() else None
    retriever = KnowledgeRetriever(settings, embed=embed, store=kb_store,
                                   session_factory=session_factory, model=model,
                                   reranker=reranker, state=knowledge_state)

    def toolset_factory(session_id: str) -> list[BaseTool]:
        return build_tools(session_factory, int(session_id), retriever, settings)

    return AppRuntime(store=DbSessionStore(session_factory, settings.max_message_chars),
                      toolset_factory=toolset_factory, retriever=retriever,
                      session_factory=session_factory, embed=embed, kb_store=kb_store,
                      knowledge_state=knowledge_state)


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
    owns_runtime = runtime is None
    if runtime is None:
        runtime = _build_production_runtime(settings, model)
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.job_runner.close()   # 先收评估子进程
        if owns_runtime and runtime.retriever is not None:
            runtime.retriever.close()

    app = FastAPI(title="wayhelp-ch04", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model
    app.state.store = runtime.store
    app.state.chat_service = service
    app.state.session_factory = runtime.session_factory
    app.state.embed = runtime.embed
    app.state.kb_store = runtime.kb_store
    # 直接构造的 AppRuntime 可不带 holder(测试/嵌入):装配补默认,KB 端点不炸
    app.state.knowledge_state = (runtime.knowledge_state
                                 if runtime.knowledge_state is not None
                                 else KnowledgeStateHolder())
    app.state.retriever = runtime.retriever
    app.state.kb_docs_dir = DEFAULT_DOCS_DIR
    app.include_router(chat_router)
    app.include_router(extract_router)
    app.include_router(kb_router)
    app.include_router(jobs_router)
    app.include_router(rag_eval_router)

    root_dir = Path(__file__).resolve().parent.parent
    app.state.rag_eval_report_path = root_dir / "evals" / "results" / "rag_eval.json"

    def _report_loader():
        try:
            return rag_eval_service.load_report(app.state.rag_eval_report_path)
        except Exception:
            return None

    job_runner = JobRunner(log_dir=root_dir / "data" / "jobs",
                           report_loader=_report_loader, cwd=root_dir)
    job_runner.register("eval-rag",
                        ["uv", "run", "python", "evals/run_retrieval_compare.py"])
    app.state.job_runner = job_runner

    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    async def chat_ui() -> FileResponse:
        return FileResponse(static_dir / "chat.html")

    @app.get("/kb", include_in_schema=False)
    async def kb_ui() -> FileResponse:
        return FileResponse(static_dir / "kb.html")

    @app.get("/rag-eval", include_in_schema=False)
    async def rag_eval_ui() -> FileResponse:
        return FileResponse(static_dir / "rag-eval.html")

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

    @app.exception_handler(KbAdminError)
    async def _(request: Request, exc: KbAdminError) -> JSONResponse:
        return JSONResponse(status_code=exc.status,
                            content=_error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422,
                            content=_error_body("invalid_request", "请求参数不合法"))

    @app.exception_handler(ReportCorruptError)
    async def _(request: Request, exc: ReportCorruptError) -> JSONResponse:
        return JSONResponse(status_code=502,
                            content=_error_body("rag_eval_report_corrupt",
                                                "评估报告文件损坏或契约不合法"))

    @app.exception_handler(Exception)
    async def _(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=_error_body("internal_error", "服务内部错误"))

    return app
