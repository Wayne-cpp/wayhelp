import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import Settings
from app.db import check_ch04_tables, check_ch07_tables, check_ch08_tables, check_ch09_tables, make_engine, make_session_factory, ping
from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.jobs.runner import JobRunner
from app.knowledge.embedding import build_embeddings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.reranker import SiliconFlowReranker
from app.knowledge.retriever import KnowledgeRetriever
from app.knowledge.state import KnowledgeState, KnowledgeStateHolder
from app.errors import (
    AppError,
    FeedbackConflictError,
    MessageTooLongError,
    ResumeConflictError,
    SessionCapacityReachedError,
    SessionNotFoundError,
    UpstreamError,
)
from app.prompts.service import SERVICE_SYSTEM_PROMPT
from app.routers.chat import router as chat_router
from app.routers.conversations import router as conversations_router
from app.routers.extract import router as extract_router
from app.routers.jobs import router as jobs_router
from app.routers.kb import router as kb_router
from app.routers.rag_eval import eval_runs_router
from app.routers.rag_eval import router as rag_eval_router
from app.services import eval_runs as eval_runs_service
from app.services import rag_eval as rag_eval_service
from app.services.rag_eval import ReportCorruptError
from app.services.chat_service import ChatService
from app.services.kb_admin import DEFAULT_DOCS_DIR, KbAdminError
from app.services.summarizer import SummaryRunner
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
        check_ch07_tables(engine)  # 缺 ch07 列/表同样启动失败,提示升级命令
        check_ch08_tables(engine)  # 缺 ch08 审计/幂等表同样启动失败,提示升级命令
        check_ch09_tables(engine)  # 缺 ch09 表/列同样启动失败,提示升级命令
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
    # 最小日志配置(master 裁决):生产侧 root 无 handler 时 wayhelp.graph 的 INFO
    # 日志(retrieve/confidence_gate 等)无处输出;basicConfig 幂等,已配环境无副作用
    logging.basicConfig()
    log_dir = Path(__file__).resolve().parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    target = str(log_dir / "app.log")
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target
               for h in root.handlers):
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(fh)
    logging.getLogger("wayhelp.graph").setLevel(logging.INFO)
    settings = settings or Settings()
    if model is None:
        model = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_tokens=settings.max_output_tokens,
            stream_usage=True,
        )
    owns_runtime = runtime is None
    if runtime is None:
        runtime = _build_production_runtime(settings, model)
    elif runtime.session_factory is not None:
        # 裁决 B:注入 session_factory 的测试/嵌入路径 store 换 DbSessionStore,与生产同构
        # (spec §13 验收 3/10:消息/低置信入池/动作端点全链路写同一 DB 账本)
        runtime = replace(runtime, store=DbSessionStore(
            runtime.session_factory, settings.max_message_chars))
    # ch07 启动预算自检(替代旧 system-prompt 探针):按真实工具面实测 SYS_TOKENS,
    # 打全分量预算日志;装不下一轮 steady 开销打 critical 交运维决策,不阻断启动
    from app.graph.agent_node import suggest_options
    from app.services.token_budget import compute_budget, measure_sys_tokens
    from app.tools.builtin import scan_builtin_specs
    from app.tools.executor import ToolFace
    probe_tools = [*ToolFace(scan_builtin_specs()).as_langchain_tools(), suggest_options]
    context_budget = compute_budget(settings, measure_sys_tokens(SERVICE_SYSTEM_PROMPT, probe_tools))
    logging.getLogger("wayhelp.graph").info(
        "context budget window=%d output=%d user_input=%d peak=%d fixed=%d sys=%d avail=%d total=%d l1=%d l2=%d",
        settings.model_context_window, settings.max_output_tokens,
        settings.max_user_input_tokens, context_budget.peak, context_budget.fixed,
        context_budget.sys_tokens, context_budget.avail, context_budget.total,
        context_budget.layer1, context_budget.layer2)
    if not context_budget.sufficient:
        logging.getLogger("wayhelp.graph").critical(
            "上下文预算不足:total=%d < steady=%d,请调大 MODEL_CONTEXT_WINDOW 或调小固定开销",
            context_budget.total, settings.steady_tokens_per_turn)

    # ch08 工具目录与 MCP 网关(spec §1.5):内置启动登记;MCP 每轮现问现拿
    from app.services.mcp_gateway import McpGateway
    from app.tools.builtin import scan_builtin_specs
    from app.tools.catalog import ToolCatalog
    tool_catalog = ToolCatalog()
    for _spec in scan_builtin_specs():
        tool_catalog.register(_spec)
    mcp_gateway = McpGateway(settings)

    from app.chains.extract_chain import build_extract_prompt

    if (
        count_tokens_approximately(build_extract_prompt().invoke({"text": ""}).to_messages())
        >= settings.max_input_tokens
    ):
        raise RuntimeError("extraction few-shot prompt alone exhausts the input token budget")

    summary_runner = SummaryRunner(runtime.store, model, settings)
    service = ChatService(runtime.store, model, settings, SERVICE_SYSTEM_PROMPT,
                          session_factory=runtime.session_factory)
    deps = GraphDeps(model=model, settings=settings, retriever=runtime.retriever,
                     store=runtime.store, system_prompt=SERVICE_SYSTEM_PROMPT,
                     context_budget=context_budget, summary_runner=summary_runner,
                     catalog=tool_catalog, mcp_gateway=mcp_gateway,
                     session_factory=runtime.session_factory)
    if not owns_runtime:  # 测试/嵌入路径:内存 checkpointer 即刻可用
        service.set_graph(build_chat_graph(deps, InMemorySaver()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sched_task = None
        if owns_runtime and settings.eval_schedule_enabled:
            # ch09 评估定时(spec §5.5):单 worker 内存 task,不引 APScheduler
            from app.services.eval_scheduler import eval_scheduler_loop
            sched_task = asyncio.create_task(
                eval_scheduler_loop(app.state.job_runner, settings))
        if owns_runtime:  # 生产:SQLite checkpointer 由 lifespan 托管,启停对称
            async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as cp:
                service.set_graph(build_chat_graph(deps, cp))
                yield
        else:
            yield
        if sched_task is not None:  # 关闭先 cancel/wait 调度器,再收尾 job runner
            sched_task.cancel()
            with suppress(asyncio.CancelledError):
                await sched_task
        await app.state.job_runner.close()
        await app.state.summary_runner.aclose()
        from app.services.langfuse_tracing import flush_langfuse
        flush_langfuse()  # ch09:进程退出前投递残余 trace 事件
        from app.tools.executor import PENDING_WRITE_TASKS
        if PENDING_WRITE_TASKS:
            await asyncio.gather(*PENDING_WRITE_TASKS, return_exceptions=True)
        await app.state.mcp_gateway.close()
        if owns_runtime and runtime.retriever is not None:
            runtime.retriever.close()

    app = FastAPI(title="wayhelp", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model
    app.state.store = runtime.store
    app.state.chat_service = service
    app.state.summary_runner = summary_runner
    app.state.session_factory = runtime.session_factory
    app.state.tool_catalog = tool_catalog
    app.state.mcp_gateway = mcp_gateway
    app.state.embed = runtime.embed
    app.state.kb_store = runtime.kb_store
    # 直接构造的 AppRuntime 可不带 holder(测试/嵌入):装配补默认,KB 端点不炸
    app.state.knowledge_state = (runtime.knowledge_state
                                 if runtime.knowledge_state is not None
                                 else KnowledgeStateHolder())
    app.state.retriever = runtime.retriever
    app.state.kb_docs_dir = DEFAULT_DOCS_DIR
    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.include_router(extract_router)
    app.include_router(kb_router)
    app.include_router(jobs_router)
    app.include_router(rag_eval_router)
    app.include_router(eval_runs_router)

    root_dir = Path(__file__).resolve().parent.parent
    app.state.rag_eval_report_path = root_dir / "evals" / "results" / "rag_eval.json"

    def _report_loader():
        try:
            return rag_eval_service.load_report(app.state.rag_eval_report_path)
        except Exception:
            return None

    job_runner = JobRunner(log_dir=root_dir / "data" / "jobs",
                           report_loader=_report_loader, cwd=root_dir,
                           on_report_published=(
                               (lambda report, by: eval_runs_service.record_run(
                                   runtime.session_factory, report, by))
                               if runtime.session_factory is not None else None))
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

    @app.exception_handler(ResumeConflictError)
    async def _(request: Request, exc: ResumeConflictError) -> JSONResponse:
        # ch06 spec §9.1:旧卡/已恢复卡/重复点击/新轮再挂起时的旧卡均适用
        return JSONResponse(status_code=409, content=_error_body(exc.code, "该选择已失效"))

    @app.exception_handler(FeedbackConflictError)
    async def _(request: Request, exc: FeedbackConflictError) -> JSONResponse:
        # ch09 spec §5.3:非最终回答/中间工具行/跨会话/反向反馈均适用
        return JSONResponse(status_code=409,
                            content=_error_body(exc.code, "该反馈目标不可用或已反馈"))

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
