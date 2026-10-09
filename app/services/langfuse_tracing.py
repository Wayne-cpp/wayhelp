"""ch09:Langfuse(SDK v3)接入。未启用时全部 no-op;任何异常只记日志不拦主路。

SDK 从环境变量读密钥(get_client):enabled 时由 ensure_env 把 settings 值写入
os.environ(进程内,不回写 .env)。trace 属性经 config metadata 键传入
(langfuse_session_id/langfuse_user_id/langfuse_trace_name/langfuse_tags),
resume 续传经 CallbackHandler(trace_context={"trace_id": ...})。
"""

import logging
import os
import sys
import uuid
from contextlib import contextmanager
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


def tracing_enabled(settings: Settings) -> bool:
    return settings.langfuse_enabled and settings.has_langfuse_key()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def ensure_env(settings: Settings) -> None:
    """把 settings 映射进 SDK 读的环境变量(幂等,不覆盖已存在的外部设置)。

    随后必须 get_client() 注册默认实例:langfuse 3.15.0 的
    get_client(public_key=...) 只在 registry 里查已初始化实例,查不到时
    静默返回 tracing 关闭的假实例(实测),handler 会整个变成 no-op。"""
    if not tracing_enabled(settings):
        return
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key.strip())
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key.strip())
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_host.rstrip("/"))
    from langfuse import get_client
    get_client()


def build_trace_config(settings: Settings, *, conversation_id: str, user_id: str,
                       trace_id: str, intent: str | None = None,
                       intent_confidence: float | None = None) -> dict:
    """每次图调用注入一次的请求级回调配置;未启用 → {}。"""
    if not tracing_enabled(settings):
        return {}
    ensure_env(settings)
    from langfuse.langchain import CallbackHandler  # 延迟 import:disabled 零开销
    handler = CallbackHandler(public_key=settings.langfuse_public_key.strip(),
                              trace_context={"trace_id": trace_id})
    metadata: dict[str, Any] = {
        "langfuse_session_id": str(conversation_id),
        "langfuse_user_id": user_id,
        "langfuse_trace_name": "chat_turn",
        "conversation_id": str(conversation_id),
        "turn_id": trace_id,
    }
    tags = ["chat"]
    if intent is not None:
        metadata["intent"] = intent
        if intent_confidence is not None:
            metadata["intent_confidence"] = intent_confidence
        tags.append(f"intent:{intent}")
    return {"callbacks": [handler], "metadata": metadata, "tags": tags}


def tag_intent(intent: str, confidence: float | None, *,
               trace_id: str | None = None) -> None:
    """classify_intent 出结果后回写当前 trace;无 trace_id/未启用 → no-op。

    不依赖 ambient OTel context(图节点体内 update_current_trace 实测静默
    丢失):按 trace_id 以 trace_context 建一个短命 span,借其 update_trace
    落 trace 属性;span 本身同时充当 trace 树上的意图标记。

    update_current_trace 是 v3 SDK 官方的 trace 属性更新 API,签名全
    keyword-only:name/user_id/session_id/version/input/output/metadata/tags/
    public;v4 起才改用 propagate_attributes,钉 3.15.0 不受影响。依据:
    https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/sdk/upgrade-path/python-v3-to-v4.mdx
    (另经 .venv langfuse/_client/client.py:1644 源码复核)。"""
    if not trace_id:
        return
    try:
        md: dict[str, Any] = {"intent": intent}
        if confidence is not None:
            md["intent_confidence"] = confidence
        span = _get_client().start_observation(
            trace_context={"trace_id": trace_id}, name="intent_tag",
            as_type="span")
        try:
            span.update_trace(name="chat_turn", tags=[f"intent:{intent}"],
                              metadata=md)
        finally:
            span.end()
    except Exception:
        logger.debug("langfuse tag_intent skipped", exc_info=True)


def flush_langfuse() -> None:
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        logger.debug("langfuse flush skipped", exc_info=True)


def _get_client():
    """模块内薄封装:测试 monkeypatch 此属性即可替换 SDK client。"""
    from langfuse import get_client
    return get_client()


def _env_tracing_ready() -> bool:
    """settings 缺省时的兜底判定:SDK get_client 读的同源环境变量是否配齐。"""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) \
        and bool(os.environ.get("LANGFUSE_SECRET_KEY"))


@contextmanager
def observation(as_type: str, name: str, input: dict | None = None,
                settings: Settings | None = None):
    """固定边界 span(spec §5.1):disabled/异常一律 yield None,绝不拦主路。

    三处边界(retriever/executor/mcp gateway)总是显式传自己持有的 settings;
    缺省时按进程环境判 enabled。异常只允许发生在首个 yield 前(client 创建/
    进入失败 → yield None);调用方 with 体的异常原样穿透并把 span 标错,
    关闭 span 自身的异常只记 debug——所有路径下主路语义不变。调用方对 yield
    出的对象判 None 后再 .update(output=..., metadata=...)。"""
    if settings is None:
        if not _env_tracing_ready():
            yield None
            return
    elif not tracing_enabled(settings):
        yield None
        return
    try:
        if settings is not None:
            ensure_env(settings)   # 幂等:密钥入 env + 预热默认 client
        cm = _get_client().start_as_current_observation(
            as_type=as_type, name=name, input=input)
        obs = cm.__enter__()
    except Exception:
        logger.debug("langfuse observation %s skipped", name, exc_info=True)
        yield None
        return
    try:
        yield obs
    except BaseException:
        try:
            cm.__exit__(*sys.exc_info())
        except Exception:
            logger.debug("langfuse observation %s close failed", name, exc_info=True)
        raise
    try:
        cm.__exit__(None, None, None)
    except Exception:
        logger.debug("langfuse observation %s close failed", name, exc_info=True)
