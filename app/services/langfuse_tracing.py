"""ch09:Langfuse(SDK v3)接入。未启用时全部 no-op;任何异常只记日志不拦主路。

SDK 从环境变量读密钥(get_client):enabled 时由 ensure_env 把 settings 值写入
os.environ(进程内,不回写 .env)。trace 属性经 config metadata 键传入
(langfuse_session_id/langfuse_user_id/langfuse_trace_name/langfuse_tags),
resume 续传经 CallbackHandler(trace_context={"trace_id": ...})。
"""

import logging
import os
import uuid
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


def tag_intent(intent: str, confidence: float | None) -> None:
    """classify_intent 出结果后回写当前 trace;无活动 trace/未启用 → no-op。

    update_current_trace 是 v3 SDK 官方的 trace 属性更新 API,签名全
    keyword-only:name/user_id/session_id/version/input/output/metadata/tags/
    public;v4 起才改用 propagate_attributes,钉 3.15.0 不受影响。依据:
    https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/sdk/upgrade-path/python-v3-to-v4.mdx
    (另经 .venv langfuse/_client/client.py:1644 源码复核)。"""
    try:
        from langfuse import get_client
        client = get_client()
        md: dict[str, Any] = {"intent": intent}
        if confidence is not None:
            md["intent_confidence"] = confidence
        client.update_current_trace(tags=[f"intent:{intent}"], metadata=md)
    except Exception:
        logger.debug("langfuse tag_intent skipped", exc_info=True)


def flush_langfuse() -> None:
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        logger.debug("langfuse flush skipped", exc_info=True)
