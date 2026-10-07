"""ch08 工具调用审计(spec §3):每次调用终结落一条;写失败不拦工具执行(红线)。"""
import asyncio
import contextlib
import logging
from dataclasses import dataclass

logger = logging.getLogger("wayhelp.tool_audit")

AUDIT_SUCCESS = "成功"
AUDIT_FAILURE = "失败"
AUDIT_TIMEOUT = "超时"
AUDIT_INVALID = "校验拦下"
AUDIT_DENIED = "权限拒绝"

_TRUNC_MARK = "…[已截断]"


@dataclass(frozen=True)
class AuditEntry:
    conversation_id: int | None
    tool_call_id: str | None
    tool_name: str
    tool_source: str          # "builtin" | "mcp"
    mcp_server: str | None
    arguments: dict | None
    result_summary: str | None
    status: str
    error_message: str | None
    retry_count: int
    duration_ms: int | None


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None or len(text) <= max_chars:
        return text
    return text[: max_chars - len(_TRUNC_MARK)] + _TRUNC_MARK


def _write_sync(session_factory, entry: AuditEntry, max_chars: int) -> None:
    from app.models import ToolAuditLog
    with session_factory() as s:
        s.add(ToolAuditLog(
            conversation_id=entry.conversation_id, tool_call_id=entry.tool_call_id,
            tool_name=entry.tool_name, tool_source=entry.tool_source,
            mcp_server=entry.mcp_server, arguments=entry.arguments,
            result_summary=_truncate(entry.result_summary, max_chars),
            status=entry.status, error_message=entry.error_message,
            retry_count=entry.retry_count, duration_ms=entry.duration_ms))
        s.commit()


async def write_audit(session_factory, entry: AuditEntry, max_chars: int = 2000) -> None:
    """shield 保护审计写;任何失败只打 error 日志——审计不得反过来拦工具执行。"""
    if session_factory is None:
        logger.error("tool_audit 无 session_factory,丢弃审计: %s %s", entry.tool_name, entry.status)
        return
    task = asyncio.ensure_future(asyncio.to_thread(_write_sync, session_factory, entry, max_chars))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise
    except Exception as exc:
        logger.error("tool_audit 写失败(不影响工具执行): %s", type(exc).__name__)
