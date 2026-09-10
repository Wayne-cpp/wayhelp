import asyncio
import time
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from app.tool_envelope import truncate_content

RETRYABLE = (TimeoutError, asyncio.TimeoutError, OperationalError, ConnectionError)
WRITE_TOOLS = {"create_ticket"}


@dataclass(frozen=True)
class ToolExecutionRecord:
    name: str
    args: dict
    ok: bool
    duration_ms: int
    retry_count: int
    error_type: str | None
    error_code: str | None = None  # 机器码(unknown_tool 等);error_type 是异常类名


@dataclass(frozen=True)
class ToolOutcome:
    message: ToolMessage
    record: ToolExecutionRecord


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool name")
        self._tools = list(tools)
        self._by_name = dict(zip(names, self._tools))

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def get(self, name: str) -> BaseTool | None:
        return self._by_name.get(name)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds: float, max_retries: int,
                 max_result_chars: int, write_tools: set[str] | None = None):
        self._registry = registry
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_chars = max_result_chars
        self._write_tools = WRITE_TOOLS if write_tools is None else write_tools

    async def execute(self, call: dict) -> ToolOutcome:
        name = call.get("name", "")
        args = call.get("args") or {}
        call_id = call.get("id") or ""
        started = time.monotonic()
        tool = self._registry.get(name)
        if tool is None:
            return self._error(name, args, call_id, "unknown_tool", "UnknownTool", started)
        try:
            tool.args_schema(**args)  # 参数校验先行,失败不重试
        except ValidationError:
            return self._error(name, args, call_id, "invalid_args", "ValidationError", started)
        if name in self._write_tools:
            return await self._run_write(tool, call, started)
        return await self._run_readonly(tool, call, started)

    async def _run_readonly(self, tool: BaseTool, call: dict, started: float) -> ToolOutcome:
        retries = 0
        while True:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, call), timeout=self._timeout
                )
                return self._success(tool, call, result, retries, started)
            except RETRYABLE as exc:
                if retries >= self._max_retries:
                    return self._error(call.get("name", ""), call.get("args") or {},
                                       call.get("id") or "", "tool_unavailable",
                                       type(exc).__name__, started, retries)
                retries += 1
                await asyncio.sleep(0.5 * 2 ** (retries - 1))
            except Exception as exc:  # 业务异常不重试
                return self._error(call.get("name", ""), call.get("args") or {},
                                   call.get("id") or "", "tool_error",
                                   type(exc).__name__, started, retries)

    async def _run_write(self, tool: BaseTool, call: dict, started: float) -> ToolOutcome:
        try:
            result = await asyncio.to_thread(tool.invoke, call)  # 不 wait_for,不重试
            return self._success(tool, call, result, 0, started)
        except Exception as exc:
            return self._error(call.get("name", ""), call.get("args") or {},
                               call.get("id") or "", "tool_error",
                               type(exc).__name__, started, 0)

    def _success(self, tool, call, result, retries, started) -> ToolOutcome:
        content = result.content if isinstance(result, ToolMessage) else str(result)
        content = truncate_content(content, True, None, self._max_chars)
        msg = ToolMessage(content=content, tool_call_id=call.get("id") or "",
                          name=tool.name, status="success")
        return ToolOutcome(msg, ToolExecutionRecord(
            tool.name, call.get("args") or {}, True,
            int((time.monotonic() - started) * 1000), retries, None, None))

    _ERROR_TEXT = {
        "unknown_tool": "调用了未注册的工具",
        "invalid_args": "工具参数不合法",
        "tool_unavailable": "工具暂时不可用(已重试)",
        "tool_error": "工具执行失败",
    }

    def _error(self, name, args, call_id, code, error_type, started, retries=0) -> ToolOutcome:
        text = truncate_content(self._ERROR_TEXT[code], False, code, self._max_chars)
        msg = ToolMessage(content=text, tool_call_id=call_id, name=name or "unknown",
                          status="error")
        return ToolOutcome(msg, ToolExecutionRecord(
            name, args, False, int((time.monotonic() - started) * 1000), retries, error_type,
            code))
