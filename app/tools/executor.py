"""ch08 统一执行引擎(spec §2):所有工具调用(内置/MCP/写)同走一处。
六步管线:查目录 → JSON Schema 校验 → 权限闸 → 归属把门 → 分发(超时/重试)→ 格式化。
写操作只由 execute_confirmed 分发(用户确认后),不自动重试;超时按「已发未必未成」。"""
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError, OperationalError

from app.knowledge.retriever import RetryableKnowledgeError
from app.services.orders import get_order
from app.services.token_budget import truncate_tool_result
from app.services.tool_audit import (
    AUDIT_DENIED, AUDIT_FAILURE, AUDIT_INVALID, AUDIT_SUCCESS, AUDIT_TIMEOUT,
    AuditEntry, write_audit,
)
from app.tools.catalog import (
    ToolSpec, TurnContext, WriteMeta, canonical_args_sha, idempotency_key_for,
)

logger = logging.getLogger("wayhelp.graph")

RETRYABLE = (TimeoutError, asyncio.TimeoutError, OperationalError, ConnectionError,
             RetryableKnowledgeError)

# 写超时后后台仍可能落地:进程级强引用集合,防 GC 回收任务;main.py lifespan 收尾 drain
PENDING_WRITE_TASKS: set[asyncio.Task] = set()


@dataclass(frozen=True)
class PendingWrite:
    """agent_node park 的写调用快照;confirm 时只信它,不信客户端字段。"""
    tool_call_id: str
    name: str
    args: dict
    args_sha256: str


@dataclass(frozen=True)
class ToolExecutionRecord:
    name: str
    args: dict
    ok: bool
    duration_ms: int
    retry_count: int
    error_type: str | None
    error_code: str | None = None
    source: str = "builtin"
    mcp_server: str | None = None
    audit_status: str = AUDIT_SUCCESS


@dataclass(frozen=True)
class ToolOutcome:
    message: ToolMessage
    record: ToolExecutionRecord


def _schema_carrier_func():
    return ""


class ToolFace:
    """每轮工具面:目录子集(含 MCP 发现条目);重名拒(spec §1.1 发现重名不覆盖)。"""

    def __init__(self, specs: list[ToolSpec]):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool name in face")
        self._specs = list(specs)
        self._by_name = {s.name: s for s in self._specs}

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def as_langchain_tools(self) -> list[StructuredTool]:
        return [StructuredTool(name=s.name, description=s.description,
                               args_schema=s.args_model, func=_schema_carrier_func)
                for s in self._specs]


def batch_violation(calls: list[dict], face: ToolFace) -> str | None:
    """批次契约(spec §2):一条模型响应最多一个写调用,且必须是最后一个。"""
    writes = [i for i, c in enumerate(calls)
              if (s := face.get(c.get("name", ""))) and s.permission == "write"]
    if not writes:
        return None
    if len(writes) > 1:
        return "一条回复中最多只能发起一个写操作"
    if writes[0] != len(calls) - 1:
        return "写操作必须是最后一步调用"
    return None


class ToolExecutor:
    def __init__(self, face: ToolFace, settings, session_factory=None, mcp=None):
        self._face = face
        self._settings = settings
        self._session_factory = session_factory
        self._mcp = mcp  # McpGateway | None(Task 8)

    # ── 有效超时:覆盖 > 政策档(faq) > 默认;写另有专档 ──
    def _read_timeout(self, name: str) -> float:
        ov = self._settings.tool_timeout_overrides.get(name)
        if ov is not None:
            return float(ov)
        if name == "query_faq":
            return self._settings.knowledge_tool_timeout_seconds
        return self._settings.tool_timeout_seconds

    def _write_timeout(self, name: str) -> float:
        ov = self._settings.tool_timeout_overrides.get(name)
        if ov is not None:
            return float(ov)
        return self._settings.tool_write_timeout_seconds

    # ── 普通路径(只读;写调用在此一律拒绝)──
    async def execute(self, call: dict, ctx: TurnContext) -> ToolOutcome:
        name = call.get("name", "")
        args = call.get("args") or {}
        call_id = call.get("id") or ""
        started = time.monotonic()
        spec = self._face.get(name)
        if spec is None:
            return await self._finish(None, name, args, call_id, "unknown_tool",
                                      "UnknownTool", started, 0, ctx,
                                      error_text="调用了未注册的工具")
        try:
            spec.args_model(**args)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                               for e in exc.errors()[:3])
            return await self._finish(spec, name, args, call_id, "invalid_args",
                                      "ValidationError", started, 0, ctx,
                                      error_text=f"工具参数不合法: {detail}")
        if spec.permission in ("write", "deny"):
            return await self._finish(spec, name, args, call_id, "permission_denied",
                                      "PermissionDenied", started, 0, ctx,
                                      error_text="没有权限执行该操作")
        if spec.ownership == "order":
            if get_order(ctx.user_id, (args.get("order_id") or "")) is None:
                content = json.dumps({"error": "订单不存在或不属于当前用户"},
                                     ensure_ascii=False)
                return await self._finish(spec, name, args, call_id, None, None,
                                          started, 0, ctx, ok=True, content=content)
        timeout = self._read_timeout(name)
        retries = 0
        max_retries = self._settings.tool_max_retries
        while True:
            try:
                raw = await self._dispatch_read(spec, args, ctx, timeout)
                content = self._format(spec, raw)
                return await self._finish(spec, name, args, call_id, None, None,
                                          started, retries, ctx, ok=True, content=content)
            except RETRYABLE as exc:
                if retries >= max_retries:
                    code = "tool_unavailable"
                    text = "工具暂时不可用(已重试)"
                    return await self._finish(spec, name, args, call_id, code,
                                              type(exc).__name__, started, retries, ctx,
                                              error_text=text,
                                              last_was_timeout=isinstance(
                                                  exc, (TimeoutError, asyncio.TimeoutError)))
                retries += 1
                await asyncio.sleep(0.5 * 2 ** (retries - 1))
            except McpToolError:      # Server 回 isError(业务失败,不重试,脱敏)
                return await self._finish(spec, name, args, call_id, "mcp_error",
                                          "McpToolError", started, retries, ctx,
                                          error_text="外部服务暂时不可用")
            except Exception as exc:
                return await self._finish(spec, name, args, call_id, "tool_error",
                                          type(exc).__name__, started, retries, ctx,
                                          error_text="工具执行失败")

    async def _dispatch_read(self, spec: ToolSpec, args: dict, ctx: TurnContext,
                             timeout: float):
        if spec.source == "mcp":
            if self._mcp is None:
                raise ConnectionError("mcp gateway unavailable")
            return await asyncio.wait_for(
                self._mcp.call(spec.mcp_server, spec.name, args), timeout=timeout)
        return await asyncio.wait_for(asyncio.to_thread(spec.fn, args, ctx), timeout=timeout)

    def _format(self, spec: ToolSpec, raw) -> str:
        content = raw if isinstance(raw, str) else str(raw)
        if spec.formatter is not None:
            content = spec.formatter(content)
        return truncate_tool_result(content, self._settings.tool_result_max_tokens)

    # ── 写确认路径(用户确认后唯一写入口;confirmed 不是模型可控参数)──
    async def execute_confirmed(self, pending: PendingWrite, ctx: TurnContext) -> ToolOutcome:
        started = time.monotonic()
        spec = self._face.get(pending.name)
        if spec is None or spec.permission != "write":
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "permission_denied", "PermissionDenied", started, 0,
                                      ctx, error_text="没有权限执行该操作")
        if ctx.conversation_id is None:
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "invalid_args", "ConversationRequired", started, 0,
                                      ctx, error_text="写操作缺少会话上下文,已拦下")
        key = idempotency_key_for(ctx.conversation_id, pending.tool_call_id)
        found = await asyncio.to_thread(self._idem_lookup, key, pending.args_sha256)
        if found is not None:
            if found == "__conflict__":
                return await self._finish(spec, pending.name, pending.args,
                                          pending.tool_call_id, "idempotency_conflict",
                                          "IdempotencyConflict", started, 0, ctx,
                                          error_text="内部一致性错误,已拦下重复写入")
            content = json.dumps({"ticket_no": found, "status": "待处理",
                                  "note": "重复提交已忽略"}, ensure_ascii=False)
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      None, None, started, 0, ctx, ok=True, content=content)
        meta = WriteMeta(key=key, args_sha256=pending.args_sha256)
        task = asyncio.ensure_future(asyncio.to_thread(spec.fn, pending.args, ctx, meta))
        PENDING_WRITE_TASKS.add(task)
        task.add_done_callback(self._write_task_done)
        try:
            raw = await asyncio.wait_for(asyncio.shield(task),
                                         timeout=self._write_timeout(pending.name))
            content = self._format(spec, raw)
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      None, None, started, 0, ctx, ok=True, content=content)
        except (TimeoutError, asyncio.TimeoutError):
            # 超时≠没执行:立即返回未知;后台任务继续,审计只此一条(spec §3)
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "write_timeout", "TimeoutError", started, 0, ctx,
                                      error_text="提交超时,结果未知,请勿重复提交;"
                                                 "如需要请查询工单列表确认")
        except IntegrityError:
            found = await asyncio.to_thread(self._idem_lookup, key, pending.args_sha256)
            if found is not None and found != "__conflict__":
                content = json.dumps({"ticket_no": found, "status": "待处理",
                                      "note": "重复提交已忽略"}, ensure_ascii=False)
                return await self._finish(spec, pending.name, pending.args,
                                          pending.tool_call_id, None, None, started, 0, ctx,
                                          ok=True, content=content)
            return await self._finish(spec, pending.name, pending.args,
                                      pending.tool_call_id, "idempotency_conflict",
                                      "IdempotencyConflict", started, 0, ctx,
                                      error_text="内部一致性错误,已拦下重复写入")
        except Exception as exc:
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "tool_error", type(exc).__name__, started, 0, ctx,
                                      error_text="工具执行失败")

    def _idem_lookup(self, key: str, args_sha256: str) -> str | None:
        """返回已建 ticket_no;key 存在但参数摘要不符 → '__conflict__';不存在 → None。
        '__conflict__' 只是标记,任何路径不得把它当 ticket_no 透出。"""
        from app.models import ToolWriteIdempotency
        if self._session_factory is None:
            return None
        with self._session_factory() as s:
            row = s.get(ToolWriteIdempotency, key)
            if row is None:
                return None
            return row.ticket_no if row.arguments_sha256 == args_sha256 else "__conflict__"

    def _write_task_done(self, task: asyncio.Task) -> None:
        PENDING_WRITE_TASKS.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("后台写任务异常(已消费,不再追加审计): %s", type(exc).__name__)
        else:
            logger.info("后台写任务完成(超时返回后落地;不追加第二条审计)")

    # ── 取消写(用户点「取消」):审计权限拒绝,arguments 记原始申请 ──
    async def deny_write(self, pending: PendingWrite, ctx: TurnContext,
                         reason: str) -> ToolOutcome:
        started = time.monotonic()
        spec = self._face.get(pending.name)
        return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                  "permission_denied", "UserCancelled", started, 0, ctx,
                                  error_text=f"用户已取消,本次未执行({reason})")

    # ── 非法批次:零执行,每个 call 一条校验拦下审计 + 错误 ToolMessage 回灌 ──
    async def mark_invalid_batch(self, calls: list[dict], ctx: TurnContext,
                                 reason: str) -> list[ToolMessage]:
        msgs = []
        for call in calls:
            spec = self._face.get(call.get("name", ""))
            out = await self._finish(spec, call.get("name", ""), call.get("args") or {},
                                     call.get("id") or "", "invalid_batch", "InvalidBatch",
                                     time.monotonic(), 0, ctx,
                                     error_text=f"工具调用批次不合法: {reason}")
            msgs.append(out.message)
        return msgs

    # ── 终结:统一产出 ToolOutcome + 落审计(失败不拦执行)──
    async def _finish(self, spec, name, args, call_id, error_code, error_type,
                      started, retries, ctx, *, ok=False, content=None,
                      error_text=None, last_was_timeout=False) -> ToolOutcome:
        duration = int((time.monotonic() - started) * 1000)
        if ok:
            audit_status = AUDIT_SUCCESS
            msg = ToolMessage(content=truncate_tool_result(
                content, self._settings.tool_result_max_tokens),
                tool_call_id=call_id, name=name or "unknown", status="success")
        else:
            if error_code in ("invalid_args", "invalid_batch") \
                    or error_type == "ConversationRequired":
                audit_status = AUDIT_INVALID
            elif error_code == "permission_denied":
                audit_status = AUDIT_DENIED
            elif error_code == "write_timeout" or last_was_timeout:
                audit_status = AUDIT_TIMEOUT
            else:
                audit_status = AUDIT_FAILURE
            msg = ToolMessage(content=error_text or "工具执行失败",
                              tool_call_id=call_id, name=name or "unknown", status="error")
        record = ToolExecutionRecord(
            name, args, ok, duration, retries, error_type, error_code,
            spec.source if spec else "builtin",
            spec.mcp_server if spec else None, audit_status)
        await write_audit(self._session_factory, AuditEntry(
            conversation_id=ctx.conversation_id, tool_call_id=call_id or None,
            tool_name=name or "unknown", tool_source=record.source,
            mcp_server=record.mcp_server, arguments=args or None,
            result_summary=msg.content if ok or error_code in (
                "write_timeout",) else None,
            status=audit_status,
            error_message=None if ok else (error_text or "")[:500],
            retry_count=retries, duration_ms=duration),
            self._settings.audit_result_max_chars)
        return ToolOutcome(msg, record)


class McpToolError(Exception):
    """gateway 抛出:Server 回 CallToolResult.isError=true(业务失败,不重试,脱敏)。"""
