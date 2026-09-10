import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Union

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.chains.tool_chat_chain import (
    build_messages_with_tools,
    finalize_tool_calls,
    fit_tool_context,
    merge_tool_call_chunks,
)
from app.config import Settings
from app.errors import MessageTooLongError, SessionNotFoundError
from app.sessions import SessionStore, StoredMessage
from app.tool_envelope import wrap
from app.tools.executor import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionEvent:
    session_id: str


@dataclass(frozen=True)
class DeltaEvent:
    content: str


@dataclass(frozen=True)
class ToolStartEvent:
    tool_call_id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolEndEvent:
    tool_call_id: str
    name: str
    ok: bool
    summary: str


@dataclass(frozen=True)
class DoneEvent:
    pass


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str


ChatEvent = Union[SessionEvent, DeltaEvent, ToolStartEvent, ToolEndEvent, DoneEvent, ErrorEvent]


class SessionLockRegistry:
    """懒创建 session 锁;持有+等待计数归零后删除条目。"""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    async def acquire(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        self._users[session_id] = self._users.get(session_id, 0) + 1
        await lock.acquire()
        return lock

    def release(self, session_id: str) -> None:
        lock = self._locks[session_id]
        lock.release()
        self._users[session_id] -= 1
        if self._users[session_id] == 0:
            del self._users[session_id]
            del self._locks[session_id]


@dataclass
class PreparedTurn:
    session_id: str
    user_text: str
    messages: list[BaseMessage]
    lock_key: str
    released: bool = False


class ChatService:
    def __init__(self, store: SessionStore, model: Any, settings: Settings,
                 system_prompt: str, toolset_factory: Callable[[str], list] | None = None):
        self._store = store
        self._model = model
        self._settings = settings
        self._system_prompt = system_prompt
        self._toolset_factory = toolset_factory
        self._locks = SessionLockRegistry()

    async def prepare(self, user_id: str, session_id: str | None, message: str) -> PreparedTurn:
        if len(message) > self._settings.max_message_chars:
            raise MessageTooLongError("message exceeds MAX_MESSAGE_CHARS")
        if session_id is None:
            sid = await self._store.create(user_id)
        else:
            sid = session_id
            if not await self._store.exists(sid, user_id):
                raise SessionNotFoundError("session not found")
        await self._locks.acquire(sid)
        try:
            history = await self._store.snapshot(sid)
            messages = build_messages_with_tools(
                self._system_prompt, history, message, self._settings.max_input_tokens
            )
        except Exception:
            self._locks.release(sid)
            raise
        return PreparedTurn(sid, message, messages, sid)

    def release_turn(self, turn: PreparedTurn) -> None:
        if turn.released:
            return
        turn.released = True
        self._locks.release(turn.lock_key)

    async def stream(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        tools = self._toolset_factory(turn.session_id) if self._toolset_factory else []
        registry = ToolRegistry(tools)
        executor = ToolExecutor(registry, self._settings.tool_timeout_seconds,
                                self._settings.tool_max_retries,
                                self._settings.max_tool_result_chars)
        first_model = self._model.bind_tools(registry.tools) if tools else self._model
        agen = first_model.astream(turn.messages)
        visible_chars = 0
        committed = False
        try:
            yield SessionEvent(turn.session_id)
            text_parts: list[str] = []
            acc: dict[int, dict] = {}
            finish_reason: str | None = None
            try:
                async for chunk in agen:
                    meta = getattr(chunk, "response_metadata", None) or {}
                    if meta.get("finish_reason"):
                        finish_reason = meta["finish_reason"]
                    chunks = getattr(chunk, "tool_call_chunks", None) or []
                    if chunks:
                        merge_tool_call_chunks(acc, chunks)
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if not text:
                        continue
                    visible_chars += len(text)
                    if visible_chars > self._settings.max_message_chars:
                        yield ErrorEvent("output_too_long", "回复超出长度限制")
                        return
                    text_parts.append(text)
                    yield DeltaEvent(text)
            except Exception as exc:
                logger.warning("chat first call upstream error: %s", type(exc).__name__)
                yield ErrorEvent("upstream_error", "上游模型暂时不可用")
                return
            finally:
                with contextlib.suppress(Exception):
                    await agen.aclose()

            if finish_reason == "length":
                yield ErrorEvent("output_too_long", "回复超出长度限制")
                return

            calls, broken = finalize_tool_calls(acc)
            if broken or not self._tool_calls_legal(calls):
                yield ErrorEvent("invalid_tool_call", "工具调用申请不合法")
                return

            tool_messages: list[ToolMessage] = []
            ai_with_calls: AIMessage | None = None
            if calls:
                ai_with_calls = AIMessage(content="".join(text_parts), tool_calls=calls)
                for call in calls:
                    yield ToolStartEvent(call["id"], call["name"], call["args"])
                    outcome = await executor.execute(call)
                    tool_messages.append(outcome.message)
                    logger.info("tool %s ok=%s retries=%d ms=%d err=%s",
                                outcome.record.name, outcome.record.ok,
                                outcome.record.retry_count, outcome.record.duration_ms,
                                outcome.record.error_type)
                    yield ToolEndEvent(call["id"], call["name"], outcome.record.ok,
                                       outcome.message.content[:80])
                second_messages = fit_tool_context(
                    [*turn.messages, ai_with_calls, *tool_messages],
                    self._settings.max_input_tokens,
                    protected_from=len(turn.messages),
                )
                if second_messages is None:
                    yield ErrorEvent("tool_context_too_long", "工具结果超出上下文预算")
                    return
                final_parts: list[str] = []
                finish2: str | None = None
                agen2 = self._model.astream(second_messages)  # 不绑工具,单轮收敛
                try:
                    async for chunk in agen2:
                        meta = getattr(chunk, "response_metadata", None) or {}
                        if meta.get("finish_reason"):
                            finish2 = meta["finish_reason"]
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if not text:
                            continue
                        visible_chars += len(text)
                        if visible_chars > self._settings.max_message_chars:
                            yield ErrorEvent("output_too_long", "回复超出长度限制")
                            return
                        final_parts.append(text)
                        yield DeltaEvent(text)
                except Exception as exc:
                    logger.warning("chat second call upstream error: %s", type(exc).__name__)
                    yield ErrorEvent("upstream_error", "上游模型暂时不可用")
                    return
                finally:
                    with contextlib.suppress(Exception):
                        await agen2.aclose()
                if finish2 == "length":
                    yield ErrorEvent("output_too_long", "回复超出长度限制")
                    return
                final_text = "".join(final_parts)
            else:
                final_text = "".join(text_parts)

            if not final_text.strip():
                yield ErrorEvent("empty_response", "上游返回了空回复")
                return

            stored = [StoredMessage("user", turn.user_text)]
            if calls:
                stored.append(StoredMessage("assistant", "".join(text_parts) or None,
                                            tool_calls=calls))
                for tm in tool_messages:
                    stored.append(StoredMessage(
                        "tool",
                        wrap(tm.content, tm.status != "error",
                             None if tm.status != "error" else "tool_error",
                             self._settings.max_tool_result_chars),
                        tool_call_id=tm.tool_call_id,
                    ))
            stored.append(StoredMessage("assistant", final_text))
            commit_task = asyncio.ensure_future(self._store.commit_turn(turn.session_id, stored))
            try:
                await asyncio.shield(commit_task)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await commit_task  # 等事务落地
                raise
            committed = True
            yield DoneEvent()
        finally:
            self.release_turn(turn)

    def _tool_calls_legal(self, calls: list[dict]) -> bool:
        if len(calls) > self._settings.max_tool_calls_per_turn:
            return False
        ids = [c["id"] for c in calls]
        if len(set(ids)) != len(ids):
            return False
        if any(not c["id"] or len(c["id"]) > 64 for c in calls):
            return False
        if sum(1 for c in calls if c["name"] == "create_ticket") > 1:
            return False
        return True
