import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Union

from langchain_core.messages import BaseMessage

from app.chains.chat_chain import build_chat_messages, check_input_budget
from app.config import Settings
from app.errors import MessageTooLongError, SessionNotFoundError
from app.sessions import SessionStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionEvent:
    session_id: str


@dataclass(frozen=True)
class DeltaEvent:
    content: str


@dataclass(frozen=True)
class DoneEvent:
    pass


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str


ChatEvent = Union[SessionEvent, DeltaEvent, DoneEvent, ErrorEvent]


@dataclass
class PreparedTurn:
    session_id: str
    user_text: str
    messages: list[BaseMessage]
    lock: asyncio.Lock
    released: bool = False


class ChatService:
    def __init__(self, store: SessionStore, model: Any, settings: Settings, system_prompt: str):
        self._store = store
        self._model = model
        self._settings = settings
        self._system_prompt = system_prompt
        self._locks: dict[str, asyncio.Lock] = {}

    async def prepare(self, session_id: uuid.UUID | None, message: str) -> PreparedTurn:
        if len(message) > self._settings.max_message_chars:
            raise MessageTooLongError("message exceeds MAX_MESSAGE_CHARS")
        check_input_budget(self._system_prompt, message, self._settings.max_input_tokens)
        if session_id is None:
            sid = self._store.create()
            self._locks[sid] = asyncio.Lock()
        else:
            sid = str(session_id)
            if not self._store.exists(sid):
                raise SessionNotFoundError("session not found")
        lock = self._locks[sid]
        await lock.acquire()
        try:
            history = self._store.snapshot(sid)
            messages = build_chat_messages(
                self._system_prompt, history, message, self._settings.max_input_tokens
            )
        except Exception:
            lock.release()
            raise
        return PreparedTurn(sid, message, messages, lock)

    def release_turn(self, turn: PreparedTurn) -> None:
        if turn.released:
            return
        turn.released = True
        turn.lock.release()

    async def stream(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        agen = self._model.astream(turn.messages)
        try:
            yield SessionEvent(turn.session_id)
            parts: list[str] = []
            total_chars = 0
            finish_reason: str | None = None
            try:
                async for chunk in agen:
                    meta = getattr(chunk, "response_metadata", None) or {}
                    if meta.get("finish_reason"):
                        finish_reason = meta["finish_reason"]
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if not text:
                        continue
                    total_chars += len(text)
                    if total_chars > self._settings.max_message_chars:
                        yield ErrorEvent("output_too_long", "回复超出长度限制")
                        return
                    parts.append(text)
                    yield DeltaEvent(text)
            except Exception as exc:
                logger.warning("chat stream upstream error: %s", type(exc).__name__)
                yield ErrorEvent("upstream_error", "上游模型暂时不可用")
                return
            full = "".join(parts)
            if finish_reason == "length":
                yield ErrorEvent("output_too_long", "回复超出长度限制")
                return
            if not full.strip():
                yield ErrorEvent("empty_response", "上游返回了空回复")
                return
            self._store.commit_turn(turn.session_id, turn.user_text, full)
            yield DoneEvent()
        finally:
            with contextlib.suppress(Exception):
                await agen.aclose()
            self.release_turn(turn)
