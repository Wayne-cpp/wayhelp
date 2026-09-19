import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Union

from langchain_core.messages import AIMessageChunk

from app.chains.chat_chain import check_input_budget
from app.config import Settings
from app.errors import (
    AppError, MessageTooLongError, ResumeConflictError, SessionNotFoundError,
)
from app.graph.errors import TurnAbortError
from app.graph.state import new_turn_state
from app.prompts.service import FALLBACK_ANSWER
from app.sessions import SessionStore

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


@dataclass(frozen=True)
class CitationsEvent:
    citations: list[dict]


@dataclass(frozen=True)
class SuggestActionsEvent:
    source_message_id: str
    options: list[dict]


@dataclass(frozen=True)
class OrderSelectorEvent:
    """ch06(spec §8):图挂起后绑定 interrupt ID 的订单选择器帧。"""
    interrupt_id: str
    orders: list[dict]


ChatEvent = Union[SessionEvent, DeltaEvent, ToolStartEvent, ToolEndEvent, DoneEvent,
                  ErrorEvent, CitationsEvent, SuggestActionsEvent, OrderSelectorEvent]


class SessionLockRegistry:
    """懒创建 session 锁;持有+等待计数归零后删除条目。"""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    async def acquire(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        self._users[session_id] = self._users.get(session_id, 0) + 1
        try:
            await lock.acquire()
        except asyncio.CancelledError:
            # 等待中被取消:未持有锁,只回滚计数(不得 release 未获得的锁)
            self._retire(session_id)
            raise
        return lock

    def release(self, session_id: str) -> None:
        lock = self._locks[session_id]
        lock.release()
        self._retire(session_id)

    def _retire(self, session_id: str) -> None:
        """计数减一;归零即删条目(懒清理)。"""
        self._users[session_id] -= 1
        if self._users[session_id] == 0:
            del self._users[session_id]
            del self._locks[session_id]


@dataclass
class PreparedTurn:
    session_id: str
    user_text: str
    lock_key: str
    user_id: str = ""
    resume_command: Any = None   # langgraph Command;resume 路径传 Command 不传 dict
    released: bool = False


class ChatService:
    def __init__(self, store: SessionStore, model: Any, settings: Settings,
                 system_prompt: str, session_factory=None, graph=None):
        self._store = store
        self._model = model
        self._settings = settings
        self._system_prompt = system_prompt
        self._session_factory = session_factory
        self._graph = graph
        self._locks = SessionLockRegistry()

    def set_graph(self, graph) -> None:
        self._graph = graph

    async def prepare(self, user_id: str, session_id: str | None, message: str) -> PreparedTurn:
        if len(message) > self._settings.max_message_chars:
            raise MessageTooLongError("message exceeds MAX_MESSAGE_CHARS")
        check_input_budget(self._system_prompt, message, self._settings.max_input_tokens)
        if session_id is None:
            sid = await self._store.create(user_id)
        else:
            sid = session_id
            if not await self._store.exists(sid, user_id):
                raise SessionNotFoundError("session not found")
        await self._locks.acquire(sid)
        return PreparedTurn(sid, message, sid, user_id=user_id)

    async def prepare_resume(self, user_id: str, session_id: str,
                             interrupt_id: str, order_id: str) -> PreparedTurn:
        """ch06(spec §9.1):锁内核验 pending interrupt/候选/订单归属,失败释锁。
        409 必须服务端自校验——LangGraph 对不匹配 resume ID 是静默忽略,不报错。"""
        from langgraph.types import Command
        from app.services.orders import get_order
        if not await self._store.exists(session_id, user_id):
            raise SessionNotFoundError("session not found")
        await self._locks.acquire(session_id)
        try:
            config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
            st = await self._graph.aget_state(config)
            match = None
            for task in st.tasks:
                for intr in task.interrupts:
                    if intr.id == interrupt_id:
                        match = intr
            value = getattr(match, "value", None) or {}
            if match is None or value.get("type") != "order_selector":
                raise ResumeConflictError("no matching pending order selection")
            candidates = {o.get("order_id") for o in value.get("orders") or []}
            if order_id not in candidates:
                raise ResumeConflictError("order not in pending candidates")
            if get_order(user_id, order_id) is None:
                raise SessionNotFoundError("session not found")  # 404 不泄露
        except Exception:
            self._locks.release(session_id)
            raise
        return PreparedTurn(session_id=session_id, user_text="", lock_key=session_id,
                            user_id=user_id,
                            resume_command=Command(resume={interrupt_id: {"order_id": order_id}}))

    def release_turn(self, turn: PreparedTurn) -> None:
        if turn.released:
            return
        turn.released = True
        self._locks.release(turn.lock_key)

    async def stream(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        try:
            if self._graph is None:
                yield ErrorEvent("internal_error", "服务未就绪")
                return
            yield SessionEvent(turn.session_id)
            config = {"configurable": {"thread_id": turn.session_id,
                                       "user_id": turn.user_id}}
            graph_input = (turn.resume_command if turn.resume_command is not None
                           else new_turn_state(turn.user_text))
            try:
                async for mode, payload in self._graph.astream(
                        graph_input, config,
                        stream_mode=["messages", "custom"]):
                    if mode == "messages":
                        chunk, meta = payload
                        meta = meta or {}
                        if meta.get("langgraph_node") != "main_agent":
                            continue  # 分类/检索内部调用的 token 不外发
                        if "chat_visible" not in (meta.get("tags") or []):
                            continue  # 节点内部分类/改写等调用不外发(ch06 spec §8)
                        if not isinstance(chunk, AIMessageChunk):
                            # 裁决护栏:langgraph 1.2.x messages 模式还会发节点级
                            # HumanMessage 回显与聚合终帧 AIMessage(同带节点元数据),
                            # 均非模型 token,一律不外发
                            continue
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if text:
                            yield DeltaEvent(text)
                    else:
                        event = self._translate_custom(payload)
                        if event is not None:
                            yield event
            except TurnAbortError:
                return  # error 帧已由节点经 writer 发出;不提交、不发 DONE
            except Exception:
                logger.exception("chat graph error")
                yield ErrorEvent("internal_error", "服务内部错误")
                return
            async for event in self._pending_selector_events(config):
                yield event
            yield DoneEvent()
        finally:
            self.release_turn(turn)

    async def _pending_selector_events(self, config) -> AsyncIterator[ChatEvent]:
        """图挂起(仍在会话锁内):读 pending interrupt,发绑定其 ID 的订单选择器(spec §8)。"""
        st = await self._graph.aget_state(config)
        for task in st.tasks:
            for intr in task.interrupts:
                value = getattr(intr, "value", None) or {}
                if value.get("type") == "order_selector":
                    yield OrderSelectorEvent(interrupt_id=intr.id,
                                             orders=value.get("orders") or [])
                    return

    @staticmethod
    def _translate_custom(payload) -> ChatEvent | None:
        kind = payload.get("kind")
        if kind == "fixed_delta":
            return DeltaEvent(payload["content"])
        if kind == "tool_start":
            return ToolStartEvent(payload["tool_call_id"], payload["name"], payload["args"])
        if kind == "tool_end":
            return ToolEndEvent(payload["tool_call_id"], payload["name"],
                                payload["ok"], payload["summary"])
        if kind == "citations":
            return CitationsEvent(payload["citations"])
        if kind == "suggest_actions":
            return SuggestActionsEvent(payload["source_message_id"], payload["options"])
        if kind == "error":
            return ErrorEvent(payload["code"], payload["message"])
        return None

    async def create_ticket_from_action(self, user_id: str, session_id: str,
                                        source_message_id: str, ticket_type: str) -> str:
        if self._session_factory is None:
            raise AppError("action_unavailable")
        task = asyncio.ensure_future(asyncio.to_thread(
            self._create_ticket_sync, user_id, session_id, source_message_id, ticket_type))
        try:
            return await asyncio.shield(task)  # 写操作:取消时等事务落地再放行
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await task
            raise

    def _create_ticket_sync(self, user_id, session_id, source_message_id, ticket_type) -> str:
        from app.models import Conversation, Message
        from app.store_db import _as_db_id
        from app.tools.business import write_ticket
        cid = _as_db_id(session_id)
        mid = _as_db_id(source_message_id)
        if cid is None or mid is None:
            raise SessionNotFoundError("session not found")
        with self._session_factory() as s:  # 同一 Session 完成归属/消息查询与写入
            conv = s.get(Conversation, cid)
            if conv is None or conv.user_id != user_id:
                raise SessionNotFoundError("session not found")
            msg = s.get(Message, mid)
            if msg is None or msg.conversation_id != cid or msg.role != "user":
                raise SessionNotFoundError("session not found")  # 404 不泄露其他会话内容
            ticket_no = write_ticket(s, cid, msg.content or "用户通过快捷操作请求建单",
                                     ticket_type)
            s.commit()  # 失败整体回滚;不改 conv.status
            return ticket_no
