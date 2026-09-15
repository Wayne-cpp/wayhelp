import asyncio
import contextlib
import json
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
from app.prompts.service import REFUSAL_ANSWER
from app.sessions import LowConfidenceRecord, SessionStore, StoredMessage
from app.tool_envelope import wrap
from app.tools.executor import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)

# 第二次调用若模型只产出 tool_calls(本轮一律不执行)而无任何文本,以该话术兜底作答
FALLBACK_ANSWER = "抱歉,暂时没有查到相关信息。您可以换个说法问我,或回复「转人工」,让人工客服帮您处理。"


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


ChatEvent = Union[SessionEvent, DeltaEvent, ToolStartEvent, ToolEndEvent, DoneEvent,
                  ErrorEvent, CitationsEvent]


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
        visible_chars = 0
        committed = False
        try:
            # 工具装配须在 finally 覆盖内:factory/registry/bind_tools/astream
            # 任一抛异常都经 finally 释放锁,不得永久持锁
            ts = self._toolset_factory(turn.session_id) if self._toolset_factory else None
            if ts is None:
                tools, trace = [], None
            elif hasattr(ts, "retrieval_trace"):  # TurnToolset(T7)
                tools, trace = ts.tools, ts.retrieval_trace
            else:  # 旧装配仍直接给工具列表(test_orchestration 等)
                tools, trace = list(ts), None
            registry = ToolRegistry(tools)
            executor = ToolExecutor(registry, self._settings.tool_timeout_seconds,
                                    self._settings.tool_max_retries,
                                    self._settings.max_tool_result_chars,
                                    tool_policies={"query_faq": (
                                        self._settings.knowledge_tool_timeout_seconds, 0)})
            first_model = self._model.bind_tools(registry.tools) if tools else self._model
            agen = first_model.astream(turn.messages)
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
            tool_error_codes: list[str | None] = []
            ai_with_calls: AIMessage | None = None
            if calls:
                ai_with_calls = AIMessage(content="".join(text_parts), tool_calls=calls)
                for call in calls:
                    yield ToolStartEvent(call["id"], call["name"], call["args"])
                    outcome = await executor.execute(call)
                    tool_messages.append(outcome.message)
                    tool_error_codes.append(outcome.record.error_code)
                    logger.info("tool %s ok=%s retries=%d ms=%d err=%s",
                                outcome.record.name, outcome.record.ok,
                                outcome.record.retry_count, outcome.record.duration_ms,
                                outcome.record.error_type)
                    yield ToolEndEvent(call["id"], call["name"], outcome.record.ok,
                                       outcome.message.content[:80])
                if trace is not None and trace.status == "low_confidence":
                    # 检索硬闸门:低置信命中时不发起第二次模型调用,直接固定话术拒答
                    yield DeltaEvent(REFUSAL_ANSWER)
                    final_text = REFUSAL_ANSWER
                else:
                    second_messages = fit_tool_context(
                        [*turn.messages, ai_with_calls, *tool_messages],
                        self._settings.max_input_tokens,
                        protected_from=len(turn.messages) - 1,  # 当前 human 的下标(turn.messages 末位)
                    )
                    if second_messages is None:
                        yield ErrorEvent("tool_context_too_long", "工具结果超出上下文预算")
                        return
                    final_parts: list[str] = []
                    finish2: str | None = None
                    # 第二次同样绑定工具:给工具意图结构化通道,避免其以标记语法裸文本泄漏;
                    # 但本轮不再执行任何 tool_calls(不聚合不推帧,徽章只代表真实执行)
                    second_model = self._model.bind_tools(registry.tools) if tools else self._model
                    agen2 = second_model.astream(second_messages)
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
                    if not final_text.strip():
                        # 模型只给了未执行的 tool_calls:落库须剥离它们,以兜底话术作答
                        yield DeltaEvent(FALLBACK_ANSWER)
                        final_text = FALLBACK_ANSWER
            else:
                final_text = "".join(text_parts)

            if not final_text.strip():
                yield ErrorEvent("empty_response", "上游返回了空回复")
                return

            # 拒答入池判定(与 commit 同事务):硬闸门拒答 → retrieval_low_conf;
            # 检索 ok 但模型自评后精确输出拒答话术 → self_check;tool_error 不入池
            low_conf: LowConfidenceRecord | None = None
            if trace is not None:
                cid = int(turn.session_id) if turn.session_id.isdecimal() else None
                if trace.status == "low_confidence" and final_text.strip() == REFUSAL_ANSWER:
                    r = trace.result
                    low_conf = LowConfidenceRecord(
                        raw_question=turn.user_text, source="retrieval_low_conf",
                        reason=json.dumps({
                            "requested_strategy": r.requested_strategy,
                            "effective_strategy": r.effective_strategy,
                            "top1": r.confidence_score,
                            "threshold": r.confidence_threshold,
                            "note": r.note}, ensure_ascii=False),
                        conversation_id=cid)
                elif trace.status == "ok" and final_text.strip() == REFUSAL_ANSWER:
                    refs = [{"ref_no": e["ref_no"], "chunk_id": e["chunk_id"]}
                            for e in (trace.evidence or [])]
                    low_conf = LowConfidenceRecord(
                        raw_question=turn.user_text, source="self_check",
                        reason=json.dumps({"evidence_refs": refs}, ensure_ascii=False),
                        conversation_id=cid)

            stored = [StoredMessage("user", turn.user_text)]
            if calls:
                stored.append(StoredMessage("assistant", "".join(text_parts) or None,
                                            tool_calls=calls))
                for tm, error_code, call in zip(tool_messages, tool_error_codes, calls):
                    metadata = None
                    if (call["name"] == "query_faq" and tm.status != "error"
                            and trace is not None and trace.evidence is not None):
                        metadata = {"citations": trace.evidence, "retrieval": {
                            "requested_strategy": trace.result.requested_strategy,
                            "effective_strategy": trace.result.effective_strategy,
                            "confidence_score": trace.result.confidence_score,
                            "confidence_threshold": trace.result.confidence_threshold,
                            "leg_counts": trace.result.leg_counts}}
                    stored.append(StoredMessage(
                        "tool",
                        wrap(tm.content, tm.status != "error",
                             None if tm.status != "error" else error_code,
                             self._settings.max_tool_result_chars, metadata=metadata),
                        tool_call_id=tm.tool_call_id,
                    ))
            stored.append(StoredMessage("assistant", final_text))
            commit_task = asyncio.ensure_future(
                self._store.commit_turn(turn.session_id, stored, low_confidence=low_conf))
            try:
                await asyncio.shield(commit_task)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await commit_task  # 等事务落地
                raise
            committed = True
            if (trace is not None and trace.status == "ok" and trace.evidence
                    and final_text.strip() != REFUSAL_ANSWER):
                yield CitationsEvent(trace.evidence)
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
        if sum(1 for c in calls if c["name"] == "query_faq") > 1:
            return False  # query_faq 每轮最多一次(完整问题一次传入)
        return True
