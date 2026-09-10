import asyncio
import logging

import pytest
from langchain_core.messages import HumanMessage

from app.errors import MessageTooLongError, SessionNotFoundError
from app.services.chat_service import (
    ChatService,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    SessionEvent,
)
from app.sessions import InMemorySessionStore
from tests.conftest import TEST_USER_ID, FakeChunk, FakeStreamModel, make_settings

SYSTEM = "你是电商售后客服小蜜。"


def make_service(script, **settings_over):
    settings = make_settings(**settings_over)
    store = InMemorySessionStore(
        settings.max_sessions, settings.max_messages_per_session, settings.max_message_chars
    )
    model = FakeStreamModel(script)
    return ChatService(store, model, settings, SYSTEM), store, model


async def collect(service, turn):
    return [e async for e in service.stream(turn)]


async def test_happy_path_commits_turn():
    service, store, model = make_service(["你好", ",我是", "小蜜"])
    turn = await service.prepare(TEST_USER_ID, None,"你好")
    events = await collect(service, turn)
    assert isinstance(events[0], SessionEvent)
    deltas = [e.content for e in events if isinstance(e, DeltaEvent)]
    assert deltas == ["你好", ",我是", "小蜜"]
    assert isinstance(events[-1], DoneEvent)
    sid = events[0].session_id
    snap = await store.snapshot(sid)
    assert [m.content for m in snap] == ["你好", "你好,我是小蜜"]


async def test_prepare_reuse_existing_session():
    service, store, _ = make_service(["答"])
    turn = await service.prepare(TEST_USER_ID, None,"第一轮")
    await collect(service, turn)
    sid = turn.session_id
    turn2 = await service.prepare(TEST_USER_ID, sid, "第二轮")
    await collect(service, turn2)
    assert [m.role for m in await store.snapshot(sid)] == ["user", "assistant"] * 2


async def test_prepare_unknown_session_404():
    service, _, _ = make_service([])
    import uuid

    with pytest.raises(SessionNotFoundError):
        await service.prepare(TEST_USER_ID, str(uuid.uuid4()), "hi")


async def test_overlong_input_no_session_created():
    service, store, _ = make_service([], max_message_chars=10)
    with pytest.raises(MessageTooLongError):
        await service.prepare(TEST_USER_ID, None,"这" * 20)
    assert store._sessions == {}


async def test_release_turn_idempotent():
    service, _, _ = make_service([])
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    assert turn.lock_key in service._locks._locks  # 持锁期间 registry 保有该 session 条目
    service.release_turn(turn)
    assert turn.lock_key not in service._locks._locks
    service.release_turn(turn)  # 第二次调用不炸,条目保持已清理
    assert turn.lock_key not in service._locks._locks


async def test_current_input_enters_prompt_exactly_once():
    service, _, model = make_service(["ok"])
    turn = await service.prepare(TEST_USER_ID, None,"独一无二的问题")
    await collect(service, turn)
    sent = model.received[0]
    humans = [m for m in sent if isinstance(m, HumanMessage)]
    assert sum(1 for m in humans if m.content == "独一无二的问题") == 1


async def test_upstream_error_no_commit_lock_released():
    service, store, _ = make_service(["部分", RuntimeError("boom")])
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    events = await collect(service, turn)
    err = [e for e in events if isinstance(e, ErrorEvent)]
    assert err and err[0].code == "upstream_error"
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert await store.snapshot(turn.session_id) == []
    assert turn.lock_key not in service._locks._locks


async def test_upstream_error_log_sanitized(caplog):
    service, _, _ = make_service(["部分", RuntimeError("boom")])
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    with caplog.at_level(logging.WARNING, logger="app.services.chat_service"):
        await collect(service, turn)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "upstream error" in joined
    assert "RuntimeError" in joined
    assert "boom" not in joined


async def test_output_too_long_cancels_no_commit():
    service, store, _ = make_service(["太" * 30, "多" * 30, "还" * 30], max_message_chars=50)
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    events = await collect(service, turn)
    codes = [e.code for e in events if isinstance(e, ErrorEvent)]
    assert codes == ["output_too_long"]
    assert await store.snapshot(turn.session_id) == []


async def test_finish_reason_length_is_failure():
    service, store, _ = make_service(["被截断的回答", ("finish", "length")])
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    events = await collect(service, turn)
    assert any(isinstance(e, ErrorEvent) and e.code == "output_too_long" for e in events)
    assert await store.snapshot(turn.session_id) == []


async def test_empty_response_is_failure():
    service, store, _ = make_service(["", "  "])
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    events = await collect(service, turn)
    assert any(isinstance(e, ErrorEvent) and e.code == "empty_response" for e in events)
    assert await store.snapshot(turn.session_id) == []


async def test_done_send_fail_keeps_full_turn():
    service, store, _ = make_service(["完整回答"])
    turn = await service.prepare(TEST_USER_ID, None,"hi")
    agen = service.stream(turn)
    async for event in agen:
        if isinstance(event, DoneEvent):
            break  # 模拟 [DONE] 帧发送失败:消费者拿到 Done 后立即断开
    await agen.aclose()
    assert [m.content for m in await store.snapshot(turn.session_id)] == ["hi", "完整回答"]
    assert turn.lock_key not in service._locks._locks


async def test_cancel_before_commit_no_partial_turn():
    gate = asyncio.Event()
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = GatedModel(gate)
    service = ChatService(store, model, settings, SYSTEM)

    holder = {}

    async def run():
        turn = await service.prepare(TEST_USER_ID, None,"hi")
        holder["turn"] = turn
        async for _ in service.stream(turn):
            pass

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    assert len(model.received) == 1  # 已进入模型、尚未提交
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    turn = holder["turn"]
    assert await store.snapshot(turn.session_id) == []
    assert turn.lock_key not in service._locks._locks


async def test_tool_context_too_long_when_dropping_human_would_fit():
    """预算卡在「含当前 human 超限 / 不含当前 human 达标」窗口:必须发 tool_context_too_long,
    不得丢掉当前 human 后在没有用户问题的上下文里静默第二次调用(review I1 调用点 off-by-one)。"""
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def query_order(order_id: str) -> str:
        """查订单"""
        return "长" * 400

    # count_tokens_approximately: [sys,human,ai,tool]=151 > 147 >= [sys,ai,tool]=144
    settings = make_settings(max_input_tokens=147)
    store = InMemorySessionStore(10, 10, 100)
    model = FakeStreamModel([
        ("tool", [{"name": "query_order", "args": "{\"order_id\": \"1001\"}",
                   "id": "call_1", "index": 0}]),
        ("then", ["最终答复"]),
    ])
    service = ChatService(store, model, settings, SYSTEM,
                          toolset_factory=lambda sid: [query_order])
    turn = await service.prepare(TEST_USER_ID, None, "查订单 1001 的物流")
    events = await collect(service, turn)
    codes = [e.code for e in events if isinstance(e, ErrorEvent)]
    assert codes == ["tool_context_too_long"]
    assert len(model.received) == 1  # 第二次调用不得发生(旧写法会删当前 human 后继续)
    assert await store.snapshot(turn.session_id) == []


class GatedModel(FakeStreamModel):
    """每次 astream 在首尾 delta 之间等待 gate;"finish" 后计数。"""

    def __init__(self, gate: asyncio.Event):
        super().__init__([])
        self.gate = gate

    async def astream(self, messages):
        self.received.append(messages)
        yield FakeChunk("开始")
        await self.gate.wait()
        yield FakeChunk("结束")


async def _run_full(service, session_id, message):
    turn = await service.prepare(TEST_USER_ID, session_id, message)
    events = [e async for e in service.stream(turn)]
    return events, turn.session_id


async def test_same_session_serialized():
    gate = asyncio.Event()
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = GatedModel(gate)
    service = ChatService(store, model, settings, SYSTEM)

    task1 = asyncio.create_task(_run_full(service, None, "一"))
    await asyncio.sleep(0.05)
    assert len(model.received) == 1  # 第一个流已进入模型并持锁
    sid = next(iter(store._sessions))
    task2 = asyncio.create_task(_run_full(service, sid, "二"))
    await asyncio.sleep(0.05)
    assert len(model.received) == 1  # 同 session 第二个请求在等锁,未进模型
    gate.set()
    (events1, sid1), (events2, sid2) = await asyncio.gather(task1, task2)
    assert sid1 == sid2
    assert len(model.received) == 2
    assert [m.content for m in await store.snapshot(sid1)] == ["一", "开始结束", "二", "开始结束"]


async def test_different_sessions_concurrent():
    gate = asyncio.Event()
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = GatedModel(gate)
    service = ChatService(store, model, settings, SYSTEM)

    task1 = asyncio.create_task(_run_full(service, None, "甲"))
    task2 = asyncio.create_task(_run_full(service, None, "乙"))
    await asyncio.sleep(0.05)
    assert len(model.received) == 2  # 不同 session 同时进入模型
    gate.set()
    (_, sid1), (_, sid2) = await asyncio.gather(task1, task2)
    assert sid1 != sid2
    assert [m.content for m in await store.snapshot(sid1)] == ["甲", "开始结束"]
    assert [m.content for m in await store.snapshot(sid2)] == ["乙", "开始结束"]
