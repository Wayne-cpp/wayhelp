"""spec §11 安全网测试(评审 I4):副作用分界 / shield 窗口 / 第二次调用预算不足。

实现走读判定无误,本文件是守卫网(非先红后绿);各用例的红性以临时变异
app/services/chat_service.py 后能失败来验证,变异已还原,见批次总结。
"""

import asyncio
import threading
import time

import pytest
from langchain_core.tools import tool as lc_tool

from app.models import Conversation, Ticket
from app.services.chat_service import (
    ChatService,
    DoneEvent,
    ErrorEvent,
    ToolEndEvent,
)
from app.sessions import InMemorySessionStore
from app.store_db import DbSessionStore
from app.tools.business import build_tools
from tests.conftest import TEST_USER_ID, FakeChunk, FakeStreamModel, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)

SYSTEM = "system"
USER = TEST_USER_ID

TICKET_CALL = [{
    "name": "create_ticket",
    "args": "{\"description\": \"发票没收到\", \"ticket_type\": \"咨询\"}",
    "id": "call_1",
    "index": 0,
}]


def _db_ticket_state(sf, cid: int) -> tuple[str, list[str]]:
    """SQL 抽查(§11 验收同源):conversation 状态与该会话的工单号列表。"""
    with sf() as s:
        conv = s.get(Conversation, cid)
        tickets = s.query(Ticket).filter(Ticket.conversation_id == cid).all()
    return conv.status, [t.ticket_no for t in tickets]


def _ticket_service(db_session_factory, model):
    """真实 DB store + 真实 build_tools(create_ticket 写库)的 ChatService 装配。"""
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    service = ChatService(store, model, make_settings(), SYSTEM,
                          toolset_factory=lambda sid: build_tools(db_session_factory, int(sid)))
    return service, store


# --- a) 副作用分界(§6.4 / §10):工单事务成功后,后续失败/取消不得回滚副作用,
#         也不得让本轮 messages 落半个 turn -----------------------------------


async def test_ticket_survives_second_model_failure(db_session_factory):
    model = FakeStreamModel([
        ("tool", TICKET_CALL),
        ("then", [RuntimeError("boom")]),
    ])
    service, store = _ticket_service(db_session_factory, model)
    sid = await store.create(USER)
    turn = await service.prepare(USER, sid, "帮我转人工")
    events = [e async for e in service.stream(turn)]
    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1 and ends[0].ok is True and ends[0].name == "create_ticket"
    codes = [e.code for e in events if isinstance(e, ErrorEvent)]
    assert codes == ["upstream_error"]
    assert not any(isinstance(e, DoneEvent) for e in events)
    status, ticket_nos = _db_ticket_state(db_session_factory, int(sid))
    assert status == "已转人工"
    assert len(ticket_nos) == 1 and len(ticket_nos[0]) == 27  # T + 14 位时间 + 12 位 hex
    assert await store.snapshot(sid) == []  # 本轮 messages 不提交
    assert turn.lock_key not in service._locks._locks


class TicketThenHangModel:
    """第一次 astream 返回 create_ticket 工单申请;第二次 astream 阻塞在 gate 上。"""

    def __init__(self, gate: asyncio.Event):
        self.gate = gate
        self.received: list = []
        self.second_started = asyncio.Event()

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.received.append(messages)
        if len(self.received) == 1:
            yield FakeChunk("", tool_call_chunks=TICKET_CALL)
        else:
            self.second_started.set()
            await self.gate.wait()
            yield FakeChunk("最终回答")


async def test_ticket_survives_client_cancel_during_second_call(db_session_factory):
    model = TicketThenHangModel(asyncio.Event())
    service, store = _ticket_service(db_session_factory, model)
    sid = await store.create(USER)
    holder, seen = {}, []

    async def run():
        turn = await service.prepare(USER, sid, "帮我转人工")
        holder["turn"] = turn
        async for e in service.stream(turn):
            seen.append(e)

    task = asyncio.create_task(run())
    await model.second_started.wait()  # 建单已执行完,已进入第二次模型调用
    assert any(isinstance(e, ToolEndEvent) and e.ok for e in seen)
    status, ticket_nos = _db_ticket_state(db_session_factory, int(sid))
    assert status == "已转人工" and len(ticket_nos) == 1  # 取消前副作用已独立落库
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await store.snapshot(sid) == []  # 取消后本轮 messages 仍不提交
    assert not any(isinstance(e, DoneEvent) for e in seen)
    assert holder["turn"].lock_key not in service._locks._locks


# --- b) shield 窗口(§5.4):提交事务进行中取消,锁在事务落地后才释放 ----------


async def test_cancel_during_commit_lock_released_only_after_transaction(
        db_session_factory, monkeypatch):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    entered = threading.Event()
    state = {"commit_done": False}
    real_commit = store._commit_sync

    def slow_commit(session_id, messages, low_confidence=None):
        entered.set()
        time.sleep(0.3)  # 拉宽"事务进行中"窗口
        real_commit(session_id, messages, low_confidence)
        state["commit_done"] = True

    monkeypatch.setattr(store, "_commit_sync", slow_commit)
    service = ChatService(store, FakeStreamModel(["最终回答"]), make_settings(), SYSTEM)
    holder = {}

    async def run():
        turn = await service.prepare(USER, sid, "hi")
        holder["turn"] = turn
        async for _ in service.stream(turn):
            pass

    task = asyncio.create_task(run())
    await asyncio.to_thread(entered.wait)  # commit 已开始、尚未落地
    turn = holder["turn"]
    assert turn.lock_key in service._locks._locks  # 事务落地前 registry 条目仍在(锁未释放)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["commit_done"] is True  # await task 返回即事务已落地,锁释放在其后
    assert turn.lock_key not in service._locks._locks
    assert [m.content for m in await store.snapshot(sid)] == ["hi", "最终回答"]


# --- c) 第二次调用预算不足(§5.3 / §10):error 帧收尾、其后无 [DONE]、不提交 ---
# 与 test_chat_service.test_tool_context_too_long_when_dropping_human_would_fit(I1)
# 分工:I1 守「丢当前 human 恰能放下」的 off-by-one 窗口;本例守「全丢历史也
# 放不下」的常规路径 + 末帧/无多余提交断言。


async def test_tool_context_too_long_is_final_frame_no_commit():
    @lc_tool
    def query_order(order_id: str) -> str:
        """查订单"""
        return "长" * 3000

    settings = make_settings(max_input_tokens=200, max_tool_result_chars=100000)
    store = InMemorySessionStore(10, 10, 100)
    model = FakeStreamModel([
        ("tool", [{"name": "query_order", "args": "{\"order_id\": \"1001\"}",
                   "id": "call_1", "index": 0}]),
        ("then", ["不应发生的第二次回答"]),
    ])
    service = ChatService(store, model, settings, SYSTEM,
                          toolset_factory=lambda sid: [query_order])
    turn = await service.prepare(TEST_USER_ID, None, "查订单 1001")
    events = [e async for e in service.stream(turn)]
    codes = [e.code for e in events if isinstance(e, ErrorEvent)]
    assert codes == ["tool_context_too_long"]
    assert isinstance(events[-1], ErrorEvent)  # error 帧收尾(SSE 层等价于 error 后无 [DONE])
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert len(model.received) == 1  # 预算不足,第二次调用不发起
    assert await store.snapshot(turn.session_id) == []  # 不产生多余提交
    assert turn.lock_key not in service._locks._locks
