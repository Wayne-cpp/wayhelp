import asyncio
import json
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
    SessionLockRegistry,
)
from app.sessions import InMemorySessionStore
from tests.conftest import (
    TEST_USER_ID,
    FakeChunk,
    FakeStreamModel,
    UserBoundMemoryStore,
    make_settings,
)

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


async def test_acquire_cancelled_while_waiting_rolls_back_user_count():
    """M2:等待锁的协程被取消,_users 计数必须回滚,不得残留。"""
    reg = SessionLockRegistry()
    await reg.acquire("s1")  # 持有者
    waiter = asyncio.create_task(reg.acquire("s1"))
    await asyncio.sleep(0.05)  # 等待者已挂起在 lock.acquire
    assert reg._users["s1"] == 2
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert reg._users["s1"] == 1  # 回滚等待者的计数,不影响持有者
    reg.release("s1")
    assert "s1" not in reg._users and "s1" not in reg._locks  # release 语义不变


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


async def test_toolset_factory_failure_releases_lock():
    """M3:工具装配(toolset_factory/registry/bind_tools/astream)抛异常时,
    锁必须随 finally 释放,不得永久持锁。"""
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)

    def boom(sid):
        raise RuntimeError("toolset boom")

    service = ChatService(store, FakeStreamModel(["答"]), settings, SYSTEM,
                          toolset_factory=boom)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    with pytest.raises(RuntimeError):
        await collect(service, turn)
    assert turn.lock_key not in service._locks._locks


async def test_bind_tools_failure_releases_lock():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def query_order(order_id: str) -> str:
        """查订单"""
        return "ok"

    class BindBoomModel(FakeStreamModel):
        def bind_tools(self, tools):
            raise RuntimeError("bind boom")

    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    service = ChatService(store, BindBoomModel(["答"]), settings, SYSTEM,
                          toolset_factory=lambda sid: [query_order])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    with pytest.raises(RuntimeError):
        await collect(service, turn)
    assert turn.lock_key not in service._locks._locks


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


async def test_stored_tool_envelope_keeps_real_error_code():
    """M1:落库 tool 行 envelope 的 error_code 必须是 executor 的真实码
    (unknown_tool / invalid_args),不得一律写 tool_error。"""
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def strict_tool(n: int) -> str:
        """严格参数"""
        return str(n)

    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = FakeStreamModel([
        ("tool", [
            {"name": "ghost_tool", "args": "{\"x\": \"1\"}", "id": "call_1", "index": 0},
            {"name": "strict_tool", "args": "{\"n\": \"不是数字\"}", "id": "call_2", "index": 1},
        ]),
        ("then", ["最终答复"]),
    ])
    service = ChatService(store, model, settings, SYSTEM,
                          toolset_factory=lambda sid: [strict_tool])
    turn = await service.prepare(TEST_USER_ID, None, "两个工具调用")
    events = await collect(service, turn)
    assert any(isinstance(e, DoneEvent) for e in events)  # 工具失败不阻断本轮
    tool_rows = [m for m in await store.snapshot(turn.session_id) if m.role == "tool"]
    assert len(tool_rows) == 2
    codes = {m.tool_call_id: json.loads(m.content)["error_code"] for m in tool_rows}
    assert codes == {"call_1": "unknown_tool", "call_2": "invalid_args"}


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


# ---- T9:检索硬闸门 / 自评拒答 / citations 帧 ----

FAQ_CALL = {"name": "query_faq", "args": "{\"keyword\": \"能寄到日本吗\"}",
            "id": "c1", "index": 0}


def _faq_hit():
    from app.knowledge.retriever import KnowledgeHit
    return KnowledgeHit(5, 0.9, "faq", "能寄到日本吗",
                        "目前仅支持中国大陆地区配送。", None, 0, "配送/服务范围")


class LowConfRetriever:
    """零命中 → low_confidence=True(硬闸门素材)。"""

    def search(self, q, **kw):
        from app.knowledge.query_understanding import passthrough_plan
        from app.knowledge.retriever import RetrievalResult
        return RetrievalResult([], "hybrid_rerank", "hybrid_rerank", None, 0.5,
                               True, None, passthrough_plan(q), {"dense": 0, "bm25": 0})


class OkRetriever:
    """一条高置信命中(chunk_id=5,Top-1 0.9 ≥ 阈值 0.5)。"""

    def search(self, q, **kw):
        from app.knowledge.query_understanding import passthrough_plan
        from app.knowledge.retriever import RetrievalResult
        return RetrievalResult([_faq_hit()], "hybrid_rerank", "hybrid_rerank",
                               0.9, 0.5, False, None, passthrough_plan(q),
                               {"dense": 1, "bm25": 1, "fused": 1})


def make_faq_service(retriever, script):
    from app.tools.business import build_tools

    settings = make_settings()
    store = UserBoundMemoryStore(1000, 100, 8000)

    def factory(sid):
        return build_tools(None, 1, retriever=retriever, settings=settings)

    model = FakeStreamModel(script)
    return ChatService(store, model, settings, SYSTEM, factory), store, model


async def test_hard_gate_refusal_skips_second_call():
    from app.prompts.service import REFUSAL_ANSWER
    from app.services.chat_service import CitationsEvent

    svc, store, model = make_faq_service(LowConfRetriever(), [
        ("tool", [FAQ_CALL]),
        ("finish", "tool_calls"),
        ("then", ["不应被调用"]),
    ])
    turn = await svc.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in svc.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == REFUSAL_ANSWER
    assert len(model.received) == 1                  # 第二次模型调用未发生
    assert not any(isinstance(e, CitationsEvent) for e in events)  # 拒答不推引用帧
    rec = store.low_confidence[0]
    assert rec.source == "retrieval_low_conf"
    for key in ("requested_strategy", "effective_strategy", "top1", "threshold", "note"):
        assert key in rec.reason
    assert rec.conversation_id is None               # 内存会话 id 非十进制 → None


async def test_self_check_refusal_pools_self_check():
    from app.prompts.service import REFUSAL_ANSWER
    from app.services.chat_service import CitationsEvent

    svc, store, model = make_faq_service(OkRetriever(), [
        ("tool", [FAQ_CALL]),
        ("finish", "tool_calls"),
        ("then", [REFUSAL_ANSWER]),   # 检索 ok,模型第二次调用精确输出拒答话术
    ])
    turn = await svc.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in svc.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == REFUSAL_ANSWER
    assert len(model.received) == 2                   # 自评路径仍走第二次调用
    assert not any(isinstance(e, CitationsEvent) for e in events)
    rec = store.low_confidence[0]
    assert rec.source == "self_check"
    assert '"chunk_id": 5' in rec.reason and "evidence_refs" in rec.reason


async def test_citations_event_pushed_with_evidence():
    from app.services.chat_service import CitationsEvent
    from app.tool_envelope import unwrap, unwrap_metadata

    svc, store, model = make_faq_service(OkRetriever(), [
        ("tool", [FAQ_CALL]),
        ("finish", "tool_calls"),
        ("then", ["目前仅支持中国大陆地区配送 [1]"]),
    ])
    turn = await svc.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in svc.stream(turn)]
    cit = next(e for e in events if isinstance(e, CitationsEvent))
    assert cit.citations[0]["ref_no"] == 1 and cit.citations[0]["chunk_id"] == 5
    # 顺序:citations 在最后一个 delta 之后、Done 之前
    types = [type(e).__name__ for e in events]
    assert types.index("CitationsEvent") < types.index("DoneEvent")
    assert types.index("CitationsEvent") > max(
        i for i, t in enumerate(types) if t == "DeltaEvent")
    assert store.low_confidence == []                 # 正常作答不入池
    # 三处同一份列表:citations 帧 = query_faq 出参 evidence = envelope v2 metadata
    tool_row = next(m for m in await store.snapshot(turn.session_id) if m.role == "tool")
    body, ok = unwrap(tool_row.content)
    assert ok and json.loads(body)["evidence"] == cit.citations
    md = unwrap_metadata(tool_row.content)
    assert md["citations"] == cit.citations
    assert md["retrieval"]["confidence_score"] == 0.9
    assert md["retrieval"]["effective_strategy"] == "hybrid_rerank"


async def test_tool_error_not_pooled():
    from app.knowledge.query_understanding import passthrough_plan
    from app.knowledge.retriever import NOTE_REBUILDING, RetrievalResult
    from app.services.chat_service import CitationsEvent

    class RebuildingRetriever:
        def search(self, q, **kw):
            return RetrievalResult([], "hybrid_rerank", "hybrid_rerank", None, 0.5,
                                   True, NOTE_REBUILDING, passthrough_plan(q), {})

    svc, store, model = make_faq_service(RebuildingRetriever(), [
        ("tool", [FAQ_CALL]),
        ("finish", "tool_calls"),
        ("then", ["知识库正在维护,请稍后再试"]),
    ])
    turn = await svc.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in svc.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == "知识库正在维护,请稍后再试"     # 系统故障走第二次模型如实说明
    assert len(model.received) == 2
    assert not any(isinstance(e, CitationsEvent) for e in events)
    assert store.low_confidence == []                 # 系统故障不入池


async def test_query_faq_twice_in_one_turn_rejected():
    chunks = [
        {"name": "query_faq", "args": "{\"keyword\": \"a\"}", "id": "c1", "index": 0},
        {"name": "query_faq", "args": "{\"keyword\": \"b\"}", "id": "c2", "index": 1},
    ]
    svc, store, model = make_faq_service(OkRetriever(), [("tool", chunks)])
    turn = await svc.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in svc.stream(turn)]
    assert events[-1].code == "invalid_tool_call"      # query_faq 每轮最多一次
    assert await store.snapshot(turn.session_id) == []
