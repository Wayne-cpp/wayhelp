"""图驱动版 ChatService 契约测试(Task 13 重写)。

退役:旧「两次调用」编排专属契约(调用次数钉、落库剥离第二轮 tool_calls、
empty_response 帧、query_faq 场景)。SSE 帧协议契约见 test_chat_api_tools;
锁释放/aclose 的 HTTP 层契约见 test_chat_api 既有用例。
"""
import asyncio
import json
import logging
import uuid

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from app.errors import MessageTooLongError, SessionNotFoundError
from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.prompts.service import (
    AGENT_BUDGET_ANSWER,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    KB_UNAVAILABLE_ANSWER,
    REFUSAL_ANSWER,
)
from app.services.chat_service import (
    ChatService,
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    SessionEvent,
    SessionLockRegistry,
    SuggestActionsEvent,
)
from app.sessions import InMemorySessionStore
from tests.conftest import TEST_USER_ID, ScriptedChatModel, make_settings

SYSTEM = "你是电商售后客服小蜜。"

# 分类段脚本(经 ainvoke 消耗);其后每段归 main_agent 的一次 astream
BUSINESS = ['{"intent":"订单","needs_knowledge":false}']
KNOWLEDGE = ['{"intent":"售后","needs_knowledge":true}']
GEN = ['{"mode":"general"}']  # ch06 Task 7:售后脚本须经 refund_scope(general)进 refund_policy
CHITCHAT = ['{"intent":"闲聊","needs_knowledge":false}']


def make_service(scripts, retriever=None, store=None, **settings_over):
    settings = make_settings(**settings_over)
    store = store or InMemorySessionStore(
        settings.max_sessions, settings.max_messages_per_session, settings.max_message_chars)
    model = ScriptedChatModel(scripts=[list(s) for s in scripts])
    service = ChatService(store, model, settings, SYSTEM)
    deps = GraphDeps(model=model, settings=settings, retriever=retriever,
                     store=store, system_prompt=SYSTEM)
    service.set_graph(build_chat_graph(deps, InMemorySaver()))
    return service, store, model


# ---- prepare:长度闸 / token 预算早闸 / 会话解析 ----

async def test_happy_path_commits_turn():
    service, store, _ = make_service([BUSINESS, ["你好", ",我是", "小蜜"]])
    turn = await service.prepare(TEST_USER_ID, None, "你好")
    events = [e async for e in service.stream(turn)]
    assert isinstance(events[0], SessionEvent)
    deltas = [e.content for e in events if isinstance(e, DeltaEvent)]
    assert deltas == ["你好", ",我是", "小蜜"]
    assert isinstance(events[-1], DoneEvent)
    sid = events[0].session_id
    snap = await store.snapshot(sid)
    assert [m.content for m in snap] == ["你好", "你好,我是小蜜"]


async def test_prepare_reuse_existing_session():
    service, store, _ = make_service([BUSINESS, ["答"],
                                       ['{"resolved_query": ""}'],  # 第二轮 understand 透传
                                       BUSINESS, ["答二"]])
    turn = await service.prepare(TEST_USER_ID, None, "第一轮")
    [e async for e in service.stream(turn)]
    sid = turn.session_id
    turn2 = await service.prepare(TEST_USER_ID, sid, "第二轮")
    [e async for e in service.stream(turn2)]
    assert [m.role for m in await store.snapshot(sid)] == ["user", "assistant"] * 2


async def test_prepare_unknown_session_404():
    service, _, _ = make_service([BUSINESS])
    with pytest.raises(SessionNotFoundError):
        await service.prepare(TEST_USER_ID, str(uuid.uuid4()), "hi")


async def test_overlong_input_no_session_created():
    service, store, _ = make_service([BUSINESS], max_message_chars=10)
    with pytest.raises(MessageTooLongError):
        await service.prepare(TEST_USER_ID, None, "这" * 20)
    assert store._sessions == {}


async def test_input_token_budget_gate_in_prepare():
    """check_input_budget 早闸:系统提示 + 当前输入超预算时,建会话前即拒绝。"""
    service, store, _ = make_service([BUSINESS], max_input_tokens=5)
    with pytest.raises(MessageTooLongError):
        await service.prepare(TEST_USER_ID, None, "一段肯定远超五个 token 的输入文本")
    assert store._sessions == {}


# ---- 锁:注册表语义与图驱动的释放 ----

async def test_release_turn_idempotent():
    service, _, _ = make_service([BUSINESS])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    assert turn.lock_key in service._locks._locks  # 持锁期间 registry 保有该 session 条目
    service.release_turn(turn)
    assert turn.lock_key not in service._locks._locks
    service.release_turn(turn)  # 第二次调用不炸,条目保持已清理
    assert turn.lock_key not in service._locks._locks


async def test_acquire_cancelled_while_waiting_rolls_back_user_count():
    """等待锁的协程被取消,_users 计数必须回滚,不得残留。"""
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
    assert "s1" not in reg._users and "s1" not in reg._locks


# ---- 上游错误帧(分类段 / agent 段)与脱敏 ----

async def test_classify_upstream_error_no_commit_lock_released():
    service, store, _ = make_service([[RuntimeError("boom")]])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = [e async for e in service.stream(turn)]
    err = [e for e in events if isinstance(e, ErrorEvent)]
    assert err and err[0].code == "upstream_error"
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert await store.snapshot(turn.session_id) == []
    assert turn.lock_key not in service._locks._locks


async def test_agent_upstream_error_after_partial_delta():
    service, store, _ = make_service([BUSINESS, ["部分", RuntimeError("boom")]])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = [e async for e in service.stream(turn)]
    deltas = [e.content for e in events if isinstance(e, DeltaEvent)]
    assert deltas == ["部分"]
    assert [e.code for e in events if isinstance(e, ErrorEvent)] == ["upstream_error"]
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert await store.snapshot(turn.session_id) == []
    assert turn.lock_key not in service._locks._locks


async def test_upstream_error_log_sanitized(caplog):
    service, _, _ = make_service([[RuntimeError("boom")]])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    with caplog.at_level(logging.WARNING, logger="wayhelp.graph"):
        [e async for e in service.stream(turn)]
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "upstream error" in joined
    assert "RuntimeError" in joined
    assert "boom" not in joined


async def test_bind_tools_failure_internal_error_lock_released():
    """节点内装配异常(非 TurnAbort)→ 图异常 → internal_error 帧,锁必须释放。"""

    class BindBoomModel(ScriptedChatModel):
        def bind_tools(self, tools, **kwargs):
            raise RuntimeError("bind boom")

    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = BindBoomModel(scripts=[BUSINESS, ["答"]])
    service = ChatService(store, model, settings, SYSTEM)
    deps = GraphDeps(model=model, settings=settings, retriever=None,
                     store=store, system_prompt=SYSTEM)
    service.set_graph(build_chat_graph(deps, InMemorySaver()))
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = [e async for e in service.stream(turn)]
    assert [e.code for e in events if isinstance(e, ErrorEvent)] == ["internal_error"]
    assert await store.snapshot(turn.session_id) == []
    assert turn.lock_key not in service._locks._locks


# ---- 长度与预算护栏(agent 段语义) ----

async def test_output_too_long_cancels_no_commit():
    service, store, _ = make_service(
        [BUSINESS, ["太" * 30, "多" * 30, "还" * 30]], max_message_chars=50)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = [e async for e in service.stream(turn)]
    codes = [e.code for e in events if isinstance(e, ErrorEvent)]
    assert codes == ["output_too_long"]
    assert await store.snapshot(turn.session_id) == []


async def test_finish_reason_length_is_failure():
    service, store, _ = make_service([BUSINESS, ["被截断的回答", ("finish", "length")]])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = [e async for e in service.stream(turn)]
    assert any(isinstance(e, ErrorEvent) and e.code == "output_too_long" for e in events)
    assert await store.snapshot(turn.session_id) == []


async def test_agent_budget_answer_delta():
    """max_agent_tokens=1:首轮预留即超,一次模型调用都不发起,兜底话术照常提交。"""
    service, store, model = make_service([BUSINESS, ["不应被调用"]], max_agent_tokens=1)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = [e async for e in service.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == AGENT_BUDGET_ANSWER
    assert isinstance(events[-1], DoneEvent)
    assert len(model.scripts) == 1  # agent 段脚本未消耗
    assert [m.content for m in await store.snapshot(turn.session_id)] == [
        "hi", AGENT_BUDGET_ANSWER]


# ---- 工具往返与落库 envelope ----

async def test_stored_tool_envelope_keeps_real_error_code():
    """落库 tool 行 envelope 的 error_code 必须是 executor 的真实码
    (unknown_tool / invalid_args),不得一律写 tool_error。"""
    ghost = {"name": "ghost_tool", "args": "{\"x\": \"1\"}", "id": "call_1", "index": 0}
    bad = {"name": "query_order", "args": "{\"order_id\": {}}", "id": "call_2", "index": 1}
    service, store, _ = make_service([BUSINESS, [("tool", [ghost, bad])], ["最终答复"]])
    turn = await service.prepare(TEST_USER_ID, None, "两个工具调用")
    events = [e async for e in service.stream(turn)]
    assert any(isinstance(e, DoneEvent) for e in events)  # 工具失败不阻断本轮
    tool_rows = [m for m in await store.snapshot(turn.session_id) if m.role == "tool"]
    assert len(tool_rows) == 2
    codes = {m.tool_call_id: json.loads(m.content)["error_code"] for m in tool_rows}
    assert codes == {"call_1": "unknown_tool", "call_2": "invalid_args"}


# ---- 提交与取消 ----

async def test_done_send_fail_keeps_full_turn():
    service, store, _ = make_service([BUSINESS, ["完整回答"]])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    agen = service.stream(turn)
    async for event in agen:
        if isinstance(event, DoneEvent):
            break  # 模拟 [DONE] 帧发送失败:消费者拿到 Done 后立即断开
    await agen.aclose()
    assert [m.content for m in await store.snapshot(turn.session_id)] == ["hi", "完整回答"]
    assert turn.lock_key not in service._locks._locks


class GatedScriptedModel(ScriptedChatModel):
    """agent 段 astream:先出一个 token,挂起等 gate,再出收尾 token。
    langchain-core 1.6.2 的 ainvoke 会走覆写后的 _astream 聚合,故按首条消息
    区分调用方:HumanMessage=分类节点(照常弹脚本),SystemMessage=main_agent。"""

    gate: object = None
    received: list = Field(default_factory=list)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import HumanMessage
        if isinstance(messages[0], HumanMessage):
            async for chunk in super()._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
            return
        self.received.append(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content="开始"))
        await self.gate.wait()
        yield ChatGenerationChunk(message=AIMessageChunk(content="结束"))


async def test_cancel_before_commit_no_partial_turn():
    gate = asyncio.Event()
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = GatedScriptedModel(scripts=[BUSINESS], gate=gate)
    service = ChatService(store, model, settings, SYSTEM)
    deps = GraphDeps(model=model, settings=settings, retriever=None,
                     store=store, system_prompt=SYSTEM)
    service.set_graph(build_chat_graph(deps, InMemorySaver()))

    holder = {}

    async def run():
        turn = await service.prepare(TEST_USER_ID, None, "hi")
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


async def _run_full(service, session_id, message):
    turn = await service.prepare(TEST_USER_ID, session_id, message)
    events = [e async for e in service.stream(turn)]
    return events, turn.session_id


async def test_same_session_serialized():
    gate = asyncio.Event()
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = GatedScriptedModel(scripts=[BUSINESS, ['{"resolved_query": ""}'], BUSINESS],
                               gate=gate)  # 第二轮开头 understand 罐头透传(有历史)
    service = ChatService(store, model, settings, SYSTEM)
    deps = GraphDeps(model=model, settings=settings, retriever=None,
                     store=store, system_prompt=SYSTEM)
    service.set_graph(build_chat_graph(deps, InMemorySaver()))

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
    assert [m.content for m in await store.snapshot(sid1)] == [
        "一", "开始结束", "二", "开始结束"]


async def test_different_sessions_concurrent():
    gate = asyncio.Event()
    settings = make_settings()
    store = InMemorySessionStore(10, 10, 100)
    model = GatedScriptedModel(scripts=[BUSINESS, BUSINESS], gate=gate)
    service = ChatService(store, model, settings, SYSTEM)
    deps = GraphDeps(model=model, settings=settings, retriever=None,
                     store=store, system_prompt=SYSTEM)
    service.set_graph(build_chat_graph(deps, InMemorySaver()))

    task1 = asyncio.create_task(_run_full(service, None, "甲"))
    task2 = asyncio.create_task(_run_full(service, None, "乙"))
    await asyncio.sleep(0.05)
    assert len(model.received) == 2  # 不同 session 同时进入模型
    gate.set()
    (_, sid1), (_, sid2) = await asyncio.gather(task1, task2)
    assert sid1 != sid2
    assert [m.content for m in await store.snapshot(sid1)] == ["甲", "开始结束"]
    assert [m.content for m in await store.snapshot(sid2)] == ["乙", "开始结束"]


# ---- 固定回复节点与 suggest_actions ----

async def test_chitchat_zero_model_calls():
    service, store, model = make_service([CHITCHAT])
    turn = await service.prepare(TEST_USER_ID, None, "你好")
    events = [e async for e in service.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == CHITCHAT_REPLY
    assert len(model.scripts) == 0  # 分类段已消耗,固定回复零模型调用
    assert isinstance(events[-1], DoneEvent)


async def test_complaint_fixed_reply_and_suggest_actions_event():
    service, store, _ = make_service([['{"intent":"投诉","needs_knowledge":false}']])
    turn = await service.prepare(TEST_USER_ID, None, "服务太差,我要投诉")
    events = [e async for e in service.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == COMPLAINT_REPLY
    sugg = next(e for e in events if isinstance(e, SuggestActionsEvent))
    assert sugg.source_message_id  # log 提交后回传的本轮 user 消息 id
    assert [o["action"] for o in sugg.options] == ["transfer_human", "create_ticket"]
    assert sugg.options[1]["ticket_type"] == "投诉"
    assert isinstance(events[-1], DoneEvent)


# ---- T9 语义(经预检索路径):硬闸门 / 自评拒答 / citations / 故障不入池 ----

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


async def test_hard_gate_refusal_skips_agent():
    service, store, model = make_service(
        [KNOWLEDGE, GEN, ["不应被调用"]], retriever=LowConfRetriever())
    turn = await service.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in service.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == REFUSAL_ANSWER
    assert len(model.scripts) == 1                  # 硬闸门不发起模型调用
    assert not any(isinstance(e, CitationsEvent) for e in events)  # 拒答不推引用帧
    assert isinstance(events[-1], DoneEvent)
    rec = store.low_confidence[0]
    assert rec.source == "retrieval_low_conf"
    for key in ("requested_strategy", "effective_strategy", "top1", "threshold", "note"):
        assert key in rec.reason
    assert rec.conversation_id is None               # 内存会话 id 非十进制 → None


async def test_self_check_refusal_pools_self_check():
    service, store, model = make_service(
        [KNOWLEDGE, GEN, [REFUSAL_ANSWER]], retriever=OkRetriever())
    turn = await service.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in service.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == REFUSAL_ANSWER
    assert len(model.scripts) == 0                  # 自评路径走模型(脚本已消耗)
    assert not any(isinstance(e, CitationsEvent) for e in events)
    rec = store.low_confidence[0]
    assert rec.source == "self_check"
    assert '"chunk_id": 5' in rec.reason and "evidence_refs" in rec.reason


async def test_citations_event_pushed_with_evidence():
    service, store, _ = make_service(
        [KNOWLEDGE, GEN, ["目前仅支持中国大陆地区配送 [1]"]], retriever=OkRetriever())
    turn = await service.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in service.stream(turn)]
    cit = next(e for e in events if isinstance(e, CitationsEvent))
    assert cit.citations[0]["ref_no"] == 1 and cit.citations[0]["chunk_id"] == 5
    # 顺序:citations 在最后一个 delta 之后、Done 之前(log 提交成功后才发)
    types = [type(e).__name__ for e in events]
    assert types.index("CitationsEvent") < types.index("DoneEvent")
    assert types.index("CitationsEvent") > max(
        i for i, t in enumerate(types) if t == "DeltaEvent")
    assert store.low_confidence == []               # 正常作答不入池


async def test_kb_unavailable_not_pooled():
    from app.knowledge.query_understanding import passthrough_plan
    from app.knowledge.retriever import NOTE_REBUILDING, RetrievalResult

    class RebuildingRetriever:
        def search(self, q, **kw):
            return RetrievalResult([], "hybrid_rerank", "hybrid_rerank", None, 0.5,
                                   True, NOTE_REBUILDING, passthrough_plan(q), {})

    service, store, model = make_service(
        [KNOWLEDGE, GEN, ["不应被调用"]], retriever=RebuildingRetriever())
    turn = await service.prepare(TEST_USER_ID, None, "能寄到日本吗")
    events = [e async for e in service.stream(turn)]
    deltas = "".join(e.content for e in events if isinstance(e, DeltaEvent))
    assert deltas == KB_UNAVAILABLE_ANSWER          # 系统故障固定话术,不入 Agent
    assert len(model.scripts) == 1
    assert not any(isinstance(e, CitationsEvent) for e in events)
    assert store.low_confidence == []               # 系统故障不入池
