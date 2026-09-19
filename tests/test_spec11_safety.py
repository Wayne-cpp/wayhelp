"""spec §11 安全网测试(Task 13 按图语义重接):提交 shield 窗口 / 释锁 / 上下文预算。

性质归属:取消不提交半轮与同 session 串行 → test_chat_service;写工单唯一通道
的归属/消息校验 → test_chat_action;写工具 shield → test_tools。
旧「聊天中建单后模型失败/取消仍保留工单」场景退役:聊天图零建单副作用,
create_ticket/query_faq 均不在 Agent 注册表内。
"""

import asyncio
import threading

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.services.chat_service import ChatService, DoneEvent, ErrorEvent
from app.sessions import InMemorySessionStore
from tests.conftest import TEST_USER_ID, ScriptedChatModel, make_settings

SYSTEM = "system"
USER = TEST_USER_ID

BUSINESS = ['{"intent":"订单","confidence":0.9}']
KNOWLEDGE = ['{"intent":"售后","confidence":0.9}']
GEN = ['{"mode":"general"}']  # ch06 Task 7:售后脚本须经 refund_scope(general)进 refund_policy


def _service(model, settings=None, store=None, retriever=None):
    settings = settings or make_settings()
    store = store or InMemorySessionStore(10, 10, 100)
    service = ChatService(store, model, settings, SYSTEM)
    deps = GraphDeps(model=model, settings=settings, retriever=retriever,
                     store=store, system_prompt=SYSTEM)
    service.set_graph(build_chat_graph(deps, InMemorySaver()))
    return service, store


# --- a) shield 窗口(§5.4):log 节点提交事务进行中取消,锁在事务落地后才释放 ---

class SlowCommitStore(InMemorySessionStore):
    """commit_turn 进入后 sleep 0.3s 再落地,拉宽「事务进行中」窗口。"""

    def __init__(self, *args, entered: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = entered
        self.commit_done = False

    async def commit_turn(self, session_id, messages, low_confidence=None):
        from app.sessions import CommitTurnResult  # noqa: F401  (返回类型参照)
        self.entered.set()
        await asyncio.sleep(0.3)
        result = await super().commit_turn(session_id, messages,
                                           low_confidence=low_confidence)
        self.commit_done = True
        return result


async def test_cancel_during_commit_lock_released_only_after_transaction():
    entered = threading.Event()
    store = SlowCommitStore(10, 10, 100, entered=entered)
    model = ScriptedChatModel(scripts=[BUSINESS, ["最终回答"]])
    service, _ = _service(model, store=store)
    sid = await store.create(USER)
    holder = {}

    async def run():
        turn = await service.prepare(USER, sid, "hi")
        holder["turn"] = turn
        async for _ in service.stream(turn):
            pass

    task = asyncio.create_task(run())
    await asyncio.to_thread(entered.wait)  # commit 已开始、尚未落地
    turn = holder["turn"]
    assert turn.lock_key in service._locks._locks  # 事务落地前锁未释放
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.commit_done is True  # await task 返回即事务已落地,锁释放在其后
    assert turn.lock_key not in service._locks._locks
    assert [m.content for m in await store.snapshot(sid)] == ["hi", "最终回答"]


# --- b) 上下文预算(§5.3 / §10,agent 段前置预算):error 帧收尾、其后无 [DONE]、
#         一次模型调用都不发起、不提交 ---
# 与 test_chat_service 的预算兜底(max_agent_tokens → AGENT_BUDGET_ANSWER)分工:
# 本例守 build_agent_context 放不下证据时的 tool_context_too_long abort 路径。

async def test_tool_context_too_long_is_final_frame_no_commit():
    class LongAnswerRetriever:
        def search(self, q, **kw):
            from app.knowledge.query_understanding import passthrough_plan
            from app.knowledge.retriever import KnowledgeHit, RetrievalResult
            hit = KnowledgeHit(5, 0.9, "faq", q, "答" * 3000, None, 0, "配送/服务范围")
            return RetrievalResult([hit], "hybrid_rerank", "hybrid_rerank",
                                   0.9, 0.5, False, None, passthrough_plan(q),
                                   {"dense": 1, "bm25": 1, "fused": 1})

    settings = make_settings(max_input_tokens=200, max_tool_result_chars=100000)
    store = InMemorySessionStore(10, 10, 100)
    model = ScriptedChatModel(scripts=[KNOWLEDGE, GEN, ["不应发生的回答"]])
    service, store = _service(model, settings=settings, store=store,
                              retriever=LongAnswerRetriever())
    turn = await service.prepare(TEST_USER_ID, None, "查政策")
    events = [e async for e in service.stream(turn)]
    codes = [e.code for e in events if isinstance(e, ErrorEvent)]
    assert codes == ["tool_context_too_long"]
    assert isinstance(events[-1], ErrorEvent)  # error 帧收尾(SSE 层等价于 error 后无 [DONE])
    assert not any(isinstance(e, DoneEvent) for e in events)
    assert len(model.scripts) == 1  # 预算前置:agent 一次模型调用都没发起
    assert await store.snapshot(turn.session_id) == []  # 不产生多余提交
    assert turn.lock_key not in service._locks._locks
