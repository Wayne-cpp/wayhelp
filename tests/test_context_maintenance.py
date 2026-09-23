# tests/test_context_maintenance.py — ch07 Task 10 图集成:
# log 节点盖章 db_id 进 checkpoint + 层1 降级持久化 + 层2 超预算触发后台摘要。
# harness 镜像 tests/test_graph_smoke.py 的构造方式:build_chat_graph + InMemorySaver
# + conftest ScriptedChatModel(scripts=[])(空脚本 → 每次调用回 "(空)":understand 降级
# 透传、classify 兜底「其他」→ other_fallback 固定答复,零工具往返)。
import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from app.sessions import InMemorySessionStore
from app.services.summarizer import SummaryRunner
from app.services.token_budget import compute_budget
from tests.conftest import ScriptedChatModel, make_settings


def _tiny_settings():
    # 极小窗口:BUDGET 只够装一两轮 → 强制降级 + 触发摘要
    # (该配置下 layer1=350 / layer2=150 token;问题文本各垫 "长"*120 使每轮约
    #  170 token——3 轮即超层1,降级后层2 约 172 > 150 触发摘要级联)
    return make_settings(model_context_window=3000, max_output_tokens=200,
                         max_user_input_tokens=200, max_agent_steps=1,
                         tool_result_max_tokens=100, rerank_top_k=1,
                         history_target_turns=50, steady_tokens_per_turn=10,
                         safety_margin_tokens=0, summary_projection_tokens=200)


async def test_checkpoint_stamped_and_degrade_and_summary():
    settings = _tiny_settings()
    store = InMemorySessionStore(10, 1000, 8000)
    model = ScriptedChatModel(scripts=[])
    budget = compute_budget(settings, sys_tokens=0)
    runner = SummaryRunner(store, model, settings)
    deps = GraphDeps(model=model, settings=settings, retriever=None, store=store,
                     system_prompt="SYS", context_budget=budget,
                     summary_runner=runner)
    graph = build_chat_graph(deps, InMemorySaver())
    sid = await store.create("u1")
    config = {"configurable": {"thread_id": sid, "user_id": "u1"}}
    for q in ["第一问订单A1001" + "长" * 120, "第二问" + "长" * 120,
              "第三问" + "长" * 120, "第四问" + "长" * 120]:
        async for _ in graph.astream(new_turn_state(q), config,
                                     stream_mode=["custom"]):
            pass
    st = await graph.aget_state(config)
    msgs = st.values["messages"]
    assert msgs and all(
        m.additional_kwargs.get("db_id") for m in msgs
        if m.type in ("human", "ai"))                       # 盖章进 checkpoint
    meta = await store.get_context_meta(sid, "u1")
    assert meta.layer1_from is not None                     # 层1 降级已持久化
    await asyncio.sleep(0.1)                                # 摘要任务后台落地
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary is not None and meta.summary_upto is not None
    await runner.aclose()


async def test_summary_does_not_block_turn():
    settings = _tiny_settings()
    store = InMemorySessionStore(10, 1000, 8000)
    model = ScriptedChatModel(scripts=[])
    runner = SummaryRunner(store, model, settings)
    deps = GraphDeps(model=model, settings=settings, retriever=None, store=store,
                     system_prompt="SYS",
                     context_budget=compute_budget(settings, sys_tokens=0),
                     summary_runner=runner)
    graph = build_chat_graph(deps, InMemorySaver())
    sid = await store.create("u1")
    config = {"configurable": {"thread_id": sid, "user_id": "u1"}}
    async for _ in graph.astream(new_turn_state("触发摘要的一问" + "长" * 300), config,
                                 stream_mode=["custom"]):
        pass
    # astream 返回即本轮结束;此时摘要任务可能尚未完成,但 meta 不因此炸
    meta = await store.get_context_meta(sid, "u1")
    assert meta is not None
    # 摘要任务落地后再 aclose:取消 in-flight 任务会让 CancelledError 逃出
    # summarizer.aclose 的 suppress(Exception)(Task 7 行为,非本任务改动面)
    await asyncio.sleep(0.05)
    await runner.aclose()
