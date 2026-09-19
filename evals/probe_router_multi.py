# evals/probe_router_multi.py —— 真模型多轮剧本 probe(手跑,烧额度,不进 pytest)
#
# in-process 建图(ChatOpenAI 真模型 + 真 retriever + InMemorySaver + InMemorySessionStore),
# 走 ChatService 生产驱动路径跑三个多轮剧本,逐轮打印:
#   raw → resolved → intent(confidence) → route → refund_mode → 工具调用 → active_order
# 退出码 = 断言失败数(每轮期望 intent/mode/是否挂起/消解锚点见 SCRIPTS 表)。
# 剧本(spec §13):
#   1. 物流 → 退款指代(这单=1111-1001,唯一候选直通不挂起) → 切回物流
#   2. 如何申请退款(general) → 我要申请退款(order_specific 挂起 → 脚本内 resume 1111-1001)
#   3. 这个能退吗(无历史无焦点挂起 → resume) → 这单到哪了(助手答复不复述订单号,靠 active_order 消解)
# 前置:docker compose up -d(MySQL)、.env 在线、data/milvus_lite.db 已建库。
# 用法: uv run python evals/probe_router_multi.py
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.db import make_engine, make_session_factory, ping
from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.knowledge.embedding import build_embeddings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.reranker import SiliconFlowReranker
from app.knowledge.retriever import KnowledgeRetriever
from app.prompts.service import SERVICE_SYSTEM_PROMPT
from app.services.chat_service import (
    ChatService, DeltaEvent, DoneEvent, OrderSelectorEvent, ToolStartEvent,
)
from app.sessions import InMemorySessionStore

USER_ID = "11111111-1111-1111-1111-111111111111"  # 订单命名空间 1111-1001..1004(conftest 同款)


@dataclass
class TurnSpec:
    text: str
    intent: str
    mode: str | None = None      # business 轮期望 None(new_turn_state 重置)
    suspended: bool = False
    anchors: list[str] = field(default_factory=list)  # resolved 须含其一(空=不查)
    resume: str | None = None    # 挂起后脚本内 resume 的订单号


SCRIPTS = [
    ("剧本1 物流→退款指代→物流切换", [
        TurnSpec("订单 1111-1001 到哪了", "物流"),
        TurnSpec("这个能退吗", "退款退货", mode="order_specific",
                 anchors=["1111-1001"]),      # 历史指代消解出订单号→唯一候选直通
        TurnSpec("顺便说下物流到哪了", "物流", anchors=["1111-1001"]),
    ]),
    ("剧本2 通用退款→个案挂起→resume", [
        TurnSpec("如何申请退款", "退款退货", mode="general"),
        TurnSpec("我要申请退款", "退款退货", mode="order_specific", suspended=True,
                 resume="1111-1001"),
    ]),
    ("剧本3 无号挂起→resume→active_order 消解", [
        TurnSpec("这个能退吗", "退款退货", mode="order_specific", suspended=True,
                 resume="1111-1001"),
        TurnSpec("这单到哪了", "物流", anchors=["1111-1001"]),
    ]),
]


async def _stream(service, turn) -> list:
    return [ev async for ev in service.stream(turn)]


def _tools(events) -> list[str]:
    return [e.name for e in events if isinstance(e, ToolStartEvent)]


def _text(events) -> str:
    return "".join(e.content for e in events if isinstance(e, DeltaEvent))


async def _view(graph, sid) -> dict:
    st = await graph.aget_state({"configurable": {"thread_id": sid, "user_id": USER_ID}})
    return st.values


async def run_turn(service, graph, sid, spec: TurnSpec) -> list[str]:
    fails: list[str] = []
    turn = await service.prepare(USER_ID, sid, spec.text)
    events = await _stream(service, turn)
    sel = next((e for e in events if isinstance(e, OrderSelectorEvent)), None)
    v = await _view(graph, sid)
    resolved, intent = v.get("resolved_query"), v.get("intent")
    conf, route, mode = v.get("intent_confidence"), v.get("route"), v.get("refund_mode")
    active = (v.get("active_order") or {}).get("order_id")
    print(f"  raw={spec.text!r}")
    print(f"    resolved={resolved!r}")
    print(f"    intent={intent}({conf}) route={route} mode={mode} "
          f"gate={v.get('retrieval_status')} 挂起={sel is not None}")
    print(f"    tools={_tools(events) or '无'} active_order={active or '无'}")
    print(f"    reply={_text(events)[:70]!r}")
    if intent != spec.intent:
        fails.append(f"intent 期望 {spec.intent!r} 实得 {intent!r}")
    if mode != spec.mode:
        fails.append(f"refund_mode 期望 {spec.mode!r} 实得 {mode!r}")
    if (sel is not None) != spec.suspended:
        fails.append(f"挂起 期望 {spec.suspended} 实得 {sel is not None}")
    if spec.anchors and not any(a in (resolved or "") for a in spec.anchors):
        fails.append(f"消解锚点 {spec.anchors} 未命中")
    if not any(isinstance(e, DoneEvent) for e in events):
        fails.append("本轮未正常收尾(缺 [DONE])")
    if spec.resume and sel is not None:
        rturn = await service.prepare_resume(USER_ID, sid, sel.interrupt_id, spec.resume)
        revents = await _stream(service, rturn)
        rv = await _view(graph, sid)
        ractive = (rv.get("active_order") or {}).get("order_id")
        print(f"    resume→{spec.resume}: tools={_tools(revents) or '无'} "
              f"active_order={ractive or '无'} reply={_text(revents)[:70]!r}")
        if any(isinstance(e, OrderSelectorEvent) for e in revents):
            fails.append("resume 后仍发 order_selector")
        if ractive != spec.resume:
            fails.append(f"resume 后 active_order 期望 {spec.resume!r} 实得 {ractive!r}")
        if not any(isinstance(e, DoneEvent) for e in revents):
            fails.append("resume 轮未正常收尾(缺 [DONE])")
    print(f"    {'PASS' if not fails else 'FAIL ' + '; '.join(fails)}")
    return fails


async def main() -> int:
    settings = Settings()
    engine = make_engine(settings.database_url)
    ping(engine)
    sf = make_session_factory(engine)
    model = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key,
                       base_url=settings.openai_base_url,
                       max_tokens=settings.max_output_tokens, stream_usage=True)
    embed = build_embeddings(settings)
    kb = MilvusKnowledgeStore(settings.milvus_uri, settings.embedding_dim)
    reranker = SiliconFlowReranker(settings) if settings.has_rerank_key() else None
    retriever = KnowledgeRetriever(settings, embed=embed, store=kb, session_factory=sf,
                                   model=model, reranker=reranker)
    store = InMemorySessionStore(settings.max_sessions, settings.max_messages_per_session,
                                 settings.max_message_chars)
    service = ChatService(store, model, settings, SERVICE_SYSTEM_PROMPT)
    deps = GraphDeps(model=model, settings=settings, retriever=retriever, store=store,
                     system_prompt=SERVICE_SYSTEM_PROMPT)
    graph = build_chat_graph(deps, InMemorySaver())
    service.set_graph(graph)
    print(f"model={settings.model_name} strategy={settings.knowledge_strategy} "
          f"expand={settings.refund_expand_enabled} rerank={'有' if reranker else '无'}")
    total = 0
    try:
        for title, turns in SCRIPTS:
            print("=" * 78)
            print(title)
            sid = await store.create(USER_ID)
            for spec in turns:
                total += len(await run_turn(service, graph, sid, spec))
        print("=" * 78)
        print(f"断言失败数:{total}")
    finally:
        retriever.close()
        engine.dispose()
    return total


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
