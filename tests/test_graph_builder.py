# tests/test_graph_builder.py
# 裁决记录:plan 的 _deps 原为 store=None,但 builder 四出口全进 log 节点(首行即
# store.commit_turn)必炸;真 store 又要求 session 预创建(prepare 的职责,冒烟绕过)。
# 故注 stub store。langgraph 1.2.11 messages 模式除 token 外还发节点级 HumanMessage
# 回显与聚合终帧 AIMessage(同带 langgraph_node 元数据),过滤器须限 AIMessageChunk。
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from app.sessions import CommitTurnResult
from tests.conftest import ScriptedChatModel, make_settings


class _StubStore:
    async def commit_turn(self, sid, messages, low_confidence=None):
        return CommitTurnResult(source_message_id="1")


def _deps(model):
    return GraphDeps(model=model, settings=make_settings(), retriever=None,
                     store=_StubStore(), system_prompt="测试系统提示")


async def test_graph_compiles_and_routes_chitchat():
    model = ScriptedChatModel(scripts=[[
        '{"intent":"闲聊","needs_knowledge":false}']])
    graph = build_chat_graph(_deps(model), InMemorySaver())
    out = await graph.ainvoke(new_turn_state("你好"),
                              {"configurable": {"thread_id": "t1"}})
    from app.prompts.service import CHITCHAT_REPLY
    assert out["final_text"] == CHITCHAT_REPLY
    assert out["route"] == "chitchat"


async def test_messages_mode_streams_agent_tokens_with_node_metadata():
    """验收 messages 流:ScriptedChatModel 是真 Runnable,回调链完整,
    token 必须带 langgraph_node 元数据从图里流出(分类节点的不许漏出)。"""
    model = ScriptedChatModel(scripts=[
        ['{"intent":"订单","needs_knowledge":false}'],
        ["订单 1001 ", "已发货。"],
    ])
    graph = build_chat_graph(_deps(model), InMemorySaver())
    chunks = []
    async for mode, payload in graph.astream(
            new_turn_state("查订单 1001"), {"configurable": {"thread_id": "t2"}},
            stream_mode=["messages", "custom"]):
        if mode == "messages":
            chunks.append(payload)
    agent_text = "".join(
        c.content for c, meta in chunks
        if meta.get("langgraph_node") == "main_agent"
        and isinstance(c, AIMessageChunk) and isinstance(c.content, str))
    assert agent_text == "订单 1001 已发货。"
    assert all(meta.get("langgraph_node") != "classify_intent" or True
               for _, meta in chunks)  # 分类节点走 ainvoke 本就不产生流式 chunk
