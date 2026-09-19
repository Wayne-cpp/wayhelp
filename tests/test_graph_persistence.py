# tests/test_graph_persistence.py
"""SQLite 文件级持久化:关闭重开后跨轮历史仍在;失败轮临时消息不进历史。"""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from app.prompts.service import CHITCHAT_REPLY
from app.sessions import InMemorySessionStore
from tests.conftest import ScriptedChatModel, make_settings


def _deps(model, store):
    return GraphDeps(model=model, settings=make_settings(), retriever=None,
                     store=store, system_prompt="测试")


async def test_sqlite_checkpoint_survives_reopen(tmp_path):
    db = str(tmp_path / "cp.db")
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u1")  # 脚手架(Task 13 裁决①同因,经裁决 C):直接
    cfg = {"configurable": {"thread_id": sid}}  # ainvoke 不经 prepare,真 store 须预创建会话
    async with AsyncSqliteSaver.from_conn_string(db) as cp:
        graph = build_chat_graph(_deps(
            ScriptedChatModel(scripts=[['{"intent":"闲聊","needs_knowledge":false}']]),
            store), cp)
        await graph.ainvoke(new_turn_state("你好"), cfg)
    # 关闭后重开同一文件(第二轮有历史,开头多一段 understand 透传罐头)
    async with AsyncSqliteSaver.from_conn_string(db) as cp2:
        graph2 = build_chat_graph(_deps(
            ScriptedChatModel(scripts=[['{"resolved_query": ""}'],
                                       ['{"intent":"闲聊","needs_knowledge":false}']]),
            store), cp2)
        out = await graph2.ainvoke(new_turn_state("在吗"), cfg)
        texts = [m.content for m in out["messages"]]
        assert "你好" in texts and CHITCHAT_REPLY in texts  # 第一轮历史仍在
        assert texts.count(CHITCHAT_REPLY) == 2  # 两轮答复都进了历史


async def test_failed_turn_leaves_no_messages(tmp_path):
    db = str(tmp_path / "cp2.db")
    store = InMemorySessionStore(10, 100, 8000)
    cfg = {"configurable": {"thread_id": "s2"}}
    async with AsyncSqliteSaver.from_conn_string(db) as cp:
        graph = build_chat_graph(_deps(
            ScriptedChatModel(scripts=[[ConnectionError("down")]]), store), cp)
        with pytest.raises(Exception):
            await graph.ainvoke(new_turn_state("查订单"), cfg)
        state = await graph.aget_state(cfg)
        assert not state.values.get("messages")  # 失败轮不追加跨轮历史
