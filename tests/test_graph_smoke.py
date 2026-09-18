# tests/test_graph_smoke.py
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class _S(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    tag: str


async def test_langgraph_memory_checkpointer_roundtrip():
    def node(state):
        return {"tag": state["tag"] or "seen"}

    g = StateGraph(_S)
    g.add_node("node", node)
    g.add_edge(START, "node")
    g.add_edge("node", END)
    graph = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    out = await graph.ainvoke({"messages": [HumanMessage("hi")], "tag": ""}, config)
    assert out["tag"] == "seen"
    # 同 thread 第二次调用继承 messages(add_messages 累积)
    out2 = await graph.ainvoke({"tag": "again"}, config)
    assert len(out2["messages"]) == 1 and out2["tag"] == "again"


async def test_async_sqlite_saver_memory():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        g = StateGraph(_S)
        g.add_node("node", lambda s: {"tag": "sqlite"})
        g.add_edge(START, "node")
        g.add_edge("node", END)
        graph = g.compile(checkpointer=saver)
        out = await graph.ainvoke({"tag": ""}, {"configurable": {"thread_id": "t2"}})
        assert out["tag"] == "sqlite"
