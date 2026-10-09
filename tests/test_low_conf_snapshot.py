"""ch09 落池快照:Top N 截取/500 字符截断/三字段随 commit 落库(内存+DB 同语义)。"""
import json

from app.services.retrieval_snapshot import snapshot_top_chunks
from app.sessions import LowConfidenceRecord
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)


def _hit(i, answer="答"):
    return {"chunk_id": i, "score": 0.9 - i * 0.1, "section_path": f"s/{i}",
            "questions": f"问{i}", "answer": answer, "category": "c",
            "source_doc": None, "chunk_index": i}


def test_snapshot_top_n_and_truncation():
    hits = [_hit(i, answer="长" * 600) for i in range(1, 6)]
    snap = snapshot_top_chunks(hits, 3)
    assert [c["chunk_id"] for c in snap] == [1, 2, 3]
    assert snap[0]["question"] == "问1"
    assert len(snap[0]["answer"]) == 500
    assert snapshot_top_chunks([], 3) is None
    assert snapshot_top_chunks(None, 3) is None


def test_build_low_conf_carries_snapshot():
    from app.graph.nodes import _build_low_conf
    state = {"low_conf_source": "retrieval_low_conf", "raw_query": "能开专票吗",
             "resolved_query": "能否开具增值税专用发票",
             "turn_message_id": 77, "retrieval_status": "low_confidence",
             "low_conf_reason": {"top1": 0.01, "threshold": 0.5},
             "final_text": "", "evidence": [],
             "retrieval_result": {"hits": [_hit(1), _hit(2)]}}
    rec = _build_low_conf(state, 42, 3)
    assert rec.source == "retrieval_low_conf"
    assert rec.resolved_question == "能否开具增值税专用发票"
    assert rec.turn_message_id == 77
    assert [c["chunk_id"] for c in rec.retrieved_chunks] == [1, 2]
    json.dumps(rec.retrieved_chunks)  # 必须可 JSON 序列化


def test_commit_turn_writes_snapshot_columns(db_session_factory):
    from app.store_db import DbSessionStore
    from app.models import LowConfidenceQuestion
    import asyncio

    store = DbSessionStore(db_session_factory, 8000)

    async def go():
        sid = await store.create("u1")
        await store.commit_turn(
            sid,
            [__import__("app.sessions", fromlist=["StoredMessage"]).StoredMessage("user", "q"),
             __import__("app.sessions", fromlist=["StoredMessage"]).StoredMessage("assistant", "a")],
            low_confidence=LowConfidenceRecord(
                raw_question="q", source="self_check", reason="{}",
                conversation_id=int(sid),
                retrieved_chunks=[{"chunk_id": 1, "score": 0.5}],
                resolved_question="qq", turn_message_id=None))
        return sid

    asyncio.run(go())
    with db_session_factory() as s:
        row = s.query(LowConfidenceQuestion).first()
        assert row.resolved_question == "qq"
        assert row.retrieved_chunks == [{"chunk_id": 1, "score": 0.5}]
        assert row.process_status == "pending"   # 飞轮默认待处理


async def test_log_stamps_turn_anchors_in_state():
    """spec §5.3:log 提交成功后 turn_messages(带 db_id)与两锚点必须进 state 输出,
    不能只在局部变量盖章(用户反馈回捞按 checkpoint 锚点定位完成轮)。"""
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph

    from app.graph.nodes import GraphDeps, build_log_node
    from app.graph.state import ChatGraphState, new_turn_state
    from app.sessions import InMemorySessionStore
    from tests.conftest import make_settings

    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    uid = await store.append_user_message(sid, "你好")  # prepare 落库 → 行 1
    st = new_turn_state("你好", user_db_id=int(uid))
    st.update({"final_text": "您好", "route": "chitchat"})
    st["turn_messages"].append(AIMessage(content="您好"))
    g = StateGraph(ChatGraphState)  # writer 需图上下文,经编译图驱动(同 test_graph_nodes)
    g.add_node("log", build_log_node(GraphDeps(model=None, settings=make_settings(),
                                               retriever=None, store=store)))
    g.add_edge(START, "log")
    g.add_edge("log", END)
    out = await g.compile().ainvoke(st, config={"configurable": {"thread_id": sid}})
    assert out["turn_message_id"] == int(uid)       # 用户轮次锚点
    assert out["final_assistant_message_id"] == 2   # 本轮最后一条落库 assistant 行
    assert [m.additional_kwargs.get("db_id") for m in out["turn_messages"]] == [1, 2]
