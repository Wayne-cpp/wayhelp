import asyncio

import httpx
import pytest

from app.main import AppRuntime, create_app
from app.store_db import DbSessionStore
from app.sessions import ContextMeta, StoredMessage
from app.tool_envelope import wrap
from tests.conftest import FakeStreamModel, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)

USER = "11111111-1111-1111-1111-111111111111"


def _turn(sid_msgs=None):
    return [StoredMessage("user", "问"), StoredMessage("assistant", "答")]


async def test_create_exists_snapshot_commit(db_session_factory):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    assert sid.isdigit()
    assert await store.exists(sid, USER) is True
    assert await store.exists(sid, "other-user") is False
    assert await store.snapshot(sid) == []
    await store.commit_turn(sid, _turn())
    snap = await store.snapshot(sid)
    assert [(m.role, m.content) for m in snap] == [("user", "问"), ("assistant", "答")]


async def test_tool_turn_roundtrip(db_session_factory):
    """ch07 定稿:tool 行不落库——带 tool 行的 commit 整轮 ValueError;
    纯工具调用步(assistant content=None + tool_calls)的新形态正常逐行落库。"""
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    calls = [{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"}]
    envelope = wrap("查到 1 条", True, None, 4000)
    with pytest.raises(ValueError, match="tool"):
        await store.commit_turn(sid, [
            StoredMessage("user", "退货政策?"),
            StoredMessage("assistant", None, tool_calls=calls),
            StoredMessage("tool", envelope, tool_call_id="call_1"),
            StoredMessage("assistant", "退货政策是……"),
        ])
    assert await store.snapshot(sid) == []  # 校验先于落库,整轮不进 DB
    r = await store.commit_turn(sid, [
        StoredMessage("user", "退货政策?"),
        StoredMessage("assistant", None, tool_calls=calls),
        StoredMessage("assistant", "退货政策是……"),
    ])
    assert len(r.message_ids) == 3
    snap = await store.snapshot(sid)
    assert [m.role for m in snap] == ["user", "assistant", "assistant"]
    assert snap[1].tool_calls[0]["name"] == "query_faq" and snap[1].content is None


async def test_commit_rolls_back_on_failure(db_session_factory, monkeypatch):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)

    from sqlalchemy.orm import Session

    real_flush = Session.flush

    # 实测整个 _commit_sync 只有一次 flush(execute(update) 触发的 autoflush;此后
    # session 已净,commit() 不再 flush),计划原文的"第二次引爆"永不满足 —— 在该次
    # flush 即引爆,同样构成事务中途失败,断言(无半个 turn)不变
    state = {"n": 0}

    def counting_flush(self):
        state["n"] += 1
        if state["n"] >= 1:
            raise RuntimeError("boom")
        return real_flush(self)

    monkeypatch.setattr(Session, "flush", counting_flush)
    with pytest.raises(RuntimeError):
        await store.commit_turn(sid, _turn())
    monkeypatch.undo()
    assert await store.snapshot(sid) == []  # 无半个 turn


async def test_invalid_turn_rejected(db_session_factory):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    with pytest.raises(ValueError):
        await store.commit_turn(sid, [StoredMessage("assistant", "答")])
    with pytest.raises(ValueError):
        # 孤儿 tool 消息
        await store.commit_turn(sid, [
            StoredMessage("user", "问"),
            StoredMessage("tool", wrap("x", True, None, 4000), tool_call_id="call_9"),
            StoredMessage("assistant", "答"),
        ])


async def test_survives_restart(db_session_factory):
    """模拟应用重启:新建 DbSessionStore 实例,同一会话仍可读写。"""
    store1 = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store1.create(USER)
    await store1.commit_turn(sid, _turn())
    store2 = DbSessionStore(db_session_factory, max_message_chars=8000)
    assert await store2.exists(sid, USER) is True
    assert len(await store2.snapshot(sid)) == 2


async def test_uuid_session_id_form_404_on_db_store(db_session_factory):
    """spec §4:规范 UUID 形态 session_id 打生产 DB adapter 按不存在处理(API 404,而非 500)。"""
    runtime = AppRuntime(
        store=DbSessionStore(db_session_factory, max_message_chars=8000),
        toolset_factory=lambda sid: [],
    )
    app = create_app(settings=make_settings(), model=FakeStreamModel(["答"]),
                     runtime=runtime)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": USER,
            "session_id": "00000000-0000-0000-0000-000000000000",
            "message": "hi",
        })
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_not_found"


async def test_uuid_session_id_treated_as_missing(db_session_factory):
    """内存 store 对未知会话 exists→False;DB store 对 UUID 形态(非自己创建的合法形态)对齐为不存在。"""
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = "00000000-0000-0000-0000-000000000000"
    assert await store.exists(sid, USER) is False
    assert await store.snapshot(sid) == []


async def test_decimal_beyond_bigint_treated_as_missing(db_session_factory):
    """schema 放行 19 位十进制,超出 BIGINT 正范围的同样按不存在处理,不查库。"""
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = "9223372036854775808"  # 2^63;[1-9]\d{0,18} 仍放行但越界
    assert await store.exists(sid, USER) is False
    assert await store.snapshot(sid) == []


def test_commit_turn_with_low_confidence(db_session_factory):
    from app.sessions import LowConfidenceRecord, StoredMessage
    from app.store_db import DbSessionStore
    from app.models import LowConfidenceQuestion
    store = DbSessionStore(db_session_factory, 8000)
    sid = asyncio.run(store.create("u1"))
    asyncio.run(store.commit_turn(sid, [
        StoredMessage("user", "能寄到日本吗"),
        StoredMessage("assistant", "抱歉,这个问题超出了我目前掌握的资料范围,已为您记录,稍后可转人工客服进一步核实。"),
    ], low_confidence=LowConfidenceRecord(
        raw_question="能寄到日本吗", source="retrieval_low_conf",
        reason='{"top1": 0.01}', conversation_id=int(sid))))
    with db_session_factory() as s:
        rows = s.query(LowConfidenceQuestion).all()
        assert len(rows) == 1 and rows[0].source == "retrieval_low_conf"
        assert rows[0].conversation_id == int(sid)


async def test_commit_turn_returns_db_user_message_id(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create(USER)
    r = await store.commit_turn(sid, _turn())
    assert r.source_message_id.isdecimal()
    # 返回的正是本轮 user 行 id:再查库核对
    from app.models import Message
    def _check():
        with db_session_factory() as s:
            msg = s.get(Message, int(r.source_message_id))
            assert msg is not None and msg.role == "user"
            assert msg.conversation_id == int(sid)
    await asyncio.to_thread(_check)


def test_check_ch07_tables_ok(db_engine):
    from app.db import check_ch07_tables
    check_ch07_tables(db_engine)  # 不抛即过


def test_check_ch07_tables_missing(db_engine):
    from sqlalchemy import text
    from app.db import check_ch07_tables
    with db_engine.connect() as conn:
        conn.execute(text("ALTER TABLE conversations DROP COLUMN layer1_from_msg_id"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch07-ddl"):
            check_ch07_tables(db_engine)
    finally:
        with db_engine.connect() as conn:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN layer1_from_msg_id "
                              "BIGINT UNSIGNED NULL AFTER summary_upto_msg_id"))
            conn.commit()


# ---- ch07 Task 5:逐行 ids / 只读查询 / 摘要与锚点事务 CAS ----
# helper 命名 _tool_turn 避让本文件既有 2 行式 _turn;新形态 = 中间纯工具调用步


def _tool_turn(user="你好", final="在的"):
    return [StoredMessage("user", user),
            StoredMessage("assistant", None,
                          tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("assistant", final)]


async def test_commit_turn_ids_and_no_tool_rows(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, _tool_turn())
    assert len(r.message_ids) == 3 and r.source_message_id == r.message_ids[0]
    assert [int(i) for i in r.message_ids] == sorted(int(i) for i in r.message_ids)


async def test_context_meta_and_move_layer1_from(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    assert (await store.get_context_meta(sid, "u1")) == ContextMeta(None, None, None)
    r1 = await store.commit_turn(sid, _tool_turn())
    r2 = await store.commit_turn(sid, _tool_turn("第二问", "第二答"))
    last1 = int(r1.message_ids[-1])
    mid = int(r1.message_ids[1])
    assert await store.move_layer1_from(sid, "u1", mid) is False      # 非完整轮边界
    assert await store.move_layer1_from(sid, "u1", last1) is True     # 第一轮末 = 边界
    assert await store.move_layer1_from(sid, "u1", last1) is False    # 不前移
    assert await store.move_layer1_from(sid, "u1", int(r2.message_ids[-1])) is True
    assert await store.move_layer1_from(sid, "other", last1) is False
    meta = await store.get_context_meta(sid, "u1")
    assert meta.layer1_from == int(r2.message_ids[-1])


async def test_append_summary_cas_and_projection(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r1 = await store.commit_turn(sid, _tool_turn())
    upto = int(r1.message_ids[-1])
    bad = await store.append_summary(sid, 0, upto, "段一", 10_000)
    assert not bad.applied and bad.reason == "beyond-layer1"          # layer1_from 未设
    await store.move_layer1_from(sid, "u1", upto)
    ok = await store.append_summary(sid, 0, upto, "用户问过订单A1001退款", 10_000)
    assert ok.applied and ok.seq == 1
    dup = await store.append_summary(sid, 0, upto, "重复", 10_000)
    assert not dup.applied and dup.reason == "anchor-moved"           # CAS 不回退
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary_upto == upto and "A1001" in meta.summary


async def test_list_conversations_and_messages(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid1 = await store.create("u1")
    await store.commit_turn(sid1, _tool_turn())
    sid2 = await store.create("u1")
    convs = await store.list_conversations("u1")
    assert [c.id for c in convs] == [sid2, sid1]                       # 新在前
    assert convs[1].preview == "你好" and convs[1].summarized is False
    msgs = await store.list_messages(sid1, "u1")
    # spec §6:list_messages 是前端回放 DTO,过滤 content=NULL 的工具调用 assistant 行
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert all(m.content for m in msgs)                                # content=NULL 已过滤
    assert await store.list_messages(sid1, "other") is None
    assert await store.list_conversations("other") == []


async def test_checkpoint_records_and_span(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, _tool_turn())
    recs = await store.list_checkpoint_records(sid, "u1")
    assert [x.role for x in recs] == ["user", "assistant", "assistant"]
    rows = await store.fetch_span_texts(sid, 0, int(r.message_ids[1]))
    assert [x[1] for x in rows] == ["user", "assistant"]
    assert await store.list_checkpoint_records(sid, "other") is None


async def test_summary_cascade_delete(db_session_factory):
    from sqlalchemy import text
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, _tool_turn())
    upto = int(r.message_ids[-1])
    await store.move_layer1_from(sid, "u1", upto)
    await store.append_summary(sid, 0, upto, "段一", 10_000)
    with db_session_factory() as s:
        # messages 的 FK(ch02)不带级联,先清消息行;本用例钉的是 ch07 summaries 的 CASCADE
        s.execute(text("DELETE FROM messages WHERE conversation_id = :cid"), {"cid": int(sid)})
        s.execute(text("DELETE FROM conversations WHERE id = :cid"), {"cid": int(sid)})
        s.commit()
        left = s.execute(text("SELECT COUNT(*) FROM conversation_summaries")).scalar()
    assert left == 0                                                   # 外键级联
