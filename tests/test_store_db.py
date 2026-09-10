import asyncio

import pytest

from app.store_db import DbSessionStore
from app.sessions import StoredMessage
from app.tool_envelope import wrap
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
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    calls = [{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"}]
    envelope = wrap("查到 1 条", True, None, 4000)
    await store.commit_turn(sid, [
        StoredMessage("user", "退货政策?"),
        StoredMessage("assistant", None, tool_calls=calls),
        StoredMessage("tool", envelope, tool_call_id="call_1"),
        StoredMessage("assistant", "退货政策是……"),
    ])
    snap = await store.snapshot(sid)
    assert [m.role for m in snap] == ["user", "assistant", "tool", "assistant"]
    assert snap[1].tool_calls[0]["name"] == "query_faq"
    assert snap[2].tool_call_id == "call_1" and snap[2].content == envelope


async def test_commit_rolls_back_on_failure(db_session_factory, monkeypatch):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)

    from sqlalchemy.orm import Session

    real_flush = Session.flush

    def boom(self):
        if getattr(self, "_boom_armed", False):
            raise RuntimeError("simulated mid-transaction failure")
        return real_flush(self)

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
