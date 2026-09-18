import pytest

from app.errors import SessionCapacityReachedError
from app.sessions import CommitTurnResult, InMemorySessionStore, LowConfidenceRecord, StoredMessage
from tests.conftest import TEST_USER_ID


def make_store(**kw):
    defaults = dict(max_sessions=3, max_messages_per_session=4, max_message_chars=100)
    defaults.update(kw)
    return InMemorySessionStore(**defaults)


def _pair(question: str, answer: str) -> list[StoredMessage]:
    return [StoredMessage("user", question), StoredMessage("assistant", answer)]


async def test_create_and_exists():
    s = make_store()
    sid = await s.create(TEST_USER_ID)
    assert await s.exists(sid, TEST_USER_ID)
    assert not await s.exists("00000000-0000-0000-0000-000000000000", TEST_USER_ID)


async def test_snapshot_is_readonly_copy():
    s = make_store()
    sid = await s.create(TEST_USER_ID)
    await s.commit_turn(sid, _pair("问", "答"))
    snap = await s.snapshot(sid)
    assert snap == [StoredMessage("user", "问"), StoredMessage("assistant", "答")]
    snap.clear()
    assert len(await s.snapshot(sid)) == 2


async def test_commit_turn_appends_pair():
    s = make_store()
    sid = await s.create(TEST_USER_ID)
    await s.commit_turn(sid, _pair("q1", "a1"))
    await s.commit_turn(sid, _pair("q2", "a2"))
    roles = [m.role for m in await s.snapshot(sid)]
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_drops_oldest_complete_turn_when_full():
    # limit = max(max_messages, max_tool_calls + 3);令两者相等(4)以保持原断言语义
    s = make_store(max_messages_per_session=4, max_tool_calls_per_turn=1)
    sid = await s.create(TEST_USER_ID)
    for i in range(4):
        await s.commit_turn(sid, _pair(f"q{i}", f"a{i}"))
    msgs = await s.snapshot(sid)
    assert len(msgs) == 4
    assert [m.content for m in msgs] == ["q2", "a2", "q3", "a3"]


async def test_capacity_reached_raises():
    s = make_store(max_sessions=2)
    await s.create(TEST_USER_ID)
    await s.create(TEST_USER_ID)
    with pytest.raises(SessionCapacityReachedError):
        await s.create(TEST_USER_ID)


async def test_sessions_isolated():
    s = make_store()
    a, b = await s.create(TEST_USER_ID), await s.create(TEST_USER_ID)
    await s.commit_turn(a, _pair("q", "r"))
    assert await s.snapshot(b) == []


async def test_commit_rejects_overlong_text():
    s = make_store(max_message_chars=5)
    sid = await s.create(TEST_USER_ID)
    with pytest.raises(ValueError):
        await s.commit_turn(sid, _pair("123456", "ok"))


async def test_commit_turn_records_low_confidence():
    s = make_store()
    sid = await s.create(TEST_USER_ID)
    rec = LowConfidenceRecord(raw_question="能寄到日本吗", source="retrieval_low_conf",
                              reason='{"top1": 0.01}', conversation_id=None)
    await s.commit_turn(sid, _pair("能寄到日本吗", "抱歉,超出了资料范围,已记录。"),
                        low_confidence=rec)
    assert s.low_confidence == [rec]
    # 不传(默认 None)不入池
    await s.commit_turn(sid, _pair("q", "a"))
    assert len(s.low_confidence) == 1


async def test_low_confidence_not_recorded_when_turn_invalid():
    """同成同败:turn 校验不过时低置信度记录不落,与 DB 侧事务语义对齐。"""
    s = make_store()
    sid = await s.create(TEST_USER_ID)
    rec = LowConfidenceRecord(raw_question="x", source="self_check",
                              reason=None, conversation_id=None)
    with pytest.raises(ValueError):
        await s.commit_turn(sid, [StoredMessage("assistant", "答")], low_confidence=rec)
    assert s.low_confidence == []


async def test_commit_turn_returns_source_message_id():
    s = InMemorySessionStore(10, 100, 8000)
    sid = await s.create("u")
    r1 = await s.commit_turn(sid, _pair("q1", "a1"))
    r2 = await s.commit_turn(sid, _pair("q2", "a2"))
    assert isinstance(r1, CommitTurnResult) and isinstance(r2, CommitTurnResult)
    assert r1.source_message_id != r2.source_message_id  # 稳定且递增的唯一 ID
    assert r1.source_message_id.isdecimal() and r2.source_message_id.isdecimal()


async def test_validate_turn_accepts_multiple_tool_groups():
    s = InMemorySessionStore(10, 100, 8000)
    sid = await s.create("u")
    await s.commit_turn(sid, [
        StoredMessage("user", "先查订单再查物流"),
        StoredMessage("assistant", None, tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", "env1", tool_call_id="c1"),
        StoredMessage("assistant", None, tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2", "type": "tool_call"}]),
        StoredMessage("tool", "env2", tool_call_id="c2"),
        StoredMessage("assistant", "订单已发货,派送中"),
    ])  # 不抛异常即通过


async def test_validate_turn_rejects_dangling_second_group():
    s = InMemorySessionStore(10, 100, 8000)
    sid = await s.create("u")
    with pytest.raises(ValueError):
        await s.commit_turn(sid, [
            StoredMessage("user", "q"),
            StoredMessage("assistant", None, tool_calls=[
                {"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("tool", "env1", tool_call_id="c1"),
            StoredMessage("assistant", None, tool_calls=[  # 第二组缺 ToolMessage
                {"name": "query_logistics", "args": {}, "id": "c2", "type": "tool_call"}]),
            StoredMessage("assistant", "答"),
        ])
