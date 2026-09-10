import pytest

from app.errors import SessionCapacityReachedError
from app.sessions import InMemorySessionStore, StoredMessage
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
