import pytest

from app.errors import SessionCapacityReachedError
from app.sessions import InMemorySessionStore, StoredMessage


def make_store(**kw):
    defaults = dict(max_sessions=3, max_messages_per_session=4, max_message_chars=100)
    defaults.update(kw)
    return InMemorySessionStore(**defaults)


def test_create_and_exists():
    s = make_store()
    sid = s.create()
    assert s.exists(sid)
    assert not s.exists("00000000-0000-0000-0000-000000000000")


def test_snapshot_is_readonly_copy():
    s = make_store()
    sid = s.create()
    s.commit_turn(sid, "问", "答")
    snap = s.snapshot(sid)
    assert snap == [StoredMessage("user", "问"), StoredMessage("assistant", "答")]
    snap.clear()
    assert len(s.snapshot(sid)) == 2


def test_commit_turn_appends_pair():
    s = make_store()
    sid = s.create()
    s.commit_turn(sid, "q1", "a1")
    s.commit_turn(sid, "q2", "a2")
    roles = [m.role for m in s.snapshot(sid)]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_drops_oldest_complete_turn_when_full():
    s = make_store(max_messages_per_session=4)
    sid = s.create()
    for i in range(4):
        s.commit_turn(sid, f"q{i}", f"a{i}")
    msgs = s.snapshot(sid)
    assert len(msgs) == 4
    assert [m.content for m in msgs] == ["q2", "a2", "q3", "a3"]


def test_capacity_reached_raises():
    s = make_store(max_sessions=2)
    s.create()
    s.create()
    with pytest.raises(SessionCapacityReachedError):
        s.create()


def test_sessions_isolated():
    s = make_store()
    a, b = s.create(), s.create()
    s.commit_turn(a, "q", "r")
    assert s.snapshot(b) == []


def test_commit_rejects_overlong_text():
    s = make_store(max_message_chars=5)
    sid = s.create()
    with pytest.raises(ValueError):
        s.commit_turn(sid, "123456", "ok")
