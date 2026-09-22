import pytest

from app.errors import SessionCapacityReachedError
from app.sessions import (
    CommitTurnResult, ContextMeta, ConversationItem, ConversationMessage,
    InMemorySessionStore, LowConfidenceRecord, PersistedTurn, StoredMessage, validate_turn,
)
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


@pytest.mark.xfail(reason="ch07 Task 4:validate_turn 停收 tool 行;本用例的 tool 行 envelope "
                         "契约随 Task 10 _to_stored 定稿后改写摘除", raises=ValueError)
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


# ---- ch07 Task 4:逐行 message_ids / validate_turn 停收 tool 行 / 锚点与摘要 ----

def _store() -> InMemorySessionStore:
    return InMemorySessionStore(max_sessions=10, max_messages_per_session=100,
                                max_message_chars=8000, max_tool_calls_per_turn=5)


def _turn() -> list[StoredMessage]:
    return [StoredMessage("user", "你好"),
            StoredMessage("assistant", None, tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("assistant", "在的")]


@pytest.mark.asyncio
async def test_commit_returns_per_row_ids():
    s = _store()
    sid = await s.create("u1")
    r = await s.commit_turn(sid, _turn())
    assert r.source_message_id == r.message_ids[0] and len(r.message_ids) == 3
    assert len(set(r.message_ids)) == 3


def test_validate_turn_rejects_tool_rows():
    bad = [StoredMessage("user", "hi"),
           StoredMessage("assistant", None, tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
           StoredMessage("tool", "{}", tool_call_id="c1"),
           StoredMessage("assistant", "好")]
    with pytest.raises(ValueError, match="tool"):
        validate_turn(bad, 5)
    validate_turn(_turn(), 5)  # 无 tool 行的新形态合法


@pytest.mark.asyncio
async def test_memory_anchor_methods():
    s = _store()
    sid = await s.create("u1")
    assert (await s.get_context_meta(sid, "u1")) == ContextMeta(None, None, None)
    r = await s.commit_turn(sid, _turn())
    last_id = int(r.message_ids[-1])
    assert await s.move_layer1_from(sid, "u1", last_id) is True     # 末条是完整轮边界
    assert await s.move_layer1_from(sid, "u1", last_id) is False    # 相等不前移 → False


@pytest.mark.asyncio
async def test_memory_summary_append_and_projection():
    s = _store()
    sid = await s.create("u1")
    r = await s.commit_turn(sid, _turn())
    upto = int(r.message_ids[-1])
    assert await s.move_layer1_from(sid, "u1", upto)
    res = await s.append_summary(sid, 0, upto, "用户问过订单A1001退款", 10_000)
    assert res.applied and res.seq == 1
    meta = await s.get_context_meta(sid, "u1")
    assert meta.summary_upto == upto and "A1001" in (meta.summary or "")
    again = await s.append_summary(sid, 0, upto, "重复提交", 10_000)
    assert not again.applied and again.reason                         # CAS:from_id 已前移


@pytest.mark.asyncio
async def test_memory_list_and_span():
    s = _store()
    sid = await s.create("u1")
    await s.commit_turn(sid, _turn())
    convs = await s.list_conversations("u1")
    assert len(convs) == 1 and convs[0].preview == "你好" and convs[0].summarized is False
    msgs = await s.list_messages(sid, "u1")
    # spec §6:list_messages 是前端回放 DTO,过滤 content=NULL 的工具调用 assistant 行
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert await s.list_messages(sid, "other") is None                # 归属不符 → None
    rows = await s.fetch_span_texts(sid, 0, 10**9)
    assert [r[1] for r in rows] == ["user", "assistant", "assistant"]  # 摘要取材不过滤 content
    recs = await s.list_checkpoint_records(sid, "u1")
    assert recs is not None and len(recs) == 3
