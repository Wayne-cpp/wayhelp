# tests/test_feedback.py
"""ch09 反馈:归属 404/非法目标 409/幂等/down 落池+回捞/同事务/并发撞键。"""

import asyncio
import dataclasses
import json

import httpx
import pytest

from app.errors import FeedbackConflictError, SessionNotFoundError
from app.main import create_app
from app.models import ChatFeedback, Conversation, LowConfidenceQuestion, Message
from app.services import feedback
from tests.conftest import TEST_USER_ID, FakeStreamModel, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)


def _seed_turn(sf, user_id="u1"):
    with sf() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        s.flush()
        u = Message(conversation_id=conv.id, role="user", content="怎么退货")
        s.add(u)
        s.flush()
        a = Message(conversation_id=conv.id, role="assistant", content="答复")
        s.add(a)
        s.commit()
        return conv.id, u.id, a.id


class _NoHistoryGraph:
    async def aget_state_history(self, config):
        return
        yield  # pragma: no cover(checkpoint 缺失 → missing)


class _Snap:
    def __init__(self, values):
        self.values = values


class _Msg:
    def __init__(self, db_id=None):
        self.additional_kwargs = {} if db_id is None else {"db_id": db_id}


class _HistoryGraph:
    """按给定顺序产出 checkpoint 快照(aget_state_history 新→旧)。"""

    def __init__(self, snaps):
        self._snaps = snaps

    async def aget_state_history(self, config):
        for snap in self._snaps:
            yield snap


_HITS = [{"chunk_id": 7, "score": 0.91, "section_path": "退货>流程",
          "questions": "怎么退货", "answer": "七天内可退"}]


def _submit(sf, graph, **kw):
    args = dict(settings=make_settings(), session_factory=sf, graph=graph,
                user_id="u1", on_pooled=None)
    args.update(kw)
    return asyncio.run(feedback.submit_feedback(**args))


def test_down_pools_question_and_feedback_atomically(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    pooled = []
    out = _submit(db_session_factory, _NoHistoryGraph(),
                  conversation_id=str(cid), assistant_message_id=str(aid),
                  sentiment="down", on_pooled=lambda: pooled.append(1))
    assert pooled == [1]  # 提交成功后才通知飞轮(spec §5.4)
    assert out["status"] == "recorded"
    with db_session_factory() as s:
        fb = s.query(ChatFeedback).first()
        lcq = s.query(LowConfidenceQuestion).first()
        assert fb.sentiment == "down"
        assert fb.low_confidence_question_id == lcq.id
        assert lcq.source == "user_feedback"
        assert lcq.raw_question == "怎么退货"
        assert lcq.retrieved_chunks is None       # checkpoint 缺失 → NULL 仍落池
        assert lcq.resolved_question is None
        assert lcq.turn_message_id == uid
        assert lcq.process_status == "pending"
        assert json.loads(lcq.reason) == {
            "assistant_message_id": aid, "turn_message_id": uid,
            "retrieval": "missing"}


def test_ownership_404_and_illegal_target_409(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    with pytest.raises(SessionNotFoundError):
        _submit(db_session_factory, _NoHistoryGraph(), user_id="u2",
                conversation_id=str(cid), assistant_message_id=str(aid),
                sentiment="down")
    with db_session_factory() as s:  # 中间工具调用 assistant 行:409
        mid = Message(conversation_id=cid, role="assistant", content=None,
                      tool_calls=[{"name": "t", "args": {}, "id": "x", "type": "tool_call"}])
        s.add(mid)
        s.commit()
        mid_id = mid.id
    with pytest.raises(FeedbackConflictError):
        _submit(db_session_factory, _NoHistoryGraph(),
                conversation_id=str(cid), assistant_message_id=str(mid_id),
                sentiment="down")
    with pytest.raises(FeedbackConflictError):  # 跨会话消息:409 不泄露他轮内容
        other_cid, _, other_aid = _seed_turn(db_session_factory, user_id="u9")
        _submit(db_session_factory, _NoHistoryGraph(),
                conversation_id=str(cid), assistant_message_id=str(other_aid),
                sentiment="down")
    with pytest.raises(SessionNotFoundError):  # 会话本身不存在:统一 404
        _submit(db_session_factory, _NoHistoryGraph(),
                conversation_id="999999", assistant_message_id=str(aid),
                sentiment="down")


def test_idempotent_same_sentiment_and_conflict_reverse(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    first = _submit(db_session_factory, _NoHistoryGraph(),
                    conversation_id=str(cid), assistant_message_id=str(aid),
                    sentiment="down")
    again = _submit(db_session_factory, _NoHistoryGraph(),
                    conversation_id=str(cid), assistant_message_id=str(aid),
                    sentiment="down")
    assert again["status"] == "duplicate"
    assert again["feedback_id"] == first["feedback_id"]
    with db_session_factory() as s:
        assert s.query(LowConfidenceQuestion).count() == 1   # 不产生重复 lcq
    with pytest.raises(FeedbackConflictError):
        _submit(db_session_factory, _NoHistoryGraph(),
                conversation_id=str(cid), assistant_message_id=str(aid),
                sentiment="up")


def test_up_records_without_lcq(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    out = _submit(db_session_factory, _NoHistoryGraph(),
                  conversation_id=str(cid), assistant_message_id=str(aid),
                  sentiment="up")
    assert out["status"] == "recorded"
    with db_session_factory() as s:
        assert s.query(ChatFeedback).count() == 1
        assert s.query(LowConfidenceQuestion).count() == 0


def test_concurrent_same_key_prior_returns_duplicate(db_session_factory, monkeypatch):
    """并发撞键:_load 不可见的抢先同向提交 → _write IntegrityError → prior 判定。
    修正点:此时必须报 duplicate(不是 recorded),不留多余 lcq,也不通知飞轮。"""
    cid, uid, aid = _seed_turn(db_session_factory)
    pooled = []

    async def _race_recover(graph, conversation_id, aid_, turn_id, top_n):
        # 模拟并发:回捞窗口内另一请求抢先提交同向反馈(同键同 sentiment)
        with db_session_factory() as s:
            s.add(ChatFeedback(conversation_id=int(conversation_id),
                               assistant_message_id=aid_, turn_message_id=turn_id,
                               sentiment="down"))
            s.commit()
        return {"chunks": None, "resolved": None, "retrieval": "missing"}

    monkeypatch.setattr(feedback, "_recover_snapshot", _race_recover)
    out = _submit(db_session_factory, _NoHistoryGraph(),
                  conversation_id=str(cid), assistant_message_id=str(aid),
                  sentiment="down", on_pooled=lambda: pooled.append(1))
    assert out["status"] == "duplicate"
    with db_session_factory() as s:
        prior = s.query(ChatFeedback).one()
        assert out["feedback_id"] == str(prior.id)
        assert s.query(LowConfidenceQuestion).count() == 0  # 撞键整体回滚
    assert pooled == []  # duplicate 不通知飞轮(spec §5.4)


def test_down_recovers_snapshot_from_matching_checkpoint(db_session_factory):
    """精确锚点回捞:跳过无锚点新轮,命中 (final_assistant_message_id, turn_message_id)
    且盖章 turn_messages 首尾相符的完成轮;跨轮累积 messages 即使含 db_id 也不看。"""
    cid, uid, aid = _seed_turn(db_session_factory)
    snaps = [
        _Snap({"final_assistant_message_id": None, "turn_message_id": None}),  # 新轮/旧格式
        _Snap({"final_assistant_message_id": aid, "turn_message_id": uid,
               "turn_messages": [_Msg(uid), _Msg(aid)],
               "messages": [_Msg(999)],  # 诱饵:累积 messages 含 db_id 也不得参与定位
               "retrieval_result": {"hits": _HITS}, "resolved_query": "退货流程"}),
    ]
    out = _submit(db_session_factory, _HistoryGraph(snaps),
                  conversation_id=str(cid), assistant_message_id=str(aid),
                  sentiment="down")
    assert out["status"] == "recorded"
    with db_session_factory() as s:
        lcq = s.query(LowConfidenceQuestion).one()
        assert [c["chunk_id"] for c in lcq.retrieved_chunks] == [7]
        assert lcq.retrieved_chunks[0]["score"] == 0.91
        assert lcq.retrieved_chunks[0]["answer"] == "七天内可退"
        assert lcq.resolved_question == "退货流程"
        assert json.loads(lcq.reason)["retrieval"] == "hit"


def test_down_stamp_mismatch_pools_null_snapshot(db_session_factory):
    """锚点相符但盖章 turn_messages 首尾不符 → mismatch,NULL 快照仍按账本原话落池。"""
    cid, uid, aid = _seed_turn(db_session_factory)
    snaps = [_Snap({"final_assistant_message_id": aid, "turn_message_id": uid,
                    "turn_messages": [_Msg(uid)],  # 尾未盖到最终回答行
                    "retrieval_result": {"hits": _HITS}, "resolved_query": "退货流程"})]
    _submit(db_session_factory, _HistoryGraph(snaps),
            conversation_id=str(cid), assistant_message_id=str(aid),
            sentiment="down")
    with db_session_factory() as s:
        lcq = s.query(LowConfidenceQuestion).one()
        assert lcq.retrieved_chunks is None
        assert lcq.resolved_question is None
        assert lcq.raw_question == "怎么退货"
        assert json.loads(lcq.reason)["retrieval"] == "mismatch"


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _make_app(db_session_factory):
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  session_factory=db_session_factory)
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=runtime)


def _post(client, **body):
    return client.post("/v1/chat/feedback", json=body)


async def test_feedback_endpoint_contract(db_session_factory):
    """HTTP 层错误契约:200 recorded/duplicate、反向 409、归属 404 不泄露、非法 id 422。"""
    cid, uid, aid = _seed_turn(db_session_factory, user_id=TEST_USER_ID)
    app = _make_app(db_session_factory)
    base = {"user_id": TEST_USER_ID, "conversation_id": str(cid),
            "assistant_message_id": str(aid)}
    async with await _client(app) as client:
        r = await _post(client, **base, sentiment="down")
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"
        fid = r.json()["feedback_id"]

        r2 = await _post(client, **base, sentiment="down")
        assert r2.status_code == 200
        assert r2.json() == {"status": "duplicate", "feedback_id": fid}

        r3 = await _post(client, **base, sentiment="up")
        assert r3.status_code == 409
        assert r3.json()["error"]["code"] == "feedback_conflict"

        r4 = await _post(client, **{**base, "user_id": "22222222-2222-2222-2222-222222222222"},
                         sentiment="down")
        assert r4.status_code == 404
        assert r4.json()["error"]["code"] == "session_not_found"
        assert "u1" not in r4.text and "u2" not in r4.text  # 不泄露归属细节

        r5 = await _post(client, **{**base, "assistant_message_id": "abc"},
                         sentiment="down")
        assert r5.status_code == 422
        assert r5.json()["error"]["code"] == "invalid_request"


def test_turn_written_by_store_is_feedback_eligible(db_session_factory):
    """验收回归:生产写路径(commit_turn)落库的最终回答行必须可反馈。

    显式 tool_calls=None 经默认 JSON 列落成 JSON null 而非 SQL NULL,
    _validate_turn/annotate 的 SQL is_(None) 全落空 → 线上 👎 一律 409。"""
    from app.sessions import StoredMessage
    from app.store_db import DbSessionStore

    async def _seed():
        store = DbSessionStore(db_session_factory, max_message_chars=8000)
        sid = await store.create("u1")
        r = await store.commit_turn(sid, [StoredMessage("user", "怎么退货"),
                                          StoredMessage("assistant", "答复")])
        return sid, r.message_ids
    sid, ids = asyncio.run(_seed())
    out = _submit(db_session_factory, _NoHistoryGraph(),
                  conversation_id=sid, assistant_message_id=ids[1],
                  sentiment="down")
    assert out["status"] == "recorded"
    with db_session_factory() as s:  # 钉死写入形态:无调用即 SQL NULL
        row = s.get(Message, int(ids[1]))
        assert row.tool_calls is None
        assert s.query(Message).filter(Message.id == int(ids[1]),
                                       Message.tool_calls.is_(None)).count() == 1
