# tests/test_feedback_annotate.py
"""ch09 Task 8:历史消息反馈态标注——只有当轮最终回答 eligible,已反馈行带回 sentiment。"""

import asyncio

from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)


def test_annotate_marks_only_final_answers(db_session_factory):
    from tests.test_feedback import _seed_turn
    from app.services import feedback

    cid, uid, aid = _seed_turn(db_session_factory)
    asyncio.run(feedback.submit_feedback(
        settings=make_settings(), session_factory=db_session_factory, graph=None,
        user_id="u1", conversation_id=str(cid),
        assistant_message_id=str(aid), sentiment="up", on_pooled=None))
    msgs = [{"id": str(uid), "role": "user", "content": "怎么退货"},
            {"id": str(aid), "role": "assistant", "content": "答复"}]
    out = asyncio.run(asyncio.to_thread(
        feedback.annotate_messages, db_session_factory, cid, msgs))
    assert out[0]["feedback_eligible"] is False
    assert out[1]["feedback_eligible"] is True
    assert out[1]["feedback_sentiment"] == "up"


def test_annotate_multiturn_marks_each_final_and_keeps_other_sentiment_none(
        db_session_factory):
    """两轮各自末条 assistant 才 eligible;只反馈过的轮带 sentiment,其余 None。"""
    from tests.test_feedback import _seed_turn
    from app.services import feedback

    cid, uid, aid = _seed_turn(db_session_factory)
    with db_session_factory() as s:  # 第二轮:u2 → a2(前面还有一条纯文本中间 assistant)
        from app.models import Message
        u2 = Message(conversation_id=cid, role="user", content="还要多久")
        s.add(u2)
        s.flush()
        mid = Message(conversation_id=cid, role="assistant", content="中间思考")
        s.add(mid)
        s.flush()
        a2 = Message(conversation_id=cid, role="assistant", content="最终答复二")
        s.add(a2)
        s.commit()
        u2_id, mid_id, a2_id = u2.id, mid.id, a2.id
    asyncio.run(feedback.submit_feedback(
        settings=make_settings(), session_factory=db_session_factory, graph=None,
        user_id="u1", conversation_id=str(cid),
        assistant_message_id=str(a2_id), sentiment="down", on_pooled=None))
    msgs = [{"id": str(uid), "role": "user", "content": "怎么退货"},
            {"id": str(aid), "role": "assistant", "content": "答复"},
            {"id": str(u2_id), "role": "user", "content": "还要多久"},
            {"id": str(mid_id), "role": "assistant", "content": "中间思考"},
            {"id": str(a2_id), "role": "assistant", "content": "最终答复二"}]
    out = asyncio.run(asyncio.to_thread(
        feedback.annotate_messages, db_session_factory, cid, msgs))
    got = {m["id"]: (m["feedback_eligible"], m["feedback_sentiment"]) for m in out}
    assert got[str(aid)] == (True, None)      # 第一轮末条:eligible,未反馈
    assert got[str(mid_id)] == (False, None)  # 轮中非末条:不 eligible
    assert got[str(a2_id)] == (True, "down")  # 第二轮末条:eligible 且已反馈
