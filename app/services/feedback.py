"""ch09:👍👎 反馈——先校验账本(404/409),再尽力回捞 checkpoint 检索快照,
down 落池与反馈行同事务,唯一键 (conversation_id, assistant_message_id) 持久幂等。"""

import asyncio
import json
import logging

from sqlalchemy.exc import IntegrityError

from app.errors import FeedbackConflictError, SessionNotFoundError
from app.models import ChatFeedback, Conversation, LowConfidenceQuestion, Message
from app.services.retrieval_snapshot import snapshot_top_chunks

logger = logging.getLogger(__name__)


def _validate_turn(s, cid: int, aid: int, user_id: str) -> tuple[int, str]:
    """返回 (turn_message_id, raw_question);非法目标抛 409,归属失败抛 404。"""
    conv = s.get(Conversation, cid)
    if conv is None or conv.user_id != user_id:
        raise SessionNotFoundError("session not found")
    msg = s.get(Message, aid)
    if msg is None or msg.conversation_id != cid:
        raise FeedbackConflictError("message not in conversation")
    if (msg.role != "assistant" or not (msg.content or "").strip()
            or msg.tool_calls):
        raise FeedbackConflictError("not a final answer message")
    prev_user = (s.query(Message.id)
                 .filter(Message.conversation_id == cid, Message.role == "user",
                         Message.id < aid)
                 .order_by(Message.id.desc()).limit(1).first())
    if prev_user is None:
        raise FeedbackConflictError("no preceding user message")
    turn_id = prev_user[0]
    nxt_user = (s.query(Message.id)
                .filter(Message.conversation_id == cid, Message.role == "user",
                        Message.id > aid)
                .order_by(Message.id).limit(1).first())
    bound = (s.query(Message)
             .filter(Message.conversation_id == cid, Message.role == "assistant",
                     Message.id > turn_id,
                     Message.content.isnot(None), Message.tool_calls.is_(None)))
    bound = bound.filter(Message.id < nxt_user[0]) if nxt_user else bound
    last_final = bound.order_by(Message.id.desc()).first()
    if last_final is None or last_final.id != aid:
        raise FeedbackConflictError("not the final answer of its turn")
    raw = s.get(Message, turn_id)
    return turn_id, raw.content or ""


async def _recover_snapshot(graph, conversation_id: str, aid: int, turn_id: int,
                            top_n: int) -> dict:
    """精确匹配完成轮锚点回捞;任何缺失/异常 → 降级(NULL 快照仍落池)。

    只接受 final_assistant_message_id==aid 且 turn_message_id==turn_id 的完成轮,
    再校验已盖章 turn_messages 首尾 id;禁止在跨轮累积 state.messages 按 db_id
    定位(后续每轮都会含旧消息,会串轮)。"""
    config = {"configurable": {"thread_id": str(conversation_id)}}
    try:
        async for snap in graph.aget_state_history(config):
            vals = snap.values or {}
            if (vals.get("final_assistant_message_id") != aid
                    or vals.get("turn_message_id") != turn_id):
                continue
            stamped = [(m.additional_kwargs or {}).get("db_id")
                       for m in (vals.get("turn_messages") or [])]
            stamped = [int(x) for x in stamped if x is not None]
            if not stamped or stamped[0] != turn_id or stamped[-1] != aid:
                return {"chunks": None, "resolved": None, "retrieval": "mismatch"}
            hits = (vals.get("retrieval_result") or {}).get("hits")
            return {"chunks": snapshot_top_chunks(hits, top_n),
                    "resolved": vals.get("resolved_query") or None,
                    "retrieval": "hit" if hits else "not_run"}
    except Exception:
        logger.warning("feedback checkpoint recover failed conv=%s",
                       conversation_id, exc_info=True)
    return {"chunks": None, "resolved": None, "retrieval": "missing"}


def annotate_messages(session_factory, conversation_id: int,
                      messages: list[dict]) -> list[dict]:
    """给历史消息补 feedback_eligible/feedback_sentiment;无反馈列表演进为全 None。

    最终回答口径与 _validate_turn 一致:两个相邻 user 行之间(或末尾)最后一条
    合法 assistant(有正文、无 tool_calls,以账本行为准);id 均按 int 比较。"""
    with session_factory() as s:
        rows = (s.query(Message.id).filter(
            Message.conversation_id == conversation_id,
            Message.role == "assistant", Message.content.isnot(None),
            Message.tool_calls.is_(None)).order_by(Message.id).all())
        fbs = {r.assistant_message_id: r.sentiment
               for r in s.query(ChatFeedback)
               .filter_by(conversation_id=conversation_id).all()}
    final_ids = set()
    user_bounds = [int(m["id"]) for m in messages if m["role"] == "user"]
    bounds = sorted(user_bounds) + [1 << 62]
    for lo, hi in zip([0, *bounds[:-1]], bounds):
        cands = [i for i, in rows if lo < i < hi]
        if cands:
            final_ids.add(cands[-1])
    out = []
    for m in messages:
        mid = int(m["id"])
        out.append({**m,
                    "feedback_eligible": m["role"] == "assistant" and mid in final_ids,
                    "feedback_sentiment": fbs.get(mid)})
    return out


async def submit_feedback(*, settings, session_factory, graph, user_id: str,
                          conversation_id: str, assistant_message_id: str,
                          sentiment: str, on_pooled=None) -> dict:
    cid = int(conversation_id)
    aid = int(assistant_message_id)

    def _load():
        with session_factory() as s:
            turn_id, raw = _validate_turn(s, cid, aid, user_id)
            existing = (s.query(ChatFeedback)
                        .filter_by(conversation_id=cid, assistant_message_id=aid)
                        .first())
            return turn_id, raw, existing

    turn_id, raw, existing = await asyncio.to_thread(_load)
    if existing is not None:
        if existing.sentiment == sentiment:
            return {"status": "duplicate", "feedback_id": str(existing.id)}
        raise FeedbackConflictError("feedback already recorded with other sentiment")

    recovered = {"chunks": None, "resolved": None, "retrieval": "missing"}
    if sentiment == "down" and graph is not None:
        recovered = await _recover_snapshot(
            graph, conversation_id, aid, turn_id,
            settings.low_conf_snapshot_top_n)

    def _write() -> tuple[ChatFeedback, bool]:
        with session_factory() as s:
            try:
                lcq = None
                if sentiment == "down":
                    lcq = LowConfidenceQuestion(
                        conversation_id=cid, raw_question=raw, source="user_feedback",
                        reason=json.dumps(
                            {"assistant_message_id": aid, "turn_message_id": turn_id,
                             "retrieval": recovered["retrieval"]}, ensure_ascii=False),
                        retrieved_chunks=recovered["chunks"],
                        resolved_question=recovered["resolved"],
                        turn_message_id=turn_id)
                    s.add(lcq)
                    s.flush()
                fb = ChatFeedback(conversation_id=cid, assistant_message_id=aid,
                                  turn_message_id=turn_id, sentiment=sentiment,
                                  low_confidence_question_id=(
                                      lcq.id if lcq else None))
                s.add(fb)
                s.commit()   # down:lcq 与反馈行同事务,失败整体回滚
                return fb, True
            except IntegrityError:   # 并发同键:读既有记录判定,不留半条
                s.rollback()
                prior = (s.query(ChatFeedback)
                         .filter_by(conversation_id=cid, assistant_message_id=aid)
                         .first())
                if prior is not None and prior.sentiment == sentiment:
                    return prior, False
                raise FeedbackConflictError(
                    "feedback already recorded with other sentiment")

    fb, created = await asyncio.to_thread(_write)
    if created and sentiment == "down" and on_pooled is not None:
        on_pooled()   # 提交成功后才通知飞轮(spec §5.4);并发 prior 命中是 duplicate,不通知
    status = "recorded" if created else "duplicate"
    return {"status": status, "feedback_id": str(fb.id)}
