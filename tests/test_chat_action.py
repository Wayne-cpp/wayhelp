# tests/test_chat_action.py
"""动作端点测试:不依赖聊天链路,直接用 db fixtures 播种会话与消息行
(这样 Task 13 换图驱动后本文件无需改动)。"""

import asyncio
import dataclasses

import httpx
import pytest

from app.main import create_app
from app.models import Conversation, Message, Ticket
from tests.conftest import FakeStreamModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _make_app(db_session_factory):
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  session_factory=db_session_factory)
    # 模型永远不会被调用(测试只打 /v1/chat/action);Task 13 换图驱动后亦然
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=runtime)


def _seed_conversation(db_session_factory) -> tuple[str, str]:
    """播种一会话 + 一条 user 消息,返回 (session_id, source_message_id)。"""
    with db_session_factory() as s:
        conv = Conversation(user_id=TEST_USER_ID)
        s.add(conv)
        s.flush()
        msg = Message(conversation_id=conv.id, role="user",
                      content="我要投诉你们的服务")
        s.add(msg)
        s.flush()
        s.commit()
        return str(conv.id), str(msg.id)


async def test_action_creates_ticket_without_touching_conv_status(db_session_factory):
    sid, mid = _seed_conversation(db_session_factory)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": mid,
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticket_no"].startswith("T") and body["status"] == "待处理"

    def _check():
        with db_session_factory() as s:
            t = s.get(Ticket, body["ticket_no"])
            assert t.description == "我要投诉你们的服务" and t.ticket_type == "投诉"
            assert s.get(Conversation, int(sid)).status == "进行中"  # 不置已转人工
    await asyncio.to_thread(_check)


async def test_action_404_when_message_not_in_session(db_session_factory):
    sid, _ = _seed_conversation(db_session_factory)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": "999999",
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 404


async def test_action_404_on_role_or_owner_mismatch(db_session_factory):
    sid, mid = _seed_conversation(db_session_factory)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        # 会话归属不符
        resp = await client.post("/v1/chat/action", json={
            "user_id": "22222222-2222-2222-2222-222222222222", "session_id": sid,
            "source_message_id": mid, "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 404


async def test_action_404_on_assistant_message(db_session_factory):
    """source_message_id 指向 assistant 消息 → 404(只允许绑定本轮用户消息)。"""
    sid, _ = _seed_conversation(db_session_factory)
    with db_session_factory() as s:
        ai = Message(conversation_id=int(sid), role="assistant", content="您好")
        s.add(ai)
        s.flush()
        s.commit()
        ai_id = str(ai.id)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": ai_id,
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 404


async def test_action_422_on_bad_params(db_session_factory):
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": "not-a-uuid", "session_id": "1", "source_message_id": "abc",
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 422
