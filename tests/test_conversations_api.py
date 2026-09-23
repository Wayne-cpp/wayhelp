# tests/test_conversations_api.py
"""ch07 只读会话接口(spec §11):侧栏列表 + 历史回放,零写动作。

DB 版 harness 照抄 test_chat_action.py 先例(注入 session_factory → 裁决 B
换 DbSessionStore):计划钉的 "T" in created_at(ISO 8601)只有真 DB 行才有
datetime,内存 store 的 list_conversations 该列恒 None。异步写库 + 异步
httpx 客户端(仓库惯例,asyncio_mode=auto),免 asyncio.run 包装。"""

import dataclasses

import httpx

from app.main import create_app
from app.sessions import StoredMessage
from tests.conftest import FakeStreamModel, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)


def _make_app(db_session_factory):
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  session_factory=db_session_factory)
    # 模型永远不会被调用(测试只打只读 GET)
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=runtime)
    return app, app.state.store  # 裁决 B 后已是 DbSessionStore,路由同源


async def _seed_summarized_turn(store) -> str:
    """一会话一轮对话 + 层1前移 + 摘要投影,返回 session_id。"""
    sid = await store.create("u1")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001能退吗"),
                                      StoredMessage("assistant", "可以,政策允许")])
    upto = int(r.message_ids[-1])
    await store.move_layer1_from(sid, "u1", upto)
    await store.append_summary(sid, 0, upto, "用户问订单A1001退款", 10_000)
    return sid


async def test_conversations_empty(db_session_factory):
    app, _ = _make_app(db_session_factory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        r = await client.get("/api/conversations", params={"user_id": "u1"})
        assert r.status_code == 200 and r.json() == []


async def test_conversations_list_and_messages_flow(db_session_factory):
    app, store = _make_app(db_session_factory)
    sid = await _seed_summarized_turn(store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        lst = await client.get("/api/conversations", params={"user_id": "u1"})
        assert lst.status_code == 200
        item = lst.json()[0]
        assert item["id"] == sid and item["preview"] == "订单A1001能退吗"
        assert item["summarized"] is True and "T" in item["created_at"]  # ISO 8601
        msgs = await client.get(f"/api/conversations/{sid}/messages",
                                params={"user_id": "u1"})
        assert [m["role"] for m in msgs.json()] == ["user", "assistant"]
        assert msgs.json()[0]["content"] == "订单A1001能退吗"


async def test_conversations_messages_404_on_owner_mismatch(db_session_factory):
    app, store = _make_app(db_session_factory)
    sid = await _seed_summarized_turn(store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        nf = await client.get(f"/api/conversations/{sid}/messages",
                              params={"user_id": "other"})
        assert nf.status_code == 404  # 归属错误与不存在同形,不泄露
