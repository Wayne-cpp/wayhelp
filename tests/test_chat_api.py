import asyncio
import json

import httpx
import pytest

from app.main import create_app
from app.routers.chat import event_stream
from tests.conftest import (
    TEST_USER_ID,
    FakeStreamModel,
    UserBoundMemoryStore,
    make_runtime,
    make_settings,
)
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)
from tests.test_ch05_acceptance import (
    _FakeRetriever,
    _client,
    _make_app,
    _result,
    _turn,
    _types,
)


def make_app(script, **over):
    settings = make_settings(**over)
    return create_app(settings=settings, model=FakeStreamModel(script),
                      runtime=make_runtime(tools=[], store=UserBoundMemoryStore(
                          settings.max_sessions, settings.max_messages_per_session,
                          settings.max_message_chars)))


async def post_stream(app, payload):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        async with client.stream("POST", "/v1/chat/stream", json=payload) as resp:
            lines = [line async for line in resp.aiter_lines() if line]
            return resp.status_code, lines


def parse_frames(lines):
    frames = []
    for line in lines:
        assert line.startswith("data: "), line
        payload = line[len("data: "):]
        frames.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return frames


async def test_sse_headers():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        async with client.stream("POST", "/v1/chat/stream", json={"user_id": TEST_USER_ID, "message": "hi"}) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers["cache-control"] == "no-cache"


async def test_blank_message_422():
    app = make_app([])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={"user_id": TEST_USER_ID, "message": "  "})
        assert resp.status_code == 422


async def test_unknown_session_404():
    app = make_app([])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/v1/chat/stream",
            json={"user_id": TEST_USER_ID, "session_id": "00000000-0000-0000-0000-000000000000", "message": "hi"},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_not_found"


async def test_session_capacity_503():
    app = make_app(["ok"], max_sessions=1)
    await post_stream(app, {"user_id": TEST_USER_ID, "message": "占用唯一容量"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={"user_id": TEST_USER_ID, "message": "再来一个"})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "session_capacity_reached"


async def test_upstream_error_frame_no_done():
    app = make_app(["一半", RuntimeError("boom")])
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    frames = parse_frames(lines)
    assert frames[-1]["type"] == "error"
    assert frames[-1]["code"] == "upstream_error"
    assert "[DONE]" not in frames
    assert "boom" not in json.dumps(frames, ensure_ascii=False)


async def test_unhandled_exception_500_sanitized():
    app = make_app([])

    @app.get("/_boom")
    async def _boom():
        raise RuntimeError("boom")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/_boom")
        assert resp.status_code == 500
        assert resp.json() == {"error": {"code": "internal_error", "message": "服务内部错误"}}
        assert "boom" not in resp.text


async def test_aclose_before_iteration_releases_lock():
    app = make_app(["x"])
    service = app.state.chat_service
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    g = event_stream(service, turn)
    await g.aclose()  # 从未迭代
    assert turn.lock_key not in service._locks._locks


async def test_aclose_after_partial_iteration_releases_lock():
    app = make_app(["你", "好"])
    service = app.state.chat_service
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    g = event_stream(service, turn)
    first = await g.__aiter__().__anext__()  # 消费到第一个事件(session 帧)
    assert first.startswith("data: ")
    await g.aclose()
    assert turn.lock_key not in service._locks._locks


async def test_second_turn_carries_context():
    model = FakeStreamModel(["回答一"])
    app = create_app(settings=make_settings(), model=model,
                     runtime=make_runtime(tools=[]))
    _, lines1 = await post_stream(app, {"user_id": TEST_USER_ID, "message": "记住数字42"})
    sid = parse_frames(lines1)[0]["session_id"]
    await post_stream(app, {"user_id": TEST_USER_ID, "session_id": sid, "message": "我刚说的数字是?"})
    sent = model.received[1]
    texts = [m.content for m in sent]
    assert "记住数字42" in texts and "回答一" in texts


def test_app_starts_without_embedding_key(db_session_factory):
    from app.config import Settings
    from app.main import create_app
    settings = Settings(_env_file=None, **{
        "openai_base_url": "http://test/v1", "openai_api_key": "test-key",
        "model_name": "test-model",
        "database_url": Settings().test_database_url,
        "embedding_api_key": "",
    })
    app = create_app(settings=settings, model=object())  # 生产 runtime 路径
    assert app.state.chat_service is not None  # 缺 Key 也能启动


# ── ch06 Task 9:resume 404/409/并发测试矩阵(spec §9.1;端点代码 Task 7 已落地,此处补钉)──

OTHER_USER = "22222222-2222-2222-2222-222222222222"

# 挂起构造复用 test_refund_flow:轮1 classify(首轮 understand 跳过)→ scope → 挂起;
# resume 轮:refund_prepare 重跑(零模型)→ expand → main_agent 收尾
_SUSPEND_SCRIPTS = [
    ['{"intent":"退款退货","confidence":0.9}'],
    ['{"mode":"order_specific"}'],
    ['{"queries":[]}'],
    ["订单 1111-1001 在 7 天无理由期内,可以退。"],
]


async def test_resume_404_and_409_matrix(db_session_factory):
    app = _make_app(list(_SUSPEND_SCRIPTS), retriever=_FakeRetriever(_result()),
                    db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        assert _types(frames) == ["session", "order_selector", "[DONE]"]
        sel = frames[1]
        # 归属错误 → 404(统一响应,不泄露详情)
        r = await client.post("/v1/chat/resume", json={
            "user_id": OTHER_USER, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "session_not_found"
        # interrupt_id 不匹配 → 409
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": "deadbeef", "order_id": "1111-1001"})
        assert r.status_code == 409
        # 订单不在卡片候选 → 409
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-9999"})
        assert r.status_code == 409
        # 正确 resume → 200;旧卡立即失效(重放 → 409)
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert r.status_code == 200
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert r.status_code == 409


async def test_concurrent_resume_only_one_wins(db_session_factory):
    """并发 resume 同一挂起实例:只有一个消费成功,另一个在锁内重新校验后 409(spec §9.1)。"""
    # resume 轮已落库 → 后续新轮从 understand 罐头开始(闲聊固定话术无模型调用)
    scripts = _SUSPEND_SCRIPTS + [
        ['{"resolved_query": ""}'],
        ['{"intent":"闲聊","confidence":0.95}'],
    ]
    app = _make_app(scripts, retriever=_FakeRetriever(_result()),
                    db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        sel = frames[1]
        body = {"user_id": TEST_USER_ID, "session_id": sid,
                "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"}
        r1, r2 = await asyncio.gather(
            client.post("/v1/chat/resume", json=body),
            client.post("/v1/chat/resume", json=body))
        assert sorted([r1.status_code, r2.status_code]) == [200, 409]
        # 锁已释放:后续新消息可正常进入
        frames2, _ = await _turn(client, "你好", sid)
        assert "[DONE]" in _types(frames2)
