import httpx

from app.main import create_app
from tests.conftest import TEST_USER_ID, FakeStreamModel, make_runtime, make_settings
from tests.test_chat_api import parse_frames, post_stream

TOOL_CHUNKS = [
    {"name": "query_order", "args": "{\"order_id\": \"1001\"}", "id": "call_1", "index": 0},
]


def make_app(script):
    return create_app(settings=make_settings(), model=FakeStreamModel(script),
                      runtime=make_runtime())


async def test_tool_frames_over_sse():
    app = make_app([("tool", TOOL_CHUNKS), ("then", ["答", "复"])])
    payload = {"user_id": TEST_USER_ID, "message": "查订单"}
    status, lines = await post_stream(app, payload)
    assert status == 200
    frames = parse_frames(lines)
    assert frames[0]["type"] == "session"
    assert frames[1]["type"] == "tool_start"
    assert frames[1]["name"] == "query_order" and frames[1]["args"] == {"order_id": "1001"}
    assert frames[1]["tool_call_id"] == "call_1"
    assert frames[2]["type"] == "tool_end" and frames[2]["ok"] is True
    assert frames[3]["type"] == "delta" and frames[3]["content"] == "答"
    assert frames[-1] == "[DONE]"


async def test_tool_end_summary_capped_80():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def verbose(x: str) -> str:
        """超长返回"""
        return "长" * 500

    app = create_app(settings=make_settings(), model=FakeStreamModel(
        [("tool", [{"name": "verbose", "args": "{\"x\": \"1\"}", "id": "c1", "index": 0}]),
         ("then", ["答"])]
    ), runtime=make_runtime(tools=[verbose]))
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    frames = parse_frames(lines)
    end = next(f for f in frames if isinstance(f, dict) and f.get("type") == "tool_end")
    assert len(end["summary"]) <= 80


async def test_invalid_user_id_422():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream",
                                 json={"user_id": "not-a-uuid", "message": "hi"})
        assert resp.status_code == 422


async def test_user_mismatch_404():
    app = make_app(["答"])
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    sid = parse_frames(lines)[0]["session_id"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": "22222222-2222-2222-2222-222222222222",
            "session_id": sid, "message": "hi"})
        assert resp.status_code == 404


async def test_bad_session_id_form_422():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": TEST_USER_ID, "session_id": "abc!!", "message": "hi"})
        assert resp.status_code == 422
