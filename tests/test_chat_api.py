import json

import httpx
import pytest

from app.main import create_app
from tests.conftest import FakeStreamModel, make_settings


def make_app(script, **over):
    return create_app(settings=make_settings(**over), model=FakeStreamModel(script))


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


async def test_sse_happy_path_frames():
    app = make_app(["你", "好"])
    status, lines = await post_stream(app, {"message": "在吗"})
    assert status == 200
    frames = parse_frames(lines)
    assert frames[0]["type"] == "session"
    assert [f["content"] for f in frames[1:-1]] == ["你", "好"]
    assert frames[-1] == "[DONE]"


async def test_sse_json_escaping():
    app = make_app(['带"引号"和\n换行'])
    _, lines = await post_stream(app, {"message": "在吗"})
    frames = parse_frames(lines)  # json.loads 不炸即转义正确
    assert frames[1]["content"] == '带"引号"和\n换行'


async def test_sse_headers():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        async with client.stream("POST", "/v1/chat/stream", json={"message": "hi"}) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers["cache-control"] == "no-cache"


async def test_blank_message_422():
    app = make_app([])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={"message": "  "})
        assert resp.status_code == 422


async def test_unknown_session_404():
    app = make_app([])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/v1/chat/stream",
            json={"session_id": "00000000-0000-0000-0000-000000000000", "message": "hi"},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "session_not_found"


async def test_session_capacity_503():
    app = make_app(["ok"], max_sessions=1)
    await post_stream(app, {"message": "占用唯一容量"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={"message": "再来一个"})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "session_capacity_reached"


async def test_upstream_error_frame_no_done():
    app = make_app(["一半", RuntimeError("boom")])
    status, lines = await post_stream(app, {"message": "hi"})
    frames = parse_frames(lines)
    assert frames[-1]["type"] == "error"
    assert frames[-1]["code"] == "upstream_error"
    assert "[DONE]" not in frames
    assert "boom" not in json.dumps(frames, ensure_ascii=False)


async def test_second_turn_carries_context():
    model = FakeStreamModel(["回答一"])
    app = create_app(settings=make_settings(), model=model)
    _, lines1 = await post_stream(app, {"message": "记住数字42"})
    sid = parse_frames(lines1)[0]["session_id"]
    await post_stream(app, {"session_id": sid, "message": "我刚说的数字是?"})
    sent = model.received[1]
    texts = [m.content for m in sent]
    assert "记住数字42" in texts and "回答一" in texts
