import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_settings


async def test_root_serves_chat_page():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="messages"' in resp.text
    assert "/v1/chat/stream" in resp.text
    assert "1784959384051.jpg" in resp.text


async def test_brand_mark_asset_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/1784959384051.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0
