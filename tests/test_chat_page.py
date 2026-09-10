import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings


async def test_root_serves_chat_page():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="messages"' in resp.text
    assert "/v1/chat/stream" in resp.text
    assert "1784959384051.jpg" in resp.text


async def test_brand_mark_asset_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/1784959384051.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


async def test_chat_page_tool_badge_hooks():
    from pathlib import Path
    html = (Path(__file__).parent.parent / "app" / "static" / "chat.html").read_text(encoding="utf-8")
    assert "wayhelp_user_id" in html
    assert "crypto.randomUUID" in html
    assert "tool_start" in html and "tool_end" in html
    assert "tool-badge" in html


def test_chat_page_settle_badge_safe_lookup():
    from pathlib import Path
    html = (Path(__file__).parent.parent / "app" / "static" / "chat.html").read_text(encoding="utf-8")
    # settleBadge 须用 Map 索引徽标:tool_call_id 含引号/反斜杠时拼 CSS 选择器会失效或注入
    assert "badgeById = new Map()" in html
    assert "badgeById.set(id, badge)" in html
    assert "badgeById.get(id)" in html
    # dataset 写入侧保持不变(赋值安全);旧的属性选择器字符串拼接不得复活
    assert "badge.dataset.toolCallId = id" in html
    assert '[data-tool-call-id="' not in html
