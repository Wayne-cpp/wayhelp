from pathlib import Path

import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings

HTML = Path(__file__).parent.parent / "app" / "static" / "topics.html"


async def test_topics_page_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/topics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_page_markers():
    html = HTML.read_text(encoding="utf-8")
    for marker in ('id="classify-btn"', 'id="stat-total"', 'id="stat-classified"',
                   'id="stat-unclassified"', 'id="meta-line"', 'id="topics-chart"',
                   'id="state-banner"',
                   "/api/topics/classify", "/api/topics/distribution",
                   "按用户轮次去重,多标签问题可计入多个类目",
                   "--accent: #c96442"):
        assert marker in html


def test_page_no_innerhtml_injection():
    assert "innerHTML" not in HTML.read_text(encoding="utf-8")
