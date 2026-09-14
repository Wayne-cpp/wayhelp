from pathlib import Path

import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings

KB_HTML = Path(__file__).parent.parent / "app" / "static" / "kb.html"


async def test_kb_page_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/kb")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="gauge"' in resp.text


def test_kb_page_six_sections_and_gauge():
    html = KB_HTML.read_text(encoding="utf-8")
    for marker in ('id="gauge"', 'id="sec-manual"', 'id="sec-materials"',
                   'id="sec-mining"', 'id="sec-vectorize"', 'id="sec-search"',
                   'id="sec-recent"'):
        assert marker in html
    assert "--accent: #c96442" in html  # 复用 chat 页设计 tokens


def test_kb_page_api_hooks():
    html = KB_HTML.read_text(encoding="utf-8")
    for path in ("/kb/api/state", "/kb/api/manual/preview", "/kb/api/manual/ingest",
                 "/kb/api/build", "/kb/api/preview", "/kb/api/mine",
                 "/kb/api/vectorize", "/kb/api/reset", "/kb/api/search"):
        assert path in html
    assert html.count("confirm(") >= 2  # build 与 reset 二次确认
    assert 'id="manual-vectorize"' in html        # 顺手向量化勾选框
    assert 'id="manual-vectorize-hint"' in html   # 无 key 禁用提示
