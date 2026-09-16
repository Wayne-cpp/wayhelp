from pathlib import Path

import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings

HTML = Path(__file__).parent.parent / "app" / "static" / "rag-eval.html"


async def test_rag_eval_page_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/rag-eval")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="kpis"' in resp.text


def test_page_sections_and_hooks():
    html = HTML.read_text(encoding="utf-8")
    for marker in ('id="report-header"', 'id="gates-banner"', 'id="kpis"',
                   'id="retrieval"', 'id="generation"', 'id="summary"',
                   'id="ledger"', 'id="rerun"', 'id="log-window"',
                   'id="metric-tabs"', 'id="summary-table"', 'id="fabricated-details"',
                   'id="ledger-tabs"', 'id="ledger-pager"', 'id="dispose-modal"',
                   'id="run-btn"'):
        assert marker in html
    for path in ("/api/rag-eval/report", "/api/rag-eval/faith-cases",
                 "/api/jobs/eval-rag/run", "/api/jobs/eval-rag"):
        assert path in html
    assert "--accent: #c96442" in html      # 复用设计 tokens


def test_page_no_innerhtml_injection():
    """外部字符串只走 textContent/createTextNode;SVG 走 DOM API(规格:防存储型注入)。"""
    html = HTML.read_text(encoding="utf-8")
    assert "innerHTML" not in html
    assert "createElementNS" in html        # SVG 经 DOM API 创建
    assert "不可用" in html                  # null 展示分支
