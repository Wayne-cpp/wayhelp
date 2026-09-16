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


def test_nav_has_rag_eval():
    html = KB_HTML.read_text(encoding="utf-8")
    assert "/rag-eval" in html


def test_kb_page_api_hooks():
    html = KB_HTML.read_text(encoding="utf-8")
    for path in ("/kb/api/state", "/kb/api/manual/preview", "/kb/api/manual/ingest",
                 "/kb/api/build", "/kb/api/preview", "/kb/api/mine",
                 "/kb/api/vectorize", "/kb/api/reset", "/kb/api/search"):
        assert path in html
    assert html.count("confirm(") >= 2  # build 与 reset 二次确认
    assert 'id="manual-vectorize"' in html        # 顺手向量化勾选框
    assert 'id="manual-vectorize-hint"' in html   # 无 key 禁用提示


def test_kb_page_rebuild_and_strategy():
    html = KB_HTML.read_text(encoding="utf-8")
    assert "/kb/api/rebuild" in html              # 重建索引(全量重置)入口
    assert "knowledge_state" in html              # 闸条区状态卡
    assert 'id="rebuild-btn"' in html and "全量重置" in html
    assert "清空全部知识块" in html                # 重建文案须含清空警示
    for opt in ('value="dense"', 'value="bm25"', 'value="hybrid"',
                'value="hybrid_rerank"'):
        assert opt in html                        # strategy 下拉四策略
    assert 'id="search-strategy"' in html and 'id="search-scope"' in html
    for opt in ('value="faq"', 'value="policy"', 'value="product_spec"',
                'value="after_sales_manual"', 'value="qa_mined"', 'value="manual"'):
        assert opt in html                        # scope 下拉六枚举
    assert "rebuilding" in html                   # rebuilding 时写按钮禁用
    assert "data-write" in html
