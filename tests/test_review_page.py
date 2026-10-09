"""ch09 /review 审核页 + rag-eval 趋势/成本区块的页面钉(Vibe 例外,字符串断言)。"""
from pathlib import Path

import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings

REVIEW_HTML = Path(__file__).parent.parent / "app" / "static" / "review.html"
RAG_EVAL_HTML = Path(__file__).parent.parent / "app" / "static" / "rag-eval.html"


async def test_review_page_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/review")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="status-tabs"' in resp.text


def test_review_page_tabs_and_list_columns():
    html = REVIEW_HTML.read_text(encoding="utf-8")
    for needle in ('id="status-tabs"', "待审", "写入中", "通过", "驳回", "处理失败",
                   "立即处理", "occurrence_count", "ai_suggested_answer",
                   "created_at"):
        assert needle in html


def test_review_page_api_hooks():
    html = REVIEW_HTML.read_text(encoding="utf-8")
    for needle in ("/api/review", "/detail", "/approve", "/reject",
                   "/api/flywheel/run", "/api/flywheel/failures",
                   "/api/flywheel/questions/", "/retry"):
        assert needle in html
    # 详情抽屉:归并原话 + 召回快照逐条原文/得分
    assert 'id="detail-drawer"' in html
    assert "raw_question" in html and "retrieved_chunks" in html and "score" in html
    # 通过流:可编辑核准答案(默认 AI 示例);写入中:冻结答案 + 错误 + 重试
    assert "approved_answer" in html and "last_write_error" in html


def test_rag_eval_page_trend_block_hooks():
    html = RAG_EVAL_HTML.read_text(encoding="utf-8")
    assert 'id="sec-trend"' in html and 'id="trend-chart"' in html   # 趋势区块 mount id
    assert "/api/eval-runs" in html
    # 分组断线键:版本三元组任一变化即新起一组
    for key in ("corpus_version", "dataset_version", "evidence_confidence_version"):
        assert key in html
    # 三条线:recall_at_10 / mrr / faithfulness
    for metric in ("recall_at_10", "mrr", "faithfulness"):
        assert metric in html


def test_rag_eval_page_cost_block_hooks():
    html = RAG_EVAL_HTML.read_text(encoding="utf-8")
    assert 'id="sec-cost"' in html and 'id="cost-chart"' in html
    assert "/api/stats/cost-by-intent" in html
    assert "Langfuse 未启用" in html       # 未启用时区块文案
    assert "missing_usage_observations" in html  # 缺 usage 计数可见
    assert "incomplete" in html            # 超上限截断可见
