import json

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.retriever import (
    KnowledgeRetriever, RetryableKnowledgeError, _as_retryable,
)
from app.models import KnowledgeChunk
from app.tools.business import build_tools
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

# 注:计划原文用 httpx 构造异常;锁定 openai 3.10.0 内部改 import httpx2
# (openai/_exceptions.py:6),两包构造签名一致已实测,此处用 httpx2 与 SDK 同源。


class FakeEmbeddings:
    """按关键词定向向量:「邮费」→ e1,「登录」→ e2,其余 → e3(与库存向量都不相似)。"""
    def embed_query(self, text):
        if "邮费" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "登录" in text:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]

    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]  # 检索测试不经此路径


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "retriever.db"), dim=4)
    yield s
    s.close()


def _seed_knowledge(sf, store):
    with sf() as s:
        rows = [
            KnowledgeChunk(category="商品FAQ", questions="运费怎么算",
                           answer="满 99 包邮,未满 8 元。", content_type="faq",
                           source_doc="knowledge_docs/商品FAQ.md", chunk_index=1,
                           vectorize_status="done", vector_id="1"),
            KnowledgeChunk(category="账户", questions="忘记密码",
                           answer="登录页点忘记密码。", content_type="faq",
                           source_doc="knowledge_docs/商品FAQ.md", chunk_index=2,
                           vectorize_status="done", vector_id="2"),
        ]
        s.add_all(rows)
        s.flush()
        ids = [r.id for r in rows]
        s.commit()
    store.ensure_collection()
    store.upsert([(ids[0], [1.0, 0.0, 0.0, 0.0]), (ids[1], [0.0, 1.0, 0.0, 0.0])])
    return ids


def _retriever(settings, store, sf, embed=None):
    return KnowledgeRetriever(settings, embed=embed or FakeEmbeddings(),
                              store=store, session_factory=sf)


def test_search_topk_threshold_order(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.5)
    ids = _seed_knowledge(db_session_factory, store)
    hits, note = _retriever(settings, store, db_session_factory).search("邮费是多少")
    assert note is None
    assert [h.chunk_id for h in hits] == [ids[0]]  # e2 方向被阈值滤掉
    assert hits[0].score >= 0.5
    assert hits[0].source_doc == "knowledge_docs/商品FAQ.md" and hits[0].chunk_index == 1


def test_query_faq_contract_unchanged(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.5)
    _seed_knowledge(db_session_factory, store)
    retriever = _retriever(settings, store, db_session_factory)
    tools = build_tools(db_session_factory, 1, retriever)
    faq = next(t for t in tools if t.name == "query_faq")
    out = json.loads(faq.invoke({"keyword": "邮费是多少"}))
    assert set(out) == {"results"}
    assert set(out["results"][0]) == {"question", "answer", "category"}
    assert out["results"][0]["answer"] == "满 99 包邮,未满 8 元。"
    out2 = json.loads(faq.invoke({"keyword": "登录"}))
    assert out2["results"] and "忘记密码" in out2["results"][0]["question"]


def test_query_faq_no_hit_note(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.99)
    _seed_knowledge(db_session_factory, store)
    retriever = _retriever(settings, store, db_session_factory)
    faq = next(t for t in build_tools(db_session_factory, 1, retriever)
               if t.name == "query_faq")
    out = json.loads(faq.invoke({"keyword": "完全不沾边的问题xyz"}))
    assert out["results"] == [] and out["note"]


def test_missing_file_not_created(store):
    settings = make_settings()
    r = KnowledgeRetriever(settings, embed=FakeEmbeddings(), store=store,
                           session_factory=None)
    hits, note = r.search("邮费")
    assert hits == [] and note == "知识库尚未建立"
    assert store.file_exists() is False  # 缺文件时不创建文件/客户端


def test_missing_collection(store):
    store._cli()  # 建文件但不建集合
    store.close()

    class CountingEmbed(FakeEmbeddings):
        calls = 0

        def embed_query(self, text):
            type(self).calls += 1
            return super().embed_query(text)

    settings = make_settings()
    r = KnowledgeRetriever(settings, embed=CountingEmbed(), store=store,
                           session_factory=None)
    hits, note = r.search("邮费")
    assert hits == [] and note == "知识库尚未建立"
    assert CountingEmbed.calls == 0  # 集合未建:不打远程 embedding 调用(spec §8 顺序)


def test_disabled_without_key(db_session_factory, store):
    settings = make_settings(embedding_api_key="")
    r = KnowledgeRetriever(settings, embed=None, store=store,
                           session_factory=db_session_factory)
    assert r.enabled is False
    hits, note = r.search("邮费")
    assert hits == [] and note == "知识检索未配置"


def test_probe_returns_all_topk_with_passed_flags(db_session_factory, store):
    """自测旁路:不按阈值过滤,压线命中带 passed=False;search 行为不变。"""
    settings = make_settings(knowledge_min_score=0.5)
    ids = _seed_knowledge(db_session_factory, store)
    r = _retriever(settings, store, db_session_factory)
    hits, note = r.probe("邮费是多少", top_k=2, min_score=0.99)
    assert note is None
    assert [h.chunk_id for h in hits] == [ids[0], ids[1]]  # Top-2 全回,含被阈值滤掉的
    assert hits[0].passed is True and hits[0].score >= 0.99
    assert hits[0].score >= hits[1].score
    assert hits[1].passed is False and hits[1].score < 0.99
    assert hits[1].source_doc == "knowledge_docs/商品FAQ.md"
    # 低阈值下同一查询两条都过线
    hits2, _ = r.probe("邮费是多少", top_k=2, min_score=0.0)
    assert all(h.passed for h in hits2)


def test_probe_note_semantics(db_session_factory, store):
    settings = make_settings(embedding_api_key="")
    r = KnowledgeRetriever(settings, embed=None, store=store,
                           session_factory=db_session_factory)
    hits, note = r.probe("邮费", top_k=5, min_score=0.6)
    assert hits == [] and note == "知识检索未配置"
    r2 = _retriever(make_settings(), store, db_session_factory)
    hits2, note2 = r2.probe("邮费", top_k=5, min_score=0.6)
    assert hits2 == [] and note2 == "知识库尚未建立"
    assert store.file_exists() is False  # 与 search 一样不创建文件


def test_retryable_mapping():
    req = httpx2.Request("POST", "http://x/v1/embeddings")
    assert _as_retryable(APIConnectionError(request=req)) is not None
    assert _as_retryable(APITimeoutError(request=req)) is not None
    resp429 = httpx2.Response(429, request=req)
    assert _as_retryable(RateLimitError("x", response=resp429, body=None)) is not None
    resp500 = httpx2.Response(500, request=req)
    assert _as_retryable(APIStatusError("x", response=resp500, body=None)) is not None
    resp401 = httpx2.Response(401, request=req)
    assert _as_retryable(APIStatusError("x", response=resp401, body=None)) is None
    assert _as_retryable(ValueError("bad")) is None


def test_executor_retries_retryable_knowledge_error():
    import asyncio
    from langchain_core.tools import tool
    from app.tools.executor import ToolExecutor, ToolRegistry

    calls = {"n": 0}

    @tool
    def flaky(q: str) -> str:
        """t"""
        calls["n"] += 1
        raise RetryableKnowledgeError("boom")

    ex = ToolExecutor(ToolRegistry([flaky]), timeout_seconds=5, max_retries=2,
                      max_result_chars=4000)
    # 注:计划原文 call dict 缺 "type": "tool_call",langchain 会把整个信封当
    # args 校验(缺 q → ValidationError,工具体不执行);补上信封字段,断言不变。
    outcome = asyncio.run(ex.execute({"name": "flaky", "args": {"q": "x"}, "id": "1",
                                      "type": "tool_call"}))
    assert outcome.record.ok is False
    assert outcome.record.error_code == "tool_unavailable"
    assert calls["n"] == 3  # 1 + 2 次重试


def test_executor_no_retry_on_plain_error():
    import asyncio
    from langchain_core.tools import tool
    from app.tools.executor import ToolExecutor, ToolRegistry

    calls = {"n": 0}

    @tool
    def broken(q: str) -> str:
        """t"""
        calls["n"] += 1
        raise ValueError("auth failed")

    ex = ToolExecutor(ToolRegistry([broken]), timeout_seconds=5, max_retries=2,
                      max_result_chars=4000)
    outcome = asyncio.run(ex.execute({"name": "broken", "args": {"q": "x"}, "id": "1",
                                      "type": "tool_call"}))
    assert outcome.record.error_code == "tool_error"
    assert calls["n"] == 1
