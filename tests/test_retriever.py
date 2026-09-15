import json

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.knowledge.ingest import vector_text
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.query_understanding import passthrough_plan
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
    """按关键词定向向量:「邮费」→ e0,「登录」→ e1,其余 → e3。
    种子块向量恰为 e0/e1,故「其余」方向与全部库存向量正交(低置信场景)。"""
    def embed_query(self, text):
        if "邮费" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "登录" in text:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]

    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]  # 检索测试不经此路径


class FakeReranker:
    """按 documents 下标逆序给分,模拟重排;fail=True 时返回降级。"""
    def __init__(self, fail=False): self._fail = fail
    def rerank(self, query, documents, top_n, deadline=None):
        from app.knowledge.reranker import RerankOutcome
        if self._fail:
            return RerankOutcome(False, [], "rerank_http_500")
        n = len(documents)
        return RerankOutcome(True, [(i, (n - i) / n) for i in range(n)][:top_n], None)


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "retriever.db"), dim=4)
    yield s
    s.close()


def _plan(q):
    return passthrough_plan(q)


def _seed_knowledge(sf, store):
    with sf() as s:
        rows = [
            KnowledgeChunk(category="商品FAQ", questions="运费怎么算",
                           answer="满 99 包邮,未满 8 元。", content_type="faq",
                           section_path="商品FAQ > 邮费怎么算",
                           source_doc="knowledge_docs/商品FAQ.md", chunk_index=1,
                           vectorize_status="done", vector_id="1"),
            KnowledgeChunk(category="账户", questions="忘记密码",
                           answer="登录页点忘记密码。", content_type="faq",
                           section_path="账户 > 忘记密码",
                           source_doc="knowledge_docs/商品FAQ.md", chunk_index=2,
                           vectorize_status="done", vector_id="2"),
        ]
        s.add_all(rows)
        s.flush()
        ids = [r.id for r in rows]
        s.commit()
    texts = [vector_text(r.category, r.questions, r.answer)
             for r in (rows[0], rows[1])]
    store.ensure_collection()
    store.upsert([
        (ids[0], [1.0, 0.0, 0.0, 0.0], texts[0], "faq"),
        (ids[1], [0.0, 1.0, 0.0, 0.0], texts[1], "faq"),
    ])
    return ids


def _retriever(settings, store, sf, embed=None, reranker=None):
    return KnowledgeRetriever(settings, embed=embed or FakeEmbeddings(),
                              store=store, session_factory=sf, reranker=reranker)


# ─── 四级管道行为(spec §5)────────────────────────────────────────────────────


def test_hybrid_rerank_pipeline(store, db_session_factory):
    ids = _seed_knowledge(db_session_factory, store)
    r = _retriever(make_settings(), store, db_session_factory, reranker=FakeReranker())
    res = r.search("邮费怎么算", query_plan=_plan("邮费怎么算"))
    assert res.requested_strategy == "hybrid_rerank" == res.effective_strategy
    assert res.hits and res.low_confidence is False
    assert res.hits[0].chunk_id == ids[0]
    assert res.hits[0].section_path is not None     # KnowledgeHit 补 section_path
    assert res.leg_counts["dense"] > 0 and res.leg_counts["bm25"] >= 0
    assert res.leg_counts["fused"] >= 1  # 融合腿至少回回了 Top-1


def test_rerank_degrade_falls_back_to_hybrid(store, db_session_factory):
    _seed_knowledge(db_session_factory, store)
    r = _retriever(make_settings(), store, db_session_factory,
                   reranker=FakeReranker(fail=True))
    res = r.search("邮费怎么算", query_plan=_plan("邮费怎么算"))
    assert res.effective_strategy == "hybrid" and res.note
    assert res.confidence_threshold == make_settings().hybrid_min_score  # 降级换阈值


def test_strategy_bm25_only(store, db_session_factory):
    _seed_knowledge(db_session_factory, store)
    r = _retriever(make_settings(), store, db_session_factory, reranker=FakeReranker())
    res = r.search("MH-LP100", strategy="bm25", query_plan=_plan("MH-LP100"))
    assert res.effective_strategy == "bm25" and res.leg_counts == {"bm25": len(res.hits)}


def test_zero_hits_low_confidence(store, db_session_factory):
    # 计划原文查询「登录」与种子 e1 块相似且 FakeReranker 恒给 top1=1.0,按原文
    # 不可红;改用「发票怎么开」(FakeEmbeddings 其余→e3,与库存向量全正交)+
    # dense 腿 + 显式阈值,验证「定向到不相似向量 → 低置信」的原意。
    _seed_knowledge(db_session_factory, store)
    r = _retriever(make_settings(), store, db_session_factory, reranker=FakeReranker())
    res = r.search("发票怎么开", strategy="dense", min_score=0.5,
                   query_plan=_plan("发票怎么开"))
    assert res.effective_strategy == "dense"
    assert res.low_confidence is True
    assert res.confidence_score < res.confidence_threshold


def test_unconfigured_note(db_session_factory):
    r = KnowledgeRetriever(make_settings(), embed=None,
                           session_factory=db_session_factory)
    res = r.search("x", query_plan=_plan("x"))
    assert res.note == "知识检索未配置" and res.hits == [] and res.low_confidence is True


# ─── 建库前置状态(spec §8 顺序:状态判断先于任何远程调用)──────────────────────


def test_missing_file_not_created(store):
    r = KnowledgeRetriever(make_settings(), embed=FakeEmbeddings(), store=store,
                           session_factory=None)
    res = r.search("邮费", query_plan=_plan("邮费"))
    assert res.hits == [] and res.note == "知识库尚未建立" and res.low_confidence is True
    assert store.file_exists() is False  # 缺文件时不创建文件/客户端


def test_missing_collection(store):
    store._cli()  # 建文件但不建集合
    store.close()

    class CountingEmbed(FakeEmbeddings):
        calls = 0

        def embed_query(self, text):
            type(self).calls += 1
            return super().embed_query(text)

    r = KnowledgeRetriever(make_settings(), embed=CountingEmbed(), store=store,
                           session_factory=None)
    res = r.search("邮费", query_plan=_plan("邮费"))
    assert res.hits == [] and res.note == "知识库尚未建立"
    assert CountingEmbed.calls == 0  # 集合未建:不打远程 embedding 调用(spec §8 顺序)


def test_disabled_without_key(db_session_factory, store):
    r = KnowledgeRetriever(make_settings(embedding_api_key=""), embed=None,
                           store=store, session_factory=db_session_factory)
    assert r.enabled is False
    res = r.search("邮费", query_plan=_plan("邮费"))
    assert res.hits == [] and res.note == "知识检索未配置"


# ─── query_faq 契约(T7 将改造 business.py,这里经遗留适配保活契约断言)────────


class _LegacyFaqRetriever:
    """T6 过渡:business.py 仍是 (hits, note) 二元组 + 阈值过滤的旧契约,
    把新 RetrievalResult 投影回旧形态;T7 工具层改造后随测试一并删除。"""

    def __init__(self, retriever):
        self._r = retriever

    def search(self, query, min_score=None):
        res = self._r.search(query, strategy="dense", min_score=min_score)
        hits = [h for h in res.hits if h.score >= res.confidence_threshold]
        return hits, res.note


def test_query_faq_contract_unchanged(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.5)
    _seed_knowledge(db_session_factory, store)
    retriever = _LegacyFaqRetriever(
        _retriever(settings, store, db_session_factory))
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
    retriever = _LegacyFaqRetriever(
        _retriever(settings, store, db_session_factory))
    faq = next(t for t in build_tools(db_session_factory, 1, retriever)
               if t.name == "query_faq")
    out = json.loads(faq.invoke({"keyword": "完全不沾边的问题xyz"}))
    assert out["results"] == [] and out["note"]


# ─── 异常分级(_as_retryable 语义不变)─────────────────────────────────────────


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
