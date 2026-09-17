import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.knowledge.ingest import vector_text
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.query_understanding import QueryPlan, passthrough_plan
from app.knowledge.retriever import (
    KnowledgeHit, KnowledgeRetriever, RetryableKnowledgeError, _as_retryable,
    confidence_from_scores,
)
from app.models import KnowledgeChunk
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


# ─── 闸门置信信号(四策略评估选拔,胜者冻结进配置)─────────────────────────────


def test_confidence_from_scores_math():
    assert confidence_from_scores([], "top1") is None
    assert confidence_from_scores([0.5], "top1") == 0.5
    assert confidence_from_scores([0.5], "margin12") == 0.5       # 单命中:s1 按 0
    assert confidence_from_scores([0.6, 0.4, 0.2], "margin12") == pytest.approx(0.2)
    assert confidence_from_scores([0.6, 0.4], "product") == pytest.approx(0.12)
    assert confidence_from_scores([0.6, 0.6, 0.6, 0.6, 0.6], "ratio5") == 1.0
    assert confidence_from_scores([0.5, 0.25, 0.25, 0.0, 0.0],
                                  "ratio5") == pytest.approx(2.5)  # 0.5 / mean(1.0/5)
    assert confidence_from_scores([0.0, 0.0], "ratio5") is None   # 全 0 分布不可比
    with pytest.raises(ValueError):
        confidence_from_scores([0.5], "nope")


def test_confidence_signal_applies_only_to_hybrid_rerank(store, db_session_factory):
    _seed_knowledge(db_session_factory, store)
    s = make_settings(rerank_confidence_signal="margin12")
    r = _retriever(s, store, db_session_factory, reranker=FakeReranker())
    res = r.search("邮费怎么算", query_plan=_plan("邮费怎么算"))
    # FakeReranker 对 2 块给分 1.0/0.5 → margin12 = 0.5(非 top1 的 1.0)
    assert res.effective_strategy == "hybrid_rerank"
    assert res.confidence_score == pytest.approx(0.5)
    res_dense = r.search("邮费怎么算", strategy="dense", query_plan=_plan("邮费怎么算"))
    assert res_dense.confidence_score == res_dense.hits[0].score   # dense 臂不受信号配置影响


def test_confidence_signal_default_is_top1(store, db_session_factory):
    _seed_knowledge(db_session_factory, store)
    r = _retriever(make_settings(), store, db_session_factory, reranker=FakeReranker())
    res = r.search("邮费怎么算", query_plan=_plan("邮费怎么算"))
    assert res.confidence_score == res.hits[0].score == 1.0


# ─── 多意图支路(QueryPlan.sub_queries,仅 hybrid_rerank)───────────────────────


class IntentReranker:
    """按 query 与文档的关键词重合给分,模拟「重排只认当前 query 表达的意图」:
    query 含「故障」时只有故障文档高分,含「时限」时只有时限文档高分。"""
    def rerank(self, query, documents, top_n, deadline=None):
        from app.knowledge.reranker import RerankOutcome
        ranking = []
        for i, doc in enumerate(documents):
            hit = ("故障" in query and "故障" in doc) or ("时限" in query and "时限" in doc)
            ranking.append((i, 0.9 if hit else 0.1))
        ranking.sort(key=lambda x: -x[1])   # python sort 稳定:同分保持原序
        return RerankOutcome(True, ranking[:top_n], None)


def _seed_intent_knowledge(sf, store):
    with sf() as s:
        rows = [
            KnowledgeChunk(category="售后", questions="常见故障自查",
                           answer="报错可自查滚筒。", content_type="faq",
                           section_path="售后手册 > 常见故障自查",
                           source_doc="d.md", chunk_index=1,
                           vectorize_status="done", vector_id="i1"),
            KnowledgeChunk(category="售后", questions="常见问题处理时限",
                           answer="维修 7 个工作日。", content_type="faq",
                           section_path="售后手册 > 常见问题处理时限",
                           source_doc="d.md", chunk_index=2,
                           vectorize_status="done", vector_id="i2"),
        ]
        s.add_all(rows)
        s.flush()
        ids = [r.id for r in rows]
        s.commit()
    store.ensure_collection()
    store.upsert([(ids[0], [0.0, 0.0, 1.0, 0.0], "故障自查", "faq"),
                  (ids[1], [0.0, 0.0, 0.9, 0.1], "处理时限", "faq")])
    return ids


def test_multi_intent_subquery_boosts_second_intent(store, db_session_factory):
    ids = _seed_intent_knowledge(db_session_factory, store)
    r = _retriever(make_settings(), store, db_session_factory, reranker=IntentReranker())
    # 对照:无子查询 → 主问两意图关键词都不含,两榜同分按融合序,故障块在前
    plan_plain = QueryPlan("猫砂盆报错能自修吗维修要等几天", (), None, False, None)
    res0 = r.search("猫砂盆报错能自修吗维修要等几天", query_plan=plan_plain)
    assert res0.hits[0].chunk_id == ids[0] and res0.hits[0].score == pytest.approx(0.1)
    # 多意图:子查询「维修处理时限」分榜重排 → 时限块 0.9 升到榜首
    plan_sub = QueryPlan("猫砂盆报错能自修吗维修要等几天", (), None, False, None,
                         ("维修处理时限",))
    res = r.search("猫砂盆报错能自修吗维修要等几天", query_plan=plan_sub)
    assert res.hits[0].chunk_id == ids[1] and res.hits[0].score == pytest.approx(0.9)
    assert res.confidence_score == pytest.approx(0.9)   # top1 置信分取合并后榜首
    assert res.leg_counts["sub_queries"] == 1
    assert res.effective_strategy == "hybrid_rerank"


def test_merge_sub_intents_pins_low_score_into_topn(store, db_session_factory):
    """子意图榜头名分数很低、跌出 top_n 时,保底从榜尾换入(不置顶污染 top1)。"""
    ids = _seed_intent_knowledge(db_session_factory, store)

    class _LowReranker:
        def rerank(self, query, documents, top_n, deadline=None):
            from app.knowledge.reranker import RerankOutcome
            return RerankOutcome(True, [(i, 0.05 - i * 0.01)
                                        for i in range(len(documents))][:top_n], None)

    r = _retriever(make_settings(), store, db_session_factory, reranker=_LowReranker())
    main = [KnowledgeHit(9000 + i, 0.9 - i * 0.01, "c", "q", "a", None, None, None)
            for i in range(12)]                        # 12 块高分主榜,无需落库
    sub_raws = [("子查询", [(ids[0], 9.9), (ids[1], 9.8)])]  # 两意图块低分
    merged = r._merge_sub_intents(main, sub_raws, None)
    top_ids = [h.chunk_id for h in merged[:10]]
    assert ids[0] in top_ids and ids[1] in top_ids      # 都被保底拉回 top_n
    assert merged[0].chunk_id == 9000                    # 榜首仍是主榜最高分
    assert [h.chunk_id for h in merged[:8]] == [9000 + i for i in range(8)]
    assert {merged[8].chunk_id, merged[9].chunk_id} == set(ids)   # 钉在榜尾区段


def test_sub_intent_rerank_failure_keeps_main_ranking(store, db_session_factory):
    _seed_intent_knowledge(db_session_factory, store)

    class _SubFailReranker:
        def __init__(self): self.calls = 0
        def rerank(self, query, documents, top_n, deadline=None):
            from app.knowledge.reranker import RerankOutcome
            self.calls += 1
            if self.calls == 1:   # 主榜正常;子意图臂失败
                return RerankOutcome(True, [(i, 1.0 - i * 0.1)
                                            for i in range(len(documents))], None)
            return RerankOutcome(False, [], "rerank_http_500")

    r = _retriever(make_settings(), store, db_session_factory, reranker=_SubFailReranker())
    plan = QueryPlan("猫砂盆报错能自修吗维修要等几天", (), None, False, None,
                     ("维修处理时限",))
    res = r.search("猫砂盆报错能自修吗维修要等几天", query_plan=plan)
    assert res.effective_strategy == "hybrid_rerank"     # 子臂失败不降级
    assert [h.score for h in res.hits] == [pytest.approx(1.0), pytest.approx(0.9)]
