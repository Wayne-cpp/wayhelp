"""知识库管理后台 API(/kb/api/*)测试。

DB 端点走 dbfixtures(Docker MySQL wayhelp_test);store/embed 用内存 fake,
绝不真连 Milvus/SiliconFlow。
"""

import math

import httpx

from app.main import AppRuntime, create_app
from app.models import (
    Conversation,
    KnowledgeChunk,
    Message,
    QaExtractionStaging,
    QaMiningProgress,
)
from tests.conftest import FakeStreamModel, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class FakeKbStore:
    """内存版 Milvus store 协议:cosine 相似度确定性可算。
    upsert 兼容旧二元组(测试侧种子)与 T3 起的四元组 (id, vector, text, scope)。"""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.vectors: dict[int, list[float]] = {}
        self.texts: dict[int, str] = {}
        self.deleted: list[int] = []

    def file_exists(self):
        return True

    def has_collection(self):
        return True

    def ensure_collection(self):
        pass

    def upsert(self, rows):
        for row in rows:
            self.vectors[row[0]] = row[1]
            if len(row) > 2:
                self.texts[row[0]] = row[2]

    def all_ids(self):
        return set(self.vectors)

    def num_entities(self):
        return len(self.vectors)

    def delete_by_ids(self, ids):
        for i in ids:
            self.vectors.pop(i, None)
            self.texts.pop(i, None)
            self.deleted.append(i)

    @staticmethod
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def search_dense(self, vector, top_k, scope=None):
        scored = sorted(((i, self._cos(v, vector)) for i, v in self.vectors.items()),
                        key=lambda t: -t[1])
        return scored[:top_k]

    def search_bm25(self, text, top_k, scope=None):
        terms = [t for t in text.split() if t]
        scored = ((i, sum(1.0 for t in terms if t in self.texts.get(i, "")))
                  for i in self.vectors)
        hits = [(i, s) for i, s in scored if s > 0]
        return sorted(hits, key=lambda t: -t[1])[:top_k]

    def hybrid(self, vector, text, top_k, scope=None):
        # 朴素 RRF(k=60),与真实 store 的融合行为同构
        legs = [self.search_dense(vector, top_k), self.search_bm25(text, top_k)]
        fused: dict[int, float] = {}
        for leg in legs:
            for rank, (i, _) in enumerate(leg, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        return sorted(fused.items(), key=lambda t: -t[1])[:top_k]

    def close(self):
        pass


class FakeEmbed:
    """含「邮费」→ e0 轴,其余 → e1 轴(dim 1024)。"""

    def _v(self, text):
        v = [0.0] * 1024
        v[0 if "邮费" in text else 1] = 1.0
        return v

    def embed_query(self, text):
        return self._v(text)

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]


class BoomEmbed(FakeEmbed):
    def embed_documents(self, texts):
        raise RuntimeError("embedding api down")


def make_kb_app(settings=None, model=None, sf=None, store=None, embed=None,
                docs_dir=None):
    rt = make_runtime(tools=[])
    runtime = AppRuntime(store=rt.store, toolset_factory=rt.toolset_factory,
                         session_factory=sf, embed=embed, kb_store=store)
    app = create_app(settings=settings or make_settings(),
                     model=model or FakeStreamModel([]), runtime=runtime)
    if docs_dir is not None:
        app.state.kb_docs_dir = docs_dir
    return app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


DOC_A = """---
type: faq
---
# 商品FAQ

## 邮费怎么算

满 99 包邮,未满 8 元。
"""

DOC_B = """---
type: policy
---
# 售后政策

## 退货

七天无理由退货。
"""


def _write_docs(tmp_path, a=DOC_A, b=DOC_B):
    d = tmp_path / "docs"
    d.mkdir()
    if a is not None:
        (d / "a.md").write_text(a, encoding="utf-8")
    if b is not None:
        (d / "b.md").write_text(b, encoding="utf-8")
    return d


def _seed_chunk(sf, **kw):
    base = dict(category="c", questions="q", answer="a", content_type="faq",
                is_key_clause=False, vectorize_status="pending")
    base.update(kw)
    with sf() as s:
        row = KnowledgeChunk(**base)
        s.add(row)
        s.flush()
        s.commit()
        return row.id


# ─── state 聚合 ──────────────────────────────────────────────────────────────


async def test_state_aggregation(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    id1 = _seed_chunk(sf, category="商品FAQ", questions="邮费怎么算", answer="满 99 包邮。",
                      is_key_clause=True, vectorize_status="done",
                      source_doc="knowledge_docs/商品FAQ.md", chunk_index=1)
    id2 = _seed_chunk(sf, category="售后政策", questions="退货", answer="七天无理由。",
                      vectorize_status="done",
                      source_doc="knowledge_docs/退货政策.md", chunk_index=1)
    _seed_chunk(sf, category="对话挖掘", questions="保修", answer="一年。",
                content_type="qa_mined")  # pending,source_doc NULL
    store.upsert([(id1, [1.0] + [0.0] * 1023), (id2, [0.0, 1.0] + [0.0] * 1022)])
    with sf() as s:
        s.add_all([
            QaExtractionStaging(batch_no="b1", source_ref="conv:1",
                                question="q1", answer="a1", status="extracted"),
            QaExtractionStaging(batch_no="b1", source_ref="conv:2",
                                question="q2", answer="a2", status="kept"),
            QaExtractionStaging(batch_no="b2", source_ref="conv:3",
                                question="q3", answer="a3", status="discarded"),
        ])
        from datetime import datetime
        s.add_all([
            QaMiningProgress(conversation_id=1, batch_no="b1", qa_count=1,
                             extracted_at=datetime(2026, 9, 1)),
            QaMiningProgress(conversation_id=2, batch_no="b2", qa_count=0,
                             extracted_at=datetime(2026, 9, 2)),
        ])
        s.commit()
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.get("/kb/api/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["embedding_configured"] is True
    g = data["gauge"]
    assert g["mysql_count"] == 3 and g["pending_count"] == 1
    assert g["milvus_count"] == 2 and g["key_clause_count"] == 1
    assert g["consistent"] is False and g["ids_match"] is False
    assert g["pointers_ok"] is True and g["no_pending"] is False
    m = data["mining"]
    assert (m["extracted"], m["kept"], m["discarded"]) == (1, 1, 1)
    assert m["batches"] == 2
    assert len(m["items"]["kept"]) == 1 and m["items"]["kept"][0]["question"] == "q2"
    v = data["vectorize"]
    assert (v["pending"], v["done"], v["milvus"]) == (1, 2, 2)
    assert len(v["pending_samples"]) == 1
    recent = data["recent"]
    assert len(recent) == 3 and recent[0]["id"] > recent[-1]["id"]  # id 倒序
    assert recent[0]["content_type"] == "qa_mined"


async def test_state_consistent_when_aligned(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    id1 = _seed_chunk(sf, vectorize_status="done",
                      source_doc="knowledge_docs/a.md", chunk_index=1)
    store.upsert([(id1, [1.0] + [0.0] * 1023)])
    app = make_kb_app(sf=sf, store=store, embed=FakeEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        data = (await client.get("/kb/api/state")).json()
    g = data["gauge"]
    assert g["consistent"] is True and g["ids_match"] is True and g["no_pending"] is True
    assert data["embedding_configured"] is False  # make_settings 默认不带 key


# ─── 手工录入 ────────────────────────────────────────────────────────────────


async def test_manual_preview_pure_chunking():
    """preview 只走切块,不需要 DB/Milvus 依赖。"""
    app = make_kb_app()
    async with _client(app) as client:
        resp = await client.post("/kb/api/manual/preview", json={
            "doc_type": "policy", "title": "补充条款",
            "markdown": "## 时效\n\n签收后 7 天内。\n\n## 范围\n\n定制商品不支持。"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_count"] == 2
    c = data["chunks"][1]
    assert c["questions"] == "范围" and c["is_key_clause"] is True  # 命中「不支持」
    assert {"is_table", "has_overlap", "is_hard_cut"} <= set(c)


async def test_manual_preview_chunking_error_400():
    app = make_kb_app()
    async with _client(app) as client:
        resp = await client.post("/kb/api/manual/preview", json={
            "doc_type": "faq", "title": "没有二级标题", "markdown": "正文没有问答结构。"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


async def test_manual_preview_validation_422_error_contract():
    app = make_kb_app()
    async with _client(app) as client:
        resp = await client.post("/kb/api/manual/preview", json={
            "doc_type": "unknown", "title": "t", "markdown": "x"})
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"error"} and body["error"]["code"] == "invalid_request"


async def test_manual_ingest_roundtrip(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/manual/ingest", json={
            "doc_type": "faq", "title": "发票补充", "markdown": "## 发票怎么开\n\n下单后申请即可。",
            "vectorize": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["vectorized"] is True and len(data["chunk_ids"]) == 1
    cid = data["chunk_ids"][0]
    with sf() as s:
        row = s.query(KnowledgeChunk).filter_by(id=cid).one()
        assert row.source_doc is None and row.chunk_index is None
        assert row.prev_chunk_id is None and row.next_chunk_id is None  # 挖掘块惯例
        assert row.content_type == "faq" and row.vectorize_status == "done"
        assert row.vector_id == str(cid)
    assert store.all_ids() == {cid}  # MySQL id 与 Milvus pk 对齐


async def test_manual_ingest_without_vectorize_leaves_pending(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    app = make_kb_app(sf=sf, store=store, embed=FakeEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/manual/ingest", json={
            "doc_type": "policy", "title": "条款", "markdown": "## 退货\n\n七天无理由。",
            "vectorize": False})
    assert resp.status_code == 200
    with sf() as s:
        row = s.query(KnowledgeChunk).one()
        assert row.vectorize_status == "pending"
    assert store.all_ids() == set()


async def test_manual_ingest_vectorize_failure_leaves_pending(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=BoomEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/manual/ingest", json={
            "doc_type": "faq", "title": "发票补充", "markdown": "## 发票怎么开\n\n下单后申请。",
            "vectorize": True})
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "vectorize_failed"
    with sf() as s:
        row = s.query(KnowledgeChunk).one()
        assert row.vectorize_status == "pending"  # 块留 pending


# ─── embedding_not_configured 契约 ───────────────────────────────────────────


async def test_embedding_not_configured_contract():
    """向量化/挖掘/勾选向量化的手工录入:无 key → 503 embedding_not_configured。"""
    settings = make_settings(embedding_api_key="")
    app = make_kb_app(settings=settings, sf=object(), store=FakeKbStore(), embed=None)
    async with _client(app) as client:
        r1 = await client.post("/kb/api/vectorize")
        r2 = await client.post("/kb/api/mine")
        r3 = await client.post("/kb/api/manual/ingest", json={
            "doc_type": "faq", "title": "t", "markdown": "## q\n\na。", "vectorize": True})
    for resp in (r1, r2, r3):
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "embedding_not_configured"


async def test_kb_deps_unavailable():
    app = make_kb_app()  # runtime 不带 session_factory/kb_store
    async with _client(app) as client:
        resp = await client.get("/kb/api/state")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "kb_unavailable"


# ─── build / preview / vectorize / reset ─────────────────────────────────────


async def test_build_full_pipeline(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    docs = _write_docs(tmp_path)
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=docs)
    async with _client(app) as client:
        resp = await client.post("/kb/api/build")
    assert resp.status_code == 200
    data = resp.json()
    assert data["skipped_vectorize"] is False
    with sf() as s:
        rows = s.query(KnowledgeChunk).all()
        assert len(rows) == 2
        assert all(r.vectorize_status == "done" for r in rows)
        ids = {r.id for r in rows}
    assert store.all_ids() == ids


async def test_build_without_key_phase1_only(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    docs = _write_docs(tmp_path)
    app = make_kb_app(settings=make_settings(embedding_api_key=""),
                      sf=sf, store=store, embed=None, docs_dir=docs)
    async with _client(app) as client:
        resp = await client.post("/kb/api/build")
    assert resp.status_code == 200
    data = resp.json()
    assert data["skipped_vectorize"] is True and data["pending_count"] == 2
    with sf() as s:
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 2
    assert store.num_entities() == 0


async def test_build_changed_doc_conflict_409(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    docs = _write_docs(tmp_path)
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=docs)
    async with _client(app) as client:
        assert (await client.post("/kb/api/build")).status_code == 200
        (docs / "a.md").write_text(DOC_A + "\n## 新问答\n\n新内容。\n", encoding="utf-8")
        resp = await client.post("/kb/api/build")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "ingest_failed"


async def test_preview_diff_report(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    docs = _write_docs(tmp_path)
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=docs)
    async with _client(app) as client:
        r0 = await client.post("/kb/api/preview")
        assert {d["status"] for d in r0.json()["docs"]} == {"new"}
        assert (await client.post("/kb/api/build")).status_code == 200
        r1 = await client.post("/kb/api/preview")
        assert {d["status"] for d in r1.json()["docs"]} == {"unchanged"}
        (docs / "a.md").write_text(DOC_A + "\n## 新问答\n\n新内容。\n", encoding="utf-8")
        (docs / "c.md").write_text(DOC_B, encoding="utf-8")
        r2 = await client.post("/kb/api/preview")
    by_name = {d["name"]: d for d in r2.json()["docs"]}
    assert by_name["a.md"]["status"] == "changed"
    assert "需先 reset" in by_name["a.md"]["note"]
    assert by_name["b.md"]["status"] == "unchanged"
    assert by_name["c.md"]["status"] == "new"


async def test_vectorize_endpoint(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    _seed_chunk(sf, category="商品FAQ", questions="邮费怎么算", answer="满 99 包邮。")
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/vectorize")
    assert resp.status_code == 200
    assert resp.json()["pending_count"] == 0
    with sf() as s:
        row = s.query(KnowledgeChunk).one()
        assert row.vectorize_status == "done" and row.vector_id == str(row.id)
    assert store.all_ids() == {row.id}


async def test_vectorize_failure_500(db_session_factory, tmp_path):
    sf = db_session_factory
    _seed_chunk(sf)
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=FakeKbStore(), embed=BoomEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/vectorize")
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "vectorize_failed"
    with sf() as s:
        assert s.query(KnowledgeChunk).one().vectorize_status == "pending"


async def test_reset_selective_rebuild(db_session_factory, tmp_path):
    """选择性重建:文档块删除重建,手工块与 qa_mined 块保留。"""
    sf = db_session_factory
    store = FakeKbStore()
    old1 = _seed_chunk(sf, category="旧", questions="旧1", answer="旧。",
                       vectorize_status="done",
                       source_doc="knowledge_docs/old.md", chunk_index=1)
    old2 = _seed_chunk(sf, category="旧", questions="旧2", answer="旧。",
                       vectorize_status="done",
                       source_doc="knowledge_docs/old.md", chunk_index=2)
    manual_id = _seed_chunk(sf, category="手工", questions="手", answer="工。",
                            vectorize_status="done")
    mined_id = _seed_chunk(sf, category="对话挖掘", questions="挖", answer="掘。",
                           content_type="qa_mined", vectorize_status="done")
    with sf() as s:  # 文档块指针按惯例互指
        r1 = s.query(KnowledgeChunk).filter_by(id=old1).one()
        r2 = s.query(KnowledgeChunk).filter_by(id=old2).one()
        r1.next_chunk_id = old2
        r2.prev_chunk_id = old1
        s.commit()
    store.upsert([(i, [1.0] + [0.0] * 1023) for i in (old1, old2, manual_id, mined_id)])
    docs = _write_docs(tmp_path)  # 全新文档 a.md/b.md
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      sf=sf, store=store, embed=FakeEmbed(), docs_dir=docs)
    async with _client(app) as client:
        resp = await client.post("/kb/api/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["removed"] == 2 and data["skipped_vectorize"] is False
    with sf() as s:
        rows = s.query(KnowledgeChunk).all()
        by_id = {r.id: r for r in rows}
        assert old1 not in by_id and old2 not in by_id  # 旧文档块已删
        assert manual_id in by_id and mined_id in by_id  # 手工/挖掘保留
        new_doc_rows = [r for r in rows if r.source_doc is not None]
        assert len(new_doc_rows) == 2  # 新文档重建
        assert all(r.vectorize_status == "done" for r in new_doc_rows)
        ids = set(by_id)
    assert set(store.deleted) == {old1, old2}
    assert store.all_ids() == ids  # 两库主键集合仍一致(含保留块)


# ─── 挖掘与检索自测 ──────────────────────────────────────────────────────────


class StubStructuredModel:
    """with_structured_output 返回脚本化 parsed(与 tests/test_mining.py 同型)。"""

    def __init__(self, parsed):
        self._parsed = parsed

    def with_structured_output(self, schema, method=None, include_raw=False):
        outer = self

        class _R:
            def invoke(self, messages):
                return {"parsed": outer._parsed, "parsing_error": None}

        return _R()


async def test_mine_endpoint(db_session_factory, tmp_path):
    from app.knowledge.mining import MiningBatchResult, MinedConversation, MinedQA
    sf = db_session_factory
    store = FakeKbStore()
    with sf() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        s.add(Message(conversation_id=conv.id, role="user", content="邮费多少"))
        s.add(Message(conversation_id=conv.id, role="assistant", content="满 99 包邮"))
        s.commit()
        cid = conv.id
    model = StubStructuredModel(MiningBatchResult(conversations=[
        MinedConversation(conversation_id=cid,
                          items=[MinedQA(question="邮费政策", answer="满 99 包邮")])]))
    app = make_kb_app(settings=make_settings(embedding_api_key="fake"),
                      model=model, sf=sf, store=store, embed=FakeEmbed(),
                      docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/mine")
    assert resp.status_code == 200
    with sf() as s:
        chunk = s.query(KnowledgeChunk).one()
        assert chunk.content_type == "qa_mined" and chunk.vectorize_status == "done"
        assert s.query(QaExtractionStaging).one().status == "kept"
        assert s.query(QaMiningProgress).count() == 1
    assert store.all_ids() == {chunk.id}


async def test_search_probe_passed_flags(db_session_factory, tmp_path):
    sf = db_session_factory
    store = FakeKbStore()
    id1 = _seed_chunk(sf, category="商品FAQ", questions="邮费怎么算",
                      answer="满 99 包邮。", vectorize_status="done",
                      source_doc="knowledge_docs/商品FAQ.md", chunk_index=1)
    id2 = _seed_chunk(sf, category="售后政策", questions="退货时效",
                      answer="七天。", vectorize_status="done",
                      source_doc="knowledge_docs/退货政策.md", chunk_index=1)
    store.upsert([(id1, [1.0] + [0.0] * 1023), (id2, [0.0, 1.0] + [0.0] * 1022)])
    app = make_kb_app(sf=sf, store=store, embed=FakeEmbed(), docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/search", json={
            "query": "邮费怎么收", "top_k": 2, "min_score": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["note"] is None
    assert [h["chunk_id"] for h in data["hits"]] == [id1, id2]  # Top-2 全回
    assert data["hits"][0]["passed"] is True and data["hits"][0]["score"] >= 0.5
    assert data["hits"][1]["passed"] is False  # 压线灰显由页面负责
    assert data["hits"][1]["source_doc"] == "knowledge_docs/退货政策.md"


async def test_search_probe_degraded_note(db_session_factory, tmp_path):
    """无 key 时自测走降级 note 语义,不报错。"""
    sf = db_session_factory
    app = make_kb_app(settings=make_settings(embedding_api_key=""),
                      sf=sf, store=FakeKbStore(), embed=None, docs_dir=tmp_path)
    async with _client(app) as client:
        resp = await client.post("/kb/api/search", json={"query": "邮费"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["note"] == "知识检索未配置" and data["hits"] == []
