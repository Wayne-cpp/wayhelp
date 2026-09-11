import pytest
import sqlalchemy.orm

from app.knowledge.ingest import IngestError, resolve_source_doc, run_ingest, vectorize_pending
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import KnowledgeChunk
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

DOC_A = """---
type: faq
---
# 商品FAQ

## 运费怎么算

满 99 包邮,未满 8 元。

## 发货时间

16 点前当天发。
"""

DOC_B = """---
type: policy
---
# 售后政策

## 退货

七天无理由。
"""


class FakeEmbeddings:
    """确定性假向量:维度 4,内容按文本 hash 可区分。"""
    dim = 4

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    def _v(self, text):
        h = abs(hash(text)) % 100
        return [h / 100.0, (100 - h) / 100.0, 0.5, 0.5]


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "ingest.db"), dim=4)
    yield s
    s.close()


@pytest.fixture()
def settings():
    return make_settings()


def _write_docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text(DOC_A, encoding="utf-8")
    (d / "b.md").write_text(DOC_B, encoding="utf-8")
    return d


def test_ingest_full_pipeline(db_session_factory, store, settings, tmp_path):
    docs = _write_docs(tmp_path)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        assert len(rows) == 3
        assert all(r.vectorize_status == "done" for r in rows)
        assert all(r.vector_id == str(r.id) for r in rows)
        faq_rows = [r for r in rows if r.source_doc.endswith("a.md")]
        assert len(faq_rows) == 2  # 运费 + 发货
        by_idx = {r.chunk_index: r for r in faq_rows}
        assert by_idx[1].next_chunk_id == by_idx[2].id
        assert by_idx[2].prev_chunk_id == by_idx[1].id
        assert by_idx[1].prev_chunk_id is None and by_idx[2].next_chunk_id is None
    assert store.all_ids() == {r.id for r in rows}  # 两库主键集合一致


def test_ingest_rerun_reuses_ids(db_session_factory, store, settings, tmp_path):
    docs = _write_docs(tmp_path)
    run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs)
    with db_session_factory() as s:
        first_ids = [r.id for r in s.query(KnowledgeChunk).order_by(KnowledgeChunk.id)]
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        assert [r.id for r in rows] == first_ids  # 原样重跑 ID 不变、不新增行


def test_ingest_changed_document_rejected(db_session_factory, store, settings, tmp_path):
    docs = _write_docs(tmp_path)
    run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs)
    (docs / "a.md").write_text(DOC_A + "\n## 新问答\n\n新内容。\n", encoding="utf-8")
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 1
    with db_session_factory() as s:  # 旧内容不被覆盖
        assert s.query(KnowledgeChunk).count() == 3


def test_same_content_different_docs_both_stored(db_session_factory, store, settings, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "x.md").write_text(DOC_B, encoding="utf-8")
    (d / "y.md").write_text(DOC_B, encoding="utf-8")
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, d) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 2  # 同内容不同文档分别落行


def test_interrupted_vectorize_resumes(db_session_factory, store, settings, tmp_path, monkeypatch):
    docs = _write_docs(tmp_path)
    real_upsert = store.upsert
    state = {"calls": 0}

    def boom_once(rows):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("simulated crash")
        return real_upsert(rows)

    monkeypatch.setattr(store, "upsert", boom_once)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 1
    with db_session_factory() as s:  # 第一次:全部落 MySQL,向量中断
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 3
    monkeypatch.setattr(store, "upsert", real_upsert)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 0
        ids = {r.id for r in s.query(KnowledgeChunk).all()}
    assert store.all_ids() == ids  # 重跑补齐,主键集合一致


def test_dimension_mismatch_aborts(db_session_factory, store, settings, tmp_path):
    class BadDim(FakeEmbeddings):
        def _v(self, text):
            return [0.1, 0.2]  # 维度 2 ≠ 4
    docs = _write_docs(tmp_path)
    assert run_ingest(settings, db_session_factory, BadDim(), store, docs) == 1
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="done").count() == 0


def test_resolve_source_doc(tmp_path):
    from pathlib import Path
    import app
    repo_root = Path(app.__file__).resolve().parent.parent
    inside = repo_root / "knowledge_docs" / "商品FAQ.md"
    assert resolve_source_doc(inside) == "knowledge_docs/商品FAQ.md"
    outside = tmp_path / "x.md"
    assert resolve_source_doc(outside) == outside.resolve().as_posix()


def test_phase1_failure_rolls_back_document(db_session_factory, store, settings, tmp_path, monkeypatch):
    """Phase 1 文档事务中途失败:整体回滚不留半截,重跑干净入库(spec §11)。"""
    docs = _write_docs(tmp_path)
    real_commit = sqlalchemy.orm.Session.commit
    state = {"calls": 0}

    def boom_once(self):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("simulated crash")
        return real_commit(self)

    monkeypatch.setattr(sqlalchemy.orm.Session, "commit", boom_once)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 1
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 0  # 整体回滚
    monkeypatch.setattr(sqlalchemy.orm.Session, "commit", real_commit)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).all()
        assert len(rows) == 3
        assert all(r.vectorize_status == "done" for r in rows)


def test_milvus_written_mysql_uncommitted_window_resumes(db_session_factory, store, settings, tmp_path, monkeypatch):
    """「Milvus upsert 已成功、MySQL 回填提交前崩溃」窗口:重跑 upsert 幂等补齐(spec §11)。"""
    docs = _write_docs(tmp_path)
    real_upsert = store.upsert
    real_commit = sqlalchemy.orm.Session.commit
    state = {"armed": False}

    def upsert_then_arm(rows):
        result = real_upsert(rows)
        state["armed"] = True  # 下一次 MySQL 提交崩溃,复现该窗口
        return result

    def commit_maybe_boom(self):
        if state["armed"]:
            state["armed"] = False
            raise RuntimeError("simulated crash after upsert")
        return real_commit(self)

    monkeypatch.setattr(store, "upsert", upsert_then_arm)
    monkeypatch.setattr(sqlalchemy.orm.Session, "commit", commit_maybe_boom)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 1
    with db_session_factory() as s:  # 窗口状态:MySQL 仍 pending,Milvus 已有向量
        rows = s.query(KnowledgeChunk).all()
        assert {r.vectorize_status for r in rows} == {"pending"}
        ids = {r.id for r in rows}
    assert store.all_ids() == ids

    monkeypatch.setattr(store, "upsert", real_upsert)
    monkeypatch.setattr(sqlalchemy.orm.Session, "commit", real_commit)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 0
        ids = {r.id for r in s.query(KnowledgeChunk).all()}
    assert store.all_ids() == ids and len(ids) == 3  # upsert 幂等,无重复向量
