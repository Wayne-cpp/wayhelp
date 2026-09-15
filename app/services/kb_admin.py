"""知识库管理后台服务层(/kb 页面):状态聚合 + 同步动作。

一切写库复用 ingest/mining 现有函数(run_ingest/vectorize_pending/run_mining),
MySQL id 与 Milvus pk 1:1 对齐的不变量不由本模块另起路径。写动作(build/
vectorize/mine/reset)拿全局作业互斥锁;读(state/search)不拿锁,可与作业并发。
"""

import threading
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.knowledge.chunking import Chunk, ChunkingError, chunk_document
from app.knowledge.ingest import (
    REPO_ROOT,
    IngestError,
    _same_chunk,
    check_consistency,
    resolve_source_doc,
    run_ingest,
    vectorize_pending,
)
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.mining import run_mining
from app.knowledge.retriever import KnowledgeRetriever
from app.models import KnowledgeChunk, QaExtractionStaging, QaMiningProgress

DEFAULT_DOCS_DIR = REPO_ROOT / "knowledge_docs"


class KbAdminError(Exception):
    code = "kb_error"
    status = 500

    def __init__(self, message: str | None = None):
        self.message = message or self.code
        super().__init__(self.message)


class KbUnavailableError(KbAdminError):
    code = "kb_unavailable"
    status = 503


class EmbeddingNotConfiguredError(KbAdminError):
    code = "embedding_not_configured"
    status = 503


class JobBusyError(KbAdminError):
    code = "job_busy"
    status = 409


class InvalidInputError(KbAdminError):
    code = "invalid_input"
    status = 400


class IngestFailedError(KbAdminError):
    code = "ingest_failed"
    status = 409


class VectorizeFailedError(KbAdminError):
    code = "vectorize_failed"
    status = 500


class MiningFailedError(KbAdminError):
    code = "mining_failed"
    status = 500


_JOB_LOCK = threading.Lock()


@contextmanager
def _job_lock():
    if not _JOB_LOCK.acquire(blocking=False):
        raise JobBusyError("另一个知识库作业正在进行,请稍后重试")
    try:
        yield
    finally:
        _JOB_LOCK.release()


def _truncate(text: str, limit: int = 40) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _chunk_dict(c: Chunk) -> dict:
    return {"category": c.category, "questions": c.questions, "answer": c.answer,
            "section_path": c.section_path, "content_type": c.content_type,
            "is_key_clause": c.is_key_clause, "is_table": c.is_table,
            "has_overlap": c.has_overlap, "is_hard_cut": c.is_hard_cut}


# ─── 读:状态聚合 ────────────────────────────────────────────────────────────


def _material_entry(settings: Settings, path: Path) -> dict:
    """单份建库材料:切块统计 + 特性标签 + 逐块明细(供页面预览区细看)。"""
    source_doc = resolve_source_doc(path)
    entry: dict = {"name": path.name, "source_doc": source_doc, "content_type": None,
                   "chunk_count": 0, "key_clause_count": 0,
                   "features": {"table": False, "overlap": False, "hard_cut": False},
                   "error": None, "chunks": []}
    try:
        chunks = chunk_document(path.read_text(encoding="utf-8"), source=source_doc,
                                max_chars=settings.max_chunk_chars,
                                overlap_chars=settings.chunk_overlap_chars)
    except ChunkingError as exc:
        entry["error"] = str(exc)
        return entry
    entry.update(
        content_type=chunks[0].content_type,
        chunk_count=len(chunks),
        key_clause_count=sum(1 for c in chunks if c.is_key_clause),
        features={"table": any(c.is_table for c in chunks),
                  "overlap": any(c.has_overlap for c in chunks),
                  "hard_cut": any(c.is_hard_cut for c in chunks)},
        chunks=[_chunk_dict(c) for c in chunks])
    return entry


def _mining_state(s) -> dict:
    counts = dict(
        s.query(QaExtractionStaging.status, func.count())
        .group_by(QaExtractionStaging.status).all())
    batches = s.query(func.count(func.distinct(QaMiningProgress.batch_no))).scalar()
    items: dict[str, list[dict]] = {}
    for status in ("extracted", "kept", "discarded"):
        rows = (s.query(QaExtractionStaging).filter_by(status=status)
                .order_by(QaExtractionStaging.id.desc()).limit(50).all())
        items[status] = [
            {"id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
             "question": r.question, "answer": r.answer,
             "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
            for r in rows]
    return {"extracted": counts.get("extracted", 0), "kept": counts.get("kept", 0),
            "discarded": counts.get("discarded", 0), "batches": batches or 0,
            "items": items}


def get_state(settings: Settings, session_factory: sessionmaker,
              store: MilvusKnowledgeStore,
              docs_dir: Path = DEFAULT_DOCS_DIR) -> dict:
    """一次聚合 /kb 页面六区块全部数据。"""
    report = check_consistency(session_factory, store)
    with session_factory() as s:
        key_clauses = (s.query(KnowledgeChunk).filter_by(is_key_clause=True).count())
        done = (s.query(KnowledgeChunk)
                .filter_by(vectorize_status="done").count())
        pending_samples = [
            {"id": r.id, "category": r.category,
             "questions": _truncate(r.questions),
             "source_doc": r.source_doc, "content_type": r.content_type}
            for r in (s.query(KnowledgeChunk).filter_by(vectorize_status="pending")
                      .order_by(KnowledgeChunk.id).limit(10).all())]
        recent = [
            {"id": r.id, "content_type": r.content_type, "category": r.category,
             "questions": _truncate(r.questions), "is_key_clause": bool(r.is_key_clause),
             "vectorize_status": r.vectorize_status, "source_doc": r.source_doc,
             "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
            for r in (s.query(KnowledgeChunk)
                      .order_by(KnowledgeChunk.id.desc()).limit(20).all())]
        mining = _mining_state(s)
    return {
        "embedding_configured": settings.has_embedding_key(),
        "gauge": {"mysql_count": report.mysql_count,
                  "pending_count": report.pending_count,
                  "milvus_count": report.milvus_count,
                  "key_clause_count": key_clauses,
                  "consistent": report.consistent,
                  "no_pending": report.no_pending,
                  "pointers_ok": report.pointers_ok,
                  "ids_match": report.ids_match},
        "materials": [_material_entry(settings, p)
                      for p in sorted(docs_dir.glob("*.md"),
                                      key=lambda p: resolve_source_doc(p))],
        "mining": mining,
        "vectorize": {"pending": report.pending_count, "done": done,
                      "milvus": report.milvus_count,
                      "pending_samples": pending_samples},
        "recent": recent,
    }


# ─── 手工录入 ────────────────────────────────────────────────────────────────


def _manual_text(doc_type: str, title: str, markdown: str) -> str:
    title, markdown = title.strip(), markdown.strip()
    if doc_type not in ("faq", "policy", "manual"):
        raise InvalidInputError(f"未知文档类型: {doc_type!r}")
    if not title or not markdown:
        raise InvalidInputError("标题与正文不能为空")
    return f"---\ntype: {doc_type}\n---\n\n# {title}\n\n{markdown}\n"


def _manual_chunks(settings: Settings, doc_type: str, title: str,
                   markdown: str) -> list[Chunk]:
    text = _manual_text(doc_type, title, markdown)
    try:
        return chunk_document(text, source=f"manual:{title}",
                              max_chars=settings.max_chunk_chars,
                              overlap_chars=settings.chunk_overlap_chars)
    except ChunkingError as exc:
        raise InvalidInputError(str(exc)) from exc


def manual_preview(settings: Settings, doc_type: str, title: str,
                   markdown: str) -> dict:
    chunks = _manual_chunks(settings, doc_type, title, markdown)
    return {"chunk_count": len(chunks), "chunks": [_chunk_dict(c) for c in chunks]}


def manual_ingest(settings: Settings, session_factory: sessionmaker, embed,
                  store: MilvusKnowledgeStore, doc_type: str, title: str,
                  markdown: str, vectorize: bool = True) -> dict:
    """插入 knowledge_chunks(source_doc/chunk_index/prev/next 按挖掘块惯例置 NULL);
    vectorize=True 时随即 vectorize_pending,中途失败块留 pending 并报错。"""
    chunks = _manual_chunks(settings, doc_type, title, markdown)
    if vectorize and embed is None:
        raise EmbeddingNotConfiguredError(
            "未配置 EMBEDDING_API_KEY,无法向量化;可取消「顺手向量化」仅入库")
    with session_factory() as s:
        try:
            rows = []
            for c in chunks:
                row = KnowledgeChunk(
                    category=c.category, questions=c.questions, answer=c.answer,
                    section_path=c.section_path, content_type=c.content_type,
                    is_key_clause=c.is_key_clause, source_doc=None, chunk_index=None,
                    vectorize_status="pending")
                s.add(row)
                rows.append(row)
            s.flush()
            ids = [r.id for r in rows]
            s.commit()
        except Exception as exc:
            s.rollback()
            raise KbAdminError(f"手工录入入库失败: {type(exc).__name__}") from exc
    if vectorize:
        try:
            store.ensure_collection()
            vectorize_pending(settings, session_factory, embed, store)
        except IngestError as exc:
            raise VectorizeFailedError(
                f"已入库 {len(ids)} 块(留 pending),向量化失败: {exc}") from exc
    return {"chunk_ids": ids, "vectorized": vectorize,
            "message": f"已入库 {len(ids)} 块" + (",已向量化" if vectorize else "(待向量化)")}


# ─── 写动作(作业互斥锁内)────────────────────────────────────────────────────


def build_kb(settings: Settings, session_factory: sessionmaker, embed,
             store: MilvusKnowledgeStore,
             docs_dir: Path = DEFAULT_DOCS_DIR) -> dict:
    """一键建库。无 embedding key 时自动只跑 Phase 1,留下 pending 不报错。"""
    with _job_lock():
        if embed is None:
            rc = run_ingest(settings, session_factory, None, store, docs_dir,
                            skip_vectorize=True)
            if rc != 0:
                raise IngestFailedError("切块入库失败,详见服务日志")
            pending = check_consistency(session_factory, store).pending_count
            return {"skipped_vectorize": True, "pending_count": pending,
                    "message": f"未配置 EMBEDDING_API_KEY:已完成切块入库,"
                               f"留下 {pending} 个块待向量化"}
        rc = run_ingest(settings, session_factory, embed, store, docs_dir)
        if rc != 0:
            raise IngestFailedError("建库失败:文档或切分配置已变更(需先 reset)"
                                    "或其他错误,详见服务日志")
        return {"skipped_vectorize": False, "pending_count": 0,
                "message": "建库完成:无 pending,指针完整,两库主键集合一致"}


def preview_docs(settings: Settings, session_factory: sessionmaker,
                 docs_dir: Path = DEFAULT_DOCS_DIR) -> dict:
    """差异报告(干跑):逐份 新增/未变/已变更,已变更标红由页面负责。"""
    items = []
    for path in sorted(docs_dir.glob("*.md"), key=lambda p: resolve_source_doc(p)):
        source_doc = resolve_source_doc(path)
        item: dict = {"name": path.name, "source_doc": source_doc}
        try:
            chunks = chunk_document(path.read_text(encoding="utf-8"),
                                    source=source_doc,
                                    max_chars=settings.max_chunk_chars,
                                    overlap_chars=settings.chunk_overlap_chars)
        except ChunkingError as exc:
            item.update(status="changed", error=str(exc),
                        note="切块失败;build 将硬报错,需先修复文档或 reset")
            items.append(item)
            continue
        with session_factory() as s:
            existing = (s.query(KnowledgeChunk).filter_by(source_doc=source_doc)
                        .order_by(KnowledgeChunk.chunk_index).all())
        if not existing:
            status = "new"
        elif len(existing) == len(chunks) and all(
                _same_chunk(r, c) for r, c in zip(existing, chunks)):
            status = "unchanged"
        else:
            status = "changed"
        item.update(status=status, chunk_count=len(chunks))
        if status == "changed":
            item["note"] = "与库内同 source_doc 行不一致:build 将硬报错,需先 reset"
        items.append(item)
    return {"docs": items}


def vectorize_kb(settings: Settings, session_factory: sessionmaker, embed,
                 store: MilvusKnowledgeStore) -> dict:
    if embed is None:
        raise EmbeddingNotConfiguredError("未配置 EMBEDDING_API_KEY,无法向量化")
    with _job_lock():
        store.ensure_collection()
        try:
            vectorize_pending(settings, session_factory, embed, store)
        except IngestError as exc:
            raise VectorizeFailedError(f"向量化失败(批次留 pending): {exc}") from exc
        report = check_consistency(session_factory, store)
        return {"pending_count": report.pending_count,
                "milvus_count": report.milvus_count,
                "message": "向量化完成,无 pending" if report.no_pending
                           else f"向量化完成,仍有 {report.pending_count} 个块待向量化"}


def mine_kb(settings: Settings, session_factory: sessionmaker, model, embed,
            store: MilvusKnowledgeStore) -> dict:
    if embed is None:
        raise EmbeddingNotConfiguredError("未配置 EMBEDDING_API_KEY,无法挖掘")
    with _job_lock():
        rc = run_mining(settings, session_factory, model, embed, store)
        if rc != 0:
            raise MiningFailedError("挖掘完成但存在失败批次,详见服务日志")
        return {"message": "挖掘完成"}


def reset_kb(settings: Settings, session_factory: sessionmaker, embed,
             store: MilvusKnowledgeStore,
             docs_dir: Path = DEFAULT_DOCS_DIR) -> dict:
    """选择性重建:只清 source_doc 非 NULL 的文档块(向量 + 行)后重新建库;
    手工块与 qa_mined 块一律保留。"""
    with _job_lock():
        with session_factory() as s:
            ids = [r[0] for r in s.query(KnowledgeChunk.id)
                   .filter(KnowledgeChunk.source_doc.isnot(None)).all()]
        store.delete_by_ids(ids)
        with session_factory() as s:
            try:
                # 先摘掉文档内自引用指针,再多行删除(InnoDB 自引用 FK 逐行检查)
                s.query(KnowledgeChunk).filter(
                    KnowledgeChunk.source_doc.isnot(None)).update(
                    {"prev_chunk_id": None, "next_chunk_id": None},
                    synchronize_session=False)
                s.query(KnowledgeChunk).filter(
                    KnowledgeChunk.source_doc.isnot(None)).delete(
                    synchronize_session=False)
                s.commit()
            except Exception as exc:
                s.rollback()
                raise KbAdminError(
                    f"reset 删除 MySQL 行失败: {type(exc).__name__}") from exc
        if embed is None:
            rc = run_ingest(settings, session_factory, None, store, docs_dir,
                            skip_vectorize=True)
            if rc != 0:
                raise IngestFailedError(
                    f"已清除 {len(ids)} 个文档块,但重建切块失败,详见服务日志")
            pending = check_consistency(session_factory, store).pending_count
            return {"removed": len(ids), "skipped_vectorize": True,
                    "pending_count": pending,
                    "message": f"已清除 {len(ids)} 个文档块并重新切块入库;"
                               f"未配置 EMBEDDING_API_KEY,留下 {pending} 个块待向量化"}
        rc = run_ingest(settings, session_factory, embed, store, docs_dir)
        if rc != 0:
            raise IngestFailedError(
                f"已清除 {len(ids)} 个文档块,但重建失败,详见服务日志")
        return {"removed": len(ids), "skipped_vectorize": False,
                "pending_count": 0,
                "message": f"已重建 {len(ids)} 个文档块,双写一致"}


# ─── 检索自测 ────────────────────────────────────────────────────────────────


def search_probe(settings: Settings, session_factory: sessionmaker, embed,
                 store: MilvusKnowledgeStore, query: str, top_k: int,
                 min_score: float) -> dict:
    # T6 过渡:自测旁路固定 dense 腿(与旧 probe 行为一致);此处未装配 reranker,
    # 若放行默认 hybrid_rerank 会恒降级。T11 接入 strategy/scope 透传与 reranker。
    retriever = KnowledgeRetriever(settings, embed=embed, store=store,
                                   session_factory=session_factory)
    return retriever.probe(query, top_k, min_score, strategy="dense")
