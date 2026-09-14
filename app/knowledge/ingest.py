"""建库两阶段流水线(spec §6):Phase 1 文档事务落 MySQL(pending),
Phase 2 向量化 upsert Milvus 并回填 done。中断重跑幂等。"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.knowledge.chunking import ChunkingError, chunk_document
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import KnowledgeChunk

REPO_ROOT = Path(__file__).resolve().parents[2]
VECTORIZE_BATCH = 32


class IngestError(Exception):
    pass


def vector_text(category: str, questions: str, answer: str) -> str:
    return f"{category}\n{questions}\n{answer}"


def resolve_source_doc(path: Path) -> str:
    real = path.resolve()
    try:
        rel = real.relative_to(REPO_ROOT)
        ident = rel.as_posix()
    except ValueError:
        ident = real.as_posix()
    if len(ident) > 255:
        raise IngestError(f"来源路径超过 255 字符: {ident}")
    return ident


def vectorize_pending(settings: Settings, session_factory: sessionmaker,
                      embed, store: MilvusKnowledgeStore) -> None:
    """Phase 2:pending 批次向量化。任一批失败抛 IngestError,该批保持 pending。"""
    while True:
        with session_factory() as s:
            rows = (s.query(KnowledgeChunk)
                    .filter_by(vectorize_status="pending")
                    .order_by(KnowledgeChunk.id)
                    .limit(VECTORIZE_BATCH).all())
            if not rows:
                return
            payloads = [(r.id, vector_text(r.category, r.questions, r.answer)) for r in rows]
        try:
            vectors = embed.embed_documents([t for _, t in payloads])
            if len(vectors) != len(payloads):
                raise IngestError(
                    f"返回向量数 {len(vectors)} ≠ 输入 {len(payloads)}")
            for v in vectors:
                if len(v) != store.dim:
                    raise IngestError(f"向量维度 {len(v)} ≠ 集合维度 {store.dim}")
            store.upsert(list(zip([i for i, _ in payloads], vectors)))
            with session_factory() as s:
                for chunk_id, _ in payloads:
                    s.query(KnowledgeChunk).filter_by(id=chunk_id).update(
                        {"vector_id": str(chunk_id), "vectorize_status": "done"})
                s.commit()
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError(f"向量化批次失败: {type(exc).__name__}: {exc}") from exc


def _same_chunk(row: KnowledgeChunk, chunk) -> bool:
    return (row.category == chunk.category and row.questions == chunk.questions
            and row.answer == chunk.answer and row.section_path == chunk.section_path
            and row.content_type == chunk.content_type
            and bool(row.is_key_clause) == chunk.is_key_clause)


def _load_document(session_factory: sessionmaker, path: Path, settings: Settings) -> None:
    source_doc = resolve_source_doc(path)
    try:
        chunks = chunk_document(path.read_text(encoding="utf-8"), source=source_doc,
                                max_chars=settings.max_chunk_chars,
                                overlap_chars=settings.chunk_overlap_chars)
    except ChunkingError as exc:
        raise IngestError(str(exc)) from exc
    with session_factory() as s:
        try:
            existing = (s.query(KnowledgeChunk)
                        .filter_by(source_doc=source_doc)
                        .order_by(KnowledgeChunk.chunk_index).all())
            if existing:
                if (len(existing) != len(chunks)
                        or not all(_same_chunk(r, c) for r, c in zip(existing, chunks))):
                    raise IngestError(
                        f"已导入文档或切分配置发生变化: {source_doc}(不覆盖旧知识)")
                rows = existing  # 原样重跑:复用 ID,仍重建指针
            else:
                rows = []
                for i, c in enumerate(chunks, start=1):
                    row = KnowledgeChunk(
                        category=c.category, questions=c.questions, answer=c.answer,
                        section_path=c.section_path, content_type=c.content_type,
                        is_key_clause=c.is_key_clause, source_doc=source_doc,
                        chunk_index=i, vectorize_status="pending")
                    s.add(row)
                    rows.append(row)
                s.flush()  # 拿自增 ID
            for i, row in enumerate(rows):
                row.prev_chunk_id = rows[i - 1].id if i > 0 else None
                row.next_chunk_id = rows[i + 1].id if i + 1 < len(rows) else None
            s.commit()  # 整篇文档只提交一次;中断则整体回滚
        except IngestError:
            s.rollback()
            raise
        except Exception as exc:
            s.rollback()
            raise IngestError(f"文档入库失败 {source_doc}: {type(exc).__name__}") from exc


def _pointer_error(session_factory: sessionmaker) -> str | None:
    """prev/next 指针完整性;None = 完整。"""
    with session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
    by_doc: dict[str, list[KnowledgeChunk]] = {}
    for r in rows:
        if r.source_doc is not None:
            by_doc.setdefault(r.source_doc, []).append(r)
    for doc, doc_rows in by_doc.items():
        doc_rows.sort(key=lambda r: r.chunk_index)
        for i, r in enumerate(doc_rows):
            expect_prev = doc_rows[i - 1].id if i > 0 else None
            expect_next = doc_rows[i + 1].id if i + 1 < len(doc_rows) else None
            if r.prev_chunk_id != expect_prev or r.next_chunk_id != expect_next:
                return f"指针不完整: {doc} chunk_index={r.chunk_index}"
    return None


@dataclass(frozen=True)
class ConsistencyReport:
    """双写一致性只读报告(check_consistency 返回,不抛异常)。"""
    consistent: bool
    mysql_count: int
    milvus_count: int
    pending_count: int
    no_pending: bool
    pointers_ok: bool
    ids_match: bool


def check_consistency(session_factory: sessionmaker,
                      store: MilvusKnowledgeStore) -> ConsistencyReport:
    """与 _verify 同口径的三项检查:pending 归零、指针完整、两库主键集合相等。"""
    with session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        pending = sum(1 for r in rows if r.vectorize_status == "pending")
        mysql_ids = {r.id for r in rows}
    milvus_ids = store.all_ids()
    no_pending = pending == 0
    pointers_ok = _pointer_error(session_factory) is None
    ids_match = mysql_ids == milvus_ids
    return ConsistencyReport(
        consistent=no_pending and pointers_ok and ids_match,
        mysql_count=len(mysql_ids), milvus_count=len(milvus_ids),
        pending_count=pending, no_pending=no_pending,
        pointers_ok=pointers_ok, ids_match=ids_match)


def _verify(session_factory: sessionmaker, store: MilvusKnowledgeStore) -> None:
    with session_factory() as s:
        pending = s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count()
        if pending:
            raise IngestError(f"仍有 {pending} 个 pending 块未向量化")
        mysql_ids = {r[0] for r in s.query(KnowledgeChunk.id).all()}
    err = _pointer_error(session_factory)
    if err is not None:
        raise IngestError(err)
    milvus_ids = store.all_ids()
    if milvus_ids != mysql_ids:
        raise IngestError(
            f"两库主键集合不一致: 仅 Milvus {sorted(milvus_ids - mysql_ids)},"
            f"仅 MySQL {sorted(mysql_ids - milvus_ids)}")


def run_ingest(settings: Settings, session_factory: sessionmaker, embed,
               store: MilvusKnowledgeStore, docs_dir: Path,
               *, skip_vectorize: bool = False) -> int:
    """skip_vectorize=True 时只跑 Phase 1 切块落库(embed 可为 None),
    校验只做指针完整性(pending 归零与两库主键比对留给向量化后)。"""
    try:
        paths = sorted(docs_dir.glob("*.md"), key=lambda p: resolve_source_doc(p))
        if not paths:
            raise IngestError(f"目录无 Markdown 文档: {docs_dir}")
        if skip_vectorize:
            for path in paths:
                _load_document(session_factory, path, settings)
            err = _pointer_error(session_factory)
            if err is not None:
                raise IngestError(err)
            with session_factory() as s:
                pending = (s.query(KnowledgeChunk)
                           .filter_by(vectorize_status="pending").count())
            print(f"[ingest] 完成(仅切块入库): {pending} 个块待向量化")
            return 0
        store.ensure_collection()
        vectorize_pending(settings, session_factory, embed, store)  # resume 历史 pending
        for path in paths:
            _load_document(session_factory, path, settings)
        vectorize_pending(settings, session_factory, embed, store)
        _verify(session_factory, store)
    except IngestError as exc:
        print(f"[ingest] 失败: {exc}")
        return 1
    print("[ingest] 完成: 无 pending,指针完整,两库主键集合一致")
    return 0
