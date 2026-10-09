"""ch09 审核队列服务:读路径 + 驳回 + 失败重试 + approve 写回(知识库锁任务)。"""

from sqlalchemy import func

from app.knowledge.ingest import vectorize_pending
from app.knowledge.state import KnowledgeState
from app.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue

PAGE_SIZE = 20


class ReviewError(Exception):
    code = "review_error"
    status = 500
    def __init__(self, message=None):
        self.message = message or self.code
        super().__init__(self.message)


class ReviewNotFoundError(ReviewError):
    code = "review_not_found"
    status = 404


class ReviewConflictError(ReviewError):
    code = "review_conflict"
    status = 409


class ReviewUnavailableError(ReviewError):
    code = "review_unavailable"
    status = 503


class ReviewWriteError(ReviewError):
    code = "review_write_failed"
    status = 500


class ReviewKbNotReadyError(ReviewError):
    code = "review_kb_not_ready"
    status = 409


def _item(r: ReviewQueue) -> dict:
    return {"id": r.id, "normalized_question": r.normalized_question,
            "occurrence_count": r.occurrence_count,
            "ai_suggested_answer": r.ai_suggested_answer,
            "review_status": r.review_status,
            "approved_answer": r.approved_answer,
            "knowledge_chunk_ids": r.knowledge_chunk_ids,
            "last_write_error": r.last_write_error,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}


def list_reviews(sf, status: str | None, page: int, page_size: int = PAGE_SIZE) -> dict:
    with sf() as s:
        q = s.query(ReviewQueue)
        if status:
            q = q.filter_by(review_status=status)
        total = q.count()
        rows = (q.order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
    return {"items": [_item(r) for r in rows], "total": total}


def get_review_detail(sf, review_id: int) -> dict:
    with sf() as s:
        r = s.get(ReviewQueue, review_id)
        if r is None:
            raise ReviewNotFoundError("review not found")
        srcs = (s.query(LowConfidenceQuestion)
                .filter_by(matched_review_id=review_id)
                .order_by(LowConfidenceQuestion.id).all())
        out = _item(r)
        out["sources"] = [{
            "lcq_id": x.id, "raw_question": x.raw_question,
            "resolved_question": x.resolved_question, "source": x.source,
            "retrieved_chunks": x.retrieved_chunks,
            "created_at": x.created_at.strftime("%Y-%m-%d %H:%M:%S")} for x in srcs]
        return out


def reject(sf, review_id: int) -> dict:
    with sf() as s:
        r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
        if r is None:
            raise ReviewNotFoundError("review not found")
        if r.review_status != "待审":
            raise ReviewConflictError("仅待审可驳回;写入中/通过/驳回不可变更")
        r.review_status = "驳回"
        s.commit()
        return {"review_status": "驳回"}


def retry_failed_question(sf, lcq_id: int) -> None:
    with sf() as s:
        r = s.query(LowConfidenceQuestion).filter_by(id=lcq_id).with_for_update().first()
        if r is None:
            raise ReviewNotFoundError("question not found")
        if r.process_status != "failed":
            raise ReviewConflictError("仅 failed 可人工重试")
        r.process_status = "pending"
        r.attempt_count = 0
        r.next_attempt_at = None
        r.last_error = None
        s.commit()


def list_failed_questions(sf, page: int, page_size: int = PAGE_SIZE) -> dict:
    with sf() as s:
        q = s.query(LowConfidenceQuestion).filter_by(process_status="failed")
        total = q.count()
        rows = (q.order_by(LowConfidenceQuestion.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
    return {"items": [{"id": r.id, "raw_question": r.raw_question,
                       "source": r.source, "attempt_count": r.attempt_count,
                       "last_error": r.last_error,
                       "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                      for r in rows], "total": total}


# ─── approve 写回(知识库锁任务)─────────────────────────────────────────────


def _faq_chunks_for(settings, normalized: str, answer: str):
    """复用 ch03 手工 FAQ 切分(frontmatter faq + chunk_document),source 标签 review;
    ## 二级标题即 normalized_question——faq 切分契约里问题取二级标题,只放 H1 会整篇无块。"""
    from app.knowledge.chunking import chunk_document
    from app.services.kb_admin import _manual_text
    return chunk_document(_manual_text("faq", "审核补充", f"## {normalized}\n\n{answer}"),
                          source="review",
                          max_chars=settings.max_chunk_chars,
                          overlap_chars=settings.chunk_overlap_chars)


def _freeze_chunks(s, settings, review: ReviewQueue, answer: str) -> list[int]:
    """冻结事务内:已有 review:<id> 块复用 id(写入中重试/重放部分完成,内容已由
    答案一致性守卫保证),否则新建;chunk_index 从 1 起,靠既有唯一键
    (source_doc, chunk_index) 防重复并维护 prev/next 指针。"""
    source_doc = f"review:{review.id}"
    existing = (s.query(KnowledgeChunk).filter_by(source_doc=source_doc)
                .order_by(KnowledgeChunk.chunk_index).all())
    if existing:
        return [c.id for c in existing]
    chunks = _faq_chunks_for(settings, review.normalized_question, answer)
    rows = []
    for i, c in enumerate(chunks, start=1):
        row = KnowledgeChunk(category="审核补充", questions=c.questions,
                             answer=c.answer, section_path=c.section_path,
                             content_type="faq", is_key_clause=c.is_key_clause,
                             source_doc=source_doc, chunk_index=i,
                             vectorize_status="pending")
        s.add(row)
        rows.append(row)
    s.flush()
    for i, row in enumerate(rows):
        row.prev_chunk_id = rows[i - 1].id if i > 0 else None
        row.next_chunk_id = rows[i + 1].id if i + 1 < len(rows) else None
    s.flush()
    return [r.id for r in rows]


def _save_write_error(sf, review_id: int, exc: Exception) -> None:
    """短事务记 last_write_error(截 500);失败不掩盖原始异常。"""
    try:
        with sf() as s:
            r = s.query(ReviewQueue).filter_by(id=review_id).first()
            if r is not None:
                r.last_write_error = f"{type(exc).__name__}: {exc}"[:500]
                s.commit()
    except Exception:
        pass


def approve(settings, sf, embed, store, review_id: int,
            approved_answer: str | None, state=None) -> dict:
    """冻结事务(写入中) → 向量化 → CAS 通过;锁序固定:知识库锁 → review 行锁,
    冻结事务提交后释放 DB 行锁、向量化阶段继续持知识库锁(外部 embedding 不持行锁)。"""
    if embed is None:
        raise ReviewConflictError("未配置 EMBEDDING_API_KEY,无法发布到知识库")
    if state is not None and state.get() != KnowledgeState.READY.value:
        raise ReviewKbNotReadyError(
            f"知识库状态为 {state.get()},须先 rebuild 恢复全局一致性")
    from app.services.kb_admin import job_lock
    with job_lock():   # 与 build/vectorize/mine/reset/rebuild/manual_ingest 互斥;忙即 409
        with sf() as s:
            r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
            if r is None:
                raise ReviewNotFoundError("review not found")
            if r.review_status == "驳回":
                raise ReviewConflictError("已驳回不可通过")
            if r.review_status == "通过":
                if approved_answer and approved_answer.strip() != r.approved_answer:
                    raise ReviewConflictError("已通过不可改答案")
                return {"review_status": "通过",
                        "knowledge_chunk_ids": r.knowledge_chunk_ids}
            if r.review_status == "写入中":
                if approved_answer and approved_answer.strip() != r.approved_answer:
                    raise ReviewConflictError("写入中答案已冻结,仅可原样重试")
            else:   # 待审 → 冻结
                ans = (approved_answer or "").strip()
                if not ans:
                    raise ReviewConflictError("首次通过必须提供核准答案")
                r.approved_answer = ans
            ids = _freeze_chunks(s, settings, r, r.approved_answer)
            r.knowledge_chunk_ids = ids
            r.review_status = "写入中"
            s.commit()   # 冻结事务:状态+答案+块+索引原子提交
        try:
            store.ensure_collection()
            vectorize_pending(settings, sf, embed, store)   # 扫全库 pending;写入中=已核准
        except Exception as exc:
            _save_write_error(sf, review_id, exc)
            raise ReviewWriteError(f"向量化失败,可重试: {type(exc).__name__}") from exc
        with sf() as s:   # 短事务 CAS:写入中 → 通过
            r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
            if r is None or r.review_status != "写入中":
                raise ReviewConflictError("状态已变更")
            pending = (s.query(KnowledgeChunk)
                       .filter_by(source_doc=f"review:{review_id}",
                                  vectorize_status="pending").count())
            if pending:
                raise ReviewWriteError("仍有 pending 块,重试")
            r.review_status = "通过"
            r.approved_at = func.now()
            r.last_write_error = None
            s.commit()
            return {"review_status": "通过", "knowledge_chunk_ids": r.knowledge_chunk_ids}


def replay_reviews_locked(settings, sf, embed, store) -> None:
    """full rebuild 持知识库锁调用(不递归取锁):清表后按 id 升序重放(通过,写入中)
    的冻结 FAQ 块、刷新 knowledge_chunk_ids 与 prev/next,统一 vectorize 后逐条复核
    无 pending;写入中收敛为通过(approved_at=now),通过保留原 approved_at;
    待审/驳回从不重放。失败抛出让 rebuild 置 REBUILD_REQUIRED——review_queue
    恢复源保留,下次 rebuild 可继续重建。"""
    with sf() as s:
        reviews = (s.query(ReviewQueue)
                   .filter(ReviewQueue.review_status.in_(["通过", "写入中"]))
                   .order_by(ReviewQueue.id).all())
        ids = [r.id for r in reviews]
    if not ids:
        return
    with sf() as s:
        for review_id in ids:
            r = s.query(ReviewQueue).filter_by(id=review_id).first()
            if not (r.approved_answer or "").strip():
                raise ReviewWriteError(f"review:{review_id} 无冻结答案,无法重放")
            r.knowledge_chunk_ids = _freeze_chunks(s, settings, r, r.approved_answer)
        s.commit()
    store.ensure_collection()
    vectorize_pending(settings, sf, embed, store)
    for review_id in ids:
        with sf() as s:   # 逐条复核:无 pending 才收敛/保通过
            r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
            pending = (s.query(KnowledgeChunk)
                       .filter_by(source_doc=f"review:{review_id}",
                                  vectorize_status="pending").count())
            if pending:
                raise ReviewWriteError(
                    f"review:{review_id} 重放后仍有 {pending} 个 pending 块")
            if r.review_status == "写入中":
                r.review_status = "通过"
                r.approved_at = func.now()
                r.last_write_error = None
            s.commit()
