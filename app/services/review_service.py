"""ch09 审核队列服务:读路径 + 驳回 + 失败重试;approve 写回在知识库锁任务。"""

from app.models import LowConfidenceQuestion, ReviewQueue

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
