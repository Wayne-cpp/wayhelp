"""ch09 审核/飞轮 API(/api/review,/api/flywheel):薄壳路由,同步 service 经
asyncio.to_thread 进线程池;flywheel run/retry 成功后在事件环内 notify worker
(Event 非线程安全,不能放进 to_thread),worker 缺失时安全降级仍 202。
approve 写回在 service 内取知识库全局锁(与 kb 管理动作互斥),路由只透传
app.state 的 settings/embed/kb_store/knowledge_state(照 kb.py _deps 惯例)。"""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import ReviewApproveRequest
from app.services import review_service
from app.services.review_service import ReviewUnavailableError

router = APIRouter(prefix="/api")


def _sf(request: Request):
    sf = request.app.state.session_factory
    if sf is None:
        raise ReviewUnavailableError("审核依赖未装配(session_factory)")
    return sf


def _notify_worker(request: Request) -> bool:
    worker = getattr(request.app.state, "flywheel_worker", None)
    if worker is not None:
        worker.notify()
        return True
    return False


@router.get("/review")
async def review_list(request: Request, status: str | None = None, page: int = 1):
    return await asyncio.to_thread(review_service.list_reviews, _sf(request),
                                   status, page)


@router.get("/review/{review_id}/detail")
async def review_detail(review_id: int, request: Request):
    return await asyncio.to_thread(review_service.get_review_detail,
                                   _sf(request), review_id)


@router.post("/review/{review_id}/approve")
async def review_approve(review_id: int, body: ReviewApproveRequest,
                         request: Request):
    sf = _sf(request)
    store = request.app.state.kb_store
    if store is None:
        raise ReviewUnavailableError("审核写回依赖未装配(kb_store)")
    return await asyncio.to_thread(
        review_service.approve, request.app.state.settings, sf,
        request.app.state.embed, store, review_id, body.approved_answer,
        request.app.state.knowledge_state)


@router.post("/review/{review_id}/reject")
async def review_reject(review_id: int, request: Request):
    return await asyncio.to_thread(review_service.reject, _sf(request), review_id)


@router.post("/flywheel/run")
async def flywheel_run(request: Request):
    return JSONResponse(status_code=202, content={"notified": _notify_worker(request)})


@router.get("/flywheel/failures")
async def flywheel_failures(request: Request, page: int = 1):
    return await asyncio.to_thread(review_service.list_failed_questions,
                                   _sf(request), page)


@router.post("/flywheel/questions/{question_id}/retry")
async def flywheel_retry(question_id: int, request: Request):
    await asyncio.to_thread(review_service.retry_failed_question,
                            _sf(request), question_id)
    return JSONResponse(status_code=202, content={"notified": _notify_worker(request)})
