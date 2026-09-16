"""RAG 评估页 API(/api/rag-eval/*):报告只读 + faith_cases 台账分页/处置。"""

import asyncio

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.schemas import FaithCasePatchRequest
from app.services import rag_eval as svc

router = APIRouter(prefix="/api/rag-eval")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code,
                                                               "message": message}})


def _sf(request: Request):
    sf = request.app.state.session_factory
    if sf is None:
        return None
    return sf


@router.get("/report")
async def rag_eval_report(request: Request):
    path = request.app.state.rag_eval_report_path
    data = await asyncio.to_thread(svc.load_report, path)   # 坏 → ReportCorruptError → 502
    if data is None:
        return {"empty": True}
    return data


@router.get("/faith-cases")
async def faith_cases(request: Request,
                      status: str = Query("未解决"),
                      page: int = Query(1, ge=1),
                      size: int = Query(10, ge=1, le=50)):
    if status not in svc.STATUSES:
        return _err(422, "invalid_request", "status 须为 未解决/已解决/无需解决")
    sf = _sf(request)
    if sf is None:
        return _err(503, "rag_eval_unavailable", "数据库依赖未装配")
    path = request.app.state.rag_eval_report_path

    def _work():
        try:
            report = svc.load_report(path)
        except svc.ReportCorruptError:
            report = None   # 报告损坏不阻断台账:按「无当前报告」口径
        data = svc.list_faith_cases(sf, status, page, size)
        data["stats"] = svc.compute_stats(report, sf)
        return data

    return await asyncio.to_thread(_work)


@router.patch("/faith-cases/{row_id}")
async def faith_case_patch(row_id: int, body: FaithCasePatchRequest, request: Request):
    sf = _sf(request)
    if sf is None:
        return _err(503, "rag_eval_unavailable", "数据库依赖未装配")
    resolution = body.resolution.strip()
    if not 1 <= len(resolution) <= 300:
        return _err(422, "invalid_request", "处置说明须为 1-300 字符")
    item = await asyncio.to_thread(svc.update_faith_case, sf, row_id,
                                   body.status, resolution)
    if item is None:
        return _err(404, "faith_case_not_found", "个案不存在")
    return item
