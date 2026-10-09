"""ch09 成本统计 API(/api/stats/*):薄壳路由,阻塞的 Langfuse 拉取经
asyncio.to_thread 进线程池;错误经 CostStatsError → main 统一错误契约。"""

import asyncio

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.services.cost_stats import cost_by_intent

router = APIRouter(prefix="/api/stats")


@router.get("/cost-by-intent")
async def stats_cost_by_intent(request: Request, days: int = Query(7)):
    """spec §5.1:days 仅 7|30 两档时间窗。"""
    if days not in (7, 30):
        return JSONResponse(status_code=422,
                            content={"error": {"code": "invalid_request",
                                               "message": "days 仅支持 7 或 30"}})
    return await asyncio.to_thread(cost_by_intent, request.app.state.settings, days)
