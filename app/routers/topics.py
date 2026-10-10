"""ch10 主题归类 API(/api/topics/*):薄壳,同步 service 走 asyncio.to_thread。"""

import asyncio

from fastapi import APIRouter, Request

from app.services import topic_classify

router = APIRouter(prefix="/api")


def _deps(request: Request):
    return request.app.state.settings, request.app.state.session_factory


@router.post("/topics/classify")
async def topics_classify(request: Request):
    settings, sf = _deps(request)
    return await asyncio.to_thread(topic_classify.run_classification, settings, sf)


@router.get("/topics/distribution")
async def topics_distribution(request: Request):
    settings, sf = _deps(request)
    return await asyncio.to_thread(topic_classify.get_distribution, settings, sf)
