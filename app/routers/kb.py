"""知识库管理后台 API(/kb/api/*):同步端点,阻塞工作经 asyncio.to_thread 进线程池。"""

import asyncio

from fastapi import APIRouter, Request

from app.schemas import KbManualRequest, KbSearchRequest
from app.services import kb_admin

router = APIRouter(prefix="/kb/api")


def _deps(request: Request):
    """settings/session_factory/kb_store;缺装配 → 503 kb_unavailable。"""
    sf = request.app.state.session_factory
    store = request.app.state.kb_store
    if sf is None or store is None:
        raise kb_admin.KbUnavailableError("知识库管理依赖未装配(session_factory/kb_store)")
    return request.app.state.settings, sf, store


@router.get("/state")
async def kb_state(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.get_state, settings, sf, store,
                                   request.app.state.knowledge_state,
                                   request.app.state.kb_docs_dir)


@router.post("/manual/preview")
async def kb_manual_preview(body: KbManualRequest, request: Request):
    return await asyncio.to_thread(kb_admin.manual_preview,
                                   request.app.state.settings,
                                   body.doc_type, body.title, body.markdown)


@router.post("/manual/ingest")
async def kb_manual_ingest(body: KbManualRequest, request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.manual_ingest, settings, sf,
                                   request.app.state.embed, store,
                                   body.doc_type, body.title, body.markdown,
                                   body.vectorize)


@router.post("/build")
async def kb_build(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.build_kb, settings, sf,
                                   request.app.state.embed, store,
                                   request.app.state.kb_docs_dir)


@router.post("/preview")
async def kb_preview(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.preview_docs, settings, sf,
                                   request.app.state.kb_docs_dir)


@router.post("/mine")
async def kb_mine(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.mine_kb, settings, sf,
                                   request.app.state.model,
                                   request.app.state.embed, store)


@router.post("/vectorize")
async def kb_vectorize(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.vectorize_kb, settings, sf,
                                   request.app.state.embed, store)


@router.post("/reset")
async def kb_reset(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.reset_kb, settings, sf,
                                   request.app.state.embed, store,
                                   request.app.state.kb_docs_dir)


@router.post("/rebuild")
async def kb_rebuild(request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.rebuild_index, settings, sf,
                                   request.app.state.embed, store,
                                   request.app.state.knowledge_state,
                                   request.app.state.kb_docs_dir)


@router.post("/search")
async def kb_search(body: KbSearchRequest, request: Request):
    settings, sf, store = _deps(request)
    return await asyncio.to_thread(kb_admin.search_probe, settings, sf,
                                   request.app.state.embed, store,
                                   body.query, body.top_k, body.min_score,
                                   body.strategy, body.scope,
                                   request.app.state.retriever)
