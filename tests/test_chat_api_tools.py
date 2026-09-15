import httpx

from app.main import create_app
from app.prompts.service import SERVICE_SYSTEM_PROMPT
from tests.conftest import (
    TEST_USER_ID,
    FakeStreamModel,
    UserBoundMemoryStore,
    make_runtime,
    make_settings,
)
from tests.test_chat_api import parse_frames, post_stream

TOOL_CHUNKS = [
    {"name": "query_order", "args": "{\"order_id\": \"1001\"}", "id": "call_1", "index": 0},
]


def make_app(script):
    return create_app(settings=make_settings(), model=FakeStreamModel(script),
                      runtime=make_runtime())


async def test_tool_frames_over_sse():
    app = make_app([("tool", TOOL_CHUNKS), ("then", ["答", "复"])])
    payload = {"user_id": TEST_USER_ID, "message": "查订单"}
    status, lines = await post_stream(app, payload)
    assert status == 200
    frames = parse_frames(lines)
    assert frames[0]["type"] == "session"
    assert frames[1]["type"] == "tool_start"
    assert frames[1]["name"] == "query_order" and frames[1]["args"] == {"order_id": "1001"}
    assert frames[1]["tool_call_id"] == "call_1"
    assert frames[2]["type"] == "tool_end" and frames[2]["ok"] is True
    assert frames[3]["type"] == "delta" and frames[3]["content"] == "答"
    assert frames[-1] == "[DONE]"


async def test_tool_end_summary_capped_80():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def verbose(x: str) -> str:
        """超长返回"""
        return "长" * 500

    app = create_app(settings=make_settings(), model=FakeStreamModel(
        [("tool", [{"name": "verbose", "args": "{\"x\": \"1\"}", "id": "c1", "index": 0}]),
         ("then", ["答"])]
    ), runtime=make_runtime(tools=[verbose]))
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    frames = parse_frames(lines)
    end = next(f for f in frames if isinstance(f, dict) and f.get("type") == "tool_end")
    assert len(end["summary"]) <= 80


async def test_invalid_user_id_422():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream",
                                 json={"user_id": "not-a-uuid", "message": "hi"})
        assert resp.status_code == 422


async def test_user_mismatch_404():
    app = make_app(["答"])
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    sid = parse_frames(lines)[0]["session_id"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": "22222222-2222-2222-2222-222222222222",
            "session_id": sid, "message": "hi"})
        assert resp.status_code == 404


async def test_bad_session_id_form_422():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": TEST_USER_ID, "session_id": "abc!!", "message": "hi"})
        assert resp.status_code == 422


def test_system_prompt_rule_demo_data_honesty():
    assert "演示数据" in SERVICE_SYSTEM_PROMPT
    assert "不得承诺" in SERVICE_SYSTEM_PROMPT


# ---- T9:citations 帧 / 硬闸门拒答(SSE 层) ----

FAQ_CALL = {"name": "query_faq", "args": "{\"keyword\": \"能寄到日本吗\"}",
            "id": "c1", "index": 0}


class _OkRetriever:
    """一条高置信命中(chunk_id=5)。"""

    def search(self, q, **kw):
        from app.knowledge.query_understanding import passthrough_plan
        from app.knowledge.retriever import KnowledgeHit, RetrievalResult
        hit = KnowledgeHit(5, 0.9, "faq", "能寄到日本吗",
                           "目前仅支持中国大陆地区配送。", None, 0, "配送/服务范围")
        return RetrievalResult([hit], "hybrid_rerank", "hybrid_rerank",
                               0.9, 0.5, False, None, passthrough_plan(q),
                               {"dense": 1, "bm25": 1, "fused": 1})


class _LowConfRetriever:
    """零命中 → low_confidence=True。"""

    def search(self, q, **kw):
        from app.knowledge.query_understanding import passthrough_plan
        from app.knowledge.retriever import RetrievalResult
        return RetrievalResult([], "hybrid_rerank", "hybrid_rerank", None, 0.5,
                               True, None, passthrough_plan(q), {"dense": 0, "bm25": 0})


def _faq_app(retriever, script):
    from app.main import AppRuntime, create_app
    from app.tools.business import build_tools

    settings = make_settings()
    return create_app(settings=settings, model=FakeStreamModel(script),
                      runtime=AppRuntime(store=UserBoundMemoryStore(1000, 100, 8000),
                                         toolset_factory=lambda sid: build_tools(
                                             None, 1, retriever=retriever,
                                             settings=settings)))


async def test_citations_frame_over_sse():
    app = _faq_app(_OkRetriever(), [
        ("tool", [FAQ_CALL]),
        ("finish", "tool_calls"),
        ("then", ["目前仅支持中国大陆地区配送 [1]"]),
    ])
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "能寄到日本吗"})
    assert status == 200
    frames = parse_frames(lines)
    types = [f["type"] if isinstance(f, dict) else f for f in frames]
    assert "citations" in types
    cit = next(f for f in frames if isinstance(f, dict) and f.get("type") == "citations")
    assert cit["citations"][0]["ref_no"] == 1 and cit["citations"][0]["chunk_id"] == 5
    # 顺序:最后一个 delta 之后、[DONE] 之前
    assert types.index("citations") > max(i for i, t in enumerate(types) if t == "delta")
    assert frames[-1] == "[DONE]"


async def test_hard_gate_refusal_frame_over_sse():
    from app.prompts.service import REFUSAL_ANSWER

    app = _faq_app(_LowConfRetriever(), [
        ("tool", [FAQ_CALL]),
        ("finish", "tool_calls"),
        ("then", ["不应被调用"]),
    ])
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "能寄到日本吗"})
    assert status == 200
    frames = parse_frames(lines)
    types = [f["type"] if isinstance(f, dict) else f for f in frames]
    assert "citations" not in types
    deltas = "".join(f["content"] for f in frames
                     if isinstance(f, dict) and f.get("type") == "delta")
    assert deltas == REFUSAL_ANSWER
    assert frames[-1] == "[DONE]"
