"""SSE 帧协议契约(Task 13 重写):经 LangGraph 图驱动,ScriptedChatModel 触发。

query_faq 用例已随工具退出聊天而删除;硬闸门/citations 场景从 query_faq 路径
迁到预检索路径(经 runtime.retriever 注假检索器)。test_chat_api 迁来的
帧序/JSON 转义契约也落在本文件(FakeStreamModel 非 Runnable 不发电回调,
无法再驱动 messages 流)。
"""
import httpx

from app.main import AppRuntime, create_app
from app.prompts.service import REFUSAL_ANSWER, SERVICE_SYSTEM_PROMPT
from tests.conftest import TEST_USER_ID, ScriptedChatModel, UserBoundMemoryStore, make_settings
from tests.test_chat_api import parse_frames, post_stream

BUSINESS = ['{"intent":"订单","needs_knowledge":false}']
KNOWLEDGE = ['{"intent":"售后","needs_knowledge":true}']
GEN = ['{"mode":"general"}']  # ch06 Task 7:售后脚本须经 refund_scope(general)进 refund_policy
COMPLAINT = ['{"intent":"投诉","needs_knowledge":false}']

ORDER_CALL = [{"name": "query_order", "args": "{\"order_id\": \"1111-1001\"}",
               "id": "call_1", "index": 0}]


def make_app(scripts, retriever=None):
    """scripts:每次模型调用一段,第一段必为分类输出,其后归 main_agent。"""
    return create_app(
        settings=make_settings(),
        model=ScriptedChatModel(scripts=[list(s) for s in scripts]),
        runtime=AppRuntime(store=UserBoundMemoryStore(1000, 100, 8000),
                           toolset_factory=lambda sid: [], retriever=retriever))


# ---- SSE 帧序协议(自 test_chat_api 迁入) ----

async def test_sse_frame_sequence_session_deltas_done():
    app = make_app([BUSINESS, ["你", "好"]])
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "在吗"})
    assert status == 200
    frames = parse_frames(lines)
    assert frames[0]["type"] == "session"
    assert [f["content"] for f in frames[1:-1]] == ["你", "好"]
    assert frames[-1] == "[DONE]"


async def test_sse_json_escaping():
    app = make_app([BUSINESS, ['带"引号"和\n换行']])
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "在吗"})
    frames = parse_frames(lines)  # json.loads 不炸即转义正确
    assert frames[1]["content"] == '带"引号"和\n换行'


async def test_main_agent_chat_visible_tag_gates_deltas():
    """chat_visible tag 过滤集成钉:main_agent 的 astream 带 per-call tags,经
    messages 流 metadata.tags 外发;分类 JSON 等其余节点调用不外发。若 per-call
    config tags 不进 metadata(langgraph 行为偏差),deltas 将为空 → 红(spec §8)。"""
    app = make_app([BUSINESS, ["答复正文"]])
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "查订单"})
    assert status == 200
    frames = parse_frames(lines)
    deltas = "".join(f["content"] for f in frames
                     if isinstance(f, dict) and f["type"] == "delta")
    assert deltas == "答复正文"
    assert "intent" not in deltas  # 分类 token 不外发


# ---- 工具帧 ----

async def test_tool_frames_over_sse():
    app = make_app([BUSINESS, [("tool", ORDER_CALL)], ["答", "复"]])
    payload = {"user_id": TEST_USER_ID, "message": "查订单"}
    status, lines = await post_stream(app, payload)
    assert status == 200
    frames = parse_frames(lines)
    assert frames[0]["type"] == "session"
    assert frames[1]["type"] == "tool_start"
    assert frames[1]["name"] == "query_order" and frames[1]["args"] == {"order_id": "1111-1001"}
    assert frames[1]["tool_call_id"] == "call_1"
    assert frames[2]["type"] == "tool_end" and frames[2]["ok"] is True
    assert frames[3]["type"] == "delta" and frames[3]["content"] == "答"
    assert frames[-1] == "[DONE]"


async def test_tool_end_summary_capped_80():
    logistics_call = [{"name": "query_logistics",
                       "args": "{\"order_id\": \"1111-1001\"}",
                       "id": "call_1", "index": 0}]
    app = make_app([BUSINESS, [("tool", logistics_call)], ["答"]])
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    frames = parse_frames(lines)
    end = next(f for f in frames if isinstance(f, dict) and f.get("type") == "tool_end")
    assert len(end["summary"]) == 80  # query_logistics 的 mock JSON 远超 80,必被截断


# ---- suggest_actions 帧(投诉固定回复) ----

async def test_suggest_actions_frame_over_sse():
    app = make_app([COMPLAINT])
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "我要投诉"})
    assert status == 200
    frames = parse_frames(lines)
    types = [f["type"] if isinstance(f, dict) else f for f in frames]
    sugg = next(f for f in frames if isinstance(f, dict)
                and f.get("type") == "suggest_actions")
    assert set(sugg) == {"type", "source_message_id", "options"}
    assert sugg["source_message_id"]  # 绑定产生本轮 user 消息
    assert [o["action"] for o in sugg["options"]] == ["transfer_human", "create_ticket"]
    assert sugg["options"][1]["ticket_type"] == "投诉"
    # 顺序:最后一个 delta 之后、[DONE] 之前(log 提交成功后才发)
    assert types.index("suggest_actions") > max(
        i for i, t in enumerate(types) if t == "delta")
    assert frames[-1] == "[DONE]"


# ---- 请求校验(与流无关,原样保留) ----

async def test_invalid_user_id_422():
    app = make_app([BUSINESS])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream",
                                 json={"user_id": "not-a-uuid", "message": "hi"})
        assert resp.status_code == 422


async def test_user_mismatch_404():
    app = make_app([BUSINESS])
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    sid = parse_frames(lines)[0]["session_id"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": "22222222-2222-2222-2222-222222222222",
            "session_id": sid, "message": "hi"})
        assert resp.status_code == 404


async def test_bad_session_id_form_422():
    app = make_app([BUSINESS])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": TEST_USER_ID, "session_id": "abc!!", "message": "hi"})
        assert resp.status_code == 422


def test_system_prompt_rule_demo_data_honesty():
    assert "演示数据" in SERVICE_SYSTEM_PROMPT
    assert "不得承诺" in SERVICE_SYSTEM_PROMPT


# ---- 预检索路径:硬闸门拒答 / citations(自 query_faq 路径迁来) ----

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


async def test_citations_frame_over_sse():
    app = make_app([KNOWLEDGE, GEN, ["目前仅支持中国大陆地区配送 [1]"]],
                   retriever=_OkRetriever())
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
    app = make_app([KNOWLEDGE, GEN, ["不应被调用"]], retriever=_LowConfRetriever())
    status, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "能寄到日本吗"})
    assert status == 200
    frames = parse_frames(lines)
    types = [f["type"] if isinstance(f, dict) else f for f in frames]
    assert "citations" not in types
    deltas = "".join(f["content"] for f in frames
                     if isinstance(f, dict) and f.get("type") == "delta")
    assert deltas == REFUSAL_ANSWER
    assert frames[-1] == "[DONE]"
