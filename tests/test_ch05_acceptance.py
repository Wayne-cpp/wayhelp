# tests/test_ch05_acceptance.py
"""spec §13 验收标准 1-11 的集成钉。帧序:session → delta/tool_* → citations? →
suggest_actions? → [DONE](失败路径无 DONE/无按钮)。"""

import asyncio
import json
import logging

import httpx
import pytest

from app.knowledge.retriever import (
    NOTE_NOT_BUILT, NOTE_REBUILDING, KnowledgeHit, RetrievalResult,
)
from app.main import create_app
from app.prompts.service import (
    AGENT_BUDGET_ANSWER, CHITCHAT_REPLY, KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER,
)
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


# ── 基建 ──

def _hit(score=0.9):
    return KnowledgeHit(chunk_id=7, score=score, category="policy",
                        questions="退货政策", answer="7 天无理由退货。",
                        source_doc="returns-policy.md", chunk_index=0,
                        section_path="退货政策")


def _result(low=False, note=None, hits=None, score=0.9):
    return RetrievalResult(
        hits=hits if hits is not None else [_hit()],
        requested_strategy="hybrid_rerank",
        effective_strategy="hybrid_rerank", confidence_score=score,
        confidence_threshold=0.0553, low_confidence=low, note=note,
        query_plan=None, leg_counts={"dense": 1})


class _FakeRetriever:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def search(self, query, **kw):
        self.calls.append(query)
        return self._result


def _make_app(scripts, retriever=None, db_sf=None, **settings_over):
    import dataclasses
    runtime = make_runtime(tools=[])
    if retriever is not None:
        runtime = dataclasses.replace(runtime, retriever=retriever)
    if db_sf is not None:
        runtime = dataclasses.replace(runtime, session_factory=db_sf)
    return create_app(settings=make_settings(**settings_over),
                      model=ScriptedChatModel(scripts=scripts), runtime=runtime)


async def _turn(client, message, session_id=None):
    """发一轮聊天,返回 (frames, session_id)。"""
    resp = await client.post("/v1/chat/stream", json={
        "user_id": TEST_USER_ID, "session_id": session_id, "message": message})
    assert resp.status_code == 200
    frames = []
    async for line in resp.aiter_lines():
        if line.startswith("data:"):
            data = line[5:].strip()
            frames.append(data if data == "[DONE]" else json.loads(data))
    sid = next((f["session_id"] for f in frames
                if isinstance(f, dict) and f.get("type") == "session"), session_id)
    return frames, sid


def _types(frames):
    return [f if isinstance(f, str) else f["type"] for f in frames]


def _deltas(frames):
    return "".join(f["content"] for f in frames
                   if isinstance(f, dict) and f["type"] == "delta")


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


# ── 验收 1:政策类问题日志可见强制检索节点;纯业务查询不检索 ──
# ch06 Task 7:refund 走正式子流程,断言从 retrieve/confidence_gate 迁至
# refund_scope/refund_policy;脚本须插 refund_scope 的 mode 段(general)。

async def test_a1_knowledge_runs_retrieve_and_logs(caplog):
    app = _make_app(
        [['{"intent":"退款退货","confidence":0.9}'],
         ['{"mode":"general"}'],
         ["7 天无理由退货[1]。"]],
        retriever=_FakeRetriever(_result()))
    async with await _client(app) as client:
        with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
            frames, _ = await _turn(client, "退货政策是什么")
    assert "node=refund_scope" in caplog.text and "node=refund_policy" in caplog.text
    assert _deltas(frames) == "7 天无理由退货[1]。"


async def test_a1_business_skips_retrieve(caplog):
    rt = _FakeRetriever(_result())
    app = _make_app(
        [['{"intent":"物流","confidence":0.9}'],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c1",
                     "args": '{"order_id":"1001"}'}])],
         ["派送中。"]],
        retriever=rt)
    async with await _client(app) as client:
        with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
            await _turn(client, "订单 1001 的物流到哪了")
    assert rt.calls == [] and "node=refund_policy" not in caplog.text


# ── 验收 2:Agent 自调工具作答 ──

async def test_a2_agent_calls_logistics_tool():
    app = _make_app(
        [['{"intent":"物流","confidence":0.9}'],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c1",
                     "args": '{"order_id":"1111-1001"}'}])],
         ["您的订单由顺丰承运,派送中。"]])
    async with await _client(app) as client:
        frames, _ = await _turn(client, "订单 1111-1001 的物流到哪了")
    starts = [f for f in frames if isinstance(f, dict) and f["type"] == "tool_start"]
    assert [s["name"] for s in starts] == ["query_logistics"]
    assert "派送中" in _deltas(frames)


# ── 验收 3:投诉双按钮;点旧按钮建单绑定原消息;都不点继续正常聊(验收 8 合并) ──

async def test_a3_complaint_two_independent_buttons_and_old_button_binds_original(
        db_session_factory):
    app = _make_app(
        [['{"intent":"投诉","confidence":0.95}'],
         ['{"resolved_query": ""}'],                # 第二轮 understand 罐头透传(有历史)
         ['{"intent":"订单","confidence":0.9}'],
         ["订单状态良好。"]],
        db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "我要投诉你们的服务")
        sug = [f for f in frames if isinstance(f, dict) and f["type"] == "suggest_actions"]
        assert len(sug) == 1
        opts = sug[0]["options"]
        assert [o["action"] for o in opts] == ["transfer_human", "create_ticket"]
        assert opts[1]["ticket_type"] == "投诉"
        # 帧序:suggest_actions 在最后一个 delta 之后、DONE 之前
        assert _types(frames).index("suggest_actions") > len(_types(frames)) - 3
        assert _types(frames)[-1] == "[DONE]"
        mid1 = sug[0]["source_message_id"]
        # 不点按钮继续正常聊(产生新用户消息)
        frames2, _ = await _turn(client, "顺便查下订单 1001", sid)
        assert "[DONE]" in _types(frames2)
        # 回头点第一条消息的旧按钮:建的仍是投诉那条
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": mid1,
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 200

        def _check():
            from app.models import Ticket
            with db_session_factory() as s:
                t = s.get(Ticket, resp.json()["ticket_no"])
                assert t.description == "我要投诉你们的服务"
        await asyncio.to_thread(_check)


# ── 验收 4:闲聊固定话术,零生成调用 ──

async def test_a4_chitchat_fixed_reply():
    app = _make_app([['{"intent":"闲聊","confidence":0.9}']])
    async with await _client(app) as client:
        frames, _ = await _turn(client, "你好")
    assert _deltas(frames) == CHITCHAT_REPLY
    # 只消费了分类一段脚本;闲聊回复零生成调用
    assert app.state.model.scripts == []


# ── 验收 5:先订单后物流,ReAct 多步 ──

async def test_a5_multi_step_react():
    app = _make_app(
        [['{"intent":"订单","confidence":0.9}'],
         [("tool", [{"index": 0, "name": "query_order", "id": "c1",
                     "args": '{"order_id":"1111-1001"}'}])],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c2",
                     "args": '{"order_id":"1111-1001"}'}])],
         ["订单已发货,顺丰派送中。"]])
    async with await _client(app) as client:
        frames, _ = await _turn(client, "订单 1111-1001 买的是什么,到哪了")
    starts = [f["name"] for f in frames
              if isinstance(f, dict) and f["type"] == "tool_start"]
    assert starts == ["query_order", "query_logistics"]


# ── 验收 6:故障不入池 / 零命中入池 / 自评拒答入池 ──

async def test_a6_pooling_rules(db_session_factory):
    from app.models import LowConfidenceQuestion
    # 维护态:不入池,回 KB_UNAVAILABLE_ANSWER(维修寄修=通用政策问 → mode general)
    app = _make_app([['{"intent":"售后","confidence":0.9}'],
                     ['{"mode":"general"}']],
                    retriever=_FakeRetriever(_result(note=NOTE_REBUILDING,
                                                     low=True, hits=[], score=None)),
                    db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "维修寄修流程是什么")
        assert _deltas(frames) == KB_UNAVAILABLE_ANSWER
        assert not any(isinstance(f, dict) and f["type"] == "citations" for f in frames)
    # 零命中:入 retrieval_low_conf
    app = _make_app([['{"intent":"售后","confidence":0.9}'],
                     ['{"mode":"general"}']],
                    retriever=_FakeRetriever(_result(note=NOTE_NOT_BUILT,
                                                     low=True, hits=[], score=None)),
                    db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid2 = await _turn(client, "偏门问题甲")
        assert _deltas(frames) == REFUSAL_ANSWER
    # 高分但模型自评拒答:入 self_check
    app = _make_app([['{"intent":"退款退货","confidence":0.9}'],
                     ['{"mode":"general"}'],
                     [REFUSAL_ANSWER]],
                    retriever=_FakeRetriever(_result()), db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid3 = await _turn(client, "定制商品能退吗")
        assert _deltas(frames) == REFUSAL_ANSWER

    def _check():
        with db_session_factory() as s:
            rows = s.query(LowConfidenceQuestion).order_by(
                LowConfidenceQuestion.id).all()
            sources = [(r.raw_question, r.source) for r in rows]
            assert ("维修寄修流程是什么", "retrieval_low_conf") not in sources
            assert ("偏门问题甲", "retrieval_low_conf") in sources
            assert ("定制商品能退吗", "self_check") in sources
    await asyncio.to_thread(_check)


# ── 验收 7:checkpoint 跨轮无临时状态串用 ──

async def test_a7_state_reset_between_turns():
    rt = _FakeRetriever(_result(low=True, hits=[], score=0.01))
    app = _make_app(
        [['{"intent":"退款退货","confidence":0.9}'],   # 第一轮:低置信被拒
         ['{"mode":"general"}'],                    # refund_scope(火星特产无个案订单)
         ['{"resolved_query": ""}'],                # 第二轮 understand 罐头透传(有历史)
         ['{"intent":"闲聊","confidence":0.9}'],      # 第二轮:闲聊
         ],
        retriever=rt)
    async with await _client(app) as client:
        f1, sid = await _turn(client, "火星特产能退吗")
        assert _deltas(f1) == REFUSAL_ANSWER
        f2, _ = await _turn(client, "你好呀", sid)
        assert _deltas(f2) == CHITCHAT_REPLY
        # 第二轮不得出现第一轮的低置信标记副作用:无 citations、无 suggest_actions
        assert not any(isinstance(f, dict) and f["type"] in ("citations", "suggest_actions")
                       for f in f2)


# ── 验收 9:预算在发起下一次模型调用前生效 ──

async def test_a9_budget_blocks_before_call():
    model = ScriptedChatModel(scripts=[['{"intent":"订单","confidence":0.9}'],
                                       ["不应出现的文本"]])
    runtime = make_runtime(tools=[])
    app = create_app(settings=make_settings(max_agent_tokens=1), model=model,
                     runtime=runtime)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "查订单")
    assert _deltas(frames) == AGENT_BUDGET_ANSWER
    assert "不应出现" not in _deltas(frames)
    assert model.scripts == [["不应出现的文本"]]  # 第二轮脚本根本没被消费


# ── 验收 10:多步消息整体提交;按钮帧在 commit 后 ──

async def test_a10_multi_step_persisted_as_groups(db_session_factory):
    app = _make_app(
        [['{"intent":"订单","confidence":0.9}'],
         [("tool", [{"index": 0, "name": "query_order", "id": "c1",
                     "args": '{"order_id":"1111-1001"}'}])],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c2",
                     "args": '{"order_id":"1111-1001"}'}])],
         ["查好了。"]],
        db_sf=db_session_factory)
    async with await _client(app) as client:
        _, sid = await _turn(client, "先查订单再查物流")

    def _check():
        from app.models import Message
        with db_session_factory() as s:
            rows = (s.query(Message).filter_by(conversation_id=int(sid))
                    .order_by(Message.id).all())
            roles = [r.role for r in rows]
            assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
            assert rows[1].tool_calls[0]["id"] == "c1" and rows[2].tool_call_id == "c1"
            assert rows[3].tool_calls[0]["id"] == "c2" and rows[4].tool_call_id == "c2"
    await asyncio.to_thread(_check)


# ── 验收 11:聊天 Agent 无建单/转人工权限 ──

async def test_a11_agent_has_no_write_power(db_session_factory):
    app = _make_app(
        [['{"intent":"订单","confidence":0.9}'],  # a11 特例:售后会走 refund 临时链落 gate_fallback,本用例只验写权限
         [("tool", [{"index": 0, "name": "create_ticket", "id": "c9",
                     "args": '{"description":"x","ticket_type":"投诉"}'}])],
         [("tool", [{"index": 0, "name": "suggest_options", "id": "s1",
                     "args": '{"options":["转人工","建工单"],"ticket_type":"投诉"}'}])],
         ["建议您点击下方按钮。"]],
        db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "我现在就要你帮我建工单并转人工")
        ends = [f for f in frames if isinstance(f, dict) and f["type"] == "tool_end"]
        assert ends[0]["ok"] is False  # 伪造 create_ticket → unknown_tool
        sug = [f for f in frames if isinstance(f, dict) and f["type"] == "suggest_actions"]
        assert len(sug) == 1  # 伪工具建议正常发出

    def _check():
        from app.models import Conversation, Ticket
        with db_session_factory() as s:
            assert s.query(Ticket).count() == 0  # 没写库
            assert s.get(Conversation, int(sid)).status == "进行中"  # 没置已转人工
    await asyncio.to_thread(_check)
