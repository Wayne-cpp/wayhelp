# tests/test_interrupted_turn_persistence.py — ch07 Task 16(中断轮持久化 P1 修复)
"""方向 A 钉:prepare 先落库本轮用户消息(interrupt 挂起/被新消息取代也不丢),
log_turn commit 幂等(用户行已在库只补 assistant 行,db_id 显式传递禁止内容对齐)。

覆盖:
a) interrupt → 新消息取代 → MySQL(经 DbSessionStore)含 turn1 用户行;
b) GET /api/conversations/{id}/messages 回载含 turn1;
c) 层装视图/understand 历史含 turn1(带 prepare 盖章的 db_id,resume 完成路径);
d) 摘要覆盖区间(fetch_span_texts (0, layer1_from])含 turn1。

harness:DB 版照 test_chat_api 先例(注入 session_factory → 裁决 B 换 DbSessionStore),
脚本消费序同 test_refund_flow(挂起轮 classify → scope;resume 轮 expand → main_agent)。
"""

import asyncio
import dataclasses
import logging

from pydantic import Field

from app.main import create_app
from app.prompts.service import CHITCHAT_REPLY
from tests.conftest import TEST_USER_ID, ScriptedChatModel, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册)
from tests.test_ch05_acceptance import (
    _FakeRetriever, _client, _make_app, _result, _turn, _types,
)

TURN1 = "我要把之前买的保温杯退款"
TURN2 = "算了,先不退了"


class _PromptSpyModel(ScriptedChatModel):
    """记录每次 _generate 收到的末条消息文本(摘要/understand prompt 断言用)。"""

    prompts: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages[-1].content if messages else "")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def _rows(store, sid):
    """经 DbSessionStore 读 messages 表全量行(不滤 content),返回 (id, role, content)。"""
    recs = await store.list_checkpoint_records(sid, TEST_USER_ID)
    return [(int(r.id), r.role, r.content) for r in recs]


def _config(sid):
    return {"configurable": {"thread_id": sid, "user_id": TEST_USER_ID}}


# ── a) + b):被新消息取代的中断轮,用户行仍在账本、仍在回载 ──

async def test_interrupt_replaced_keeps_user_row_in_mysql_and_reload(db_session_factory):
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],   # 轮1 classify(首轮 understand 跳过)
        ['{"mode":"order_specific"}'],                # 轮1 scope → 挂起
        ['{"intent":"闲聊","confidence":0.95}'],      # 轮2 取代挂起轮(仍无历史,understand 跳过)
    ]
    app = _make_app(scripts, retriever=_FakeRetriever(_result()), db_sf=db_session_factory)
    store = app.state.store  # 裁决 B 后是 DbSessionStore
    async with await _client(app) as client:
        frames, sid = await _turn(client, TURN1)
        assert _types(frames) == ["session", "order_selector", "[DONE]"]  # 挂起:无 log_turn
        # (a) 挂起当下:本轮用户行已落 MySQL(P1 修复点:旧版此刻库里空)
        assert [(r[1], r[2]) for r in await _rows(store, sid)] == [("user", TURN1)]
        frames2, _ = await _turn(client, TURN2, sid)
        assert "order_selector" not in _types(frames2) and _types(frames2)[-1] == "[DONE]"
        # (b) 回载:首条是 turn1 用户行,其后才是取代轮
        r = await client.get(f"/api/conversations/{sid}/messages",
                             params={"user_id": TEST_USER_ID})
        assert r.status_code == 200
        assert [(m["role"], m["content"]) for m in r.json()] == [
            ("user", TURN1), ("user", TURN2), ("assistant", CHITCHAT_REPLY)]
        lst = await client.get("/api/conversations", params={"user_id": TEST_USER_ID})
        assert lst.json()[0]["preview"] == TURN1


# ── c):resume 完成路径——prepare 盖章的 db_id 进 checkpoint,历史含 turn1 ──

async def test_resume_path_history_carries_prepare_db_id(db_session_factory, caplog):
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],   # 轮1 classify → scope → 挂起
        ['{"mode":"order_specific"}'],
        ['{"queries":[]}'],                           # resume:expand 罐头
        ["订单 1111-1001 在 7 天无理由期内,可以退。"],  # resume:main_agent 收尾
        ['{"resolved_query": ""}'],                   # 轮3 understand(有历史,透传)
        ['{"intent":"闲聊","confidence":0.95}'],      # 轮3 classify
    ]
    app = _make_app(scripts, retriever=_FakeRetriever(_result()), db_sf=db_session_factory)
    store = app.state.store
    async with await _client(app) as client:
        frames, sid = await _turn(client, TURN1)
        sel = frames[1]
        rows = await _rows(store, sid)
        assert [(r[1], r[2]) for r in rows] == [("user", TURN1)]  # 挂起即落库
        u1_id = rows[0][0]
        # 挂起态 checkpoint:本轮输入已带 prepare 落库的 db_id(显式传递,非内容对齐)
        st = await app.state.chat_service._graph.aget_state(_config(sid))
        first = st.values["turn_messages"][0]
        assert first.content == TURN1
        assert first.additional_kwargs.get("db_id") == u1_id
        # resume 完成点卡:用户消息当初 prepare 已落库 → 只补 assistant 行,不重复插
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert resp.status_code == 200
        rows2 = await _rows(store, sid)
        assert [(r[1], r[2]) for r in rows2] == [
            ("user", TURN1), ("assistant", "订单 1111-1001 在 7 天无理由期内,可以退。")]
        assert rows2[1][0] > rows2[0][0]               # assistant 行后插
        # 轮3:understand 层装历史含 turn1(带 db_id,无需 reconcile)
        with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
            await _turn(client, "在吗", sid)
        assert "history_ctx" in caplog.text and TURN1 in caplog.text
        assert "context_reconcile_failed" not in caplog.text
        st2 = await app.state.chat_service._graph.aget_state(_config(sid))
        m0 = st2.values["messages"][0]
        assert m0.content == TURN1 and m0.additional_kwargs.get("db_id") == u1_id


# ── d):摘要覆盖区间含被取代的 turn1(摘要器读 MySQL 区间,非 checkpoint)──

async def test_summary_span_includes_interrupted_turn(db_session_factory):
    model = _PromptSpyModel(scripts=[
        ['{"intent":"退款退货","confidence":0.9}'],   # 轮1 classify → scope → 挂起
        ['{"mode":"order_specific"}'],
        ['{"intent":"闲聊","confidence":0.95}'],      # 轮2 取代并完成
        ["用户要退保温杯订单。"],                     # 摘要模型罐头
    ])
    runtime = dataclasses.replace(make_runtime(tools=[]), retriever=_FakeRetriever(_result()),
                                  session_factory=db_session_factory)
    app = create_app(settings=make_settings(), model=model, runtime=runtime)
    store = app.state.store
    async with await _client(app) as client:
        frames, sid = await _turn(client, TURN1)
        assert _types(frames)[1] == "order_selector"   # 挂起
        frames2, _ = await _turn(client, TURN2, sid)   # 取代并完成
        assert _types(frames2)[-1] == "[DONE]"
    rows = await _rows(store, sid)
    assert [r[1] for r in rows] == ["user", "user", "assistant"]  # 账本含被取代的 turn1
    upto = rows[-1][0]                                 # 完成轮 assistant 行 = 轮边界
    assert await store.move_layer1_from(sid, TEST_USER_ID, upto)
    app.state.summary_runner.maybe_trigger(sid, TEST_USER_ID)
    await asyncio.sleep(0.1)                           # 后台摘要落地
    summ_prompts = [p for p in model.prompts if "待压缩对话" in p]
    assert summ_prompts and TURN1 in summ_prompts[0]   # 区间 (0, upto] 含 turn1 原文
    assert TURN2 in summ_prompts[0]
    meta = await store.get_context_meta(sid, TEST_USER_ID)
    assert meta.summary_upto == upto


# ── 单元:commit_turn 幂等(用户行已由 prepare 落库)──

async def test_commit_turn_idempotent_with_prepared_user_row_db(db_session_factory):
    from app.sessions import StoredMessage
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    uid = await store.append_user_message(sid, "订单A1001能退吗")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001能退吗"),
                                      StoredMessage("assistant", "可以退")],
                                user_row_id=int(uid))
    assert r.source_message_id == uid                  # 按钮仍绑本轮用户行
    ids = [int(x) for x in r.message_ids]
    assert ids[0] == int(uid) and ids[1] > ids[0]
    recs = await store.list_checkpoint_records(sid, "u1")
    assert [x.role for x in recs] == ["user", "assistant"]  # 用户行不重复插


async def test_memory_append_user_and_idempotent_commit():
    from app.sessions import InMemorySessionStore, StoredMessage
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u1")
    uid = await store.append_user_message(sid, "订单A1001能退吗")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001能退吗"),
                                      StoredMessage("assistant", "可以退")],
                                user_row_id=int(uid))
    assert r.source_message_id == uid
    snap = await store.snapshot(sid)
    assert [(m.role, m.content) for m in snap] == [
        ("user", "订单A1001能退吗"), ("assistant", "可以退")]
    recs = await store.list_checkpoint_records(sid, "u1")
    assert [x.role for x in recs] == ["user", "assistant"]
    # 完成轮末尾仍是合法层1边界(id 与行对齐未被幂等路径打乱)
    assert await store.move_layer1_from(sid, "u1", int(r.message_ids[-1]))


async def test_memory_trim_drops_dangling_head_with_first_turn():
    """悬垂用户行(prepare 落库、轮未完成)参与超限裁剪:随首个完整轮整体裁掉,
    不把 assistant 行错切到轮首(行/id 映射不串位)。"""
    from app.sessions import InMemorySessionStore, StoredMessage
    store = InMemorySessionStore(10, 8, 8000)          # limit = max(8, 5+3) = 8
    sid = await store.create("u1")
    await store.append_user_message(sid, "挂起没答")   # 悬垂头
    for i in (2, 3, 4, 5):
        await store.commit_turn(sid, [StoredMessage("user", f"问{i}"),
                                      StoredMessage("assistant", f"答{i}")])
    snap = await store.snapshot(sid)
    assert [m.content for m in snap] == ["问3", "答3", "问4", "答4", "问5", "答5"]
    msgs = await store.list_messages(sid, "u1")
    assert [m.content for m in msgs] == ["问3", "答3", "问4", "答4", "问5", "答5"]
