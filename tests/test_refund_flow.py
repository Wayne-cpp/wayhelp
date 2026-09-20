# tests/test_refund_flow.py
"""ch06 退款子流程集成钉:缺号挂起→订单卡→按 interrupt ID resume→落库一轮(spec §13)。

脚本消费序(ScriptedChatModel 按模型调用顺序逐段消费):
- 挂起轮:classify → refund_scope →(refund_prepare 零模型调用)挂起;
- resume 轮:不重跑 understand/classify/refund_scope(spec §2),refund_prepare 从头
  重跑后 interrupt() 返回选单值 → refund_policy expand → main_agent 每步一段;
- 挂起轮不落 messages → 下一轮仍是「无历史」首轮,understand 跳过(裁决见 dev-notes)。
"""

import json

from app.main import create_app
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.test_ch05_acceptance import _FakeRetriever, _client, _result, _turn, _types


def _app(scripts, retriever, **over):
    import dataclasses
    runtime = dataclasses.replace(make_runtime(tools=[]), retriever=retriever)
    return create_app(settings=make_settings(**over),
                      model=ScriptedChatModel(scripts=scripts), runtime=runtime)


def _parse_sse(resp):
    return [json.loads(l[5:]) if l.startswith("data:") and l[5:].strip() != "[DONE]"
            else "[DONE]" for l in resp.text.splitlines() if l.strip()]


async def test_missing_order_suspends_then_resume_completes_single_turn():
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],      # classify(首轮无历史,understand 跳过)
        ['{"mode":"order_specific"}'],                  # refund_scope
        # resume 后重跑 refund_prepare→refund_policy:expand 罐头
        ['{"queries":["保温杯退货期限是多久","退货需要承担运费吗"]}'],
        # main_agent 第一步:先建议按钮(最终答复后脚本不再被消费,顺序不可反)
        [("tool", [{"index": 0, "name": "suggest_options", "id": "s1",
                    "args": '{"options":["申请退款"]}'}])],
        # main_agent 收尾答复
        ["订单 1111-1001 在 7 天无理由期内,可以退[1]。请点下方按钮提交退款申请。"],
    ]
    app = _app(scripts, _FakeRetriever(_result()))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        types = _types(frames)
        assert types == ["session", "order_selector", "[DONE]"]  # 挂起轮无 delta/无按钮
        sel = frames[1]
        assert sel["interrupt_id"]
        assert [o["order_id"] for o in sel["orders"]][0] == "1111-1001"
        # resume
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert resp.status_code == 200
        frames2 = _parse_sse(resp)
        types2 = [f if isinstance(f, str) else f["type"] for f in frames2]
        assert types2[0] == "session" and types2[-1] == "[DONE]"
        assert "order_selector" not in types2            # 恢复成功不重发卡
        text = "".join(f.get("content", "") for f in frames2
                       if isinstance(f, dict) and f["type"] == "delta")
        assert "可以退" in text
        sug = [f for f in frames2 if isinstance(f, dict) and f["type"] == "suggest_actions"]
        assert sug and sug[0]["options"][0]["action"] == "refund_form"
        assert sug[0]["options"][0]["order_id"] == "1111-1001"
        # 完成轮建立跨轮订单焦点:绑定所选订单与本轮用户消息(spec §10)
        snap = await app.state.chat_service._graph.aget_state(
            {"configurable": {"thread_id": sid, "user_id": TEST_USER_ID}})
        assert snap.values["active_order"] == {
            "order_id": "1111-1001",
            "source_message_id": snap.values["source_message_id"]}
        # 旧卡重放 → 409(已完成图无 pending)
        resp2 = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert resp2.status_code == 409


async def test_resume_markers_in_tool_step_text_still_cites():
    """引用判定看跨步累计可见文本(dev-notes Task 13 观察→FIX C):真模型形态是
    步 1 正文带 [n] 角标 + suggest_options 同步调用,步 2 短收尾无角标——若只看
    末步文本,用户已看到死角标却永不出引用卡。SSE 必须仍发 citations 帧。"""
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],      # classify(首轮无历史,understand 跳过)
        ['{"mode":"order_specific"}'],                  # refund_scope
        ['{"queries":[]}'],                             # resume:expand 罐头
        # main_agent 步 1:正文角标 + suggest_options 同一步(真实失败形状)
        ["订单 1111-1001 在 7 天无理由期内,可以退[1]。",
         ("tool", [{"index": 0, "name": "suggest_options", "id": "s1",
                    "args": '{"options":["申请退款"]}'}])],
        # 步 2:无角标短收尾
        ["请点击下方按钮提交退款申请。"],
    ]
    app = _app(scripts, _FakeRetriever(_result()))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        sel = frames[1]
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": sel["interrupt_id"], "order_id": "1111-1001"})
        assert resp.status_code == 200
        frames2 = _parse_sse(resp)
        types2 = [f if isinstance(f, str) else f["type"] for f in frames2]
        # 可见流不受影响:两步正文都外发
        text = "".join(f.get("content", "") for f in frames2
                       if isinstance(f, dict) and f["type"] == "delta")
        assert "可以退[1]" in text and "请点击下方按钮" in text
        # 角标在步 1:citations 帧必须发,且在最后一个 delta 之后、[DONE] 之前
        assert "citations" in types2
        cit = next(f for f in frames2 if isinstance(f, dict) and f["type"] == "citations")
        assert cit["citations"][0]["ref_no"] == 1
        assert types2.index("citations") > max(
            i for i, t in enumerate(types2) if t == "delta")
        assert types2[-1] == "[DONE]"
        # 建议按钮不受影响
        sug = [f for f in frames2 if isinstance(f, dict) and f["type"] == "suggest_actions"]
        assert sug and sug[0]["options"][0]["action"] == "refund_form"


async def test_resume_survives_sqlite_reopen():
    """挂起态落 SQLite 文件;关库重开后同一 interrupt ID 仍可恢复(spec §13)。"""
    import dataclasses
    import os
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from app.graph.builder import build_chat_graph
    from app.graph.nodes import GraphDeps
    from app.prompts.service import SERVICE_SYSTEM_PROMPT
    db_path = "./data/test_ch06_resume.db"
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],
        ['{"mode":"order_specific"}'],
        ['{"queries":[]}'],
        ["订单 1111-1003 是定制商品,不支持 7 天无理由退货[1]。"],
    ]
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  retriever=_FakeRetriever(_result()))
    settings = make_settings(checkpoint_db_path=db_path)
    model = ScriptedChatModel(scripts=scripts)
    app = create_app(settings=settings, model=model, runtime=runtime)
    # 换成文件级 checkpointer 的图(与生产同构;测试后清理文件)
    async with AsyncSqliteSaver.from_conn_string(db_path) as cp:
        service = app.state.chat_service
        deps = GraphDeps(model=model, settings=settings,
                         retriever=app.state.retriever, store=app.state.store,
                         system_prompt=SERVICE_SYSTEM_PROMPT)
        service.set_graph(build_chat_graph(deps, cp))
        async with await _client(app) as client:
            frames, sid = await _turn(client, "这个能退吗")
            sel = frames[1]
    # 关库重开(模拟进程重启):新 saver 实例,同文件同 thread
    async with AsyncSqliteSaver.from_conn_string(db_path) as cp2:
        service.set_graph(build_chat_graph(deps, cp2))
        async with await _client(app) as client:
            resp = await client.post("/v1/chat/resume", json={
                "user_id": TEST_USER_ID, "session_id": sid,
                "interrupt_id": sel["interrupt_id"], "order_id": "1111-1003"})
            assert resp.status_code == 200
            assert "定制商品" in resp.text
    for suffix in ("", "-shm", "-wal"):
        try:
            os.remove(f"{db_path}{suffix}")
        except FileNotFoundError:
            pass


async def test_new_message_overrides_suspension_and_stale_card_409():
    """挂起后发新消息:新轮正常完成;旧卡 ID 立即失效(409);新轮再挂起时旧卡仍 409。"""
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],   # 轮1:classify → scope → 挂起
        ['{"mode":"order_specific"}'],
        # 轮2:挂起轮不落 messages → 仍无历史,understand 跳过(红绿钉裁决,见 dev-notes)
        ['{"intent":"闲聊","confidence":0.95}'],      # 轮2:闲聊完成
        ['{"resolved_query": ""}'],                    # 轮3:轮2 已落库 → understand 罐头
        ['{"intent":"退款退货","confidence":0.9}'],   # 轮3:再次挂起(新 interrupt ID)
        ['{"mode":"order_specific"}'],
        # 末次 resume(新卡):expand + main_agent 收尾
        ['{"queries":[]}'],
        ["已收到您的退款诉求。"],
    ]
    app = _app(scripts, _FakeRetriever(_result()))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        old_sel = frames[1]
        frames2, _ = await _turn(client, "算了,打个招呼", sid)
        assert "order_selector" not in _types(frames2)
        # 旧卡失效
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": old_sel["interrupt_id"], "order_id": "1111-1001"})
        assert r.status_code == 409
        # 轮3 再次挂起:产生新 ID,旧卡依旧 409
        frames3, _ = await _turn(client, "那个能退吗", sid)
        new_sel = frames3[1]
        assert new_sel["interrupt_id"] != old_sel["interrupt_id"]
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": old_sel["interrupt_id"], "order_id": "1111-1001"})
        assert r.status_code == 409
        # 新卡可用:ID 校验通过 → 200
        r = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": new_sel["interrupt_id"], "order_id": "1111-1001"})
        assert r.status_code == 200
