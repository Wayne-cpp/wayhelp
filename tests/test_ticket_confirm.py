"""ch08 工单预览(spec §5):挂起轮帧序 session → ticket_preview → [DONE];
确认/取消全链在 Task 11 resume 分派落地后补全。"""
import json

from app.main import create_app
from app.models import Ticket, ToolAuditLog, ToolWriteIdempotency
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401
from tests.test_ch05_acceptance import _client, _turn, _types

_SCRIPTS = [
    ['{"intent":"订单","confidence":0.9}'],          # classify(首轮 understand 跳过)
    [("tool", [{"index": 0, "name": "create_ticket", "id": "w1",
                "args": '{"description":"商品有质量问题,要求换货","ticket_type":"售后"}'}])],
    ["已为您提交售后工单,工单号见上方卡片。"],         # Task 11 接通后 main_agent 收尾
]


async def test_write_call_suspends_with_ticket_preview_frame():
    app = create_app(settings=make_settings(),
                     model=ScriptedChatModel(scripts=list(_SCRIPTS)),
                     runtime=make_runtime(tools=[]))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "帮我建个工单")
    assert _types(frames) == ["session", "ticket_preview", "[DONE]"]
    card = frames[1]
    assert card["interrupt_id"]
    assert card["ticket_type"] == "售后"
    assert "质量问题" in card["description"]


def _app(scripts, db_sf=None):
    import dataclasses
    runtime = make_runtime(tools=[])
    if db_sf is not None:
        runtime = dataclasses.replace(runtime, session_factory=db_sf)
    return create_app(settings=make_settings(),
                      model=ScriptedChatModel(scripts=scripts), runtime=runtime)


def _parse(resp):
    return [json.loads(l[5:]) if l.startswith("data:") and l[5:].strip() != "[DONE]"
            else "[DONE]" for l in resp.text.splitlines() if l.strip()]


async def _suspend(client):
    frames, sid = await _turn(client, "帮我建个工单")
    card = frames[1]
    assert card["type"] == "ticket_preview"
    return sid, card["interrupt_id"]


async def test_confirm_lands_ticket_idempotency_audit(db_session_factory):
    app = _app(list(_SCRIPTS), db_sf=db_session_factory)
    async with await _client(app) as client:
        sid, iid = await _suspend(client)
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "confirm"})
        assert resp.status_code == 200
        frames2 = _parse(resp)
        types2 = [f if isinstance(f, str) else f["type"] for f in frames2]
        assert types2[0] == "session" and types2[-1] == "[DONE]"
        assert "ticket_preview" not in types2          # 恢复成功不重发卡
        text = "".join(f.get("content", "") for f in frames2
                       if isinstance(f, dict) and f["type"] == "delta")
        assert "工单" in text
        with db_session_factory() as s:
            tickets = s.query(Ticket).all()
            assert len(tickets) == 1 and tickets[0].ticket_type == "售后"
            idem = s.query(ToolWriteIdempotency).all()
            assert len(idem) == 1 and idem[0].ticket_no == tickets[0].ticket_no
            audits = s.query(ToolAuditLog).filter_by(tool_name="create_ticket").all()
            assert [a.status for a in audits] == ["成功"]
            assert audits[0].conversation_id == int(sid)
            # 旧卡重放 → 409(幂等表有记录也不放行,spec §5.3)
            resp2 = await client.post("/v1/chat/resume", json={
                "user_id": TEST_USER_ID, "session_id": sid,
                "interrupt_id": iid, "decision": "confirm"})
            assert resp2.status_code == 409


async def test_cancel_writes_denied_audit_no_ticket(db_session_factory):
    app = _app(list(_SCRIPTS), db_sf=db_session_factory)
    async with await _client(app) as client:
        sid, iid = await _suspend(client)
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "cancel"})
        assert resp.status_code == 200
        _parse(resp)
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 0
        assert s.query(ToolWriteIdempotency).count() == 0
        audits = s.query(ToolAuditLog).filter_by(tool_name="create_ticket").all()
        assert [a.status for a in audits] == ["权限拒绝"]
        assert audits[0].arguments["description"] == "商品有质量问题,要求换货"


async def test_ticket_preview_resume_requires_decision(db_session_factory):
    app = _app(list(_SCRIPTS), db_sf=db_session_factory)
    async with await _client(app) as client:
        sid, iid = await _suspend(client)
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid, "interrupt_id": iid})
        assert resp.status_code == 409          # 字段不对按契约 409(spec §5.3)


async def test_order_selector_contract_unchanged(db_session_factory):
    """旧契约逐字保持:order_selector 必须 order_id;decision 不顶用。"""
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],
        ['{"mode":"order_specific"}'],
        ['{"queries":["保温杯退货期限"]}'],
        ["订单 1111-1001 可以退[1]。"],
    ]
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    import dataclasses
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  retriever=_FakeRetriever(_result()))
    app = create_app(settings=make_settings(),
                     model=ScriptedChatModel(scripts=scripts), runtime=runtime)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        assert _types(frames) == ["session", "order_selector", "[DONE]"]
        iid = frames[1]["interrupt_id"]
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "confirm"})   # 缺 order_id → 409
        assert resp.status_code == 409
        resp2 = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "order_id": "1111-1001"}) # 旧路径照常放行
        assert resp2.status_code == 200


async def test_ticket_confirm_survives_sqlite_reopen(db_session_factory, tmp_path):
    """SQLite 文件级持久化钉(照 test_resume_survives_sqlite_reopen 先例):
    挂起 → 关库重开新 saver → confirm 仍放行落票。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command
    from app.graph.builder import build_chat_graph
    from app.graph.nodes import GraphDeps
    from app.graph.state import new_turn_state
    from app.store_db import DbSessionStore
    from app.tools.builtin import scan_builtin_specs
    from app.tools.catalog import ToolCatalog
    catalog = ToolCatalog()
    for s in scan_builtin_specs():
        catalog.register(s)
    settings = make_settings()
    store = DbSessionStore(db_session_factory, settings.max_message_chars)
    sid = await store.create(TEST_USER_ID)
    scripts = [
        ['{"intent":"订单","confidence":0.9}'],
        [("tool", [{"index": 0, "name": "create_ticket", "id": "w1",
                    "args": '{"description":"商品有质量问题,要求换货","ticket_type":"售后"}'}])],
        ["已为您提交售后工单。"],
    ]
    deps = GraphDeps(model=ScriptedChatModel(scripts=scripts), settings=settings,
                     retriever=None, store=store, system_prompt="",
                     catalog=catalog, mcp_gateway=None,
                     session_factory=db_session_factory)
    db = str(tmp_path / "cp.db")
    config = {"configurable": {"thread_id": sid, "user_id": TEST_USER_ID}}
    iid = None
    async with AsyncSqliteSaver.from_conn_string(db) as cp:
        graph = build_chat_graph(deps, cp)
        async for _ in graph.astream(new_turn_state("帮我建个工单"), config,
                                     stream_mode=["custom"]):
            pass
        st = await graph.aget_state(config)
        for task in st.tasks:
            for intr in task.interrupts:
                if (getattr(intr, "value", None) or {}).get("type") == "ticket_preview":
                    iid = intr.id
    assert iid
    async with AsyncSqliteSaver.from_conn_string(db) as cp2:   # 关库重开
        graph2 = build_chat_graph(deps, cp2)
        async for _ in graph2.astream(Command(resume={iid: {"decision": "confirm"}}),
                                      config, stream_mode=["custom"]):
            pass
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 1
