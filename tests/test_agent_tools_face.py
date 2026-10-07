"""ch08 工具面装配与批次契约(spec §1.4/§2):business 面含 create_ticket、
物流内置已下线;非法批次(写不在最后)整批拦下、零执行、逐 call 审计、模型恢复收尾。"""
import json

from app.main import create_app
from app.models import ToolAuditLog
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401
from tests.test_ch05_acceptance import _client, _turn, _types


async def test_business_face_has_create_ticket_no_logistics():
    model = ScriptedChatModel(scripts=[
        ['{"intent":"订单","confidence":0.9}'],   # classify(首轮 understand 跳过)
        ["好的。"],                                # main_agent 一步收尾
    ])
    app = create_app(settings=make_settings(), model=model, runtime=make_runtime(tools=[]))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "我的订单怎么样了")
    assert _types(frames) == ["session", "delta", "[DONE]"]
    bound = set(model.bound or [])
    assert {"query_order", "query_product", "query_faq", "create_ticket",
            "suggest_options"} <= bound
    assert "query_logistics" not in bound          # 内置下线,MCP 未配置时缺席(spec §4.1)


async def test_invalid_batch_write_not_last_recovers(db_session_factory):
    scripts = [
        ['{"intent":"订单","confidence":0.9}'],
        [("tool", [{"index": 0, "name": "create_ticket", "id": "c1",
                    "args": '{"description":"商品质量问题","ticket_type":"售后"}'},
                   {"index": 1, "name": "query_order", "id": "c2",
                    "args": '{"order_id":"1111-1001"}'}])],
        ["好的,请问还有什么可以帮您?"],
    ]
    import dataclasses
    runtime = dataclasses.replace(make_runtime(tools=[]), session_factory=db_session_factory)
    app = create_app(settings=make_settings(), model=ScriptedChatModel(scripts=scripts),
                     runtime=runtime)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "帮我建个工单顺便查下订单")
    assert _types(frames) == ["session", "delta", "[DONE]"]  # 零执行:无 tool_start
    with db_session_factory() as s:
        rows = s.query(ToolAuditLog).filter(ToolAuditLog.tool_call_id.in_(["c1", "c2"])).all()
    assert len(rows) == 2 and {r.status for r in rows} == {"校验拦下"}
