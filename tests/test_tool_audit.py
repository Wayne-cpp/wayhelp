import logging

from app.services.tool_audit import (
    AUDIT_DENIED, AUDIT_FAILURE, AUDIT_INVALID, AUDIT_SUCCESS, AUDIT_TIMEOUT,
    AuditEntry, write_audit,
)
from app.models import ToolAuditLog
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _entry(**over):
    base = dict(conversation_id=1, tool_call_id="c1", tool_name="query_order",
                tool_source="builtin", mcp_server=None, arguments={"order_id": "X"},
                result_summary="{}", status=AUDIT_SUCCESS, error_message=None,
                retry_count=0, duration_ms=5)
    base.update(over)
    return AuditEntry(**base)


async def test_all_five_statuses_land(db_session_factory):
    for i, st in enumerate((AUDIT_SUCCESS, AUDIT_FAILURE, AUDIT_TIMEOUT,
                            AUDIT_INVALID, AUDIT_DENIED)):
        await write_audit(db_session_factory, _entry(tool_call_id=f"c{i}", status=st))
    with db_session_factory() as s:
        got = {r.tool_call_id: r.status for r in s.query(ToolAuditLog).all()}
    assert got == {f"c{i}": st for i, st in enumerate(
        (AUDIT_SUCCESS, AUDIT_FAILURE, AUDIT_TIMEOUT, AUDIT_INVALID, AUDIT_DENIED))}


async def test_result_summary_truncated(db_session_factory):
    await write_audit(db_session_factory, _entry(result_summary="汉" * 5000), max_chars=100)
    with db_session_factory() as s:
        row = s.query(ToolAuditLog).filter_by(tool_call_id="c1").one()
    assert len(row.result_summary) < 5000
    assert row.result_summary.endswith("…[已截断]")


async def test_audit_failure_never_raises(db_session_factory, caplog):
    class BoomFactory:  # 构造即炸,模拟审计库不可用
        def __call__(self):
            raise RuntimeError("db down")
    with caplog.at_level(logging.ERROR):
        await write_audit(BoomFactory(), _entry())  # 不抛
    assert any("tool_audit" in r.message for r in caplog.records)


async def test_null_conversation_allowed(db_session_factory):
    await write_audit(db_session_factory, _entry(conversation_id=None))
    with db_session_factory() as s:
        row = s.query(ToolAuditLog).filter_by(tool_call_id="c1").one()
    assert row.conversation_id is None
