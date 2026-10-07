"""ch08 DDL 钉:两表存在、ORM 可写可读、启动探针缺表即炸。"""
import pytest
from sqlalchemy import text

from app.db import check_ch08_tables
from app.models import ToolAuditLog, ToolWriteIdempotency
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_tables_exist(db_engine):
    with db_engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SHOW TABLES"))}
    assert {"tool_audit_logs", "tool_write_idempotency"} <= tables


def test_audit_row_roundtrip(db_session_factory):
    with db_session_factory() as s:
        s.add(ToolAuditLog(
            conversation_id=None, tool_call_id="c1", tool_name="query_order",
            tool_source="builtin", mcp_server=None, arguments={"order_id": "X"},
            result_summary="{}", status="成功", error_message=None,
            retry_count=0, duration_ms=3))
        s.commit()
        row = s.query(ToolAuditLog).filter_by(tool_call_id="c1").one()
        assert row.status == "成功" and row.tool_source == "builtin"


def test_idempotency_row_roundtrip(db_session_factory):
    with db_session_factory() as s:
        s.add(ToolWriteIdempotency(idempotency_key="k" * 64,
                                   arguments_sha256="a" * 64, ticket_no="T1"))
        s.commit()
        row = s.get(ToolWriteIdempotency, "k" * 64)
        assert row.ticket_no == "T1"


def test_startup_probe_raises_when_missing(db_engine):
    with db_engine.connect() as conn:  # 两表都删:重建脚本整份重放才不会 1050
        conn.execute(text("DROP TABLE IF EXISTS tool_write_idempotency"))
        conn.execute(text("DROP TABLE IF EXISTS tool_audit_logs"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch08"):
            check_ch08_tables(db_engine)
    finally:
        from tests.dbfixtures import _split_statements, DDL_PATHS
        with db_engine.connect() as c2:
            for stmt in _split_statements(DDL_PATHS[-1].read_text(encoding="utf-8")):
                c2.execute(text(stmt))
            c2.commit()
