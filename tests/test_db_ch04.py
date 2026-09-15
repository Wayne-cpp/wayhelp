import pytest
from sqlalchemy import text

from app.db import check_ch04_tables
from tests.dbfixtures import db_engine  # noqa: F401


def test_ch04_tables_exist(db_engine):
    check_ch04_tables(db_engine)  # 不抛即过


def test_ch04_tables_missing_raises(db_engine):
    with db_engine.connect() as conn:
        # 两张一起删:finally 重放整份 04 DDL,只删一张会 1050(low_confidence_questions 已存在)
        conn.execute(text("DROP TABLE faith_cases"))
        conn.execute(text("DROP TABLE low_confidence_questions"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch04"):
            check_ch04_tables(db_engine)
    finally:
        from tests.dbfixtures import DDL_PATHS, _split_statements
        ddl = [p for p in DDL_PATHS if p.name == "04-ddl.sql"][0]
        with db_engine.connect() as conn:
            for stmt in _split_statements(ddl.read_text(encoding="utf-8")):
                conn.execute(text(stmt))
            conn.commit()
