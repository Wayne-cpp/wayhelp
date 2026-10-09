import pytest
from sqlalchemy import text

from app.db import check_ch04_tables
from tests.dbfixtures import db_engine  # noqa: F401


def test_ch04_tables_exist(db_engine):
    check_ch04_tables(db_engine)  # 不抛即过


def test_ch04_tables_missing_raises(db_engine):
    with db_engine.connect() as conn:
        # 两张一起删:finally 重放整份 04 DDL,只删一张会 1050(low_confidence_questions 已存在)
        # ch09 起 chat_feedback 外键引用 lcq,须先关外键检查再 DROP(errno 3730)
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("DROP TABLE faith_cases"))
        conn.execute(text("DROP TABLE low_confidence_questions"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch04"):
            check_ch04_tables(db_engine)
    finally:
        from tests.dbfixtures import DDL_PATHS, _split_statements
        ddl = [p for p in DDL_PATHS if p.name == "04-ddl.sql"][0]
        alter07 = [p for p in DDL_PATHS if p.name == "07-ddl.sql"][0]
        with db_engine.connect() as conn:
            for stmt in _split_statements(ddl.read_text(encoding="utf-8")):
                conn.execute(text(stmt))
            # 04 只建原始 lcq;ch09 八列/索引/fk_lcq_review 外键由 07 的 ALTER 补回。
            # 不能用子串 "low_confidence_questions" in stmt 过滤:chat_feedback 建表语句的
            # REFERENCES 子句也含该串,整份重放会 1050;按语句前缀精确匹配 ALTER。
            for stmt in _split_statements(alter07.read_text(encoding="utf-8")):
                if stmt.startswith("ALTER TABLE low_confidence_questions"):
                    conn.execute(text(stmt))
            conn.commit()
