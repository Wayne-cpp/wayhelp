"""Docker MySQL 测试库基建:session 级建库建表,function 级清表。

需要 Docker 在线且 `docker compose up -d` 已启动;连接失败时显式 pytest.exit,
不静默跳过(spec §11)。
"""

from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import Settings
from app.db import make_engine, make_session_factory

DDL_PATH = Path(__file__).resolve().parent.parent / "db" / "init" / "01-ddl.sql"
TABLES = ("messages", "tickets", "faq", "conversations")  # 先子后父


def _split_statements(sql_text: str) -> list[str]:
    # 01-ddl.sql 为用户手写的已知文件,语句内不含分号(已核实);剔除注释行
    statements = []
    for chunk in sql_text.split(";\n"):
        lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


@pytest.fixture(scope="session")
def db_engine():
    settings = Settings()  # 读真实 .env 的 TEST_* 字段
    try:
        admin = make_engine(settings.test_admin_database_url)
        with admin.connect() as conn:
            conn.execute(text("CREATE DATABASE IF NOT EXISTS wayhelp_test "
                              "CHARACTER SET utf8mb4"))
    except Exception as exc:
        pytest.exit(f"DB 测试需要 Docker MySQL 在线(docker compose up -d): "
                    f"{type(exc).__name__}", returncode=3)
    engine = make_engine(settings.test_database_url)
    ddl = DDL_PATH.read_text(encoding="utf-8")
    with engine.connect() as conn:
        # 用户 DDL 不带 IF NOT EXISTS,重跑会 1050;先按先子后父清场再重放,保证幂等
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table in TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        for stmt in _split_statements(ddl):
            conn.execute(text(stmt))
        conn.commit()
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session_factory(db_engine):
    sf = make_session_factory(db_engine)
    with db_engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table in TABLES:
            conn.execute(text(f"TRUNCATE TABLE {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.commit()
    return sf
