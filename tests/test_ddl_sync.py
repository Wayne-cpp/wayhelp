from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# (db/init 序号, sql/ 章节名):两侧 DDL 必须逐字节一致(db/init 供 compose 自动建表,sql/ 供人工执行)
DDL_PAIRS = [("01", "ch02"), ("03", "ch03"), ("04", "ch04")]


@pytest.mark.parametrize(("init_name", "sql_stem"), DDL_PAIRS)
def test_ddl_byte_identical(init_name, sql_stem):
    assert (ROOT / "sql" / f"{sql_stem}-ddl.sql").read_bytes() == (
        ROOT / "db" / "init" / f"{init_name}-ddl.sql"
    ).read_bytes()
