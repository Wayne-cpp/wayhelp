"""faq seed 的标注式断言:验收 2 必须命中,验收 3 必须漏召回(spec §11)。"""

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

SEED_PATH = Path(__file__).resolve().parent.parent / "db" / "init" / "02-seed.sql"


@pytest.fixture()
def seeded_factory(db_engine):
    from app.db import make_session_factory
    with db_engine.connect() as conn:
        for table in ("messages", "tickets", "faq", "conversations"):
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            conn.execute(text(f"TRUNCATE TABLE {table}"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.execute(text(SEED_PATH.read_text(encoding="utf-8")))
        conn.commit()
    return make_session_factory(db_engine)


def _count_like(sf, keyword):
    with sf() as s:
        return s.execute(text(
            "SELECT COUNT(*) FROM faq WHERE question LIKE :p OR answer LIKE :p"
        ), {"p": f"%{keyword}%"}).scalar()


async def test_seed_has_no_youfei_anywhere(db_engine):
    """02-seed.sql 数据行不得出现「邮费」二字。

    计划原文断言全文件不含,但 seed 首行注释(meta 描述验收 3)本身含该词,
    自相矛盾;剥注释行后断言,与 dbfixtures._split_statements 同一约定。
    """
    data_text = "\n".join(
        ln for ln in SEED_PATH.read_text(encoding="utf-8").splitlines()
        if not ln.strip().startswith("--")
    )
    assert "邮费" not in data_text


async def test_return_policy_hit(seeded_factory):
    assert await asyncio.to_thread(_count_like, seeded_factory, "退货政策") >= 1


async def test_youfei_miss(seeded_factory):
    assert await asyncio.to_thread(_count_like, seeded_factory, "邮费") == 0
