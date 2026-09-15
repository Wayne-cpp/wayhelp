"""faith_cases upsert 复发语义(ch04 T12):一题一行;已解决复发退回未解决。"""

from datetime import datetime

from app.models import FaithCase
from evals.run_retrieval_compare import upsert_faith_case
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_upsert_insert_then_recurrence(db_session_factory):
    with db_session_factory() as s:
        upsert_faith_case(s, eval_id="A43", bucket="A_policy", query="q",
                          answer="a1", reason="r1", citations=[{"n": 1}], judge_model="m")
        s.commit()
    with db_session_factory() as s:  # 人工标已解决
        row = s.query(FaithCase).one()
        row.status = "已解决"; row.resolution = "补了文档"
        row.resolved_at = datetime(2026, 9, 15, 12, 0, 0)
        s.commit()
    with db_session_factory() as s:  # 复发:同一 eval_id 再判编造
        upsert_faith_case(s, eval_id="A43", bucket="A_policy", query="q",
                          answer="a2", reason="r2", citations=[{"n": 2}], judge_model="m")
        s.commit()
    with db_session_factory() as s:
        row = s.query(FaithCase).one()
        assert row.seen_count == 2 and row.answer == "a2"
        assert row.status == "未解决" and row.resolution is None   # 复发退回,处置清空
        assert row.resolved_at == datetime(2026, 9, 15, 12, 0, 0)  # DDL 语义:复发后仍保留
