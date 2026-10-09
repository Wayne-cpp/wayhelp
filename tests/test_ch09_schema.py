"""ch09 DDL/模型契约:表列对齐 DDL、启动校验缺表报错。"""
import pytest
from sqlalchemy import text

from app.db import check_ch09_tables
from app.models import ChatFeedback, EvalRun, LowConfidenceQuestion, ReviewQueue
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_models_have_ch09_columns():
    lcq = {c.name for c in LowConfidenceQuestion.__table__.columns}
    assert {"retrieved_chunks", "resolved_question", "turn_message_id",
            "matched_review_id", "process_status", "attempt_count",
            "next_attempt_at", "last_error"} <= lcq
    rq = {c.name for c in ReviewQueue.__table__.columns}
    assert {"normalized_question", "ai_suggested_answer", "occurrence_count",
            "review_status", "approved_answer", "approved_at",
            "knowledge_chunk_ids", "last_write_error"} <= rq
    assert {c.name for c in ChatFeedback.__table__.columns} >= {
        "conversation_id", "assistant_message_id", "turn_message_id",
        "sentiment", "low_confidence_question_id"}
    assert {c.name for c in EvalRun.__table__.columns} >= {
        "run_id", "triggered_by", "dataset_size", "corpus_mode",
        "corpus_version", "dataset_version", "metrics"}


def test_check_ch09_tables_passes(db_engine):
    check_ch09_tables(db_engine)  # 不抛即过


def test_check_ch09_tables_fails_when_missing(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("DROP TABLE chat_feedback"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch09"):
            check_ch09_tables(db_engine)
    finally:
        from pathlib import Path
        ddl = (Path(__file__).resolve().parent.parent
               / "db" / "init" / "07-ddl.sql").read_text(encoding="utf-8")
        import tests.dbfixtures as fx
        with db_engine.connect() as conn:
            for stmt in fx._split_statements(ddl):  # 重放 07 恢复现场
                if "chat_feedback" in stmt:
                    conn.execute(text(stmt))
            conn.commit()


def test_review_status_enum_values(db_session_factory):
    with db_session_factory() as s:
        s.add(ReviewQueue(normalized_question="如何申请开发票?"))
        s.commit()
        row = s.query(ReviewQueue).first()
        assert row.review_status == "待审"
        assert row.occurrence_count == 1
