"""ch10 topic_classifications 表:DDL 配对、启动校验、ORM 往返。"""

import pytest
from sqlalchemy import text

from app.db import check_ch10_tables
from app.models import LowConfidenceQuestion, TopicClassification
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_check_ch10_tables_passes(db_engine):
    check_ch10_tables(db_engine)  # 不抛即过


def test_orm_roundtrip(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        lcq = LowConfidenceQuestion(raw_question="买大了想退", source="self_check",
                                    process_status="processed")
        s.add(lcq)
        s.flush()
        s.add(TopicClassification(
            question_id=lcq.id, labels=["退换货", "尺码"],
            classifier_version="ab" * 32,
            classifier_metadata={"model_version": "m", "taxonomy_version": "t",
                                 "effective_threshold": 0.5, "decoder_version": "d"}))
        s.commit()
    with sf() as s:
        row = s.query(TopicClassification).filter_by(question_id=lcq.id).one()
        assert row.labels == ["退换货", "尺码"]
        assert len(row.classifier_version) == 64
        assert row.classifier_metadata["decoder_version"] == "d"
        assert row.classified_at is not None
        # uk_question_id 幂等:同 question_id 再插必撞唯一键
        s.add(TopicClassification(question_id=lcq.id, labels=["物流"],
                                  classifier_version="cd" * 32,
                                  classifier_metadata={}))
        with pytest.raises(Exception):
            s.flush()
