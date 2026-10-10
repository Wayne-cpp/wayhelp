"""主题分布统计口径(spec §7.4):轮次去重/多标签计数/空表/stale/版本冲突。"""

import pytest

from app.models import Conversation, LowConfidenceQuestion, TopicClassification
from app.services.topic_classify import (TopicVersionConflictError,
                                         get_distribution)
from training.labels import label_names, taxonomy_version
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

V1 = "a" * 64
META_OK = {"model_version": "m", "taxonomy_version": taxonomy_version(),
           "effective_threshold": 0.5, "labels": label_names(),
           "decoder_version": "topic-decoder-v1"}
META_STALE = {**META_OK, "taxonomy_version": "0" * 64}


def _mk(sf, lcq_kw, tc=None):
    """建一行 lcq(可选挂一条分类结果)。

    conversation_id 有真外键,先按需补 conversations 父行
    (同 test_topic_classify._seed_lcq 桩惯例,修桩不修语义)。
    """
    with sf() as s:
        cid = lcq_kw.get("conversation_id")
        if cid is not None and s.get(Conversation, cid) is None:
            s.add(Conversation(id=cid, user_id="topic-distribution-test"))
            s.flush()
        q = LowConfidenceQuestion(raw_question=lcq_kw.pop("raw_question", "q"),
                                  source="self_check", process_status="processed",
                                  **lcq_kw)
        s.add(q)
        s.flush()
        if tc is not None:
            s.add(TopicClassification(question_id=q.id, labels=tc["labels"],
                                      classifier_version=tc.get("v", V1),
                                      classifier_metadata=tc.get("meta", META_OK)))
        s.commit()
        return q.id


def test_empty_table(db_session_factory):
    # 计划草案写 _empty_sf;按括注直接用 db_session_factory(fixture 自带清表)
    out = get_distribution(make_settings(), db_session_factory)
    assert out["total"] == 0 and out["classified"] == 0
    assert out["classifier_version"] is None and out["stale"] is False
    assert len(out["counts"]) == 17 and all(c["count"] == 0 for c in out["counts"])


def test_multilabel_counts_and_turn_dedup(db_session_factory):
    sf = db_session_factory
    # 同轮两行共享结果:只算一个问题;多标签进两个横条
    _mk(sf, {"raw_question": "买大了想退", "conversation_id": 1, "turn_message_id": 1},
        tc={"labels": ["退换货", "尺码"]})
    _mk(sf, {"raw_question": "买大了想退", "conversation_id": 1, "turn_message_id": 1},
        tc={"labels": ["退换货", "尺码"]})
    _mk(sf, {"raw_question": "发票呢"}, tc={"labels": ["发票"]})
    _mk(sf, {"raw_question": "没人理我"})  # 未归类
    out = get_distribution(make_settings(), sf)
    assert out["total"] == 3 and out["classified"] == 2 and out["unclassified"] == 1
    counts = {c["label"]: c["count"] for c in out["counts"]}
    assert counts["退换货"] == 1 and counts["尺码"] == 1 and counts["发票"] == 1
    assert sum(counts.values()) == 3 > out["classified"]  # 多标签计数可大于已归类数
    assert out["classifier_version"] == V1 and out["stale"] is False
    # 降序,计数并列按权威顺序
    assert [c["label"] for c in out["counts"]][:3] == ["退换货", "尺码", "发票"]


def test_group_incomplete_counts_unclassified(db_session_factory):
    sf = db_session_factory
    _mk(sf, {"raw_question": "q", "conversation_id": 2, "turn_message_id": 5},
        tc={"labels": ["物流"]})
    _mk(sf, {"raw_question": "q", "conversation_id": 2, "turn_message_id": 5})  # 同组缺结果
    out = get_distribution(make_settings(), sf)
    assert out["classified"] == 0 and out["unclassified"] == 1
    assert {c["label"]: c["count"] for c in out["counts"]}["物流"] == 0


def test_stale_taxonomy_zeroes_counts(db_session_factory):
    sf = db_session_factory
    _mk(sf, {"raw_question": "q"}, tc={"labels": ["物流"], "meta": META_STALE})
    out = get_distribution(make_settings(), sf)
    assert out["stale"] is True and out["classified"] == 0 and out["unclassified"] == 1
    assert all(c["count"] == 0 for c in out["counts"])
    assert out["classifier_version"] == V1  # 旧版本信息保留


def test_version_conflict_409(db_session_factory):
    sf = db_session_factory
    _mk(sf, {"raw_question": "a"}, tc={"labels": ["物流"], "v": V1})
    _mk(sf, {"raw_question": "b"}, tc={"labels": ["发票"], "v": "b" * 64})
    with pytest.raises(TopicVersionConflictError):
        get_distribution(make_settings(), sf)
