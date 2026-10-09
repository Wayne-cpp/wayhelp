"""ch09 飞轮:解析契约/退避/同义合并累加/新建缺口/失败退避与到限 failed。"""
import asyncio
import json

from app.services.flywheel import (
    backoff_seconds, parse_dedup_output, parse_standardize_output, process_pending,
)
from app.models import LowConfidenceQuestion, ReviewQueue
from app.config import Settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _settings(**kw):
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d",
                flywheel_retry_base_seconds=60, flywheel_retry_max_seconds=3600)
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_parse_standardize():
    assert parse_standardize_output(
        '{"normalized_question": "如何开发票?", "suggested_answer": "在订单页…"}'
    ) == ("如何开发票?", "在订单页…")
    assert parse_standardize_output("not json") is None
    assert parse_standardize_output('{"normalized_question": ""}') is None


def test_parse_dedup():
    assert parse_dedup_output('{"matched_id": 3}', {3, 5}) == 3
    assert parse_dedup_output('{"matched_id": null}', {3}) is None
    assert parse_dedup_output('{"matched_id": 99}', {3}) is False   # 不在候选集
    assert parse_dedup_output("garbage", {3}) is False


def test_backoff():
    s = _settings()
    assert backoff_seconds(s, 1) == 60
    assert backoff_seconds(s, 2) == 120
    assert backoff_seconds(s, 99) == 3600


class _FakeModel:
    """标准化/查重脚本化应答:按输入内容分流。"""
    def __init__(self, std_map, dedup_id):
        self._std = std_map
        self._dedup_id = dedup_id

    async def ainvoke(self, messages):
        text = messages[0].content
        if "待审队列候选" in text:
            return type("R", (), {"content": json.dumps(
                {"matched_id": self._dedup_id})})()
        q = self._std.get(text, ("标准化问题?", "示例答案。"))
        return type("R", (), {"content": json.dumps(
            {"normalized_question": q[0], "suggested_answer": q[1]})})()


def _seed_lcq(sf, raw="发票咋开啊", resolved="如何开具发票?"):
    with sf() as s:
        row = LowConfidenceQuestion(raw_question=raw, source="self_check",
                                    resolved_question=resolved)
        s.add(row)
        s.commit()
        return row.id


def test_new_gap_creates_review_row(db_session_factory):
    _seed_lcq(db_session_factory)
    model = _FakeModel({}, None)
    stats = asyncio.run(process_pending(_settings(), db_session_factory, model))
    assert stats["created"] == 1
    with db_session_factory() as s:
        rq = s.query(ReviewQueue).first()
        lcq = s.query(LowConfidenceQuestion).first()
        assert rq.normalized_question == "标准化问题?"
        assert rq.occurrence_count == 1 and rq.review_status == "待审"
        assert lcq.process_status == "processed"
        assert lcq.matched_review_id == rq.id


def test_synonym_merges_and_counts(db_session_factory):
    _seed_lcq(db_session_factory)
    model = _FakeModel({}, None)
    asyncio.run(process_pending(_settings(), db_session_factory, model))
    with db_session_factory() as s:
        rid = s.query(ReviewQueue).first().id
    # 第二条在首轮后落池:同批两条会被首轮一并排空(查重判 null 各建一行),
    # 后到同义经查重归并才是本测试要钉的语义
    _seed_lcq(db_session_factory, raw="开发票在哪里", resolved="在哪里开发票?")
    model2 = _FakeModel({}, rid)   # 第二条查重命中第一条
    stats = asyncio.run(process_pending(_settings(), db_session_factory, model2))
    assert stats["merged"] == 1
    with db_session_factory() as s:
        assert s.query(ReviewQueue).count() == 1
        assert s.query(ReviewQueue).first().occurrence_count == 2
        assert s.query(LowConfidenceQuestion).filter_by(
            process_status="processed").count() == 2


def test_failure_backoff_and_terminal_failed(db_session_factory):
    _seed_lcq(db_session_factory)

    class BadModel:
        async def ainvoke(self, messages):
            return type("R", (), {"content": "garbage"})()

    s = _settings(flywheel_max_attempts=2)
    asyncio.run(process_pending(s, db_session_factory, BadModel()))
    with db_session_factory() as ss:
        row = ss.query(LowConfidenceQuestion).first()
        assert row.process_status == "pending" and row.attempt_count == 1
        assert row.next_attempt_at is not None and row.last_error
        assert row.next_attempt_at > row.created_at   # DB 钟写入(spec §5.4)
        row.next_attempt_at = None   # 测试直接放行到期的等待
        ss.commit()
    asyncio.run(process_pending(s, db_session_factory, BadModel()))
    with db_session_factory() as ss:
        row = ss.query(LowConfidenceQuestion).first()
        assert row.process_status == "failed" and row.next_attempt_at is None
