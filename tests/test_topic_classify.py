"""归类作业核心:归组/幂等/原子性/并发锁(spec §7.3 + §10 幂等矩阵)。"""

import httpx
import pytest

from app.models import Conversation, LowConfidenceQuestion, TopicClassification
from app.services import topic_classify
from app.services.topic_classify import (TopicContractError, TopicInferenceError,
                                         TopicJobBusyError, run_classification)
from training.labels import (classifier_version, decode_scores, label_names,
                             taxonomy_version)
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

NAMES = label_names()
THRESHOLD = 0.5
MODEL_VERSION = "m" * 64
VERSION = classifier_version(MODEL_VERSION, THRESHOLD)
METADATA = {"model_version": MODEL_VERSION, "taxonomy_version": taxonomy_version(),
            "effective_threshold": THRESHOLD, "labels": NAMES,
            "decoder_version": "topic-decoder-v1"}


def _scores(hit_indexes):
    return [0.9 if i in hit_indexes else 0.01 for i in range(17)]


def _fake_client(labels_map):
    """labels_map: text -> hit 下标元组;返回 httpx.Client(MockTransport)。"""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metadata":
            return httpx.Response(200, json={
                "model_version": MODEL_VERSION, "taxonomy_version": taxonomy_version(),
                "classifier_version": VERSION, "effective_threshold": THRESHOLD,
                "labels": NAMES, "decoder_version": "topic-decoder-v1"})
        import json as _json
        texts = _json.loads(request.content.decode())["texts"]
        batch = [_scores(labels_map[t]) for t in texts]
        return httpx.Response(200, json={
            "labels": [decode_scores(s, THRESHOLD) for s in batch],
            "scores": batch, "classifier_version": VERSION})
    return httpx.Client(transport=httpx.MockTransport(handler))


def _seed_lcq(sf, rows):
    """rows: list of dict(raw_question, resolved_question, cid, mid, status)。"""
    ids = []
    with sf() as s:
        # conversation_id 有真外键,先按用到的 cid 建父行(显式 id,与 cid 对齐)
        for cid in sorted({r["cid"] for r in rows if r.get("cid") is not None}):
            s.add(Conversation(id=cid, user_id="topic-classify-test"))
        for r in rows:
            q = LowConfidenceQuestion(
                raw_question=r["raw_question"],
                resolved_question=r.get("resolved_question"),
                conversation_id=r.get("cid"), turn_message_id=r.get("mid"),
                source=r.get("source", "self_check"),
                process_status=r.get("status", "processed"))
            s.add(q)
            s.flush()
            ids.append(q.id)
        s.commit()
    return ids


def _settings(**kw):
    return make_settings(topic_inference_url="http://fake-inference", **kw)


def test_happy_path_classifies_and_writes(db_session_factory):
    _seed_lcq(db_session_factory, [
        {"raw_question": "买大了想退", "resolved_question": "买大了想退货"},
        {"raw_question": "发票怎么开", "resolved_question": None},
    ])
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: _fake_client(
                                 {"买大了想退货": (0, 2), "发票怎么开": (3,)}))
    assert out["classified_questions"] == 2 and out["skipped_questions"] == 0
    assert out["written_rows"] == 2 and out["classifier_version"] == VERSION
    with db_session_factory() as s:
        rows = s.query(TopicClassification).order_by(
            TopicClassification.question_id).all()
    assert rows[0].labels == ["退换货", "尺码"]  # 冻结顺序
    assert rows[0].classifier_version == VERSION
    assert rows[0].classifier_metadata["taxonomy_version"] == taxonomy_version()
    assert rows[1].labels == ["发票"]


def test_same_turn_rows_share_one_inference(db_session_factory):
    """同轮自动落池+👎(同 conversation_id+turn_message_id)共享一次推理与结果。"""
    _seed_lcq(db_session_factory, [
        {"raw_question": "这个太大想退", "cid": 1, "mid": 101, "source": "retrieval_low_conf"},
        {"raw_question": "这个太大想退", "cid": 1, "mid": 101, "source": "user_feedback"},
        {"raw_question": "这个太大想退", "cid": 1, "mid": 102},  # 不同轮同文本,不合并
    ])
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: _fake_client(
                                 {"这个太大想退": (0, 2)}))
    assert out["classified_questions"] == 2  # 3 行归 2 组(同轮两行合一)
    assert out["written_rows"] == 3          # 同组两行写相同结果
    with db_session_factory() as s:
        rows = s.query(TopicClassification).all()
    assert {tuple(r.labels) for r in rows} == {("退换货", "尺码")}


def test_same_version_skip_complete_groups(db_session_factory):
    ids = _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    client = _fake_client({"买大了想退": (0,)})
    run_classification(_settings(), db_session_factory, client_factory=lambda **kw: client)
    # 第二次:同版本已有完整结果 → 跳过,零写入。
    # metadata 仍会拉取(校验目标版本);推理不应发生——labels_map 为空的
    # fake client 收到任何 classify 文本即 KeyError,即为「不应发生」的探针。
    metadata_only = _fake_client({})
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: metadata_only)
    assert out["classified_questions"] == 0 and out["skipped_questions"] == 1
    assert out["written_rows"] == 0


def test_inference_down_zero_write(db_session_factory):
    _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    def dead(**kw):
        c = httpx.Client(transport=httpx.MockTransport(
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))))
        return c
    with pytest.raises(TopicInferenceError):
        run_classification(_settings(), db_session_factory, client_factory=dead)
    with db_session_factory() as s:
        assert s.query(TopicClassification).count() == 0  # 零写入


def test_second_batch_failure_zero_write(db_session_factory):
    """第二批失败,第一批也不得落库(原子性,spec §7.3/§10)。"""
    texts = [f"问题{i}号" for i in range(3)]
    _seed_lcq(db_session_factory, [{"raw_question": t} for t in texts])
    def flaky(**kw):
        def handler(request):
            if request.url.path == "/metadata":
                return _fake_client({}).get("http://x/metadata")  # 复用 metadata 响应
            import json as _json
            batch = _json.loads(request.content.decode())["texts"]
            if len(batch) < 3:  # batch_size=2:第二批
                return httpx.Response(502, json={"error": "boom"})
            s = [_scores((0,)) for _ in batch]
            return httpx.Response(200, json={
                "labels": [decode_scores(x, THRESHOLD) for x in s],
                "scores": s, "classifier_version": VERSION})
        return httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(TopicInferenceError):
        run_classification(_settings(topic_classify_batch_size=2),
                           db_session_factory, client_factory=flaky)
    with db_session_factory() as s:
        assert s.query(TopicClassification).count() == 0


def test_contract_violation_zero_write(db_session_factory):
    """响应标签与分数解码不一致 → 契约失败,零写入。"""
    _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    def bad(**kw):
        def handler(request):
            if request.url.path == "/metadata":
                return _fake_client({}).get("http://x/metadata")
            s = [_scores((0, 2))]
            return httpx.Response(200, json={
                "labels": [["物流"]], "scores": s, "classifier_version": VERSION})
        return httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(TopicContractError):
        run_classification(_settings(), db_session_factory, client_factory=bad)
    with db_session_factory() as s:
        assert s.query(TopicClassification).count() == 0


def test_job_lock_busy(db_session_factory):
    assert topic_classify._JOB_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(TopicJobBusyError):
            run_classification(_settings(), db_session_factory,
                               client_factory=lambda **kw: None)
    finally:
        topic_classify._JOB_LOCK.release()


def test_lock_released_after_failure(db_session_factory):
    """失败后锁已释放,可重试(spec §10)。"""
    _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    def dead(**kw):
        return httpx.Client(transport=httpx.MockTransport(
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))))
    with pytest.raises(TopicInferenceError):
        run_classification(_settings(), db_session_factory, client_factory=dead)
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: _fake_client({"买大了想退": (0,)}))
    assert out["classified_questions"] == 1
