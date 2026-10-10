"""ch10 旁路批量归类作业(spec §7.3)。

时序:非阻塞作业锁 → 拉推理 metadata 定目标版本 → 一致性读事务取快照并归组
→ 关事务后分批推理(全暂存内存、逐批校验契约)→ 全部成功才开唯一写事务原子提交。
任何推理/契约失败 503 零写入;DB 失败异常上抛 500 回滚。
"""

from __future__ import annotations

import math
import threading
from datetime import datetime

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.models import LowConfidenceQuestion, TopicClassification
from training.labels import (DECODER_VERSION, canonical_json, classifier_version,
                             decode_scores, label_names, taxonomy_version)

_JOB_LOCK = threading.Lock()  # 进程内独占;单 Uvicorn worker 前提(spec §7.3)


class TopicError(Exception):
    code = "topic_error"


class TopicJobBusyError(TopicError):
    code = "job_busy"


class TopicInferenceError(TopicError):
    code = "inference_unavailable"


class TopicContractError(TopicError):
    code = "inference_contract"


class TopicVersionConflictError(TopicError):
    code = "classification_version_conflict"


def question_key(row: LowConfidenceQuestion) -> tuple:
    """问题键:双锚点齐 → 轮次;缺一 → 行本身(spec §1)。"""
    if row.conversation_id is not None and row.turn_message_id is not None:
        return ("turn", row.conversation_id, row.turn_message_id)
    return ("row", row.id)


def _fetch_metadata(client: httpx.Client, base_url: str) -> dict:
    try:
        resp = client.get(f"{base_url}/metadata")
        resp.raise_for_status()
        meta = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise TopicInferenceError(f"推理服务 metadata 不可达:{type(exc).__name__}") from exc
    if not isinstance(meta, dict):
        raise TopicContractError("metadata 响应须为 JSON 对象")
    if (meta.get("labels") != label_names()
            or meta.get("taxonomy_version") != taxonomy_version()
            or meta.get("decoder_version") != DECODER_VERSION):
        raise TopicContractError("推理服务 metadata 与权威类目表不一致")
    th = meta.get("effective_threshold")
    if isinstance(th, bool) or not isinstance(th, (int, float)) or not 0 < th < 1:
        raise TopicContractError("effective_threshold 非法")  # (0,1) 有限数:inf/NaN 均不满足区间
    if not isinstance(meta.get("model_version"), str) or not meta["model_version"]:
        raise TopicContractError("model_version 缺失")
    return meta


def _snapshot(sf) -> list[tuple[LowConfidenceQuestion, TopicClassification | None]]:
    """一致性读事务快照:lcq LEFT JOIN topic_classifications;读完关事务再发网络请求。"""
    with sf() as s:
        rows = s.execute(
            select(LowConfidenceQuestion, TopicClassification)
            .outerjoin(TopicClassification,
                       TopicClassification.question_id == LowConfidenceQuestion.id)
            .order_by(LowConfidenceQuestion.id)).all()
        s.expunge_all()
    return [(r[0], r[1]) for r in rows]


def _representative_text(rows: list[LowConfidenceQuestion]) -> str | None:
    """组内按 id 升序:先找第一个非空 resolved_question,全空回退第一个非空 raw_question。"""
    ordered = sorted(rows, key=lambda r: r.id)
    for field in ("resolved_question", "raw_question"):
        for r in ordered:
            text = (getattr(r, field) or "").strip()
            if text:
                return text
    return None


def _result_is_current(tc: TopicClassification | None, target_version: str,
                       store_meta_canonical: str, names: set[str]) -> bool:
    """整组跳过的单行判定(spec §7.3:已有合法结果 + 版本 == 目标 + metadata 规范化一致)。"""
    if tc is None or tc.classifier_version != target_version:
        return False
    labels = tc.labels
    if not (isinstance(labels, list) and labels
            and all(isinstance(x, str) for x in labels) and set(labels) <= names):
        return False
    return canonical_json(tc.classifier_metadata or {}) == store_meta_canonical


def _validate_batch(payload: dict, expected_count: int, threshold: float,
                    expected_version: str) -> None:
    if not isinstance(payload, dict):
        raise TopicContractError("classify 响应须为 JSON 对象")
    if payload.get("classifier_version") != expected_version:
        raise TopicContractError("批次 classifier_version 与作业目标版本不一致")
    labels, scores = payload.get("labels"), payload.get("scores")
    if not (isinstance(labels, list) and isinstance(scores, list)
            and len(labels) == expected_count and len(scores) == expected_count):
        raise TopicContractError("批次条数不符")
    for lbs, scs in zip(labels, scores):
        if not (isinstance(scs, list) and len(scs) == len(label_names())
                and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        and math.isfinite(x) and 0.0 <= x <= 1.0 for x in scs)):
            raise TopicContractError("分数须为 17 维 [0,1] 有限浮点")
        if lbs != decode_scores([float(x) for x in scs], threshold):
            raise TopicContractError("标签与分数解码不一致")


def run_classification(settings, session_factory, *, client_factory=None) -> dict:
    if not _JOB_LOCK.acquire(blocking=False):
        raise TopicJobBusyError("已有归类作业在进行")
    try:
        return _run(settings, session_factory, client_factory)
    finally:
        _JOB_LOCK.release()


def _run(settings, sf, client_factory) -> dict:
    factory = client_factory or (
        lambda **kw: httpx.Client(timeout=settings.topic_inference_timeout_seconds, **kw))
    client = factory()
    base = settings.topic_inference_url.rstrip("/")
    meta = _fetch_metadata(client, base)
    threshold = float(meta["effective_threshold"])
    # 目标版本由本地按冻结口径算出,不信响应自报值(版本是三元组纯函数,spec §7.3)
    target_version = classifier_version(meta["model_version"], threshold)

    groups: dict[tuple, list[tuple[LowConfidenceQuestion, TopicClassification | None]]] = {}
    for lcq, tc in _snapshot(sf):
        groups.setdefault(question_key(lcq), []).append((lcq, tc))

    store_meta = _metadata_for_store(meta)
    store_meta_canonical = canonical_json(store_meta)
    names = set(label_names())

    todo: list[tuple[list[tuple[LowConfidenceQuestion, TopicClassification | None]], str]] = []
    skipped = 0
    for members in groups.values():
        if all(_result_is_current(tc, target_version, store_meta_canonical, names)
               for _, tc in members):
            skipped += 1  # 整组已有同版本完整一致结果 → 跳过(同版本只补漏)
            continue
        text = _representative_text([lcq for lcq, _ in members])
        if text is None:
            skipped += 1  # 全组无可用文本:跳过不写
            continue
        todo.append((members, text))

    results: list[tuple[list[tuple[LowConfidenceQuestion, TopicClassification | None]], list[str]]] = []
    batch_size = settings.topic_classify_batch_size
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        try:
            resp = client.post(f"{base}/classify",
                               json={"texts": [text for _, text in chunk]})
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TopicInferenceError(
                f"推理批次 {start // batch_size + 1} 失败:{type(exc).__name__}") from exc
        _validate_batch(payload, len(chunk), threshold, target_version)
        for (members, _), lbs in zip(chunk, payload["labels"]):
            results.append((members, lbs))

    written = 0
    if results:
        now = datetime.now()
        rows_payload = [{"question_id": lcq.id, "labels": lbs,
                         "classifier_version": target_version,
                         "classifier_metadata": store_meta,
                         "classified_at": now}
                        for members, lbs in results for lcq, _ in members]
        with sf() as s:  # 唯一写事务:一次 commit,异常整笔回滚
            stmt = mysql_insert(TopicClassification).values(rows_payload)
            stmt = stmt.on_duplicate_key_update(
                labels=stmt.inserted.labels,
                classifier_version=stmt.inserted.classifier_version,
                classifier_metadata=stmt.inserted.classifier_metadata,
                classified_at=stmt.inserted.classified_at)
            s.execute(stmt)
            s.commit()
        written = len(rows_payload)
    return {"classifier_version": target_version,
            "classified_questions": len(results),
            "skipped_questions": skipped,
            "written_rows": written}


def _metadata_for_store(meta: dict) -> dict:
    return {"model_version": meta["model_version"],
            "taxonomy_version": meta["taxonomy_version"],
            "effective_threshold": float(meta["effective_threshold"]),
            "labels": meta["labels"],
            "decoder_version": meta["decoder_version"]}
