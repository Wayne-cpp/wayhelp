"""ch09 飞轮流水线:lcq pending → 标准化 → 查重 → review_queue。

lifespan 单 worker;Event 唤醒 + DB 退避到期自醒;每条 lcq 独立事务;
写回事务行锁复核状态(spec §5.4)。同步 DB 经 asyncio.to_thread,模型异步 await。
"""

import asyncio
import contextlib
import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage
from sqlalchemy import func, or_, text

from app.config import Settings
from app.models import LowConfidenceQuestion, ReviewQueue
from app.prompts.flywheel import DEDUP_PROMPT, STANDARDIZE_PROMPT

logger = logging.getLogger(__name__)


class RetryDedupError(Exception):
    """查重命中目标在写回时已非待审:外层重读候选重判一次。"""


def parse_standardize_output(text: str) -> tuple[str, str] | None:
    """提取 {"normalized_question","suggested_answer"};两键均非空 str 否则 None。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q, a = data.get("normalized_question"), data.get("suggested_answer")
    if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
        return q, a
    return None


def parse_dedup_output(text: str, candidate_ids: set[int]) -> int | None | bool:
    """命中且 id 在候选集 → id;{"matched_id": null} → None 新建;其余 → False 视为失败。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    mid = data.get("matched_id")
    if mid is None:
        return None
    if isinstance(mid, bool) or not isinstance(mid, int):
        return False
    return mid if mid in candidate_ids else False


def backoff_seconds(settings: Settings, attempt_count: int) -> float:
    return min(settings.flywheel_retry_max_seconds,
               settings.flywheel_retry_base_seconds * 2 ** (attempt_count - 1))


# ── 同步 DB 操作(一律经 asyncio.to_thread 调用)──


def _fetch_due(sf, limit: int) -> list[LowConfidenceQuestion]:
    with sf() as s:
        return list(s.query(LowConfidenceQuestion).filter(
            LowConfidenceQuestion.process_status == "pending",
            LowConfidenceQuestion.matched_review_id.is_(None),
            or_(LowConfidenceQuestion.next_attempt_at.is_(None),
                LowConfidenceQuestion.next_attempt_at <= func.now()),
        ).order_by(LowConfidenceQuestion.id).limit(limit).all())


def _fetch_candidates(sf, limit: int) -> list[ReviewQueue]:
    with sf() as s:
        return list(s.query(ReviewQueue)
                    .filter_by(review_status="待审")
                    .order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
                    .limit(limit).all())


def _commit_match(sf, lcq_id: int, normalized: str, suggested: str,
                  match_id: int | None) -> str:
    """写回:锁 lcq 复核 pending/未归并;命中则锁目标复核仍待审。
    返回 "merged" | "created" | "skipped"(状态已变,重放安全)。"""
    with sf() as s:
        row = s.query(LowConfidenceQuestion).filter_by(id=lcq_id).with_for_update().first()
        if row is None or row.process_status != "pending" \
                or row.matched_review_id is not None:
            return "skipped"
        if match_id is not None:
            target = s.query(ReviewQueue).filter_by(id=match_id).with_for_update().first()
            if target is None or target.review_status != "待审":
                raise RetryDedupError(f"target review {match_id} state changed")
            target.occurrence_count += 1
            rid = target.id
        else:
            target = ReviewQueue(normalized_question=normalized,
                                 ai_suggested_answer=suggested)
            s.add(target)
            s.flush()
            rid = target.id
        row.matched_review_id = rid
        row.process_status = "processed"
        row.next_attempt_at = None
        row.last_error = None
        s.commit()
        return "merged" if match_id is not None else "created"


def _mark_failure(sf, lcq_id: int, exc: Exception, settings: Settings) -> str:
    """失败记账:attempt_count+1,退避用 DB 钟写 next_attempt_at;到限转 failed。
    返回 "retried" | "failed" | "skipped"(行已不在处理面)。"""
    with sf() as s:
        row = s.query(LowConfidenceQuestion).filter_by(id=lcq_id).with_for_update().first()
        if row is None or row.process_status != "pending" \
                or row.matched_review_id is not None:
            return "skipped"
        row.attempt_count += 1
        row.last_error = f"{type(exc).__name__}: {exc}"[:500]
        if row.attempt_count >= settings.flywheel_max_attempts:
            row.process_status = "failed"
            row.next_attempt_at = None
        else:
            secs = int(backoff_seconds(settings, row.attempt_count))
            row.next_attempt_at = s.query(
                func.date_add(func.now(), text(f"INTERVAL {secs} SECOND"))).scalar()
        s.commit()
        return "failed" if row.process_status == "failed" else "retried"


# ── 异步流水线 ──


async def _standardize(model, row) -> tuple[str, str]:
    # 惯例照 nodes.py:replace 占位,prompt 内字面 JSON 花括号不被 format 误伤
    prompt = (STANDARDIZE_PROMPT
              .replace("{resolved_question}",
                       row.resolved_question or row.raw_question)
              .replace("{raw_question}", row.raw_question))
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    text_out = resp.content if isinstance(resp.content, str) else ""
    parsed = parse_standardize_output(text_out)
    if parsed is None:
        raise ValueError(f"standardize output unparsable: {text_out[:120]!r}")
    return parsed


async def _dedup(model, normalized: str, candidates) -> int | None:
    if not candidates:   # 队列空 = 必然新建,省一次模型调用
        return None
    lines = "\n".join(f"{c.id}: {c.normalized_question}" for c in candidates)
    prompt = (DEDUP_PROMPT
              .replace("{normalized_question}", normalized)
              .replace("{candidates}", lines))
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    result = parse_dedup_output(
        resp.content if isinstance(resp.content, str) else "",
        {c.id for c in candidates})
    if result is False:
        raise ValueError(f"dedup output invalid: "
                         f"{(resp.content if isinstance(resp.content, str) else '')[:120]!r}")
    return result


def _bump(stats: dict, outcome: str) -> None:
    if outcome in ("merged", "created"):
        stats["processed"] += 1
        stats[outcome] += 1


async def _fail(sf, settings: Settings, lcq_id: int, exc: Exception, stats: dict) -> None:
    outcome = await asyncio.to_thread(_mark_failure, sf, lcq_id, exc, settings)
    if outcome in ("retried", "failed"):
        stats[outcome] += 1


async def process_pending(settings: Settings, session_factory, model) -> dict:
    """排空所有到期 pending 行(spec §5.4);只能由飞轮 worker 调用。"""
    stats = {"processed": 0, "merged": 0, "created": 0, "retried": 0, "failed": 0}
    while True:
        rows = await asyncio.to_thread(_fetch_due, session_factory,
                                       settings.flywheel_batch_size)
        if not rows:
            return stats
        for row in rows:
            try:
                norm = await _standardize(model, row)
                candidates = await asyncio.to_thread(
                    _fetch_candidates, session_factory,
                    settings.review_queue_match_limit)
                match_id = await _dedup(model, norm[0], candidates)
                outcome = await asyncio.to_thread(
                    _commit_match, session_factory, row.id, norm[0], norm[1], match_id)
                _bump(stats, outcome)
            except RetryDedupError:
                try:   # 候选状态变了:重读候选重判一次,仍失败走退避
                    candidates = await asyncio.to_thread(
                        _fetch_candidates, session_factory,
                        settings.review_queue_match_limit)
                    match_id = await _dedup(model, norm[0], candidates)
                    outcome = await asyncio.to_thread(
                        _commit_match, session_factory, row.id, norm[0], norm[1],
                        match_id)
                    _bump(stats, outcome)
                except Exception as exc:
                    await _fail(session_factory, settings, row.id, exc, stats)
            except Exception as exc:
                await _fail(session_factory, settings, row.id, exc, stats)


class FlywheelWorker:
    """单 worker 循环:先清 Event 再扫描;无到期行按最早 next_attempt_at 定时醒,
    无未来行只等 Event;循环异常记日志按有上限退避恢复,不退出不忙转。"""

    def __init__(self, settings: Settings, session_factory, model):
        self._settings = settings
        self._sf = session_factory
        self._model = model
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def notify(self) -> None:
        self._event.set()

    def start(self) -> None:
        if self._task is None and self._sf is not None:
            self._task = asyncio.create_task(self._loop())

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def run_now(self) -> None:
        self.notify()   # 「立即处理」入口:只加速唤醒,不绕过退避/另建 task

    async def _loop(self) -> None:
        while True:
            self._event.clear()          # 先清再扫,等待建立期间的通知不丢
            try:
                stats = await process_pending(self._settings, self._sf, self._model)
                if any(stats.values()):
                    logger.info("flywheel pass %s", stats)
                wait = await asyncio.to_thread(self._next_due_in)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("flywheel loop error")
                wait = min(self._settings.flywheel_retry_max_seconds, 60.0)
            try:
                if wait is None:
                    await self._event.wait()
                else:
                    await asyncio.wait_for(self._event.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass

    def _next_due_in(self) -> float | None:
        with self._sf() as s:
            nxt = (s.query(func.min(LowConfidenceQuestion.next_attempt_at))
                   .filter(LowConfidenceQuestion.process_status == "pending",
                           LowConfidenceQuestion.next_attempt_at.isnot(None))
                   .scalar())
        if nxt is None:
            return None
        return max((nxt - datetime.now()).total_seconds(), 0.5)
