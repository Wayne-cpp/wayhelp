"""ch09 评估定时:下一触发点计算;开关;异常后续调度。"""

import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app.services.eval_scheduler import eval_scheduler_loop, next_run_at
from tests.conftest import make_settings

TZ = ZoneInfo("Asia/Shanghai")
NY = ZoneInfo("America/New_York")


def test_next_run_at_same_day_and_next_day():
    now = datetime(2026, 10, 9, 2, 30, tzinfo=TZ)
    assert next_run_at(now, 3, TZ) == datetime(2026, 10, 9, 3, 0, tzinfo=TZ)
    now2 = datetime(2026, 10, 9, 3, 0, tzinfo=TZ)  # 恰在整点:严格大于 → 次日
    assert next_run_at(now2, 3, TZ) == datetime(2026, 10, 10, 3, 0, tzinfo=TZ)


def test_next_run_at_dst_repeat_day_picks_first_occurrence():
    # 2026-11-01 美东夏令时结束:本地 01:00–01:59 出现两次(fold=0 EDT → fold=1 EST)
    first = next_run_at(datetime(2026, 11, 1, 0, 30, tzinfo=NY), 1, NY)
    assert first == datetime(2026, 11, 1, 1, 0, tzinfo=NY)
    assert first.utcoffset() == timedelta(hours=-4)  # 落在第一次出现的 01:00(EDT)


def test_next_run_at_dst_repeat_runs_once_per_calendar_day():
    # 首次 01:00(EDT)已触发:即使墙钟回拨后当天还有第二次 01:00(EST),也只算下一天
    after_first_fire = datetime(2026, 11, 1, 1, 30, tzinfo=NY)
    nxt = next_run_at(after_first_fire, 1, NY)
    assert nxt == datetime(2026, 11, 2, 1, 0, tzinfo=NY)
    assert nxt.utcoffset() == timedelta(hours=-5)  # 次日已是 EST


def test_settings_schedule_defaults():
    s = make_settings()
    assert s.eval_schedule_enabled is True
    assert s.eval_schedule_hour == 3
    assert s.eval_schedule_timezone == "Asia/Shanghai"
    with pytest.raises(ValidationError):
        make_settings(eval_schedule_hour=24)  # Field(ge=0, le=23) 契约


async def test_loop_triggers_once_then_cancel():
    fired = []

    class FakeRunner:
        async def run(self, name, *, triggered_by="手动"):
            fired.append((name, triggered_by))
            return True

    sleeps = []

    async def fake_sleep(sec):  # 第二次 sleep 即取消,模拟服务关闭
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    now = datetime(2026, 10, 9, 2, 30, tzinfo=TZ)  # 固定时钟:目标 03:00 恒定
    s = make_settings(eval_schedule_enabled=True, eval_schedule_hour=3)
    task = asyncio.create_task(
        eval_scheduler_loop(FakeRunner(), s, sleep=fake_sleep,
                            now=lambda: now))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fired == [("eval-rag", "定时")]
    assert sleeps == [1800.0, 1800.0]  # 02:30 → 03:00


async def test_loop_failure_reschedules_next_day(caplog):
    calls = []

    class FlakyRunner:
        async def run(self, name, *, triggered_by="手动"):
            calls.append((name, triggered_by))
            if len(calls) == 1:
                raise RuntimeError("eval 炸了")
            return True

    state = {"now": datetime(2026, 10, 9, 2, 30, tzinfo=TZ)}
    sleeps = []

    async def fake_sleep(sec):
        sleeps.append(sec)
        if len(sleeps) >= 3:  # 第二轮跑完后取消
            raise asyncio.CancelledError
        state["now"] += timedelta(days=1)  # 睡到第二天到点

    s = make_settings(eval_schedule_hour=3)
    task = asyncio.create_task(
        eval_scheduler_loop(FlakyRunner(), s, sleep=fake_sleep,
                            now=lambda: state["now"]))
    with caplog.at_level(logging.ERROR, logger="app.services.eval_scheduler"):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == 2  # 第一轮炸了,第二天照常再跑(异常不传播)
    assert "eval scheduler run failed" in caplog.text
    assert sleeps[:2] == [1800.0, 1800.0]


async def test_loop_sleeps_real_instant_delta_across_dst():
    # 2026-10-31 23:30 EDT → 目标 11-01 03:00 EST:真实间隔 4.5h;墙钟差只有
    # 3.5h,按墙钟休眠会在 DST 回拨夜提前一小时触发(spec §5.5 timezone-aware)
    state = {"now": datetime(2026, 10, 31, 23, 30, tzinfo=NY)}
    sleeps = []

    async def fake_sleep(sec):
        sleeps.append(sec)
        raise asyncio.CancelledError

    s = make_settings(eval_schedule_timezone="America/New_York",
                      eval_schedule_hour=3)
    task = asyncio.create_task(
        eval_scheduler_loop(object(), s, sleep=fake_sleep,
                            now=lambda: state["now"]))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sleeps == [4.5 * 3600]
