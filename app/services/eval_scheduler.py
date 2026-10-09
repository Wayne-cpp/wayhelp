"""ch09:评估定时调度(lifespan 单 task,不引 APScheduler)。"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def next_run_at(now: datetime, hour: int, tz: ZoneInfo) -> datetime:
    """严格大于 now 的下一个本地 hour:00(DST 重复同日只一次)。"""
    local = now.astimezone(tz)
    candidate = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate = (local + timedelta(days=1)).replace(
            hour=hour, minute=0, second=0, microsecond=0)
    return candidate


async def eval_scheduler_loop(job_runner, settings, *,
                              sleep=asyncio.sleep, now=None) -> None:
    """每天本地 hour:00 触发 eval-rag(spec §5.5);可取消;异常记日志算下一天。"""
    tz = ZoneInfo(settings.eval_schedule_timezone)
    clock = now or (lambda: datetime.now(tz))
    while True:
        target = next_run_at(clock(), settings.eval_schedule_hour, tz)
        # 休眠按 UTC 瞬时差算:DST 切换夜同 tzinfo 的墙钟差 ≠ 真实间隔,
        # 按墙钟睡会在回拨夜提前一小时触发
        delta = (target.astimezone(timezone.utc)
                 - clock().astimezone(timezone.utc)).total_seconds()
        await sleep(max(delta, 0.1))
        try:
            await job_runner.run("eval-rag", triggered_by="定时")
        except Exception:
            logger.exception("eval scheduler run failed")  # 不传播,算下一天
