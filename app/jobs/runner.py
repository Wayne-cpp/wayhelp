"""应用内后台作业:白名单命令 + asyncio 子进程 + 日志文件 + 报告驱动的成败判定。

单并发(同一把锁完成「检查+登记」);内存态重启即丢,日志文件持久。
成败不看 exit_code:子进程结束后读固定报告,meta.run_id 更新即 ok
(脚本因质量门槛退出 1 也属正常完成),quality_passed 取 gates.passed。
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOG_TAIL_BYTES = 8192
_CLOSE_TIMEOUT_S = 5


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobInfo:
    name: str
    status: str = "idle"            # idle|running|ok|failed
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    quality_passed: bool | None = None
    report_run_id: str | None = None
    error: str | None = None
    proc: object = field(default=None, repr=False)  # asyncio.subprocess.Process


class JobRunner:
    def __init__(self, log_dir, report_loader=None, cwd=None):
        self._log_dir = Path(log_dir)
        self._report_loader = report_loader  # callable() -> dict|None(已校验报告)
        self._cwd = str(cwd) if cwd else None
        self._cmds: dict[str, list[str]] = {}
        self._jobs: dict[str, JobInfo] = {}
        self._lock = asyncio.Lock()

    def register(self, name: str, cmd: list[str]) -> None:
        self._cmds[name] = list(cmd)

    def names(self) -> tuple[str, ...]:
        return tuple(self._cmds)

    def _current_run_id(self):
        if self._report_loader is None:
            return None
        try:
            report = self._report_loader() or {}
        except Exception:
            return None
        return (report.get("meta") or {}).get("run_id")

    async def run(self, name: str) -> bool:
        if name not in self._cmds:
            raise KeyError(name)
        async with self._lock:  # 检查+登记原子化:两个并发 POST 只放一个
            job = self._jobs.get(name)
            if job is not None and job.status == "running":
                return False
            job = JobInfo(name=name, status="running", started_at=_utcnow())
            self._jobs[name] = job
        asyncio.create_task(self._exec(job))
        return True

    async def _exec(self, job: JobInfo) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{job.name}.log"
        prev_run_id = self._current_run_id()
        try:
            with open(log_path, "wb") as log:
                job.proc = await asyncio.create_subprocess_exec(
                    *self._cmds[job.name], stdout=log,
                    stderr=asyncio.subprocess.STDOUT, cwd=self._cwd)
                job.exit_code = await job.proc.wait()
        except Exception as exc:  # 启动失败(uv 不存在等)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            with open(log_path, "ab") as log:
                log.write(("\n[runner] 启动失败: " + job.error + "\n").encode())
        else:
            report = None
            if self._report_loader is not None:
                try:
                    report = self._report_loader()
                except Exception:
                    report = None
            run_id = (report or {}).get("meta", {}).get("run_id")
            if self._report_loader is None:
                job.status = "ok" if job.exit_code == 0 else "failed"
            elif run_id and run_id != prev_run_id:
                job.status = "ok"
                gates = report.get("gates") or {}
                job.quality_passed = bool(gates.get("passed"))
                job.report_run_id = run_id
            else:
                job.status = "failed"
        finally:
            job.finished_at = _utcnow()
            job.proc = None

    def status(self, name: str) -> dict:
        if name not in self._cmds:
            raise KeyError(name)
        job = self._jobs.get(name) or JobInfo(name=name)
        tail = ""
        log_path = self._log_dir / f"{name}.log"
        if log_path.exists():
            tail = log_path.read_bytes()[-LOG_TAIL_BYTES:].decode("utf-8", errors="replace")
        return {"name": name, "status": job.status, "started_at": job.started_at,
                "finished_at": job.finished_at, "exit_code": job.exit_code,
                "quality_passed": job.quality_passed, "report_run_id": job.report_run_id,
                "error": job.error, "log_tail": tail}

    async def close(self) -> None:
        for job in self._jobs.values():
            proc = job.proc
            if job.status == "running" and proc is not None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_CLOSE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                job.status = "failed"
                job.error = "server shutdown"
                job.finished_at = _utcnow()
