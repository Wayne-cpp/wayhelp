"""/api/jobs 路由:白名单作业名、409 并发、报告驱动 ok/failed。"""

import asyncio
import json
import sys

import httpx
import pytest

from app.jobs.runner import JobRunner
from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings


def _report(run_id, passed=True):
    return {"meta": {"run_id": run_id}, "gates": {"passed": passed}}


def _make_app(tmp_path, cmd, prev_report=None):
    """装测试 runner:loader 读 tmp 目录报告文件;cmd 模拟评估(自己写新报告)。"""
    rp = tmp_path / "rag_eval.json"
    if prev_report is not None:
        rp.write_text(json.dumps(prev_report), encoding="utf-8")
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))

    def _loader():
        if not rp.exists():
            return None
        return json.loads(rp.read_text(encoding="utf-8"))

    runner = JobRunner(tmp_path, report_loader=_loader)
    runner.register("eval-rag", cmd)
    app.state.job_runner = runner            # 替换默认 runner,绝不真跑评估
    app.state.rag_eval_report_path = rp
    return app


async def _wait_terminal(app, name, timeout=5.0):
    runner = app.state.job_runner
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        if runner.status(name)["status"] != "running":
            break
    return runner.status(name)


def _write_report_cmd(tmp_path, report, exit_code=0):
    """模拟评估脚本的命令:写新报告 + 打日志 + 按 exit_code 退出。"""
    rp = tmp_path / "rag_eval.json"
    return [sys.executable, "-c",
            "import json,sys,pathlib;"
            f"pathlib.Path({str(rp)!r}).write_text(json.dumps({report!r}));"
            f"print('done');sys.exit({exit_code})"]


async def test_run_and_status(tmp_path):
    app = _make_app(tmp_path, _write_report_cmd(tmp_path, _report("r1")))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/api/jobs/eval-rag")
        assert resp.status_code == 200 and resp.json()["status"] == "idle"
        assert resp.json()["exit_code"] is None
        resp = await client.post("/api/jobs/eval-rag/run")
        assert resp.status_code == 200 and resp.json() == {"started": True}
        st = await _wait_terminal(app, "eval-rag")
        assert st["status"] == "ok" and "done" in st["log_tail"]
        resp = await client.get("/api/jobs/eval-rag")
        body = resp.json()
        assert body["status"] == "ok" and body["exit_code"] == 0
        assert body["quality_passed"] is True and body["report_run_id"] == "r1"
        assert "done" in body["log_tail"]


async def test_unknown_job_404(tmp_path):
    app = _make_app(tmp_path, [sys.executable, "-c", "pass"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.get("/api/jobs/nope")).status_code == 404
        assert (await client.post("/api/jobs/nope/run")).status_code == 404


async def test_conflict_409(tmp_path):
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    runner = JobRunner(tmp_path)
    runner.register("eval-rag",
                    [sys.executable, "-c", "import time; time.sleep(1.0)"])
    app.state.job_runner = runner
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.post("/api/jobs/eval-rag/run")).status_code == 200
        resp = await client.post("/api/jobs/eval-rag/run")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "job_running"
    await _wait_terminal(app, "eval-rag")


async def test_quality_failed_but_ok(tmp_path):
    """新报告 + gates.passed=false + exit 1:status=ok,quality_passed=false(报告驱动)。"""
    cmd = _write_report_cmd(tmp_path, _report("r-new", passed=False), exit_code=1)
    app = _make_app(tmp_path, cmd, prev_report=_report("r-old"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/api/jobs/eval-rag/run")
        st = await _wait_terminal(app, "eval-rag")
    assert st["status"] == "ok" and st["quality_passed"] is False
    assert st["exit_code"] == 1 and st["report_run_id"] == "r-new"


async def test_no_new_report_is_failed(tmp_path):
    """脚本没产出新报告(exit 1)→ failed,quality_passed=null,旧报告不动。"""
    app = _make_app(tmp_path, [sys.executable, "-c", "import sys; sys.exit(1)"],
                    prev_report=_report("r-old"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/api/jobs/eval-rag/run")
        st = await _wait_terminal(app, "eval-rag")
    assert st["status"] == "failed" and st["quality_passed"] is None
    assert st["report_run_id"] is None
