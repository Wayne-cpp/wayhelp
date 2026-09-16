"""JobRunner:状态机 / 日志捕获 / 单并发 / 报告驱动的成败判定 / close 回收。"""

import asyncio
import json
import sys

import pytest

from app.jobs.runner import JobRunner


def _report(run_id, passed=True):
    return {"meta": {"run_id": run_id}, "gates": {"passed": passed}}


async def test_run_ok_and_log(tmp_path):
    r = JobRunner(tmp_path)                 # 无 loader:exit 0 → ok
    r.register("t", [sys.executable, "-c", "print('hello-eval')"])
    assert await r.run("t") is True
    for _ in range(100):
        await asyncio.sleep(0.02)
        if r.status("t")["status"] != "running":
            break
    st = r.status("t")
    assert st["status"] == "ok" and st["exit_code"] == 0
    assert "hello-eval" in st["log_tail"]
    assert st["started_at"] and st["finished_at"]
    with pytest.raises(KeyError):
        r.status("nope")


async def test_conflict_while_running(tmp_path):
    r = JobRunner(tmp_path)
    r.register("t", [sys.executable, "-c", "import time; time.sleep(1.0)"])
    assert await r.run("t") is True
    assert await r.run("t") is False          # 运行中重复触发
    for _ in range(200):
        await asyncio.sleep(0.02)
        if r.status("t")["status"] != "running":
            break
    assert r.status("t")["status"] == "ok"    # 无 loader:exit 0 → ok


async def test_report_drives_verdict(tmp_path):
    """exit 1 但产出新 run_id 报告 → ok + quality_passed=false;无新报告 → failed。"""
    rp = tmp_path / "rag_eval.json"
    rp.write_text(json.dumps(_report("old")), encoding="utf-8")
    loader = lambda: json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    r = JobRunner(tmp_path, report_loader=loader)
    write_and_fail = (
        "import json,sys,pathlib;"
        f"pathlib.Path({str(rp)!r}).write_text(json.dumps({_report('new', False)!r}));"
        "sys.exit(1)")
    r.register("t", [sys.executable, "-c", write_and_fail])
    await r.run("t")
    for _ in range(200):
        await asyncio.sleep(0.02)
        if r.status("t")["status"] != "running":
            break
    st = r.status("t")
    assert st["status"] == "ok" and st["quality_passed"] is False
    assert st["exit_code"] == 1 and st["report_run_id"] == "new"

    r2 = JobRunner(tmp_path, report_loader=loader)  # 不更新报告 → run_id 不变 → failed
    r2.register("t", [sys.executable, "-c", "import sys; sys.exit(1)"])
    await r2.run("t")
    for _ in range(200):
        await asyncio.sleep(0.02)
        if r2.status("t")["status"] != "running":
            break
    st2 = r2.status("t")
    assert st2["status"] == "failed" and st2["quality_passed"] is None


async def test_close_terminates(tmp_path):
    r = JobRunner(tmp_path)
    r.register("t", [sys.executable, "-c", "import time; time.sleep(60)"])
    await r.run("t")
    await asyncio.sleep(0.2)
    await r.close()
    assert r.status("t")["status"] == "failed"
