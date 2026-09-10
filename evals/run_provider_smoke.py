"""端点能力 smoke test:经 HTTP 验证流式增量与结构化输出。

用法: uv run python evals/run_provider_smoke.py   (需要项目根 .env)
通过标准:聊天 >= 2 个 delta 且收到 [DONE];/v1/extract 返回四键齐全。失败退出码 1。
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = {"order_id", "intent", "expectation", "summary"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(base: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/docs", timeout=1.0).status_code == 200:
                return True
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    return False


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
    )
    failures = []
    try:
        if not wait_ready(base, time.monotonic() + 20):
            print("FAIL: 应用未在 20s 内就绪(检查 .env 与依赖)")
            return 1
        if proc.poll() is not None:
            print(f"FAIL: 应用子进程已退出(exit={proc.returncode}),响应并非来自 smoke 自建服务")
            return 1

        deltas = []
        t_first = t_last = t_done = None
        saw_done = False
        start = time.monotonic()
        try:
            with httpx.stream(
                "POST", f"{base}/v1/chat/stream",
                json={"message": "请用 150 字以上详细介绍你们店的退货退款流程。"},
                timeout=60,
            ) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        saw_done = True
                        t_done = time.monotonic() - start
                        break
                    frame = json.loads(payload)
                    if frame.get("type") == "delta":
                        deltas.append(frame["content"])
                        now = time.monotonic() - start
                        t_first = t_first if t_first is not None else now
                        t_last = now
                    elif frame.get("type") == "error":
                        failures.append(f"聊天返回错误帧: {frame}")
                        break
        except httpx.TransportError as exc:
            failures.append(f"网络故障(非能力问题,需排查连接): {type(exc).__name__}")

        print(f"delta 数: {len(deltas)},首 delta {t_first}s,末 delta {t_last}s,DONE {t_done}s")
        if len(deltas) < 2:
            failures.append(f"仅 {len(deltas)} 个增量,不足以证明流式能力")
        if not saw_done:
            failures.append("未收到 [DONE]")

        try:
            r = httpx.post(f"{base}/v1/extract", json={"text": "订单 20260101001 收到就是坏的,我要退货退款"}, timeout=60)
            body = r.json()
            if r.status_code != 200 or not REQUIRED_KEYS.issubset(body.keys()):
                failures.append(f"结构化输出能力不支持: status={r.status_code} body={body}")
            else:
                print(f"结构化输出 OK: {body}")
        except httpx.TransportError as exc:
            failures.append(f"网络故障(非能力问题,需排查连接): {type(exc).__name__}")
        except json.JSONDecodeError as exc:
            failures.append(f"结构化输出能力不支持: 响应非 JSON: {type(exc).__name__}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    if failures:
        print("\nSMOKE FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nSMOKE PASS: 当前 endpoint/model/config 具备流式与结构化输出能力")
    return 0


if __name__ == "__main__":
    sys.exit(main())
