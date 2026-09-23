"""20+ 轮上下文长跑(烧额度,手跑):
  默认窗口: uv run python evals/probe_context.py --scenario default
  演示配置: 先用演示 env 起服,再 uv run python evals/probe_context.py --scenario demo
起服(演示): MODEL_CONTEXT_WINDOW=13000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
  MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
  no_proxy=127.0.0.1,localhost uv run uvicorn --factory app.main:create_app
"""
import argparse
import re
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
USER = "00000000-0000-0000-0000-000000000001"   # 与 chat.html 演示账号一致

SCRIPT = (["我的订单A1001说可以申请退款,帮我看看进度", "退款要多久到账", "好的先这样"]
          + [f"顺便问下保温杯的第{i}个问题:保温多久" for i in range(1, 19)]
          + ["最开始那个订单后来怎么说"])


def chat(client, sid, text):
    with client.stream("POST", f"{BASE}/v1/chat/stream",
                       json={"user_id": USER, "session_id": sid, "message": text},
                       timeout=120) as r:
        sid_out, answer = sid, []
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            import json
            ev = json.loads(payload)
            if ev.get("type") == "session":
                sid_out = ev["session_id"]
            elif ev.get("type") == "delta":
                answer.append(ev["content"])
        return sid_out, "".join(answer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["default", "demo"], required=True)
    ap.add_argument("--log", default="log/app.log")
    args = ap.parse_args()
    log_start = Path(args.log).stat().st_size if Path(args.log).exists() else 0
    sid, t0 = None, time.monotonic()
    with httpx.Client(trust_env=False) as client:   # 代理红线:本机不走代理
        for i, q in enumerate(SCRIPT, 1):
            sid, ans = chat(client, sid, q)
            print(f"[{i:02d}] {q[:24]}… -> {ans[:60]}")
    elapsed = time.monotonic() - t0
    tail = Path(args.log).read_text(encoding="utf-8")[log_start:]
    degraded = "层1 降级" in tail
    triggered = "summary trigger" in tail
    done = "summary done" in tail
    print(f"elapsed={elapsed:.1f}s degraded={degraded} trigger={triggered} done={done}")
    if args.scenario == "default":
        assert not degraded and not triggered and not done, "默认窗口 20 轮不该触发降级/摘要"
    else:
        assert degraded and triggered and done, "演示配置应出现完整级联"
    print("末轮回答(人工核对是否答对早期订单):", ans)
    return 0


if __name__ == "__main__":
    sys.exit(main())
