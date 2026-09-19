# evals/probe_intent.py —— 真模型跑标注样例,打印判定表;退出码=失败数
# 单轮行:{"query","intent"} 两字段,直接分类。
# 多轮行:额外带 "history": [[用户,助手],...] 与可选 "resolved_must_contain": [候选锚点...]:
#   先把 history 渲染进 UNDERSTAND_PROMPT 做指代消解(跳一),再把 resolved 喂 INTENT_PROMPT
#   分类(跳二);断言两跳——消解结果命中任一锚点、分类 intent 与标注一致。
# 用法: uv run python evals/probe_intent.py
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.graph.nodes import parse_intent_output, parse_understand_output
from app.prompts.intent import INTENT_PROMPT
from app.prompts.understand import UNDERSTAND_PROMPT


def _history_block(history: list[list[str]]) -> str:
    """[[用户,助手],...] → UNDERSTAND_PROMPT 的 {history_block}(与 understand_query 节点同格式)。"""
    lines = []
    for turn in history:
        if len(turn) >= 1:
            lines.append(f"用户:{turn[0]}")
        if len(turn) >= 2:
            lines.append(f"助手:{turn[1]}")
    return "对话历史:\n" + "\n".join(lines)


async def main() -> int:
    s = Settings()
    model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                       base_url=s.openai_base_url, max_tokens=200)
    rows = [json.loads(l) for l in Path(__file__).with_name("intent_samples.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = 0
    for r in rows:
        raw = r["query"]
        resolved = raw
        hop1_ok, hop1_note = True, ""
        if r.get("history"):
            prompt = (UNDERSTAND_PROMPT
                      .replace("{history_block}", _history_block(r["history"]))
                      .replace("{query}", raw))
            resp = await model.ainvoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else ""
            parsed = parse_understand_output(text)
            resolved = (parsed or "").strip() or raw
            anchors = r.get("resolved_must_contain") or []
            hop1_ok = parsed is not None and (not anchors
                                              or any(a in resolved for a in anchors))
            hop1_note = f" 消解={'OK' if hop1_ok else 'BAD'}"
        resp = await model.ainvoke([HumanMessage(
            content=INTENT_PROMPT.replace("{query}", resolved))])
        got = parse_intent_output(resp.content if isinstance(resp.content, str) else "")
        got_intent = got[0] if got else None
        ok = got_intent == r["intent"] and hop1_ok
        fails += 0 if ok else 1
        arrow = f"{raw!r} → {resolved!r}" if r.get("history") else f"{raw!r}"
        print(f"{'OK ' if ok else 'BAD'} {arrow}: 期望 {r['intent']!r} 实得 {got}{hop1_note}")
    print(f"\n{len(rows) - fails}/{len(rows)} 命中")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
