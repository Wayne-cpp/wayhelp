# evals/probe_intent.py —— 真模型跑 8 条标注样例,打印判定表;退出码=失败数
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.graph.nodes import parse_intent_output
from app.prompts.intent import INTENT_PROMPT


async def main() -> int:
    s = Settings()
    model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                       base_url=s.openai_base_url, max_tokens=200)
    rows = [json.loads(l) for l in Path(__file__).with_name("intent_samples.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = 0
    for r in rows:
        resp = await model.ainvoke([HumanMessage(content=INTENT_PROMPT.replace("{query}", r["query"]))])
        got = parse_intent_output(resp.content if isinstance(resp.content, str) else "")
        ok = got == (r["intent"], r["needs_knowledge"])
        fails += 0 if ok else 1
        print(f"{'OK ' if ok else 'BAD'} {r['query']!r}: 期望 {(r['intent'], r['needs_knowledge'])} 实得 {got}")
    print(f"\n{len(rows) - fails}/{len(rows)} 命中")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
