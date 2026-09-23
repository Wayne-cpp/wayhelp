"""摘要 prompt 标注样例验证(烧额度,手跑):uv run python evals/probe_summary.py"""
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Settings                      # noqa: E402
from app.prompts.summary import SUMMARY_PROMPT        # noqa: E402


def main() -> int:
    from langchain_openai import ChatOpenAI
    s = Settings()
    model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                       base_url=s.openai_base_url, max_tokens=400)
    cases = [json.loads(x) for x in
             Path(__file__).with_name("summary_cases.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    failures = 0
    for c in cases:
        prompt = (SUMMARY_PROMPT.replace("{background}", "(无)")
                  .replace("{transcript}", "\n".join(c["dialogue"]))
                  .replace("{max_chars}", str(s.summary_max_chars)))
        out = model.invoke([HumanMessage(content=prompt)]).content.strip()
        ok = (all(k in out for k in c["must_include"])
              and all(k not in out for k in c["must_not_include"])
              and len(out) <= c["max_len"])
        print(f"[{'PASS' if ok else 'FAIL'}] {c['name']}: {out}")
        failures += 0 if ok else 1
    print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
