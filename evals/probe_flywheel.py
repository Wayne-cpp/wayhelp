# evals/probe_flywheel.py —— 真模型跑飞轮标注样例,打印标准化/查重判定表;退出码=失败数
# 样例行:{"raw","resolved","expect_normalized_contains":[锚点...],
#          "candidates":[{"id","question"}],"expect_match_id":<id|null>}
# 两跳断言(与 app/services/flywheel.py 同一代码路径):
#   跳一标准化:resolved+raw → STANDARDIZE_PROMPT,normalized_question 命中任一锚点;
#   跳二查重(候选非空才跑):DEDUP_PROMPT,matched_id 与标注一致(null = 判新建)。
# 用法: uv run python evals/probe_flywheel.py(实跑留 Task 16 验收,不进 pytest)
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.services.flywheel import _dedup, _standardize


async def main() -> int:
    s = Settings()
    model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                       base_url=s.openai_base_url, max_tokens=200)
    rows = [json.loads(l) for l in Path(__file__).with_name("flywheel_samples.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = 0
    for r in rows:
        lcq = SimpleNamespace(raw_question=r["raw"],
                              resolved_question=r.get("resolved"))
        try:
            normalized, _answer = await _standardize(model, lcq)
        except Exception as exc:
            normalized, std_ok, fail_note = None, False, f"标准化异常 {type(exc).__name__}"
        else:
            anchors = r.get("expect_normalized_contains") or []
            std_ok = not anchors or any(a in normalized for a in anchors)
            fail_note = "" if std_ok else "标准化锚点未命中"
        match_note = ""
        if not r.get("candidates"):
            got_match, dedup_ok = None, True   # 候选空 = 生产路径直接新建,不调模型
        else:
            cands = [SimpleNamespace(id=c["id"], normalized_question=c["question"])
                     for c in r["candidates"]]
            try:
                got_match = await _dedup(model, normalized, cands)
            except Exception as exc:
                got_match, dedup_ok = False, False
                match_note = f" 查重异常 {type(exc).__name__}"
            else:
                dedup_ok = got_match == r.get("expect_match_id")
                match_note = "" if dedup_ok else \
                    f" 期望 match={r.get('expect_match_id')!r} 实得 {got_match!r}"
        ok = normalized is not None and std_ok and dedup_ok
        fails += 0 if ok else 1
        print(f"{'OK ' if ok else 'BAD'} {r['raw']!r} → {normalized!r}: "
              f"match={got_match!r}{(' ' + fail_note) if fail_note else ''}{match_note}")
    print(f"\n{len(rows) - fails}/{len(rows)} 命中")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
