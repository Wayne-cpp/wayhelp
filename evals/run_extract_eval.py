"""售后提取评估:真实模型跑 evals/extract_cases.jsonl。

用法: uv run python evals/run_extract_eval.py   (需要项目根 .env)
达标线(spec §10):三字段宏平均 >= 0.80 且任一字段 >= 0.70,否则退出码 1。
"""
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from app.chains.extract_chain import EXTRACT_INSTRUCTION, _FEW_SHOT, run_extraction
from app.config import Settings
from app.schemas import Expectation, Intent

CASES = Path(__file__).parent / "extract_cases.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
FIELDS = ["order_id", "intent", "expectation"]
MIN_MACRO = 0.80
MIN_PER_FIELD = 0.70


def load_cases():
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case id 重复"
    intents = {}
    expectations = {}
    with_oid = sum(1 for c in cases if c["order_id"] is not None)
    for c in cases:
        intents[c["intent"]] = intents.get(c["intent"], 0) + 1
        key = "null" if c["expectation"] is None else c["expectation"]
        expectations[key] = expectations.get(key, 0) + 1
    problems = []
    if len(cases) < 21:
        problems.append(f"样例数 {len(cases)} < 21")
    for intent in Intent:
        n = intents.get(intent.value, 0)
        if n < 3:
            problems.append(f"intent {intent.value} 仅 {n} 条 < 3")
    for exp in Expectation:
        n = expectations.get(exp.value, 0)
        if n < 2:
            problems.append(f"expectation {exp.value} 仅 {n} 次 < 2")
    if expectations.get("null", 0) < 2:
        problems.append(f"expectation null 仅 {expectations.get('null', 0)} 次 < 2")
    if with_oid < 7 or len(cases) - with_oid < 7:
        problems.append("有/无订单号样例不足 7 条")
    return cases, problems


async def main() -> int:
    cases, problems = load_cases()
    if problems:
        print("评估集覆盖不满足要求:")
        for p in problems:
            print(" -", p)
        return 1

    settings = Settings()
    model_kwargs = dict(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        max_tokens=settings.max_output_tokens,
    )
    temperature_note = "temperature=0"
    model = ChatOpenAI(temperature=0, **model_kwargs)

    prompt_hash = hashlib.sha256(
        (EXTRACT_INSTRUCTION + json.dumps(_FEW_SHOT, ensure_ascii=False)).encode()
    ).hexdigest()[:12]

    per_field_hits = {f: 0 for f in FIELDS}
    exact_hits = 0
    oid_hits = {"with": [0, 0], "without": [0, 0]}
    records = []
    for case in cases:
        try:
            got = await run_extraction(
                model, settings.structured_output_method, case["text"], settings.max_input_tokens
            )
            got_fields = {f: getattr(got, f) for f in FIELDS}
            got_fields = {f: (v.value if hasattr(v, "value") else v) for f, v in got_fields.items()}
            error = None
        except Exception as exc:  # 异常/解析失败 = 三字段全错
            got_fields = {f: None for f in FIELDS}
            error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            hits = {f: False for f in FIELDS}
        else:
            hits = {f: got_fields[f] == case[f] for f in FIELDS}
        for f in FIELDS:
            per_field_hits[f] += int(hits[f])
        exact_hits += int(all(hits.values()))
        group = "with" if case["order_id"] is not None else "without"
        oid_hits[group][0] += int(hits["order_id"])
        oid_hits[group][1] += 1
        summary_note = "" if error is None else f" [ERROR] {error}"
        print(f"{case['id']}: {'OK ' if all(hits.values()) else 'FAIL'} "
              f"expect=({case['order_id']},{case['intent']},{case['expectation']}) "
              f"got=({got_fields['order_id']},{got_fields['intent']},{got_fields['expectation']}){summary_note}")
        records.append({"id": case["id"], "text": case["text"], "expected": {f: case[f] for f in FIELDS},
                        "got": got_fields, "hits": hits, "error": error,
                        "summary_text": None if error else got.summary})

    n = len(cases)
    per_field = {f: per_field_hits[f] / n for f in FIELDS}
    macro = sum(per_field.values()) / 3
    exact = exact_hits / n
    print(f"\n字段准确率: {per_field}")
    print(f"三字段宏平均: {macro:.3f} (达标线 {MIN_MACRO})")
    print(f"整条 exact-match: {exact:.3f} (诊断指标)")
    for g, (h, total) in oid_hits.items():
        print(f"order_id 准确率({g}): {h}/{total} = {h / total:.3f}")

    passed = macro >= MIN_MACRO and all(v >= MIN_PER_FIELD for v in per_field.values())
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps({
        "endpoint": settings.openai_base_url, "model": settings.model_name,
        "structured_output_method": settings.structured_output_method,
        "generation": {"temperature": temperature_note, "max_tokens": settings.max_output_tokens},
        "prompt_sha256_12": prompt_hash,
        "per_field_accuracy": per_field, "macro_avg": macro, "exact_match": exact,
        "passed": passed, "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out}")
    print("summary 字段请人工核对 evals/results 中各记录,并登记 evals/manual_review.md")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
