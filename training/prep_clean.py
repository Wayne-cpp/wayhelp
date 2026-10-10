"""步骤1 清洗:确定性 clean_text(全量)+ LLM 错别字校对(ID 对齐校验)。

LLM 校对仅离线数据准备用;错位/重复/遗漏 → 整批保留原文并记日志(spec §4.1)。
CLI: uv run --group train python -m training.prep_clean <in.jsonl> <out.jsonl> [--real]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from training.preprocessing import clean_text
from training.prep_records import (make_chat_model, read_jsonl, write_jsonl)

log = logging.getLogger(__name__)


def deterministic_clean(records) -> list[dict]:
    for r in records:
        r["text"] = clean_text(r["text_raw"])
    return records


def llm_polish(records, llm, batch_size: int = 20):
    """llm 须实现 polish_batch(list[{sample_id,text}]) -> list[{sample_id,text}]。
    返回 (records, failed_batches)。ID 集合不一致或条数不符 → 整批保留原文。"""
    failed = 0
    out = list(records)
    for start in range(0, len(out), batch_size):
        batch = out[start:start + batch_size]
        expect_ids = [r["sample_id"] for r in batch]
        try:
            fixed = llm.polish_batch(
                [{"sample_id": r["sample_id"], "text": r["text"]} for r in batch])
        except Exception as exc:
            log.warning("polish 批 %d 异常保留原文:%s", start // batch_size, exc)
            failed += 1
            continue
        got_ids = [f.get("sample_id") for f in fixed] if isinstance(fixed, list) else []
        if sorted(got_ids) != sorted(expect_ids) or len(set(got_ids)) != len(got_ids):
            log.warning("polish 批 %d ID 不对齐,整批保留原文", start // batch_size)
            failed += 1
            continue
        by_id = {f["sample_id"]: f["text"] for f in fixed}
        for r in batch:
            r["text"] = by_id[r["sample_id"]]
    return out, failed


class _ChatPolisher:
    """把 ChatOpenAI 适配成 llm_polish 需要的 polish_batch 接口(仅 CLI 用)。"""

    def __init__(self, llm):
        self._llm = llm

    def polish_batch(self, batch):
        prompt = ("你是中文错别字校对器。输入 JSON 数组(元素 {\"sample_id\":...,\"text\":...}),"
                  "输出同长度、同顺序、sample_id 原样的数组;只修正明显错别字/漏字/语法硬伤,"
                  "保持口语与原意,不改写风格;不确定就原样返回文本。只输出 JSON 数组。\n"
                  + json.dumps(batch, ensure_ascii=False))
        resp = self._llm.invoke([("user", prompt)])
        content = resp.content if isinstance(resp.content, str) else ""
        start, end = content.find("["), content.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("polish 响应不含 JSON 数组")
        return json.loads(content[start:end + 1])


def merge_real(records, session_factory):
    """并入 lcq 真实池:只走确定性清洗(用户原话不做 LLM 改写),文本精确去重。
    返回 (合并后记录, 读出条数, 并入条数, 快照时间)。"""
    from training.prep_records import load_real_questions
    snapshot_at = datetime.now(timezone.utc).isoformat()
    fresh = deterministic_clean(load_real_questions(session_factory))
    seen = {r["text"] for r in records}
    added = []
    for r in fresh:
        if r["text"] and r["text"] not in seen:
            seen.add(r["text"])
            added.append(r)
    return records + added, len(fresh), len(added), snapshot_at


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清洗 + LLM 错别字校对")
    parser.add_argument("infile", help="输入 jsonl(如 raw_synthetic.jsonl)")
    parser.add_argument("outfile", help="输出 jsonl(如 cleaned_synthetic.jsonl)")
    parser.add_argument("--real", action="store_true",
                        help="并入 lcq 真实池(确定性清洗,不走 LLM 校对)")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(argv)

    from app.config import Settings
    records = read_jsonl(args.infile)
    deterministic_clean(records)
    records, failed = llm_polish(records, _ChatPolisher(make_chat_model(Settings())),
                                 batch_size=args.batch_size)
    print(f"[prep_clean] 输入 {len(records)} 条,LLM 校对失败批 {failed}")
    if args.real:
        from app.db import make_engine, make_session_factory
        engine = make_engine(Settings().database_url)
        try:
            records, total, added, snapshot_at = merge_real(
                records, make_session_factory(engine))
        finally:
            engine.dispose()
        print(f"[prep_clean] 真实池快照 {snapshot_at}:读出 {total} 条,"
              f"去重后并入 {added} 条")
    write_jsonl(args.outfile, records)
    print(f"[prep_clean] 写出 {len(records)} 条 → {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
