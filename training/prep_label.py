"""步骤标注:LLM 预标(QA/真实数据)+ 人工抽审导出 + corrections 回灌(spec §3)。

CLI 子命令:prelabel / export-review / apply-corrections。
"""

from __future__ import annotations

import argparse
import json
import logging
import random

from training.labels import label_names
from training.prep_records import make_chat_model, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def parse_prelabel(text: str, valid_names: list[str] | None = None) -> list[str]:
    names = valid_names or label_names()
    start, end = text.find("["), text.rfind("]")
    raw = json.loads(text[start:end + 1]) if start >= 0 else []
    keep = [n for n in names if n in set(raw) and n != "其他"]
    return keep if keep else ["其他"]  # 零具体命中兜底「其他」


def disagreements(records) -> list[dict]:
    """合成数据:预标与意图标签不一致的进人工裁决队列。"""
    return [r for r in records
            if r.get("intended_labels") is not None
            and sorted(r["labels"]) != sorted(r["intended_labels"])]


def apply_corrections(records, corrections) -> list[dict]:
    by_id = {c["sample_id"]: c["labels"] for c in corrections}
    for r in records:
        if r["sample_id"] in by_id:
            r["labels"] = list(by_id[r["sample_id"]])
    return records


def export_review_markdown(records, path, per_class: float = 0.1, seed: int = 42):
    """每类随机抽 ~10% 导出人工抽审 markdown(spec §3)。"""
    rng = random.Random(seed)
    lines = ["# 标注人工抽审", ""]
    by_label: dict[str, list[dict]] = {}
    for r in records:
        for lb in r["labels"]:
            by_label.setdefault(lb, []).append(r)
    picked = []
    for lb, rs in sorted(by_label.items()):
        k = max(1, round(len(rs) * per_class))
        picked.extend(rng.sample(rs, min(k, len(rs))))
    for r in picked:
        lines.append(f"- `{r['sample_id']}` {r['text']} → {json.dumps(r['labels'], ensure_ascii=False)}")
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return len(picked)


class _ChatPrelabeler:
    """把 ChatOpenAI 适配成 prelabel_records 需要的 label_batch 接口(仅 CLI 用)。"""

    def __init__(self, llm):
        self._llm = llm

    def label_batch(self, batch):
        prompt = ("你是电商客服问题多标签分类器。类目只能从 "
                  f"{json.dumps(label_names(), ensure_ascii=False)} 中选。"
                  "输入 JSON 数组(元素 {\"sample_id\":...,\"text\":...}),"
                  "输出同长度、同顺序、sample_id 原样的数组,元素为 "
                  "{\"sample_id\":...,\"labels\":[...]};一句话字面提到几个诉求就打几个标签,"
                  "都不沾边输出 [\"其他\"]。只输出 JSON 数组。\n"
                  + json.dumps(batch, ensure_ascii=False))
        resp = self._llm.invoke([("user", prompt)])
        content = resp.content if isinstance(resp.content, str) else ""
        start, end = content.find("["), content.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("prelabel 响应不含 JSON 数组")
        return json.loads(content[start:end + 1])


def prelabel_records(records, labeler, batch_size: int = 20):
    """与 prep_clean.llm_polish 同款 ID 对齐校验;失败批保留原 labels。
    返回 (records, failed_batches)。"""
    failed = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        expect_ids = [r["sample_id"] for r in batch]
        try:
            fixed = labeler.label_batch(
                [{"sample_id": r["sample_id"], "text": r["text"]} for r in batch])
        except Exception as exc:
            log.warning("prelabel 批 %d 异常保留原标签:%s", start // batch_size, exc)
            failed += 1
            continue
        got_ids = [f.get("sample_id") for f in fixed] if isinstance(fixed, list) else []
        if sorted(got_ids) != sorted(expect_ids) or len(set(got_ids)) != len(got_ids):
            log.warning("prelabel 批 %d ID 不对齐,保留原标签", start // batch_size)
            failed += 1
            continue
        by_id = {f["sample_id"]: f.get("labels") or [] for f in fixed}
        for r in batch:
            r["labels"] = parse_prelabel(
                json.dumps(by_id[r["sample_id"]], ensure_ascii=False))
    return records, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 预标 + 抽审导出 + 回灌")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prelabel", help="LLM 批量预标(jsonl → jsonl)")
    p.add_argument("infile")
    p.add_argument("outfile")
    p.add_argument("--batch-size", type=int, default=20)
    p = sub.add_parser("export-review", help="每类 ~10% 抽审 markdown")
    p.add_argument("infile")
    p.add_argument("outfile")
    p = sub.add_parser("apply-corrections",
                       help="corrections.jsonl 回灌(逐条 {sample_id, labels})")
    p.add_argument("infile")
    p.add_argument("corrections")
    p.add_argument("outfile")
    args = parser.parse_args(argv)

    if args.command == "prelabel":
        from app.config import Settings
        records = read_jsonl(args.infile)
        labeler = _ChatPrelabeler(make_chat_model(Settings()))
        records, failed = prelabel_records(records, labeler, args.batch_size)
        print(f"[prelabel] 预标 {len(records)} 条,失败批 {failed},"
              f"与意图分歧 {len(disagreements(records))} 条(人工裁决队列)")
        write_jsonl(args.outfile, records)
    elif args.command == "export-review":
        records = read_jsonl(args.infile)
        picked = export_review_markdown(records, args.outfile)
        print(f"[export-review] 抽审 {picked} 条 → {args.outfile}")
    else:
        records = apply_corrections(read_jsonl(args.infile),
                                    read_jsonl(args.corrections))
        write_jsonl(args.outfile, records)
        print(f"[apply-corrections] 回灌完成 {len(records)} 条 → {args.outfile},"
              f"剩余分歧 {len(disagreements(records))} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
