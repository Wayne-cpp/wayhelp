"""步骤4 数据增强:LLM 同义词替换/句式微调,只扩训练集(spec §4.4)。

派生样本继承标签、来源组与训练集归属;与三份现有文本重复的候选丢弃;
不移动验证/测试样本凑数量。CLI 的 LLM 改写部分手跑验证,不进 pytest。
CLI: uv run --group train python -m training.prep_augment \
       --manifest data/topic_classifier/split_manifest.json [--target-ratio 1.3]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from training.prep_records import make_chat_model, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def plan_augmentation(train_records, all_texts, target_ratio: float = 1.3) -> list[dict]:
    """计算要增强哪些样本:目标 train * target_ratio,每条 parent ≤1 派生。"""
    need = max(0, round(len(train_records) * target_ratio) - len(train_records))
    return [{"parent_id": r["sample_id"], "text": r["text"]}
            for r in train_records[:need]]


def apply_augmentation(train_records, variants: dict[str, str], all_texts,
                       assignment: dict[str, str]) -> list[dict]:
    by_id = {r["sample_id"]: r for r in train_records}
    out = []
    for pid, text in variants.items():
        parent = by_id.get(pid)
        if parent is None or assignment.get(pid) != "train":
            continue  # 只扩训练集
        text = (text or "").strip()
        if not text or text in all_texts:
            continue  # 重复候选丢弃
        out.append({"sample_id": f"aug-{pid}", "source": parent["source"],
                    "source_group_id": parent["source_group_id"],
                    "text_raw": text, "text": text,
                    "labels": list(parent["labels"]),
                    "intended_labels": list(parent["intended_labels"])
                    if parent.get("intended_labels") else None,
                    "parent_id": pid})
    return out


def llm_rewrite(plan, llm, batch_size: int = 20):
    """镜像 prep_clean.llm_polish 的 ID 对齐校验;llm 须实现
    rewrite_batch(list[{sample_id,text}]) -> list[{sample_id,text}]。
    返回 (variants: {parent_id: 改写文本}, failed_batches)。
    异常/ID 错位/条数不符 → 整批候选丢弃并记日志——派生样本没有「原文」可回退,
    原样保留必与既有文本重复而被丢弃,不如直接不生成。"""
    variants, failed = {}, 0
    for start in range(0, len(plan), batch_size):
        batch = plan[start:start + batch_size]
        expect_ids = [p["parent_id"] for p in batch]
        try:
            fixed = llm.rewrite_batch(
                [{"sample_id": p["parent_id"], "text": p["text"]} for p in batch])
        except Exception as exc:
            log.warning("rewrite 批 %d 异常丢弃:%s", start // batch_size, exc)
            failed += 1
            continue
        got_ids = [f.get("sample_id") for f in fixed] if isinstance(fixed, list) else []
        if sorted(got_ids) != sorted(expect_ids) or len(set(got_ids)) != len(got_ids):
            log.warning("rewrite 批 %d ID 不对齐,整批丢弃", start // batch_size)
            failed += 1
            continue
        for f in fixed:
            variants[f["sample_id"]] = f.get("text") or ""
    return variants, failed


class _ChatRewriter:
    """把 ChatOpenAI 适配成 llm_rewrite 需要的 rewrite_batch 接口(仅 CLI 用)。"""

    def __init__(self, llm):
        self._llm = llm

    def rewrite_batch(self, batch):
        prompt = ("你是中文客服问句改写器。输入 JSON 数组(元素 {\"sample_id\":...,\"text\":...}),"
                  "输出同长度、同顺序、sample_id 原样的数组;每条换一种口语说法:"
                  "同义词替换/语序调整/语气词增删,保持原意与全部诉求不变、不增删信息,"
                  "不修正错别字;改写与原文至少差两个字。只输出 JSON 数组。\n"
                  + json.dumps(batch, ensure_ascii=False))
        resp = self._llm.invoke([("user", prompt)])
        content = resp.content if isinstance(resp.content, str) else ""
        start, end = content.find("["), content.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("rewrite 响应不含 JSON 数组")
        return json.loads(content[start:end + 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="训练集数据增强:LLM 同义改写,只扩 train 到 ~target-ratio 倍")
    parser.add_argument("--manifest", default="data/topic_classifier/split_manifest.json",
                        help="split_manifest.json(同目录读 train/val/test.jsonl,"
                             "增强后原地改写 train.jsonl)")
    parser.add_argument("--target-ratio", type=float, default=1.3)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assignment = {e["sample_id"]: e["split"] for e in manifest["entries"]}
    data_dir = Path(args.manifest).parent
    records = {split: read_jsonl(data_dir / f"{split}.jsonl")
               for split in ("train", "val", "test")}
    # 幂等重跑:剥离既有派生样本(parent_id 非 None),从基线重算,避免 aug- 套 aug-
    base_train = [r for r in records["train"] if r.get("parent_id") is None]
    all_texts = {r["text"] for rs in records.values() for r in rs}

    plan = plan_augmentation(base_train, all_texts, target_ratio=args.target_ratio)
    print(f"[prep_augment] train 基线 {len(base_train)} 条,计划增强 {len(plan)} 条"
          f"(target_ratio={args.target_ratio})")
    if not plan:
        print("[prep_augment] 无需增强,train.jsonl 未改写")
        return 0

    from app.config import Settings
    variants, failed = llm_rewrite(plan, _ChatRewriter(make_chat_model(Settings())),
                                   batch_size=args.batch_size)
    aug = apply_augmentation(base_train, variants, all_texts, assignment)
    write_jsonl(data_dir / "train.jsonl", base_train + aug)
    print(f"[prep_augment] LLM 失败批 {failed},改写 {len(variants)} 条,"
          f"重复/无效丢弃 {len(variants) - len(aug)} 条;"
          f"train.jsonl {len(base_train)} → {len(base_train) + len(aug)} 条"
          f"(val/test 未动)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
