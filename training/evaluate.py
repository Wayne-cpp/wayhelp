"""留出测试评测:逐类 P/R/F1/support + one-vs-rest 混淆矩阵 + 错例导出(spec §6)。

CLI: uv run --group train python -m training.evaluate \
       --ckpt models/topic_classifier/_work/best
读 test.jsonl 与 _work/threshold.json,报告落 evals/results/<UTC时间戳>_topic_eval.md,
错例导出 <UTC时间戳>_topic_errors.md(产物 gitignore,不进 pytest、不入库)。
纯函数 compute_metrics_per_class/render_report 只依赖 numpy + training.labels,
torch/transformers 延迟到 CLI 路径,保证 dev 闭包可收集。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.labels import OTHER, decode_scores

DATA_DIR = Path("data/topic_classifier")
EVAL_RESULTS = Path("evals/results")
MAX_LEN = 64


def compute_metrics_per_class(scores, truths, threshold, names) -> dict:
    """逐类 one-vs-rest:预测走统一解码(零命中兜底「其他」),金标取 truths 列;
    「其他」是解码兜底不单列指标(spec §6)。"""
    per_class = {}
    tps = fps = fns = 0
    for idx, name in enumerate(names):
        if name == OTHER:
            continue
        tp = fp = fn = tn = 0
        for row, truth in zip(scores, truths):
            pred = decode_scores([float(x) for x in row], threshold, names)
            p = name in pred
            g = bool(truth[idx])
            tp += p and g
            fp += p and not g
            fn += (not p) and g
            tn += (not p) and not g
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        support = int(sum(bool(t[idx]) for t in truths))
        per_class[name] = {"precision": round(precision, 3),
                           "recall": round(recall, 3), "f1": round(f1, 3),
                           "support": support,
                           "confusion": {"tp": int(tp), "fp": int(fp),
                                         "fn": int(fn), "tn": int(tn)}}
        tps += tp
        fps += fp
        fns += fn
    f1s = [c["f1"] for c in per_class.values()]
    micro_p = tps / (tps + fps) if tps + fps else 0.0
    micro_r = tps / (tps + fns) if tps + fns else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)
    return {"per_class": per_class,
            "micro_f1": round(micro_f1, 3),
            "macro_f1": round(sum(f1s) / len(f1s), 3)}


def render_report(metrics: dict, meta: dict) -> str:
    pinned = ("model_version", "dataset_version", "effective_threshold",
              "decoder_version")
    lines = ["# 主题分类器留出测试报告", ""]
    lines += [f"- {k}: {meta[k]}" for k in pinned if k in meta]
    lines += [f"- {k}: {v}" for k, v in meta.items() if k not in pinned]
    lines += ["", "## 逐类指标", "",
              "| 类目 | P | R | F1 | support |",
              "| --- | --- | --- | --- | --- |"]
    for name, c in metrics["per_class"].items():
        lines.append(f"| {name} | {c['precision']:.3f} | {c['recall']:.3f} "
                     f"| {c['f1']:.3f} | {c['support']} |")
    lines += ["", f"- micro-F1: {metrics['micro_f1']:.3f}",
              f"- macro-F1: {metrics['macro_f1']:.3f}",
              "", "## 每类混淆矩阵(one-vs-rest)", ""]
    for name, c in metrics["per_class"].items():
        cm = c["confusion"]
        lines.append(f"### {name}(support={c['support']})")
        lines.append(f"- TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
        lines.append("")
    return "\n".join(lines)


def _export_errors(path: Path, records, scores, truths, names, threshold) -> int:
    """FP/FN 错例导出:预测集合 ≠ 金标集合的样本逐条列出(spec §6 人工抽判输入)。"""
    lines = ["# 主题分类器留出测试错例(FP/FN)", ""]
    n = 0
    for r, row, truth in zip(records, scores, truths):
        pred = decode_scores([float(x) for x in row], threshold, names)
        gold = [nm for nm, v in zip(names, truth) if v]
        if set(pred) == set(gold):
            continue
        n += 1
        fps = [lb for lb in pred if lb not in gold]
        fns = [lb for lb in gold if lb not in pred]
        detail = " ".join(f"{lb}={float(row[names.index(lb)]):.3f}"
                          for lb in dict.fromkeys(fps + fns))
        lines.append(f"- `{r['sample_id']}` {r['text']}")
        lines.append(f"  - 金标:{gold} / 预测:{pred}(FP:{fps} FN:{fns};分数 {detail})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def _predict_scores(ckpt: Path, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """CPU 加载 best checkpoint → sigmoid 分数(与训练侧 compute_metrics 同口径)。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start + batch_size], max_length=MAX_LEN,
                            truncation=True, padding=True, return_tensors="pt")
            out.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    if not out:
        return np.zeros((0, 17), dtype=np.float32)
    return np.concatenate(out)


def _truncated_share(ckpt: Path, texts: list[str]) -> float:
    """截断比例:tokenize(含特殊符号)超 MAX_LEN 的测试文本占比。"""
    from transformers import AutoTokenizer
    if not texts:
        return 0.0
    tok = AutoTokenizer.from_pretrained(ckpt)
    cut = sum(len(tok(t)["input_ids"]) > MAX_LEN for t in texts)
    return round(cut / len(texts), 4)


def _model_version(ckpt: Path) -> str:
    """导出前(checkpoint 阶段)以权重文件 SHA-256 充当 model_version 记录入报告。"""
    import hashlib
    weights = ckpt / "model.safetensors"
    if not weights.exists():
        return f"ckpt:{ckpt}"
    h = hashlib.sha256()
    with weights.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from datetime import datetime, timezone

    from training.labels import DECODER_VERSION, label_names
    from training.prep_records import read_jsonl

    ap = argparse.ArgumentParser(description="ch10 主题分类器留出测试评测")
    ap.add_argument("--ckpt", default="models/topic_classifier/_work/best")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(EVAL_RESULTS))
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt)
    data_dir = Path(args.data_dir)
    names = label_names()
    records = read_jsonl(data_dir / "test.jsonl")
    threshold = float(json.loads(
        (ckpt.parent / "threshold.json").read_text(encoding="utf-8")
    )["effective_threshold"])

    texts = [r["text"] for r in records]
    truths = np.array([[1 if n in r["labels"] else 0 for n in names]
                       for r in records])
    scores = _predict_scores(ckpt, texts)
    metrics = compute_metrics_per_class(scores, truths, threshold, names)

    manifest_path = data_dir / "split_manifest.json"
    dataset_version = (json.loads(manifest_path.read_text(encoding="utf-8"))
                       ["dataset_version"] if manifest_path.exists() else "unknown")
    meta = {"model_version": _model_version(ckpt),
            "dataset_version": dataset_version,
            "effective_threshold": threshold,
            "decoder_version": DECODER_VERSION,
            "checkpoint": str(ckpt),
            "test_size": len(records),
            "synthetic": sum(r["source"] == "synthetic" for r in records),
            "real": sum(r["source"] == "real" for r in records),
            "multilabel_share": round(
                sum(len(r["labels"]) > 1 for r in records) / max(1, len(records)), 4),
            "truncated_share": _truncated_share(ckpt, texts)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"{ts}_topic_eval.md"
    report_path.write_text(render_report(metrics, meta), encoding="utf-8")
    errors_path = out_dir / f"{ts}_topic_errors.md"
    n_err = _export_errors(errors_path, records, scores, truths, names, threshold)
    print(f"[evaluate] micro_f1={metrics['micro_f1']} macro_f1={metrics['macro_f1']}")
    print(f"[evaluate] 报告 {report_path}(错例 {n_err} 条 → {errors_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
