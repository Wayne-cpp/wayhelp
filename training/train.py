"""RoBERTa-wwm-ext 全参微调(CPU),early stopping + 阈值扫描(spec §5)。

CLI: uv run --group train python -m training.train [--smoke]
产物:models/topic_classifier_work/best/(checkpoint)+ val_scores.npy + threshold.json
(_work 迁出模型包目录:包目录 models/topic_classifier/ 是导出 target,导出前会被清空)。
torch/transformers/datasets 全部延迟进 main,dev 闭包可 import 收集。

锁版 API 核对(transformers 4.57.6,Context7 + 安装源内省):
Trainer 无 tokenizer 形参只用 processing_class;TrainingArguments 用 eval_strategy
(evaluation_strategy 已移除)、use_cpu;EarlyStoppingCallback 需
metric_for_best_model + eval_strategy;problem_type="multi_label_classification"
走 BCEWithLogitsLoss,labels 须为 float。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from training.labels import decode_scores, label_names
from training.prep_records import read_jsonl

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
WORK_DIR = Path("models/topic_classifier_work")
DATA_DIR = Path("data/topic_classifier")
MAX_LEN = 64
SEED = 42


def _load(split: str):
    records = read_jsonl(DATA_DIR / f"{split}.jsonl")
    names = label_names()
    texts = [r["text"] for r in records]
    y = np.array([[1.0 if n in r["labels"] else 0.0 for n in names]
                  for r in records], dtype=np.float32)
    return texts, y


def main(smoke: bool = False):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments)

    names = label_names()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], max_length=MAX_LEN,
                         truncation=True, padding="max_length")

    train_texts, train_y = _load("train")
    val_texts, val_y = _load("val")
    if smoke:  # 冒烟:50 条验证管线,不要求类别覆盖(spec §4.3)
        train_texts, train_y = train_texts[:50], train_y[:50]
        val_texts, val_y = val_texts[:20], val_y[:20]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = Dataset.from_dict(
        {"text": train_texts, "labels": train_y.tolist()}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict(
        {"text": val_texts, "labels": val_y.tolist()}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(names),
        problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)})

    def compute_metrics(eval_pred):
        scores = 1 / (1 + np.exp(-eval_pred.predictions))  # sigmoid
        tp = fp = fn = 0
        for row, truth in zip(scores, eval_pred.label_ids):
            pred = set(decode_scores([float(x) for x in row], 0.5, names))
            gold = {names[i] for i, v in enumerate(truth) if v}
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        return {"micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}

    args = TrainingArguments(
        output_dir=str(WORK_DIR), learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=10, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="micro_f1",
        seed=SEED, use_cpu=True, report_to=[], save_total_limit=2)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    best = WORK_DIR / "best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)

    # 阈值扫描:选定 checkpoint 后,验证集 0.05–0.95(spec §5)
    from training.threshold import sweep_global_threshold
    out = trainer.predict(val_ds)
    scores = 1 / (1 + np.exp(-out.predictions))
    threshold = sweep_global_threshold(scores, val_y)
    np.save(WORK_DIR / "val_scores.npy", scores)
    (WORK_DIR / "threshold.json").write_text(
        json.dumps({"effective_threshold": threshold}, indent=2), encoding="utf-8")
    print(f"[train] train={len(train_texts)} val={len(val_texts)} "
          f"best 已存 {best},阈值={threshold}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
