"""评测报告渲染:逐类 P/R/F1/support + 每类混淆矩阵(spec §6)。"""

import numpy as np

from training.evaluate import compute_metrics_per_class, render_report


def test_per_class_metrics_and_confusion():
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.9, 0.9], [0.1, 0.1]])
    truths = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    m = compute_metrics_per_class(scores, truths, 0.5, ["退换货", "尺码"])
    a = m["per_class"]["退换货"]
    assert a["precision"] == 1.0 and a["recall"] == 1.0 and a["support"] == 2
    # 「退换货」one-vs-rest:r1/r3 命中且为金 → tp=2;r2 只预测尺码、r4 零命中
    # 兜底「其他」不在此列 → 均为 tn(计划草案 {tp:2,fp:1,fn:0,tn:1} 与其自身
    # precision==1.0 断言矛盾:fp=1 则 precision=0.667,按实际行为修正)
    cm = a["confusion"]
    assert cm == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert "micro_f1" in m and "macro_f1" in m


def test_render_report_contains_sections():
    m = {"per_class": {"物流": {"precision": 1.0, "recall": 0.5, "f1": 0.667,
                                "support": 2,
                                "confusion": {"tp": 1, "fp": 0, "fn": 1, "tn": 8}}},
         "micro_f1": 0.9, "macro_f1": 0.667}
    md = render_report(m, {"model_version": "m", "dataset_version": "d",
                           "effective_threshold": 0.5, "decoder_version": "topic-decoder-v1"})
    assert "# 主题分类器留出测试报告" in md
    assert "| 物流 | 1.000 | 0.500 | 0.667 | 2 |" in md
    assert "混淆矩阵" in md and "topic-decoder-v1" in md
