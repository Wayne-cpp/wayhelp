"""验证集全局阈值扫描(spec §5):0.05–0.95 步长 0.01,micro-F1 最优;
并列先取最接近 0.5,仍并列取较大者。解码口径与 labels.decode_scores 完全一致。"""

from __future__ import annotations

import numpy as np

from training.labels import decode_scores, label_names


def _micro_f1(scores: np.ndarray, truths: np.ndarray, threshold: float) -> float:
    # 名单取冻结类目表前 k 个(k=scores 列数):k=17 时与线上解码完全同口径,
    # 窄矩阵供测试用(decode_scores 维度校验按名单长度)
    names = label_names()[: scores.shape[1]]
    tp = fp = fn = 0
    for row, truth in zip(scores, truths):
        pred = set(decode_scores([float(x) for x in row], threshold, names))
        gold = {names[i] for i, v in enumerate(truth) if v}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def sweep_global_threshold(scores: np.ndarray, truths: np.ndarray) -> float:
    best_f1, candidates = -1.0, []
    for i in range(5, 96):  # 0.05..0.95
        th = i / 100
        f1 = _micro_f1(scores, truths, th)
        if f1 > best_f1 + 1e-12:
            best_f1, candidates = f1, [th]
        elif abs(f1 - best_f1) <= 1e-12:
            candidates.append(th)
    candidates.sort(key=lambda t: (abs(t - 0.5), -t))
    return candidates[0]
