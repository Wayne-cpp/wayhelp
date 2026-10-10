"""阈值扫描:范围/步长/并列规则(spec §5)。"""

import numpy as np

from training.threshold import sweep_global_threshold


def test_sweep_picks_best_micro_f1():
    # 0.11–0.40 网格区间 micro-F1=1.0 平台(0.4>=th 命中、0.1<th 未命中),
    # 并列取最接近 0.5 → 0.40(计划草案期望 0.3 与其自身数据不符,按实际行为修正)
    scores = np.array([[0.4, 0.1], [0.1, 0.4]])
    truths = np.array([[1, 0], [0, 1]])
    assert sweep_global_threshold(scores, truths) == 0.4


def test_tie_prefers_closer_to_half_then_larger():
    # 所有候选阈值 F1 相同 → 取最接近 0.5;仍并列取较大者
    scores = np.array([[0.99, 0.01]])
    truths = np.array([[1, 0]])
    got = sweep_global_threshold(scores, truths)
    assert got == 0.5
