"""生成段聚合口径(rag-eval 规格:生成指标口径一节)。"""

import pytest

from evals.rag_eval_report import (
    aggregate_generation, check_generation_invariants,
)


def _row(cid, bucket, refused=False, judge=None, coverage=None,
         should_refuse=False, degraded=False):
    r = {"id": cid, "bucket": bucket, "should_refuse": should_refuse,
         "low_confidence": False, "refused": refused, "answer": "a",
         "degraded": degraded, "evidence": [], "judge": judge}
    if coverage is not None:
        r["coverage"] = coverage
    return r


def test_aggregate_counts_and_ratios():
    rows = [
        _row("A2", "A_policy", judge="faithful", coverage=1.0),
        _row("A4", "A_policy", judge="fabricated", coverage=0.5),
        _row("C2", "C_colloquial", refused=True),              # 拒答 coverage 记 0
        _row("D2", "D_absent", refused=True, should_refuse=True),
        _row("D4", "D_absent", judge=None, should_refuse=True),  # D 未拒答
    ]
    agg = aggregate_generation(rows)
    assert agg["answerable_total"] == 3 and agg["answered"] == 2
    assert agg["answerable_refused"] == 1
    assert agg["judged"] == 2 and agg["faithful"] == 1 and agg["fabricated"] == 1
    assert agg["judge_errors"] == 0
    assert agg["answer_coverage"] == pytest.approx(1.5 / 3)   # 拒答记 0,分母含拒答
    assert agg["faithful_rate"] == pytest.approx(0.5)         # 分母只含裁判成功的
    assert agg["d_total"] == 2 and agg["d_refused"] == 1
    assert agg["d_refuse_rate"] == pytest.approx(0.5)
    assert agg["by_bucket"]["C_colloquial"]["coverage"] == 0.0
    assert agg["by_bucket"]["A_policy"]["faithful_rate"] == pytest.approx(0.5)
    check_generation_invariants(agg)                          # 不抛即过


def test_aggregate_judge_error_nulls():
    """任一裁判失败:策略与受影响桶的比率为 null;未受影响桶照常。"""
    rows = [
        _row("A2", "A_policy", judge="judge_error"),
        _row("C2", "C_colloquial", judge="faithful", coverage=1.0),
    ]
    agg = aggregate_generation(rows)
    assert agg["answer_coverage"] is None and agg["faithful_rate"] is None
    assert agg["by_bucket"]["A_policy"]["coverage"] is None
    assert agg["by_bucket"]["A_policy"]["faithful_rate"] is None
    assert agg["by_bucket"]["C_colloquial"]["faithful_rate"] == 1.0
    assert agg["judge_errors"] == 1 and agg["judged"] == 1
    check_generation_invariants(agg)   # judged + judge_errors == answered 仍成立


def test_check_generation_invariants_raises():
    with pytest.raises(RuntimeError):
        check_generation_invariants({"answered": 2, "answerable_refused": 1,
                                     "answerable_total": 4,   # 2+1 != 4
                                     "judged": 2, "judge_errors": 0,
                                     "faithful": 1, "fabricated": 1,
                                     "d_refused": 0, "d_total": 0})
