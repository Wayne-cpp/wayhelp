"""生成段聚合口径(rag-eval 规格:生成指标口径一节)。"""

import json

import pytest

from evals.rag_eval_report import (
    aggregate_generation, build_rag_eval, check_generation_invariants,
    new_run_id, publish_rag_eval, validate_rag_eval,
)
from evals.run_retrieval_compare import STRATEGIES


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


class _S:  # 最小 settings
    embedding_model = "bge-m3"
    rerank_model = "bge-reranker-v2-m3"
    model_name = "m"


def _cases():
    return [{"id": "A1", "bucket": "A_policy", "split": "calibration"},
            {"id": "A2", "bucket": "A_policy", "split": "test"},
            {"id": "D2", "bucket": "D_absent", "split": "test"}]


def _metrics():
    tm = {}
    for s in STRATEGIES:
        tm[s] = {"threshold": 0.5, "section_recall_at_5": 0.1, "section_recall_at_10": 0.2,
                 "complete_hit_at_5": 0.3, "complete_hit_at_10": 0.4, "mrr_at_10": 0.5,
                 "d_refuse_correct_rate": 1.0, "over_refusal_rate": 0.0,
                 "by_bucket": {"A_policy": {"section_recall_at_10": 0.2, "mrr_at_10": 0.5,
                                            "recall5": 0.1, "evidence_coverage": 0.4,
                                            "refused": 0}}}
    return tm


def _gen_agg():
    base = aggregate_generation([
        _row("A2", "A_policy", judge="faithful", coverage=1.0),
        _row("D2", "D_absent", refused=True, should_refuse=True),
    ])
    return {s: dict(base) for s in STRATEGIES}


def _build():
    return build_rag_eval(
        run_id="r1", ts="2026-09-16T00:00:00+00:00", elapsed_s=1.5, settings=_S(),
        cases=_cases(), corpus_chunks=50,
        thresholds={s: {"threshold": 0.5} for s in STRATEGIES},
        test_metrics=_metrics(), gen_agg=_gen_agg(),
        faithfulness_cases=[], gates={"judge_error_zero": True,
                                      "bm25_b_model_section_recall_at_10": 1.0,
                                      "bm25_b_model_gate": True, "passed": True})


def test_build_rag_eval_shape():
    rep = _build()
    assert set(rep) == {"meta", "retrieval", "generation", "gates"}
    m = rep["meta"]
    assert m["total_cases"] == 3 and m["calibration_cases"] == 1 and m["test_cases"] == 2
    assert m["answerable_test_cases"] == 1 and m["d_test_cases"] == 1
    assert m["corpus_chunks"] == 50 and m["embedding_model"] == "bge-m3"
    dense = rep["retrieval"]["dense"]
    assert dense["overall"] == {"mrr": 0.5, "recall5": 0.1,
                                "evidence_coverage": 0.4, "sr10": 0.2}
    assert dense["by_bucket"]["A_policy"] == {"mrr": 0.5, "recall5": 0.1,
                                              "evidence_coverage": 0.4}
    assert rep["generation"]["online_strategy"] == "hybrid_rerank"
    assert rep["generation"]["dense"]["answer_coverage"] == 1.0
    validate_rag_eval(rep)  # 不抛即过


def test_validate_rejects_bad():
    import pytest
    with pytest.raises(ValueError):
        validate_rag_eval({"meta": {}})
    bad = _build()
    bad["generation"]["dense"]["answered"] = 99   # 破坏不变量
    with pytest.raises(ValueError):
        validate_rag_eval(bad)


def test_publish_atomic(tmp_path):
    old = tmp_path / "rag_eval.json"
    old.write_text('{"old": true}', encoding="utf-8")
    path = publish_rag_eval(_build(), tmp_path)
    assert path == old
    assert json.loads(old.read_text(encoding="utf-8"))["meta"]["run_id"] == "r1"
    assert not list(tmp_path.glob(".rag_eval-*"))

    # 校验失败:旧文件字节不变,不留临时文件
    import pytest
    bad = _build()
    bad["generation"]["dense"]["fabricated"] = 99
    with pytest.raises(ValueError):
        publish_rag_eval(bad, tmp_path)
    assert json.loads(old.read_text(encoding="utf-8"))["meta"]["run_id"] == "r1"
    assert not list(tmp_path.glob(".rag_eval-*"))


def test_new_run_id_unique():
    from datetime import datetime, timezone
    a = new_run_id(datetime.now(timezone.utc))
    b = new_run_id(datetime.now(timezone.utc))
    assert a != b and len(a) == len("20260916T024957123456Z")
