"""ch09 评估版本化:内容哈希稳定;冻结 artifact 与 settings 不一致即拒跑。"""
import json

import pytest

from evals.rag_eval_report import build_rag_eval, validate_rag_eval
from evals.run_retrieval_compare import (
    STRATEGIES, check_evidence_frozen, content_sha256, load_frozen_evidence,
)
from tests.conftest import make_settings

# 与「显式 uncalibrated settings」完全一致的 artifact:check 放行
# (2026-10-09 回填后 config 默认已是校准版,旧口径一律显式声明)
UNCALIBRATED_ARTIFACT = {
    "version": "uncalibrated",
    "weights": {"top1": 1.0, "count": 0.0, "margin": 0.0},
    "min_effective_score": 0.0,
    "threshold": 0.0553,
}


def _uncal_settings():
    return make_settings(evidence_confidence_version="uncalibrated",
                         evidence_weight_top1=1.0, evidence_weight_count=0.0,
                         evidence_weight_margin=0.0,
                         evidence_min_effective_score=0.0,
                         evidence_min_confidence=0.0553)


def test_content_sha256_stable_and_order_independent(tmp_path):
    (tmp_path / "b.md").write_text("B", encoding="utf-8")
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    h1 = content_sha256([tmp_path / "a.md", tmp_path / "b.md"])
    h2 = content_sha256([tmp_path / "b.md", tmp_path / "a.md"])
    assert h1 == h2 and len(h1) == 64
    (tmp_path / "a.md").write_text("AA", encoding="utf-8")
    assert content_sha256([tmp_path / "a.md", tmp_path / "b.md"]) != h1


def test_frozen_mismatch_rejected():
    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")  # 默认 uncalibrated
    artifact = {"version": "v2026", "weights": {"top1": 0.6, "count": 0.2,
                "margin": 0.2}, "min_effective_score": 0.05, "threshold": 0.4}
    with pytest.raises(SystemExit):
        check_evidence_frozen(s, artifact)


def test_frozen_match_accepted():
    check_evidence_frozen(_uncal_settings(), UNCALIBRATED_ARTIFACT)   # 不抛即过


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(version="v2"),                       # 版本不一致
    lambda a: a.update(threshold=0.9),                      # 阈值不一致
    lambda a: a.update(min_effective_score=0.05),           # 有效分下限不一致
    lambda a: a["weights"].update(top1=0.6),                # 任一权重不一致
    lambda a: a["weights"].update(count=0.2),
    lambda a: a["weights"].update(margin=0.2),
    lambda a: a.pop("threshold"),                           # 键缺失同样拒跑
    lambda a: a.pop("weights"),
])
def test_frozen_any_field_mismatch_or_missing_rejected(mutate):
    artifact = json.loads(json.dumps(UNCALIBRATED_ARTIFACT))
    mutate(artifact)
    with pytest.raises(SystemExit):
        check_evidence_frozen(_uncal_settings(), artifact)   # 基线一致,变异即拒


def test_load_frozen_evidence_roundtrip(tmp_path):
    path = tmp_path / "evidence_confidence.json"
    path.write_text(json.dumps(UNCALIBRATED_ARTIFACT), encoding="utf-8")
    assert load_frozen_evidence(path) == UNCALIBRATED_ARTIFACT


def test_load_frozen_evidence_missing_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit) as ei:
        load_frozen_evidence(tmp_path / "absent.json")
    assert "--calibrate-evidence" in str(ei.value)   # 整轮失败并提示先校准


def test_load_frozen_evidence_corrupt_exits(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_frozen_evidence(path)


# ─── rag_eval.json meta 版本键 + hybrid_rerank overall recall_at_10 ──────────


class _S:  # 最小 settings
    embedding_model = "bge-m3"
    rerank_model = "bge-reranker-v2-m3"
    model_name = "m"


def _report():
    tm = {s: {"threshold": 0.5, "section_recall_at_5": 0.1,
              "section_recall_at_10": 0.2, "complete_hit_at_5": 0.3,
              "complete_hit_at_10": 0.4, "mrr_at_10": 0.5,
              "d_refuse_correct_rate": 1.0, "over_refusal_rate": 0.0,
              "by_bucket": {}} for s in STRATEGIES}
    gen = {s: {"answerable_total": 1, "answered": 1, "answerable_refused": 0,
               "judged": 1, "faithful": 1, "fabricated": 0, "judge_errors": 0,
               "answer_coverage": 1.0, "faithful_rate": 1.0, "d_total": 1,
               "d_refused": 1, "d_refuse_rate": 1.0, "degraded": 0,
               "by_bucket": {}} for s in STRATEGIES}
    return build_rag_eval(
        run_id="r1", ts="2026-10-09T00:00:00+00:00", elapsed_s=1.0,
        settings=_S(), cases=[{"id": "A2", "bucket": "A_policy", "split": "test"},
                              {"id": "D2", "bucket": "D_absent", "split": "test"}],
        corpus_chunks=50, thresholds={s: {"threshold": 0.5} for s in STRATEGIES},
        test_metrics=tm, gen_agg=gen, faithfulness_cases=[],
        gates={"passed": True},
        corpus_version="c" * 64, dataset_version="d" * 64,
        evidence_confidence_version="uncalibrated")


def test_build_rag_eval_meta_versions_and_recall_at_10():
    rep = _report()
    m = rep["meta"]
    assert m["corpus_mode"] == "knowledge_docs_baseline"
    assert m["corpus_version"] == "c" * 64 and m["dataset_version"] == "d" * 64
    assert m["evidence_confidence_version"] == "uncalibrated"
    hr = rep["retrieval"]["hybrid_rerank"]["overall"]
    # recall_at_10 = 既有 section_recall@10 聚合,与 recall5 并列;只加在在线策略臂
    assert hr["recall_at_10"] == 0.2 == hr["sr10"]
    assert "recall_at_10" not in rep["retrieval"]["dense"]["overall"]
    validate_rag_eval(rep)   # 不抛即过


def test_validate_requires_version_meta():
    rep = _report()
    stripped = {**rep, "meta": {k: v for k, v in rep["meta"].items()
                                if k != "corpus_mode"}}
    with pytest.raises(ValueError):
        validate_rag_eval(stripped)
