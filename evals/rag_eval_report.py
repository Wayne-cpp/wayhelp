"""rag_eval.json 规范化产物:生成段聚合口径、计数不变量、构建、契约校验、原子发布。

规格:docs/superpowers/specs/2026-09-16-rag-eval-page-design.md(生成指标口径一节)。
"""
import json
import os
import tempfile
from pathlib import Path

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
ANSWERABLE_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
GENERATION_COUNTERS = ("answerable_total", "answered", "answerable_refused", "judged",
                       "faithful", "fabricated", "judge_errors", "d_total", "d_refused",
                       "degraded")
REPORT_FILENAME = "rag_eval.json"


def _bucket_agg(bp: list[dict]) -> dict:
    judged = [r for r in bp if r["judge"] in ("faithful", "fabricated")]
    has_error = any(r["judge"] == "judge_error" for r in bp)
    faithful = sum(1 for r in judged if r["judge"] == "faithful")
    return {
        "total": len(bp),
        "answered": sum(1 for r in bp if not r["refused"]),
        "refused": sum(1 for r in bp if r["refused"]),
        "judged": len(judged),
        "faithful": faithful,
        "fabricated": len(judged) - faithful,
        "judge_errors": sum(1 for r in bp if r["judge"] == "judge_error"),
        # 拒答 coverage 记 0,分母 = 桶全部 test 题;裁判失败 → null(不静默缩分母)
        "coverage": None if has_error else sum(r.get("coverage", 0.0) for r in judged) / len(bp),
        "faithful_rate": None if (has_error or not judged) else faithful / len(judged),
    }


def aggregate_generation(rows: list[dict]) -> dict:
    """单策略生成段聚合(test 分片)。rows 见 _run_generation 行结构。"""
    answerable = [r for r in rows if not r["should_refuse"]]
    d_rows = [r for r in rows if r["should_refuse"]]
    judged = [r for r in answerable if r["judge"] in ("faithful", "fabricated")]
    errors = [r for r in answerable if r["judge"] == "judge_error"]
    faithful = sum(1 for r in judged if r["judge"] == "faithful")
    has_error = bool(errors)
    return {
        "answerable_total": len(answerable),
        "answered": sum(1 for r in answerable if not r["refused"]),
        "answerable_refused": sum(1 for r in answerable if r["refused"]),
        "judged": len(judged),
        "faithful": faithful,
        "fabricated": len(judged) - faithful,
        "judge_errors": len(errors),
        "answer_coverage": (None if has_error or not answerable
                            else sum(r.get("coverage", 0.0) for r in judged) / len(answerable)),
        "faithful_rate": None if (has_error or not judged) else faithful / len(judged),
        "d_total": len(d_rows),
        "d_refused": sum(1 for r in d_rows if r["refused"]),
        "d_refuse_rate": (sum(1 for r in d_rows if r["refused"]) / len(d_rows)
                          if d_rows else None),
        "degraded": sum(1 for r in rows if r["degraded"]),
        "by_bucket": {b: _bucket_agg([r for r in answerable if r["bucket"] == b])
                      for b in ANSWERABLE_BUCKETS
                      if any(r["bucket"] == b for r in answerable)},
    }


def check_generation_invariants(agg: dict) -> None:
    """发布前校验计数不变量;违反即 RuntimeError(本轮不发布)。"""
    ok = (agg["answered"] + agg["answerable_refused"] == agg["answerable_total"]
          and agg["judged"] + agg["judge_errors"] == agg["answered"]
          and agg["faithful"] + agg["fabricated"] == agg["judged"]
          and agg["d_refused"] <= agg["d_total"])
    if not ok:
        raise RuntimeError(f"生成段计数不变量违反: {agg}")


def new_run_id(now) -> str:
    """含微秒的 UTC 紧凑时间戳;连续运行不重复。"""
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def build_rag_eval(*, run_id, ts, elapsed_s, settings, cases, corpus_chunks,
                   thresholds, test_metrics, gen_agg, faithfulness_cases, gates) -> dict:
    test = [c for c in cases if c["split"] == "test"]
    retrieval = {}
    for s in STRATEGIES:
        m = test_metrics[s]
        retrieval[s] = {
            "threshold": thresholds[s]["threshold"],
            "ungated": bool(thresholds[s].get("ungated")),
            "overall": {"mrr": m["mrr_at_10"], "recall5": m["section_recall_at_5"],
                        "evidence_coverage": m["complete_hit_at_10"],
                        "sr10": m["section_recall_at_10"]},
            "by_bucket": {b: {"mrr": bb["mrr_at_10"], "recall5": bb["recall5"],
                              "evidence_coverage": bb["evidence_coverage"]}
                          for b, bb in m["by_bucket"].items()},
        }
        if thresholds[s].get("signal"):   # hybrid_rerank:本轮校准胜出的闸门置信信号
            retrieval[s]["signal"] = thresholds[s]["signal"]
    return {
        "meta": {"run_id": run_id, "ts": ts, "elapsed_s": round(elapsed_s, 1),
                 "total_cases": len(cases),
                 "calibration_cases": len(cases) - len(test),
                 "test_cases": len(test),
                 "answerable_test_cases": sum(1 for c in test
                                              if c["bucket"] in ANSWERABLE_BUCKETS),
                 "d_test_cases": sum(1 for c in test if c["bucket"] == "D_absent"),
                 "corpus_chunks": corpus_chunks,
                 "embedding_model": settings.embedding_model,
                 "rerank_model": settings.rerank_model,
                 "eval_model": settings.model_name,
                 "judge_model": settings.model_name},
        "retrieval": retrieval,
        "generation": {"online_strategy": "hybrid_rerank",
                       **{s: gen_agg[s] for s in STRATEGIES},
                       "faithfulness_cases": faithfulness_cases},
        "gates": gates,
    }


def validate_rag_eval(data) -> None:
    """发布前契约校验;不合法即 ValueError。"""
    if not isinstance(data, dict) or any(
            k not in data for k in ("meta", "retrieval", "generation", "gates")):
        raise ValueError("rag_eval 缺顶层键 meta/retrieval/generation/gates")
    meta = data["meta"]
    for k in ("run_id", "ts", "total_cases", "calibration_cases", "test_cases",
              "answerable_test_cases", "d_test_cases", "corpus_chunks",
              "embedding_model", "rerank_model", "eval_model", "judge_model"):
        if k not in meta:
            raise ValueError(f"rag_eval meta 缺 {k}")
    gen = data["generation"]
    if gen.get("online_strategy") != "hybrid_rerank":
        raise ValueError("online_strategy 非 hybrid_rerank")
    if not isinstance(gen.get("faithfulness_cases"), list):
        raise ValueError("faithfulness_cases 非列表")
    for s in STRATEGIES:
        r = data["retrieval"].get(s)
        if not isinstance(r, dict) or "overall" not in r or "by_bucket" not in r:
            raise ValueError(f"rag_eval retrieval 缺 {s}")
        g = gen.get(s)
        if not isinstance(g, dict) or any(k not in g for k in GENERATION_COUNTERS):
            raise ValueError(f"rag_eval generation 缺 {s} 计数")
        try:
            check_generation_invariants(g)
        except RuntimeError as exc:          # 不变量违反统一视为校验失败
            raise ValueError(str(exc)) from exc


def publish_rag_eval(data, results_dir) -> Path:
    """原子发布:先校验;临时文件写全后 os.replace;失败清理临时文件并保留旧报告。"""
    validate_rag_eval(data)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / REPORT_FILENAME
    fd, tmp = tempfile.mkstemp(dir=results_dir, prefix=".rag_eval-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
