"""rag_eval.json 规范化产物:生成段聚合口径、计数不变量、构建、契约校验、原子发布。

规格:docs/superpowers/specs/2026-09-16-rag-eval-page-design.md(生成指标口径一节)。
"""

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
