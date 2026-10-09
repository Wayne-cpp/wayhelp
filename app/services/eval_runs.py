"""ch09:eval_runs 落表——报告发布回调幂等记录一轮评估。"""

from sqlalchemy.exc import IntegrityError

from app.models import EvalRun

# spec §6 指标契约八键(缺失存 None,不用 0 伪造)
_METRIC_KEYS = ("online_strategy", "recall_at_10", "mrr", "faithfulness",
                "coverage", "sr10", "quality_passed", "evidence_confidence_version")


def _metrics_of(report: dict) -> dict:
    overall = ((report.get("retrieval") or {}).get("hybrid_rerank") or {}).get("overall") or {}
    gen = report.get("generation") or {}
    strategy = gen.get("online_strategy") or "hybrid_rerank"
    arm = gen.get(strategy) or {}   # rag_eval.json 契约:策略臂平铺在 generation 下
    meta = report.get("meta") or {}
    gates = report.get("gates") or {}
    return {
        "online_strategy": strategy,
        "recall_at_10": overall.get("recall_at_10"),
        "mrr": overall.get("mrr"),
        "faithfulness": arm.get("faithful_rate"),
        "coverage": overall.get("evidence_coverage"),
        "sr10": overall.get("sr10"),
        "quality_passed": bool(gates.get("passed")),
        "evidence_confidence_version": meta.get("evidence_confidence_version"),
    }


def record_run(session_factory, report: dict, triggered_by: str) -> bool:
    """meta.run_id 唯一幂等;缺失指标存 None,不用 0 伪造(spec §6)。"""
    meta = report.get("meta") or {}
    row = EvalRun(run_id=meta["run_id"], triggered_by=triggered_by,
                  dataset_size=int(meta["test_cases"]),
                  corpus_mode=meta.get("corpus_mode", "knowledge_docs_baseline"),
                  corpus_version=meta["corpus_version"],
                  dataset_version=meta["dataset_version"],
                  metrics=_metrics_of(report))
    with session_factory() as s:
        s.add(row)
        try:
            s.commit()
            return True
        except IntegrityError:
            s.rollback()
            return False


def list_runs(session_factory, limit: int = 90) -> list[dict]:
    with session_factory() as s:
        rows = (s.query(EvalRun).order_by(EvalRun.created_at.desc(),
                                          EvalRun.id.desc()).limit(limit).all())
    out = [{"run_id": r.run_id, "triggered_by": r.triggered_by,
            "dataset_size": r.dataset_size, "corpus_mode": r.corpus_mode,
            "corpus_version": r.corpus_version, "dataset_version": r.dataset_version,
            "metrics": r.metrics,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
           for r in rows]
    return out[::-1]  # 升序供趋势图直接画
