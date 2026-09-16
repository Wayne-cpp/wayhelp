"""/rag-eval 页面数据服务:固定报告加载/契约校验、faith_cases 台账查询与处置、两口径统计。

report 与 stats 共用 load_report(契约校验同一处);报告损坏时 faith-cases 降级为
「无当前报告」口径,台账处置不被报告文件阻断。
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func

from app.models import FaithCase

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
STATUSES = ("未解决", "已解决", "无需解决")


class ReportCorruptError(Exception):
    """rag_eval.json 存在但解析/契约不合法。"""


def validate_report(data) -> None:
    if not isinstance(data, dict) or any(
            k not in data for k in ("meta", "retrieval", "generation", "gates")):
        raise ReportCorruptError("缺顶层键 meta/retrieval/generation/gates")
    meta = data["meta"]
    if not isinstance(meta, dict) or any(
            k not in meta for k in ("run_id", "ts", "total_cases", "test_cases",
                                    "embedding_model", "judge_model")):
        raise ReportCorruptError("meta 字段不完整")
    gen = data["generation"]
    if (not isinstance(gen, dict) or "online_strategy" not in gen
            or not isinstance(gen.get("faithfulness_cases"), list)):
        raise ReportCorruptError("generation 字段不完整")
    ret = data["retrieval"]
    if not isinstance(ret, dict) or any(s not in ret for s in STRATEGIES):
        raise ReportCorruptError("retrieval 缺策略")


def load_report(path) -> dict | None:
    """不存在 → None;损坏 → ReportCorruptError;合法 → dict。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportCorruptError(f"报告解析失败: {type(exc).__name__}") from exc
    validate_report(data)
    return data


def normalize_citations(raw) -> dict:
    """历史兼容:裸列表 → {run_id: None, evidence: raw, cited_refs: []};None → 空。"""
    if isinstance(raw, dict):
        return {"run_id": raw.get("run_id"),
                "evidence": raw.get("evidence") or [],
                "cited_refs": raw.get("cited_refs") or []}
    if isinstance(raw, list):
        return {"run_id": None, "evidence": raw, "cited_refs": []}
    return {"run_id": None, "evidence": [], "cited_refs": []}


def _parse_reason(raw):
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


def _item(row: FaithCase) -> dict:
    return {"id": row.id, "case_id": row.eval_id, "bucket": row.bucket,
            "query": row.query, "strategy": row.strategy, "answer": row.answer,
            "unsupported_claims": _parse_reason(row.reason),
            "citations": normalize_citations(row.citations),
            "judge_model": row.judge_model, "status": row.status,
            "seen_count": row.seen_count,
            "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
            "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
            "resolution": row.resolution,
            "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None}


def list_faith_cases(sf, status: str, page: int, size: int) -> dict:
    with sf() as s:
        q = s.query(FaithCase).filter_by(status=status)
        total = q.count()
        rows = (q.order_by(FaithCase.last_seen_at.desc(), FaithCase.id.desc())
                 .offset((page - 1) * size).limit(size).all())
        return {"total": total, "page": page, "size": size,
                "items": [_item(r) for r in rows]}


def ledger_counts(sf) -> dict:
    with sf() as s:
        rows = s.query(FaithCase.status, func.count()).group_by(FaithCase.status).all()
    counts = {st: 0 for st in STATUSES}
    for status, n in rows:
        counts[status] = counts.get(status, 0) + n
    return {"total": sum(counts.values()), **counts}


def compute_stats(report: dict | None, sf) -> dict:
    """judge_rate = 上线 fabricated/judged(有 judge_error 或 judged=0 → null);
    confirmed_rate 分子只数本轮报告判出且台账快照 run_id 一致的题;
    有缺快照 → current_run_missing>0 且 confirmed_rate=null。"""
    stats = {"judge_rate": None, "confirmed_rate": None,
             "current_run_missing": 0, "ledger": ledger_counts(sf)}
    if report is None:
        return stats
    gen = report["generation"]
    online = gen[gen["online_strategy"]]
    if online.get("judge_errors") == 0 and online.get("judged", 0) > 0:
        stats["judge_rate"] = online["fabricated"] / online["judged"]
    case_ids = [c["case_id"] for c in gen["faithfulness_cases"]]
    if not case_ids:
        return stats
    run_id = report["meta"]["run_id"]
    with sf() as s:
        rows = s.query(FaithCase).filter(FaithCase.eval_id.in_(case_ids)).all()
    matched = {r.eval_id: r for r in rows
               if normalize_citations(r.citations)["run_id"] == run_id}
    stats["current_run_missing"] = sum(1 for cid in case_ids if cid not in matched)
    if stats["current_run_missing"] == 0:
        confirmed = sum(1 for r in matched.values() if r.status == "已解决")
        stats["confirmed_rate"] = confirmed / len(case_ids)
    return stats


def update_faith_case(sf, row_id: int, status: str, resolution: str) -> dict | None:
    """处置:status ∈ {已解决, 无需解决};resolution 去白后保存(规格:保存去白值);
    写朴素 UTC resolved_at。"""
    with sf() as s:
        row = s.get(FaithCase, row_id)
        if row is None:
            return None
        row.status = status
        row.resolution = resolution.strip()
        row.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        s.commit()
        return _item(row)
