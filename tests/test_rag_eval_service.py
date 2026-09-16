"""rag_eval 数据服务:报告加载/契约校验、citations 兼容、台账分页与统计、处置。"""

import json
from datetime import datetime

import pytest

from app.models import FaithCase
from app.services import rag_eval as svc
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _report(run_id="r1", fabricated=1, judged=4, judge_errors=0, case_ids=("A2",)):
    return {
        "meta": {"run_id": run_id, "ts": "t", "total_cases": 300,
                 "calibration_cases": 150, "test_cases": 150,
                 "answerable_test_cases": 120, "d_test_cases": 30,
                 "corpus_chunks": 50, "embedding_model": "e", "rerank_model": "r",
                 "eval_model": "m", "judge_model": "m"},
        "retrieval": {s: {"threshold": 0.5, "ungated": False,
                          "overall": {"mrr": 0.1, "recall5": 0.1,
                                      "evidence_coverage": 0.1, "sr10": 0.1},
                          "by_bucket": {}}
                      for s in ("dense", "bm25", "hybrid", "hybrid_rerank")},
        "generation": {"online_strategy": "hybrid_rerank",
                       "hybrid_rerank": {"fabricated": fabricated, "judged": judged,
                                          "judge_errors": judge_errors},
                       "faithfulness_cases": [{"case_id": c} for c in case_ids]},
        "gates": {"passed": True},
    }


def _seed(s, case_id, run_id="r1", status="未解决", last=None):
    row = FaithCase(eval_id=case_id, bucket="A_policy", query="q", answer="a",
                    reason='[{"claim": "c", "reason": "r"}]',
                    citations={"run_id": run_id, "evidence": [{"ref_no": 1}],
                               "cited_refs": [1]},
                    judge_model="m", status=status)
    if last:
        row.last_seen_at = last
    s.add(row)
    return row


def test_normalize_citations_legacy():
    assert svc.normalize_citations(None) == {"run_id": None, "evidence": [],
                                             "cited_refs": []}
    out = svc.normalize_citations([{"ref_no": 1}])
    assert out == {"run_id": None, "evidence": [{"ref_no": 1}], "cited_refs": []}
    out = svc.normalize_citations({"run_id": "r1", "evidence": [], "cited_refs": [2]})
    assert out["run_id"] == "r1" and out["cited_refs"] == [2]


def test_load_report_missing_corrupt_ok(tmp_path):
    assert svc.load_report(tmp_path / "none.json") is None
    bad = tmp_path / "rag_eval.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(svc.ReportCorruptError):
        svc.load_report(bad)
    bad.write_text('{"meta": {}}', encoding="utf-8")       # 契约不合法
    with pytest.raises(svc.ReportCorruptError):
        svc.load_report(bad)
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    assert svc.load_report(good)["meta"]["run_id"] == "r1"


def test_list_pagination_and_order(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        same = datetime(2026, 9, 16, 10, 0, 0)
        for cid in ("A1", "A2", "A3"):
            _seed(s, cid, last=same)                       # 同刻 → id DESC 定序
        _seed(s, "B1", status="已解决")
        s.commit()
    out = svc.list_faith_cases(sf, "未解决", page=1, size=2)
    assert out["total"] == 3 and len(out["items"]) == 2
    assert [i["case_id"] for i in out["items"]] == ["A3", "A2"]
    page2 = svc.list_faith_cases(sf, "未解决", page=2, size=2)
    assert [i["case_id"] for i in page2["items"]] == ["A1"]
    item = out["items"][0]
    assert isinstance(item["id"], int) and item["case_id"] == "A3"
    assert item["unsupported_claims"] == [{"claim": "c", "reason": "r"}]
    assert item["citations"]["run_id"] == "r1"
    assert svc.list_faith_cases(sf, "已解决", 1, 10)["total"] == 1


def test_compute_stats(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2", status="已解决")                    # 本轮判出且已确认
        _seed(s, "C2", run_id="r-old")                     # 旧轮快照,不算本轮
        s.commit()
    rep = _report(run_id="r1", fabricated=2, judged=4, case_ids=("A2", "B2"))
    stats = svc.compute_stats(rep, sf)
    assert stats["judge_rate"] == 0.5                      # fabricated / judged
    assert stats["current_run_missing"] == 1               # B2 无本轮台账快照
    assert stats["confirmed_rate"] is None                 # 缺失 → 不输出残缺口径
    assert stats["ledger"]["total"] == 2 and stats["ledger"]["已解决"] == 1

    with sf() as s:                                        # 补齐 B2 本轮快照(未解决)
        _seed(s, "B2", run_id="r1")
        s.commit()
    stats = svc.compute_stats(rep, sf)
    assert stats["current_run_missing"] == 0
    assert stats["confirmed_rate"] == 0.5                  # 2 题中 1 题已解决

    rep_err = _report(run_id="r1", judge_errors=1)
    assert svc.compute_stats(rep_err, sf)["judge_rate"] is None

    stats = svc.compute_stats(None, sf)                    # 无报告:两口径 null,台账照给
    assert stats["judge_rate"] is None and stats["confirmed_rate"] is None
    assert stats["current_run_missing"] == 0 and stats["ledger"]["total"] == 3


def test_update_faith_case(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2")
        s.commit()
        rid = s.query(FaithCase).one().id
    item = svc.update_faith_case(sf, rid, "已解决", "  已补售后文档  ")
    assert item["status"] == "已解决" and item["resolution"] == "已补售后文档"
    assert item["resolved_at"] is not None
    with sf() as s:
        row = s.query(FaithCase).one()
        assert row.resolved_at.tzinfo is None              # 朴素 UTC(与既有列一致)
    assert svc.update_faith_case(sf, 999999, "已解决", "x") is None
