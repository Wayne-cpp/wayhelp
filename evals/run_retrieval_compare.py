"""四策略检索对比 + 生成段 Faithfulness 评估(spec §8)。

用法:
  uv run python evals/run_retrieval_compare.py [--max-d-pass 0.10]
前置:EMBEDDING_API_KEY 必填;RERANK_API_KEY 空时回退 EMBEDDING_API_KEY;
主库须已执行 ch04 DDL(faith_cases 写主业务库);服务已停(Lite 独占)。
产物:evals/results/{UTC 时间戳}_compare.json + .md
"""
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db import check_ch04_tables, make_engine, make_session_factory
from app.knowledge.embedding import build_embeddings
from app.knowledge.evalset import BUCKETS, covered_groups, load_compare_cases
from app.knowledge.ingest import run_ingest
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.query_understanding import plan_query
from app.knowledge.reranker import SiliconFlowReranker
from app.knowledge.retriever import KnowledgeRetriever, assemble_evidence
from app.knowledge.scope import derive_scope
from app.models import KnowledgeChunk
from app.prompts.faithfulness import JUDGE_PROMPT
from app.prompts.service import REFUSAL_ANSWER, SERVICE_SYSTEM_PROMPT
from evals.run_knowledge_eval import _prepare_eval_db

ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).parent / "retrieval_compare.txt"
RESULTS_DIR = Path(__file__).parent / "results"
STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
DEFAULT_MAX_D_PASS = 0.10
MIN_B_RECALL = 0.5   # 硬门槛:B_model test 桶 bm25 SectionRecall@10 下限


def section_recall_at_k(hit_paths: list[str], groups, k: int) -> float:
    if not groups:
        return 0.0
    return covered_groups(hit_paths[:k], groups) / len(groups)


def complete_hit_at_k(hit_paths: list[str], groups, k: int) -> float:
    return 1.0 if groups and covered_groups(hit_paths[:k], groups) == len(groups) else 0.0


def mrr_at_10(hit_paths: list[str], groups) -> float:
    for rank, p in enumerate(hit_paths[:10], start=1):
        if covered_groups([p], groups):
            return 1.0 / rank
    return 0.0


def _score_threshold(pos, neg, t: float) -> tuple[float, float, float]:
    """阈值 t 下的 (D 误通过率, 正例误拒率, pass-adjusted SectionRecall@10)。"""
    d_pass = sum(1 for s in neg if s["top1"] is not None and s["top1"] >= t)
    d_rate = d_pass / len(neg) if neg else 0.0
    refused = sum(1 for s in pos if s["top1"] is None or s["top1"] < t)
    over_refusal = refused / len(pos) if pos else 0.0
    par = sum(s["recall10"] for s in pos
              if s["top1"] is not None and s["top1"] >= t) / len(pos) if pos else 0.0
    return d_rate, over_refusal, par


def _threshold_candidates(samples: list[dict]) -> tuple[list[float], list[dict], list[dict]]:
    candidates = sorted({s["top1"] for s in samples if s["top1"] is not None})
    pos = [s for s in samples if not s["should_refuse"]]
    neg = [s for s in samples if s["should_refuse"]]
    return candidates, pos, neg


def choose_strategy_threshold(samples: list[dict], max_d_pass: float) -> dict:
    """只用 calibration:candidates = 全部观测 Top-1 分;约束 D 误通过率 ≤ max_d_pass;
    约束内最大化 pass-adjusted SectionRecall@10(被拒 case recall 计 0);
    并列 → 误拒率低 → 阈值小。无可行 → SystemExit。"""
    candidates, pos, neg = _threshold_candidates(samples)
    best = None
    for t in candidates:
        d_rate, over_refusal, par = _score_threshold(pos, neg, t)
        if d_rate > max_d_pass:
            continue
        key = (par, -over_refusal, -t)
        if best is None or key > best[0]:
            best = (key, {"threshold": t, "d_pass_rate": d_rate,
                          "over_refusal_rate": over_refusal,
                          "pass_adjusted_recall": par})
    if best is None:
        raise SystemExit("[compare] 校准失败:不存在满足 D 桶误通过约束的阈值")
    return best[1]


def _pareto_rows(samples: list[dict], max_d_pass: float) -> list[dict]:
    """报告用:全部满足 D 约束的候选阈值及其三项指标(升序)。"""
    candidates, pos, neg = _threshold_candidates(samples)
    rows = []
    for t in candidates:
        d_rate, over_refusal, par = _score_threshold(pos, neg, t)
        if d_rate <= max_d_pass:
            rows.append({"threshold": t, "d_pass_rate": d_rate,
                         "over_refusal_rate": over_refusal,
                         "pass_adjusted_recall": par})
    return rows


_JUDGE_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_judge_output(text: str) -> dict:
    m = _JUDGE_JSON_RE.search(text or "")
    if not m:
        raise ValueError("judge output unparseable")
    data = json.loads(m.group(0))
    if data.get("verdict") not in ("faithful", "fabricated"):
        raise ValueError("judge verdict invalid")
    claims = data.get("unsupported_claims")
    refs = data.get("cited_refs")
    if not isinstance(claims, list) or not isinstance(refs, list):
        raise ValueError("judge fields invalid")
    return {"verdict": data["verdict"],
            "unsupported_claims": claims,
            "cited_refs": [int(x) for x in refs]}


def upsert_faith_case(s, *, eval_id, bucket, query, answer, reason, citations,
                      judge_model) -> None:
    """一题一行;重判更新快照 + seen_count+1;已解决复发退回未解决并清空 resolution。"""
    from app.models import FaithCase
    row = s.query(FaithCase).filter_by(eval_id=eval_id).first()
    now = datetime.now()
    if row is None:
        s.add(FaithCase(eval_id=eval_id, bucket=bucket, query=query,
                        strategy="hybrid_rerank", answer=answer, reason=reason,
                        citations=citations, judge_model=judge_model,
                        first_seen_at=now, last_seen_at=now))
        return
    row.answer = answer
    row.reason = reason
    row.citations = citations
    row.judge_model = judge_model
    row.seen_count += 1
    row.last_seen_at = now
    if row.status == "已解决":
        row.status = "未解决"
        row.resolution = None   # 处置清空;resolved_at 保留,用来标「复发过」


def _case_dict(c) -> dict:
    """EvalCase(frozen)→ 可携带 plan/retrieval 结果的可变 dict。"""
    return {"id": c.id, "bucket": c.bucket, "query": c.query, "gt_groups": c.gt_groups,
            "expect_points": c.expect_points, "should_refuse": c.should_refuse,
            "split": c.split}


def _test_metrics(test_cases: list[dict], strat: str, threshold: float) -> dict:
    """test 分片指标:recall 族宏平均只计 A/B/C/E 正例(被拒计 0);D 只进拒答正确率。"""
    per = []
    for c in test_cases:
        r = c["retrieval"][strat]
        passed = r["top1"] is not None and r["top1"] >= threshold
        paths = r["paths"]
        per.append({
            "bucket": c["bucket"], "should_refuse": c["should_refuse"], "passed": passed,
            "recall5": section_recall_at_k(paths, c["gt_groups"], 5) if passed else 0.0,
            "recall10": section_recall_at_k(paths, c["gt_groups"], 10) if passed else 0.0,
            "ch5": complete_hit_at_k(paths, c["gt_groups"], 5) if passed else 0.0,
            "ch10": complete_hit_at_k(paths, c["gt_groups"], 10) if passed else 0.0,
            "mrr10": mrr_at_10(paths, c["gt_groups"]) if passed else 0.0,
        })
    pos = [p for p in per if not p["should_refuse"]]
    d = [p for p in per if p["should_refuse"]]

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    by_bucket = {}
    for b in BUCKETS:
        bp = [p for p in pos if p["bucket"] == b]
        if bp:
            by_bucket[b] = {"section_recall_at_10": avg([p["recall10"] for p in bp]),
                            "refused": sum(1 for p in bp if not p["passed"])}
    return {
        "threshold": threshold,
        "section_recall_at_5": avg([p["recall5"] for p in pos]),
        "section_recall_at_10": avg([p["recall10"] for p in pos]),
        "complete_hit_at_5": avg([p["ch5"] for p in pos]),
        "complete_hit_at_10": avg([p["ch10"] for p in pos]),
        "mrr_at_10": avg([p["mrr10"] for p in pos]),
        "d_refuse_correct_rate": avg([1.0 if not p["passed"] else 0.0 for p in d]),
        "over_refusal_rate": avg([1.0 if not p["passed"] else 0.0 for p in pos]),
        "by_bucket": by_bucket,
    }


def _simulate_second_turn(model, query: str, evidence: list[dict]) -> str:
    """模拟第二轮消息:System(SERVICE) + Human(query) + AIMessage(query_faq tool_call)
    + ToolMessage(query_faq 出参 JSON),同步 invoke 生成答案。"""
    payload = {"effective_strategy": "hybrid_rerank", "low_confidence": False,
               "note": None, "evidence": evidence}
    call_id = "call_compare"
    messages = [
        SystemMessage(content=SERVICE_SYSTEM_PROMPT),
        HumanMessage(content=query),
        AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": query},
                                           "id": call_id, "type": "tool_call"}]),
        ToolMessage(content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=call_id, name="query_faq"),
    ]
    resp = model.invoke(messages)
    return resp.content if isinstance(resp.content, str) else ""


def _judge(model, query: str, answer: str, evidence: list[dict]) -> tuple[dict | None, str | None]:
    """忠实度裁判;失败(模型异常/解析非法)重试一次,再失败返回 (None, 错误描述)。"""
    prompt = (JUDGE_PROMPT.replace("{query}", query).replace("{answer}", answer)
              .replace("{evidence}", json.dumps(evidence, ensure_ascii=False)))
    err = None
    for _ in range(2):
        try:
            resp = model.invoke([("user", prompt)])
            return (parse_judge_output(resp.content
                                       if isinstance(resp.content, str) else ""), None)
        except Exception as exc:  # 模型异常与 ValueError 同样计入重试
            err = f"{type(exc).__name__}: {exc}"
    return None, err


def _generation_segment(test_cases: list[dict], threshold: float, settings, model,
                        main_sf) -> tuple[list[dict], int]:
    """生成段(test 分片,hybrid_rerank):低置信不调模型直接拒答话术;非拒答的
    A/B/C/E case 交 judge;编造落 faith_cases 主库。返回 (逐 case 行, judge_error 数)。"""
    rows, judge_errors = [], 0
    for c in test_cases:
        r = c["retrieval"]["hybrid_rerank"]
        low = r["top1"] is None or r["top1"] < threshold
        if low:
            answer, evidence = REFUSAL_ANSWER, []
        else:
            evidence = [e.to_dict() for e in assemble_evidence(
                r["hits"], max_items=settings.rerank_top_n,
                budget_chars=settings.max_tool_result_chars, overhead_chars=200)]
            answer = _simulate_second_turn(model, c["query"], evidence)
        refused = answer.strip() == REFUSAL_ANSWER
        row = {"id": c["id"], "bucket": c["bucket"], "low_confidence": low,
               "refused": refused, "answer": answer, "judge": None}
        if not refused and not c["should_refuse"]:
            verdict, err = _judge(model, c["query"], answer, evidence)
            if verdict is None:
                row["judge"] = "judge_error"
                row["judge_error"] = err
                judge_errors += 1
            else:
                row["judge"] = verdict["verdict"]
                row["unsupported_claims"] = verdict["unsupported_claims"]
                row["cited_refs"] = verdict["cited_refs"]
                if verdict["verdict"] == "fabricated":
                    with main_sf() as s:
                        upsert_faith_case(s, eval_id=c["id"], bucket=c["bucket"],
                                          query=c["query"], answer=answer,
                                          reason=json.dumps(verdict["unsupported_claims"],
                                                            ensure_ascii=False),
                                          citations=evidence,
                                          judge_model=settings.model_name)
                        s.commit()
        rows.append(row)
    return rows, judge_errors


def _render_md(out: dict) -> str:
    lines = ["# 四策略检索对比 + 生成段 Faithfulness 评估(ch04 T12)", "",
             f"- 时间:{out['ts']};max_d_pass={out['max_d_pass']};"
             f"语料:{out['corpus']};cases={out['cases_file']}", "",
             "## 校准冻结阈值(calibration 分片 = 每桶奇数编号)", "",
             "| strategy | threshold | D误通过率 | 误拒率 | pass-adj SR@10 |",
             "|---|---|---|---|---|"]
    for strat in STRATEGIES:
        th = out["thresholds"][strat]
        lines.append(f"| {strat} | {th['threshold']:.4f} | {th['d_pass_rate']:.3f} | "
                     f"{th['over_refusal_rate']:.3f} | {th['pass_adjusted_recall']:.3f} |")
    lines += ["", "## test 分片指标(偶数编号;被拒正例计 0,recall 族只宏平均 A/B/C/E)", "",
              "| strategy | SR@5 | SR@10 | CH@5 | CH@10 | MRR@10 | D拒答正确率 | 误拒率 |",
              "|---|---|---|---|---|---|---|---|"]
    for strat in STRATEGIES:
        m = out["test_metrics"][strat]
        lines.append(f"| {strat} | {m['section_recall_at_5']:.3f} | "
                     f"{m['section_recall_at_10']:.3f} | {m['complete_hit_at_5']:.3f} | "
                     f"{m['complete_hit_at_10']:.3f} | {m['mrr_at_10']:.3f} | "
                     f"{m['d_refuse_correct_rate']:.3f} | {m['over_refusal_rate']:.3f} |")
    gen = out["generation"]
    lines += ["", "## 生成段(test 分片,hybrid_rerank)", "",
              f"- 拒答 {gen['refused']} / faithful {gen['faithful']} / "
              f"fabricated {gen['fabricated']} / judge_error {gen['judge_errors']}",
              "", "### 编造 / 裁判失败 case", ""]
    flagged = [r for r in gen["rows"]
               if r["judge"] in ("fabricated", "judge_error")]
    if flagged:
        lines += [f"- {r['id']}({r['bucket']}):{r['judge']}" for r in flagged]
    else:
        lines.append("- (无)")
    gates = out["gates"]
    lines += ["", "## 硬门槛", "",
              f"- judge_error == 0:{'通过' if gates['judge_error_zero'] else '未过'}",
              f"- bm25 B_model test SR@10 ≥ {MIN_B_RECALL}:"
              f"{gates['bm25_b_model_section_recall_at_10']:.3f}"
              f"({'通过' if gates['bm25_b_model_gate'] else '未过'})",
              "", f"**结论:{'通过' if gates['passed'] else '未通过'}**", "",
              "> 阈值只从 calibration 冻结;test 指标按冻结阈值离线施加;"
              "> 全量 300 case 逐条 top1/pass/recall 见同名 .json 的 diagnosis。", ""]
    return "\n".join(lines)


def _parse_max_d_pass(argv: list[str]) -> float:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        raise SystemExit(0)
    if "--max-d-pass" not in argv:
        return DEFAULT_MAX_D_PASS
    i = argv.index("--max-d-pass")
    try:
        value = float(argv[i + 1])
    except (IndexError, ValueError):
        raise SystemExit("用法: run_retrieval_compare.py [--max-d-pass 0.10]")
    if not 0.0 <= value <= 1.0:
        raise SystemExit("--max-d-pass 须在 [0,1]")
    return value


def main(argv: list[str]) -> int:
    max_d_pass = _parse_max_d_pass(argv)
    settings = Settings()
    if not settings.has_embedding_key():
        print("[compare] embedding_api_key 未配置", file=sys.stderr)
        return 2
    if not settings.has_rerank_key():
        print("[compare] rerank/embedding key 均未配置", file=sys.stderr)
        return 2
    main_engine = make_engine(settings.database_url)
    try:  # faith_cases 写主业务库:先只读校验 ch04 表,再做任何远程调用
        check_ch04_tables(main_engine)
    except RuntimeError as exc:
        print(f"[compare] {exc}", file=sys.stderr)
        return 2
    main_sf = make_session_factory(main_engine)
    model = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key,
                       base_url=settings.openai_base_url,
                       max_tokens=settings.max_output_tokens)
    embed = build_embeddings(settings)
    engine = _prepare_eval_db(settings)  # 复用 ch03 版:独立评估库重放 03-ddl
    store = MilvusKnowledgeStore(str(Path(tempfile.mkdtemp()) / "compare_milvus.db"),
                                 settings.embedding_dim)
    try:
        store.ensure_collection()
        eval_sf = make_session_factory(engine)
        if run_ingest(settings, eval_sf, embed, store, ROOT / "knowledge_docs") != 0:
            print("[compare] 语料建库失败", file=sys.stderr)
            return 1
        with eval_sf() as s:  # 读回 {chunk_id: (section_path, answer, scope)}
            chunk_meta = {r.id: (r.section_path or "", r.answer,
                                 derive_scope(r.content_type, r.source_doc))
                          for r in s.query(KnowledgeChunk).all()}
        corpus_paths = [m[0] for m in chunk_meta.values()]
        cases = [_case_dict(c) for c in load_compare_cases(CASES)]
        for c in cases:  # 覆盖断言:每个正例 AND 组至少一个别名命中语料 section_path
            for g in c["gt_groups"]:
                assert covered_groups(corpus_paths, (g,)) >= 1, \
                    f"{c['id']} GT 组未命中语料: {g}"
        for c in cases:  # QueryPlan 冻结:每 case 改写一次,四策略共用
            c["plan"] = plan_query(model, c["query"], enabled=settings.query_rewrite_enabled,
                                   timeout_seconds=settings.rerank_timeout_seconds,
                                   model_name=settings.model_name)
        retriever = KnowledgeRetriever(
            settings, embed=embed, store=store, session_factory=eval_sf, model=model,
            reranker=SiliconFlowReranker(settings))
        for c in cases:  # 检索段:四策略 × 全量 300,min_score=-1 阈值离线施加
            c["retrieval"] = {}
            for strat in STRATEGIES:
                res = retriever.search(c["query"], strategy=strat, min_score=-1.0,
                                       query_plan=c["plan"])
                c["retrieval"][strat] = {
                    "hits": res.hits,
                    "paths": [h.section_path or "" for h in res.hits],
                    "top1": res.confidence_score,
                    "effective_strategy": res.effective_strategy, "note": res.note,
                }
        calibration = [c for c in cases if c["split"] == "calibration"]
        thresholds, pareto = {}, {}
        for strat in STRATEGIES:
            samples = [{"bucket": c["bucket"], "should_refuse": c["should_refuse"],
                        "top1": c["retrieval"][strat]["top1"],
                        "recall10": section_recall_at_k(c["retrieval"][strat]["paths"],
                                                        c["gt_groups"], 10)}
                       for c in calibration]
            thresholds[strat] = choose_strategy_threshold(samples, max_d_pass)
            pareto[strat] = _pareto_rows(samples, max_d_pass)
            th = thresholds[strat]
            print(f"[compare] {strat}: threshold={th['threshold']:.4f} "
                  f"d_pass={th['d_pass_rate']:.3f} "
                  f"over_refusal={th['over_refusal_rate']:.3f} "
                  f"par={th['pass_adjusted_recall']:.3f} "
                  f"(可行候选 {len(pareto[strat])} 个)")
        test_cases = [c for c in cases if c["split"] == "test"]
        metrics = {strat: _test_metrics(test_cases, strat, thresholds[strat]["threshold"])
                   for strat in STRATEGIES}
        gen_rows, judge_errors = _generation_segment(
            test_cases, thresholds["hybrid_rerank"]["threshold"], settings, model, main_sf)
        bm25_b = metrics["bm25"]["by_bucket"]["B_model"]["section_recall_at_10"]
        gates = {"judge_error_zero": judge_errors == 0,
                 "bm25_b_model_section_recall_at_10": bm25_b,
                 "bm25_b_model_gate": bm25_b >= MIN_B_RECALL}
        gates["passed"] = gates["judge_error_zero"] and gates["bm25_b_model_gate"]
        diagnosis = []  # 全量 300(含 calibration)逐条:top1 / 冻结阈值 pass / recall10
        for c in cases:
            entry = {"id": c["id"], "bucket": c["bucket"], "split": c["split"],
                     "should_refuse": c["should_refuse"], "strategies": {}}
            for strat in STRATEGIES:
                r = c["retrieval"][strat]
                t = thresholds[strat]["threshold"]
                entry["strategies"][strat] = {
                    "top1": r["top1"],
                    "passed": r["top1"] is not None and r["top1"] >= t,
                    "recall10": section_recall_at_k(r["paths"], c["gt_groups"], 10)}
            diagnosis.append(entry)
        out = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "max_d_pass": max_d_pass, "model": settings.model_name,
            "judge_model": settings.model_name,
            "embedding_model": settings.embedding_model,
            "corpus_chunks": len(chunk_meta), "cases_file": CASES.name,
            "corpus": "knowledge_docs/",
            "thresholds": thresholds, "pareto": pareto,
            "test_metrics": metrics,
            "generation": {
                "strategy": "hybrid_rerank",
                "refused": sum(1 for r in gen_rows if r["refused"]),
                "faithful": sum(1 for r in gen_rows if r["judge"] == "faithful"),
                "fabricated": sum(1 for r in gen_rows if r["judge"] == "fabricated"),
                "judge_errors": judge_errors, "rows": gen_rows,
            },
            "diagnosis": diagnosis, "gates": gates,
        }
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = RESULTS_DIR / f"{stamp}_compare.json"
        md_path = RESULTS_DIR / f"{stamp}_compare.md"
        json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        md_path.write_text(_render_md(out), encoding="utf-8")
        print(f"[compare] 报告: {json_path}")
        print(f"[compare] 门槛: judge_error={judge_errors} "
              f"bm25_B_SR@10={bm25_b:.3f}(>= {MIN_B_RECALL}) "
              f"-> {'通过' if gates['passed'] else '未通过'}")
        return 0 if gates["passed"] else 1
    finally:
        store.close()
        engine.dispose()
        main_engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
