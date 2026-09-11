"""知识检索评估(spec §11):真实硅基流动 embedding + 真实 Milvus Lite + 独立 MySQL 评估库。

用法:
  uv run python evals/run_knowledge_eval.py --dump-corpus   # 建评估语料并打印可标注块
  uv run python evals/run_knowledge_eval.py                 # 完整评估:校准冻结阈值 → test 验收
达标线:test 分片 Recall@5 宏平均 ≥ 0.90 且负例误召回率 ≤ 0.10,且 p_youfei 必须命中。
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.knowledge.embedding import build_embeddings
from app.knowledge.ingest import run_ingest
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.retriever import KnowledgeRetriever
from app.models import KnowledgeChunk
from tests.dbfixtures import _split_statements

ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).parent / "knowledge_recall.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
EVAL_DB = "wayhelp_eval"
YOUFEI_CASE = "p_youfei"
YOUFEI_POINTS = ["满99元包邮", "8元"]  # 空白规范化后包含判定
MIN_RECALL = 0.90
MAX_FPR = 0.10
TOP_K = 5


def case_recall(returned: set, relevant: set) -> float:
    if not relevant:
        return 0.0
    return len(returned & relevant) / len(relevant)


def macro_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def false_recall_rate(returned_nonempty: list[bool]) -> float:
    return sum(1 for b in returned_nonempty if b) / len(returned_nonempty) if returned_nonempty else 0.0


def choose_threshold(cases: list[dict], max_fpr: float) -> tuple[float, float]:
    """只用 calibration 分片:先满足 FPR ≤ max_fpr,再最大化宏平均召回,并列取低阈值
    (对负例留最大间隔;2026-09-11 勘误:原「取高」实测压在正例分数悬崖边上,泛化差)。
    返回 (threshold, calibration_recall);无可行阈值直接 SystemExit。
    候选按升序遍历,仅在召回严格更大时更新,实现并列取低。"""
    candidates = {-1.0, 1.0}
    for c in cases:
        candidates.update(score for _, score in c["hits"])
    best_t, best_recall = None, -1.0
    for t in sorted(candidates):
        recalls, negs = [], []
        for c in cases:
            returned = {k for k, s in c["hits"] if s >= t}
            if c["answerable"]:
                recalls.append(case_recall(returned, c["relevant"]))
            else:
                negs.append(bool(returned))
        if false_recall_rate(negs) > max_fpr:
            continue
        recall = macro_average(recalls)
        if recall > best_recall:
            best_t, best_recall = t, recall
    if best_t is None:
        raise SystemExit("[eval] 校准失败: 不存在满足负例约束的阈值")
    return best_t, best_recall


def _prepare_eval_db(settings: Settings):
    admin = make_engine(settings.test_admin_database_url)
    with admin.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {EVAL_DB} CHARACTER SET utf8mb4"))
        conn.commit()
    # 脆点:依赖默认测试库名 wayhelp_test 做前缀替换,用户改过 TEST_DATABASE_URL 则直接报错
    if "/wayhelp_test" not in settings.test_database_url:
        raise SystemExit("[eval] test_database_url 不含 /wayhelp_test,无法派生评估库名")
    url = settings.test_database_url.replace("/wayhelp_test", f"/{EVAL_DB}")
    engine = make_engine(url)
    ddl = (ROOT / "db" / "init" / "03-ddl.sql").read_text(encoding="utf-8")
    with engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in ("knowledge_chunks", "qa_extraction_staging", "qa_mining_progress"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        for stmt in _split_statements(ddl):
            conn.execute(text(stmt))
        conn.commit()
    return engine


def _build_corpus(settings: Settings, engine, store, embed) -> dict[int, tuple[str, int]]:
    """独立评估库 + 临时 Milvus,只含固定 knowledge_docs 语料(不混入挖掘 QA)。"""
    sf = make_session_factory(engine)
    if run_ingest(settings, sf, embed, store, ROOT / "knowledge_docs") != 0:
        raise SystemExit("[eval] 语料建库失败")
    with sf() as s:
        rows = s.query(KnowledgeChunk).all()
        return {r.id: (r.source_doc, r.chunk_index) for r in rows}


def _load_cases(corpus_keys: set[tuple[str, int]]) -> list[dict]:
    cases = [json.loads(ln) for ln in CASES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case id 重复"
    pos = [c for c in cases if c["answerable"]]
    neg = [c for c in cases if not c["answerable"]]
    assert len(pos) >= 20 and len(neg) >= 20, "正负例各需 ≥ 20"
    for split in ("calibration", "test"):
        assert sum(1 for c in pos if c["split"] == split) >= 10
        assert sum(1 for c in neg if c["split"] == split) >= 10
    for c in cases:
        assert c["split"] in ("calibration", "test")
        if not c["answerable"]:
            assert not c["relevant_chunks"], "负例相关块必须为空"
        for ref in c["relevant_chunks"]:
            key = (ref["source_doc"], ref["chunk_index"])
            assert key in corpus_keys, f"标注引用不存在的块: {key}"
    youfei = [c for c in cases if c["id"] == YOUFEI_CASE]
    assert len(youfei) == 1 and youfei[0]["split"] == "test" and youfei[0]["answerable"]
    for c in cases:  # 派生可比对的关键集合,hits 里的 key 也是 (source_doc, chunk_index)
        c["relevant"] = frozenset((r["source_doc"], r["chunk_index"])
                                  for r in c["relevant_chunks"])
    return cases


def main(argv: list[str]) -> int:
    settings = Settings()
    if not settings.has_embedding_key():
        print("[eval] embedding_api_key 未配置", file=sys.stderr)
        return 2
    embed = build_embeddings(settings)
    engine = _prepare_eval_db(settings)
    tmp = Path(tempfile.mkdtemp()) / "eval_milvus.db"
    store = MilvusKnowledgeStore(str(tmp), settings.embedding_dim)
    try:
        id_to_key = _build_corpus(settings, engine, store, embed)
        key_to_answer = {}
        sf = make_session_factory(engine)
        with sf() as s:
            for r in s.query(KnowledgeChunk).all():
                key_to_answer[(r.source_doc, r.chunk_index)] = r.answer
        if "--dump-corpus" in argv:
            RESULTS_DIR.mkdir(exist_ok=True)
            snapshot = [{"chunk_id": i, "source_doc": k[0], "chunk_index": k[1]}
                        for i, k in sorted(id_to_key.items())]
            (RESULTS_DIR / "corpus_snapshot.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            for row in snapshot:
                print(row)
            return 0
        cases = _load_cases(set(id_to_key.values()))
        retriever = KnowledgeRetriever(settings, embed=embed, store=store,
                                       session_factory=sf)
        for c in cases:  # 同一次检索链路,min_score=-1 拿原始分,阈值离线施加
            hits, _ = retriever.search(c["query"], min_score=-1.0)
            c["hits"] = [(id_to_key[h.chunk_id], h.score) for h in hits]
        calibration = [c for c in cases if c["split"] == "calibration"]
        threshold, cal_recall = choose_threshold(calibration, MAX_FPR)
        if cal_recall < MIN_RECALL:
            print(f"[eval] 校准失败: calibration 召回 {cal_recall:.3f} < {MIN_RECALL}",
                  file=sys.stderr)
            return 1
        test_cases = [c for c in cases if c["split"] == "test"]
        recalls, negs, per_case = [], [], []
        for c in test_cases:
            returned = {k for k, s in c["hits"] if s >= threshold}
            if c["answerable"]:
                r = case_recall(returned, c["relevant"])
                recalls.append(r)
            else:
                negs.append(bool(returned))
            per_case.append({"id": c["id"], "answerable": c["answerable"],
                             "returned": sorted(str(k) for k in returned),
                             "relevant": sorted(str((x["source_doc"], x["chunk_index"]))
                                                for x in c["relevant_chunks"])})
        recall = macro_average(recalls)
        fpr = false_recall_rate(negs)
        youfei = next(c for c in test_cases if c["id"] == YOUFEI_CASE)
        youfei_hit_keys = {k for k, s in youfei["hits"] if s >= threshold}
        youfei_ok = any(
            all(pt in "".join(key_to_answer[k].split()) for pt in YOUFEI_POINTS)
            for k in youfei_hit_keys if k in key_to_answer)
        ok = recall >= MIN_RECALL and fpr <= MAX_FPR and youfei_ok
        RESULTS_DIR.mkdir(exist_ok=True)
        out = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": settings.embedding_model, "threshold": threshold, "top_k": TOP_K,
            "test_recall_at_5": recall, "test_false_recall_rate": fpr,
            "youfei_hit": youfei_ok, "cases": per_case,
            "cases_file": CASES.name, "corpus": "knowledge_docs/",
        }
        path = RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] threshold={threshold:.3f} recall@5={recall:.3f} "
              f"fpr={fpr:.3f} youfei={youfei_ok} -> {path}")
        if not ok:
            print("[eval] 未达标", file=sys.stderr)
            return 1
        return 0
    finally:
        store.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
