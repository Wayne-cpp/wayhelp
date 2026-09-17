"""一次性诊断:E_multi 失分题(E6/E30/E56)的四级管道逐段可视。

对每题打印:QueryPlan(改写+同义词+子意图)→ dense/bm25/融合 三段中各 GT 组的
最佳名次 → 重排 top15(旧单榜口径,对照)→ 生产路径 top-N(含多意图支路)。
用法: uv run python evals/probe_emulti.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db import make_session_factory
from app.knowledge.embedding import build_embeddings
from app.knowledge.evalset import covered_groups, load_compare_cases
from app.knowledge.ingest import run_ingest, vector_text
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.query_understanding import plan_query
from app.knowledge.reranker import SiliconFlowReranker
from app.knowledge.retriever import KnowledgeRetriever, _hit_text
from app.models import KnowledgeChunk
from evals.run_knowledge_eval import _prepare_eval_db

ROOT = Path(__file__).resolve().parent.parent
PROBE_IDS = {"E2", "E6", "E24", "E26", "E30", "E40", "E56"}


def main() -> int:
    settings = Settings()
    engine = _prepare_eval_db(settings)
    store = MilvusKnowledgeStore(str(Path(tempfile.mkdtemp()) / "probe_milvus.db"),
                                 settings.embedding_dim)
    try:
        store.ensure_collection()
        sf = make_session_factory(engine)
        embed = build_embeddings(settings)
        if run_ingest(settings, sf, embed, store, ROOT / "knowledge_docs") != 0:
            return 1
        with sf() as s:
            chunks = {r.id: r for r in s.query(KnowledgeChunk).all()}
        model = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key,
                           base_url=settings.openai_base_url,
                           max_tokens=settings.max_output_tokens,
                           extra_body={"thinking": {"type": "disabled"}})
        reranker = SiliconFlowReranker(settings)
        retriever = KnowledgeRetriever(settings, embed=embed, store=store,
                                       session_factory=sf, model=model,
                                       reranker=reranker)
        cases = [c for c in load_compare_cases(ROOT / "evals" / "retrieval_compare.txt")
                 if c.id in PROBE_IDS]
        for c in sorted(cases, key=lambda x: x.id):
            print("=" * 78)
            print(f"{c.id}  {c.query}")
            print(f"GT 组: {[list(g) for g in c.gt_groups]}")
            plan = plan_query(model, c.query, enabled=settings.query_rewrite_enabled,
                              timeout_seconds=settings.rerank_timeout_seconds,
                              model_name=settings.model_name)
            print(f"改写: {plan.standard_query}")
            print(f"同义词: {list(plan.synonyms)}")
            print("子意图:", list(plan.sub_queries) or "(无)")
            vector = embed.embed_query(plan.standard_query)
            bm25_text = " ".join([plan.standard_query, *plan.synonyms]).strip()
            legs = {
                "dense": store.search_dense(vector, 50, None),
                "bm25": store.search_bm25(bm25_text, 50, None),
                "fused": store.hybrid(vector, bm25_text, 50, None),
            }
            for name, hits in legs.items():
                paths = [chunks[i].section_path or "" for i, _ in hits if i in chunks]
                marks = []
                for gi, g in enumerate(c.gt_groups):
                    rank = next((r + 1 for r, p in enumerate(paths)
                                 if covered_groups([p], (g,))), None)
                    marks.append(f"组{gi + 1}={'#' + str(rank) if rank else 'MISS'}")
                print(f"  {name:6s} top50: {'  '.join(marks)}")
            fused = legs["fused"]
            ids = [i for i, _ in fused if i in chunks]
            docs = [_hit_text(chunks[i]) for i in ids]
            outcome = reranker.rerank(plan.standard_query, docs, 15)
            print("  重排 top15(单榜对照):")
            for rank, (i, score) in enumerate(outcome.ranking, 1):
                path = chunks[ids[i]].section_path or ""
                hit = "".join(f" [GT组{gi + 1}]" for gi, g in enumerate(c.gt_groups)
                              if covered_groups([path], (g,)))
                print(f"    #{rank:2d} {score:.4f}  {path}{hit}")
            res = retriever.search(c.query, strategy="hybrid_rerank",
                                   min_score=-1.0, query_plan=plan)
            print(f"  生产路径 top{len(res.hits)}"
                  f"(sub_queries={res.leg_counts.get('sub_queries', 0)},"
                  f" confidence={res.confidence_score}):")
            for rank, h in enumerate(res.hits, 1):
                hit = "".join(f" [GT组{gi + 1}]" for gi, g in enumerate(c.gt_groups)
                              if covered_groups([h.section_path or ""], (g,)))
                print(f"    #{rank:2d} {h.score:.4f}  {h.section_path}{hit}")
    finally:
        store.close()
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
