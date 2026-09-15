"""语料-评估集对齐校验(spec §11):离线切块,断言每个正例 AND 组至少一个别名命中
section_path,且每个 expect_point 有原文支撑;输出 case → GT 组 → 命中块映射。

用法: uv run python evals/validate_corpus.py   # 全部通过 exit 0,否则 exit 1 并逐条打印
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.knowledge.chunking import chunk_document
from app.knowledge.evalset import covered_groups, load_compare_cases
from app.knowledge.ingest import resolve_source_doc

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    # 离线校验:不用真实 key/.env,占位必填项,只取切块参数(conftest.make_settings 同法)
    settings = Settings(
        _env_file=None,
        openai_base_url="http://offline/v1", openai_api_key="offline",
        model_name="offline", database_url="mysql+pymysql://u:p@127.0.0.1:9/offline",
    )
    chunks = []
    for p in sorted((ROOT / "knowledge_docs").glob("*.md")):
        chunks += chunk_document(p.read_text(encoding="utf-8"), source=resolve_source_doc(p),
                                 max_chars=settings.max_chunk_chars,
                                 overlap_chars=settings.chunk_overlap_chars)
    paths = [c.section_path or "" for c in chunks]
    corpus_text = "".join("".join((c.questions + c.answer).split()) for c in chunks)
    fails = 0
    for case in load_compare_cases(ROOT / "evals" / "retrieval_compare.txt"):
        if case.should_refuse:
            continue
        uncovered = [g for g in case.gt_groups
                     if covered_groups(paths, (g,)) == 0]
        missing_points = [pt for pt in case.expect_points
                          if "".join(pt.split()) not in corpus_text]
        if uncovered or missing_points:
            fails += 1
            print(f"[FAIL] {case.id} 未覆盖组={uncovered} 缺要点={missing_points}")
    print(f"[validate] {len(chunks)} 块;失败 {fails} 条")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
