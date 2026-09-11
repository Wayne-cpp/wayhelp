from evals.run_knowledge_eval import (
    case_recall, choose_threshold, false_recall_rate, macro_average,
)


def test_case_recall():
    assert case_recall({("d", 1), ("d", 2)}, {("d", 1)}) == 1.0
    assert case_recall(set(), {("d", 1)}) == 0.0
    assert case_recall({("d", 1)}, {("d", 1), ("d", 2)}) == 0.5


def test_macro_average():
    assert macro_average([1.0, 0.5]) == 0.75


def test_false_recall_rate():
    assert false_recall_rate([True, False, False, False]) == 0.25
    assert false_recall_rate([]) == 0.0


def test_choose_threshold_prefers_feasible_then_recall_then_higher():
    # 负例分 0.8/0.6,正例相关块分 0.7:满足 FPR 约束的阈值只剩 > 0.8,召回为 0 也照选
    cases = [
        {"answerable": False, "relevant": set(), "hits": [("n1", 0.8)]},
        {"answerable": False, "relevant": set(), "hits": [("n2", 0.6)]},
        {"answerable": True, "relevant": {("d", 1)}, "hits": [(("d", 1), 0.7)]},
    ]
    t, cal_recall = choose_threshold(cases, max_fpr=0.10)
    assert t > 0.8
    assert cal_recall == 0.0


def test_choose_threshold_maximizes_recall_with_tie_break():
    cases = [
        {"answerable": False, "relevant": set(), "hits": [("n1", 0.3)]},
        {"answerable": True, "relevant": {("d", 1)}, "hits": [(("d", 1), 0.9)]},
        {"answerable": True, "relevant": {("d", 2)}, "hits": [(("d", 2), 0.6)]},
    ]
    t, cal_recall = choose_threshold(cases, max_fpr=0.10)
    assert 0.3 < t <= 0.6  # 两个正例都保住的最高阈值
    assert cal_recall == 1.0


def test_choose_threshold_no_feasible_exits():
    import pytest
    cases = [{"answerable": False, "relevant": set(), "hits": [("n1", 1.0)]}]
    with pytest.raises(SystemExit):
        choose_threshold(cases, max_fpr=0.10)  # 负例满分,任何阈值都误召回


def test_load_cases_derives_relevant_set():
    """_load_cases 必须把 relevant_chunks 派生成可比对的关键集合(relevant),
    否则 choose_threshold/main 真实运行 KeyError(单测全用现成 relevant 没暴露)。"""
    from pathlib import Path
    from app.knowledge.chunking import chunk_document
    from evals.run_knowledge_eval import _load_cases
    corpus_keys = set()
    for p in sorted(Path("knowledge_docs").glob("*.md")):
        src = f"knowledge_docs/{p.name}"
        for i, _ in enumerate(
                chunk_document(p.read_text(encoding="utf-8"), source=src,
                               max_chars=500, overlap_chars=80), start=1):
            corpus_keys.add((src, i))
    assert len(corpus_keys) == 14
    cases = _load_cases(corpus_keys)
    assert all(isinstance(c["relevant"], frozenset) for c in cases)
    youfei = next(c for c in cases if c["id"] == "p_youfei")
    assert youfei["relevant"] == frozenset({("knowledge_docs/商品FAQ.md", 1)})


def test_choose_threshold_tie_prefers_lower():
    """并列(召回/FPR 相同)取低阈值,对负例留最大间隔——spec 勘误 2026-09-11。
    实测教训:取高会压在正例分数悬崖边上,测试片三个换说法问题被阈值卡死。"""
    cases = [
        {"answerable": False, "relevant": set(), "hits": [("n1", 0.3)]},
        {"answerable": True, "relevant": {("d", 1)},
         "hits": [(("d", 1), 0.9), (("d", 2), 0.5)]},  # 0.5 为同 case 的非相关命中
    ]
    t, cal_recall = choose_threshold(cases, max_fpr=0.10)
    assert t == 0.5  # 0.5 与 0.9 在召回/FPR 上并列,取低
    assert cal_recall == 1.0
