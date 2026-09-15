"""证据组装(spec §5/§6):唯一列表截断 → 首尾展示排位 → 预算整条弃 → 裁剪后编号。"""

from app.knowledge.retriever import Evidence, KnowledgeHit, assemble_evidence


def _hit(i):
    return KnowledgeHit(i, 1.0 - i * 0.01, "类目", f"问{i}", f"答{i}", "doc", i, f"章{i}")


def test_display_permutation_n10():
    ev = assemble_evidence([_hit(i) for i in range(1, 12)], max_items=10,
                           budget_chars=10**6, overhead_chars=0)
    assert [e.chunk_id for e in ev] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]  # 首尾最优
    assert [e.ref_no for e in ev] == list(range(1, 11))


def test_display_permutation_n6():
    ev = assemble_evidence([_hit(i) for i in range(1, 7)], max_items=10,
                           budget_chars=10**6, overhead_chars=0)
    assert [e.chunk_id for e in ev] == [1, 3, 5, 6, 4, 2]


def test_display_permutation_n1():
    ev = assemble_evidence([_hit(1)], max_items=10, budget_chars=10**6,
                           overhead_chars=0)
    assert [e.chunk_id for e in ev] == [1]


def test_budget_cut_keeps_whole_objects():
    hits = [_hit(i) for i in range(1, 11)]
    ev = assemble_evidence(hits, max_items=10, budget_chars=250, overhead_chars=0)
    assert 0 < len(ev) < 10
    assert [e.ref_no for e in ev] == list(range(1, len(ev) + 1))  # 裁剪后才编号
    import json
    assert len(json.dumps([e.to_dict() for e in ev], ensure_ascii=False)) <= 250


def test_budget_zero_yields_empty():
    assert assemble_evidence([_hit(1)], max_items=10, budget_chars=1,
                             overhead_chars=0) == []
