from evals.run_knowledge_eval import (
    case_recall, choose_threshold, false_recall_rate, macro_average,
)
from evals.run_retrieval_compare import (
    choose_strategy_threshold, complete_hit_at_k, mrr_at_10, parse_judge_output,
    section_recall_at_k,
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


def test_load_cases_derives_relevant_set(tmp_path, monkeypatch):
    """_load_cases 必须把 relevant_chunks 派生成可比对的关键集合(relevant),
    否则 choose_threshold/main 真实运行 KeyError(单测全用现成 relevant 没暴露)。
    ch04 语料换血后 knowledge_recall.jsonl 标注已失效(引用已删除的旧文档名),
    改用合成标注集钉同一回归,不再依赖真实 knowledge_docs。"""
    import json
    from evals import run_knowledge_eval as rke

    def case(cid, split, answerable, chunks):
        return {"id": cid, "split": split, "query": f"q{cid}", "answerable": answerable,
                "relevant_chunks": [{"source_doc": d, "chunk_index": i} for d, i in chunks],
                "answer_points": []}

    synthetic = [case(f"p{i:02d}", "calibration" if i % 2 else "test", True, [("d1.md", i)])
                 for i in range(1, 22)]           # 21 正例:奇 cal / 偶 test
    synthetic.append(case("p_youfei", "test", True, [("d2.md", 3)]))  # 凑足 22 正例
    synthetic += [case(f"n{i:02d}", "calibration" if i % 2 else "test", False, [])
                  for i in range(1, 21)]          # 20 负例
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(map(json.dumps, synthetic)), encoding="utf-8")
    monkeypatch.setattr(rke, "CASES", path)

    corpus_keys = {("d1.md", i) for i in range(1, 22)} | {("d2.md", 3)}
    cases = rke._load_cases(corpus_keys)
    assert all(isinstance(c["relevant"], frozenset) for c in cases)
    youfei = next(c for c in cases if c["id"] == "p_youfei")
    assert youfei["relevant"] == frozenset({("d2.md", 3)})
    assert next(c for c in cases if not c["answerable"])["relevant"] == frozenset()


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


# ---- ch04 T12:四策略对比脚本纯函数(spec §8)----

G = (("MH-LP100",), ("保修说明", "质量问题与保修"))


def test_section_recall():
    assert section_recall_at_k(["规格 > MH-LP100 款"], G, 1) == 0.5
    assert section_recall_at_k(["无关"], G, 10) == 0.0
    assert complete_hit_at_k(["规格 > MH-LP100 款", "售后 > 保修说明"], G, 2) == 1.0
    assert complete_hit_at_k(["规格 > MH-LP100 款"], G, 5) == 0.0


def test_mrr():
    assert mrr_at_10(["无关", "规格 > MH-LP100 款"], G) == 0.5
    assert mrr_at_10(["无关"], G) == 0.0


def test_choose_threshold_pareto():
    samples = [
        {"bucket": "D_absent", "should_refuse": True, "top1": 0.9, "recall10": 0.0},
        {"bucket": "D_absent", "should_refuse": True, "top1": 0.1, "recall10": 0.0},
        {"bucket": "A_policy", "should_refuse": False, "top1": 0.8, "recall10": 1.0},
        {"bucket": "A_policy", "should_refuse": False, "top1": 0.2, "recall10": 0.5},
        {"bucket": "A_policy", "should_refuse": False, "top1": None, "recall10": 0.0},
    ]
    # max_d_pass=0.5:阈值 >0.1 才挡住一条 D;候选 t=0.2 时 D 通过 1/2、正例 0.8 过
    out = choose_strategy_threshold(samples, max_d_pass=0.5)
    assert out["threshold"] == 0.2
    assert out["d_pass_rate"] == 0.5
    # 并列取「误拒率低 → 阈值小」:t=0.2 与 t=0.8 的 pass-adjusted recall 同为 0.5 时取 0.2
    assert out["pass_adjusted_recall"] == 0.5


def test_choose_threshold_no_feasible():
    samples = [{"bucket": "D_absent", "should_refuse": True, "top1": 0.9, "recall10": 0.0}]
    with __import__("pytest").raises(SystemExit):
        choose_strategy_threshold(samples, max_d_pass=0.0)


def test_parse_judge_output():
    out = parse_judge_output('{"verdict": "fabricated", "unsupported_claims": '
                             '[{"claim": "c", "reason": "r"}], "cited_refs": [1, 3], '
                             '"coverage": 0.5}')
    assert out["verdict"] == "fabricated" and out["cited_refs"] == [1, 3]
    assert out["coverage"] == 0.5
    import pytest
    with pytest.raises(ValueError):
        parse_judge_output("not json")
    with pytest.raises(ValueError):
        parse_judge_output('{"verdict": "maybe", "unsupported_claims": [], '
                           '"cited_refs": [], "coverage": 0.5}')


def test_parse_judge_output_coverage_validation():
    """coverage 缺失/字符串/bool/NaN/越界 → ValueError(计入重试与 judge_error)。"""
    import pytest
    base = '"verdict": "faithful", "unsupported_claims": [], "cited_refs": []'
    for payload in (
        "{" + base + "}",                                  # 缺失
        "{" + base + ', "coverage": "0.5"}',               # 字符串
        "{" + base + ', "coverage": true}',                # bool
        "{" + base + ', "coverage": NaN}',                 # 非有限(json.loads 会放行 NaN)
        "{" + base + ', "coverage": 1.5}',                 # 越界
    ):
        with pytest.raises(ValueError):
            parse_judge_output(payload)
    out = parse_judge_output("{" + base + ', "coverage": 1}')   # int 合法
    assert out["coverage"] == 1.0 and isinstance(out["coverage"], float)


def test_ungated_threshold_fallback():
    from evals.run_retrieval_compare import _ungated_threshold
    samples = [
        {"bucket": "D_absent", "should_refuse": True, "top1": 0.9, "recall10": 0.0},
        {"bucket": "D_absent", "should_refuse": True, "top1": None, "recall10": 0.0},
        {"bucket": "A_policy", "should_refuse": False, "top1": 0.8, "recall10": 1.0},
        {"bucket": "A_policy", "should_refuse": False, "top1": None, "recall10": 0.0},
    ]
    out = _ungated_threshold(samples)
    assert out["threshold"] is None and out["ungated"] is True
    assert out["d_pass_rate"] == 0.5        # 不设闸:有命中的 D 全放过
    assert out["over_refusal_rate"] == 0.5  # 无命中正例计误拒
    assert out["pass_adjusted_recall"] == 0.5


def test_test_metrics_ungated_pass_semantics():
    from evals.run_retrieval_compare import _test_metrics
    cases = [
        {"bucket": "A_policy", "should_refuse": False, "gt_groups": G,
         "retrieval": {"hybrid": {"top1": 0.03, "paths": ["规格 > MH-LP100 款"]}}},
        {"bucket": "D_absent", "should_refuse": True, "gt_groups": (),
         "retrieval": {"hybrid": {"top1": None, "paths": []}}},
    ]
    m = _test_metrics(cases, "hybrid", None)   # ungated:有命中即过闸
    assert m["threshold"] is None
    assert m["section_recall_at_10"] == 0.5    # G 两组只命中一组
    assert m["d_refuse_correct_rate"] == 1.0   # D 无命中 -> 拒,计正确


def test_test_metrics_by_bucket_three_metrics():
    """by_bucket 聚合 MRR/Recall@5/证据覆盖度(CH@10);D 桶不进 by_bucket。"""
    from evals.run_retrieval_compare import _test_metrics
    G2 = (("规格",), ("保修",))
    cases = [
        {"bucket": "A_policy", "should_refuse": False, "gt_groups": G2,
         "retrieval": {"dense": {"top1": 0.9, "paths": ["售后 > 保修说明", "产品规格 > 规格"]}}},
        {"bucket": "A_policy", "should_refuse": False, "gt_groups": G2,
         "retrieval": {"dense": {"top1": 0.8, "paths": ["无关"]}}},
        {"bucket": "C_colloquial", "should_refuse": False, "gt_groups": G2,
         "retrieval": {"dense": {"top1": 0.7, "paths": ["产品规格 > 规格"]}}},
        {"bucket": "D_absent", "should_refuse": True, "gt_groups": (),
         "retrieval": {"dense": {"top1": None, "paths": []}}},
    ]
    m = _test_metrics(cases, "dense", 0.5)
    a = m["by_bucket"]["A_policy"]
    assert a["mrr_at_10"] == 0.5            # (1.0 + 0.0) / 2
    assert a["recall5"] == 0.5
    assert a["evidence_coverage"] == 0.5    # CH@10:(1 + 0) / 2
    assert a["section_recall_at_10"] == 0.5 and a["refused"] == 0
    c = m["by_bucket"]["C_colloquial"]
    assert c["recall5"] == 0.5 and c["evidence_coverage"] == 0.0 and c["mrr_at_10"] == 1.0
    assert "D_absent" not in m["by_bucket"]


def test_simulate_second_turn_uses_real_effective_strategy():
    """rerank 降级时工具出参用实际 effective_strategy,不再硬编码 hybrid_rerank。"""
    from evals.run_retrieval_compare import _simulate_second_turn

    class _Cap:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            class R:
                content = "答"
            return R()

    m = _Cap()
    _simulate_second_turn(m, "q", [], "hybrid")
    import json as _json
    payload = _json.loads(m.messages[-1].content)
    assert payload["effective_strategy"] == "hybrid"


def test_run_generation_counts():
    """_run_generation:低置信硬拒答 / 正常裁判 / D 桶不送裁判 / degraded 标记。"""
    from evals.run_retrieval_compare import _run_generation

    class _Fake:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            class R:
                content = "答"
            if isinstance(messages, list) and messages and isinstance(
                    messages[0], tuple):                       # 裁判调用
                R.content = ('{"verdict": "faithful", "unsupported_claims": [], '
                             '"cited_refs": [1], "coverage": 1.0}')
            return R()

    class _Settings:
        rerank_top_n = 10
        max_tool_result_chars = 4000

    cases = [
        {"id": "A2", "bucket": "A_policy", "should_refuse": False,
         "query": "q1", "expect_points": ("p",),
         "retrieval": {"dense": {"top1": 0.9, "hits": [], "effective_strategy": "dense"}}},
        {"id": "A4", "bucket": "A_policy", "should_refuse": False,
         "query": "q2", "expect_points": ("p",),
         "retrieval": {"dense": {"top1": 0.1, "hits": [], "effective_strategy": "dense"}}},
        {"id": "D2", "bucket": "D_absent", "should_refuse": True,
         "query": "q3", "expect_points": (),
         "retrieval": {"dense": {"top1": None, "hits": [], "effective_strategy": "dense"}}},
    ]
    rows = _run_generation(cases, "dense", 0.5, _Settings(), _Fake())
    a2, a4, d2 = rows
    assert a2["judge"] == "faithful" and a2["coverage"] == 1.0 and not a2["refused"]
    assert a4["refused"] and a4["low_confidence"] and a4["judge"] is None
    assert d2["refused"] and d2["judge"] is None               # D 桶不送裁判
    assert all(not r["degraded"] for r in rows)
