import pytest

from app.knowledge.evalset import (EvalCase, covered_groups, load_compare_cases,
                                   parse_gt)


def test_parse_gt_and_or():
    g = parse_gt("MH-W40 + 保修说明 | 质量问题与保修 + 维修寄修")
    assert g == (("MH-W40",), ("保修说明", "质量问题与保修"), ("维修寄修",))


def test_parse_gt_rejects_bad_syntax():
    for bad in ("", "a +", "a || b", "+ a", "a + + b"):
        with pytest.raises(ValueError):
            parse_gt(bad)


def test_covered_groups():
    paths = ["商品规格手册 > 智能猫砂盆 Pro(型号 MH-LP100)", "售后手册 > 维修寄修流程"]
    groups = parse_gt("MH-LP100 + 维修寄修 + 可开票类型")
    assert covered_groups(paths, groups) == 2


def test_load_real_evalset():
    cases = load_compare_cases("evals/retrieval_compare.txt")
    assert len(cases) == 300
    by_bucket = {}
    for c in cases:
        by_bucket.setdefault(c.bucket, []).append(c)
    assert set(by_bucket) == {"A_policy", "B_model", "C_colloquial", "D_absent", "E_multi"}
    for bucket, items in by_bucket.items():
        assert len(items) == 60
        assert sum(1 for c in items if c.split == "calibration") == 30
        assert sum(1 for c in items if c.split == "test") == 30
    for c in by_bucket["D_absent"]:
        assert c.should_refuse and not c.gt_groups and not c.expect_points
    for c in cases:
        if c.bucket != "D_absent":
            assert not c.should_refuse and c.gt_groups
    assert next(c for c in cases if c.id == "D3").query == "发票能不能用外币金额开具"
