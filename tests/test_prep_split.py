"""分层抽样:来源组不可拆、三份无交集、种子可复现、17 类覆盖、manifest 冻结。"""

import pytest

from training.labels import label_names, taxonomy_version
from training.prep_split import (dedupe_by_text, freeze_manifest, split_report,
                                 stratified_split)

NAMES = label_names()


def _rec(i, label, text=None, group=None, source="synthetic"):
    sid = f"s{i:03d}"
    return {"sample_id": sid, "source": source,
            "source_group_id": group or sid,
            "text_raw": text or f"文本{i}", "text": text or f"文本{i}",
            "labels": [label], "intended_labels": [label], "parent_id": None}


def _corpus(per_class=10):
    return [_rec(i * per_class + j, NAMES[i])
            for i in range(17) for j in range(per_class)]


def test_split_ratios_and_coverage():
    recs = _corpus()
    assignment = stratified_split(recs, seed=42)
    assert set(assignment) == {r["sample_id"] for r in recs}
    train = [r for r in recs if assignment[r["sample_id"]] == "train"]
    val = [r for r in recs if assignment[r["sample_id"]] == "val"]
    test = [r for r in recs if assignment[r["sample_id"]] == "test"]
    assert abs(len(train) / len(recs) - 0.8) < 0.06
    assert abs(len(val) / len(recs) - 0.1) < 0.06
    # 三份均覆盖 17 类
    for part in (train, val, test):
        assert {lb for r in part for lb in r["labels"]} == set(NAMES)
    # 种子可复现
    assert stratified_split(recs, seed=42) == assignment


def test_source_group_indivisible():
    recs = _corpus()
    # 把第 0 类前两条捏成同一来源组(模拟同轮真实问题)
    recs[0]["source_group_id"] = recs[1]["source_group_id"] = "turn-1-1"
    assignment = stratified_split(recs, seed=42)
    assert assignment["s000"] == assignment["s001"]


def test_dedupe_merges_groups_and_conflicts():
    a = _rec(0, "物流", text="一样的话")
    b = _rec(1, "物流", text="一样的话")
    c = _rec(2, "发票", text="一样的话")
    merged, conflicts = dedupe_by_text([a, b])
    assert len(merged) == 1 and conflicts == []
    assert set(merged[0]["source_group_id"].split("+")) == {"s000", "s001"}
    _, conflicts2 = dedupe_by_text([a, c])
    assert len(conflicts2) == 1  # 同文异标进人工裁决


def test_manifest_frozen_and_stable():
    recs = _corpus()
    assignment = stratified_split(recs, seed=42)
    m1 = freeze_manifest(recs, assignment, taxonomy_version(), seed=42)
    m2 = freeze_manifest(recs, assignment, taxonomy_version(), seed=42)
    assert m1["dataset_version"] == m2["dataset_version"] and len(m1["dataset_version"]) == 64
    assert m1["seed"] == 42 and m1["taxonomy_version"] == taxonomy_version()
    # 改一个样本标签 → 数据版本变
    recs[0]["labels"] = ["物流"]
    m3 = freeze_manifest(recs, assignment, taxonomy_version(), seed=42)
    assert m3["dataset_version"] != m1["dataset_version"]


def test_split_report_fields():
    recs = _corpus()
    assignment = stratified_split(recs, seed=42)
    rep = split_report(recs, assignment)
    assert set(rep) >= {"actual_ratios", "per_class_support", "multilabel_share"}
