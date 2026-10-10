"""数据增强:只扩训练集、标签继承、≤1 派生、跨集文本重复丢弃(spec §4.4)。"""

from training.prep_augment import apply_augmentation, plan_augmentation


def _rec(i, labels=None):
    return {"sample_id": f"s{i:03d}", "source": "synthetic",
            "source_group_id": f"g{i:03d}", "text_raw": f"原文{i}",
            "text": f"原文{i}", "labels": labels or ["物流"],
            "intended_labels": labels or ["物流"], "parent_id": None}


def test_plan_respects_target_ratio():
    train = [_rec(i) for i in range(10)]
    plan = plan_augmentation(train, all_texts={r["text"] for r in train},
                             target_ratio=1.3)
    assert len(plan) == 3  # 10 → 13,扩 3 条
    assert all(p["parent_id"] in {r["sample_id"] for r in train} for p in plan)


def test_apply_inherits_and_dedupes():
    train = [_rec(0, ["退换货", "尺码"]), _rec(1)]
    existing = {"原文0", "原文1", "验证集文本"}
    variants = {"s000": "买大啦想退掉", "s001": "验证集文本"}
    out = apply_augmentation(train, variants, existing, {"s000": "train", "s001": "train"})
    assert len(out) == 1  # 与既有文本重复的丢弃
    aug = out[0]
    assert aug["parent_id"] == "s000" and aug["labels"] == ["退换货", "尺码"]
    assert aug["source_group_id"] == "g000"  # 继承来源组(不可拆一致性)
    assert aug["sample_id"].startswith("aug-s000")


def test_augment_never_touches_val_test():
    train = [_rec(0)]
    out = apply_augmentation(train, {"s000": "换个说法"},
                             {"原文0"}, {"s000": "train"})
    assert all(r["parent_id"] == "s000" for r in out)
    # 只接受 train 归属的 parent:val 里的样本不得被增强
    out2 = apply_augmentation([_rec(9)], {"s009": "x"}, {"原文9"},
                              {"s009": "val"})
    assert out2 == []
