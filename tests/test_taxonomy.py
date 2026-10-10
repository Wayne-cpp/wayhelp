"""taxonomy 冻结钉 + 统一解码口径(spec §2/§3)。"""

from training.labels import (DECODER_VERSION, canonical_json, classifier_version,
                             decode_scores, label_names, load_taxonomy,
                             taxonomy_version)

EXPECTED = ["退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动",
            "价保", "支付", "订单修改", "库存补货", "商品信息", "保修维修",
            "账号", "会员积分", "评价", "其他"]


def test_taxonomy_frozen():
    tax = load_taxonomy()
    assert [t["name"] for t in tax] == EXPECTED  # 类名与顺序钉死
    assert len(tax) == 17
    assert all(t["boundary"].strip() for t in tax)  # 边界非空
    assert len(taxonomy_version()) == 64


def test_decode_threshold_inclusive():
    names = label_names()
    scores = [0.0] * 17
    scores[names.index("退换货")] = 0.5
    assert decode_scores(scores, 0.5) == ["退换货"]  # >= 等号命中


def test_decode_zero_hit_falls_back_to_other():
    scores = [0.9 if i == 16 else 0.01 for i in range(17)]  # 只有「其他」高分
    assert decode_scores(scores, 0.5) == ["其他"]  # 「其他」分数不参与判定,零命中兜底


def test_decode_other_mutually_exclusive_and_order():
    names = label_names()
    scores = [0.0] * 17
    scores[names.index("尺码")] = 0.9
    scores[names.index("退换货")] = 0.8
    scores[names.index("其他")] = 0.99
    assert decode_scores(scores, 0.5) == ["退换货", "尺码"]  # 冻结顺序,剔除「其他」


def test_decode_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        decode_scores([0.1, 0.2], 0.5)


def test_classifier_version_stable_and_sensitive():
    v1 = classifier_version("m" * 64, 0.5)
    assert v1 == classifier_version("m" * 64, 0.5) and len(v1) == 64
    assert v1 != classifier_version("m" * 64, 0.51)
    assert v1 != classifier_version("m" * 64, 0.5, decoder_version="v2")
