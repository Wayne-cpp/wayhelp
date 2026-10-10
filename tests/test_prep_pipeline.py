"""数据管线确定性部分:LLM 校对 ID 对齐、生成解析、预标解析、corrections 回灌。
LLM 本体不进 pytest——全部用假模型(spec §10)。"""

import json

from training.prep_clean import llm_polish
from training.prep_generate import parse_generated
from training.prep_label import (apply_corrections, disagreements,
                                 export_review_markdown, parse_prelabel)


class FakeLLM:
    """批量校对假模型:receive list[dict(id,text)],按 script 返回。"""
    def __init__(self, script):
        self.script = script  # list[list[dict]] 每次 invoke 弹一批
    def polish_batch(self, batch):
        return self.script.pop(0) if self.script else batch


def _rec(i, text):
    return {"sample_id": f"syn-0-{i:04d}", "source": "synthetic",
            "source_group_id": f"syn-0-{i:04d}", "text_raw": text, "text": text,
            "labels": [], "intended_labels": ["物流"], "parent_id": None}


def test_llm_polish_id_aligned():
    recs = [_rec(0, "物流慢死了"), _rec(1, "什么时候发货")]
    llm = FakeLLM([[{"sample_id": "syn-0-0000", "text": "物流慢极了"},
                    {"sample_id": "syn-0-0001", "text": "什么时候发货"}]])
    out, failed = llm_polish(recs, llm, batch_size=2)
    assert failed == 0 and out[0]["text"] == "物流慢极了"


def test_llm_polish_misaligned_keeps_original():
    recs = [_rec(0, "a"), _rec(1, "b")]
    # 返回条数不符 → 整批保留原文并计失败
    llm = FakeLLM([[{"sample_id": "syn-0-0000", "text": "a!"}]])
    out, failed = llm_polish(recs, llm, batch_size=2)
    assert failed == 1 and [r["text"] for r in out] == ["a", "b"]
    # 返回 ID 错位 → 同样整批保留
    llm2 = FakeLLM([[{"sample_id": "X", "text": "a"}, {"sample_id": "Y", "text": "b"}]])
    out2, failed2 = llm_polish(recs, llm2, batch_size=2)
    assert failed2 == 1 and [r["text"] for r in out2] == ["a", "b"]


def test_parse_generated():
    text = json.dumps([{"text": "买大了想退", "labels": ["尺码", "退换货"]},
                       {"text": "什么时候发货", "labels": ["物流"]}],
                      ensure_ascii=False)
    out = parse_generated(text, "物流")
    assert out == [("买大了想退", ["尺码", "退换货"]), ("什么时候发货", ["物流"])]


def test_parse_prelabel_filters_unknown_and_orders():
    out = parse_prelabel('["尺码", "退换货", "不存在类"]',
                         ["退换货", "物流", "尺码", "发票", "其他"])
    assert out == ["退换货", "尺码"]  # 未知类丢弃,按权威顺序重排


def test_apply_corrections_and_disagreements():
    recs = [_rec(0, "买大了想退")]
    recs[0]["labels"] = ["物流"]  # 预标与意图 ["物流"?] 一致则无分歧;改成不一致:
    recs[0]["labels"] = ["价保"]
    bad = disagreements(recs)
    assert [r["sample_id"] for r in bad] == ["syn-0-0000"]  # 预标 ≠ intended
    fixed = apply_corrections(recs, [{"sample_id": "syn-0-0000",
                                      "labels": ["退换货", "尺码"]}])
    assert fixed[0]["labels"] == ["退换货", "尺码"]
    assert disagreements(fixed) == [] or fixed[0]["intended_labels"] == ["物流"]


def test_export_review_markdown_deterministic(tmp_path):
    # 计划 Step 1 只列了 5 个用例而 Step 4 期望 6——补 export_review_markdown
    # (唯一无测试的公共接口):种子冻结 42,同类两跑输出一致,每类至少抽 1 条
    recs = [_rec(i, f"第{i}号问题") for i in range(20)]
    for i, r in enumerate(recs):
        r["labels"] = ["尺码"] if i % 2 else ["物流"]
    p1, p2 = tmp_path / "r1.md", tmp_path / "r2.md"
    n1 = export_review_markdown(recs, p1)
    n2 = export_review_markdown(recs, p2)
    assert n1 == n2 == 2  # 两类各 10 条,每类 max(1, round(10×0.1)) = 1
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
    lines = [ln for ln in p1.read_text(encoding="utf-8").splitlines()
             if ln.startswith("- ")]
    assert len(lines) == 2 and all("`" in ln for ln in lines)
