import pytest

from app.knowledge.chunking import ChunkingError, chunk_document

POLICY_DOC = """---
type: policy
---
# 售后政策

## 退货

七天无理由退货。定制类商品不支持退货。

## 换货

十五天内可申请换货。
"""

FAQ_DOC = """---
type: faq
---
# 商品FAQ

## 运费怎么算

单笔订单实付满 99 元包邮,未满收 8 元基础运费。

## 什么时候发货

工作日 16 点前当天发。
"""


def test_policy_headings_split_and_fields():
    chunks = chunk_document(POLICY_DOC, source="d.md", max_chars=500, overlap_chars=80)
    assert len(chunks) == 2
    c0, c1 = chunks
    assert c0.questions == "退货" and c0.category == "售后政策"
    assert c0.section_path == "售后政策 > 退货"
    assert "七天无理由" in c0.answer
    assert c0.is_key_clause is True  # 命中「不支持」
    assert c1.questions == "换货" and c1.is_key_clause is False
    assert all(c.content_type == "policy" for c in chunks)


def test_faq_questions_from_real_heading():
    chunks = chunk_document(FAQ_DOC, source="f.md", max_chars=500, overlap_chars=80)
    assert [c.questions for c in chunks] == ["运费怎么算", "什么时候发货"]
    assert all(c.category == "商品FAQ" for c in chunks)
    assert all(c.content_type == "faq" for c in chunks)


def test_frontmatter_missing_or_bad():
    with pytest.raises(ChunkingError):
        chunk_document("# 无头\n\n正文。", source="x.md", max_chars=500, overlap_chars=80)
    with pytest.raises(ChunkingError):
        chunk_document("---\ntype: unknown\n---\n# t\n\n正文。",
                       source="x.md", max_chars=500, overlap_chars=80)


def test_missing_h1_and_empty_body():
    with pytest.raises(ChunkingError):
        chunk_document("---\ntype: policy\n---\n## 只有二级\n\n正文。",
                       source="x.md", max_chars=500, overlap_chars=80)
    with pytest.raises(ChunkingError):
        chunk_document("---\ntype: policy\n---\n# 只有标题\n",
                       source="x.md", max_chars=500, overlap_chars=80)


def test_heading_only_section_produces_no_chunk():
    doc = "---\ntype: policy\n---\n# T\n\n## 空节\n\n## 有内容\n\n正文一句。"
    chunks = chunk_document(doc, source="x.md", max_chars=500, overlap_chars=80)
    assert len(chunks) == 1 and chunks[0].questions == "有内容"


def test_long_paragraph_splits_by_sentence_with_overlap():
    body = "第一句很长啊。第二句也不短呢。第三句更长了。"
    doc = f"---\ntype: policy\n---\n# T\n\n## S\n\n{body}"
    chunks = chunk_document(doc, source="x.md", max_chars=20, overlap_chars=10)
    assert len(chunks) == 2
    assert chunks[0].answer == "第一句很长啊。第二句也不短呢。"
    # 重叠 = 上一块末尾 ≤10 的完整句后缀「第二句也不短呢。」
    assert chunks[1].answer == "第二句也不短呢。\n第三句更长了。"
    assert all(len(c.answer) <= 20 for c in chunks)


def test_no_overlap_across_sections():
    doc = ("---\ntype: policy\n---\n# T\n\n## A\n\n甲句结尾在此。"
           "\n\n## B\n\n乙句开头在此。")
    chunks = chunk_document(doc, source="x.md", max_chars=500, overlap_chars=10)
    assert chunks[1].answer == "乙句开头在此。"  # 不跨 section 重叠


def test_hard_cut_when_no_sentence_punctuation():
    body = "无标点" * 30  # 90 字无句末标点
    doc = f"---\ntype: policy\n---\n# T\n\n## S\n\n{body}"
    chunks = chunk_document(doc, source="x.md", max_chars=40, overlap_chars=10)
    assert len(chunks) == 3  # 90/40 硬切
    assert all(len(c.answer) <= 40 for c in chunks)
    assert not chunks[1].answer.startswith(chunks[0].answer[-10:])  # 硬切尾不当作重叠内容
    assert chunks[1].answer == body[40:80]


def test_table_header_copied_into_every_piece():
    table = ("| 项目 | 标准 |\n|---|---|\n| 退货时效 | 签收后7天 |\n"
             "| 换货时效 | 签收后15天 |\n| 运费险 | 支持首重 |")
    doc = f"---\ntype: manual\n---\n# 手册\n\n## 时效\n\n{table}"
    chunks = chunk_document(doc, source="m.md", max_chars=40, overlap_chars=10)
    assert len(chunks) == 3
    header = "| 项目 | 标准 |\n|---|---|"
    for c in chunks:
        assert c.answer.startswith(header)
        assert len(c.answer) <= 40
    data_rows = "".join(c.answer for c in chunks)
    assert "退货时效" in data_rows and "换货时效" in data_rows and "运费险" in data_rows


def test_table_row_too_long_errors_with_line_no():
    table = "| 项目 | 标准 |\n|---|---|\n| 超长 | " + "x" * 100 + " |"
    doc = f"---\ntype: manual\n---\n# 手册\n\n## 时效\n\n{table}"
    with pytest.raises(ChunkingError) as exc_info:
        chunk_document(doc, source="m.md", max_chars=40, overlap_chars=10)
    assert exc_info.value.line_no is not None


def test_faq_long_answer_chunks_share_question():
    answer = "第一点说明。第二点说明。第三点说明。"
    doc = f"---\ntype: faq\n---\n# 商品FAQ\n\n## 保修多久\n\n{answer}"
    chunks = chunk_document(doc, source="f.md", max_chars=15, overlap_chars=6)
    assert len(chunks) >= 2
    assert all(c.questions == "保修多久" for c in chunks)


def test_subheading_stays_in_faq_answer():
    doc = ("---\ntype: faq\n---\n# 商品FAQ\n\n## 退货流程\n\n### 第一步\n\n提交申请。"
           "\n\n### 第二步\n\n寄回商品。")
    chunks = chunk_document(doc, source="f.md", max_chars=500, overlap_chars=80)
    assert len(chunks) == 1
    assert "### 第一步" in chunks[0].answer and "### 第二步" in chunks[0].answer
