from app.knowledge.scope import SCOPES, derive_scope


def test_derive_scope():
    assert derive_scope("manual", "knowledge_docs/product-specs.md") == "product_spec"
    assert derive_scope("manual", "knowledge_docs/after-sales-manual.md") == "after_sales_manual"
    assert derive_scope("faq", "knowledge_docs/product-faq.md") == "faq"
    assert derive_scope("policy", "knowledge_docs/returns-policy.md") == "policy"
    assert derive_scope("qa_mined", None) == "qa_mined"
    assert derive_scope("faq", None) == "manual"          # 手工录入统一 manual
    assert set(SCOPES) == {"faq", "policy", "product_spec",
                           "after_sales_manual", "qa_mined", "manual"}
