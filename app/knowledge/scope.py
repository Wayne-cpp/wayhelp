"""元数据过滤的 scope 维度(spec §4.1):有限枚举,由 content_type + source_doc 单一派生。"""

SCOPES = ("faq", "policy", "product_spec", "after_sales_manual", "qa_mined", "manual")

_SCOPE_BY_SOURCE_DOC = {
    "knowledge_docs/product-specs.md": "product_spec",
    "knowledge_docs/after-sales-manual.md": "after_sales_manual",
}


def derive_scope(content_type: str | None, source_doc: str | None) -> str:
    if content_type == "qa_mined":
        return "qa_mined"
    if source_doc is not None:
        if source_doc in _SCOPE_BY_SOURCE_DOC:
            return _SCOPE_BY_SOURCE_DOC[source_doc]
        if content_type in ("faq", "policy", "manual"):
            return content_type
    return "manual"  # 手工录入与未知来源统一 manual
