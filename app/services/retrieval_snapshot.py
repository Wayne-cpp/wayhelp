"""ch09:落池召回片段快照(retrieval_result.hits → 审核页展示形态)。"""


def snapshot_top_chunks(hits: list[dict] | None, top_n: int,
                        *, answer_chars: int = 500) -> list[dict] | None:
    """截 Top N;answer 截 answer_chars 防爆。hits 为 asdict(KnowledgeHit) 形态。
    空/None → None(没走检索的入口语义:NULL 而不是空数组)。"""
    if not hits:
        return None
    out = []
    for h in hits[:top_n]:
        out.append({"chunk_id": h.get("chunk_id"), "score": h.get("score"),
                    "section_path": h.get("section_path"),
                    "question": h.get("questions"),
                    "answer": (h.get("answer") or "")[:answer_chars]})
    return out or None
