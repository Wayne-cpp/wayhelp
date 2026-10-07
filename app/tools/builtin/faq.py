"""ch08 内置 query_faq:无参(以本轮 resolved_query 检索),同轮缓存,置信闸状态写 ctx.faq_trace。"""
import json
from dataclasses import dataclass

from pydantic import BaseModel

from app.knowledge.retriever import (
    NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED, RetrievalResult,
    assemble_evidence,
)
from app.tools.builtin import builtin_tool


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class FaqRetrievalTrace:
    """图专用 query_faq 的单轮显式 interface;首轮调用写一次,后续走缓存。"""
    calls: int = 0
    status: str = "not_called"   # ok | low_confidence | unavailable | tool_error
    result: RetrievalResult | None = None
    evidence: list[dict] | None = None
    error_code: str | None = None
    _cached: str | None = None


class QueryFaqArgs(BaseModel):
    pass


@builtin_tool(name="query_faq",
              description="检索知识库,获取政策/流程/规格/常见问题资料。无需参数,以用户本轮完整问题检索。返回检索策略、低置信标记与证据列表。同轮重复调用返回同一结果。",
              args_model=QueryFaqArgs, budget_priority=100)
def query_faq_fn(args: dict, ctx) -> str:
    trace = ctx.faq_trace
    settings = ctx.settings
    trace.calls += 1
    if trace.calls > 1 and trace._cached is not None:
        return trace._cached
    if ctx.retriever is None:
        trace.status = "unavailable"
        trace.error_code = "kb_unconfigured"
        trace._cached = _json({"evidence": [], "note": "知识检索未配置",
                               "low_confidence": False, "effective_strategy": None})
        return trace._cached
    result = ctx.retriever.search(ctx.resolved_query, scope=None)
    if result.note in (NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED):
        trace.status = "unavailable"
        trace.error_code = (
            "kb_rebuilding" if result.note == NOTE_REBUILDING
            else "kb_rebuild_required" if result.note == NOTE_REBUILD_REQUIRED
            else "kb_unconfigured")
        trace._cached = _json({"evidence": [], "note": result.note,
                               "low_confidence": False,
                               "effective_strategy": result.effective_strategy})
        return trace._cached
    trace.result = result
    if result.low_confidence:
        trace.status = "low_confidence"
        trace._cached = _json({"evidence": [], "note": result.note,
                               "low_confidence": True,
                               "effective_strategy": result.effective_strategy})
        return trace._cached
    evidence = [e.to_dict() for e in assemble_evidence(
        result.hits, max_items=settings.rerank_top_k,
        budget_chars=settings.max_tool_result_chars, overhead_chars=200)]
    trace.status = "ok"
    trace.evidence = evidence
    trace._cached = _json({"effective_strategy": result.effective_strategy,
                           "low_confidence": False, "note": result.note,
                           "evidence": evidence})
    return trace._cached
