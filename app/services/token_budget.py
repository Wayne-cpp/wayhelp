"""ch07 上下文预算:唯一 token 估算尺 + 从模型窗口倒推历史预算(spec §4/§5)。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F), (0xFF00, 0xFFEF))
PER_MESSAGE_OVERHEAD = 4
TOOL_CALL_OVERHEAD_TOKENS = 50      # ReAct 每步 tool_call JSON 开销(PEAK 公式)
EVIDENCE_TOKENS_PER_DOC = 300       # 证据预留/条
LAYER2_HEAD_CHARS = 50              # 层2 assistant 答复保留开头字数
TRUNCATED_MARK = "…[已截断]"
ELLIPSIS = "……"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text
              if any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES))
    return cjk + math.ceil((len(text) - cjk) / 4)


def estimate_message(msg) -> int:
    content = msg.content if isinstance(getattr(msg, "content", None), str) else ""
    total = estimate_tokens(content) + PER_MESSAGE_OVERHEAD
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        total += estimate_tokens(json.dumps(tool_calls, ensure_ascii=False))
    return total


def estimate_messages(msgs) -> int:
    return sum(estimate_message(m) for m in msgs)


def truncate_to_tokens(text: str, budget: int) -> str:
    """二分截断,保证 estimate_tokens(结果) <= budget。"""
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def truncate_tool_result(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    return truncate_to_tokens(text, max_tokens - estimate_tokens(TRUNCATED_MARK)) + TRUNCATED_MARK


def build_summary_projection(segments: list[str], budget_tokens: int) -> str:
    """方案 A(spec §9):所有段按 seq 序等分配额度;放得下留全文,放不下 head+tail
    截短加省略标记;段数不丢,不改原文。"""
    if not segments or budget_tokens <= 0:
        return ""
    full = "\n\n".join(segments)
    if estimate_tokens(full) <= budget_tokens:
        return full
    per = budget_tokens // len(segments)
    out = []
    for seg in segments:
        if estimate_tokens(seg) <= per:
            out.append(seg)
            continue
        body = max(0, per - estimate_tokens(ELLIPSIS))
        head = truncate_to_tokens(seg, body * 2 // 3)
        tail_budget = body - estimate_tokens(head)
        tail = truncate_to_tokens(seg[::-1], tail_budget)[::-1] if tail_budget > 0 else ""
        out.append(head + ELLIPSIS + tail)
    return "\n\n".join(out)


def measure_sys_tokens(system_prompt: str, tools) -> int:
    """系统提示 + 工具 schema 的估算开销(启动时实测,spec §5 SYS_TOKENS)。"""
    parts = [system_prompt]
    for t in tools:
        parts.append(t.name)
        parts.append(t.description or "")
        try:
            parts.append(json.dumps(t.args_schema.model_json_schema(), ensure_ascii=False))
        except Exception:
            pass
    return estimate_tokens("\n".join(parts)) + PER_MESSAGE_OVERHEAD


@dataclass(frozen=True)
class ContextBudget:
    avail: int
    total: int
    layer1: int
    layer2: int
    peak: int
    fixed: int
    sys_tokens: int
    sufficient: bool      # total >= steady_tokens_per_turn(一轮都装不下 → 启动报警)


def compute_budget(settings, sys_tokens: int) -> ContextBudget:
    peak = settings.max_agent_steps * (settings.tool_result_max_tokens
                                       + TOOL_CALL_OVERHEAD_TOKENS)
    fixed = (sys_tokens + settings.rerank_top_k * EVIDENCE_TOKENS_PER_DOC
             + settings.summary_projection_tokens + settings.safety_margin_tokens)
    avail = max(0, settings.model_context_window - settings.max_output_tokens
                - settings.max_user_input_tokens - peak - fixed)
    total = min(settings.history_target_turns * settings.steady_tokens_per_turn, avail)
    layer1 = int(total * 0.7)
    return ContextBudget(avail=avail, total=total, layer1=layer1, layer2=total - layer1,
                         peak=peak, fixed=fixed, sys_tokens=sys_tokens,
                         sufficient=total >= settings.steady_tokens_per_turn)
