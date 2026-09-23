"""ch07 三层历史组装(spec §7):纯函数;DB 读由调用方注入。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import trim_messages

from app.services.token_budget import (
    LAYER2_HEAD_CHARS, estimate_message, estimate_messages, estimate_tokens,
)

TOOL_OMITTED = "[工具结果已省略: {}]"


def tool_omitted_text(name: str | None) -> str:
    return TOOL_OMITTED.format(name or "tool")


@dataclass
class LayeredView:
    layer2_rendered: list[BaseMessage]
    layer1: list[BaseMessage]
    summary: str | None
    l1_tokens: int
    l2_tokens: int
    summary_tokens: int
    span_meta: dict = field(default_factory=dict)


def _db_id(msg) -> int | None:
    v = (getattr(msg, "additional_kwargs", None) or {}).get("db_id")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _effective_ids(messages) -> list[int]:
    """顺序扫描:无 db_id 的消息(未落库的 ToolMessage)继承最近已盖章 id。"""
    out, cur = [], 0
    for m in messages:
        mid = _db_id(m)
        if mid is not None:
            cur = mid
        out.append(cur)
    return out


def _render_layer2(msg) -> BaseMessage:
    if isinstance(msg, ToolMessage):
        return msg.model_copy(update={"content": tool_omitted_text(msg.name)})
    if isinstance(msg, AIMessage):
        content = msg.content if isinstance(msg.content, str) else ""
        if len(content) > LAYER2_HEAD_CHARS:
            return msg.model_copy(
                update={"content": content[:LAYER2_HEAD_CHARS] + "…"})
    return msg  # 用户原话不动


def build_layered_view(messages, meta, settings) -> LayeredView:
    summary_upto = meta.summary_upto or 0
    layer1_from = meta.layer1_from if meta.layer1_from is not None else -1  # NULL=负无穷,全部层1
    layer2, layer1 = [], []
    for m, eff in zip(messages, _effective_ids(messages)):
        if eff <= summary_upto:
            continue
        (layer2 if eff <= layer1_from else layer1).append(m)
    rendered = [_render_layer2(m) for m in layer2]
    return LayeredView(layer2_rendered=rendered, layer1=layer1, summary=meta.summary,
                       l1_tokens=estimate_messages(layer1),
                       l2_tokens=estimate_messages(rendered),
                       summary_tokens=estimate_tokens(meta.summary or ""),
                       span_meta={"summary_upto": meta.summary_upto,
                                  "layer1_from": meta.layer1_from})


def needs_reconcile(messages) -> bool:
    return any(not isinstance(m, ToolMessage) and _db_id(m) is None for m in messages)


def _ua_key_msg(m) -> tuple:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    content = m.content if isinstance(m.content, str) else ""
    calls = json.dumps(getattr(m, "tool_calls", None) or None,
                       ensure_ascii=False, sort_keys=True)
    return (role, content, calls)


def _ua_key_rec(r) -> tuple:
    calls = json.dumps(r.tool_calls or None, ensure_ascii=False, sort_keys=True)
    return (r.role, r.content or "", calls)


def reconcile_checkpoint_ids(messages, records):
    """旧 checkpoint(无 db_id)与 messages 表只读对齐;无法唯一对齐 → None(spec §2)。"""
    if not needs_reconcile(messages):
        return list(messages)
    ua_msgs = [(i, m) for i, m in enumerate(messages) if not isinstance(m, ToolMessage)]
    ua_recs = [r for r in records if r.role in ("user", "assistant")]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    tool_recs = [r for r in records if r.role == "tool"]
    if len({_ua_key_msg(m) for _, m in ua_msgs}) != len(ua_msgs):
        return None  # 重复候选:不猜 ID
    if len({_ua_key_rec(r) for r in ua_recs}) != len(ua_recs):
        return None
    if len(ua_msgs) != len(ua_recs):
        return None  # 末尾未完成轮/长度不一
    out = list(messages)
    for (i, m), r in zip(ua_msgs, ua_recs):
        if _ua_key_msg(m) != _ua_key_rec(r):
            return None  # 顺序不一致
        out[i] = m.model_copy(update={
            "additional_kwargs": {**m.additional_kwargs, "db_id": int(r.id)}})
    if tool_recs:  # legacy tool 行:有序按 tool_call_id 配对;新库无 tool 行则跳过
        if len(tool_msgs) != len(tool_recs):
            return None
        for m, r in zip(tool_msgs, tool_recs):
            if (m.tool_call_id or "") != (r.tool_call_id or ""):
                return None
    return out


def render_history_text(view: LayeredView) -> str:
    parts = []
    if view.summary:
        parts.append("早期对话摘要:\n" + view.summary)
    lines = []
    for m in [*view.layer2_rendered, *view.layer1]:
        if isinstance(m, HumanMessage):
            lines.append(f"用户:{m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(tool_omitted_text(m.name))  # 滑窗文本里一律一行标识
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"助手:{m.content}")
    if lines:
        parts.append("对话历史:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def evaluate_degrade(merged_messages, layer1_from: int | None, budget_l1: int) -> int | None:
    """层1 超预算 → 返回新的 layer1_from(被保留首条前一个持久化 id);否则 None。"""
    base = layer1_from if layer1_from is not None else -1
    span = [m for m, eff in zip(merged_messages, _effective_ids(merged_messages))
            if eff > base]
    if not span or estimate_messages(span) <= budget_l1:
        return None
    kept = trim_messages(span, max_tokens=budget_l1, token_counter=estimate_messages,
                         strategy="last", start_on=HumanMessage,
                         include_system=False, allow_partial=False)
    ids = {id(m): eff for m, eff in zip(span, _effective_ids(span))}
    if not kept:
        return _effective_ids(span)[-1]  # 单轮就超:层1 空,边界推到最末
    first_kept = ids.get(id(kept[0]))
    if first_kept is None:
        return None
    earlier = [eff for eff in ids.values() if eff < first_kept]
    if not earlier:
        return None  # 防御:无更早 id 可退
    return max(earlier)
