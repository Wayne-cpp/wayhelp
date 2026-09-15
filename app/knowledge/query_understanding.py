"""Query 理解(spec §5):口语问法改写归一 + 同义词扩展(只在检索侧,不入库侧拆存)。

任何失败(模型异常/超时/解析失败/空改写)都降级为原问法直查,不阻断检索。
"""

import json
import logging
import re
from dataclasses import dataclass

from app.prompts.query_understanding import QUERY_UNDERSTANDING_PROMPT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryPlan:
    standard_query: str
    synonyms: tuple[str, ...]
    rewrite_model: str | None   # 实际执行改写的模型名;降级为 None
    degraded: bool
    note: str | None


def passthrough_plan(query: str, note: str | None = None) -> QueryPlan:
    return QueryPlan(query, (), None, True, note)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_plan(text: str, raw_query: str, model_name: str | None) -> QueryPlan:
    """从模型输出解析 QueryPlan;任何不合规都降级。"""
    m = _JSON_RE.search(text or "")
    if not m:
        return passthrough_plan(raw_query, "rewrite_unparseable")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return passthrough_plan(raw_query, "rewrite_unparseable")
    std = data.get("standard_query")
    syns = data.get("synonyms")
    if not isinstance(std, str) or not std.strip():
        return passthrough_plan(raw_query, "rewrite_empty")
    if not isinstance(syns, list):
        syns = []
    syns = tuple(s.strip() for s in syns[:5] if isinstance(s, str) and s.strip())
    return QueryPlan(std.strip(), syns, model_name, False, None)


def plan_query(model, query: str, *, enabled: bool, timeout_seconds: float,
               model_name: str | None = None) -> QueryPlan:
    """生成 QueryPlan。enabled=False 或 model=None 或任何异常 → 原问法降级。
    timeout_seconds 为剩余预算提示;同步 invoke 本身无法硬中断,超时兜底由外层
    wait_for(executor 20s 总预算)承担。"""
    if not enabled or model is None:
        return passthrough_plan(query, "rewrite_disabled" if not enabled else "no_model")
    try:
        # prompt 内含字面 JSON 花括号,不能用 str.format(会把 {"standard_query"...}
        # 当替换字段),改用占位符替换
        prompt = QUERY_UNDERSTANDING_PROMPT.replace("{query}", query)
        resp = model.invoke([("user", prompt)])
        content = resp.content if isinstance(resp.content, str) else ""
    except Exception as exc:
        logger.warning("query understanding failed: %s", type(exc).__name__)
        return passthrough_plan(query, f"rewrite_error:{type(exc).__name__}")
    return parse_plan(content, query, model_name)
