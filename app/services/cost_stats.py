"""ch09 成本统计(spec §5.1):Langfuse public API 分页拉取,按 trace metadata.intent 聚合。

数据源(legacy v1 端点;Context7 复核,master 2026-10-09):
- 钉 server v3.225.11,不能用 v2 Observations API(官方文档明示需 self-hosted v4+);
  v1 端点在该版 fern 定义仍存在(fern/apis/server/definition/legacy/observations-v1.yml)。
- GET /api/public/traces:参数 page/limit/fromTimestamp/orderBy;响应 data[] + meta.totalPages;
  行含 metadata(metadata.intent 在此取;注意 GET API 可能滞后 ~10 分钟)。
- GET /api/public/observations:参数 page/limit/traceId/type(如 GENERATION);响应同构;
  行含 usageDetails/costDetails。
- usageDetails 契约:各键互斥不重叠(input 不含 input_* 等),total 是总和;
  缺 input/output 时用 total,三者都缺计入 missing_usage_observations 不静默当 0。
- 认证:Basic Auth(public_key, secret_key)。
- 依据 URL:github.com/langfuse/langfuse-docs content/docs/api-and-data-platform/features/public-api.mdx
  + content/docs/observability/features/token-and-cost-tracking.mdx + server v3.225.11 fern 定义。

聚合口径:observation 按 id 全局去重(同一行可能随翻页/多 trace 重复出现);
每逻辑 turn(trace)= 1 个 request;检索/工具 span 不计模型 token(type=GENERATION 过滤);
优先 costDetails,缺失按单价折算(token/1e6;仅有 total 无 input/output 拆分时整段按
输入单价折算);单价皆 0 时 cost=0 且 cost_available=false。无意图 trace 聚 unknown。
页面上限 MAX_TRACES=200 / MAX_OBS=2000,超限 incomplete=true。
"""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

MAX_TRACES = 200
MAX_OBS = 2000
PAGE_SIZE = 50  # v1 public API 单页上限

_DISABLED = "langfuse_disabled"
_UNAVAILABLE = "langfuse_unavailable"


class CostStatsError(Exception):
    """code ∈ {langfuse_disabled(503), langfuse_unavailable(502)};main 统一转错误契约。"""

    def __init__(self, code: str, message: str, status: int):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


def _default_client_factory(settings: Settings) -> httpx.Client:
    return httpx.Client(
        base_url=settings.langfuse_host.rstrip("/"),
        auth=(settings.langfuse_public_key.strip(), settings.langfuse_secret_key.strip()),
        timeout=10.0)


def _get_page(client: httpx.Client, path: str, params: dict) -> dict:
    try:
        resp = client.get(path, params=params)
    except httpx.HTTPError as exc:
        raise CostStatsError(_UNAVAILABLE,
                             f"Langfuse 不可达: {type(exc).__name__}", 502) from exc
    if resp.status_code >= 400:
        raise CostStatsError(_UNAVAILABLE,
                             f"Langfuse 响应异常: HTTP {resp.status_code}", 502)
    payload = resp.json()
    if not isinstance(payload, dict):
        raise CostStatsError(_UNAVAILABLE, "Langfuse 响应非预期格式", 502)
    return payload


def _total_pages(payload: dict) -> int:
    try:
        return int((payload.get("meta") or {}).get("totalPages") or 1)
    except (TypeError, ValueError):
        return 1


def _bucket(intent: str) -> dict:
    return {"intent": intent, "requests": 0, "total_tokens": 0,
            "input_tokens": 0, "output_tokens": 0, "cost": 0.0}


def cost_by_intent(settings: Settings, days: int, *, client_factory=None) -> dict:
    """时间窗 days 天内按意图聚合 token/成本;未启用 → 503 langfuse_disabled。"""
    if not (settings.langfuse_enabled and settings.has_langfuse_key()):
        raise CostStatsError(_DISABLED, "Langfuse 未启用或未配齐密钥", 503)
    factory = client_factory or _default_client_factory
    with factory(settings) as client:
        return _aggregate(settings, client, days)


def _aggregate(settings: Settings, client: httpx.Client, days: int) -> dict:
    from_timestamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    incomplete = False

    traces: list[dict] = []
    page, total_pages = 1, 1
    while page <= total_pages and len(traces) < MAX_TRACES:
        payload = _get_page(client, "/api/public/traces",
                            {"page": page, "limit": PAGE_SIZE,
                             "fromTimestamp": from_timestamp,
                             "orderBy": "timestamp.desc"})
        rows = payload.get("data") or []
        room = MAX_TRACES - len(traces)
        if len(rows) > room:
            rows = rows[:room]
            incomplete = True
        traces.extend(rows)
        total_pages = _total_pages(payload)
        page += 1
    if page <= total_pages:
        incomplete = True  # 还有页未拉,被 MAX_TRACES 截断

    in_price = settings.model_input_price_per_mtok / 1e6
    out_price = settings.model_output_price_per_mtok / 1e6
    used_cost_details = False
    seen_obs: set[str] = set()
    kept_obs = 0
    missing = 0
    buckets: dict[str, dict] = {}

    for trace in traces:
        intent = str(((trace.get("metadata") or {}).get("intent")) or "").strip()
        bucket = buckets.setdefault(intent or "unknown", _bucket(intent or "unknown"))
        bucket["requests"] += 1
        opage, ototal = 1, 1
        while opage <= ototal and kept_obs < MAX_OBS:
            payload = _get_page(client, "/api/public/observations",
                                {"page": opage, "limit": PAGE_SIZE,
                                 "traceId": trace.get("id"), "type": "GENERATION"})
            for obs in payload.get("data") or []:
                if kept_obs >= MAX_OBS:
                    incomplete = True
                    break
                oid = obs.get("id")
                if oid is None or oid in seen_obs:
                    continue
                seen_obs.add(oid)
                kept_obs += 1
                usage = obs.get("usageDetails") or {}
                inp, out, tot = (usage.get("input"), usage.get("output"),
                                 usage.get("total"))
                if inp is None and out is None and tot is None:
                    missing += 1  # 不静默当 0
                    continue
                if inp is not None:
                    bucket["input_tokens"] += inp
                if out is not None:
                    bucket["output_tokens"] += out
                if inp is not None and out is not None:
                    bucket["total_tokens"] += inp + out
                else:  # 缺一项时用 total;total 也缺则用仅有的那项
                    only = tot if tot is not None else (inp if inp is not None else out)
                    bucket["total_tokens"] += only
                details = obs.get("costDetails") or {}
                c_in, c_out = details.get("input"), details.get("output")
                c_tot = details.get("total")
                if c_in is not None or c_out is not None:
                    used_cost_details = True
                    bucket["cost"] += (c_in or 0) + (c_out or 0)
                elif c_tot is not None:
                    used_cost_details = True
                    bucket["cost"] += c_tot
                elif inp is not None and out is not None:
                    bucket["cost"] += inp * in_price + out * out_price
                else:  # 拆分未知:整段按输入单价折算(保守取较低档)
                    only = tot if tot is not None else (inp if inp is not None else out)
                    bucket["cost"] += only * in_price
            ototal = _total_pages(payload)
            opage += 1
        if opage <= ototal:
            incomplete = True  # observation 翻页被 MAX_OBS 截断

    for b in buckets.values():
        b["cost"] = round(b["cost"], 8)
    intents = [b for k, b in buckets.items() if k != "unknown"]
    intents.sort(key=lambda b: (-b["requests"], -b["total_tokens"], b["intent"]))
    return {"days": days, "incomplete": incomplete,
            "missing_usage_observations": missing,
            "cost_available": used_cost_details
            or (in_price > 0 or out_price > 0),
            "intents": intents, "unknown": buckets.get("unknown")}
