"""ch08 MCP client 网关(spec §4.2):MultiServerMCPClient 全局单例的薄封装。
每轮按 Server 独立超时现问现拿;权限/归属只认本地能力政策;某 Server 挂了跳过记 warning。"""
import asyncio
import json
import logging
import time

from app.services.langfuse_tracing import observation
from app.tools.catalog import ToolSpec, classify_mcp_tool, mcp_args_model
from app.tools.executor import McpToolError

logger = logging.getLogger("wayhelp.graph")

_MCP_BUDGET_PRIORITY = 10  # 动态工具最低档:超预算先剔(spec §1.5)


class McpGateway:
    def __init__(self, settings):
        self._settings = settings   # ch09:mcp.call span 判 enabled 用
        self._timeout = settings.mcp_discovery_timeout_seconds
        servers = {}
        if settings.mcp_logistics_url:
            servers["logistics"] = {"url": settings.mcp_logistics_url,
                                    "transport": "streamable_http"}
        if settings.mcp_after_sales_url:
            servers["after_sales"] = {"url": settings.mcp_after_sales_url,
                                      "transport": "streamable_http"}
        self._servers = tuple(servers)
        from langchain_mcp_adapters.client import MultiServerMCPClient
        self._client = MultiServerMCPClient(servers) if servers else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def discover(self) -> list[ToolSpec]:
        """每轮装配时调用;按 Server 独立超时/独立捕获,互不倒逼。"""
        if self._client is None:
            return []
        out: list[ToolSpec] = []
        for server in self._servers:
            try:
                tools = await asyncio.wait_for(
                    self._client.get_tools(server_name=server), timeout=self._timeout)
            except Exception as exc:
                logger.warning("mcp discover %s 失败(本轮跳过): %s",
                               server, type(exc).__name__)
                continue
            for t in tools:
                schema = t.args_schema if isinstance(t.args_schema, dict) \
                    else t.args_schema.model_json_schema()
                verdict = classify_mcp_tool(server, schema)
                if verdict is None:
                    logger.warning("mcp tool %s/%s 不符本地只读策略,拒进工具面",
                                   server, t.name)
                    continue
                permission, ownership = verdict
                out.append(ToolSpec(
                    name=t.name, description=t.description or "",
                    args_schema=schema, args_model=mcp_args_model(ownership),
                    source="mcp", permission=permission, ownership=ownership,
                    mcp_server=server, budget_priority=_MCP_BUDGET_PRIORITY,
                    formatter=_slim_result))
        return out

    async def call(self, server: str, name: str, args: dict) -> str:
        """外包一层 tool span(ch09 spec §5.1;disabled 零开销):input 只记
        server+工具名(args 不发 Langfuse),output 记状态/耗时。"""
        started = time.monotonic()
        with observation("tool", "mcp.call",
                         {"server": server, "tool": name},
                         settings=self._settings) as obs:
            try:
                result = await self._call(server, name, args)
            except Exception as exc:
                if obs is not None:
                    obs.update(output={"status": "error",
                                       "error": type(exc).__name__,
                                       "duration_ms": int((time.monotonic() - started) * 1000)})
                raise
            if obs is not None:
                obs.update(output={"status": "success",
                                   "duration_ms": int((time.monotonic() - started) * 1000)})
            return result

    async def _call(self, server: str, name: str, args: dict) -> str:
        if self._client is None:
            raise ConnectionError("mcp gateway not configured")
        async with self._client.session(server) as session:
            result = await session.call_tool(name, args)
        if getattr(result, "isError", False):
            raise McpToolError(f"{server}/{name} returned isError")
        parts = [getattr(c, "text", "") for c in (result.content or [])]
        return "\n".join(p for p in parts if p)

    async def close(self) -> None:
        # adapter 按调用新建 session,无长连接;保留接口供 lifespan 对称调用
        return None


def _slim_result(raw: str) -> str:
    """MCP 结果格式化(spec §2 第 6 步):JSON 原样透传(种子 mock 字段全是回答用字段),
    只确保是合法 JSON 且中文不转义;非法 JSON 原样返回(截断由引擎统一处理)。"""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (ValueError, TypeError):
        return raw
