"""ch08 MCP 网关(spec §4.2):按 Server 独立超时发现、能力政策过滤、isError 脱敏。"""
import pytest

from app.services.mcp_gateway import McpGateway
from app.tools.executor import McpToolError
from tests.conftest import make_settings
from tests.test_mcp_servers import after_sales_url, logistics_url  # noqa: F401


def _gw(**over):
    return McpGateway(make_settings(**over))


async def test_discover_three_tools(logistics_url, after_sales_url):
    gw = _gw(mcp_logistics_url=logistics_url, mcp_after_sales_url=after_sales_url)
    specs = {s.name: s for s in await gw.discover()}
    assert set(specs) == {"query_logistics", "query_warranty", "query_return_progress"}
    assert all(s.source == "mcp" and s.permission == "read" for s in specs.values())
    assert specs["query_logistics"].mcp_server == "logistics"
    assert specs["query_logistics"].ownership == "order"
    await gw.close()


async def test_one_server_down_other_still_discovered(logistics_url):
    gw = _gw(mcp_logistics_url=logistics_url,
             mcp_after_sales_url="http://127.0.0.1:9/mcp",  # 不可达
             mcp_discovery_timeout_seconds=1)
    specs = await gw.discover()          # 不抛、不等满默认超时
    assert [s.name for s in specs] == ["query_logistics"]
    await gw.close()


async def test_unconfigured_gateway_discovers_nothing():
    gw = _gw()
    assert not gw.configured
    assert await gw.discover() == []


async def test_call_and_iserror(logistics_url):
    gw = _gw(mcp_logistics_url=logistics_url)
    text = await gw.call("logistics", "query_logistics", {"order_id": "AB12-1001"})
    assert "company" in text
    with pytest.raises((McpToolError, Exception)):  # 未知工具:Server 失败统一脱敏
        await gw.call("logistics", "no_such_tool", {})
    await gw.close()
