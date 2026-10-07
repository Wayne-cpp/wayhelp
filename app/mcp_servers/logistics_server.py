"""ch08 物流 MCP Server(独立进程,spec §4.1):query_logistics 接管内置下线的物流查询。
v1 FastMCP + streamable-http;起服:uv run python -m app.mcp_servers.logistics_server"""
import os

from mcp.server.fastmcp import FastMCP

from app.mcp_servers.mock_data import logistics_trace

mcp = FastMCP("logistics", host="127.0.0.1",
              port=int(os.environ.get("WAYHELP_MCP_LOGISTICS_PORT", "8101")))


@mcp.tool()
def query_logistics(order_id: str) -> dict:
    """查询订单物流轨迹。参数 order_id 为订单号。返回物流公司、运单号与物流节点列表。"""
    return logistics_trace(order_id)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
