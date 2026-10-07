"""ch08 售后 MCP Server(独立进程,spec §4.1):查在保 / 查退货进度。
v1 FastMCP + streamable-http;起服:uv run python -m app.mcp_servers.after_sales_server"""
import os

from mcp.server.fastmcp import FastMCP

from app.mcp_servers.mock_data import return_progress, warranty_status

mcp = FastMCP("after_sales", host="127.0.0.1",
              port=int(os.environ.get("WAYHELP_MCP_AFTER_SALES_PORT", "8102")))


@mcp.tool()
def query_warranty(order_id: str) -> dict:
    """查询订单商品是否在保修期内。参数 order_id 为订单号。返回在保状态、保修到期日与说明。"""
    return warranty_status(order_id)


@mcp.tool()
def query_return_progress(order_id: str) -> dict:
    """查询订单退货进度。参数 order_id 为订单号。返回退货单号、当前节点与进度列表。"""
    return return_progress(order_id)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
