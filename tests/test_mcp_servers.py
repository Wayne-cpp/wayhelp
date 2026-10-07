"""ch08 MCP Server 冒烟(spec §4.3):真起子进程,list-tools + call-tool 钉住
v1 FastMCP API、streamable-http 传输与 /mcp 路径;同时是 adapter transport 字面量的钉。"""
import os
import socket
import subprocess
import sys
import time

import pytest

from app.mcp_servers.mock_data import (
    logistics_trace, return_progress, warranty_status,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} not ready")


@pytest.fixture(scope="module")
def logistics_url():
    port = _free_port()
    env = {**os.environ, "WAYHELP_MCP_LOGISTICS_PORT": str(port),
           "no_proxy": "127.0.0.1,localhost"}
    p = subprocess.Popen([sys.executable, "-m", "app.mcp_servers.logistics_server"],
                         env=env)
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        p.terminate()
        p.wait(10)


@pytest.fixture(scope="module")
def after_sales_url():
    port = _free_port()
    env = {**os.environ, "WAYHELP_MCP_AFTER_SALES_PORT": str(port),
           "no_proxy": "127.0.0.1,localhost"}
    p = subprocess.Popen([sys.executable, "-m", "app.mcp_servers.after_sales_server"],
                         env=env)
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        p.terminate()
        p.wait(10)


def test_mock_data_seeded_stable():
    a, b = logistics_trace("AB12-1001"), logistics_trace("AB12-1001")
    assert a == b and a["company"] and a["tracking_no"] and len(a["traces"]) >= 2
    w = warranty_status("AB12-1001")
    assert w == warranty_status("AB12-1001") and w["in_warranty"] in (True, False)
    r = return_progress("AB12-1001")
    assert r == return_progress("AB12-1001") and r["nodes"]


async def test_logistics_server_list_and_call_tools(logistics_url):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "logistics": {"url": logistics_url, "transport": "streamable_http"}})
    tools = await client.get_tools(server_name="logistics")
    names = {t.name for t in tools}
    assert names == {"query_logistics"}
    t = tools[0]
    schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
    props = schema.get("properties") or {}
    assert set(props) == {"order_id"}
    async with client.session("logistics") as session:
        result = await session.call_tool("query_logistics", {"order_id": "AB12-1001"})
    assert not result.isError
    text = result.content[0].text
    assert "顺丰" in text or "company" in text


async def test_after_sales_server_two_tools(after_sales_url):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "after_sales": {"url": after_sales_url, "transport": "streamable_http"}})
    tools = await client.get_tools(server_name="after_sales")
    assert {t.name for t in tools} == {"query_warranty", "query_return_progress"}
    async with client.session("after_sales") as session:
        r1 = await session.call_tool("query_warranty", {"order_id": "AB12-1001"})
        r2 = await session.call_tool("query_return_progress", {"order_id": "AB12-1001"})
    assert not r1.isError and not r2.isError
