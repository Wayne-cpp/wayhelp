.PHONY: eval-rag mcp-logistics mcp-after-sales
eval-rag:
	uv run python evals/run_retrieval_compare.py

mcp-logistics: ## 起物流 MCP Server(:8101)
	uv run python -m app.mcp_servers.logistics_server

mcp-after-sales: ## 起售后 MCP Server(:8102)
	uv run python -m app.mcp_servers.after_sales_server
