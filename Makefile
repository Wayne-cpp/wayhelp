.PHONY: eval-rag mcp-logistics mcp-after-sales topic-inference
eval-rag:
	uv run python evals/run_retrieval_compare.py

mcp-logistics: ## 起物流 MCP Server(:8101)
	uv run python -m app.mcp_servers.logistics_server

mcp-after-sales: ## 起售后 MCP Server(:8102)
	uv run python -m app.mcp_servers.after_sales_server

topic-inference: ## 起主题分类推理服务(:8103,需先导出模型包)
	uv run --group train python -m training.serve
