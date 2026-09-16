.PHONY: eval-rag
eval-rag:
	uv run python evals/run_retrieval_compare.py
