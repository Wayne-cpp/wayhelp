"""python -m app.jobs.mine_qa —— 从历史客服对话挖知识(spec §7)。"""

import sys

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db import make_engine, make_session_factory, ping
from app.knowledge.embedding import build_embeddings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.mining import run_mining


def main(argv: list[str]) -> int:
    settings = Settings()
    if not settings.has_embedding_key():
        print("[mining] embedding_api_key 未配置,写库前退出", file=sys.stderr)
        return 2
    engine = make_engine(settings.database_url)
    try:
        ping(engine)
    except Exception as exc:
        print(f"[mining] 数据库不可达: {type(exc).__name__}", file=sys.stderr)
        return 2
    model = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key,
                       base_url=settings.openai_base_url,
                       max_tokens=settings.max_output_tokens)
    session_factory = make_session_factory(engine)
    store = MilvusKnowledgeStore(settings.milvus_uri, settings.embedding_dim)
    try:
        return run_mining(settings, session_factory, model,
                          build_embeddings(settings), store)
    finally:
        store.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
