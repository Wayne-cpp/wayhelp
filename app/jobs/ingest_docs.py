"""python -m app.jobs.ingest_docs [docs_dir] —— 知识文档建库(spec §6)。"""

import sys
from pathlib import Path

from app.config import Settings
from app.db import make_engine, make_session_factory, ping
from app.knowledge.embedding import build_embeddings
from app.knowledge.ingest import run_ingest
from app.knowledge.milvus_store import MilvusKnowledgeStore


def main(argv: list[str]) -> int:
    docs_dir = Path(argv[1]) if len(argv) > 1 else Path("knowledge_docs")
    settings = Settings()
    if not settings.has_embedding_key():
        print("[ingest] embedding_api_key 未配置,写库前退出", file=sys.stderr)
        return 2
    engine = make_engine(settings.database_url)
    try:
        ping(engine)
    except Exception as exc:
        print(f"[ingest] 数据库不可达: {type(exc).__name__}", file=sys.stderr)
        return 2
    session_factory = make_session_factory(engine)
    store = MilvusKnowledgeStore(settings.milvus_uri, settings.embedding_dim)
    try:
        return run_ingest(settings, session_factory, build_embeddings(settings),
                          store, docs_dir)
    finally:
        store.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
