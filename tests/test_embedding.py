import pytest

from app.knowledge.embedding import build_embeddings
from tests.conftest import make_settings


def test_build_requires_key():
    with pytest.raises(ValueError, match="embedding_api_key"):
        build_embeddings(make_settings(embedding_api_key="  "))


def test_build_params():
    emb = build_embeddings(make_settings(embedding_api_key="sk-test"))
    assert emb.model == "BAAI/bge-m3"
    assert emb.check_embedding_ctx_length is False  # 第三方端点必须关 tiktoken
    assert emb.max_retries == 0                      # 重试由 executor 统一管理
    assert emb.openai_api_base == "https://api.siliconflow.cn/v1"
