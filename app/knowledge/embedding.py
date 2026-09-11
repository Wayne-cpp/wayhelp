"""嵌入客户端封装:OpenAIEmbeddings 指硅基流动 BGE-M3(spec §2/§9)。"""

from langchain_openai import OpenAIEmbeddings

from app.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    key = settings.embedding_api_key.strip()
    if not key:
        raise ValueError("embedding_api_key 未配置")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=key,
        base_url=settings.embedding_base_url,
        check_embedding_ctx_length=False,  # 第三方模型 tiktoken 不认识,发原始文本
        max_retries=0,                     # 重试统一由 executor/调用方管理,防叠乘
        request_timeout=settings.tool_timeout_seconds,
    )
