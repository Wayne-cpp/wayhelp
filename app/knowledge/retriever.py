"""在线语义检索(spec §8):query → embed → Milvus Top-K → 阈值过滤 → 回 MySQL 取原文。

内部命中记录(KnowledgeHit)带 chunk_id/score/source_doc/chunk_index,供评估比对;
query_faq 只投影 question/answer/category 三键。
"""

import logging
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import Settings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import KnowledgeChunk

logger = logging.getLogger(__name__)

NOTE_UNCONFIGURED = "知识检索未配置"
NOTE_NOT_BUILT = "知识库尚未建立"


class RetryableKnowledgeError(Exception):
    """可重试的知识检索故障(连接/超时/限流/暂时性 HTTP),由 executor 统一重试。"""


@dataclass(frozen=True)
class KnowledgeHit:
    chunk_id: int
    score: float
    category: str
    questions: str
    answer: str
    source_doc: str | None
    chunk_index: int | None


def _as_retryable(exc: Exception) -> RetryableKnowledgeError | None:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError,
                        TimeoutError, ConnectionError)):
        return RetryableKnowledgeError(type(exc).__name__)
    if isinstance(exc, APIStatusError) and (
            exc.status_code in (408, 409) or exc.status_code >= 500):
        return RetryableKnowledgeError(f"HTTP {exc.status_code}")
    return None


class KnowledgeRetriever:
    def __init__(self, settings: Settings, embed=None,
                 store: MilvusKnowledgeStore | None = None, session_factory=None):
        self._settings = settings
        self._embed = embed
        self._store = store or MilvusKnowledgeStore(settings.milvus_uri,
                                                    settings.embedding_dim)
        self._sf = session_factory

    @property
    def enabled(self) -> bool:
        return self._embed is not None

    def close(self) -> None:
        self._store.close()

    def search(self, query: str,
               min_score: float | None = None) -> tuple[list[KnowledgeHit], str | None]:
        """→ (hits, note)。note 非空 = 降级/未建库;hits 空且 note 空 = 正常无命中。
        min_score 仅供评估覆盖阈值;在线调用不传。"""
        threshold = self._settings.knowledge_min_score if min_score is None else min_score
        if not self.enabled:
            return [], NOTE_UNCONFIGURED
        if not self._store.file_exists():
            return [], NOTE_NOT_BUILT
        try:
            vector = self._embed.embed_query(query)
        except Exception as exc:
            retryable = _as_retryable(exc)
            if retryable is not None:
                raise retryable from exc
            raise
        if len(vector) != self._store._dim:
            raise ValueError(f"查询向量维度 {len(vector)} ≠ 集合维度 {self._store._dim}")
        try:
            if not self._store.has_collection():
                return [], NOTE_NOT_BUILT
            raw = self._store.search(vector, self._settings.knowledge_top_k)
        except Exception as exc:
            retryable = _as_retryable(exc)
            if retryable is not None:
                raise retryable from exc
            raise
        hits = [(i, d) for i, d in raw if d >= threshold]
        if not hits:
            return [], None
        ids = [i for i, _ in hits]
        with self._sf() as s:
            rows = {r.id: r for r in s.query(KnowledgeChunk)
                    .filter(KnowledgeChunk.id.in_(ids)).all()}
        out: list[KnowledgeHit] = []
        for chunk_id, score in hits:  # 保持 Milvus 相似度顺序
            row = rows.get(chunk_id)
            if row is None:
                continue  # 向量在而原文被删:跳过不炸
            out.append(KnowledgeHit(chunk_id, score, row.category, row.questions,
                                    row.answer, row.source_doc, row.chunk_index))
        return out, None
