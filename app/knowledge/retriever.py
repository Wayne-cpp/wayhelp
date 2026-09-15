"""在线检索(spec §5):Query 理解 → 双路召回 → RRF → Rerank 四级管道,策略参数化。

命中(KnowledgeHit)带 chunk_id/score/source_doc/chunk_index/section_path,供评估
比对与证据组装;score 是 effective_strategy 末级排序分,只在同策略内可比。
"""

import json
import logging
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import Settings
from app.knowledge.ingest import vector_text
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.query_understanding import QueryPlan, passthrough_plan, plan_query
from app.models import KnowledgeChunk

logger = logging.getLogger(__name__)

NOTE_UNCONFIGURED = "知识检索未配置"
NOTE_NOT_BUILT = "知识库尚未建立"
NOTE_REBUILDING = "知识库正在重建,请稍后重试"
NOTE_REBUILD_REQUIRED = "知识库需要重建索引"

# 分策略置信阈值:rerank 降级为 hybrid 时阈值随 effective_strategy 换挡
_THRESHOLD_ATTR = {"dense": "knowledge_min_score", "bm25": "bm25_min_score",
                   "hybrid": "hybrid_min_score", "hybrid_rerank": "rerank_min_score"}


class RetryableKnowledgeError(Exception):
    """可重试的知识检索故障(连接/超时/限流/暂时性 HTTP/总预算耗尽),由 executor 统一重试。"""


@dataclass(frozen=True)
class KnowledgeHit:
    chunk_id: int
    score: float          # effective_strategy 的末级排序分,只在同策略内比较
    category: str
    questions: str
    answer: str
    source_doc: str | None
    chunk_index: int | None
    section_path: str | None


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[KnowledgeHit]
    requested_strategy: str
    effective_strategy: str
    confidence_score: float | None   # Top-1 末级排序分;无命中为 None
    confidence_threshold: float
    low_confidence: bool
    note: str | None
    query_plan: QueryPlan | None
    leg_counts: dict[str, int]


@dataclass(frozen=True)
class Evidence:
    ref_no: int
    chunk_id: int
    section_path: str | None
    question: str
    answer: str
    category: str

    def to_dict(self) -> dict:
        return {"ref_no": self.ref_no, "chunk_id": self.chunk_id,
                "section_path": self.section_path, "question": self.question,
                "answer": self.answer, "category": self.category}


def display_permutation(n: int) -> list[int]:
    """0-based 展示序:奇数 rank 升序 + 偶数 rank 降序(最相关钉首尾)。"""
    return list(range(0, n, 2)) + list(range(1, n, 2))[::-1]


def assemble_evidence(hits: list[KnowledgeHit], *, max_items: int,
                      budget_chars: int, overhead_chars: int) -> list[Evidence]:
    """截 max_items → 首尾排位 → 按展示序累加序列化,超预算即停(整条弃,不截字段)
    → 裁剪后按最终展示序分配 ref_no。overhead_chars 为外层 JSON 固定开销的保守预留。"""
    ordered = [hits[i] for i in display_permutation(min(len(hits), max_items))]
    used = overhead_chars
    items: list[dict] = []
    for h in ordered:
        d = {"chunk_id": h.chunk_id, "section_path": h.section_path,
             "question": h.questions, "answer": h.answer, "category": h.category}
        size = len(json.dumps(d, ensure_ascii=False)) + 2  # 逗号与括号余量
        if used + size > budget_chars:
            break
        items.append(d)
        used += size
    return [Evidence(i + 1, d["chunk_id"], d["section_path"], d["question"],
                     d["answer"], d["category"]) for i, d in enumerate(items)]


def _as_retryable(exc: Exception) -> RetryableKnowledgeError | None:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError,
                        TimeoutError, ConnectionError)):
        return RetryableKnowledgeError(type(exc).__name__)
    if isinstance(exc, APIStatusError) and (
            exc.status_code in (408, 409) or exc.status_code >= 500):
        return RetryableKnowledgeError(f"HTTP {exc.status_code}")
    return None


def _hit_text(h: KnowledgeHit) -> str:
    return vector_text(h.category, h.questions, h.answer)  # 与 BM25 text 列同款三格拼接


class KnowledgeRetriever:
    def __init__(self, settings: Settings, embed=None,
                 store: MilvusKnowledgeStore | None = None, session_factory=None,
                 model=None, reranker=None, state=None):
        self._settings = settings
        self._embed = embed
        self._store = store or MilvusKnowledgeStore(settings.milvus_uri,
                                                    settings.embedding_dim)
        self._sf = session_factory
        self._model = model
        self._reranker = reranker
        self._state = state          # KnowledgeStateHolder | None(T11)

    @property
    def enabled(self) -> bool:
        return self._embed is not None

    def close(self) -> None:
        self._store.close()

    def _threshold_for(self, strategy: str) -> float:
        return getattr(self._settings, _THRESHOLD_ATTR[strategy])

    def _state_note(self) -> str | None:
        if self._state is None:
            return None
        cur = self._state.get()
        if cur == "rebuilding":
            return NOTE_REBUILDING
        if cur == "rebuild_required":
            return NOTE_REBUILD_REQUIRED
        return None

    def search(self, query: str, *, scope: str | None = None,
               strategy: str | None = None, min_score: float | None = None,
               query_plan: QueryPlan | None = None,
               deadline: float | None = None) -> RetrievalResult:
        """四级管道。阈值不做命中过滤,只驱动 low_confidence(min_score 供评估/自测覆盖)。"""
        requested = strategy or self._settings.knowledge_strategy
        threshold = self._threshold_for(requested) if min_score is None else min_score

        state_note = self._state_note()
        if state_note is not None:
            return RetrievalResult([], requested, requested, None, threshold, True,
                                   state_note, query_plan, {})
        if not self.enabled:
            return RetrievalResult([], requested, requested, None, threshold, True,
                                   NOTE_UNCONFIGURED, query_plan, {})
        if not self._store.file_exists():
            return RetrievalResult([], requested, requested, None, threshold, True,
                                   NOTE_NOT_BUILT, query_plan, {})
        try:
            if not self._store.has_collection():
                return RetrievalResult([], requested, requested, None, threshold, True,
                                       NOTE_NOT_BUILT, query_plan, {})
        except Exception as exc:
            self._raise_if_retryable(exc)
            raise

        # model 未装配(T6 过渡/自测旁路)时静默直查,不把「no_model」配置态当降级 note;
        # model 在位才走真实改写(含 rewrite_disabled/rewrite_error 降级 note)。
        plan = query_plan or (
            passthrough_plan(query) if self._model is None else
            plan_query(self._model, query, enabled=self._settings.query_rewrite_enabled,
                       timeout_seconds=self._settings.rerank_timeout_seconds,
                       model_name=self._settings.model_name))
        if deadline is not None and time.monotonic() > deadline:
            raise RetryableKnowledgeError("budget_exhausted")

        k = self._settings.retrieval_candidate_k
        dense_text = plan.standard_query
        bm25_text = " ".join([plan.standard_query, *plan.synonyms]).strip()
        leg_counts: dict[str, int] = {}
        try:
            if requested == "dense":
                raw = self._dense_leg(dense_text, k, scope)
                leg_counts["dense"] = len(raw)
            elif requested == "bm25":
                raw = self._store.search_bm25(bm25_text, k, scope)
                leg_counts["bm25"] = len(raw)
            else:
                vector = self._embed_query(dense_text)
                raw = self._store.hybrid(vector, bm25_text, k, scope)
                leg_counts = {"dense": k, "bm25": k, "fused": len(raw)}  # 融合返回;腿级命中数不可得,记请求量+融合量
        except Exception as exc:
            self._raise_if_retryable(exc)
            raise

        effective = requested
        note = plan.note if plan.degraded else None
        hits = self._hydrate(raw)

        if requested == "hybrid_rerank" and hits:
            outcome = self._rerank(plan, hits, deadline)
            if outcome.ok:
                by_idx = list(hits)
                hits = [KnowledgeHit(by_idx[i].chunk_id, score, by_idx[i].category,
                                     by_idx[i].questions, by_idx[i].answer,
                                     by_idx[i].source_doc, by_idx[i].chunk_index,
                                     by_idx[i].section_path)
                        for i, score in outcome.ranking if i < len(by_idx)]
            else:
                effective = "hybrid"
                note = outcome.note or note
                threshold = (self._threshold_for("hybrid") if min_score is None
                             else min_score)
        top_n = self._settings.rerank_top_n
        hits = hits[:top_n]
        confidence = hits[0].score if hits else None
        low = (confidence is None) or (confidence < threshold)
        return RetrievalResult(hits, requested, effective, confidence, threshold,
                               low, note, plan, leg_counts)

    def _embed_query(self, text: str) -> list[float]:
        try:
            vector = self._embed.embed_query(text)
        except Exception as exc:
            self._raise_if_retryable(exc)
            raise
        if len(vector) != self._store.dim:
            raise ValueError(f"查询向量维度 {len(vector)} ≠ 集合维度 {self._store.dim}")
        return vector

    def _dense_leg(self, text: str, k: int, scope: str | None):
        return self._store.search_dense(self._embed_query(text), k, scope)

    def _rerank(self, plan: QueryPlan, hits: list[KnowledgeHit],
                deadline: float | None):
        if self._reranker is None:
            from app.knowledge.reranker import RerankOutcome
            return RerankOutcome(False, [], "reranker_not_configured")
        docs = [_hit_text(h) for h in hits]
        return self._reranker.rerank(plan.standard_query, docs,
                                     self._settings.rerank_top_n, deadline)

    @staticmethod
    def _raise_if_retryable(exc: Exception) -> None:
        retryable = _as_retryable(exc)
        if retryable is not None:
            raise retryable from exc

    def _hydrate(self, hits: list[tuple[int, float]]) -> list[KnowledgeHit]:
        """按 id 从 MySQL 取原文,保持检索顺序;原文被删的跳过不炸。"""
        ids = [i for i, _ in hits]
        with self._sf() as s:
            rows = {r.id: r for r in s.query(KnowledgeChunk)
                    .filter(KnowledgeChunk.id.in_(ids)).all()}
        out: list[KnowledgeHit] = []
        for chunk_id, score in hits:
            row = rows.get(chunk_id)
            if row is None:
                continue
            out.append(KnowledgeHit(chunk_id, score, row.category, row.questions,
                                    row.answer, row.source_doc, row.chunk_index,
                                    row.section_path))
        return out

    def probe(self, query: str, top_k: int, min_score: float, *,
              strategy: str | None = None, scope: str | None = None) -> dict:
        """检索自测旁路:不过滤,返回策略/腿数/Top-1/阈值/低置信标记/降级 note。"""
        res = self.search(query, scope=scope, strategy=strategy,
                          min_score=min_score, query_plan=None)
        return {"note": res.note, "requested_strategy": res.requested_strategy,
                "effective_strategy": res.effective_strategy,
                "leg_counts": res.leg_counts,
                "confidence_score": res.confidence_score,
                "confidence_threshold": res.confidence_threshold,
                "low_confidence": res.low_confidence,
                "hits": [{"chunk_id": h.chunk_id, "score": h.score,
                          "category": h.category, "questions": h.questions,
                          "answer": h.answer, "section_path": h.section_path,
                          "source_doc": h.source_doc,
                          "passed": h.score >= min_score} for h in res.hits]}
