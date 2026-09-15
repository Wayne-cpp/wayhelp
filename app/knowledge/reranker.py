"""bge-reranker-v2-m3 精排客户端(spec §5/§12):硅基流动 POST /v1/rerank。

连接/超时/429/408/409/5xx 内部退避重试(0.5s/1s,最多 2 次追加);401/403/其他 4xx
零重试。任何失败耗尽返回结构化降级结果(ok=False),绝不抛给 executor 触发整链重试。
"""

import logging
import time
from dataclasses import dataclass

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankOutcome:
    ok: bool
    ranking: list[tuple[int, float]]  # (候选下标, relevance_score) 降序
    note: str | None


_RETRYABLE_STATUS = {408, 409, 429}


class SiliconFlowReranker:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self._base = settings.rerank_base_url.rstrip("/")
        self._key = settings.rerank_key()
        self._model = settings.rerank_model
        self._timeout = settings.rerank_timeout_seconds
        self._max_retries = settings.rerank_max_retries
        self._client = client or httpx.Client()

    def rerank(self, query: str, documents: list[str], top_n: int,
               deadline: float | None = None) -> RerankOutcome:
        if not self._key:
            return RerankOutcome(False, [], "rerank_key_missing")
        if not documents:
            return RerankOutcome(True, [], None)
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return RerankOutcome(False, [], "budget_exhausted")
            try:
                resp = self._client.post(
                    f"{self._base}/rerank",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"model": self._model, "query": query,
                          "documents": documents, "top_n": top_n},
                    timeout=self._timeout if remaining is None
                            else min(self._timeout, max(remaining, 0.1)),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                note = f"rerank_transport:{type(exc).__name__}"
                if attempt < attempts - 1:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                logger.warning("rerank exhausted: %s", note)
                return RerankOutcome(False, [], note)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                ranking = sorted(((int(r["index"]), float(r["relevance_score"]))
                                  for r in results), key=lambda x: -x[1])
                return RerankOutcome(True, ranking[:top_n], None)
            if resp.status_code in _RETRYABLE_STATUS or resp.status_code >= 500:
                if attempt < attempts - 1:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                return RerankOutcome(False, [], f"rerank_http_{resp.status_code}")
            return RerankOutcome(False, [], f"rerank_http_{resp.status_code}_noretry")
        raise AssertionError("unreachable")
