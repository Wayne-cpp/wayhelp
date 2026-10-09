"""ch09 正式置信闸(spec §5.2):三信号合成/边界/与旧 top1 等价退化 + retriever 闸位回归。"""
import pytest

from app.knowledge.evidence_confidence import (
    UNCALIBRATED_VERSION, EvidenceGateParams, evidence_confidence, evaluate_evidence,
    gate_params_from_settings,
)
from app.knowledge.retriever import KnowledgeHit, KnowledgeRetriever
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

P = EvidenceGateParams(weight_top1=0.6, weight_count=0.2, weight_margin=0.2,
                       min_effective_score=0.05, threshold=0.4, version="t1")

# 闸位回归用「已校准」参数(非默认):0.5/0.3/0.2 权重让 count/margin 信号真正参与合成
CALIBRATED = dict(evidence_confidence_version="t9", evidence_weight_top1=0.5,
                  evidence_weight_count=0.3, evidence_weight_margin=0.2)


# ─── 纯函数:三信号合成与边界(计划 Step 1)────────────────────────────────────


def test_empty_scores_none():
    assert evidence_confidence([], P) is None


def test_single_hit_margin_full():
    # 仅一条:margin 记满 top1;count = 1/3
    assert evidence_confidence([0.8], P) == 0.6 * 0.8 + 0.2 * (1 / 3) + 0.2 * 0.8


def test_margin_uses_gap():
    a = evidence_confidence([0.8, 0.7], P)
    b = evidence_confidence([0.8, 0.1], P)
    assert b > a   # Top1 相同,分差大者分高


def test_effective_count_capped_at_three():
    a = evidence_confidence([0.8, 0.7, 0.6, 0.5], P)
    b = evidence_confidence([0.8, 0.7, 0.6], P)
    assert a == b   # 有效证据数归一按 3 封顶


def test_gate_params_from_settings_defaults_uncalibrated():
    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")
    p = gate_params_from_settings(s)
    assert p.version == UNCALIBRATED_VERSION == "uncalibrated"   # 校准前保守等价 top1
    assert p.weight_top1 == 1.0
    assert (p.weight_count, p.weight_margin, p.min_effective_score) == (0.0, 0.0, 0.0)
    assert p.threshold == 0.0553   # 与 rerank_min_score 现行冻结值同源


def test_uncalibrated_params_equivalent_to_top1_signal():
    # 默认参数下合成分退化为 Top-1 分:闸位行为与旧 top1 信号逐位等价(硬约束)
    p = gate_params_from_settings(make_settings())
    assert evidence_confidence([0.62, 0.40, 0.31], p) == pytest.approx(0.62)
    assert evidence_confidence([0.62], p) == pytest.approx(0.62)
    conf, low = evaluate_evidence([0.05], p)   # 0.05 < 0.0553 → 旧口径同判低
    assert conf == pytest.approx(0.05) and low is True


def test_evaluate_evidence_returns_confidence_and_low():
    conf, low = evaluate_evidence([0.8, 0.7], P)
    assert conf == pytest.approx(0.6 * 0.8 + 0.2 * (2 / 3) + 0.2 * 0.1)
    assert low is False   # ≥ threshold 0.4
    conf, low = evaluate_evidence([0.2], P)
    assert conf == pytest.approx(0.6 * 0.2 + 0.2 / 3 + 0.2 * 0.2)
    assert low is True    # < 0.4
    assert evaluate_evidence([], P) == (None, True)   # 无命中恒判低


# ─── retriever 闸位回归(spec §5.2/§8:两方向 + 降级臂旧阈值)──────────────────


class _ScoresReranker:
    """固定分数重排:下标 i → scores[i](调用方保证降序)。"""
    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, documents, top_n, deadline=None):
        from app.knowledge.reranker import RerankOutcome
        return RerankOutcome(True, [(i, s) for i, s in enumerate(self._scores)][:top_n],
                             None)


def _hits(n):
    return [KnowledgeHit(i + 1, 0.1 * i, "policy", f"q{i}", f"a{i}", "d.md", i, "s")
            for i in range(n)]


def _result(scores, **overrides):
    r = KnowledgeRetriever(make_settings(**overrides), embed=object(), store=object(),
                           reranker=_ScoresReranker(scores))
    res = r.rerank_candidates("退款政策", _hits(len(scores)))
    assert res is not None and res.effective_strategy == "hybrid_rerank"
    return res


def test_gate_old_top1_low_new_synthetic_usable():
    """方向①:旧 top1 判低(0.05 < 0.0553)、新合成判可用(count 满额抬分)。"""
    scores = [0.05, 0.048, 0.047]
    old = _result(scores)   # 默认 uncalibrated → 旧 top1 口径
    assert old.confidence_score == pytest.approx(0.05)
    assert old.low_confidence is True and old.confidence_threshold == 0.0553
    new = _result(scores, **CALIBRATED, evidence_min_effective_score=0.01,
                  evidence_min_confidence=0.3)
    # 0.5*0.05 + 0.3*min(3/3,1) + 0.2*0.002 = 0.3254 ≥ 0.3 → 判可用
    assert new.confidence_score == pytest.approx(0.3254)
    assert new.low_confidence is False and new.confidence_threshold == 0.3


def test_gate_old_usable_new_synthetic_low():
    """方向②:旧判可用(top1 0.9 ≥ 0.0553)、新判低(有效下限 0.95 → count=0 压分)。"""
    scores = [0.9, 0.899, 0.898]
    old = _result(scores)
    assert old.confidence_score == pytest.approx(0.9) and old.low_confidence is False
    new = _result(scores, **CALIBRATED, evidence_min_effective_score=0.95,
                  evidence_min_confidence=0.5)
    # 0.5*0.9 + 0.3*0 + 0.2*0.001 = 0.4502 < 0.5 → 判低
    assert new.confidence_score == pytest.approx(0.4502)
    assert new.low_confidence is True


# ③ 降级臂:dense/bm25/hybrid 仍用旧阈值旧口径(与是否校准无关)

class FakeEmbeddings:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]


class FailReranker:
    def rerank(self, query, documents, top_n, deadline=None):
        from app.knowledge.reranker import RerankOutcome
        return RerankOutcome(False, [], "rerank_http_500")


@pytest.fixture()
def store(tmp_path):
    from app.knowledge.milvus_store import MilvusKnowledgeStore
    s = MilvusKnowledgeStore(str(tmp_path / "gate.db"), dim=4)
    yield s
    s.close()


def _seed(sf, store):
    from app.knowledge.ingest import vector_text
    from app.models import KnowledgeChunk
    with sf() as s:
        row = KnowledgeChunk(category="商品FAQ", questions="运费怎么算",
                             answer="满 99 包邮,未满 8 元。", content_type="faq",
                             section_path="商品FAQ > 邮费怎么算",
                             source_doc="knowledge_docs/商品FAQ.md", chunk_index=1,
                             vectorize_status="done", vector_id="1")
        s.add(row)
        s.flush()
        cid = row.id
        s.commit()
    store.ensure_collection()
    store.upsert([(cid, [1.0, 0.0, 0.0, 0.0],
                   vector_text("商品FAQ", "运费怎么算", "满 99 包邮"), "faq")])
    return cid


def _retriever(settings, store, sf, reranker=None):
    return KnowledgeRetriever(settings, embed=FakeEmbeddings(), store=store,
                              session_factory=sf, reranker=reranker)


def test_degraded_arms_keep_old_thresholds_under_calibrated_gate(store, db_session_factory):
    """③:已校准参数下降级/非重排臂不受新闸影响——rerank 失败回 hybrid 旧阈值,
    dense/bm25 仍用各自旧阈值,置信分一律 Top-1。"""
    _seed(db_session_factory, store)
    s = make_settings(**CALIBRATED, evidence_min_effective_score=0.01,
                      evidence_min_confidence=0.3)
    plain = make_settings()

    r = _retriever(s, store, db_session_factory, reranker=FailReranker())
    res = r.search("邮费怎么算")
    assert res.effective_strategy == "hybrid"
    assert res.confidence_threshold == plain.hybrid_min_score   # 降级换旧阈值
    assert res.confidence_score == res.hits[0].score            # Top-1 口径,非合成分

    res = r.search("邮费怎么算", strategy="dense")
    assert res.effective_strategy == "dense"
    assert res.confidence_threshold == plain.knowledge_min_score
    assert res.confidence_score == res.hits[0].score

    res = r.search("运费怎么算", strategy="bm25")
    assert res.effective_strategy == "bm25"
    assert res.confidence_threshold == plain.bm25_min_score
    assert res.confidence_score == res.hits[0].score


def test_search_uses_calibrated_gate_for_hybrid_rerank(store, db_session_factory):
    """search 主臂同款口径:已校准时 hybrid_rerank 走合成分(1 命中 → margin 满分)。"""
    _seed(db_session_factory, store)

    class OneHitReranker:
        def rerank(self, query, documents, top_n, deadline=None):
            from app.knowledge.reranker import RerankOutcome
            return RerankOutcome(True, [(0, 0.9)], None)

    s = make_settings(**CALIBRATED, evidence_min_effective_score=0.5,
                      evidence_min_confidence=0.7)
    r = _retriever(s, store, db_session_factory, reranker=OneHitReranker())
    res = r.search("邮费怎么算")
    assert res.effective_strategy == "hybrid_rerank"
    # 0.5*0.9 + 0.3*min(1/3,1) + 0.2*0.9 = 0.73 ≥ 0.7 → 判可用(与 top1 口径 0.9 不同值)
    assert res.confidence_score == pytest.approx(0.73)
    assert res.low_confidence is False and res.confidence_threshold == 0.7


# ─── 校准模式纯函数(spec §5.2;实跑在 Task 16,此处只钉选择逻辑与产物契约)────────


def test_calibrate_evidence_gate_picks_count_separation():
    """正例三命中满额、D 桶单命中:count 信号可分离 → 约束内通过率 1.0 的解胜出。"""
    from evals.run_retrieval_compare import calibrate_evidence_gate
    samples = [
        {"should_refuse": False, "scores": [0.5, 0.49, 0.48], "recall10": 1.0},
        {"should_refuse": False, "scores": [0.5, 0.49, 0.48], "recall10": 1.0},
        {"should_refuse": True, "scores": [0.5], "recall10": 0.0},
    ]
    best = calibrate_evidence_gate(samples, max_d_pass=0.0)
    assert best is not None
    assert best["d_pass_rate"] == 0.0 and best["pass_rate"] == 1.0
    # 唯一分离维度是有效证据数(3 vs 1/3):胜者必押 count
    assert best["weights"]["count"] == pytest.approx(1.0)
    assert best["threshold"] > 1 / 3   # 拒 D:1/3 < threshold ≤ 1.0


def test_calibrate_evidence_gate_inseparable_returns_none():
    """正负分布完全相同:任何能让正例过闸的阈值同样放行 D → 无可行点 None(退出码 2)。"""
    from evals.run_retrieval_compare import calibrate_evidence_gate
    samples = [
        {"should_refuse": False, "scores": [0.9, 0.89, 0.88], "recall10": 1.0},
        {"should_refuse": True, "scores": [0.9, 0.89, 0.88], "recall10": 0.0},
    ]
    assert calibrate_evidence_gate(samples, max_d_pass=0.0) is None


def test_build_evidence_artifact_contract():
    from evals.run_retrieval_compare import build_evidence_artifact
    best = {"weights": {"top1": 0.5, "count": 0.3, "margin": 0.2},
            "min_effective_score": 0.05, "threshold": 0.4,
            "d_pass_rate": 0.05, "pass_rate": 0.93}
    a = build_evidence_artifact(best, make_settings(), "2026-10-09T00:00:00Z")
    assert set(a) == {"version", "calibrated_at", "corpus", "model", "rerank_model",
                      "weights", "min_effective_score", "threshold",
                      "d_pass_rate", "pass_rate"}
    assert a["version"] == a["calibrated_at"] == "2026-10-09T00:00:00Z"
    assert a["corpus"] == "knowledge_docs/"
    assert a["model"] == "test-model"
    assert a["rerank_model"] == "BAAI/bge-reranker-v2-m3"
    assert a["weights"] == best["weights"] and a["threshold"] == 0.4
    assert a["d_pass_rate"] == 0.05 and a["pass_rate"] == 0.93
