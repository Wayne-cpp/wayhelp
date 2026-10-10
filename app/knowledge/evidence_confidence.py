"""ch09 正式版证据置信闸(spec §5.2):Top1 分/有效证据数/Top1-Top2 分差三信号合成。

权重、有效分下限、阈值全部由 evals/run_retrieval_compare.py --calibrate-evidence
在 calibration split 上校准后冻结进 Settings 默认值 + evals/calibration/
evidence_confidence.json(版本控制);本模块不做任何自适应。
默认 uncalibrated 参数(1.0/0.0/0.0/0.0/0.0553)下合成分退化为 Top-1 分,
闸位行为与旧 top1 信号等价——校准落地(Task 16)前不换闸。
"""

from dataclasses import dataclass

from app.config import Settings

UNCALIBRATED_VERSION = "uncalibrated"


@dataclass(frozen=True)
class EvidenceGateParams:
    weight_top1: float
    weight_count: float
    weight_margin: float
    min_effective_score: float
    threshold: float
    version: str


def gate_params_from_settings(settings: Settings) -> EvidenceGateParams:
    return EvidenceGateParams(
        weight_top1=settings.evidence_weight_top1,
        weight_count=settings.evidence_weight_count,
        weight_margin=settings.evidence_weight_margin,
        min_effective_score=settings.evidence_min_effective_score,
        threshold=settings.rerank_evidence_min_confidence,
        version=settings.evidence_confidence_version)


def evidence_confidence(scores: list[float], params: EvidenceGateParams) -> float | None:
    """降序精排分 → 0~1 合成置信分;无命中 None。仅一条时 margin 记 top1。"""
    if not scores:
        return None
    top1 = scores[0]
    margin = top1 - scores[1] if len(scores) > 1 else top1
    effective = sum(1 for s in scores if s >= params.min_effective_score)
    return (params.weight_top1 * top1
            + params.weight_count * min(effective / 3.0, 1.0)
            + params.weight_margin * margin)


def evaluate_evidence(scores: list[float],
                      params: EvidenceGateParams) -> tuple[float | None, bool]:
    """retriever 闸位入口(spec §5.2):一并返回 (confidence, low_confidence)。"""
    confidence = evidence_confidence(scores, params)
    low = confidence is None or confidence < params.threshold
    return confidence, low
