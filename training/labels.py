"""17 类权威类目加载、版本哈希与统一解码(decoder_version=topic-decoder-v1)。

纯标准库:主服务 app/services/topic_classify.py 也导入本模块,
不得引入 numpy/torch 等任何第三方依赖(spec §13)。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.json"
DECODER_VERSION = "topic-decoder-v1"
OTHER = "其他"


def load_taxonomy() -> list[dict]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def label_names() -> list[str]:
    return [t["name"] for t in load_taxonomy()]


def canonical_json(obj) -> str:
    """规范化 JSON:键排序、数组保序、UTF-8、无多余空白(spec §2)。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def taxonomy_version() -> str:
    return _sha256(canonical_json(load_taxonomy()))


def classifier_version(model_version: str, effective_threshold: float,
                       decoder_version: str = DECODER_VERSION) -> str:
    return _sha256(canonical_json({
        "model_version": model_version,
        "effective_threshold": effective_threshold,
        "decoder_version": decoder_version}))


def decode_scores(scores: Sequence[float], threshold: float,
                  labels: Sequence[str] | None = None) -> list[str]:
    """统一解码:其余 16 类分数 >= threshold 全命中;有具体类命中剔除「其他」,
    零命中输出 ["其他"];按冻结顺序返回(spec §3)。"""
    names = list(labels) if labels is not None else label_names()
    if len(scores) != len(names):
        raise ValueError(f"scores 维度 {len(scores)} 与类目数 {len(names)} 不符")
    hits = [name for s, name in zip(scores, names)
            if name != OTHER and s >= threshold]
    return hits if hits else [OTHER]
