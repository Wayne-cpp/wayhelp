"""ch10 独立推理服务(:8103):ONNX + onnxruntime CPU(spec §7.2)。

起服:make topic-inference(= uv run --group train python -m training.serve)
启动校验:包文件哈希/labels 顺序与数量/阈值合法;每个进程固定一个模型包,
换包须重启。TOPIC_LABEL_THRESHOLD 仅本进程读取(覆盖会改变 classifier_version)。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.labels import (DECODER_VERSION, classifier_version, decode_scores,
                             label_names, taxonomy_version)
from training.package import package_version

MODEL_DIR = Path("models/topic_classifier")


def load_package(model_dir) -> dict:
    pkg = Path(model_dir)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    labels_doc = json.loads((pkg / "labels.json").read_text(encoding="utf-8"))
    if labels_doc["labels"] != label_names():
        raise RuntimeError("模型包 labels 与权威类目表不一致")
    if labels_doc["taxonomy_version"] != taxonomy_version():
        raise RuntimeError("模型包 taxonomy_version 过期")
    if package_version(pkg) != manifest["model_version"]:
        raise RuntimeError("模型包文件哈希与 manifest 不符")
    th = float(labels_doc["effective_threshold"])
    if not (0 < th < 1) or not math.isfinite(th):
        raise RuntimeError("labels.json 阈值非法")
    return {"model_version": manifest["model_version"], "labels_doc": labels_doc}


class OnnxRunner:
    """真实推理:确定性清洗 → tokenizer(右截断/padding 到 64)→ onnxruntime → sigmoid。"""

    def __init__(self, model_dir):
        import onnxruntime
        from transformers import AutoTokenizer
        from training.preprocessing import clean_text
        self._clean = clean_text
        self._tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        self._sess = onnxruntime.InferenceSession(
            str(Path(model_dir) / "model.onnx"), providers=["CPUExecutionProvider"])

    def run(self, texts):
        import numpy as np
        cleaned = [self._clean(t) for t in texts]
        inputs = self._tok(cleaned, max_length=64, truncation=True,
                           padding="max_length", return_tensors="np")
        logits = self._sess.run(None, dict(inputs))[0]
        return (1 / (1 + np.exp(-logits))).tolist()


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


def create_inference_app(model_dir, *, runner=None, threshold_override=None) -> FastAPI:
    pkg = load_package(model_dir)
    labels_doc = pkg["labels_doc"]
    threshold = float(labels_doc["effective_threshold"])
    if threshold_override is not None:
        if not (0 < threshold_override < 1) or not math.isfinite(threshold_override):
            raise ValueError("TOPIC_LABEL_THRESHOLD 须为 (0,1) 有限数")
        threshold = float(threshold_override)
    cv = classifier_version(pkg["model_version"], threshold)
    names = labels_doc["labels"]
    runner = runner or OnnxRunner(model_dir)
    app = FastAPI(title="topic-inference")

    @app.get("/metadata")
    def metadata():
        return {"model_version": pkg["model_version"],
                "taxonomy_version": labels_doc["taxonomy_version"],
                "classifier_version": cv, "effective_threshold": threshold,
                "labels": names, "decoder_version": DECODER_VERSION}

    @app.post("/classify")
    def classify(req: ClassifyRequest):
        if any(not isinstance(t, str) or not t.strip() for t in req.texts):
            raise HTTPException(422, "texts 须为非空字符串数组")
        from training.preprocessing import clean_text
        if any(not clean_text(t) for t in req.texts):
            raise HTTPException(422, "清洗后为空")
        try:
            scores = runner.run(req.texts)
        except Exception as exc:
            raise HTTPException(500, f"模型推理失败:{type(exc).__name__}") from exc
        labels = [decode_scores([float(x) for x in s], threshold, names)
                  for s in scores]
        return {"labels": labels, "scores": scores, "classifier_version": cv}

    return app


def main():
    override = os.environ.get("TOPIC_LABEL_THRESHOLD")
    model_dir = Path(os.environ.get("TOPIC_MODEL_DIR", str(MODEL_DIR)))
    app = create_inference_app(
        model_dir, threshold_override=float(override) if override else None)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8103)


if __name__ == "__main__":
    main()
