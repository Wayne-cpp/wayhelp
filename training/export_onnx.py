"""ONNX 导出 + PyTorch/ONNX 一致性验证 + 模型包组装(spec §7.1)。

Context7 + 锁定版本(optimum 2.1.0 + optimum-onnx 0.1.0)双核实:
optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained(ckpt,
export=True) + save_pretrained(写 model.onnx);onnxruntime InferenceSession(
path, providers=["CPUExecutionProvider"])。
CLI: uv run --group train python -m training.export_onnx \
       --ckpt models/topic_classifier/_work/best --out models/topic_classifier
一致性:eval 模式,同输入比较 logits 与 sigmoid(atol=1e-4, rtol=1e-3),
解码标签必须完全一致;不一致即导出失败,不写 manifest。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from training.labels import (DECODER_VERSION, classifier_version, decode_scores,
                             label_names, taxonomy_version)
from training.package import package_version, write_manifest
from training.preprocessing import clean_text, preprocess_config

PROBES = [  # 覆盖 batch=1/32、短句/超长句、多标签、脱敏占位符(spec §7.1)
    "买大了想退", "发票怎么开?", *[ "短句" ] * 30,
    "这是一段" * 50 + "超长输入",
]
ATOL, RTOL = 1e-4, 1e-3


def verify_export(torch_dir: Path, onnx_dir: Path, probes: list[str]) -> None:
    import onnxruntime
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(torch_dir)
    model = AutoModelForSequenceClassification.from_pretrained(torch_dir).eval()
    sess = onnxruntime.InferenceSession(str(Path(onnx_dir) / "model.onnx"),
                                        providers=["CPUExecutionProvider"])
    names = label_names()
    for probe in probes:
        inputs = tok([clean_text(probe)], max_length=64, truncation=True,
                     padding="max_length", return_tensors="np")
        with torch.no_grad():
            t_logits = model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).logits.numpy()
        o_logits = sess.run(None, dict(inputs))[0]
        np.testing.assert_allclose(o_logits, t_logits, atol=ATOL, rtol=RTOL)
        sig = 1 / (1 + np.exp(-o_logits))
        t_sig = 1 / (1 + np.exp(-t_logits))
        np.testing.assert_allclose(sig, t_sig, atol=ATOL, rtol=RTOL)
        assert decode_scores(sig[0].tolist(), 0.5, names) == \
            decode_scores(t_sig[0].tolist(), 0.5, names)


def main(ckpt: Path, out: Path) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ort = ORTModelForSequenceClassification.from_pretrained(str(ckpt), export=True)
    ort.save_pretrained(out)  # 产 model.onnx
    AutoTokenizer.from_pretrained(ckpt).save_pretrained(out / "tokenizer")
    shutil.copy(Path("training/taxonomy.json"), out / "taxonomy.json")
    threshold = json.loads((ckpt.parent / "threshold.json").read_text())["effective_threshold"]
    (out / "labels.json").write_text(json.dumps({
        "labels": label_names(), "taxonomy_version": taxonomy_version(),
        "effective_threshold": threshold, "decoder_version": DECODER_VERSION},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "preprocess.json").write_text(json.dumps({
        **preprocess_config(), "max_len": 64, "truncation": "right",
        "padding": "right-fixed-64"}, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_export(ckpt, out, PROBES)  # 不一致抛错 → 不写 manifest,不发布
    model_version = package_version(out)
    cv = classifier_version(model_version, threshold)
    write_manifest(out, {"model_version": model_version,
                         "classifier_version": cv,
                         "model_revision": ckpt.name,
                         "consistency_check": {"atol": ATOL, "rtol": RTOL,
                                               "probes": len(PROBES),
                                               "result": "pass"}})
    print(f"[export] model_version={model_version} classifier_version={cv}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(Path(args[args.index("--ckpt") + 1]), Path(args[args.index("--out") + 1]))
