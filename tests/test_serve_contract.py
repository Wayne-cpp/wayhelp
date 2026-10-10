"""推理服务契约:metadata/422 语义/阈值覆盖(spec §7.2)。真实 ONNX 不进 pytest。"""

import json

import httpx
import pytest

from training.labels import classifier_version, label_names, taxonomy_version
from training.serve import create_inference_app


class FakeRunner:
    """确定性假推理:含「退」→ 退换货 0.9,含「大」→ 尺码 0.9,其余全 0.01。"""
    def run(self, texts):
        names = label_names()
        out = []
        for t in texts:
            s = [0.01] * 17
            if "退" in t:
                s[names.index("退换货")] = 0.9
            if "大" in t:
                s[names.index("尺码")] = 0.9
            out.append(s)
        return out


@pytest.fixture()
def pkg(tmp_path):
    import hashlib
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"fake")
    (d / "labels.json").write_text(json.dumps({
        "labels": label_names(), "taxonomy_version": taxonomy_version(),
        "effective_threshold": 0.5, "decoder_version": "topic-decoder-v1"}),
        encoding="utf-8")
    from training.package import package_version, write_manifest
    mv = package_version(d)
    write_manifest(d, {"model_version": mv,
                       "classifier_version": classifier_version(mv, 0.5)})
    return d


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_metadata(pkg):
    app = create_inference_app(pkg, runner=FakeRunner())
    async with await _client(app) as c:
        meta = (await c.get("/metadata")).json()
    assert meta["labels"] == label_names()
    assert meta["taxonomy_version"] == taxonomy_version()
    assert meta["effective_threshold"] == 0.5
    assert len(meta["classifier_version"]) == 64


async def test_classify_multilabel(pkg):
    app = create_inference_app(pkg, runner=FakeRunner())
    async with await _client(app) as c:
        resp = await c.post("/classify", json={"texts": ["买大了想退", "在吗"]})
    body = resp.json()
    assert body["labels"][0] == ["退换货", "尺码"]  # 冻结顺序
    assert body["labels"][1] == ["其他"]            # 零命中兜底
    assert len(body["scores"][0]) == 17


async def test_classify_validation_422(pkg):
    app = create_inference_app(pkg, runner=FakeRunner())
    async with await _client(app) as c:
        assert (await c.post("/classify", json={"texts": []})).status_code == 422
        assert (await c.post("/classify",
                             json={"texts": ["ok", "  "]})).status_code == 422
        assert (await c.post("/classify",
                             json={"texts": ["x"] * 257})).status_code == 422
        assert (await c.post("/classify",
                             json={"texts": [123]})).status_code == 422


async def test_threshold_override_changes_version(pkg):
    app = create_inference_app(pkg, runner=FakeRunner(), threshold_override=0.3)
    async with await _client(app) as c:
        meta = (await c.get("/metadata")).json()
    assert meta["effective_threshold"] == 0.3
    assert meta["classifier_version"] != classifier_version(
        json.loads((pkg / "manifest.json").read_text())["model_version"], 0.5)
    with pytest.raises(ValueError):
        create_inference_app(pkg, runner=FakeRunner(), threshold_override=1.5)
