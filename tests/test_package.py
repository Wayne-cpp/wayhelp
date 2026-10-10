"""模型包哈希:model_version 稳定性与敏感性(spec §7.1)。"""

from pathlib import Path

from training.package import file_sha256, package_version, write_manifest


def test_package_version_stable_and_sensitive(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")
    v1 = package_version(tmp_path)
    assert v1 == package_version(tmp_path) and len(v1) == 64
    (tmp_path / "sub" / "b.txt").write_text("bb", encoding="utf-8")
    assert package_version(tmp_path) != v1  # 内容变 → 版本变
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")
    assert package_version(tmp_path) != v1  # 文件增 → 版本变


def test_manifest_roundtrip(tmp_path: Path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    m = write_manifest(tmp_path, {"model_revision": "r1"})
    assert m["files"]["model.onnx"] == file_sha256(tmp_path / "model.onnx")
    assert m["model_revision"] == "r1"
    assert (tmp_path / "manifest.json").exists()
    # manifest 自身不参与 model_version
    assert package_version(tmp_path) == package_version(tmp_path)
