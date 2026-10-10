"""模型包哈希与 manifest(纯标准库,serve 侧校验复用,spec §7.1)。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def package_version(pkg_dir, exclude=("manifest.json",)) -> str:
    pkg = Path(pkg_dir)
    entries = []
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name not in exclude:
            rel = p.relative_to(pkg).as_posix()
            entries.append([rel, file_sha256(p)])
    return hashlib.sha256(_canon(entries).encode("utf-8")).hexdigest()


def write_manifest(pkg_dir, extra: dict) -> dict:
    pkg = Path(pkg_dir)
    files = {p.relative_to(pkg).as_posix(): file_sha256(p)
             for p in sorted(pkg.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    manifest = {"files": files, **extra}
    (pkg / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
