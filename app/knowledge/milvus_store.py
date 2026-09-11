"""Milvus Lite 集合管理(spec §4.4)。只存 id + vector 两列,原文在 MySQL。"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymilvus import MilvusClient

COLLECTION = "knowledge"


def _load_milvus_client() -> type["MilvusClient"]:
    """惰性导入并隔离 pymilvus 的 import 副作用(实测 pymilvus 3.0.1)。

    pymilvus 的 __init__ 会执行 load_dotenv()(override=False),把 CWD 下 .env
    的键写进 os.environ —— 库 import 不应改进程环境(测试进程里会污染
    Settings(_env_file=None) 的读取,如 STRUCTURED_OUTPUT_METHOD)。导入后回滚
    新增键,恢复进程原有环境。
    """
    before = set(os.environ)
    from pymilvus import MilvusClient
    for key in set(os.environ) - before:
        os.environ.pop(key, None)
    return MilvusClient


class MilvusKnowledgeStore:
    def __init__(self, uri: str, dim: int):
        self._uri = uri
        self._dim = dim
        self._client: "MilvusClient | None" = None

    def _cli(self) -> "MilvusClient":
        if self._client is None:
            self._client = _load_milvus_client()(uri=self._uri)
        return self._client

    def file_exists(self) -> bool:
        return Path(self._uri).exists()

    @property
    def dim(self) -> int:
        return self._dim

    def has_collection(self) -> bool:
        return self._cli().has_collection(COLLECTION)

    def ensure_collection(self) -> None:
        """不存在则按契约创建;已存在则校验主键/维度,不符报错(spec §6 不得自动重建)。"""
        # Milvus Lite 打开本地文件时不会自建父目录,必须先于 client 创建
        Path(self._uri).parent.mkdir(parents=True, exist_ok=True)
        cli = self._cli()
        if not cli.has_collection(COLLECTION):
            cli.create_collection(collection_name=COLLECTION, dimension=self._dim,
                                  metric_type="COSINE", auto_id=False,
                                  enable_dynamic_field=False)
            return
        info = cli.describe_collection(COLLECTION)
        fields = {f["name"]: f for f in info["fields"]}
        pk = fields.get("id") or {}
        if not pk.get("is_primary"):
            raise ValueError("knowledge 集合契约不符: id 不是主键")
        vec = fields.get("vector") or {}
        actual_dim = (vec.get("params") or {}).get("dim", vec.get("dimension"))
        if actual_dim != self._dim:
            raise ValueError(f"knowledge 集合维度不符: 期望 {self._dim},实际 {actual_dim}")

    def upsert(self, rows: list[tuple[int, list[float]]]) -> None:
        if not rows:
            return
        self._cli().upsert(COLLECTION, [{"id": i, "vector": v} for i, v in rows])

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """返回 [(chunk_id, cosine_similarity)],按相似度降序。"""
        res = self._cli().search(COLLECTION, data=[vector], limit=top_k)
        return [(h["id"], h["distance"]) for h in res[0]]

    def all_ids(self) -> set[int]:
        if not self.has_collection():
            return set()
        rows = self._cli().query(COLLECTION, filter="id >= 0", output_fields=["id"])
        return {r["id"] for r in rows}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
