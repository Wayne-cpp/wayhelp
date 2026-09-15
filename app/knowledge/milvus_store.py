"""Milvus Lite 集合管理(spec §4.4)。五列 schema:id/vector/text/sparse/scope,
BM25 由 Milvus 原生 Function 生成 sparse 列,原文权威源在 MySQL。"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from app.knowledge.scope import SCOPES

if TYPE_CHECKING:
    from pymilvus import MilvusClient

COLLECTION = "knowledge"


def _scope_filter(scope: str | None) -> str | None:
    """Lite 实测不支持 expr_params/$占位符;scope 值经 SCOPES 白名单校验后字面插值,
    枚举外的值在这里就拒掉,等效防注入(spec §4.1 意图)。"""
    if scope is None:
        return None
    if scope not in SCOPES:
        raise ValueError(f"未知 scope: {scope!r}(合法值 {SCOPES})")
    return f'scope == "{scope}"'


def _load_milvus_client() -> type["MilvusClient"]:
    """惰性导入并隔离 pymilvus 的 import 副作用(实测 pymilvus 3.0.1)。

    pymilvus/settings.py 在 import 期执行 load_dotenv()(override=False),把
    CWD 下 .env 的键写进 os.environ;且 pymilvus/orm/connections.py:596 在
    import 期就实例化 Connections 单例,按远程 URI 解析 Config.MILVUS_URI。
    我们的 .env 恰恰要求用户配置 MILVUS_URI=./data/milvus_lite.db(本地相对
    路径),不设防则 import 本身即炸 "Illegal uri"。

    对策:import 前把 MILVUS_URI 占位为 ""(override=False 不再被 .env 覆盖,
    Config 烧成 ""),import 后回滚全部新增键并复位 Config。本应用只走
    MilvusClient(uri=...) 显式传参,不用 legacy env 连接配置。
    """
    before = set(os.environ)
    had_uri = "MILVUS_URI" in os.environ
    saved_uri = os.environ.get("MILVUS_URI")
    os.environ["MILVUS_URI"] = ""
    try:
        from pymilvus import MilvusClient
        from pymilvus.settings import Config
    finally:
        for key in set(os.environ) - before:
            os.environ.pop(key, None)
        if had_uri:
            os.environ["MILVUS_URI"] = saved_uri
    Config.MILVUS_URI = ""
    Config.LEGACY_URI = ""
    return MilvusClient


class MilvusKnowledgeStore:
    def __init__(self, uri: str, dim: int):
        self._uri = uri
        self._dim = dim
        self._client: "MilvusClient | None" = None
        self._loaded = False

    def _cli(self) -> "MilvusClient":
        if self._client is None:
            self._client = _load_milvus_client()(uri=self._uri)
        return self._client

    def _ensure_loaded(self) -> None:
        """Lite 崩溃/未 close 退出后重开,集合处于 released;读写前必须显式 load。"""
        if not self._loaded:
            if self._cli().has_collection(COLLECTION):
                self._cli().load_collection(COLLECTION)
            self._loaded = True

    def file_exists(self) -> bool:
        return Path(self._uri).exists()

    @property
    def dim(self) -> int:
        return self._dim

    def has_collection(self) -> bool:
        return self._cli().has_collection(COLLECTION)

    def ensure_collection(self) -> None:
        """不存在则按契约创建;已存在则校验契约,不符报错(spec §6 不得自动重建)。"""
        # Milvus Lite 打开本地文件时不会自建父目录,必须先于 client 创建
        Path(self._uri).parent.mkdir(parents=True, exist_ok=True)
        cli = self._cli()
        if not cli.has_collection(COLLECTION):
            self._create_with_schema(cli)
        else:
            self._verify_contract(cli)
        cli.load_collection(COLLECTION)
        self._loaded = True

    def _create_with_schema(self, cli) -> None:
        _load_milvus_client()  # 幂等;保证 import 副作用隔离发生在符号 import 前
        from pymilvus import DataType, Function, FunctionType
        schema = cli.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("text", DataType.VARCHAR, max_length=4096,
                         enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("scope", DataType.VARCHAR, max_length=32)
        schema.add_function(Function(
            name="bm25_fn", function_type=FunctionType.BM25,
            input_field_names=["text"], output_field_names=["sparse"]))
        index_params = cli.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX",
                               metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25")
        cli.create_collection(COLLECTION, schema=schema, index_params=index_params)

    def _verify_contract(self, cli) -> None:
        info = cli.describe_collection(COLLECTION)
        fields = {f["name"]: f for f in info["fields"]}
        missing = {"id", "vector", "text", "sparse", "scope"} - set(fields)
        if missing:
            raise ValueError(
                f"knowledge 集合契约不符(缺列 {sorted(missing)}):需重建索引")
        if not fields["id"].get("is_primary"):
            raise ValueError("knowledge 集合契约不符: id 不是主键")
        vec = fields["vector"]
        actual_dim = (vec.get("params") or {}).get("dim", vec.get("dimension"))
        if actual_dim != self._dim:
            raise ValueError(
                f"knowledge 集合维度不符: 期望 {self._dim},实际 {actual_dim}")
        fnames = {f.get("name") for f in info.get("functions", [])}
        if "bm25_fn" not in fnames:
            raise ValueError("knowledge 集合契约不符: 缺 BM25 函数,需重建索引")

    def contract_error(self) -> str | None:
        """只读契约探针:集合不存在返回 None(未建库走 NOTE_NOT_BUILT 语义);
        存在但契约不符返回错误文案;集合不存在时不创建集合。"""
        if not self._cli().has_collection(COLLECTION):
            return None
        try:
            self._verify_contract(self._cli())
        except ValueError as exc:
            return str(exc)
        return None

    def upsert(self, rows: list[tuple[int, list[float], str, str]]) -> None:
        if not rows:
            return
        self._ensure_loaded()
        self._cli().upsert(COLLECTION, [
            {"id": i, "vector": v, "text": t, "scope": sc} for i, v, t, sc in rows])

    def search_dense(self, vector: list[float], top_k: int,
                     scope: str | None = None) -> list[tuple[int, float]]:
        self._ensure_loaded()
        kw: dict = {}
        f = _scope_filter(scope)
        if f:
            kw["filter"] = f
        res = self._cli().search(COLLECTION, data=[vector], limit=top_k, **kw)
        return [(h["id"], h["distance"]) for h in res[0]]

    def search_bm25(self, text: str, top_k: int,
                    scope: str | None = None) -> list[tuple[int, float]]:
        self._ensure_loaded()
        kw = {"anns_field": "sparse", "search_params": {"metric_type": "BM25"}}
        f = _scope_filter(scope)
        if f:
            kw["filter"] = f
        res = self._cli().search(COLLECTION, data=[text], limit=top_k, **kw)
        return [(h["id"], h["distance"]) for h in res[0]]

    def hybrid(self, vector: list[float], text: str, top_k: int,
               scope: str | None = None) -> list[tuple[int, float]]:
        """dense + BM25 双路,RRF(k=60) 融合;返回 [(chunk_id, rrf_score)]。"""
        self._ensure_loaded()
        _load_milvus_client()
        from pymilvus import AnnSearchRequest, RRFRanker
        f = _scope_filter(scope)
        dreq = AnnSearchRequest(data=[vector], anns_field="vector",
                                param={"metric_type": "COSINE"}, limit=top_k, expr=f)
        breq = AnnSearchRequest(data=[text], anns_field="sparse",
                                param={"metric_type": "BM25"}, limit=top_k, expr=f)
        res = self._cli().hybrid_search(COLLECTION, [dreq, breq], RRFRanker(k=60),
                                        limit=top_k)
        return [(h["id"], h["distance"]) for h in res[0]]

    def drop_collection(self) -> None:
        if self._cli().has_collection(COLLECTION):
            self._cli().drop_collection(COLLECTION)
        self._loaded = False

    def recreate(self) -> None:
        self.drop_collection()
        self.ensure_collection()

    def all_ids(self) -> set[int]:
        if not self.has_collection():
            return set()
        self._ensure_loaded()
        rows = self._cli().query(COLLECTION, filter="id >= 0", output_fields=["id"])
        return {r["id"] for r in rows}

    def num_entities(self) -> int:
        if not self.has_collection():
            return 0
        self._ensure_loaded()
        stats = self._cli().get_collection_stats(COLLECTION)
        return int(stats["row_count"])

    def delete_by_ids(self, ids: list[int]) -> None:
        """选择性重建用;集合不存在或 ids 空时不做事。"""
        if not ids or not self.has_collection():
            return
        self._ensure_loaded()
        self._cli().delete(COLLECTION, ids=list(ids))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._loaded = False
