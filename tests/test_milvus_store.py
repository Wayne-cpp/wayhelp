import pytest

from app.knowledge.milvus_store import COLLECTION, MilvusKnowledgeStore


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "test_milvus.db"), dim=4)
    yield s
    s.close()


def test_file_not_exists_initially(store):
    assert store.file_exists() is False


def test_ensure_creates_and_contract_ok(store):
    store.ensure_collection()
    assert store.has_collection() is True
    store.ensure_collection()  # 幂等,重复调用不炸


def test_upsert_search_roundtrip(store):
    store.ensure_collection()
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0], "智能猫砂盆 Pro 猫砂容量 9L", "product_spec"),
                  (2, [0.0, 1.0, 0.0, 0.0], "自动饮水机 水箱容量 2L", "product_spec")])
    hits = store.search_dense([0.9, 0.1, 0.0, 0.0], top_k=2)
    assert hits[0][0] == 1  # 最相似的是 id=1
    assert 0.0 < hits[0][1] <= 1.0
    assert {h[0] for h in hits} == {1, 2}


def test_upsert_same_id_replaces(store):
    store.ensure_collection()
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0], "文本甲", "faq")])
    store.upsert([(1, [0.0, 1.0, 0.0, 0.0], "文本甲", "faq")])  # 同 id 覆盖,不出重复向量
    assert store.all_ids() == {1}
    hits = store.search_dense([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert hits[0][0] == 1 and hits[0][1] > 0.9


def test_all_ids_empty_collection(store):
    assert store.all_ids() == set()
    store.ensure_collection()
    assert store.all_ids() == set()


def test_num_entities(store):
    assert store.num_entities() == 0  # 集合不存在 → 0,不建文件外的东西
    store.ensure_collection()
    assert store.num_entities() == 0
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0], "文本甲", "faq"),
                  (2, [0.0, 1.0, 0.0, 0.0], "文本乙", "faq")])
    assert store.num_entities() == 2
    store.upsert([(1, [0.0, 1.0, 0.0, 0.0], "文本甲", "faq")])  # 覆盖不增量
    assert store.num_entities() == 2


def test_delete_by_ids(store):
    store.ensure_collection()
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0], "文本甲", "faq"),
                  (2, [0.0, 1.0, 0.0, 0.0], "文本乙", "faq"),
                  (3, [0.0, 0.0, 1.0, 0.0], "文本丙", "policy")])
    store.delete_by_ids([1, 3])
    assert store.all_ids() == {2}
    assert store.num_entities() == 1
    store.delete_by_ids([])     # 空列表不炸
    store.delete_by_ids([999])  # 不存在的 id 不炸
    assert store.all_ids() == {2}


def test_delete_by_ids_without_collection(store):
    store.delete_by_ids([1, 2])  # 集合不存在:静默不炸(对空库 reset 场景)
    assert store.all_ids() == set()


def test_contract_mismatch_dim_rejected(store):
    store.ensure_collection()  # dim=4
    wrong = MilvusKnowledgeStore(store._uri, dim=8)
    with pytest.raises(ValueError, match="维度|dim"):
        wrong.ensure_collection()
    wrong.close()


def test_close_and_reopen(tmp_path):
    uri = str(tmp_path / "reopen.db")
    s1 = MilvusKnowledgeStore(uri, dim=4)
    s1.ensure_collection()
    s1.upsert([(7, [1.0, 0.0, 0.0, 0.0], "文本甲", "faq")])
    s1.close()
    s2 = MilvusKnowledgeStore(uri, dim=4)
    assert s2.all_ids() == {7}
    s2.close()


def test_ensure_collection_creates_missing_parent_dir(tmp_path):
    """全新环境 ./data 不存在时也能建库(review Critical #1)。"""
    s = MilvusKnowledgeStore(str(tmp_path / "missing" / "sub" / "milvus.db"), dim=4)
    try:
        s.ensure_collection()
        assert s.has_collection() is True
    finally:
        s.close()


def test_dotenv_milvus_uri_does_not_break_lite(tmp_path):
    """CWD/.env 带 MILVUS_URI(我们文档要求用户配置)时,pymilvus import 期
    load_dotenv 会把它烧进 Config.MILVUS_URI,Lite 本地路径被判非法;
    _load_milvus_client 必须连同 Config 一起隔离。子进程模拟真实 CLI 冷启动。"""
    import os
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path / ".env").write_text("MILVUS_URI=./data/milvus_lite.db\n", encoding="utf-8")
    repo = Path(__file__).resolve().parent.parent
    script = (
        "import os\n"
        "from app.knowledge.milvus_store import MilvusKnowledgeStore\n"
        "s = MilvusKnowledgeStore('kb.db', dim=4)\n"
        "s.ensure_collection()\n"
        "assert s.has_collection()\n"
        "assert 'MILVUS_URI' not in os.environ  # .env 污染已回滚\n"
        "s.close()\n"
        "print('OK')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(repo)}
    r = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "OK" in r.stdout


def test_reopen_after_unclean_exit_recovers(tmp_path):
    """崩溃(未 close)后重开:Lite 集合处于 released,store 读写前必须自动 load。"""
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    db = tmp_path / "crash.db"
    env = {**os.environ, "PYTHONPATH": str(repo)}
    p1 = (
        "from app.knowledge.milvus_store import MilvusKnowledgeStore\n"
        f"s = MilvusKnowledgeStore({str(db)!r}, dim=4)\n"
        "s.ensure_collection()\n"
        "s.upsert([(7, [1.0, 0.0, 0.0, 0.0], '文本甲', 'faq')])\n"
        "import os; os._exit(0)\n"  # 模拟崩溃:不 close
    )
    r1 = subprocess.run([sys.executable, "-c", p1], env=env, cwd=tmp_path,
                        capture_output=True, text=True, timeout=120)
    assert r1.returncode == 0, r1.stderr[-300:]
    p2 = (
        "from app.knowledge.milvus_store import MilvusKnowledgeStore\n"
        f"s = MilvusKnowledgeStore({str(db)!r}, dim=4)\n"
        "assert s.has_collection()\n"
        "assert s.all_ids() == {7}  # released 状态下 query 会炸,须自动 load\n"
        "assert s.search_dense([1.0, 0.0, 0.0, 0.0], top_k=1)[0][0] == 7\n"
        "s.upsert([(8, [0.0, 1.0, 0.0, 0.0], '文本乙', 'faq')])\n"
        "assert s.all_ids() == {7, 8}\n"
        "s.close()\n"
        "print('OK')\n"
    )
    r2 = subprocess.run([sys.executable, "-c", p2], env=env, cwd=tmp_path,
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr[-500:]
    assert "OK" in r2.stdout


def test_new_schema_bm25_and_hybrid(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "hybrid.db"), dim=4)
    try:
        s.ensure_collection()  # 建五列 schema
        s.upsert([
            (1, [1, 0, 0, 0], "智能猫砂盆 Pro 型号 MH-LP100 猫砂容量 9L 活性炭除臭", "product_spec"),
            (2, [0, 1, 0, 0], "自动饮水机 型号 MH-W20 水箱容量 2L 三重过滤棉", "product_spec"),
            (3, [0, 0, 1, 0], "退货政策 7 天无理由 不影响二次销售", "policy"),
        ])
        bm = s.search_bm25("MH-LP100 猫砂容量", 3)
        assert bm[0][0] == 1                       # BM25 命中型号
        dense = s.search_dense([0, 0, 1, 0], 3)
        assert dense[0][0] == 3
        hy = s.hybrid([0, 0, 1, 0], "退货", 3)
        assert hy[0][0] == 3 and hy[0][1] > 0      # RRF 分数为正(量级 ~0.03)
        scoped = s.search_bm25("退货 无理由", 3, scope="policy")
        assert [i for i, _ in scoped] == [3]       # 过滤只剩 policy
        assert s.search_bm25("退货 无理由", 3, scope="product_spec") == []
        with pytest.raises(ValueError, match="scope"):
            s.search_bm25("x", 1, scope="; DROP")  # 非枚举白名单直接拒
    finally:
        s.close()


def test_legacy_two_column_schema_rejected(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "legacy.db"), dim=4)
    try:
        cli = s._cli()
        cli.create_collection(collection_name="knowledge", dimension=4,
                              metric_type="COSINE", auto_id=False,
                              enable_dynamic_field=False)  # 旧两列快捷形态
        with pytest.raises(ValueError, match="重建索引"):
            s.ensure_collection()
    finally:
        s.close()


def test_recreate_cycle(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "re.db"), dim=4)
    try:
        s.ensure_collection()
        s.upsert([(1, [1, 0, 0, 0], "文本甲", "faq")])
        assert s.all_ids() == {1}
        s.recreate()                       # drop + 新 schema + load,_loaded 复位正确
        assert s.all_ids() == set()
        s.upsert([(2, [0, 1, 0, 0], "文本乙", "faq")])
        assert [i for i, _ in s.search_dense([0, 1, 0, 0], 1)] == [2]
    finally:
        s.close()


def test_cli_concurrent_first_access_builds_single_client(monkeypatch):
    """R2(ch06):_cli() 惰性初始化的并发契约——双检锁。构造慢(桩注入延迟)+
    屏障对齐并发首调,无锁时各线程各建 client,有锁时全线程共享同一实例。"""
    import threading
    import time

    import app.knowledge.milvus_store as mod

    constructed = []

    class _SlowCli:
        def __init__(self, uri):
            time.sleep(0.05)  # 拉宽竞态窗口(真实 MilvusClient 构造是 C++ IO)
            constructed.append(uri)

    monkeypatch.setattr(mod, "_load_milvus_client", lambda: _SlowCli)
    s = MilvusKnowledgeStore("unused.db", dim=4)
    barrier = threading.Barrier(4)
    got: list[object] = []
    lock = threading.Lock()

    def hit():
        barrier.wait(timeout=10)
        cli = s._cli()
        with lock:
            got.append(cli)

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)
    assert len(constructed) == 1            # 只构造一次
    assert all(g is got[0] for g in got)    # 全线程拿到同一实例


def test_concurrent_first_access_no_flock_conflict(tmp_path):
    """R2(ch06)实弹回归:refund_policy 多查询 asyncio.to_thread 并行(spec §6.4)
    撞进程首次 milvus 访问时,若 _cli() 无锁,两线程各建 MilvusClient → 同进程
    双 open file description 的 flock 互斥 → DataDirLockedError(Task 12 probe
    剧本1 T2 实锤的最小复现形态)。并发首调 has_collection() 必须全部无异常。"""
    import threading

    s = MilvusKnowledgeStore(str(tmp_path / "race.db"), dim=4)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def hit():
        try:
            barrier.wait(timeout=10)
            s.has_collection()
        except BaseException as exc:  # noqa: BLE001——竞态证据要原样收集
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    s.close()
