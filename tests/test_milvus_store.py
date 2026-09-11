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
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])])
    hits = store.search([0.9, 0.1, 0.0, 0.0], top_k=2)
    assert hits[0][0] == 1  # 最相似的是 id=1
    assert 0.0 < hits[0][1] <= 1.0
    assert {h[0] for h in hits} == {1, 2}


def test_upsert_same_id_replaces(store):
    store.ensure_collection()
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0])])
    store.upsert([(1, [0.0, 1.0, 0.0, 0.0])])  # 同 id 覆盖,不出重复向量
    assert store.all_ids() == {1}
    hits = store.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert hits[0][0] == 1 and hits[0][1] > 0.9


def test_all_ids_empty_collection(store):
    assert store.all_ids() == set()
    store.ensure_collection()
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
    s1.upsert([(7, [1.0, 0.0, 0.0, 0.0])])
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
        "s.upsert([(7, [1.0, 0.0, 0.0, 0.0])])\n"
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
        "assert s.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0][0] == 7\n"
        "s.upsert([(8, [0.0, 1.0, 0.0, 0.0])])\n"
        "assert s.all_ids() == {7, 8}\n"
        "s.close()\n"
        "print('OK')\n"
    )
    r2 = subprocess.run([sys.executable, "-c", p2], env=env, cwd=tmp_path,
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr[-500:]
    assert "OK" in r2.stdout
