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
