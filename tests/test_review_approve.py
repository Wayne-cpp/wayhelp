"""ch09 审核写回:冻结→写入中→通过;并发/重试/驳回约束;reset 保留 review 块。

approve 锁序固定 知识库锁 → review 行锁;向量化阶段已释放 DB 行锁但继续持知识库锁
(外部 embedding 调用不持行锁);知识库忙 → 409 job_busy 不改审核项;非 READY → 409。
reset 只清 knowledge_docs/ 命名空间;rebuild 持锁重放(通过,写入中)。
端点级:POST /api/review/{id}/approve 200/409 与错误契约。
"""

import httpx
import pytest

from app.knowledge.state import KnowledgeState, KnowledgeStateHolder
from app.main import AppRuntime, create_app
from app.models import KnowledgeChunk, ReviewQueue
from app.services import kb_admin, review_service
from app.services.review_service import (
    ReviewConflictError,
    ReviewKbNotReadyError,
    ReviewWriteError,
)
from tests.conftest import FakeStreamModel, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class _FakeEmbed:
    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _BoomEmbed:
    def embed_documents(self, texts):
        raise RuntimeError("embedding api down")


class _FlakyEmbed:
    """第 2 次 embed_documents 调用失败:第 1 次是 rebuild 重灌文档块,
    第 2 次是重放审核块——把失败精确钉在重放阶段,下次调用恢复。"""

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("replay embed down")
        return [[0.1] * 1024 for _ in texts]


class _FakeStore:
    """内存版 store 协议:approve 只需 upsert/all_ids/delete_by_ids;
    rebuild 路径另需 drop_collection/search_bm25(texts 记原文)。"""

    def __init__(self):
        self.dim = 1024
        self.ids = set()
        self.texts = {}
        self.dropped = False

    def ensure_collection(self):
        pass

    def upsert(self, payloads):
        for p in payloads:
            self.ids.add(p[0])
            self.texts[p[0]] = p[2]

    def all_ids(self):
        return set(self.ids)

    def delete_by_ids(self, ids):
        self.ids -= set(ids)
        for i in ids:
            self.texts.pop(i, None)

    def drop_collection(self):
        self.dropped = True
        self.ids.clear()
        self.texts.clear()

    def search_bm25(self, text, top_k, scope=None):
        terms = [t for t in text.split() if t]
        hits = [(i, 1.0) for i, t in self.texts.items() if any(x in t for x in terms)]
        return sorted(hits, key=lambda t: -t[1])[:top_k]


# 四段各 ~285 字:默认 max_chunk_chars=500 下必出 4 块,钉 chunk_index/prev/next
LONG_ANSWER = "\n\n".join(f"第{i}段核准正文:" + "长" * 270 + "。" for i in range(1, 5))

# rebuild 终验 BM25 smoke 依赖语料含 MH-LP100(与 test_kb_api 同款约定)
DOC_R_SPEC = """---
type: manual
---
# 规格手册

## 猫窝(型号 MH-LP100)

MH-LP100 猫窝支持机洗。
"""


def _settings():
    from app.config import Settings
    return Settings(_env_file=None, openai_base_url="http://x", openai_api_key="k",
                    model_name="m", database_url="mysql+pymysql://u:p@h/d")


def _seed(sf, **kw):
    base = dict(normalized_question="如何开发票?", ai_suggested_answer="示例答案")
    base.update(kw)
    with sf() as s:
        r = ReviewQueue(**base)
        s.add(r)
        s.commit()
        return r.id


# ─── approve 主链路 ──────────────────────────────────────────────────────────


def test_approve_full_cycle(db_session_factory):
    rid = _seed(db_session_factory)
    store = _FakeStore()
    out = review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                 store, rid, LONG_ANSWER,
                                 state=KnowledgeStateHolder())  # READY 不拦截
    assert out["review_status"] == "通过"
    with db_session_factory() as s:
        r = s.get(ReviewQueue, rid)
        assert r.approved_answer == LONG_ANSWER and r.approved_at is not None
        chunks = sorted(s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all(), key=lambda c: c.chunk_index)
        assert len(chunks) == 4
        assert [c.chunk_index for c in chunks] == [1, 2, 3, 4]  # 稳定从 1 起
        assert all(c.vectorize_status == "done" for c in chunks)
        assert all(c.content_type == "faq" and c.category == "审核补充" for c in chunks)
        assert all(c.questions == "如何开发票?" for c in chunks)  # 问题取 normalized
        by_id = {c.id: c for c in chunks}
        for c in chunks:  # 冻结块 prev/next 互指成链
            if c.next_chunk_id is not None:
                assert by_id[c.next_chunk_id].prev_chunk_id == c.id
        assert r.knowledge_chunk_ids == [c.id for c in chunks]
    assert store.ids >= {c.id for c in chunks}   # 向量已 upsert


def test_approve_conflict_rules(db_session_factory):
    rid = _seed(db_session_factory)
    with pytest.raises(ReviewConflictError):   # 首次通过必须给答案
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid, None)
    review_service.reject(db_session_factory, rid)
    with pytest.raises(ReviewConflictError):   # 驳回不可 approve
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid, "答")
    rid2 = _seed(db_session_factory)
    review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                           _FakeStore(), rid2, "答A")
    with pytest.raises(ReviewConflictError):   # 已通过改答案 409
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid2, "答B")
    out = review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                 _FakeStore(), rid2, "答A")   # 同答案幂等
    assert out["review_status"] == "通过"
    with pytest.raises(ReviewConflictError):   # 通过后不可驳回:竞争只有一种结果
        review_service.reject(db_session_factory, rid2)


def test_approve_busy_returns_409_without_state_change(db_session_factory):
    """知识库忙:approve 非阻塞取锁失败即 409 job_busy,不改审核项,锁释放后可重试。"""
    from app.services.kb_admin import JobBusyError
    rid = _seed(db_session_factory)
    with kb_admin.job_lock():   # 模拟 build/vectorize/mine/reset/rebuild/manual_ingest 进行中
        with pytest.raises(JobBusyError) as ei:
            review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                   _FakeStore(), rid, "核准答案")
        assert ei.value.status == 409 and ei.value.code == "job_busy"
    with db_session_factory() as s:
        r = s.get(ReviewQueue, rid)
        assert r.review_status == "待审" and r.approved_answer is None
        assert s.query(KnowledgeChunk).count() == 0   # 竞争者未动任何数据
    out = review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                 _FakeStore(), rid, "核准答案")
    assert out["review_status"] == "通过"


def test_approve_rejected_when_kb_not_ready(db_session_factory):
    """知识库非 READY(rebuild_required 等):approve 409,由 rebuild 恢复全局一致性。"""
    rid = _seed(db_session_factory)
    holder = KnowledgeStateHolder(initial=KnowledgeState.REBUILD_REQUIRED)
    with pytest.raises(ReviewKbNotReadyError) as ei:
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid, "核准答案", state=holder)
    assert ei.value.status == 409
    with db_session_factory() as s:
        r = s.get(ReviewQueue, rid)
        assert r.review_status == "待审" and r.approved_answer is None
        assert s.query(KnowledgeChunk).count() == 0


class _ProbeEmbed:
    """在 embedding 调用现场探测锁纪律:review 行锁可 NOWAIT 拿到
    (外部 embedding 调用不持 DB 行锁),知识库锁拿不到(向量化阶段仍持全局锁)。"""

    def __init__(self, sf, rid):
        self.sf, self.rid = sf, rid
        self.row_lock_free = None
        self.kb_lock_held = None

    def embed_documents(self, texts):
        with self.sf() as s:
            s.query(ReviewQueue).filter_by(id=self.rid).with_for_update(
                nowait=True).first()
            s.commit()
        self.row_lock_free = True
        from app.services.kb_admin import JobBusyError
        try:
            with kb_admin.job_lock():
                self.kb_lock_held = False
        except JobBusyError:
            self.kb_lock_held = True
        return [[0.1] * 1024 for _ in texts]


def test_approve_vectorize_phase_lock_discipline(db_session_factory):
    """冻结事务提交后释放 DB 行锁、向量化阶段继续持知识库锁(spec §5.6 锁序)。"""
    rid = _seed(db_session_factory)
    probe = _ProbeEmbed(db_session_factory, rid)
    out = review_service.approve(_settings(), db_session_factory, probe,
                                 _FakeStore(), rid, "核准答案")
    assert out["review_status"] == "通过"
    assert probe.row_lock_free is True    # 外部 embedding 调用不持 DB 行锁
    assert probe.kb_lock_held is True     # 向量化阶段继续持知识库锁


def test_approve_write_failure_keeps_frozen_and_retryable(db_session_factory):
    """向量化失败:保留写入中+冻结答案/pending 块+last_write_error;仅可原样重试。"""
    rid = _seed(db_session_factory)
    store = _FakeStore()
    with pytest.raises(ReviewWriteError):
        review_service.approve(_settings(), db_session_factory, _BoomEmbed(),
                               store, rid, "核准答案")
    with db_session_factory() as s:
        r = s.get(ReviewQueue, rid)
        assert r.review_status == "写入中" and r.approved_answer == "核准答案"
        assert r.approved_at is None and "embedding api down" in r.last_write_error
        chunks = s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all()
        assert chunks and all(c.vectorize_status == "pending" for c in chunks)
        frozen_ids = list(r.knowledge_chunk_ids)
    with pytest.raises(ReviewConflictError):   # 写入中不可驳回
        review_service.reject(db_session_factory, rid)
    with pytest.raises(ReviewConflictError):   # 写入中改答案 409
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid, "别的答案")
    out = review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                 store, rid, None)   # 省略答案=原样重试
    assert out["review_status"] == "通过"
    with db_session_factory() as s:
        r = s.get(ReviewQueue, rid)
        chunks = sorted(s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all(), key=lambda c: c.chunk_index)
        assert [c.id for c in chunks] == frozen_ids   # 复用冻结块,无重复
        assert all(c.vectorize_status == "done" for c in chunks)
        assert r.review_status == "通过" and r.approved_at is not None
        assert r.last_write_error is None            # 通过时清错误


# ─── reset 选择性删除 ────────────────────────────────────────────────────────


def test_reset_preserves_review_chunks(db_session_factory, monkeypatch):
    rid = _seed(db_session_factory)
    store = _FakeStore()
    review_service.approve(_settings(), db_session_factory, _FakeEmbed(), store,
                           rid, "核准答案")
    with db_session_factory() as s:   # 塞一个文档来源块 + 一个手工块(source_doc NULL)
        s.add(KnowledgeChunk(category="c", questions="q", answer="a",
                             source_doc="knowledge_docs/product-faq.md",
                             chunk_index=1, vectorize_status="done"))
        s.add(KnowledgeChunk(category="手工", questions="手", answer="工。",
                             vectorize_status="done"))
        s.commit()
    monkeypatch.setattr(kb_admin, "run_ingest",
                        lambda *a, **kw: 0)   # 隔离真语料,不重灌
    kb_admin.reset_kb(_settings(), db_session_factory, _FakeEmbed(), store)
    with db_session_factory() as s:
        docs = s.query(KnowledgeChunk).filter(
            KnowledgeChunk.source_doc.like("knowledge_docs/%")).all()
        review_rows = s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all()
        manual_rows = s.query(KnowledgeChunk).filter_by(source_doc=None).all()
    assert docs == []            # 文档块已清(真跑时随后 run_ingest 重灌)
    assert review_rows           # review 来源必须保留
    assert manual_rows           # 手工/挖掘块保留


# ─── rebuild 重放 ────────────────────────────────────────────────────────────


def test_replay_reviews_locked(db_session_factory):
    """清表后重放(通过,写入中):刷新 chunk ids/指针、统一向量化;写入中收敛为通过,
    通过保留原 approved_at;待审/驳回从不重放。"""
    sf = db_session_factory
    rid_done = _seed(sf)
    review_service.approve(_settings(), sf, _FakeEmbed(), _FakeStore(), rid_done,
                           "已通过答案")
    rid_mid = _seed(sf, review_status="写入中", approved_answer="重放答案",
                    knowledge_chunk_ids=[999999],
                    last_write_error="IngestError: 向量化批次失败")
    rid_pend = _seed(sf)                       # 待审:从不重放
    rid_rej = _seed(sf)
    review_service.reject(sf, rid_rej)         # 驳回:从不重放
    with sf() as s:
        at_done = s.get(ReviewQueue, rid_done).approved_at
        old_ids = list(s.get(ReviewQueue, rid_done).knowledge_chunk_ids)
    kb_admin._clear_knowledge_tables(sf)       # rebuild 破坏性步骤同款清表
    store = _FakeStore()
    review_service.replay_reviews_locked(_settings(), sf, _FakeEmbed(), store)
    with sf() as s:
        for rid in (rid_done, rid_mid):
            r = s.get(ReviewQueue, rid)
            chunks = sorted(s.query(KnowledgeChunk).filter_by(
                source_doc=f"review:{rid}").all(), key=lambda c: c.chunk_index)
            assert chunks and all(c.vectorize_status == "done" for c in chunks)
            assert r.knowledge_chunk_ids == [c.id for c in chunks]  # 主键已刷新
            by_id = {c.id: c for c in chunks}
            for c in chunks:
                if c.next_chunk_id is not None:
                    assert by_id[c.next_chunk_id].prev_chunk_id == c.id
        done_r = s.get(ReviewQueue, rid_done)
        assert done_r.review_status == "通过" and done_r.approved_at == at_done
        assert set(done_r.knowledge_chunk_ids).isdisjoint(old_ids)
        mid_r = s.get(ReviewQueue, rid_mid)
        assert mid_r.review_status == "通过" and mid_r.approved_at is not None
        assert mid_r.last_write_error is None
        assert s.query(KnowledgeChunk).filter(KnowledgeChunk.source_doc.in_(
            [f"review:{rid_pend}", f"review:{rid_rej}"])).count() == 0
        assert s.get(ReviewQueue, rid_pend).review_status == "待审"
        assert s.get(ReviewQueue, rid_rej).review_status == "驳回"
        all_ids = {c.id for c in s.query(KnowledgeChunk).all()}
    assert store.ids == all_ids   # 重放块与全库一致可检索


def _write_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.md").write_text(DOC_R_SPEC, encoding="utf-8")
    return docs


def test_rebuild_replays_review_chunks(db_session_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(kb_admin, "REPO_ROOT", tmp_path)  # 评估集不在场:跳过 GT 预检
    sf = db_session_factory
    rid = _seed(sf)
    store = _FakeStore()
    review_service.approve(_settings(), sf, _FakeEmbed(), store, rid, "核准答案")
    with sf() as s:
        old_ids = list(s.get(ReviewQueue, rid).knowledge_chunk_ids)
        old_at = s.get(ReviewQueue, rid).approved_at
    docs = _write_docs(tmp_path)
    holder = KnowledgeStateHolder()
    out = kb_admin.rebuild_index(_settings(), sf, _FakeEmbed(), store, holder, docs)
    assert out["message"].startswith("重建完成") and holder.get() == "ready"
    with sf() as s:
        r = s.get(ReviewQueue, rid)
        chunks = sorted(s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all(), key=lambda c: c.chunk_index)
        assert chunks and all(c.vectorize_status == "done" for c in chunks)
        assert r.knowledge_chunk_ids == [c.id for c in chunks]   # 新主键刷新
        assert set(r.knowledge_chunk_ids).isdisjoint(old_ids)   # 不复用旧 id
        assert r.review_status == "通过" and r.approved_at == old_at
        # tmp 目录不在真 REPO_ROOT 下,source_doc 是绝对路径而非 knowledge_docs/ 前缀
        doc_chunks = s.query(KnowledgeChunk).filter(
            KnowledgeChunk.source_doc.isnot(None),
            KnowledgeChunk.source_doc != f"review:{rid}").count()
    assert doc_chunks == 1
    assert store.all_ids() >= {c.id for c in chunks}   # 重放块已可检索


def test_rebuild_replay_failure_marks_rebuild_required(db_session_factory, tmp_path,
                                                        monkeypatch):
    """重放阶段失败 → rebuild_failed + rebuild_required,审核恢复源保留;下次可恢复。"""
    from app.services.kb_admin import RebuildFailedError
    monkeypatch.setattr(kb_admin, "REPO_ROOT", tmp_path)
    sf = db_session_factory
    rid = _seed(sf)
    review_service.approve(_settings(), sf, _FakeEmbed(), _FakeStore(), rid,
                           "核准答案")
    docs = _write_docs(tmp_path)
    store = _FakeStore()
    holder = KnowledgeStateHolder()
    flaky = _FlakyEmbed()
    with pytest.raises(RebuildFailedError):
        kb_admin.rebuild_index(_settings(), sf, flaky, store, holder, docs)
    assert holder.get() == "rebuild_required"
    with sf() as s:
        r = s.get(ReviewQueue, rid)   # review_queue 恢复源完整保留
        assert r.review_status == "通过" and r.approved_answer == "核准答案"
    out = kb_admin.rebuild_index(_settings(), sf, flaky, store, holder, docs)
    assert out["message"].startswith("重建完成") and holder.get() == "ready"
    with sf() as s:
        r = s.get(ReviewQueue, rid)
        chunks = s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all()
        assert chunks and r.knowledge_chunk_ids == [c.id for c in chunks]


# ─── manual_ingest 互斥 ──────────────────────────────────────────────────────


def test_manual_ingest_takes_job_lock(db_session_factory):
    """manual_ingest 与 approve/建库等写入口互斥:锁被持有时 409,不入块。"""
    from app.services.kb_admin import JobBusyError
    with kb_admin.job_lock():
        with pytest.raises(JobBusyError):
            kb_admin.manual_ingest(_settings(), db_session_factory, _FakeEmbed(),
                                   _FakeStore(), "faq", "补充", "## q\n\na。",
                                   vectorize=False)
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 0
    out = kb_admin.manual_ingest(_settings(), db_session_factory, _FakeEmbed(),
                                 _FakeStore(), "faq", "补充", "## q\n\na。",
                                 vectorize=False)
    assert out["chunk_ids"]


# ─── 端点级 POST /api/review/{id}/approve ───────────────────────────────────


def make_approve_app(sf, embed, kb_store, knowledge_state=None):
    rt = make_runtime(tools=[])
    runtime = AppRuntime(store=rt.store, toolset_factory=rt.toolset_factory,
                         session_factory=sf, embed=embed, kb_store=kb_store,
                         knowledge_state=knowledge_state or rt.knowledge_state)
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=runtime)


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_api_approve_roundtrip(db_session_factory):
    """端点级:待审+核准答案 → 200 通过链路(假 embed/store 走完冻结→向量化→CAS)。"""
    rid = _seed(db_session_factory)
    store = _FakeStore()
    app = make_approve_app(db_session_factory, _FakeEmbed(), store)
    async with _client(app) as client:
        resp = await client.post(f"/api/review/{rid}/approve",
                                 json={"approved_answer": "核准答案"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["review_status"] == "通过" and body["knowledge_chunk_ids"]
        detail = (await client.get(f"/api/review/{rid}/detail")).json()
        assert detail["review_status"] == "通过"   # _item 契约无 approved_at 字段
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").count() >= 1


async def test_api_approve_conflict_contract(db_session_factory):
    """驳回项 approve → 409 review_conflict,错误契约 {"error":{"code","message"}}。"""
    rid = _seed(db_session_factory)
    review_service.reject(db_session_factory, rid)
    app = make_approve_app(db_session_factory, _FakeEmbed(), _FakeStore())
    async with _client(app) as client:
        resp = await client.post(f"/api/review/{rid}/approve",
                                 json={"approved_answer": "答"})
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert set(err) == {"code", "message"} and err["code"] == "review_conflict"


async def test_api_approve_embed_missing_409(db_session_factory):
    """未装配 embed(无 EMBEDDING_API_KEY 同态):409,审核项保持待审不动。"""
    rid = _seed(db_session_factory)
    app = make_approve_app(db_session_factory, None, _FakeStore())
    async with _client(app) as client:
        resp = await client.post(f"/api/review/{rid}/approve",
                                 json={"approved_answer": "答"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "review_conflict"
    with db_session_factory() as s:
        assert s.get(ReviewQueue, rid).review_status == "待审"
        assert s.query(KnowledgeChunk).count() == 0
