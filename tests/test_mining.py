import pytest

from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.mining import (
    MiningBatchResult, MinedConversation, MinedQA, normalize_question, run_mining,
)
from app.models import (
    Conversation, KnowledgeChunk, Message, QaExtractionStaging, QaMiningProgress,
)
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class StubStructuredModel:
    """with_structured_output 返回脚本化 parsed;记录调用次数与输入。"""
    def __init__(self, parsed):
        self._parsed = parsed
        self.calls = 0
        self.received = None

    def with_structured_output(self, schema, method=None, include_raw=False):
        self.calls += 1
        outer = self

        class _R:
            def invoke(self, messages):
                outer.received = messages
                return {"parsed": outer._parsed, "parsing_error": None}

        return _R()


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    def embed_query(self, text):
        return [0.5, 0.5, 0.5, 0.5]


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "mining.db"), dim=4)
    yield s
    s.close()


@pytest.fixture()
def settings():
    return make_settings()


def _mk_conversation(sf, user_texts=("退货怎么弄",), answers=("七天无理由。",)):
    with sf() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        for i, (u, a) in enumerate(zip(user_texts, answers)):
            s.add(Message(conversation_id=conv.id, role="user", content=u))
            s.add(Message(conversation_id=conv.id, role="assistant", content=a))
        s.commit()
        return conv.id


def _parsed_for(conv_id, qas):
    return MiningBatchResult(conversations=[
        MinedConversation(conversation_id=conv_id,
                          items=[MinedQA(question=q, answer=a) for q, a in qas])
    ])


def test_mining_happy_path(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "支持七天无理由退货")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        chunk = s.query(KnowledgeChunk).one()
        assert chunk.content_type == "qa_mined" and chunk.category == "对话挖掘"
        assert chunk.questions == "退货政策是什么"
        assert chunk.source_doc is None and chunk.chunk_index is None
        assert chunk.vectorize_status == "done"
        prog = s.query(QaMiningProgress).one()
        assert prog.conversation_id == cid and prog.qa_count == 1
        staging = s.query(QaExtractionStaging).one()
        assert staging.source_ref == f"conv:{cid}" and staging.status == "kept"
    assert store.all_ids() == {chunk.id}


def test_mining_rerun_skips_without_llm(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "七天无理由")]))
    run_mining(settings, db_session_factory, model, FakeEmbeddings(), store)
    assert model.calls == 1
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    assert model.calls == 1  # 重跑不再调 LLM
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 1


def test_zero_qa_conversation_progress(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory, ("你好",), ("你好,请问有什么可以帮您?",))
    model = StubStructuredModel(_parsed_for(cid, []))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        prog = s.query(QaMiningProgress).one()
        assert prog.qa_count == 0
        assert s.query(KnowledgeChunk).count() == 0
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    assert model.calls == 1  # 零 QA 会话重跑也跳过


def test_dedup_within_staging(db_session_factory, store, settings):
    c1 = _mk_conversation(db_session_factory, ("邮费多少",), ("满99包邮",))
    c2 = _mk_conversation(db_session_factory, ("邮费多少?",), ("99元包邮",))
    model = StubStructuredModel(MiningBatchResult(conversations=[
        MinedConversation(conversation_id=c1, items=[MinedQA(question="邮费多少", answer="满99包邮")]),
        MinedConversation(conversation_id=c2, items=[MinedQA(question="邮费多少?", answer="99元包邮")]),
    ]))
    settings = make_settings(mining_batch_size=10)  # 两个会话同批
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        kept = s.query(QaExtractionStaging).filter_by(status="kept").all()
        discarded = s.query(QaExtractionStaging).filter_by(status="discarded").all()
        assert len(kept) == 1 and len(discarded) == 1  # 规范化后同问法,留最早 ID
        assert kept[0].question == "邮费多少"
        assert s.query(KnowledgeChunk).count() == 1


def test_dedup_against_existing_knowledge(db_session_factory, store, settings):
    with db_session_factory() as s:
        s.add(KnowledgeChunk(category="退货", questions="退货政策是什么",
                             answer="七天无理由", content_type="faq",
                             source_doc="knowledge_docs/x.md", chunk_index=1,
                             vectorize_status="done", vector_id="0"))
        s.commit()
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么?", "七天")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        assert s.query(QaExtractionStaging).one().status == "discarded"
        assert s.query(KnowledgeChunk).count() == 1


def test_batch_missing_conversation_rejected(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(MiningBatchResult(conversations=[
        MinedConversation(conversation_id=cid + 999, items=[]),  # 编造 ID
    ]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 1
    with db_session_factory() as s:  # 整批不留 staging/进度
        assert s.query(QaExtractionStaging).count() == 0
        assert s.query(QaMiningProgress).count() == 0


def test_blank_question_rejected(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("   ", "答案")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 1
    with db_session_factory() as s:
        assert s.query(QaExtractionStaging).count() == 0


def test_recover_kept_after_crash_before_load(db_session_factory, store, settings, monkeypatch):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "七天无理由")]))
    import app.knowledge.mining as mining_mod
    real_load = mining_mod._load_kept
    monkeypatch.setattr(mining_mod, "_load_kept",
                        lambda sf: (_ for _ in ()).throw(RuntimeError("crash")))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 1
    with db_session_factory() as s:  # staging/进度已提交,kept 未入库
        assert s.query(QaExtractionStaging).one().status == "kept"
        assert s.query(KnowledgeChunk).count() == 0
    monkeypatch.setattr(mining_mod, "_load_kept", real_load)
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 1  # 重放补齐,不重复插
    assert model.calls == 1


def test_normalize_question():
    assert normalize_question("邮费多少?") == normalize_question(" 邮费 多少？")
    assert normalize_question("ABC") == "abc"


def test_mining_rerun_after_staging_cleared(db_session_factory, store, settings):
    """清空 staging 后重跑:凭进度表跳过,不调 LLM 也不重放入库(spec §11)。"""
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "七天无理由")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    assert model.calls == 1
    with db_session_factory() as s:  # 去重证据消失,只剩进度表
        s.query(QaExtractionStaging).delete()
        s.commit()
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    assert model.calls == 1
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 1
