from datetime import datetime, timezone

import pytest

from app.models import Conversation, Faq, Message, Ticket
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)


async def test_four_tables_crud(db_session_factory):
    def _work(sf):
        with sf() as s:
            conv = Conversation(user_id="u-1")
            s.add(conv)
            s.flush()
            s.add(Message(conversation_id=conv.id, role="user", content="你好"))
            s.add(Faq(question="q", answer="a", category="c"))
            s.add(Ticket(ticket_no="T1", conversation_id=conv.id,
                         description="d", ticket_type="售后"))
            s.commit()
        with sf() as s:
            assert s.query(Conversation).count() == 1
            msg = s.query(Message).one()
            assert msg.role == "user" and msg.conversation_id == conv.id
            assert s.query(Faq).one().category == "c"
            ticket = s.query(Ticket).one()
            assert ticket.status == "待处理"
            conv2 = s.get(Conversation, conv.id)
            assert conv2.status == "进行中"
            # 更新
            conv2.status = "已转人工"
            s.commit()
            assert s.get(Conversation, conv.id).status == "已转人工"

    import asyncio
    await asyncio.to_thread(_work, db_session_factory)


def test_knowledge_tables_crud(db_session_factory):
    from app.models import KnowledgeChunk, QaExtractionStaging, QaMiningProgress
    with db_session_factory() as s:
        c1 = KnowledgeChunk(category="售后政策", questions="退货", answer="七天无理由。",
                            section_path="售后政策 > 退货", content_type="policy",
                            is_key_clause=True, source_doc="knowledge_docs/退货政策.md",
                            chunk_index=1, vectorize_status="pending")
        s.add(c1)
        s.flush()  # 先拿 c1.id
        c2 = KnowledgeChunk(category="售后政策", questions="换货", answer="十五天。",
                            section_path="售后政策 > 换货", content_type="policy",
                            is_key_clause=False, prev_chunk_id=c1.id,
                            source_doc="knowledge_docs/退货政策.md", chunk_index=2)
        s.add(c2)
        s.flush()  # 再拿 c2.id 回填 c1.next
        c1.next_chunk_id = c2.id
        s.add(QaExtractionStaging(batch_no="b1", source_ref="conv:1",
                                  question="q", answer="a"))
        s.add(QaMiningProgress(conversation_id=1, batch_no="b1", qa_count=1,
                               extracted_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        s.commit()
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index).all()
        assert [r.chunk_index for r in rows] == [1, 2]
        assert rows[0].next_chunk_id == rows[1].id
        assert rows[1].prev_chunk_id == rows[0].id
        assert rows[0].vectorize_status == "pending"
        assert s.query(QaMiningProgress).count() == 1


def test_source_doc_chunk_index_unique(db_session_factory):
    from app.models import KnowledgeChunk
    from sqlalchemy.exc import IntegrityError
    with db_session_factory() as s:
        s.add(KnowledgeChunk(category="c", questions="q", answer="a",
                             source_doc="d.md", chunk_index=1))
        s.commit()
    with db_session_factory() as s:
        s.add(KnowledgeChunk(category="c2", questions="q2", answer="a2",
                             source_doc="d.md", chunk_index=1))
        with pytest.raises(IntegrityError):
            s.commit()


def test_mined_qa_allows_multiple_null_source(db_session_factory):
    from app.models import KnowledgeChunk
    with db_session_factory() as s:
        for i in range(2):
            s.add(KnowledgeChunk(category="对话挖掘", questions=f"q{i}", answer="a",
                                 content_type="qa_mined", source_doc=None, chunk_index=None))
        s.commit()
        assert s.query(KnowledgeChunk).count() == 2
