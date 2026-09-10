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
