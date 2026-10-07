import json

from app.config import Settings
from app.tools.builtin import scan_builtin_specs
from app.tools.builtin.faq import FaqRetrievalTrace
from app.tools.catalog import TurnContext, WriteMeta
from tests.conftest import TEST_USER_ID, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _ctx(settings=None, **over):
    base = dict(user_id=TEST_USER_ID, conversation_id=None, resolved_query="q",
                retriever=None, settings=settings or make_settings())
    base.update(over)
    return TurnContext(**base)


def test_scan_finds_all_builtin():
    specs = {s.name: s for s in scan_builtin_specs()}
    assert {"query_order", "query_product", "query_faq", "create_ticket"} <= set(specs)
    assert "query_logistics" not in specs            # 内置已下线(spec §4.1)
    assert specs["create_ticket"].permission == "write"
    assert specs["query_order"].ownership == "order"
    assert specs["query_faq"].args_schema.get("properties", {}) == {}  # 无参


def test_query_order_bound_to_user():
    from app.tools.builtin.business import query_order_fn
    specs = {s.name: s for s in scan_builtin_specs()}
    ctx = _ctx()
    out = json.loads(query_order_fn({"order_id": "1111-1001"}, ctx))
    assert out["status"] == "已完成" and ctx.order_snapshots[0]["order_id"] == "1111-1001"
    miss = json.loads(query_order_fn({"order_id": "9999-1001"}, ctx))
    assert miss == {"error": "订单不存在或不属于当前用户"}


def test_query_product_mock_shape():
    from app.tools.builtin.business import query_product_fn
    out = json.loads(query_product_fn({"product_name": "保温杯"}, _ctx()))
    assert {"product_name", "price", "stock", "on_sale"} <= set(out)


def test_query_faq_caches_and_traces():
    from app.tools.builtin.faq import query_faq_fn
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    trace = FaqRetrievalTrace()
    fake = _FakeRetriever(_result())
    ctx = _ctx(retriever=fake, resolved_query="退货政策", faq_trace=trace)
    first = json.loads(query_faq_fn({}, ctx))
    second = json.loads(query_faq_fn({}, ctx))
    assert first == second and len(fake.calls) == 1  # 同轮二次走缓存,检索器只搜一次
    assert trace.calls == 2 and trace.status == "ok" and first["evidence"]


def test_create_ticket_writes_ticket_and_idempotency_same_tx(db_session_factory):
    from app.tools.builtin.ticket import create_ticket_fn
    from app.models import Ticket, ToolWriteIdempotency, Conversation
    with db_session_factory() as s:
        s.add(Conversation(id=1, user_id=TEST_USER_ID))
        s.commit()
    ctx = _ctx(conversation_id=1, session_factory=db_session_factory)
    meta = WriteMeta(key="k" * 64, args_sha256="a" * 64)
    out = json.loads(create_ticket_fn(
        {"description": "商品有质量问题", "ticket_type": "售后"}, ctx, meta))
    with db_session_factory() as s:
        t = s.get(Ticket, out["ticket_no"])
        assert t is not None and t.ticket_type == "售后"
        assert s.get(Conversation, 1).status == "进行中"  # 不改 conv.status
        idem = s.get(ToolWriteIdempotency, "k" * 64)
        assert idem.ticket_no == out["ticket_no"]


def test_create_ticket_without_conversation_raises():
    from app.tools.builtin.ticket import create_ticket_fn
    import pytest
    with pytest.raises(ValueError):
        create_ticket_fn({"description": "x", "ticket_type": "售后"},
                         _ctx(), WriteMeta(key="k" * 64, args_sha256="a" * 64))
