import asyncio
import json

import pytest

from app.models import Conversation, Ticket
from app.tools.business import MOCK_TOOLS, build_tools
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401  (fixture 注册,依赖需一并导入)


def _invoke(tool, **args):
    return tool.invoke({"name": tool.name, "args": args, "id": "t1", "type": "tool_call"})


def test_mock_tools_return_parseable_json():
    by_name = {t.name: t for t in MOCK_TOOLS}
    order = json.loads(_invoke(by_name["query_order"], order_id="1001").content)
    assert {"order_id", "status", "amount", "product", "created_at"} <= order.keys()
    product = json.loads(_invoke(by_name["query_product"], product_name="水杯").content)
    assert {"product_name", "price", "stock", "on_sale"} <= product.keys()
    logistics = json.loads(_invoke(by_name["query_logistics"], order_id="1001").content)
    assert {"order_id", "company", "tracking_no", "traces"} <= logistics.keys()
    assert 2 <= len(logistics["traces"]) <= 4


def test_mock_tools_arg_constraints():
    from pydantic import ValidationError
    by_name = {t.name: t for t in MOCK_TOOLS}
    with pytest.raises(ValidationError):
        _invoke(by_name["query_order"], order_id="   ")
    with pytest.raises(ValidationError):
        _invoke(by_name["query_order"], order_id="x" * 65)


async def test_create_ticket_writes_and_flips_status(db_session_factory):
    def _mk(sf):
        with sf() as s:
            c = Conversation(user_id="u")
            s.add(c)
            s.commit()
            return c.id

    cid = await asyncio.to_thread(_mk, db_session_factory)
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=cid)}
    out = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["create_ticket"], description="要求人工核实退款",
                        ticket_type="售后").content))
    assert out["ticket_no"].startswith("T") and len(out["ticket_no"]) == 27

    def _check(sf):
        with sf() as s:
            t = s.query(Ticket).one()
            assert t.conversation_id == cid and t.status == "待处理"
            assert s.get(Conversation, cid).status == "已转人工"

    await asyncio.to_thread(_check, db_session_factory)


async def test_create_ticket_rejects_bad_type(db_session_factory):
    from pydantic import ValidationError
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=1)}
    with pytest.raises(ValidationError):
        _invoke(tools["create_ticket"], description="x", ticket_type="退款")


# ─── query_faq 新契约(T7):trace / evidence / scope / 降级分支 ───────────────


class StubRetriever:
    def __init__(self, result): self._r = result; self.calls = []
    def search(self, query, **kw):
        self.calls.append(kw)
        return self._r


def _result(low=False, note=None):
    from app.knowledge.query_understanding import passthrough_plan
    from app.knowledge.retriever import KnowledgeHit, RetrievalResult
    hits = [] if low else [KnowledgeHit(5, 0.9, "商品FAQ", "运费怎么算",
                                        "满 99 包邮。", "doc", 1, "商品FAQ > 运费")]
    return RetrievalResult(hits, "hybrid_rerank", "hybrid_rerank",
                           hits[0].score if hits else None, 0.0, low, note,
                           passthrough_plan("q"), {"dense": 3, "bm25": 3})


def test_query_faq_ok_writes_trace_and_evidence():
    ts = build_tools(None, 1, retriever=StubRetriever(_result()), settings=make_settings())
    out = json.loads(ts.tools_by_name["query_faq"].invoke({"keyword": "运费"}))
    assert out["low_confidence"] is False and out["effective_strategy"] == "hybrid_rerank"
    assert out["evidence"][0]["ref_no"] == 1
    assert out["evidence"][0]["question"] == "运费怎么算"
    assert ts.retrieval_trace.status == "ok"
    assert ts.retrieval_trace.evidence == out["evidence"]   # 同一份列表(逐字段相等)


def test_query_faq_low_confidence_flag():
    ts = build_tools(None, 1, retriever=StubRetriever(_result(low=True)),
                     settings=make_settings())
    out = json.loads(ts.tools_by_name["query_faq"].invoke({"keyword": "日本"}))
    assert out["low_confidence"] is True and out["evidence"] == []
    assert ts.retrieval_trace.status == "low_confidence"


def test_query_faq_scope_passed_through():
    r = StubRetriever(_result())
    ts = build_tools(None, 1, retriever=r, settings=make_settings())
    ts.tools_by_name["query_faq"].invoke({"keyword": "保修", "scope": "policy"})
    assert r.calls[0]["scope"] == "policy"
    with pytest.raises(Exception):     # pydantic 校验:非法 scope 在参数层就被拒
        ts.tools_by_name["query_faq"].invoke({"keyword": "x", "scope": "bogus"})


def test_query_faq_rebuilding_goes_tool_error():
    ts = build_tools(None, 1, retriever=StubRetriever(
        _result(low=True, note="知识库正在重建,请稍后重试")), settings=make_settings())
    out = json.loads(ts.tools_by_name["query_faq"].invoke({"keyword": "x"}))
    assert out["evidence"] == [] and "重建" in out["note"]
    assert ts.retrieval_trace.status == "tool_error"
    assert ts.retrieval_trace.error_code == "kb_rebuilding"


def test_turn_toolset_satisfies_legacy_list_contract():
    """T7→T9 间 ChatService 仍把 factory 结果直喂 ToolRegistry 并做真值判断,
    TurnToolset 须满足列表协议(可迭代/可取长/按下标/空集为假)。"""
    from app.tools.business import RetrievalTrace, TurnToolset
    from app.tools.executor import ToolRegistry
    ts = build_tools(None, 1)
    assert len(ts) == 5
    assert ts[0].name == "query_order"
    reg = ToolRegistry(ts)
    assert {t.name for t in reg.tools} == {"query_order", "query_product",
                                           "query_logistics", "query_faq", "create_ticket"}
    assert not TurnToolset([], RetrievalTrace())   # make_runtime(tools=[]) 空工具集路径


def test_write_ticket_inserts_row_without_commit(db_session_factory):
    from app.tools.business import write_ticket
    from app.models import Conversation, Ticket
    with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        ticket_no = write_ticket(s, conv.id, "测试描述", "投诉")
        s.commit()
    assert ticket_no.startswith("T")
    with db_session_factory() as s:
        row = s.get(Ticket, ticket_no)
        assert row.description == "测试描述" and row.status == "待处理"
        assert s.get(Conversation, conv.id).status == "进行中"  # helper 不碰会话状态


def test_write_ticket_rollback_leaves_nothing(db_session_factory):
    from app.tools.business import write_ticket
    from app.models import Conversation, Ticket
    import pytest
    with pytest.raises(Exception), db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        write_ticket(s, conv.id, "x", "售后")
        raise RuntimeError("boom")  # 上下文管理器回滚
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 0


def test_create_ticket_keeps_compat_behavior(db_session_factory):
    # 旧工具:工单 + conv.status=已转人工 同事务
    from app.tools.business import build_tools
    from app.models import Conversation
    with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.commit()
        cid = conv.id
    ts = build_tools(db_session_factory, cid)
    out = ts.tools_by_name["create_ticket"].invoke(
        {"description": "要投诉", "ticket_type": "投诉"})
    import json
    assert json.loads(out)["ticket_no"].startswith("T")
    with db_session_factory() as s:
        assert s.get(Conversation, cid).status == "已转人工"


# ─── ch06 图专用只读工具 build_graph_tools(Task 5)──────────────────────────

from app.tools.business import build_graph_tools
from tests.conftest import TEST_USER_ID

OTHER_USER = "22222222-2222-2222-2222-222222222222"


def _toolset(user_id=TEST_USER_ID, retriever=None, with_faq=True):
    return build_graph_tools(user_id, "原始问题", retriever, make_settings(),
                             with_faq=with_faq)


def test_query_order_ownership():
    ts = _toolset()
    qo = ts.tools_by_name["query_order"] if hasattr(ts, "tools_by_name") else {t.name: t for t in ts.tools}["query_order"]
    import json
    ok = json.loads(qo.invoke({"order_id": "1111-1001"}))
    assert ok["order_id"] == "1111-1001" and ok["status"] == "已完成" and ok["product"] == "保温杯"
    assert ts.order_snapshots[-1]["order_id"] == "1111-1001"  # 成功才记快照
    err = json.loads(qo.invoke({"order_id": "2222-1001"}))     # 他人订单
    assert "error" in err and "error" not in ok
    err2 = json.loads(qo.invoke({"order_id": "1111-9999"}))    # 不存在
    assert "error" in err2


def test_query_logistics_deterministic_and_consistent():
    ts = _toolset()
    import json
    ql = {t.name: t for t in ts.tools}["query_logistics"]
    a = json.loads(ql.invoke({"order_id": "1111-1001"}))
    b = json.loads(ql.invoke({"order_id": "1111-1001"}))
    assert a == b                          # 确定性
    assert a["traces"][-1]["description"] == "已签收"   # 已完成订单含签收
    transit = json.loads(ql.invoke({"order_id": "1111-1004"}))  # 在途
    assert transit["traces"][-1]["description"] != "已签收"
    assert "error" in json.loads(ql.invoke({"order_id": "2222-1001"}))


def test_query_faq_uses_full_resolved_query_and_caches():
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    rt = _FakeRetriever(_result())
    ts = build_graph_tools(TEST_USER_ID, "退货运费谁承担", rt, make_settings())
    qf = {t.name: t for t in ts.tools}["query_faq"]
    import json
    r1 = json.loads(qf.invoke({}))
    assert rt.calls == ["退货运费谁承担"]      # 完整 resolved_query,无参数
    assert r1["low_confidence"] is False and r1["evidence"]
    assert ts.faq_trace.status == "ok" and ts.faq_trace.evidence == r1["evidence"]
    r2 = json.loads(qf.invoke({}))
    assert r2 == r1 and len(rt.calls) == 1     # 同轮缓存,不重复检索


def test_query_faq_low_confidence_and_maintenance():
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    from app.knowledge.retriever import NOTE_REBUILDING
    import json
    ts = build_graph_tools(TEST_USER_ID, "q",
                           _FakeRetriever(_result(low=True, hits=[], score=0.01)),
                           make_settings())
    r = json.loads({t.name: t for t in ts.tools}["query_faq"].invoke({}))
    assert r["evidence"] == [] and r["low_confidence"] is True   # 不给候选正文
    assert ts.faq_trace.status == "low_confidence"
    ts2 = build_graph_tools(TEST_USER_ID, "q",
                            _FakeRetriever(_result(note=NOTE_REBUILDING, low=True,
                                                   hits=[], score=None)),
                            make_settings())
    json.loads({t.name: t for t in ts2.tools}["query_faq"].invoke({}))
    assert ts2.faq_trace.status == "unavailable" and ts2.faq_trace.error_code == "kb_rebuilding"


def test_without_faq_and_registry_names():
    ts = _toolset(with_faq=False)
    names = [t.name for t in ts.tools]
    assert names == ["query_order", "query_product", "query_logistics"]
    ts2 = _toolset()
    assert "query_faq" in [t.name for t in ts2.tools]
    assert "create_ticket" not in [t.name for t in ts2.tools]  # 写工具永不在册
