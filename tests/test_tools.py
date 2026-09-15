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
