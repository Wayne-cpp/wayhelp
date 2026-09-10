import asyncio
import json

import pytest

from app.models import Conversation, Faq, Ticket
from app.tools.business import MOCK_TOOLS, build_tools
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


async def test_query_faq_hit_and_miss(db_session_factory):
    def _seed(sf):
        with sf() as s:
            s.add_all([
                Faq(question="退货政策是什么", answer="七天无理由", category="退货"),
                Faq(question="运费怎么算", answer="满 99 包邮", category="运费"),
            ])
            s.commit()

    await asyncio.to_thread(_seed, db_session_factory)
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=1)}
    hit = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="退货政策").content))
    # query_faq 返回 JSON 字符串: {"results": [...]}
    assert hit["results"] and "七天无理由" in hit["results"][0]["answer"]
    miss = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="邮费").content))
    assert miss["results"] == [] and "未找到" in miss["note"]


async def test_query_faq_like_wildcards_literal(db_session_factory):
    def _seed(sf):
        with sf() as s:
            from app.models import Faq
            s.add(Faq(question="100% 正品吗", answer="是", category="其他"))
            s.add(Faq(question="怎么开发票", answer="订单详情页申请", category="发票"))
            s.commit()

    await asyncio.to_thread(_seed, db_session_factory)
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=1)}
    # "%" 按字面匹配:keyword="100%" 命中;"%"(裸通配)若未转义会捞全表,
    # 转义后只命中含字面 % 的那一行(计划原断言 == [] 与其自身种子矛盾,见任务汇报)
    hit = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="100%").content))
    assert len(hit["results"]) == 1
    wild = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="%").content))
    assert [r["question"] for r in wild["results"]] == ["100% 正品吗"]


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
