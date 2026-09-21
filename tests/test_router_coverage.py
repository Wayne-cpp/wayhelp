# tests/test_router_coverage.py
"""spec §5 样例表的全链路钉:库存不依赖知识库/取消与规格走 query_faq/
通用售后不选单不扩写/个案缺号出选择器/澄清分支无工具。"""

from app.knowledge.retriever import KnowledgeHit
from tests.test_ch05_acceptance import _FakeRetriever, _result, _client, _turn, _types, _deltas
from tests.test_refund_flow import _app
from tests.conftest import TEST_USER_ID


def _scoped_result(doc="returns-policy.md", cat="policy"):
    return _result(hits=[KnowledgeHit(9, 0.9, cat, "问", "答", doc, 0, "节")])


async def test_stock_query_answers_without_knowledge():
    app = _app([['{"intent":"商品咨询","confidence":0.9}'],
                [("tool", [{"index": 0, "name": "query_product", "id": "c1",
                            "args": '{"product_name":"保温杯"}'}])],
                ["保温杯有货。"]],
               _FakeRetriever(_result()))  # retriever 在,但不得被调用
    async with await _client(app) as client:
        frames, _ = await _turn(client, "保温杯还有库存吗")
    rt = app.state.retriever
    assert rt.calls == [] and "有货" in _deltas(frames)


async def test_stock_query_survives_kb_unavailable():
    rt = _FakeRetriever(_result(note="知识库尚未建立", low=True, hits=[], score=None))
    app = _app([['{"intent":"商品咨询","confidence":0.9}'],
                [("tool", [{"index": 0, "name": "query_product", "id": "c1",
                            "args": '{"product_name":"保温杯"}'}])],
                ["保温杯有货。"]],
               rt)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "保温杯还有库存吗")
    assert "有货" in _deltas(frames)  # 纯业务查询不依赖知识库可用性


async def test_cancel_rule_uses_query_faq():
    rt = _FakeRetriever(_scoped_result(doc="product-faq.md", cat="faq"))
    app = _app([['{"intent":"订单","confidence":0.85}'],
                [("tool", [{"index": 0, "name": "query_faq", "id": "f1", "args": "{}"}])],
                ["未发货订单可以在订单页直接取消[1]。"]],
               rt)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "未发货订单如何取消")
    assert rt.calls == ["未发货订单如何取消"]
    assert "取消" in _deltas(frames)
    assert any(isinstance(f, dict) and f["type"] == "citations" for f in frames)


async def test_general_aftersales_skips_prepare_and_expand():
    rt = _FakeRetriever(_scoped_result(doc="after-sales-manual.md", cat="manual"))
    scripts = [['{"intent":"售后","confidence":0.9}'],
               ['{"mode":"general"}'],
               ["寄修流程分四步[1]……"]]
    app = _app(scripts, rt)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "维修寄修流程是什么")
    assert "order_selector" not in _types(frames)      # 通用问题不选单
    assert len(rt.calls) == 1                          # general 不扩写,单查询
    assert app.state.model.scripts == []               # expand/agent 脚本按序恰好耗尽


async def test_case_without_order_id_shows_selector():
    app = _app([['{"intent":"退款退货","confidence":0.9}'],
                ['{"mode":"order_specific"}']],
               _FakeRetriever(_result()))
    async with await _client(app) as client:
        frames, _ = await _turn(client, "这个能退吗")
    assert _types(frames) == ["session", "order_selector", "[DONE]"]


async def test_clarify_branch_no_tools_no_selector():
    scripts = [['{"intent":"售后","confidence":0.4}'],
               ['{"mode":"clarify"}'],
               ["请问您是想了解售后政策,还是查询某个订单的情况呢?"]]
    app = _app(scripts, _FakeRetriever(_result()))
    async with await _client(app) as client:
        frames, _ = await _turn(client, "嗯那个事")
    types = _types(frames)
    assert "order_selector" not in types and "tool_start" not in types
    assert "请问" in _deltas(frames)


# ── §5 样例表字面钉(评审 Minor:第 3/5/7 行此前只有语义钉)──


async def test_specs_query_uses_query_faq():
    """§5 行 3:MH-W20 水箱容量 → 商品咨询/business → query_faq 取 product-specs 规格。"""
    rt = _FakeRetriever(_scoped_result(doc="product-specs.md", cat="specs"))
    app = _app([['{"intent":"商品咨询","confidence":0.9}'],
                [("tool", [{"index": 0, "name": "query_faq", "id": "f1", "args": "{}"}])],
                ["自动饮水机 MH-W20 的水箱容量为 2L[1]。"]],
               rt)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "自动饮水机 MH-W20 的水箱容量是多少")
    assert rt.calls == ["自动饮水机 MH-W20 的水箱容量是多少"]  # 以完整原问题检索
    assert "2L" in _deltas(frames)
    assert any(isinstance(f, dict) and f["type"] == "citations" for f in frames)


async def test_how_to_refund_general_retrieves_faq():
    """§5 行 5:如何申请退款 → 退款退货/general → 不选单,强制检索 FAQ 申请步骤。"""
    rt = _FakeRetriever(_scoped_result(doc="product-faq.md", cat="faq"))
    scripts = [['{"intent":"退款退货","confidence":0.9}'],
               ['{"mode":"general"}'],
               ["在订单详情页提交退款申请并选择原因即可[1]。"]]
    app = _app(scripts, rt)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "如何申请退款")
    assert "order_selector" not in _types(frames)      # 通用问题不选单
    assert rt.calls == ["如何申请退款"]                # general 不扩写,单查询
    assert "退款申请" in _deltas(frames)
    assert any(isinstance(f, dict) and f["type"] == "citations" for f in frames)
    assert app.state.model.scripts == []               # 脚本按序恰好耗尽


async def test_warranty_order_specific_direct_retrieves_policy():
    """§5 行 7:保修期个案(订单号在问句)→ 售后/order_specific → 唯一候选直通不选单,
    检索问句附加订单事实(签收/状态),扩写支路并发,答复带引用。"""
    rt = _FakeRetriever(_scoped_result(doc="after-sales-manual.md", cat="manual"))
    scripts = [['{"intent":"售后","confidence":0.9}'],
               ['{"mode":"order_specific"}'],
               ['{"queries":["整机保修期是多久","保修期内维修收费吗"]}'],  # expand 罐头
               ["订单 1111-1001 已于 3 天前签收,按政策整机保修一年[1],仍在保修期内。"]]
    app = _app(scripts, rt)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "订单 1111-1001 的这台设备还在保修期吗")
    assert "order_selector" not in _types(frames)      # 订单号在问句,直通不选单
    assert rt.calls[0].startswith("订单 1111-1001 的这台设备还在保修期吗")
    assert "(商品:保温杯" in rt.calls[0]               # 订单事实附加进检索问句(§6.4)
    assert rt.calls[1:] == ["整机保修期是多久", "保修期内维修收费吗"]
    assert "保修" in _deltas(frames)
    assert any(isinstance(f, dict) and f["type"] == "citations" for f in frames)
    assert app.state.model.scripts == []
