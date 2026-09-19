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
