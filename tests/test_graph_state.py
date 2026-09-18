# tests/test_graph_state.py
import pytest
from langchain_core.messages import HumanMessage

from app.graph.state import INTENTS, ROUTE_TABLE, ChatGraphState, new_turn_state

EXPECTED = {
    ("物流", False): "business", ("订单", False): "business",
    ("商品咨询", False): "business", ("售后", False): "business",
    ("退款退货", False): "knowledge", ("投诉", False): "complaint",
    ("闲聊", False): "chitchat",
    **{(i, True): "knowledge" for i in
       ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")},
}


def test_intents_exact():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


@pytest.mark.parametrize("key,route", sorted(EXPECTED.items(), key=str))
def test_route_table_covers_14_combos(key, route):
    assert ROUTE_TABLE[key] == route


def test_route_table_size_and_completeness():
    assert len(ROUTE_TABLE) == 14
    for intent in INTENTS:
        for nk in (True, False):
            assert (intent, nk) in ROUTE_TABLE


def test_new_turn_state_resets_all_transients():
    st = new_turn_state("退货政策是什么")
    assert st["raw_query"] == "退货政策是什么" and st["resolved_query"] == ""
    assert st["intent"] is None and st["needs_knowledge"] is None and st["route"] is None
    assert st["retrieval_result"] is None and st["retrieval_status"] == "not_run"
    assert st["retrieval_error_code"] is None and st["evidence"] == []
    assert st["low_conf_source"] is None and st["low_conf_reason"] is None
    assert st["suggested_actions"] == [] and st["agent_steps"] == 0
    assert st["agent_tokens"] == 0 and st["token_accounting"] == "none"
    assert st["final_text"] == "" and st["source_message_id"] is None
    assert st["node_trace"] == []
    assert "messages" not in st  # 跨轮历史只来自 checkpoint
    assert len(st["turn_messages"]) == 1 and isinstance(st["turn_messages"][0], HumanMessage)
