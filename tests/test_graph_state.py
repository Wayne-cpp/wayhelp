# tests/test_graph_state.py
from langchain_core.messages import HumanMessage

from app.graph.state import INTENTS, ROUTE_TABLE, new_turn_state


def test_intents_exact():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")


def test_route_table_pure_intent_mapping():
    assert set(INTENTS) == {"物流", "订单", "商品咨询", "退款退货", "售后",
                            "投诉", "闲聊", "其他"}
    assert ROUTE_TABLE == {
        "物流": "business", "订单": "business", "商品咨询": "business",
        "退款退货": "refund", "售后": "refund",
        "投诉": "complaint", "闲聊": "chitchat", "其他": "other",
    }


def test_new_turn_state_resets_temp_fields_but_keeps_cross_turn():
    st = new_turn_state("q")
    for key in ("intent_confidence", "refund_mode", "order_context",
                "expanded_queries", "understanding_degraded"):
        assert key in st
    assert st["order_context"] is None and st["expanded_queries"] == []
    assert st["understanding_degraded"] is False
    assert "needs_knowledge" not in st
    assert "messages" not in st and "active_order" not in st  # 跨轮字段不重置


def test_new_turn_state_resets_all_transients():
    st = new_turn_state("退货政策是什么")
    assert st["raw_query"] == "退货政策是什么" and st["resolved_query"] == ""
    assert st["intent"] is None and st["route"] is None
    assert st["retrieval_result"] is None and st["retrieval_status"] == "not_run"
    assert st["retrieval_error_code"] is None and st["evidence"] == []
    assert st["low_conf_source"] is None and st["low_conf_reason"] is None
    assert st["suggested_actions"] == [] and st["agent_steps"] == 0
    assert st["agent_tokens"] == 0 and st["token_accounting"] == "none"
    assert st["final_text"] == "" and st["source_message_id"] is None
    assert st["node_trace"] == []
    assert "messages" not in st  # 跨轮历史只来自 checkpoint
    assert len(st["turn_messages"]) == 1 and isinstance(st["turn_messages"][0], HumanMessage)
