"""ch09 Langfuse 挂载:disabled 零侵入;enabled 注入回调与元数据;resume 意图续传。"""
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.services import langfuse_tracing as lt


def _settings(**kw):
    base = dict(_env_file=None, openai_base_url="http://x", openai_api_key="k",
                model_name="m", database_url="mysql+pymysql://u:p@h/d")
    base.update(kw)
    return Settings(**base)


def test_disabled_returns_empty_config():
    assert lt.build_trace_config(_settings(), conversation_id="1",
                                 user_id="u", trace_id="t") == {}
    lt.tag_intent("退款退货", 0.9)   # disabled 也不得抛
    lt.flush_langfuse()              # 同上


def test_enabled_builds_handler_and_metadata():
    cfg = lt.build_trace_config(
        _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-t",
                  langfuse_secret_key="sk-lf-t"),
        conversation_id="42", user_id="u1", trace_id="ab" * 16,
        intent="退款退货", intent_confidence=0.92)
    assert len(cfg["callbacks"]) == 1
    md = cfg["metadata"]
    assert md["langfuse_session_id"] == "42"
    assert md["langfuse_user_id"] == "u1"
    assert md["langfuse_trace_name"] == "chat_turn"
    assert md["intent"] == "退款退货" and md["intent_confidence"] == 0.92
    assert "intent:退款退货" in cfg["tags"]


def test_new_turn_state_resets_trace_id():
    from app.graph.state import new_turn_state
    st = new_turn_state("你好", user_db_id=7)
    assert st["trace_id"] is None
    assert st["turn_message_id"] == 7
    assert st["final_assistant_message_id"] is None
