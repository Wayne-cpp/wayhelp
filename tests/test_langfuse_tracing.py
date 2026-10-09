"""ch09 Langfuse 挂载:disabled 零侵入;enabled 注入回调与元数据;resume 意图续传。"""
import os

import pytest
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


def test_tag_intent_writes_trace_via_trace_context(monkeypatch):
    """验收回归:tag_intent 不得依赖 ambient OTel context(图节点体里不可靠),
    必须用 trace_context 按 trace_id 建 span 再 update_trace,否则 intent
    元数据静默丢失(线上实测 trace metadata intent 恒 None)。"""
    calls = {}

    class _Span:
        def update_trace(self, **kw):
            calls["update_trace"] = kw

        def end(self):
            calls["ended"] = True

    class _Client:
        def start_observation(self, **kw):
            calls["start"] = kw
            return _Span()

    monkeypatch.setattr(lt, "_get_client", lambda: _Client())
    lt.tag_intent("退款退货", 0.9, trace_id="ab" * 16)
    assert calls["start"]["trace_context"] == {"trace_id": "ab" * 16}
    assert calls["update_trace"]["tags"] == ["intent:退款退货"]
    assert calls["update_trace"]["name"] == "chat_turn"  # 钉住 trace 名,防被 intent_tag 兜底
    assert calls["update_trace"]["metadata"] == {"intent": "退款退货",
                                                 "intent_confidence": 0.9}
    assert calls["ended"] is True


def test_tag_intent_without_trace_id_is_noop(monkeypatch):
    class _Client:
        def start_observation(self, **kw):
            raise AssertionError("无 trace_id 不得建 span")

    monkeypatch.setattr(lt, "_get_client", lambda: _Client())
    lt.tag_intent("闲聊", None)  # 不抛、不碰 client


@pytest.fixture(autouse=True)
def _clean_langfuse_env():
    """ensure_env 直写 os.environ 不回滚——本文件测试后清掉,防跨文件污染
    (tracing 排在 config 前跑时 test_langfuse_disabled_by_default 被毒化)。"""
    keys = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in keys:
        os.environ.pop(k, None)
        if saved[k] is not None:
            os.environ[k] = saved[k]
