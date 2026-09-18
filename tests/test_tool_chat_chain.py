import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.chains.chat_chain import check_input_budget
from app.chains.tool_chat_chain import (
    _validate,
    build_agent_context,
    build_messages_with_tools,
    finalize_tool_calls,
    fit_tool_context,
    merge_tool_call_chunks,
    rebuild_messages,
)
from app.errors import MessageTooLongError
from app.sessions import StoredMessage
from app.tool_envelope import wrap

SYSTEM = "你是电商售后客服小蜜。"


def history(n_turns: int) -> list[StoredMessage]:
    out = []
    for i in range(n_turns):
        out.append(StoredMessage("user", f"问题{i} " + "长文本" * 40))
        out.append(StoredMessage("assistant", f"回答{i} " + "长文本" * 40))
    return out


def test_rebuild_plain_turn():
    msgs = rebuild_messages([StoredMessage("user", "问"), StoredMessage("assistant", "答")])
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage]


def test_rebuild_tool_turn_restores_status():
    stored = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id="c1"),
        StoredMessage("tool", wrap("失败摘要", False, "timeout", 4000), tool_call_id="c2"),
        StoredMessage("assistant", "答复"),
    ]
    # c2 没有对应申请单 -> 对不上即数据损坏
    with pytest.raises(ValueError):
        rebuild_messages(stored)


def test_rebuild_tool_turn_ok():
    stored = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id="c1"),
        StoredMessage("assistant", "答复"),
    ]
    msgs = rebuild_messages(stored)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["id"] == "c1"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].content == "结果"
    assert msgs[2].status == "success" and msgs[2].tool_call_id == "c1"


def test_rebuild_corrupted_envelope_raises():
    with pytest.raises(ValueError):
        rebuild_messages([
            StoredMessage("user", "问"),
            StoredMessage("assistant", None,
                          tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("tool", "not-json", tool_call_id="c1"),
            StoredMessage("assistant", "答"),
        ])


def test_merge_and_finalize():
    acc: dict[int, dict] = {}
    merge_tool_call_chunks(acc, [
        {"name": "query_faq", "args": "{\"keyword\": \"退", "id": "call_1", "index": 0},
    ])
    merge_tool_call_chunks(acc, [
        {"name": None, "args": "货\"}", "id": None, "index": 0},
        {"name": "query_order", "args": "{\"order_id\": \"1\"}", "id": "call_2", "index": 1},
    ])
    calls, broken = finalize_tool_calls(acc)
    assert broken is False
    assert calls == [
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"},
        {"name": "query_order", "args": {"order_id": "1"}, "id": "call_2", "type": "tool_call"},
    ]


def test_finalize_broken_json():
    acc: dict[int, dict] = {}
    merge_tool_call_chunks(acc, [
        {"name": "query_faq", "args": "{损坏", "id": "c1", "index": 0},
    ])
    calls, broken = finalize_tool_calls(acc)
    assert broken is True


def test_trim_drops_orphan_tool_group():
    # 历史里工具 turn 完整,但预算只够留最后两条 -> 头部 user 起的整个 turn 被丢
    history = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果很长" * 50, True, None, 4000), tool_call_id="c1"),
        StoredMessage("assistant", "物流答复" * 50),
        StoredMessage("user", "再问"),
        StoredMessage("assistant", "再答"),
    ]
    msgs = build_messages_with_tools("system", history, "现在呢", max_input_tokens=120)
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "再问"
    assert msgs[-1].content == "现在呢"
    # 不得出现孤儿 ToolMessage
    assert not any(isinstance(m, ToolMessage) for m in msgs)


def test_trim_keeps_complete_tool_group_when_budget_allows():
    history = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id="c1"),
        StoredMessage("assistant", "答复"),
    ]
    msgs = build_messages_with_tools("system", history, "现在呢", max_input_tokens=2000)
    assert any(isinstance(m, ToolMessage) for m in msgs)
    assert msgs[1].content == "查物流"


# ---- 以下用例自 tests/test_chat_chain.py 等价迁移(build_chat_messages 已删) ----


def test_build_without_trim_when_fits():
    msgs = build_messages_with_tools(SYSTEM, history(2), "现在的问题", 2000)
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[-1].content == "现在的问题"
    assert len(msgs) == 1 + 4 + 1


def test_trim_drops_oldest_turns_keeps_system_and_current():
    msgs = build_messages_with_tools(SYSTEM, history(10), "现在的问题", 300)
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[-1], HumanMessage)
    assert msgs[-1].content == "现在的问题"
    assert len(msgs) < 22
    # 角色顺序:system 之后 human/assistant 交替且以 human 结尾
    roles = [type(m) for m in msgs[1:]]
    assert roles[0] is HumanMessage
    assert all(a is not b for a, b in zip(roles, roles[1:]))


def test_trim_drops_oldest_not_newest():
    msgs = build_messages_with_tools(SYSTEM, history(10), "现在的问题", 300)
    contents = [m.content for m in msgs]
    assert any("问题9" in c for c in contents)
    assert not any("问题0" in c for c in contents)


def test_current_input_over_budget_raises():
    with pytest.raises(MessageTooLongError):
        check_input_budget(SYSTEM, "超" * 10000, 100)
    with pytest.raises(MessageTooLongError):
        build_messages_with_tools(SYSTEM, [], "超" * 10000, 100)


def test_validate_rejects_bad_role_order():
    # 原 test_validate_trimmed_rejects_bad_role_order 等价迁移;
    # 相邻 human 断言随旧交替约束移除(工具上下文由 tool 组完整性取代)
    good = [SystemMessage(content="s"), HumanMessage(content="q"), AIMessage(content="a"), HumanMessage(content="now")]
    _validate(good, "now", 100)  # 好序列不炸

    bad_starts_with_ai = [SystemMessage(content="s"), AIMessage(content="a"), HumanMessage(content="now")]
    with pytest.raises(RuntimeError):
        _validate(bad_starts_with_ai, "now", 100)


# ---- fit_tool_context(第二次调用预算)----


def _second_call_messages():
    """模拟 chat_service 第二次调用的入参形状:[system, ...历史..., 当前human, AI(tool_calls), Tool]。"""
    base = [
        SystemMessage(content="s"),
        HumanMessage(content="旧问题" + "甲" * 60),
        AIMessage(content="旧回答" + "甲" * 60),
        HumanMessage(content="当前问题"),
    ]
    second = [
        *base,
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1"}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="结果" * 20, tool_call_id="c1", name="query_order"),
    ]
    return base, second


def test_fit_drops_oldest_history_turn_keeps_current_human():
    # 预算够丢历史:丢最旧完整历史 turn 达标,当前 human 与工具组原样保留
    base, second = _second_call_messages()
    # count_tokens_approximately: 全量 96 > 70 >= 丢历史后 55
    fitted = fit_tool_context(second, max_input_tokens=70,
                              protected_from=len(base) - 1)  # 当前 human 下标
    assert fitted is not None
    assert [m.content for m in fitted] == [
        "s", "当前问题", "",
        "结果" * 20,
    ]
    assert not any("旧问题" in m.content or "旧回答" in m.content for m in fitted)


# ---- build_agent_context(Task 11:ReAct 每步上下文组装/裁剪)----


def _group(call_id, name):
    return [StoredMessage("assistant", None, tool_calls=[
                {"name": name, "args": {}, "id": call_id, "type": "tool_call"}]),
            StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id=call_id)]


def test_rebuild_messages_multi_group_roundtrip():
    stored = ([StoredMessage("user", "q")] + _group("c1", "query_order")
              + _group("c2", "query_logistics") + [StoredMessage("assistant", "答")])
    msgs = rebuild_messages(stored)
    assert [m.type for m in msgs] == ["human", "ai", "tool", "ai", "tool", "ai"]
    assert msgs[1].tool_calls[0]["id"] == "c1" and msgs[3].tool_calls[0]["id"] == "c2"


def test_build_agent_context_protects_current_turn_and_evidence():
    history = [HumanMessage(content="旧问" + "长" * 200), AIMessage(content="旧答" * 200)]
    turn = [HumanMessage(content="当前问题")]
    out = build_agent_context("系统", history, turn, "本轮证据文本", 10_000)
    assert [m.content for m in out] == ["系统", "旧问" + "长" * 200, "旧答" * 200,
                                        "本轮证据文本", "当前问题"]
    # 预算极小:丢历史也要保住 system + evidence + 当前 turn
    tight = build_agent_context("系统", history, turn, "本轮证据文本", 60)
    assert tight is not None
    assert tight[0].content == "系统" and tight[-1].content == "当前问题"
    assert "本轮证据文本" in [m.content for m in tight]
    # 历史丢光仍放不下 → None(调用方发 tool_context_too_long)
    assert build_agent_context("系统" * 500, [], turn, "证据" * 500, 10) is None
