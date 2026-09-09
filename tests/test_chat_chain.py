import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.chains.chat_chain import (
    _validate_trimmed,
    build_chat_messages,
    check_input_budget,
)
from app.errors import MessageTooLongError
from app.sessions import StoredMessage

SYSTEM = "你是电商售后客服小蜜。"


def history(n_turns: int) -> list[StoredMessage]:
    out = []
    for i in range(n_turns):
        out.append(StoredMessage("user", f"问题{i} " + "长文本" * 40))
        out.append(StoredMessage("assistant", f"回答{i} " + "长文本" * 40))
    return out


def test_build_without_trim_when_fits():
    msgs = build_chat_messages(SYSTEM, history(2), "现在的问题", 2000)
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[-1].content == "现在的问题"
    assert len(msgs) == 1 + 4 + 1


def test_trim_drops_oldest_turns_keeps_system_and_current():
    msgs = build_chat_messages(SYSTEM, history(10), "现在的问题", 300)
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[-1], HumanMessage)
    assert msgs[-1].content == "现在的问题"
    assert len(msgs) < 22
    # 角色顺序:system 之后 human/assistant 交替且以 human 结尾
    roles = [type(m) for m in msgs[1:]]
    assert roles[0] is HumanMessage
    assert all(a is not b for a, b in zip(roles, roles[1:]))


def test_trim_drops_oldest_not_newest():
    msgs = build_chat_messages(SYSTEM, history(10), "现在的问题", 300)
    contents = [m.content for m in msgs]
    assert any("问题9" in c for c in contents)
    assert not any("问题0" in c for c in contents)


def test_current_input_over_budget_raises():
    with pytest.raises(MessageTooLongError):
        check_input_budget(SYSTEM, "超" * 10000, 100)
    with pytest.raises(MessageTooLongError):
        build_chat_messages(SYSTEM, [], "超" * 10000, 100)


def test_validate_trimmed_rejects_bad_role_order():
    good = [SystemMessage(content="s"), HumanMessage(content="q"), AIMessage(content="a"), HumanMessage(content="now")]
    _validate_trimmed(good, "now", 100)  # 好序列不炸

    bad_starts_with_ai = [SystemMessage(content="s"), AIMessage(content="a"), HumanMessage(content="now")]
    with pytest.raises(RuntimeError):
        _validate_trimmed(bad_starts_with_ai, "now", 100)

    bad_adjacent_humans = [SystemMessage(content="s"), HumanMessage(content="q"), HumanMessage(content="now")]
    with pytest.raises(RuntimeError):
        _validate_trimmed(bad_adjacent_humans, "now", 100)
