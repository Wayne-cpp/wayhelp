from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.chains.tool_chat_chain import build_agent_context
from app.services.token_budget import estimate_messages


def _turn():
    return [HumanMessage(content="现在这句"),
            AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            ToolMessage(content="结果", tool_call_id="c1", name="query_order")]


def test_layout_order_and_background_position():
    ctx = build_agent_context("SYS", [HumanMessage(content="层2旧问")],
                              [HumanMessage(content="层1近问"), AIMessage(content="近答")],
                              _turn(), "【对话背景】\n摘要", guard_tokens=10**6)
    roles = [(type(m).__name__, m.content) for m in ctx]
    assert roles[0] == ("SystemMessage", "SYS")
    assert roles[1][1] == "层2旧问" and roles[2][1] == "层1近问"
    assert roles[4] == ("HumanMessage", "现在这句")          # 当前用户句
    assert roles[5] == ("HumanMessage", "【对话背景】\n摘要")  # 合成一条紧跟用户句
    assert isinstance(ctx[6], AIMessage) and isinstance(ctx[7], ToolMessage)


def test_background_omitted_when_empty():
    ctx = build_agent_context("SYS", [], [], _turn(), None, 10**6)
    assert [type(m).__name__ for m in ctx] == ["SystemMessage", "HumanMessage",
                                               "AIMessage", "ToolMessage"]


def test_guard_returns_none():
    assert build_agent_context("SYS" * 1, [], [], _turn(), None, guard_tokens=5) is None
    ctx = build_agent_context("SYS", [], [], _turn(), None,
                              guard_tokens=estimate_messages([SystemMessage(content="SYS"), *_turn()]))
    assert ctx is not None  # 边界值恰好放行
