# tests/test_bare_agent.py
from langchain_core.messages import AIMessage

from examples.bare_agent import BARE_BUDGET_ANSWER, run_bare_agent
from app.tools.business import query_order, query_logistics


class ScriptedSyncModel:
    """测试侧同步假模型:script 每元素是一次 invoke 的返回值。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self._script.pop(0) if self._script else AIMessage(content="(无脚本)")


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_converges_without_tool_call():
    model = ScriptedSyncModel([AIMessage(content="直接回答")])
    text, messages, steps = run_bare_agent(model, [query_order], "你好")
    assert text == "直接回答" and steps == 1 and model.calls == 1


def test_tool_result_fed_back():
    model = ScriptedSyncModel([
        _tool_call("query_order", {"order_id": "1001"}, "c1"),
        AIMessage(content="订单 1001 状态是……"),
    ])
    text, messages, steps = run_bare_agent(model, [query_order], "查订单 1001")
    assert steps == 2
    tool_msgs = [m for m in messages if m.type == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "c1"
    assert "1001" in tool_msgs[0].content  # 真实工具结果被喂回
    assert text == "订单 1001 状态是……"


def test_max_steps_fallback():
    model = ScriptedSyncModel([
        _tool_call("query_order", {"order_id": "1"}, "c1"),
        _tool_call("query_logistics", {"order_id": "1"}, "c2"),
        _tool_call("query_order", {"order_id": "2"}, "c3"),
    ])
    text, _, steps = run_bare_agent(model, [query_order, query_logistics], "一直查", max_steps=2)
    assert steps == 2 and text == BARE_BUDGET_ANSWER
