"""祛魅热身:不借任何 Agent 框架,手写最裸的「带工具的循环」。

Agent 的全部本质:调 LLM → 有 tool_calls 就执行并把结果喂回去 → 没有就收敛。
本章正式的 LangGraph 节点(app/graph/agent_node.py)与此循环同构。
运行:uv run python examples/bare_agent.py [--real]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.tools.business import MOCK_TOOLS

BARE_SYSTEM = "你是客服助手。需要数据时调用工具,拿到结果后回答用户。"
BARE_BUDGET_ANSWER = "抱歉,问题比较复杂,本轮处理已达到上限。"


def run_bare_agent(model, tools, question, max_steps=5):
    """最裸 Agent 循环。返回 (最终文本, 消息序列, 实际步数)。"""
    tool_by_name = {t.name: t for t in tools}
    messages = [SystemMessage(content=BARE_SYSTEM), HumanMessage(content=question)]
    for step in range(1, max_steps + 1):
        ai = model.bind_tools(tools).invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:  # 没有工具调用 → 收敛出答案
            return ai.content, messages, step
        for call in ai.tool_calls:  # 有 → 逐个执行并把结果喂回去
            tool = tool_by_name.get(call["name"])
            result = tool.invoke(call["args"]) if tool is not None else "调用了未注册的工具"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"],
                                        name=call["name"]))
    return BARE_BUDGET_ANSWER, messages, max_steps


class ScriptedModel:
    """examples 自包含的演示假模型(不依赖 tests/):按脚本依次应答。"""

    def __init__(self, script):
        self._script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._script.pop(0) if self._script else AIMessage(content="(无脚本)")


def _demo_script():
    return [
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2", "type": "tool_call"}]),
        AIMessage(content="订单 1001 已发货,物流正在派送中。"),
    ]


def main():
    if "--real" in sys.argv:
        from app.config import Settings
        from langchain_openai import ChatOpenAI

        s = Settings()
        model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                           base_url=s.openai_base_url, max_tokens=s.max_output_tokens)
        question = "订单 1001 到哪了?先查订单再查物流"
    else:
        model = ScriptedModel(_demo_script())
        question = "订单 1001 到哪了?先查订单再查物流(演示脚本)"
    text, messages, steps = run_bare_agent(model, MOCK_TOOLS, question)
    for m in messages:
        print(f"[{m.type}]", (m.content or "")[:120])
    print(f"\n=== 收敛于第 {steps} 步:{text}")


if __name__ == "__main__":
    main()
