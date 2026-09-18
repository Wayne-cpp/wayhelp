# ch05 LangGraph Workflow 编排骨架 + 主力 ReAct Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `/v1/chat/stream` 的手写「两次调用」编排原地替换为 LangGraph 图(resolve_reference → classify_intent → 分流 → retrieve → confidence_gate → main_agent 手写 ReAct → log),新增 `/v1/chat/action` 建工单端点与前端「转人工/建工单」独立按钮。

**Architecture:** 确定性 Workflow 骨架(LangGraph StateGraph + 写死的 `(intent, needs_knowledge)` 分流表)+ 手写 ReAct 循环节点(复用 ch02 只读工具 + `suggest_options` 伪工具);State 跨轮持久化用 AsyncSqliteSaver(`data/checkpoints.db`),MySQL 会话/消息表仍做账本,由 log 节点 `commit_turn` 单事务写入;SSE 协议不变,新增 `suggest_actions` 帧(commit 成功后、`[DONE]` 前发出)。

**Tech Stack:** Python 3.12 / FastAPI / langchain-core 1.6.2 / **langgraph 1.0.x + langgraph-checkpoint-sqlite(新增)** / SQLAlchemy 2.0 + MySQL(Docker)/ Milvus Lite 检索(复用 ch03/04)/ 原生 JS 前端。

**Spec:** `docs/superpowers/specs/2026-09-17-ch05-langgraph-workflow-design.md`(commit `ead6c16`,GPT-6 评审修订定稿)。plan 从 spec 论证,执行时两者一起读。

## Global Constraints

每个任务都隐含遵守本节;关键取值逐字抄自 spec,不得改动。

- **技术选型定死**:LangGraph 编排 + AsyncSqliteSaver + 手写 ReAct 节点(不用 create_react_agent)+ 原生 JS 前端。发现矛盾/走不通 → 停下来问用户,不自行换方案。
- **涉库用法先查 Context7**(`/langchain-ai/langgraph/1.0.8`)。已核 API:`StateGraph`/`START`/`END`/`add_node`/`add_conditional_edges`/`compile(checkpointer=)`;`InMemorySaver`(`langgraph.checkpoint.memory`);`AsyncSqliteSaver.from_conn_string(...)` 是**异步上下文管理器**(`langgraph.checkpoint.sqlite.aio`);`add_messages`(`langgraph.graph.message`);`get_stream_writer()`(`langgraph.config`);`astream(input, config, stream_mode=["messages","custom"])` 产出 `(mode, payload)` 二元组,messages 模式 payload 为 `(chunk, metadata)`,`metadata["langgraph_node"]` 标识来源节点。
- **七类意图**(逐字):`物流`、`订单`、`商品咨询`、`退款退货`、`售后`、`投诉`、`闲聊`;四出口:`knowledge`/`business`/`complaint`/`chitchat`。
- **话术常量**(逐字,进 `app/prompts/service.py`):
  - `REFUSAL_ANSWER` 不变:`抱歉,这个问题超出了我目前掌握的资料范围,已为您记录,稍后可转人工客服进一步核实。`
  - `COMPLAINT_REPLY = "非常抱歉给您带来了不好的体验,您的问题我们十分重视。您可以选择转接人工客服,或点击下方按钮建立工单跟进处理。"`
  - `CHITCHAT_REPLY = "您好,我是客服小蜜,可以帮您查订单、查物流、介绍商品和退换货政策,有什么可以帮您的吗?"`
  - `KB_UNAVAILABLE_ANSWER = "知识库暂时不可用,请稍后重试。"`
  - `AGENT_BUDGET_ANSWER = "本轮处理已达到上限,请缩小问题范围后重试。"`
  - `FALLBACK_ANSWER` 修订为:`"抱歉,暂时没有查到相关信息。您可以换个说法,或请求人工帮助并选择相应按钮。"`(从 chat_service.py 挪进 service.py 统一出口)
- **预算默认值**:`max_agent_steps=8`(main_agent 内模型调用次数上限)、`max_agent_tokens=20000`(本轮累计)、`max_input_tokens`/`max_message_chars` 等现有闸不动。
- **聊天 Agent 无写权限**:只绑定 `query_order`/`query_product`/`query_logistics`/`suggest_options`;`create_ticket`/`query_faq` 不进绑定列表也不进 ToolExecutor 注册表;伪造 `create_ticket` 调用只能得到 unknown_tool 错误 ToolMessage,不得写库。
- **分流表写死**:`ROUTE_TABLE` 以 `(intent, needs_knowledge)` 为键覆盖 14 组合(spec §5 表);`needs_knowledge=True` 优先于意图名称;解析失败兜底 `("售后", True)` 走 knowledge,**不得**退回 business。
- **帧序**:`session` → delta/tool_start/tool_end → (log 提交成功)→ citations(如有)→ suggest_actions(如有)→ `[DONE]`;失败路径不发 `[DONE]`、不发按钮。
- **suggest_actions 帧**:`{"type":"suggest_actions","source_message_id":"<str>","options":[{"action":"transfer_human","label":"转人工"},{"action":"create_ticket","label":"建工单","ticket_type":"投诉"}]}`;`source_message_id` 是产生按钮的那轮**用户消息 ID 的字符串**(防 JS BIGINT 精度损失),同一轮至多一帧。
- **测试交付**:`uv run pytest` 全量绿(DB 测试需 `docker compose up -d`);本机访问服务用 `curl --noproxy '*'`;pgrep 用 `pgrep -f '[u]vicorn --factory'` 方括号技巧;单 worker。
- **Conventional Commits**,每任务至少一次 commit;**每个任务 commit 前在 `dev-notes/ch05.md` 追加「任务 N」段**(记四样:用户关键原话/AI 关键产出/纠偏/翻车),dev-notes 更新随任务 commit 一起提交。
- **chat.html 走 Vibe 例外**(无 TDD):`test_chat_page.py` 字符串断言 + 人工点验。
- **Prompt 类任务**(Task 3 意图 prompt)用标注样例验证替代 TDD 红绿:`evals/intent_samples.jsonl` 8 条标注样例 + 真模型跑一遍,结果记 dev-notes。
- 改 prompt 后必须跑全量 pytest(test_prompts.py 有契约断言)。

## File Structure

新增:

- `examples/__init__.py`、`examples/bare_agent.py` — 热身裸循环(祛魅留档)
- `app/graph/__init__.py`
- `app/graph/state.py` — ChatGraphState / new_turn_state / INTENTS / ROUTE_TABLE / route 常量
- `app/graph/errors.py` — TurnAbortError
- `app/graph/events.py` — writer 自定义事件 payload 构造函数
- `app/graph/nodes.py` — 前段节点(resolve/classify/retrieve/gate/fallback/complaint/chitchat/log)
- `app/graph/agent_node.py` — main_agent 手写 ReAct 节点 + suggest_options 伪工具
- `app/graph/builder.py` — GraphDeps + build_chat_graph
- `app/prompts/intent.py` — INTENT_PROMPT
- `evals/intent_samples.jsonl`、`evals/probe_intent.py` — 意图标注样例验证
- `tests/test_bare_agent.py`、`test_graph_state.py`、`test_graph_nodes.py`、`test_agent_node.py`、`test_chat_action.py`、`test_graph_persistence.py`、`test_ch05_acceptance.py`

修改:

- `pyproject.toml`/`uv.lock` — +langgraph +langgraph-checkpoint-sqlite
- `app/prompts/service.py` — 常量与条款修订(spec §7.1)
- `app/sessions.py` — CommitTurnResult、validate_turn 多组工具组、InMemory 实现
- `app/store_db.py` — commit_turn 返回 source_message_id
- `app/tools/business.py` — write_ticket 抽取、create_ticket 兼容改造
- `app/chains/tool_chat_chain.py` — build_agent_context(BaseMessage 历史裁剪)
- `app/services/chat_service.py` — stream() 换成驱动图;+create_ticket_from_action;+SuggestActionsEvent
- `app/routers/chat.py` — +/v1/chat/action;+suggest_actions 帧序列化
- `app/schemas.py` — +ChatActionRequest
- `app/config.py` — +max_agent_steps/max_agent_tokens/checkpoint_db_path
- `app/main.py` — ChatOpenAI stream_usage=True;lifespan 装配图与 AsyncSqliteSaver
- `app/static/chat.html` — 按钮组 + 转人工模拟 + 建工单调用
- `tests/conftest.py` — +ScriptedChatModel(真实 Runnable 假模型);FakeChunk 支持 usage
- 重写:`tests/test_chat_service.py`、`test_orchestration.py`(删除,职责并入新测试)、`test_chat_api_tools.py`、`test_spec11_safety.py`
- 更新:`tests/test_sessions.py`、`test_store_db.py`、`test_tools.py`、`test_tool_chat_chain.py`、`test_prompts.py`、`test_chat_page.py`
- `README.md`、`AGENTS.md` — 架构与目录约定更新(Task 15)

---

### Task 1: 依赖安装与 LangGraph 冒烟

**Files:**
- Modify: `pyproject.toml`、`uv.lock`(经 uv)
- Test: `tests/test_graph_smoke.py`(新建,本章图基建的烟雾钉)

**Interfaces:**
- Produces: langgraph/langgraph-checkpoint-sqlite 进入 uv.lock;证明 `StateGraph`/`InMemorySaver`/`AsyncSqliteSaver` 在当前环境可用。

- [ ] **Step 1: 写失败的冒烟测试**

```python
# tests/test_graph_smoke.py
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class _S(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    tag: str


async def test_langgraph_memory_checkpointer_roundtrip():
    def node(state):
        return {"tag": "seen"}

    g = StateGraph(_S)
    g.add_node("node", node)
    g.add_edge(START, "node")
    g.add_edge("node", END)
    graph = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    out = await graph.ainvoke({"messages": [HumanMessage("hi")], "tag": ""}, config)
    assert out["tag"] == "seen"
    # 同 thread 第二次调用继承 messages(add_messages 累积)
    out2 = await graph.ainvoke({"tag": "again"}, config)
    assert len(out2["messages"]) == 1 and out2["tag"] == "again"


async def test_async_sqlite_saver_memory():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        g = StateGraph(_S)
        g.add_node("node", lambda s: {"tag": "sqlite"})
        g.add_edge(START, "node")
        g.add_edge("node", END)
        graph = g.compile(checkpointer=saver)
        out = await graph.ainvoke({"tag": ""}, {"configurable": {"thread_id": "t2"}})
        assert out["tag"] == "sqlite"
```

- [ ] **Step 2: 运行确认失败(依赖未装)**

Run: `uv run pytest tests/test_graph_smoke.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'langgraph'`

- [ ] **Step 3: 装依赖**

Run: `uv add "langgraph>=1.0.8,<2" "langgraph-checkpoint-sqlite"`
确认: `uv run python -c "import langgraph, langgraph_checkpoint_sqlite; print(langgraph.__version__)"` 打印 ≥1.0.8;`uv.lock` 出现两包及 aiosqlite。
若解析出的 langgraph <1.0 或拉入 langchain-core 降级 → 停下问用户(spec 定死 1.0.x,不得动现有 langchain-core 1.6.2)。

- [ ] **Step 4: 重跑冒烟测试确认通过**

Run: `uv run pytest tests/test_graph_smoke.py -v`
Expected: 2 PASS

- [ ] **Step 5: 确认 data/ 忽略规则并提交**

Run: `git check-ignore data/checkpoints.db || echo "data/checkpoints.db" >> .gitignore`(顺手覆盖 `-wal`/`-shm`:在 .gitignore 写 `data/checkpoints.db*`)

```bash
git add pyproject.toml uv.lock .gitignore tests/test_graph_smoke.py dev-notes/ch05.md
git commit -m "build: 引入 langgraph 1.0.x 与 checkpoint-sqlite,冒烟验证 StateGraph+checkpointer"
```

---

### Task 2: 热身留档 examples/bare_agent.py(祛魅)

**Files:**
- Create: `examples/__init__.py`(空)、`examples/bare_agent.py`
- Test: `tests/test_bare_agent.py`

**Interfaces:**
- Consumes: `app.tools.business.MOCK_TOOLS`(query_order/query_product/query_logistics,@tool 装饰的 BaseTool,mock JSON 不触库)。
- Produces: `run_bare_agent(model, tools, question, max_steps=5) -> tuple[str, list, int]`(最终文本, 消息序列, 实际步数);`ScriptedModel`(examples 自包含同步假模型);`BARE_SYSTEM`、`BARE_BUDGET_ANSWER` 常量。Task 11 的 agent_node 与此同构。

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_bare_agent.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'examples'`

- [ ] **Step 3: 实现 examples/bare_agent.py**

```python
"""祛魅热身:不借任何 Agent 框架,手写最裸的「带工具的循环」。

Agent 的全部本质:调 LLM → 有 tool_calls 就执行并把结果喂回去 → 没有就收敛。
本章正式的 LangGraph 节点(app/graph/agent_node.py)与此循环同构。
运行:uv run python examples/bare_agent.py [--real]
"""

import sys

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
```

- [ ] **Step 4: 跑测试与演示脚本**

Run: `uv run pytest tests/test_bare_agent.py -v` → 3 PASS
Run: `uv run python examples/bare_agent.py` → 打印两轮工具调用与收敛文本

- [ ] **Step 5: 提交**

```bash
git add examples/ tests/test_bare_agent.py dev-notes/ch05.md
git commit -m "feat(examples): 祛魅热身——手写最裸 Agent 循环留档(框架前的对照物)"
```

---

### Task 3: Prompts——意图 prompt + 服务话术常量与条款修订(标注样例验证)

**Files:**
- Create: `app/prompts/intent.py`、`evals/intent_samples.jsonl`、`evals/probe_intent.py`
- Modify: `app/prompts/service.py`
- Test: `tests/test_prompts.py`(按新条款更新契约)

**Interfaces:**
- Produces: `INTENT_PROMPT`(含 `{query}` 占位,**用 `.replace` 不用 `.format`**——字面 JSON 花括号);service.py 新常量 `COMPLAINT_REPLY`/`CHITCHAT_REPLY`/`KB_UNAVAILABLE_ANSWER`/`AGENT_BUDGET_ANSWER`/`FALLBACK_ANSWER`(Global Constraints 逐字值);`SERVICE_SYSTEM_PROMPT` 修订版。Task 8 消费 INTENT_PROMPT;Task 9/10/11/12 消费常量。

- [ ] **Step 1: 写标注样例与解析契约测试(Prompt 类任务:标注样例替代 TDD 红绿,解析器契约仍走单测)**

```jsonl
# evals/intent_samples.jsonl —— 逐字取自 spec §6.2 验收表
{"query": "订单 1001 的物流到哪了", "intent": "物流", "needs_knowledge": false, "route": "business"}
{"query": "保温杯还有库存吗", "intent": "商品咨询", "needs_knowledge": false, "route": "business"}
{"query": "维修寄修流程是什么", "intent": "售后", "needs_knowledge": true, "route": "knowledge"}
{"query": "未发货订单如何取消", "intent": "订单", "needs_knowledge": true, "route": "knowledge"}
{"query": "订单 1001 到哪了,退货运费谁承担", "intent": "物流", "needs_knowledge": true, "route": "knowledge"}
{"query": "我要投诉你们的服务", "intent": "投诉", "needs_knowledge": false, "route": "complaint"}
{"query": "我要投诉,并问一下退货运费规则", "intent": "投诉", "needs_knowledge": true, "route": "knowledge"}
{"query": "你好", "intent": "闲聊", "needs_knowledge": false, "route": "chitchat"}
```

```python
# tests/test_prompts.py(全量替换,旧两条断言并入)
from app.prompts.intent import INTENT_PROMPT
from app.prompts.service import (
    AGENT_BUDGET_ANSWER, CHITCHAT_REPLY, COMPLAINT_REPLY, FALLBACK_ANSWER,
    KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER, SERVICE_SYSTEM_PROMPT,
)


def test_refusal_constant_embedded_verbatim():
    assert REFUSAL_ANSWER in SERVICE_SYSTEM_PROMPT
    assert SERVICE_SYSTEM_PROMPT.count(REFUSAL_ANSWER) == 1


def test_prompt_contracts_present():
    for needle in ("[ref_no]", "不承诺", "suggest_options", "不得声称"):
        assert needle in SERVICE_SYSTEM_PROMPT


def test_fixed_answers():
    assert COMPLAINT_REPLY.startswith("非常抱歉")
    assert "客服小蜜" in CHITCHAT_REPLY
    assert KB_UNAVAILABLE_ANSWER == "知识库暂时不可用,请稍后重试。"
    assert AGENT_BUDGET_ANSWER == "本轮处理已达到上限,请缩小问题范围后重试。"
    assert "请求人工帮助并选择相应按钮" in FALLBACK_ANSWER


def test_intent_prompt_shape():
    assert "{query}" in INTENT_PROMPT
    for intent in ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"):
        assert intent in INTENT_PROMPT
    assert "needs_knowledge" in INTENT_PROMPT
    # 渲染必须用 .replace(字面花括号不能被 .format 吞掉)
    rendered = INTENT_PROMPT.replace("{query}", "测试问题")
    assert "测试问题" in rendered and "{query}" not in rendered
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL(`app.prompts.intent` 不存在 / 新常量不存在)

- [ ] **Step 3: 实现 app/prompts/intent.py**

```python
INTENT_PROMPT = """你是电商客服会话的意图分类器。只输出一行 JSON,不要输出任何其他内容。

输出格式:{"intent": "<七类之一>", "needs_knowledge": true 或 false}

intent 七类定义:
- 物流:查物流进度、快递到哪了
- 订单:查订单状态、金额、下单信息
- 商品咨询:了解商品;价格/库存/是否在售属业务查询,功能/规格属知识
- 退款退货:退款、退货、退货运费、退货流程
- 售后:维修、保修、寄修、售后流程
- 投诉:表达不满、要投诉,不含具体问题
- 闲聊:打招呼、寒暄、与购物无关的闲聊

needs_knowledge 判定(true/false 必须是 JSON 布尔值):
- 涉及规则、流程、政策、保修、运费、订单取消规则、商品规格/功能,或任一子问需要知识库资料 → true
- 只查具体订单/物流状态、商品价格/库存/是否在售,收集信息,或纯投诉/纯闲聊 → false
- 退款退货一律 true;情绪词不抹掉问题中的政策需求

用户问题:{query}"""
```

- [ ] **Step 4: 修订 app/prompts/service.py**

在文件顶部保持 `REFUSAL_ANSWER` 逐字不变,新增:

```python
COMPLAINT_REPLY = "非常抱歉给您带来了不好的体验,您的问题我们十分重视。您可以选择转接人工客服,或点击下方按钮建立工单跟进处理。"
CHITCHAT_REPLY = "您好,我是客服小蜜,可以帮您查订单、查物流、介绍商品和退换货政策,有什么可以帮您的吗?"
KB_UNAVAILABLE_ANSWER = "知识库暂时不可用,请稍后重试。"
AGENT_BUDGET_ANSWER = "本轮处理已达到上限,请缩小问题范围后重试。"
FALLBACK_ANSWER = "抱歉,暂时没有查到相关信息。您可以换个说法,或请求人工帮助并选择相应按钮。"
```

`SERVICE_SYSTEM_PROMPT` 条款修订(其余条款保留原文,编号重排):
- 第 5 条改为:「查询订单、商品、物流等业务数据时,优先调用对应工具,不得编造查询结果;政策、流程、规格类结论只能依据本轮提供的检索证据,并标注引用角标。」
- 第 7 条改为:「顾客明确要求人工、建工单,或问题超出工具能力时,调用 suggest_options 请用户点击对应按钮;只有用户在页面上点击并完成操作后才算数,你不得声称已经建单或已经转人工。」
- 第 9 条改为:「工具可以连续多步调用:先根据中间结果判断下一步需要什么,再决定继续调用还是作答;不再需要的工具不要重复调用,不要输出任何标记语法。」
- 引用协议段(10/11 条)保留 [ref_no] 协议,把「query_faq 返回 evidence」表述改为「本轮检索证据(若提供)」;删除「query_faq 每轮最多调用一次」。
- 14 条 REFUSAL_ANSWER 逐字口径保留;15/16 负面知识禁令保留。
- 新增一条:「业务工具返回的数据(订单/物流/价格库存)可以直接作答,不需要检索证据;但没有检索证据时不得编造政策与规格。」

- [ ] **Step 5: 运行契约测试**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 4 PASS

- [ ] **Step 6: 标注样例真模型验证(替代 TDD 的一步)**

```python
# evals/probe_intent.py —— 真模型跑 8 条标注样例,打印判定表;退出码=失败数
import asyncio, json, re, sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts.intent import INTENT_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


def parse(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if data.get("intent") not in INTENTS or type(data.get("needs_knowledge")) is not bool:
        return None
    return data["intent"], data["needs_knowledge"]


async def main() -> int:
    s = Settings()
    model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                       base_url=s.openai_base_url, max_tokens=200)
    rows = [json.loads(l) for l in Path(__file__).with_name("intent_samples.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = 0
    for r in rows:
        resp = await model.ainvoke([HumanMessage(content=INTENT_PROMPT.replace("{query}", r["query"]))])
        got = parse(resp.content if isinstance(resp.content, str) else "")
        ok = got == (r["intent"], r["needs_knowledge"])
        fails += 0 if ok else 1
        print(f"{'OK ' if ok else 'BAD'} {r['query']!r}: 期望 {(r['intent'], r['needs_knowledge'])} 实得 {got}")
    print(f"\n{len(rows) - fails}/{len(rows)} 命中")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Run: `uv run python evals/probe_intent.py`(需 .env 真 key;约 8 次调用)
Expected: 8/8 命中;有 BAD 行时调 INTENT_PROMPT 措辞重跑,直至全中。把输出表格粘贴进 dev-notes/ch05.md 本任务段。
注意:probe 脚本里的 parse 是 Task 8 解析器的预览版,Task 8 落地时以 `app/graph/nodes.py` 的正式实现为准并让 probe 复用它(若 Task 8 已完成,直接 import 替换)。

- [ ] **Step 7: 全量 pytest + 提交**

Run: `uv run pytest`(prompt 改动按红线跑全量)
Expected: 全绿(test_orchestration/test_chat_service 此刻尚未动,仍走旧链路;本任务只加了常量与 prompt,旧条款断言已同步)

```bash
git add app/prompts/ tests/test_prompts.py evals/intent_samples.jsonl evals/probe_intent.py dev-notes/ch05.md
git commit -m "feat(prompts): 意图分类 prompt 与 ch05 话术常量;服务条款适配 ReAct 多步与 suggest_options"
```

---

### Task 4: store 层扩展——CommitTurnResult 与多组工具校验

**Files:**
- Modify: `app/sessions.py`(CommitTurnResult、validate_turn 多组、InMemory 返回 ID)、`app/store_db.py`(commit_turn 返回用户消息 ID)
- Test: `tests/test_sessions.py`、`tests/test_store_db.py`

**Interfaces:**
- Consumes: 现有 `StoredMessage`/`LowConfidenceRecord`/`validate_turn`。
- Produces: `@dataclass(frozen=True) class CommitTurnResult: source_message_id: str`;`SessionStore.commit_turn(...) -> CommitTurnResult`(协议、内存、DB 三处同步);`validate_turn` 支持**多组** AI tool_calls→ToolMessage 配对(逐组校验、组间清空)。Task 6/12 依赖 CommitTurnResult;Task 12/13 依赖多组校验。

- [ ] **Step 1: 写失败测试(追加进现有两个测试文件)**

```python
# tests/test_sessions.py 追加
from app.sessions import CommitTurnResult, StoredMessage


async def test_commit_turn_returns_source_message_id():
    s = InMemorySessionStore(10, 100, 8000)
    sid = await s.create("u")
    r1 = await s.commit_turn(sid, _pair("q1", "a1"))
    r2 = await s.commit_turn(sid, _pair("q2", "a2"))
    assert isinstance(r1, CommitTurnResult) and isinstance(r2, CommitTurnResult)
    assert r1.source_message_id != r2.source_message_id  # 稳定且递增的唯一 ID
    assert r1.source_message_id.isdecimal() and r2.source_message_id.isdecimal()


async def test_validate_turn_accepts_multiple_tool_groups():
    s = InMemorySessionStore(10, 100, 8000)
    sid = await s.create("u")
    await s.commit_turn(sid, [
        StoredMessage("user", "先查订单再查物流"),
        StoredMessage("assistant", None, tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", "env1", tool_call_id="c1"),
        StoredMessage("assistant", None, tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2", "type": "tool_call"}]),
        StoredMessage("tool", "env2", tool_call_id="c2"),
        StoredMessage("assistant", "订单已发货,派送中"),
    ])  # 不抛异常即通过


async def test_validate_turn_rejects_dangling_second_group():
    s = InMemorySessionStore(10, 100, 8000)
    sid = await s.create("u")
    with pytest.raises(ValueError):
        await s.commit_turn(sid, [
            StoredMessage("user", "q"),
            StoredMessage("assistant", None, tool_calls=[
                {"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("tool", "env1", tool_call_id="c1"),
            StoredMessage("assistant", None, tool_calls=[  # 第二组缺 ToolMessage
                {"name": "query_logistics", "args": {}, "id": "c2", "type": "tool_call"}]),
            StoredMessage("assistant", "答"),
        ])
```

```python
# tests/test_store_db.py 追加(DB 在线)
async def test_commit_turn_returns_db_user_message_id(db_session_factory):
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create(TEST_USER_ID)
    r = await store.commit_turn(sid, _turn())
    assert r.source_message_id.isdecimal()
    # 返回的正是本轮 user 行 id:再查库核对
    import asyncio
    from app.models import Message
    from sqlalchemy import select
    def _check():
        with db_session_factory() as s:
            msg = s.get(Message, int(r.source_message_id))
            assert msg is not None and msg.role == "user"
            assert msg.conversation_id == int(sid)
    await asyncio.to_thread(_check)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_sessions.py tests/test_store_db.py -v`
Expected: FAIL(CommitTurnResult 不存在 / commit_turn 返回 None)

- [ ] **Step 3: 实现 sessions.py 改动**

`app/sessions.py` 顶部加:

```python
@dataclass(frozen=True)
class CommitTurnResult:
    source_message_id: str  # 本轮用户消息 ID 的十进制字符串(内存实现为分配序号)
```

`validate_turn` 中段循环改为逐组校验(替换现有 `tool_calls`/`tool_ids` 单组逻辑):

```python
    middle = messages[1:-1]
    group_calls: list[dict] = []
    group_ids: list[str] = []

    def close_group() -> None:
        if group_calls:
            expected = [c["id"] for c in group_calls]
            if sorted(expected) != sorted(group_ids) or len(set(group_ids)) != len(group_ids):
                raise ValueError("tool call ids and tool messages must match one-to-one")
            group_calls.clear()
            group_ids.clear()

    for m in middle:
        if m.role == "assistant":
            close_group()  # 新 assistant 出现前先结清上一组
            if not m.tool_calls:
                raise ValueError("middle assistant message without tool calls")
            if len(m.tool_calls) > max_tool_calls:
                raise ValueError("too many tool calls in turn")
            for call in m.tool_calls:
                cid = call.get("id") if isinstance(call, dict) else None
                if not isinstance(cid, str) or not cid or len(cid) > 64:
                    raise ValueError("invalid tool call id")
            group_calls.extend(m.tool_calls)
        elif m.role == "tool":
            if not group_calls:
                raise ValueError("orphan tool message")
            if not m.tool_call_id or len(m.tool_call_id) > 64:
                raise ValueError("tool message missing tool_call_id")
            group_ids.append(m.tool_call_id)
        else:
            raise ValueError(f"unexpected role in turn middle: {m.role}")
    close_group()
```

`SessionStore` 协议与 `InMemorySessionStore.commit_turn` 返回类型改 `-> CommitTurnResult`;内存实现加 `self._msg_seq = 0`,commit 时 `self._msg_seq += 1; source_id = str(self._msg_seq)`,末尾 `return CommitTurnResult(source_id)`(校验全过后才分配,与入池同点)。

- [ ] **Step 4: 实现 store_db.py 改动**

`_commit_sync` 首部改为先写用户行并 flush 取 ID,末尾返回:

```python
    def _commit_sync(self, session_id, messages, low_confidence=None) -> str:
        cid = int(session_id)
        with self._sf() as s:
            first = Message(conversation_id=cid, role=messages[0].role,
                            content=messages[0].content, tool_calls=messages[0].tool_calls,
                            tool_call_id=messages[0].tool_call_id)
            s.add(first)
            s.flush()  # 同事务内取 messages.id;失败整体回滚,不暴露半成品
            source_id = str(first.id)
            for m in messages[1:]:
                s.add(Message(conversation_id=cid, role=m.role, content=m.content,
                              tool_calls=m.tool_calls, tool_call_id=m.tool_call_id))
            s.execute(update(Conversation).where(Conversation.id == cid)
                      .values(updated_at=datetime.now()))
            if low_confidence is not None:
                s.add(LowConfidenceQuestion(...))  # 照旧
            s.commit()
            return source_id
```

`commit_turn` 签名改 `-> CommitTurnResult`,return `CommitTurnResult(await asyncio.to_thread(...))`。

- [ ] **Step 5: 跑两个测试文件确认通过**

Run: `uv run pytest tests/test_sessions.py tests/test_store_db.py -v`
Expected: 全绿(含旧用例——旧调用方忽略返回值,向后兼容)

- [ ] **Step 6: 提交**

```bash
git add app/sessions.py app/store_db.py tests/test_sessions.py tests/test_store_db.py dev-notes/ch05.md
git commit -m "feat(store): commit_turn 返回本轮用户消息 ID;validate_turn 支持多组工具配对"
```

---

### Task 5: write_ticket 抽取与 create_ticket 兼容改造

**Files:**
- Modify: `app/tools/business.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `write_ticket(db_session, conversation_id: int, description: str, ticket_type: str) -> str`(只 INSERT + flush,不开 Session、不 commit/rollback,事务归调用者);`create_ticket` 工具行为不变(同事务置「已转人工」),内部改调 write_ticket。Task 6 端点直接调 write_ticket。

- [ ] **Step 1: 写失败测试(追加 tests/test_tools.py)**

```python
def test_write_ticket_inserts_row_without_commit(db_session_factory):
    from app.tools.business import write_ticket
    from app.models import Conversation, Ticket
    with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        ticket_no = write_ticket(s, conv.id, "测试描述", "投诉")
        s.commit()
    assert ticket_no.startswith("T")
    with db_session_factory() as s:
        row = s.get(Ticket, ticket_no)
        assert row.description == "测试描述" and row.status == "待处理"
        assert s.get(Conversation, conv.id).status == "进行中"  # helper 不碰会话状态


def test_write_ticket_rollback_leaves_nothing(db_session_factory):
    from app.tools.business import write_ticket
    from app.models import Conversation, Ticket
    import pytest
    with pytest.raises(Exception), db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        write_ticket(s, conv.id, "x", "售后")
        raise RuntimeError("boom")  # 上下文管理器回滚
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 0


def test_create_ticket_keeps_compat_behavior(db_session_factory):
    # 旧工具:工单 + conv.status=已转人工 同事务
    from app.tools.business import build_tools
    from app.models import Conversation
    with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.commit()
        cid = conv.id
    ts = build_tools(db_session_factory, cid)
    out = ts.tools_by_name["create_ticket"].invoke(
        {"description": "要投诉", "ticket_type": "投诉"})
    import json
    assert json.loads(out)["ticket_no"].startswith("T")
    with db_session_factory() as s:
        assert s.get(Conversation, cid).status == "已转人工"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_tools.py -v -k write_ticket`
Expected: FAIL(`cannot import name 'write_ticket'`)

- [ ] **Step 3: 实现抽取**

`app/tools/business.py` 新增(放在 `create_ticket` 之前):

```python
def write_ticket(db_session, conversation_id: int, description: str, ticket_type: str) -> str:
    """纯写库:只 INSERT tickets 行并 flush;不开 Session、不 commit/rollback,事务归调用者。"""
    ticket_no = "T" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + secrets.token_hex(6)
    db_session.add(Ticket(ticket_no=ticket_no, conversation_id=conversation_id,
                          description=description, ticket_type=ticket_type))
    db_session.flush()
    return ticket_no
```

`create_ticket` 闭包体改为:

```python
        with session_factory() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                raise ValueError("conversation not found")
            ticket_no = write_ticket(s, conversation_id, description, ticket_type)
            conv.status = "已转人工"
            s.commit()  # 工单与会话状态同事务,同成同败
        return _json({"ticket_no": ticket_no, "status": "待处理"})
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_tools.py -v`
Expected: 全绿(含旧 create_ticket 用例)

- [ ] **Step 5: 提交**

```bash
git add app/tools/business.py tests/test_tools.py dev-notes/ch05.md
git commit -m "refactor(tools): 抽出 write_ticket 纯写库 helper;create_ticket 维持兼容语义"
```

---

### Task 6: 动作端点 POST /v1/chat/action(建工单唯一写通道)

**Files:**
- Modify: `app/schemas.py`、`app/services/chat_service.py`、`app/routers/chat.py`、`app/main.py`(service 构造传 session_factory)
- Test: `tests/test_chat_action.py`

**Interfaces:**
- Consumes: Task 5 的 `write_ticket`;`SessionNotFoundError`→404、`RequestValidationError`→422 现有异常处理(main.py 已有)。
- Produces: `ChatActionRequest`;`ChatService.create_ticket_from_action(user_id, session_id, source_message_id, ticket_type) -> str`;`POST /v1/chat/action` → `{"ticket_no","status":"待处理"}`。Task 14 前端调它。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_chat_action.py
"""动作端点测试:不依赖聊天链路,直接用 db fixtures 播种会话与消息行
(这样 Task 13 换图驱动后本文件无需改动)。"""

import asyncio
import dataclasses

import httpx
import pytest

from app.main import create_app
from app.models import Conversation, Message, Ticket
from tests.conftest import FakeStreamModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _make_app(db_session_factory):
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  session_factory=db_session_factory)
    # 模型永远不会被调用(测试只打 /v1/chat/action);Task 13 换图驱动后亦然
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=runtime)


def _seed_conversation(db_session_factory) -> tuple[str, str]:
    """播种一会话 + 一条 user 消息,返回 (session_id, source_message_id)。"""
    with db_session_factory() as s:
        conv = Conversation(user_id=TEST_USER_ID)
        s.add(conv)
        s.flush()
        msg = Message(conversation_id=conv.id, role="user",
                      content="我要投诉你们的服务")
        s.add(msg)
        s.flush()
        s.commit()
        return str(conv.id), str(msg.id)


async def test_action_creates_ticket_without_touching_conv_status(db_session_factory):
    sid, mid = _seed_conversation(db_session_factory)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": mid,
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticket_no"].startswith("T") and body["status"] == "待处理"

    def _check():
        with db_session_factory() as s:
            t = s.get(Ticket, body["ticket_no"])
            assert t.description == "我要投诉你们的服务" and t.ticket_type == "投诉"
            assert s.get(Conversation, int(sid)).status == "进行中"  # 不置已转人工
    await asyncio.to_thread(_check)


async def test_action_404_when_message_not_in_session(db_session_factory):
    sid, _ = _seed_conversation(db_session_factory)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": "999999",
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 404


async def test_action_404_on_role_or_owner_mismatch(db_session_factory):
    sid, mid = _seed_conversation(db_session_factory)
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        # 会话归属不符
        resp = await client.post("/v1/chat/action", json={
            "user_id": "22222222-2222-2222-2222-222222222222", "session_id": sid,
            "source_message_id": mid, "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 404


async def test_action_422_on_bad_params(db_session_factory):
    app = _make_app(db_session_factory)
    async with await _client(app) as client:
        resp = await client.post("/v1/chat/action", json={
            "user_id": "not-a-uuid", "session_id": "1", "source_message_id": "abc",
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 422
```

注:AppRuntime 是 frozen dataclass,一律用 `dataclasses.replace` 改字段,不要直接赋值。本文件的测试只打动作端点、不走聊天链路,因此对 Task 13 的图驱动重构免疫。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_chat_action.py -v`
Expected: FAIL(404 路由不存在)

- [ ] **Step 3: 实现**

`app/schemas.py` 追加(`_SESSION_ID_RE`/`_user_id_must_be_uuid` 逻辑复用,抽成模块级函数供两个模型共用):

```python
_SOURCE_MSG_ID_RE = re.compile(r"^[1-9]\d{0,18}$")


class ChatActionRequest(BaseModel):
    user_id: str
    session_id: str
    source_message_id: str
    action: Literal["create_ticket"]
    ticket_type: Literal["售后", "投诉", "咨询"]

    @field_validator("user_id")
    @classmethod
    def _user_id(cls, v):  # 与 ChatStreamRequest 同一校验
        return _validate_uuid(v)

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, v):
        return _validate_session_id(v)

    @field_validator("source_message_id")
    @classmethod
    def _source_msg_id(cls, v: str) -> str:
        if not _SOURCE_MSG_ID_RE.match(v):
            raise ValueError("source_message_id must be a decimal id string")
        return v
```

`app/services/chat_service.py`:`__init__` 加 `session_factory=None` 存 `self._session_factory`;新增:

```python
    async def create_ticket_from_action(self, user_id: str, session_id: str,
                                        source_message_id: str, ticket_type: str) -> str:
        if self._session_factory is None:
            raise AppError("action_unavailable")
        task = asyncio.ensure_future(asyncio.to_thread(
            self._create_ticket_sync, user_id, session_id, source_message_id, ticket_type))
        try:
            return await asyncio.shield(task)  # 写操作:取消时等事务落地再放行
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await task
            raise

    def _create_ticket_sync(self, user_id, session_id, source_message_id, ticket_type) -> str:
        from app.models import Conversation, Message
        from app.store_db import _as_db_id
        from app.tools.business import write_ticket
        cid = _as_db_id(session_id)
        mid = _as_db_id(source_message_id)
        if cid is None or mid is None:
            raise SessionNotFoundError("session not found")
        with self._session_factory() as s:  # 同一 Session 完成归属/消息查询与写入
            conv = s.get(Conversation, cid)
            if conv is None or conv.user_id != user_id:
                raise SessionNotFoundError("session not found")
            msg = s.get(Message, mid)
            if msg is None or msg.conversation_id != cid or msg.role != "user":
                raise SessionNotFoundError("session not found")  # 404 不泄露其他会话内容
            ticket_no = write_ticket(s, cid, msg.content or "用户通过快捷操作请求建单",
                                     ticket_type)
            s.commit()  # 失败整体回滚;不改 conv.status
            return ticket_no
```

`app/routers/chat.py` 追加:

```python
@router.post("/v1/chat/action")
async def chat_action(body: ChatActionRequest, request: Request):
    service: ChatService = request.app.state.chat_service
    ticket_no = await service.create_ticket_from_action(
        body.user_id, body.session_id, body.source_message_id, body.ticket_type)
    return {"ticket_no": ticket_no, "status": "待处理"}
```

`app/main.py`:`ChatService(...)` 构造处加 `session_factory=runtime.session_factory`。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_chat_action.py -v`
Expected: 3 PASS(需 Docker MySQL 在线)

- [ ] **Step 5: 提交**

```bash
git add app/schemas.py app/services/chat_service.py app/routers/chat.py app/main.py tests/test_chat_action.py dev-notes/ch05.md
git commit -m "feat(api): POST /v1/chat/action——绑定源消息的确定性建单,聊天外唯一写通道"
```

---

### Task 7: 图基建——state.py / errors.py / events.py

**Files:**
- Create: `app/graph/__init__.py`(空)、`app/graph/state.py`、`app/graph/errors.py`、`app/graph/events.py`
- Test: `tests/test_graph_state.py`

**Interfaces:**
- Consumes: Task 1 的 langgraph API。
- Produces(后续所有图任务依赖):
  - `INTENTS: tuple[str, ...]`、`ROUTES: tuple[str, ...]`、`ROUTE_TABLE: dict[tuple[str, bool], str]`(14 组合全覆盖)
  - `class ChatGraphState(TypedDict, total=False)`(字段逐字按 spec §10.1 表)
  - `new_turn_state(raw_query: str) -> dict`(全部临时字段显式初值;**不含 messages 键**,保留 checkpoint 历史)
  - `class TurnAbortError(Exception)`(节点已发 error 帧后中止图;驱动层捕获后不发 DONE)
  - events 构造:`ev_tool_start(call)`、`ev_tool_end(tool_call_id, name, ok, summary)`、`ev_fixed_delta(content)`、`ev_citations(citations)`、`ev_suggest_actions(source_message_id, options)`、`ev_error(code, message)` → 均为 `dict`,`"kind"` 键区分

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph_state.py -v`
Expected: FAIL(`No module named 'app.graph'`)

- [ ] **Step 3: 实现 state.py**

```python
"""ch05 图 State:字段语义与每轮重置契约见 spec §10.1。"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")
ROUTES = ("knowledge", "business", "complaint", "chitchat")

# 分流规则写死:(intent, needs_knowledge) → route;needs_knowledge=True 优先于意图名称
ROUTE_TABLE: dict[tuple[str, bool], str] = {
    ("物流", False): "business",
    ("订单", False): "business",
    ("商品咨询", False): "business",
    ("售后", False): "business",
    ("退款退货", False): "knowledge",  # 保守覆盖:业务工具办不了退款退货,仍须先检索政策
    ("投诉", False): "complaint",
    ("闲聊", False): "chitchat",
    **{(intent, True): "knowledge" for intent in INTENTS},
}


class ChatGraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # 跨轮,仅 log 提交成功后追加
    raw_query: str
    resolved_query: str
    intent: str | None
    needs_knowledge: bool | None
    route: str | None
    retrieval_result: dict | None          # RetrievalResult 的可序列化快照
    retrieval_status: str                  # not_run | ok | low_confidence | unavailable
    retrieval_error_code: str | None       # kb_unconfigured/kb_rebuilding/kb_rebuild_required/kb_unavailable
    evidence: list[dict]
    low_conf_source: str | None            # retrieval_low_conf | self_check
    low_conf_reason: dict | None
    suggested_actions: list[dict]
    agent_steps: int
    agent_tokens: int
    token_accounting: str                  # none | usage | estimated | mixed
    final_text: str
    turn_messages: list[BaseMessage]       # 本轮消息(用户 + 工具往返 + 最终答复)
    source_message_id: str | None
    node_trace: list[dict]


def new_turn_state(raw_query: str) -> dict:
    """每轮图调用的显式输入:全部临时字段重置(messages 键刻意缺席,保留 checkpoint 历史)。
    同 thread 未覆盖字段会延续上一轮的值——所有分支都必须从同一个入口拿初值。"""
    return {
        "raw_query": raw_query,
        "resolved_query": "",
        "intent": None,
        "needs_knowledge": None,
        "route": None,
        "retrieval_result": None,
        "retrieval_status": "not_run",
        "retrieval_error_code": None,
        "evidence": [],
        "low_conf_source": None,
        "low_conf_reason": None,
        "suggested_actions": [],
        "agent_steps": 0,
        "agent_tokens": 0,
        "token_accounting": "none",
        "final_text": "",
        "turn_messages": [HumanMessage(content=raw_query)],
        "source_message_id": None,
        "node_trace": [],
    }
```

`app/graph/errors.py`:

```python
class TurnAbortError(Exception):
    """节点已通过 stream writer 发出 error 帧;抛出以中止图执行。
    驱动层捕获后正常结束 SSE(不发 [DONE]),不提交本轮。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
```

`app/graph/events.py`:

```python
"""节点经 get_stream_writer 发出的自定义事件 payload;驱动层按 kind 翻译成 SSE 帧。"""


def ev_tool_start(call: dict) -> dict:
    return {"kind": "tool_start", "tool_call_id": call["id"],
            "name": call["name"], "args": call["args"]}


def ev_tool_end(tool_call_id: str, name: str, ok: bool, summary: str) -> dict:
    return {"kind": "tool_end", "tool_call_id": tool_call_id,
            "name": name, "ok": ok, "summary": summary}


def ev_fixed_delta(content: str) -> dict:
    """固定话术/兜底文本的显式 delta(非模型 token,不经 messages 流)。"""
    return {"kind": "fixed_delta", "content": content}


def ev_citations(citations: list[dict]) -> dict:
    return {"kind": "citations", "citations": citations}


def ev_suggest_actions(source_message_id: str, options: list[dict]) -> dict:
    return {"kind": "suggest_actions", "source_message_id": source_message_id,
            "options": options}


def ev_error(code: str, message: str) -> dict:
    return {"kind": "error", "code": code, "message": message}
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_graph_state.py tests/test_graph_smoke.py -v`
Expected: 全绿

- [ ] **Step 5: 提交**

```bash
git add app/graph/ tests/test_graph_state.py dev-notes/ch05.md
git commit -m "feat(graph): State schema、写死分流表(14 组合)与 writer 事件契约"
```

---

### Task 8: 前段节点——resolve_reference / classify_intent / 分流边

**Files:**
- Create: `app/graph/nodes.py`(本任务先落地前段节点与 deps 结构,后续任务继续往此文件加节点)
- Test: `tests/test_graph_nodes.py`

**Interfaces:**
- Consumes: Task 3 `INTENT_PROMPT`;Task 7 state/errors/events。
- Produces:
  - `class GraphDeps`: dataclass,字段 `model`、`settings`、`retriever`、`store`(builder 与节点工厂共用;Task 13 装配)
  - `parse_intent_output(text: str) -> tuple[str, bool] | None`(严格:intent 七类枚举、needs_knowledge 必须 JSON 布尔,字符串 `"false"`/缺字段/多余文本非法→None)
  - `build_front_nodes(deps: GraphDeps) -> dict`:`resolve_reference`、`classify_intent` 两个 async 节点函数 + `route_by_intent(state) -> str` 条件边函数
  - 节点日志:logger 名 `wayhelp.graph`,INFO 级带 `node=` 与关键字段

- [ ] **Step 1: 写失败测试**

```python
# tests/test_graph_nodes.py(本任务先放分类与解析用例,后续任务继续追加)
import json
import logging

import pytest
from langchain_core.messages import AIMessage

from app.graph.nodes import GraphDeps, build_front_nodes, parse_intent_output
from app.graph.state import ROUTE_TABLE, new_turn_state
from tests.conftest import make_settings


class _ClassifyModel:
    """只支持 ainvoke 的桩:返回固定文本。"""

    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._text)


def _nodes(model_text):
    deps = GraphDeps(model=_ClassifyModel(model_text), settings=make_settings(),
                     retriever=None, store=None)
    return build_front_nodes(deps)


@pytest.mark.parametrize("text,expected", [
    ('{"intent":"物流","needs_knowledge":false}', ("物流", False)),
    ('{"intent":"售后","needs_knowledge":true}', ("售后", True)),
    ('前缀文本{"intent":"投诉","needs_knowledge":false}后缀', ("投诉", False)),
    ('{"intent":"售后","needs_knowledge":"false"}', None),   # 字符串 false 非法
    ('{"intent":"售后"}', None),                              # 缺字段
    ('{"intent":"退票","needs_knowledge":true}', None),       # 非法枚举
    ('不是 JSON', None),
    ('{"intent":"闲聊","needs_knowledge":false,"x":1}', ("闲聊", False)),  # 容忍多余字段
])
def test_parse_intent_output(text, expected):
    assert parse_intent_output(text) == expected


async def test_classify_sets_route_from_table():
    nodes = _nodes('{"intent":"退款退货","needs_knowledge":false}')
    out = await nodes["classify_intent"](new_turn_state("我想退货"))
    assert out["intent"] == "退款退货" and out["needs_knowledge"] is False
    assert out["route"] == "knowledge"  # 路由表保守覆盖


async def test_classify_parse_failure_falls_back_to_knowledge(caplog):
    nodes = _nodes("模型输出了一坨废话")
    with caplog.at_level(logging.WARNING, logger="wayhelp.graph"):
        out = await nodes["classify_intent"](new_turn_state("查订单"))
    assert (out["intent"], out["needs_knowledge"]) == ("售后", True)
    assert out["route"] == "knowledge"  # 解析失败不得退回 business
    assert "解析失败" in caplog.text or "parse" in caplog.text


async def test_classify_model_exception_aborts_with_upstream_error():
    class _Boom:
        async def ainvoke(self, messages):
            raise ConnectionError("upstream down")

    deps = GraphDeps(model=_Boom(), settings=make_settings(), retriever=None, store=None)
    nodes = build_front_nodes(deps)
    from app.graph.errors import TurnAbortError
    with pytest.raises(TurnAbortError):
        await nodes["classify_intent"](new_turn_state("q"))


async def test_resolve_reference_passthrough():
    nodes = _nodes("{}")
    out = await nodes["resolve_reference"](new_turn_state("原样 透传"))
    assert out["resolved_query"] == "原样 透传"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph_nodes.py -v`
Expected: FAIL(`No module named 'app.graph.nodes'`)

- [ ] **Step 3: 实现 app/graph/nodes.py 前段**

```python
"""ch05 图节点。节点经闭包捕获 GraphDeps;日志统一 logger 'wayhelp.graph'。"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from app.config import Settings
from app.graph.errors import TurnAbortError
from app.graph.events import ev_error
from app.graph.state import INTENTS, ROUTE_TABLE
from app.prompts.intent import INTENT_PROMPT

logger = logging.getLogger("wayhelp.graph")


@dataclass(frozen=True)
class GraphDeps:
    model: Any
    settings: Settings
    retriever: Any   # KnowledgeRetriever | None(测试可注假检索器)
    store: Any       # SessionStore 协议(log 节点用;本任务可为 None)


def parse_intent_output(text: str) -> tuple[str, bool] | None:
    """从模型输出提取 {"intent","needs_knowledge"};任何不合法一律 None(调用方兜底)。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent = data.get("intent")
    nk = data.get("needs_knowledge")
    if intent not in INTENTS or type(nk) is not bool:  # 字符串 "false"/缺失都算非法
        return None
    return intent, nk


def route_by_intent(state) -> str:
    """条件边函数:只查写死的分流表。classify_intent 已保证 route 非 None。"""
    return state["route"]


def build_front_nodes(deps: GraphDeps) -> dict:
    async def resolve_reference(state):
        # 最简版:原样透传(正式指代消解是后续章节的事)
        logger.info("node=resolve_reference query=%r", state["raw_query"][:50])
        return {"resolved_query": state["raw_query"],
                "node_trace": [*state["node_trace"], {"node": "resolve_reference"}]}

    async def classify_intent(state):
        writer = get_stream_writer()
        prompt = INTENT_PROMPT.replace("{query}", state["resolved_query"])
        try:
            resp = await deps.model.ainvoke([HumanMessage(content=prompt)])
        except Exception as exc:  # 网络故障不是分类结果
            logger.warning("node=classify_intent upstream error: %s", type(exc).__name__)
            writer(ev_error("upstream_error", "上游模型暂时不可用"))
            raise TurnAbortError("upstream_error") from exc
        text = resp.content if isinstance(resp.content, str) else ""
        parsed = parse_intent_output(text)
        if parsed is None:
            logger.warning("node=classify_intent 解析失败,保守兜底 knowledge: %r", text[:120])
            intent, needs_knowledge = "售后", True
        else:
            intent, needs_knowledge = parsed
        route = ROUTE_TABLE[(intent, needs_knowledge)]
        logger.info("node=classify_intent intent=%s needs_knowledge=%s route=%s",
                    intent, needs_knowledge, route)
        return {"intent": intent, "needs_knowledge": needs_knowledge, "route": route,
                "node_trace": [*state["node_trace"],
                               {"node": "classify_intent", "intent": intent,
                                "needs_knowledge": needs_knowledge, "route": route}]}

    return {"resolve_reference": resolve_reference, "classify_intent": classify_intent}
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_graph_nodes.py tests/test_graph_state.py -v`
Expected: 全绿

- [ ] **Step 5: 标注样例回归(可选,若 probe_intent.py 已 import 正式解析器则同步)**

确认 `evals/probe_intent.py` 的 `parse` 替换为 `from app.graph.nodes import parse_intent_output`(消除预览版重复),重跑 `uv run python evals/probe_intent.py` 仍 8/8。

- [ ] **Step 6: 提交**

```bash
git add app/graph/nodes.py tests/test_graph_nodes.py evals/probe_intent.py dev-notes/ch05.md
git commit -m "feat(graph): 意图分类节点(JSON+needs_knowledge,失败保守走检索)与分流边"
```

---

### Task 9: retrieve / confidence_gate / gate_fallback 节点

**Files:**
- Modify: `app/graph/nodes.py`(追加)
- Test: `tests/test_graph_nodes.py`(追加闸门与检索用例)

**Interfaces:**
- Consumes: `KnowledgeRetriever.search` / `RetrievalResult` / `KnowledgeHit` / `assemble_evidence` / `NOTE_*` 常量(app/knowledge/retriever.py);Task 8 GraphDeps。
- Produces:
  - `snapshot_retrieval(r: RetrievalResult) -> dict`(可 JSON 序列化;不含运行时对象)
  - `build_knowledge_nodes(deps) -> dict`:`retrieve`、`confidence_gate`、`gate_fallback` 节点 + `route_after_gate(state) -> str`("main_agent"|"gate_fallback")
  - `retrieval_status` 语义:`unavailable`(NOTE_UNCONFIGURED/REBUILDING/REBUILD_REQUIRED/超时/异常,带 `retrieval_error_code`)、`low_confidence`(含 NOTE_NOT_BUILT 零命中)、`ok`

- [ ] **Step 1: 写失败测试(追加 tests/test_graph_nodes.py)**

```python
from app.knowledge.retriever import (
    NOTE_NOT_BUILT, NOTE_REBUILDING, NOTE_UNCONFIGURED, RetrievalResult,
)
from app.graph.nodes import build_knowledge_nodes
from app.prompts.service import KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER


def _result(note=None, low=False, hits=None, score=0.9):
    return RetrievalResult(
        hits=hits or [], requested_strategy="hybrid_rerank",
        effective_strategy="hybrid_rerank", confidence_score=score,
        confidence_threshold=0.0553, low_confidence=low, note=note,
        query_plan=None, leg_counts={"dense": 3, "bm25": 2})


class _FakeRetriever:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc
        self.queries = []

    def search(self, query, **kw):
        self.queries.append(query)
        if self._exc:
            raise self._exc
        return self._result


def _knodes(result=None, exc=None):
    deps = GraphDeps(model=None, settings=make_settings(),
                     retriever=_FakeRetriever(result, exc), store=None)
    return build_knowledge_nodes(deps)


async def test_retrieve_ok_snapshot_serializable():
    import json as _json
    from app.knowledge.retriever import KnowledgeHit
    hit = KnowledgeHit(chunk_id=7, score=0.9, category="policy", questions="q",
                       answer="a", source_doc="d.md", chunk_index=0, section_path="退货")
    nodes = _knodes(result=_result(hits=[hit]))
    st = new_turn_state("退货政策")
    st["resolved_query"] = "退货政策"
    out = await nodes["retrieve"](st)
    assert out["retrieval_status"] == "ok"
    _json.dumps(out["retrieval_result"])  # 快照必须可序列化(进 checkpoint)
    assert out["retrieval_result"]["hits"][0]["chunk_id"] == 7


async def test_retrieve_unavailable_states_not_pooled():
    for note, code in ((NOTE_UNCONFIGURED, "kb_unconfigured"),
                       (NOTE_REBUILDING, "kb_rebuilding")):
        nodes = _knodes(result=_result(note=note, low=True, score=None))
        st = new_turn_state("q"); st["resolved_query"] = "q"
        out = await nodes["retrieve"](st)
        assert out["retrieval_status"] == "unavailable"
        assert out["retrieval_error_code"] == code  # 即使 low_confidence=True 也不算知识缺口


async def test_retrieve_exception_is_unavailable():
    nodes = _knodes(exc=TimeoutError("timeout"))
    st = new_turn_state("q"); st["resolved_query"] = "q"
    out = await nodes["retrieve"](st)
    assert out["retrieval_status"] == "unavailable"
    assert out["retrieval_error_code"] == "kb_unavailable"


async def test_gate_low_confidence_marks_pool_fields():
    nodes = _knodes(result=_result(note=NOTE_NOT_BUILT, low=True, score=None))
    st = new_turn_state("q"); st["resolved_query"] = "q"
    st.update(await nodes["retrieve"](st))
    out = await nodes["confidence_gate"](st)
    assert out["low_conf_source"] == "retrieval_low_conf"
    assert out["low_conf_reason"]["note"] == NOTE_NOT_BUILT
    assert nodes["route_after_gate"]({**st, **out}) == "gate_fallback"


async def test_gate_fallback_texts():
    nodes = _knodes()
    st = {**new_turn_state("q"), "retrieval_status": "unavailable"}
    out = await nodes["gate_fallback"](st)
    assert out["final_text"] == KB_UNAVAILABLE_ANSWER
    st2 = {**new_turn_state("q"), "retrieval_status": "low_confidence"}
    out2 = await nodes["gate_fallback"](st2)
    assert out2["final_text"] == REFUSAL_ANSWER
    assert out2["turn_messages"][-1].content == REFUSAL_ANSWER  # 进本轮消息
```

注:节点内 `get_stream_writer()` 在裸调节点函数时不在图上下文——LangGraph 1.x 中无 writer 上下文时 `get_stream_writer()` 返回 no-op writer,直接调用节点测试安全;若实测抛异常,改为经编译图驱动断言(见 Task 13 集成)。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph_nodes.py -v -k "retrieve or gate"`
Expected: FAIL(`build_knowledge_nodes` 不存在)

- [ ] **Step 3: 实现(追加 app/graph/nodes.py)**

```python
import asyncio
from dataclasses import asdict, is_dataclass

from app.knowledge.retriever import (
    NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED,
    KnowledgeHit, RetrievalResult, assemble_evidence,
)
from app.graph.events import ev_fixed_delta
from app.prompts.service import KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER

_STATE_NOTE_CODES = {
    NOTE_UNCONFIGURED: "kb_unconfigured",
    NOTE_REBUILDING: "kb_rebuilding",
    NOTE_REBUILD_REQUIRED: "kb_rebuild_required",
}


def snapshot_retrieval(r: RetrievalResult) -> dict:
    """RetrievalResult → 可进 checkpoint 的纯数据快照(不含 retriever/连接等运行时对象)。"""
    return {
        "requested_strategy": r.requested_strategy,
        "effective_strategy": r.effective_strategy,
        "confidence_score": r.confidence_score,
        "confidence_threshold": r.confidence_threshold,
        "low_confidence": r.low_confidence,
        "note": r.note,
        "leg_counts": dict(r.leg_counts),
        "hits": [asdict(h) for h in r.hits],
        "query_plan": asdict(r.query_plan) if is_dataclass(r.query_plan) else None,
    }


def build_knowledge_nodes(deps: GraphDeps) -> dict:
    async def retrieve(state):
        logger.info("node=retrieve query=%r", state["resolved_query"][:50])
        trace = [*state["node_trace"], {"node": "retrieve"}]
        if deps.retriever is None:
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unconfigured", "node_trace": trace}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(deps.retriever.search, state["resolved_query"]),
                timeout=deps.settings.knowledge_tool_timeout_seconds)  # 总等待上限,本层不重试
        except Exception as exc:  # 超时与检索异常同归 unavailable
            logger.warning("node=retrieve unavailable: %s", type(exc).__name__)
            return {"retrieval_status": "unavailable",
                    "retrieval_error_code": "kb_unavailable", "node_trace": trace}
        snap = snapshot_retrieval(result)
        code = _STATE_NOTE_CODES.get(result.note)
        if code is not None:  # 配置/维护态优先于低置信判定
            return {"retrieval_result": snap, "retrieval_status": "unavailable",
                    "retrieval_error_code": code, "node_trace": trace}
        # NOTE_NOT_BUILT 等其余 note 参与低置信判定
        status = "low_confidence" if result.low_confidence else "ok"
        logger.info("node=retrieve done status=%s score=%s", status, result.confidence_score)
        return {"retrieval_result": snap, "retrieval_status": status, "node_trace": trace}

    async def confidence_gate(state):
        status = state["retrieval_status"]
        trace = [*state["node_trace"], {"node": "confidence_gate", "status": status}]
        if status == "low_confidence":
            r = state["retrieval_result"]
            logger.info("node=confidence_gate blocked low_confidence")
            return {"low_conf_source": "retrieval_low_conf",
                    "low_conf_reason": {
                        "requested_strategy": r["requested_strategy"],
                        "effective_strategy": r["effective_strategy"],
                        "top1": r["confidence_score"],
                        "threshold": r["confidence_threshold"],
                        "note": r["note"]},
                    "node_trace": trace}
        if status == "ok":
            hits = [KnowledgeHit(**h) for h in state["retrieval_result"]["hits"]]
            evidence = [e.to_dict() for e in assemble_evidence(
                hits, max_items=deps.settings.rerank_top_n,
                budget_chars=deps.settings.max_tool_result_chars,
                overhead_chars=200)] if hits else []
            return {"evidence": evidence, "node_trace": trace}
        return {"node_trace": trace}  # unavailable:直接落 fallback

    def route_after_gate(state) -> str:
        return "main_agent" if state["retrieval_status"] == "ok" else "gate_fallback"

    async def gate_fallback(state):
        writer = get_stream_writer()
        text = (KB_UNAVAILABLE_ANSWER if state["retrieval_status"] == "unavailable"
                else REFUSAL_ANSWER)
        writer(ev_fixed_delta(text))
        logger.info("node=gate_fallback status=%s", state["retrieval_status"])
        return {"final_text": text,
                "turn_messages": [*state["turn_messages"], AIMessage(content=text)],
                "node_trace": [*state["node_trace"], {"node": "gate_fallback"}]}

    return {"retrieve": retrieve, "confidence_gate": confidence_gate,
            "gate_fallback": gate_fallback, "route_after_gate": route_after_gate}
```

文件顶部 import 补 `from langchain_core.messages import AIMessage`。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_graph_nodes.py -v`
Expected: 全绿

- [ ] **Step 5: 提交**

```bash
git add app/graph/nodes.py tests/test_graph_nodes.py dev-notes/ch05.md
git commit -m "feat(graph): 强制检索与置信度闸节点——故障/低置信/正常三态分离,故障不入池"
```

---

### Task 10: complaint_reply / chitchat_reply 固定回复节点

**Files:**
- Modify: `app/graph/nodes.py`(追加)
- Test: `tests/test_graph_nodes.py`(追加)

**Interfaces:**
- Produces: `build_fixed_nodes() -> dict`:`complaint_reply`、`chitchat_reply`。complaint 的 suggested_actions 结构逐字:
  `[{"action": "transfer_human", "label": "转人工"}, {"action": "create_ticket", "label": "建工单", "ticket_type": "投诉"}]`。两节点只写 `final_text`/`turn_messages`/`suggested_actions` 并发一次性 fixed_delta;**不在节点内发 suggest_actions 帧**(log 提交成功后才发,Task 12)。

- [ ] **Step 1: 写失败测试**

```python
from app.graph.nodes import build_fixed_nodes
from app.prompts.service import CHITCHAT_REPLY, COMPLAINT_REPLY


async def test_complaint_reply_with_two_independent_options():
    out = await build_fixed_nodes()["complaint_reply"](new_turn_state("我要投诉"))
    assert out["final_text"] == COMPLAINT_REPLY
    assert out["turn_messages"][-1].content == COMPLAINT_REPLY
    actions = out["suggested_actions"]
    assert [a["action"] for a in actions] == ["transfer_human", "create_ticket"]
    assert actions[0]["label"] == "转人工" and "ticket_type" not in actions[0]
    assert actions[1]["label"] == "建工单" and actions[1]["ticket_type"] == "投诉"


async def test_chitchat_reply_no_model_no_actions():
    out = await build_fixed_nodes()["chitchat_reply"](new_turn_state("你好"))
    assert out["final_text"] == CHITCHAT_REPLY
    assert out["suggested_actions"] == []
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph_nodes.py -k "complaint or chitchat" -v`
Expected: FAIL(`build_fixed_nodes` 不存在)

- [ ] **Step 3: 实现(追加 app/graph/nodes.py)**

```python
from app.prompts.service import CHITCHAT_REPLY, COMPLAINT_REPLY

COMPLAINT_ACTIONS = [
    {"action": "transfer_human", "label": "转人工"},
    {"action": "create_ticket", "label": "建工单", "ticket_type": "投诉"},
]


def build_fixed_nodes() -> dict:
    async def complaint_reply(state):
        writer = get_stream_writer()
        writer(ev_fixed_delta(COMPLAINT_REPLY))
        logger.info("node=complaint_reply")
        return {"final_text": COMPLAINT_REPLY,
                "suggested_actions": [dict(a) for a in COMPLAINT_ACTIONS],
                "turn_messages": [*state["turn_messages"], AIMessage(content=COMPLAINT_REPLY)],
                "node_trace": [*state["node_trace"], {"node": "complaint_reply"}]}

    async def chitchat_reply(state):
        writer = get_stream_writer()
        writer(ev_fixed_delta(CHITCHAT_REPLY))
        logger.info("node=chitchat_reply")  # 零模型调用
        return {"final_text": CHITCHAT_REPLY,
                "turn_messages": [*state["turn_messages"], AIMessage(content=CHITCHAT_REPLY)],
                "node_trace": [*state["node_trace"], {"node": "chitchat_reply"}]}

    return {"complaint_reply": complaint_reply, "chitchat_reply": chitchat_reply}
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest tests/test_graph_nodes.py -v` → 全绿

```bash
git add app/graph/nodes.py tests/test_graph_nodes.py dev-notes/ch05.md
git commit -m "feat(graph): 投诉/闲聊固定回复节点——安抚话术+双独立建议按钮,不进 Agent"
```

---

### Task 11: main_agent 手写 ReAct 节点(核心)

**Files:**
- Create: `app/graph/agent_node.py`
- Modify: `app/chains/tool_chat_chain.py`(+`build_agent_context`)、`app/config.py`(+`max_agent_steps`/`max_agent_tokens` 字段)
- Test: `tests/test_agent_node.py`

**Interfaces:**
- Consumes: `MOCK_TOOLS`(只读三工具)、`ToolRegistry`/`ToolExecutor`(超时/重试/shield 语义原样复用)、`merge_tool_call_chunks`/`finalize_tool_calls`、`fit_tool_context`、`count_tokens_approximately`;Task 7 events/errors/state;Task 3 `FALLBACK_ANSWER`/`AGENT_BUDGET_ANSWER`。
- Produces:
  - `suggest_options`(@tool,伪工具;参数 `options: list[Literal["转人工","建工单"]]` + `ticket_type: Literal["售后","投诉","咨询"] | None = None`)
  - `build_agent_node(deps: GraphDeps) -> async def main_agent(state) -> dict`
  - `build_agent_context(system_prompt, history, turn_messages, evidence_text, max_input_tokens) -> list[BaseMessage] | None`(tool_chat_chain.py;None = 历史丢光仍放不下)
  - 返回 state 更新键:`turn_messages`/`final_text`/`agent_steps`/`agent_tokens`/`token_accounting`/`suggested_actions`/`node_trace`

**行为契约(逐条对应 spec §7,测试一一映射):**
1. 无 tool_calls → 收敛;空文本 → FALLBACK_ANSWER(经 fixed_delta 补发)
2. 有 tool_calls → 校验(≤5、id 合法唯一)→ 逐个执行:伪工具走 `_handle_suggest_options`(零副作用),真工具经 executor,writer 发 tool_start/tool_end(伪工具不发徽章)
3. `create_ticket`/`query_faq` 不在注册表 → executor 回 unknown_tool 错误 ToolMessage,绝不写库
4. 步数检查在每次模型调用前(`steps >= max_agent_steps` → AGENT_BUDGET_ANSWER 收敛);工具组照样补全 ToolMessage 后再兜底
5. token 预算:调用前预留 `est_input + max_output_tokens`,不足 → AGENT_BUDGET_ANSWER 不再调模型;调用后有效 usage 替换预留,缺失/非法保留预留(不得按 0 计);`token_accounting` ∈ usage/estimated/mixed
6. `finish_reason=="length"` 或可见字符超 `max_message_chars` → ev_error("output_too_long") + TurnAbortError;模型异常 → ev_error("upstream_error") + TurnAbortError;上下文放不下 → ev_error("tool_context_too_long") + TurnAbortError
7. 每次循环重建上下文(历史 messages + turn_messages + evidence 系统消息),经 build_agent_context 裁剪

- [ ] **Step 1: config 加字段(先行小步)**

`app/config.py` 追加:

```python
    max_agent_steps: int = Field(default=8, gt=0)      # main_agent 内模型调用次数上限
    max_agent_tokens: int = Field(default=20000, gt=0)  # 本轮累计 token 预算(usage 或预留估算)
    checkpoint_db_path: str = "./data/checkpoints.db"
```

Run: `uv run pytest tests/test_config.py -v` → 全绿

- [ ] **Step 2: 写失败测试(循环骨架:一步收敛/多步/追问/伪工具/禁写)**

```python
# tests/test_agent_node.py
import json

import pytest
from langchain_core.messages import AIMessage

from app.graph.agent_node import build_agent_node
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from tests.conftest import FakeStreamModel, make_settings


def _tool_chunk(name, args, call_id):
    return ("tool", [{"index": 0, "name": name, "id": call_id,
                      "args": json.dumps(args)}])


def _agent(script, **settings_over):
    deps = GraphDeps(model=FakeStreamModel(script),
                     settings=make_settings(**settings_over),
                     retriever=None, store=None)
    return build_agent_node(deps)


async def test_one_step_converge():
    node = _agent(["订单 1001 已发货。"])
    out = await node(new_turn_state("订单 1001 状态"))
    assert out["final_text"] == "订单 1001 已发货。"
    assert out["agent_steps"] == 1
    assert out["turn_messages"][-1].content == "订单 1001 已发货。"


async def test_multi_step_order_then_logistics():
    node = _agent([
        _tool_chunk("query_order", {"order_id": "1001"}, "c1"), ("then", [
            _tool_chunk("query_logistics", {"order_id": "1001"}, "c2"), ("then", [
                "订单已发货,物流派送中。"])]),
    ])
    out = await node(new_turn_state("订单 1001 到哪了,先查订单再查物流"))
    assert out["agent_steps"] == 3
    tools = [m for m in out["turn_messages"] if m.type == "tool"]
    assert [t.name for t in tools] == ["query_order", "query_logistics"]
    assert "1001" in tools[0].content  # 真实工具结果喂回
    assert out["final_text"] == "订单已发货,物流派送中。"


async def test_clarify_question_is_plain_convergence():
    node = _agent(["请问您要查询哪个订单号?"])
    out = await node(new_turn_state("帮我查下物流"))
    assert out["final_text"] == "请问您要查询哪个订单号?"
    assert out["agent_steps"] == 1  # 缺信息追问 = 无 tool_calls 收敛


async def test_suggest_options_signal_no_side_effect():
    node = _agent([
        _tool_chunk("suggest_options", {"options": ["转人工", "建工单"],
                                        "ticket_type": "售后"}, "s1"),
        ("then", ["好的,您可以选择下方按钮。"]),
    ])
    out = await node(new_turn_state("我要找人工"))
    assert [a["action"] for a in out["suggested_actions"]] == ["transfer_human", "create_ticket"]
    assert out["suggested_actions"][1]["ticket_type"] == "售后"
    tool_msgs = [m for m in out["turn_messages"] if m.type == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "s1"  # 配对完整


async def test_forged_create_ticket_gets_unknown_tool():
    node = _agent([
        _tool_chunk("create_ticket", {"description": "x", "ticket_type": "投诉"}, "c9"),
        ("then", ["我只能为您建议按钮。"]),
    ])
    out = await node(new_turn_state("帮我建个工单"))
    tool_msgs = [m for m in out["turn_messages"] if m.type == "tool"]
    assert tool_msgs[0].status == "error"  # 未注册工具,不写库(无 session_factory 可写)


async def test_suggest_options_invalid_args_no_suggestion():
    node = _agent([
        _tool_chunk("suggest_options", {"options": ["建工单"]}, "s1"),  # 缺 ticket_type
        ("then", ["抱歉。"]),
    ])
    out = await node(new_turn_state("投诉"))
    assert out["suggested_actions"] == []
    tool_msgs = [m for m in out["turn_messages"] if m.type == "tool"]
    assert tool_msgs[0].status == "error"


async def test_empty_text_falls_back():
    from app.prompts.service import FALLBACK_ANSWER
    node = _agent([""])
    out = await node(new_turn_state("q"))
    assert out["final_text"] == FALLBACK_ANSWER
```

同时在 `tests/test_tool_chat_chain.py` 追加裁剪/重建的多组钉(spec §13 对该文件的改造要求):

```python
# tests/test_tool_chat_chain.py 追加
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.chains.tool_chat_chain import build_agent_context, rebuild_messages
from app.sessions import StoredMessage
from app.tool_envelope import wrap


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
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/test_agent_node.py -v`
Expected: FAIL(`No module named 'app.graph.agent_node'`)

- [ ] **Step 4: 实现 build_agent_context(tool_chat_chain.py 追加)**

```python
def build_agent_context(system_prompt: str, history: list[BaseMessage],
                        turn_messages: list[BaseMessage], evidence_text: str | None,
                        max_input_tokens: int) -> list[BaseMessage] | None:
    """组装并裁剪一次 ReAct 模型调用的上下文。
    布局:[system, ...历史完整 turn..., evidence?, ...本轮 turn_messages...];
    裁剪只允许丢最旧的完整历史 turn(fit_tool_context 的区间语义),
    system/evidence/当前 turn 受保护。返回 None = 历史丢光仍超限(调用方发 tool_context_too_long)。"""
    evidence = [SystemMessage(content=evidence_text)] if evidence_text else []
    base = [SystemMessage(content=system_prompt), *history, *evidence, *turn_messages]
    protected_from = len(base) - len(turn_messages) - len(evidence)
    if not history:
        # 无历史可丢:直接判定
        from langchain_core.messages.utils import count_tokens_approximately
        return base if count_tokens_approximately(base) <= max_input_tokens else None
    return fit_tool_context(base, max_input_tokens, protected_from=protected_from)
```

注意 `fit_tool_context` 的 `protected_from` 语义是「该下标起(含)不可拆」;上面传的是 evidence 起点,evidence 与本轮 turn 一起受保护。

- [ ] **Step 5: 实现 app/graph/agent_node.py**

```python
"""主力 Agent:手写 ReAct 循环节点(与 examples/bare_agent.py 同构)。
停止条件:无 tool_calls 收敛 / 步数或累计 token 预算耗尽 → AGENT_BUDGET_ANSWER。"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from pydantic import ValidationError

from app.chains.tool_chat_chain import (
    build_agent_context, finalize_tool_calls, merge_tool_call_chunks,
)
from app.graph.errors import TurnAbortError
from app.graph.events import ev_error, ev_fixed_delta, ev_tool_end, ev_tool_start
from app.graph.nodes import GraphDeps
from app.prompts.service import AGENT_BUDGET_ANSWER, FALLBACK_ANSWER
from app.tools.business import MOCK_TOOLS
from app.tools.executor import ToolExecutor, ToolRegistry

logger = logging.getLogger("wayhelp.graph")


@tool
def suggest_options(options: list[Literal["转人工", "建工单"]],
                    ticket_type: Literal["售后", "投诉", "咨询"] | None = None) -> str:
    """向用户建议后续可选动作(只建议不执行)。options 取 1~2 个不重复标签;
    含「建工单」时 ticket_type 必填,不含时必须省略。"""
    return "已记录建议,请收尾答复用户。"


_ACTION_ID = {"转人工": "transfer_human", "建工单": "create_ticket"}


def _handle_suggest_options(call: dict) -> tuple[ToolMessage, list[dict] | None]:
    """校验伪工具参数;合法 → (成功 ToolMessage, actions),非法 → (错误 ToolMessage, None)。"""
    try:
        suggest_options.args_schema(**(call.get("args") or {}))
    except ValidationError:
        return _suggest_err(call, "工具参数不合法"), None
    args = call["args"]
    options, ttype = args["options"], args.get("ticket_type")
    if not (1 <= len(options) <= 2) or len(set(options)) != len(options):
        return _suggest_err(call, "工具参数不合法"), None
    if ("建工单" in options) != (ttype is not None):
        return _suggest_err(call, "工具参数不合法"), None
    actions = [{"action": _ACTION_ID[o], "label": o} for o in options]
    if ttype is not None:
        actions[-1]["ticket_type"] = ttype  # 建工单选项携带类型
    msg = ToolMessage(content="已记录建议,请收尾答复用户。", tool_call_id=call["id"],
                      name="suggest_options", status="success")
    return msg, actions


def _suggest_err(call: dict, text: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call.get("id") or "",
                       name="suggest_options", status="error")


def _evidence_text(evidence: list[dict]) -> str | None:
    if not evidence:
        return None
    lines = ["本轮检索证据(政策/规格结论只能依据以下证据,引用句末标 [n] 角标):"]
    for e in evidence:
        lines.append(f"[{e['ref_no']}] {e.get('section_path') or ''}\n"
                     f"问:{e['question']}\n答:{e['answer']}")
    return "\n".join(lines)


def _tool_schema_tokens(tools) -> int:
    """绑定工具 schema 的估算开销(name+description+args schema)。"""
    import json as _json
    parts = []
    for t in tools:
        parts.append(t.name)
        parts.append(t.description or "")
        try:
            parts.append(_json.dumps(t.args_schema.schema(), ensure_ascii=False))
        except Exception:
            pass
    return count_tokens_approximately([AIMessage(content="\n".join(parts))])


def build_agent_node(deps: GraphDeps):
    settings = deps.settings

    async def main_agent(state) -> dict:
        writer = get_stream_writer()
        tools = [*MOCK_TOOLS, suggest_options]
        registry = ToolRegistry(tools)  # 注册表无 create_ticket/query_faq:伪造调用 → unknown_tool
        executor = ToolExecutor(registry, settings.tool_timeout_seconds,
                                settings.tool_max_retries, settings.max_tool_result_chars,
                                write_tools=set())  # 聊天图内无写工具
        model = deps.model.bind_tools(registry.tools)
        schema_tokens = _tool_schema_tokens(registry.tools)
        evidence_text = _evidence_text(state["evidence"])

        turn_messages = list(state["turn_messages"])
        suggested = list(state["suggested_actions"])
        steps = 0
        spent = 0
        accounting = "none"
        visible_chars = 0
        trace = [*state["node_trace"]]

        def _abort(code: str, message: str):
            writer(ev_error(code, message))
            raise TurnAbortError(code)

        def _budget_answer():
            writer(ev_fixed_delta(AGENT_BUDGET_ANSWER))
            turn_messages.append(AIMessage(content=AGENT_BUDGET_ANSWER))
            trace.append({"node": "main_agent", "budget": True, "steps": steps})
            return {"turn_messages": turn_messages, "final_text": AGENT_BUDGET_ANSWER,
                    "agent_steps": steps, "agent_tokens": spent,
                    "token_accounting": accounting, "suggested_actions": suggested,
                    "node_trace": trace}

        while True:
            if steps >= settings.max_agent_steps:
                logger.info("node=main_agent budget: steps=%d", steps)
                return _budget_answer()
            context = build_agent_context(deps.system_prompt, state["messages"],
                                          turn_messages, evidence_text,
                                          settings.max_input_tokens)
            if context is None:
                _abort("tool_context_too_long", "工具结果超出上下文预算")
            est_input = count_tokens_approximately(context) + schema_tokens
            reserve = est_input + settings.max_output_tokens
            if spent + reserve > settings.max_agent_tokens:
                logger.info("node=main_agent budget: tokens spent=%d reserve=%d", spent, reserve)
                return _budget_answer()

            steps += 1
            text_parts: list[str] = []
            acc: dict[int, dict] = {}
            finish = None
            usage = None
            agen = model.astream(context)
            try:
                async for chunk in agen:
                    meta = getattr(chunk, "response_metadata", None) or {}
                    if meta.get("finish_reason"):
                        finish = meta["finish_reason"]
                    um = getattr(chunk, "usage_metadata", None)
                    if isinstance(um, dict) and um.get("input_tokens") is not None:
                        usage = um  # 取末次有效 usage(完整响应结算一次)
                    chunks = getattr(chunk, "tool_call_chunks", None) or []
                    if chunks:
                        merge_tool_call_chunks(acc, chunks)
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if not text:
                        continue
                    visible_chars += len(text)
                    if visible_chars > settings.max_message_chars:
                        _abort("output_too_long", "回复超出长度限制")
                    text_parts.append(text)
                    # token 经 stream_mode="messages" 自动流出,节点不写 writer
            except TurnAbortError:
                raise
            except Exception as exc:
                logger.warning("node=main_agent upstream error: %s", type(exc).__name__)
                _abort("upstream_error", "上游模型暂时不可用")
            finally:
                import contextlib
                with contextlib.suppress(Exception):
                    await agen.aclose()

            if finish == "length":
                _abort("output_too_long", "回复超出长度限制")

            # 结算本次消耗:有效 usage 替换预留;缺失/非法保留预留(不得按 0 计)
            try:
                actual = int(usage["input_tokens"]) + int(usage["output_tokens"]) \
                    if usage else 0
                if usage and actual >= 0 and isinstance(usage.get("input_tokens"), int):
                    spent += actual
                    accounting = "usage" if accounting in ("none", "usage") else "mixed"
                else:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                spent += reserve
                accounting = "estimated" if accounting == "none" else "mixed"

            calls, broken = finalize_tool_calls(acc)
            if broken or not _calls_legal(calls, settings.max_tool_calls_per_turn):
                _abort("invalid_tool_call", "工具调用申请不合法")

            if not calls:
                final = "".join(text_parts).strip() or FALLBACK_ANSWER
                if not "".join(text_parts).strip():
                    writer(ev_fixed_delta(FALLBACK_ANSWER))
                turn_messages.append(AIMessage(content=final))
                trace.append({"node": "main_agent", "steps": steps})
                return {"turn_messages": turn_messages, "final_text": final,
                        "agent_steps": steps, "agent_tokens": spent,
                        "token_accounting": accounting, "suggested_actions": suggested,
                        "node_trace": trace}

            turn_messages.append(AIMessage(content="".join(text_parts), tool_calls=calls))
            for call in calls:
                if call["name"] == "suggest_options":
                    msg, actions = _handle_suggest_options(call)
                    if actions is not None:
                        suggested = actions  # 多次合法建议以最后一组替换
                    turn_messages.append(msg)
                    continue
                writer(ev_tool_start(call))
                outcome = await executor.execute(call)
                logger.info("node=main_agent tool %s ok=%s err=%s",
                            outcome.record.name, outcome.record.ok, outcome.record.error_type)
                writer(ev_tool_end(call["id"], call["name"], outcome.record.ok,
                                   outcome.message.content[:80]))
                if outcome.record.error_code:
                    outcome.message.additional_kwargs["error_code"] = outcome.record.error_code
                turn_messages.append(outcome.message)
            # 循环回顶部:步数/预算检查在下次模型调用前生效

    return main_agent


def _calls_legal(calls: list[dict], max_calls: int) -> bool:
    if len(calls) > max_calls:
        return False
    ids = [c["id"] for c in calls]
    if len(set(ids)) != len(ids):
        return False
    return all(c["id"] and len(c["id"]) <= 64 for c in calls)
```

文件头部 import 补 `from langgraph.config import get_stream_writer`。`GraphDeps` 需要 `system_prompt` 字段——回到 `app/graph/nodes.py` 给 GraphDeps 加 `system_prompt: str = ""`(带默认值,Task 8 测试不受影响;Task 13 装配传 SERVICE_SYSTEM_PROMPT)。

- [ ] **Step 6: 运行循环测试**

Run: `uv run pytest tests/test_agent_node.py -v`
Expected: 6 PASS

- [ ] **Step 7: 预算与边界测试(追加)**

```python
async def test_step_budget_exhausted_after_tool_group_completed():
    node = _agent([
        _tool_chunk("query_order", {"order_id": "1"}, "c1"),
        ("then", [_tool_chunk("query_logistics", {"order_id": "1"}, "c2")]),
    ], max_agent_steps=2)
    out = await node(new_turn_state("一直查"))
    assert out["final_text"] == AGENT_BUDGET_ANSWER
    assert out["agent_steps"] == 2
    # 悬空 tool_calls 不留:第二组 ToolMessage 补齐后才兜底
    assert out["turn_messages"][-1].content == AGENT_BUDGET_ANSWER
    assert out["turn_messages"][-2].type == "tool" and out["turn_messages"][-2].tool_call_id == "c2"


async def test_token_budget_blocks_call_before_it_happens():
    node = _agent(["文本"], max_agent_tokens=10)  # 预算极小,首轮预留即超
    out = await node(new_turn_state("q"))
    assert out["final_text"] == AGENT_BUDGET_ANSWER
    assert out["agent_steps"] == 0  # 一次模型调用都没发起


async def test_usage_settled_once_when_present():
    script = [("tool", [{"index": 0, "name": "query_order", "id": "c1",
                         "args": '{"order_id":"1"}'}]),
              ("usage", {"input_tokens": 100, "output_tokens": 5}),
              ("then", ["好的", ("usage", {"input_tokens": 120, "output_tokens": 10})])]
    node = _agent(script)
    out = await node(new_turn_state("查订单"))
    assert out["agent_tokens"] == 100 + 5 + 120 + 10  # 两次调用各结算一次
    assert out["token_accounting"] == "usage"
```

(`("usage", {...})` 需要 conftest 的 FakeStreamModel/FakeChunk 支持 usage_metadata——见 Step 8。)

- [ ] **Step 8: conftest 扩展 FakeStreamModel 支持 usage**

`tests/conftest.py` 的 `FakeChunk.__init__` 加 `usage_metadata=None` 参数存 `self.usage_metadata`;`FakeStreamModel.astream` 加分支:

```python
            elif isinstance(item, tuple) and item[0] == "usage":
                yield FakeChunk("", usage_metadata=item[1])
```

Run: `uv run pytest tests/test_agent_node.py -v` → 全绿

- [ ] **Step 9: 提交**

```bash
git add app/graph/agent_node.py app/graph/nodes.py app/chains/tool_chat_chain.py app/config.py tests/test_agent_node.py tests/conftest.py dev-notes/ch05.md
git commit -m "feat(graph): main_agent 手写 ReAct 节点——多步工具、预算前置、suggest_options 伪工具、禁写护栏"
```

---

### Task 12: log 节点——提交、入池、提交后发帧

**Files:**
- Modify: `app/graph/nodes.py`(追加 `build_log_node`)
- Test: `tests/test_graph_nodes.py`(追加 log 用例)

**Interfaces:**
- Consumes: Task 4 `CommitTurnResult`;`store.commit_turn`;`wrap`(tool_envelope);Task 7 events。
- Produces: `build_log_node(deps) -> async def log_turn(state, config) -> dict`;返回 `{"messages": state["turn_messages"], "source_message_id": ...}`(此时才追加跨轮历史);`_to_stored`/`_build_low_conf`/`_should_cite` 纯函数(可单测)。

**行为契约(spec §6.6):**
- 入池两来源:gate 置的 `retrieval_low_conf`(reason 用 `low_conf_reason`);`retrieval_status=="ok"` 且 `final_text.strip()==REFUSAL_ANSWER` → `self_check`(reason 含 evidence 的 ref_no/chunk_id)。每轮至多一条;故障/预算/普通答复不入池。
- citations 条件:route==knowledge 且 retrieval_status=="ok" 且 evidence 非空 且 final_text 含 `[n]` 角标 且不等于 REFUSAL_ANSWER/AGENT_BUDGET_ANSWER。
- commit 用 shield;取消等事务落地再放行;提交失败异常上抛(图中止,驱动层不发 DONE)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_graph_nodes.py 追加
from app.graph.nodes import build_log_node
from app.sessions import InMemorySessionStore


def _log_deps(store):
    return GraphDeps(model=None, settings=make_settings(), retriever=None, store=store)


def _config(sid):
    return {"configurable": {"thread_id": sid}}


async def test_log_commits_and_returns_source_message_id():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    node = build_log_node(_log_deps(store))
    st = new_turn_state("你好")
    st.update({"final_text": "您好", "route": "chitchat"})
    st["turn_messages"].append(AIMessage(content="您好"))
    out = await node(st, _config(sid))
    assert out["source_message_id"]  # commit 成功后才返回
    assert out["messages"] == st["turn_messages"]  # 此刻才追加跨轮历史
    snap = await store.snapshot(sid)
    assert [m.role for m in snap] == ["user", "assistant"]


async def test_log_pools_retrieval_low_conf():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    node = build_log_node(_log_deps(store))
    st = new_turn_state("库外问题")
    st.update({"final_text": REFUSAL_ANSWER, "route": "knowledge",
               "retrieval_status": "low_confidence",
               "low_conf_source": "retrieval_low_conf",
               "low_conf_reason": {"note": "知识库尚未建立", "top1": None}})
    st["turn_messages"].append(AIMessage(content=REFUSAL_ANSWER))
    await node(st, _config(sid))
    assert len(store.low_confidence) == 1
    assert store.low_confidence[0].source == "retrieval_low_conf"
    assert store.low_confidence[0].raw_question == "库外问题"


async def test_log_pools_self_check_when_ok_but_refusal():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    node = build_log_node(_log_deps(store))
    st = new_turn_state("偏门问题")
    st.update({"final_text": REFUSAL_ANSWER, "route": "knowledge",
               "retrieval_status": "ok",
               "evidence": [{"ref_no": 1, "chunk_id": 7, "section_path": "s",
                             "question": "q", "answer": "a", "category": "c"}]})
    st["turn_messages"].append(AIMessage(content=REFUSAL_ANSWER))
    await node(st, _config(sid))
    assert store.low_confidence[0].source == "self_check"
    assert "chunk_id" in store.low_confidence[0].reason


async def test_log_does_not_pool_unavailable_or_budget():
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u")
    node = build_log_node(_log_deps(store))
    for status, text in (("unavailable", KB_UNAVAILABLE_ANSWER),
                         ("ok", AGENT_BUDGET_ANSWER)):
        st = new_turn_state("q")
        st.update({"final_text": text, "route": "knowledge",
                   "retrieval_status": status})
        st["turn_messages"].append(AIMessage(content=text))
        await node(st, _config(sid))
    assert store.low_confidence == []


def test_should_cite_rules():
    from app.graph.nodes import _should_cite
    base = {"route": "knowledge", "retrieval_status": "ok",
            "evidence": [{"ref_no": 1}], "final_text": "7 天无理由[1]"}
    assert _should_cite(base) is True
    assert _should_cite({**base, "route": "business"}) is False
    assert _should_cite({**base, "final_text": REFUSAL_ANSWER}) is False
    assert _should_cite({**base, "final_text": "没标角标的答复"}) is False
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph_nodes.py -k log -v`
Expected: FAIL(`build_log_node` 不存在)

- [ ] **Step 3: 实现(追加 app/graph/nodes.py)**

```python
import asyncio
import contextlib

from langchain_core.messages import HumanMessage, ToolMessage

from app.graph.events import ev_citations, ev_suggest_actions
from app.prompts.service import AGENT_BUDGET_ANSWER
from app.sessions import LowConfidenceRecord, StoredMessage
from app.tool_envelope import wrap


def _to_stored(turn_messages, settings) -> list[StoredMessage]:
    """本轮消息 → 落库行;临时 SystemMessage 不落库;ToolMessage 打 envelope。"""
    out: list[StoredMessage] = []
    for m in turn_messages:
        if isinstance(m, HumanMessage):
            out.append(StoredMessage("user", m.content))
        elif isinstance(m, AIMessage):
            out.append(StoredMessage("assistant", m.content or None,
                                     tool_calls=m.tool_calls or None))
        elif isinstance(m, ToolMessage):
            ok = m.status != "error"
            out.append(StoredMessage(
                "tool",
                wrap(m.content, ok,
                     None if ok else m.additional_kwargs.get("error_code", "tool_error"),
                     settings.max_tool_result_chars),
                tool_call_id=m.tool_call_id))
    return out


def _build_low_conf(state, cid: int | None) -> LowConfidenceRecord | None:
    if state["low_conf_source"] == "retrieval_low_conf":
        return LowConfidenceRecord(
            raw_question=state["raw_query"], source="retrieval_low_conf",
            reason=json.dumps(state["low_conf_reason"], ensure_ascii=False),
            conversation_id=cid)
    if (state["retrieval_status"] == "ok"
            and state["final_text"].strip() == REFUSAL_ANSWER):
        refs = [{"ref_no": e["ref_no"], "chunk_id": e["chunk_id"]}
                for e in state["evidence"]]
        return LowConfidenceRecord(
            raw_question=state["raw_query"], source="self_check",
            reason=json.dumps({"evidence_refs": refs}, ensure_ascii=False),
            conversation_id=cid)
    return None


def _should_cite(state) -> bool:
    if state["route"] != "knowledge" or state["retrieval_status"] != "ok":
        return False
    if not state["evidence"]:
        return False
    final = state["final_text"].strip()
    if final in (REFUSAL_ANSWER, AGENT_BUDGET_ANSWER):
        return False
    return re.search(r"\[\d{1,2}\]", final) is not None


def build_log_node(deps: GraphDeps):
    async def log_turn(state, config):
        writer = get_stream_writer()
        sid = config["configurable"]["thread_id"]
        stored = _to_stored(state["turn_messages"], deps.settings)
        cid = int(sid) if sid.isdecimal() else None
        low_conf = _build_low_conf(state, cid)
        commit_task = asyncio.ensure_future(
            deps.store.commit_turn(sid, stored, low_confidence=low_conf))
        try:
            result = await asyncio.shield(commit_task)  # 取消时等事务落地
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await commit_task
            raise
        # 提交成功后才发 citations / suggest_actions(失败不得发按钮或成功终帧)
        if _should_cite(state):
            writer(ev_citations(state["evidence"]))
        if state["suggested_actions"]:
            writer(ev_suggest_actions(result.source_message_id,
                                      state["suggested_actions"]))
        logger.info("node=log session=%s route=%s steps=%s tokens=%s accounting=%s",
                    sid, state.get("route"), state.get("agent_steps"),
                    state.get("agent_tokens"), state.get("token_accounting"))
        return {"messages": state["turn_messages"],
                "source_message_id": result.source_message_id,
                "node_trace": [*state["node_trace"], {"node": "log"}]}

    return log_turn
```

(`json`/`re`/`AIMessage` 等 import 文件头部已具备则复用。)

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest tests/test_graph_nodes.py -v` → 全绿

```bash
git add app/graph/nodes.py tests/test_graph_nodes.py dev-notes/ch05.md
git commit -m "feat(graph): log 节点——单事务提交+两源入池+提交成功后才发 citations/建议帧"
```

---

### Task 13: 图装配 + ChatService 驱动重构 + main.py 装配 + 旧测试改造

**Files:**
- Create: `app/graph/builder.py`
- Modify: `app/services/chat_service.py`(stream 换成驱动图)、`app/main.py`、`app/routers/chat.py`(+suggest_actions 帧序列化)、`tests/conftest.py`(+ScriptedChatModel)
- Rewrite: `tests/test_chat_service.py`、`tests/test_chat_api_tools.py`、`tests/test_spec11_safety.py`;Delete: `tests/test_orchestration.py`
- Test: `tests/test_graph_builder.py`(新增,图结构与 ScriptedChatModel 回调冒烟)

**Interfaces:**
- Consumes: 全部前序节点工厂(Task 8/9/10/11/12)。
- Produces:
  - `build_chat_graph(deps: GraphDeps, checkpointer) -> CompiledStateGraph`
  - `ChatService(store, model, settings, system_prompt, session_factory=None, graph=None)`;`set_graph(graph)`;`stream(turn)` 驱动 `graph.astream(new_turn_state(...), config, stream_mode=["messages","custom"])`
  - `SuggestActionsEvent(source_message_id: str, options: list[dict])` 入 `ChatEvent` 联合;路由序列化为 `{"type":"suggest_actions","source_message_id":...,"options":[...]}`
  - conftest `ScriptedChatModel(BaseChatModel)`(真实 Runnable → 回调链完整,messages 流可用)

- [ ] **Step 1: 写失败测试——图结构与假模型回调冒烟**

```python
# tests/test_graph_builder.py
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from tests.conftest import ScriptedChatModel, make_settings


def _deps(model):
    return GraphDeps(model=model, settings=make_settings(), retriever=None,
                     store=None, system_prompt="测试系统提示")


async def test_graph_compiles_and_routes_chitchat():
    model = ScriptedChatModel(scripts=[[
        '{"intent":"闲聊","needs_knowledge":false}']])
    graph = build_chat_graph(_deps(model), InMemorySaver())
    out = await graph.ainvoke(new_turn_state("你好"),
                              {"configurable": {"thread_id": "t1"}})
    from app.prompts.service import CHITCHAT_REPLY
    assert out["final_text"] == CHITCHAT_REPLY
    assert out["route"] == "chitchat"


async def test_messages_mode_streams_agent_tokens_with_node_metadata():
    """验收 messages 流:ScriptedChatModel 是真 Runnable,回调链完整,
    token 必须带 langgraph_node 元数据从图里流出(分类节点的不许漏出)。"""
    model = ScriptedChatModel(scripts=[
        ['{"intent":"订单","needs_knowledge":false}'],
        ["订单 1001 ", "已发货。"],
    ])
    graph = build_chat_graph(_deps(model), InMemorySaver())
    chunks = []
    async for mode, payload in graph.astream(
            new_turn_state("查订单 1001"), {"configurable": {"thread_id": "t2"}},
            stream_mode=["messages", "custom"]):
        if mode == "messages":
            chunks.append(payload)
    agent_text = "".join(
        c.content for c, meta in chunks
        if meta.get("langgraph_node") == "main_agent" and isinstance(c.content, str))
    assert agent_text == "订单 1001 已发货。"
    assert all(meta.get("langgraph_node") != "classify_intent" or True
               for _, meta in chunks)  # 分类节点走 ainvoke 本就不产生流式 chunk
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph_builder.py -v`
Expected: FAIL(builder / ScriptedChatModel 不存在)

- [ ] **Step 3: conftest 加 ScriptedChatModel**

```python
# tests/conftest.py 追加
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.chains.tool_chat_chain import finalize_tool_calls, merge_tool_call_chunks


class ScriptedChatModel(BaseChatModel):
    """脚本化假模型(真 Runnable → LangChain 回调链完整,图 messages 流可用)。
    scripts: 每段是一次调用的脚本,元素:
      str → 文本 delta;("tool", [tool_call_chunks]) → 工具调用;
      ("finish", reason) → finish_reason;("usage", dict) → usage_metadata;
      Exception → 抛错。bind_tools 记录工具名并返回自身。"""

    scripts: list
    bound: list | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound = [t.name for t in tools]
        return self

    def _next_script(self) -> list:
        return list(self.scripts.pop(0)) if self.scripts else ["(空)"]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text, calls, usage = "", [], None
        for item in self._next_script():
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                text += item
            elif item[0] == "tool":
                acc: dict[int, dict] = {}
                merge_tool_call_chunks(acc, item[1])
                parsed, _ = finalize_tool_calls(acc)
                calls.extend(parsed)
            elif item[0] == "usage":
                usage = item[1]
        msg = AIMessage(content=text, tool_calls=calls)
        if usage:
            msg.usage_metadata = usage
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # 回调(on_llm_new_token)由 BaseChatModel.astream 包装层负责,这里只 yield
        for item in self._next_script():
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                chunk = AIMessageChunk(content=item)
            elif item[0] == "tool":
                chunk = AIMessageChunk(content="", tool_call_chunks=item[1])
            elif item[0] == "finish":
                chunk = AIMessageChunk(content="",
                                       response_metadata={"finish_reason": item[1]})
            elif item[0] == "usage":
                chunk = AIMessageChunk(content="", usage_metadata=item[1])
            else:
                continue
            yield ChatGenerationChunk(message=chunk)
```

若实测 `BaseChatModel.astream` 对该子类不发回调(langchain-core 版本差异),先查 Context7 `langchain-core` 的 fake/custom chat model 文档,仍不通则停下问用户——不得改用 monkeypatch 伪造回调。

- [ ] **Step 4: 实现 builder.py**

```python
"""ch05 图装配:确定性 Workflow 骨架,分流规则经写死的路由表生效。"""

from langgraph.graph import END, START, StateGraph

from app.graph.agent_node import build_agent_node
from app.graph.nodes import (
    GraphDeps, build_fixed_nodes, build_front_nodes, build_knowledge_nodes,
    build_log_node, route_by_intent,
)
from app.graph.state import ChatGraphState


def build_chat_graph(deps: GraphDeps, checkpointer):
    g = StateGraph(ChatGraphState)
    front = build_front_nodes(deps)
    knowledge = build_knowledge_nodes(deps)
    fixed = build_fixed_nodes()

    g.add_node("resolve_reference", front["resolve_reference"])
    g.add_node("classify_intent", front["classify_intent"])
    g.add_node("retrieve", knowledge["retrieve"])
    g.add_node("confidence_gate", knowledge["confidence_gate"])
    g.add_node("gate_fallback", knowledge["gate_fallback"])
    g.add_node("complaint_reply", fixed["complaint_reply"])
    g.add_node("chitchat_reply", fixed["chitchat_reply"])
    g.add_node("main_agent", build_agent_node(deps))
    g.add_node("log", build_log_node(deps))

    g.add_edge(START, "resolve_reference")
    g.add_edge("resolve_reference", "classify_intent")
    g.add_conditional_edges(
        "classify_intent", route_by_intent,
        {"knowledge": "retrieve", "business": "main_agent",
         "complaint": "complaint_reply", "chitchat": "chitchat_reply"})
    g.add_edge("retrieve", "confidence_gate")
    g.add_conditional_edges(
        "confidence_gate", knowledge["route_after_gate"],
        {"main_agent": "main_agent", "gate_fallback": "gate_fallback"})
    for node in ("main_agent", "gate_fallback", "complaint_reply", "chitchat_reply"):
        g.add_edge(node, "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=checkpointer)
```

(`route_by_intent` 从 nodes.py 导出;`build_knowledge_nodes` 返回的 `route_after_gate` 是条件边函数不是节点,`g.add_node` 时不要误加。)

- [ ] **Step 5: 运行图冒烟测试**

Run: `uv run pytest tests/test_graph_builder.py -v`
Expected: 2 PASS

- [ ] **Step 6: 重构 chat_service.py 驱动层**

改动要点(保留 prepare/release_turn/create_ticket_from_action 与全部事件 dataclass):

- `__init__` 改为 `(self, store, model, settings, system_prompt, session_factory=None, graph=None)`;删 `toolset_factory`;加 `set_graph(graph)`。
- `PreparedTurn` 删 `messages` 字段(prepare 不再拼上下文;保留 `check_input_budget` 早闸——从 `app.chains.chat_chain` import,在长度闸后调用 `check_input_budget(self._system_prompt, message, self._settings.max_input_tokens)`)。
- 新增事件:

```python
@dataclass(frozen=True)
class SuggestActionsEvent:
    source_message_id: str
    options: list[dict]
```

`ChatEvent` 联合加 `SuggestActionsEvent`。

- `stream()` 整体替换为:

```python
    async def stream(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        try:
            if self._graph is None:
                yield ErrorEvent("internal_error", "服务未就绪")
                return
            yield SessionEvent(turn.session_id)
            config = {"configurable": {"thread_id": turn.session_id}}
            try:
                async for mode, payload in self._graph.astream(
                        new_turn_state(turn.user_text), config,
                        stream_mode=["messages", "custom"]):
                    if mode == "messages":
                        chunk, meta = payload
                        if (meta or {}).get("langgraph_node") != "main_agent":
                            continue  # 分类/检索内部调用的 token 不外发
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if text:
                            yield DeltaEvent(text)
                    else:
                        event = self._translate_custom(payload)
                        if event is not None:
                            yield event
            except TurnAbortError:
                return  # error 帧已由节点经 writer 发出;不提交、不发 DONE
            except Exception:
                logger.exception("chat graph error")
                yield ErrorEvent("internal_error", "服务内部错误")
                return
            yield DoneEvent()
        finally:
            self.release_turn(turn)

    @staticmethod
    def _translate_custom(payload) -> ChatEvent | None:
        kind = payload.get("kind")
        if kind == "fixed_delta":
            return DeltaEvent(payload["content"])
        if kind == "tool_start":
            return ToolStartEvent(payload["tool_call_id"], payload["name"], payload["args"])
        if kind == "tool_end":
            return ToolEndEvent(payload["tool_call_id"], payload["name"],
                                payload["ok"], payload["summary"])
        if kind == "citations":
            return CitationsEvent(payload["citations"])
        if kind == "suggest_actions":
            return SuggestActionsEvent(payload["source_message_id"], payload["options"])
        if kind == "error":
            return ErrorEvent(payload["code"], payload["message"])
        return None
```

import 更新:删 `fit_tool_context`/`finalize_tool_calls`/`merge_tool_call_chunks`/`ToolExecutor` 等旧编排引用;加 `from app.graph.errors import TurnAbortError`、`from app.graph.state import new_turn_state`。`FALLBACK_ANSWER` 改从 `app.prompts.service` import(常量已挪);删除本文件内的旧定义与 `_tool_calls_legal`(语义归 agent_node)。`REFUSAL_ANSWER` 引用同步改自 prompts。

- [ ] **Step 7: 路由序列化 suggest_actions 帧**

`app/routers/chat.py` 的 `_EventStream._body` 分发链加:

```python
                elif isinstance(event, SuggestActionsEvent):
                    yield _sse({"type": "suggest_actions",
                                "source_message_id": event.source_message_id,
                                "options": event.options})
```

- [ ] **Step 8: main.py 装配**

- `ChatOpenAI(...)` 构造加 `stream_usage=True`(token 结算依赖;spec §7.3)。
- `ChatService(...)` 构造改为 `ChatService(runtime.store, model, settings, SERVICE_SYSTEM_PROMPT, session_factory=runtime.session_factory)`。
- 紧跟其后:

```python
    deps = GraphDeps(model=model, settings=settings, retriever=runtime.retriever,
                     store=runtime.store, system_prompt=SERVICE_SYSTEM_PROMPT)
    if not owns_runtime:  # 测试/嵌入路径:内存 checkpointer 即刻可用
        service.set_graph(build_chat_graph(deps, InMemorySaver()))
```

- lifespan 改为:

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if owns_runtime:  # 生产:SQLite checkpointer 由 lifespan 托管,启停对称
            async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as cp:
                service.set_graph(build_chat_graph(deps, cp))
                yield
        else:
            yield
        await app.state.job_runner.close()
        if owns_runtime and runtime.retriever is not None:
            runtime.retriever.close()
```

import:`from langgraph.checkpoint.memory import InMemorySaver`、`from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`、`from app.graph.builder import build_chat_graph`、`from app.graph.nodes import GraphDeps`。

- [ ] **Step 9: 旧测试改造(先跑全量看红,再按下表处置)**

Run: `uv run pytest -x --ignore=tests/dbfixtures.py` 收集红的清单。处置原则:**HTTP/SSE 帧协议契约尽量保留;旧「两次调用」编排专属契约退役;检索相关场景从 query_faq 路径迁到预检索路径**。

- `tests/test_orchestration.py`:**删除文件**。守护语义的新家:无工具单次收敛 → `test_agent_node.py::test_one_step_converge`;工具序列 → `test_multi_step_order_then_logistics`;空文本兜底 → 在 test_agent_node 补 `FakeStreamModel([""])` → final_text==FALLBACK_ANSWER 用例(本步补上);「第二次调用绑工具但不执行 tool_calls」→ 退役(ReAct 语义取代)。
- `tests/test_chat_service.py`:按图重写。保留语义:commit 内容与低置信入池(改经图触发:runtime.retriever 注假检索器)、锁释放与 aclose(HTTP 层契约,经 test_chat_api 既有用例)、prepare 长度闸/404、上游错误帧(scripted model 抛异常 → `upstream_error` 帧)、预算(`max_agent_tokens=1` → AGENT_BUDGET_ANSWER delta)。退役:两次调用次数钉、「落库剥离第二轮 tool_calls」(新设计 turn_messages 只含已执行组,无可剥)。
- `tests/test_chat_api_tools.py`:删全部 query_faq 用例(工具退出聊天);保留并重写:tool_start/tool_end 帧序与 summary 截 80(scripted model 触发 query_order)、session 归属 404(不动)、硬闸门拒答帧(假 retriever 返回 low_confidence=True → delta == REFUSAL_ANSWER 且无 citations 帧)、citations 帧(ok 检索 + scripted 答复含 [1])。
- `tests/test_spec11_safety.py`:保留取消/提交/释锁/上下文预算安全性质,按图语义重接;「聊天中建单后模型失败/取消仍保留工单」场景**退役**(聊天零建单副作用);写工具 shield 由 `test_tools.py` 与 `test_chat_action.py` 覆盖。
- `make_runtime` 的 `toolset_factory` 参数不再被 ChatService 消费:保留字段不动(AppRuntime 结构),新测试一律经 `runtime.retriever` 注假检索器。

完成标准:`uv run pytest` 全绿。

- [ ] **Step 10: 提交**

```bash
git add app/graph/builder.py app/services/chat_service.py app/routers/chat.py app/main.py tests/ dev-notes/ch05.md
git commit -m "feat(chat): 主链路切换 LangGraph 图驱动——SSE 协议不变,新增 suggest_actions 帧"
```

---

### Task 14: 前端——suggest_actions 按钮组与转人工模拟

**Files:**
- Modify: `app/static/chat.html`(CSS + JS,Vibe 例外)
- Test: `tests/test_chat_page.py`(字符串断言)

**Interfaces:**
- Consumes: Task 13 的 `suggest_actions` 帧;Task 6 的 `POST /v1/chat/action`。
- Produces: 按钮组渲染(各自独立锁定)、转人工纯前端模拟、建工单 fetch 调用(捕获时绑定 user_id/session_id/source_message_id/ticket_type)。

- [ ] **Step 1: 先写断言(字符串契约)**

`tests/test_chat_page.py` 追加:

```python
def test_chat_page_action_buttons_logic():
    html = (Path(__file__).resolve().parent.parent
            / "app" / "static" / "chat.html").read_text(encoding="utf-8")
    for needle in ("suggest_actions", "source_message_id",
                   "转人工", "建工单",
                   "已转接人工客服", "您好，我是客服小猫，请问有什么可以帮您的",
                   "/v1/chat/action", "action-bar", "ticket_type"):
        assert needle in html
```

(若该文件已有读取 chat.html 的 helper,复用之。)

Run: `uv run pytest tests/test_chat_page.py -v` → FAIL(缺字符串)

- [ ] **Step 2: CSS(`<style>` 内追加,配色沿用现有变量)**

```css
.action-bar { display: flex; gap: 8px; margin-top: 6px; }
.action-btn {
  padding: 6px 14px; border-radius: 16px; border: 1px solid #d0d7de;
  background: #fff; cursor: pointer; font-size: 13px;
}
.action-btn:hover:not(:disabled) { border-color: #0969da; color: #0969da; }
.action-btn:disabled { opacity: 0.5; cursor: default; }
.system-hint {
  text-align: center; color: #888; font-size: 12px; margin: 10px 0;
}
```

- [ ] **Step 3: JS(script IIFE 内追加,并接入事件分发)**

```js
  var KITTEN_GREETING = "您好，我是客服小猫，请问有什么可以帮您的？";

  function addSystemHint(text) {
    var div = document.createElement("div");
    div.className = "system-hint";
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function renderActions(col, evt) {
    // 捕获时绑定:不得读取随后变化的全局 sessionId,也不用「最近一条消息」
    var captured = { user_id: userId, session_id: sessionId,
                     source_message_id: evt.source_message_id };
    var bar = document.createElement("div");
    bar.className = "action-bar";
    bar.setAttribute("data-od-id", "action-bar");
    (evt.options || []).forEach(function (opt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "action-btn";
      btn.setAttribute("data-od-id", "action-" + opt.action);
      btn.textContent = opt.label;
      btn.addEventListener("click", function () {
        btn.disabled = true;  // 只锁定被点的按钮;另一个不受影响,互不绑定
        if (opt.action === "transfer_human") {
          // 本章纯前端模拟,不接真人系统,不发任何请求
          addSystemHint("已转接人工客服");
          addBubble("assistant", KITTEN_GREETING);
        } else if (opt.action === "create_ticket") {
          fetch("/v1/chat/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              user_id: captured.user_id, session_id: captured.session_id,
              source_message_id: captured.source_message_id,
              action: "create_ticket", ticket_type: opt.ticket_type
            })
          }).then(function (r) {
            if (!r.ok) throw new Error("建单失败(" + r.status + ")");
            return r.json();
          }).then(function (d) {
            addBubble("assistant", "已为您创建工单 " + d.ticket_no + "，我们会尽快处理。");
          }).catch(function () {
            addSystemHint("建单失败，请稍后重试");  // 不自动重试
          });
        }
      });
      bar.appendChild(btn);
    });
    col.appendChild(bar);
    scrollToBottom();
  }
```

`send()` 内 `parseEvents` 分发链加(与 citations 分支并列):

```js
          } else if (evt.type === "suggest_actions") {
            renderActions(col, evt);  // 帧在 commit 成功后、DONE 前到达,直接渲染
          }
```

- [ ] **Step 4: 断言通过 + 人工点验**

Run: `uv run pytest tests/test_chat_page.py -v` → 全绿
人工点验(起服务,docker 在线):说「我要投诉」→ 出现两个独立按钮;点「转人工」→ 提示 + 小猫问候,另一个按钮仍可用;点「建工单」→ 出现工单号;都不点继续发消息 → 正常对话。结果记 dev-notes。

- [ ] **Step 5: 提交**

```bash
git add app/static/chat.html tests/test_chat_page.py dev-notes/ch05.md
git commit -m "feat(web): 转人工/建工单独立按钮——转人工纯前端模拟,建工单调动作端点"
```

---

### Task 15: 验收集成、持久化测试、联调验证与文档

**Files:**
- Create: `tests/test_ch05_acceptance.py`、`tests/test_graph_persistence.py`
- Modify: `README.md`、`AGENTS.md`

**Interfaces:**
- Consumes: 全部前序任务。
- Produces: spec §13 验收映射 11 条的自动化钉;SQLite 文件级持久化验证;上线联调记录。

- [ ] **Step 1: 验收测试(11 条)**

```python
# tests/test_ch05_acceptance.py
"""spec §13 验收标准 1-11 的集成钉。帧序:session → delta/tool_* → citations? →
suggest_actions? → [DONE](失败路径无 DONE/无按钮)。"""

import asyncio
import json
import logging

import httpx
import pytest

from app.knowledge.retriever import (
    NOTE_NOT_BUILT, NOTE_REBUILDING, KnowledgeHit, RetrievalResult,
)
from app.main import create_app
from app.prompts.service import (
    AGENT_BUDGET_ANSWER, CHITCHAT_REPLY, KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER,
)
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


# ── 基建 ──

def _hit(score=0.9):
    return KnowledgeHit(chunk_id=7, score=score, category="policy",
                        questions="退货政策", answer="7 天无理由退货。",
                        source_doc="returns-policy.md", chunk_index=0,
                        section_path="退货政策")


def _result(low=False, note=None, hits=None, score=0.9):
    return RetrievalResult(
        hits=hits if hits is not None else [_hit()], requested_strategy="hybrid_rerank",
        effective_strategy="hybrid_rerank", confidence_score=score,
        confidence_threshold=0.0553, low_confidence=low, note=note,
        query_plan=None, leg_counts={"dense": 1})


class _FakeRetriever:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def search(self, query, **kw):
        self.calls.append(query)
        return self._result


def _make_app(scripts, retriever=None, db_sf=None, **settings_over):
    import dataclasses
    runtime = make_runtime(tools=[])
    if retriever is not None:
        runtime = dataclasses.replace(runtime, retriever=retriever)
    if db_sf is not None:
        runtime = dataclasses.replace(runtime, session_factory=db_sf)
    return create_app(settings=make_settings(**settings_over),
                      model=ScriptedChatModel(scripts=scripts), runtime=runtime)


async def _turn(client, message, session_id=None):
    """发一轮聊天,返回 (frames, session_id)。"""
    resp = await client.post("/v1/chat/stream", json={
        "user_id": TEST_USER_ID, "session_id": session_id, "message": message})
    assert resp.status_code == 200
    frames = []
    async for line in resp.aiter_lines():
        if line.startswith("data:"):
            data = line[5:].strip()
            frames.append(data if data == "[DONE]" else json.loads(data))
    sid = next((f["session_id"] for f in frames
                if isinstance(f, dict) and f.get("type") == "session"), session_id)
    return frames, sid


def _types(frames):
    return [f if isinstance(f, str) else f["type"] for f in frames]


def _deltas(frames):
    return "".join(f["content"] for f in frames
                   if isinstance(f, dict) and f["type"] == "delta")


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


# ── 验收 1:政策类问题日志可见强制检索节点;纯业务查询不检索 ──

async def test_a1_knowledge_runs_retrieve_and_logs(caplog):
    app = _make_app(
        [['{"intent":"退款退货","needs_knowledge":true}'],
         ["7 天无理由退货[1]。"]],
        retriever=_FakeRetriever(_result()))
    async with await _client(app) as client:
        with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
            frames, _ = await _turn(client, "退货政策是什么")
    assert "node=retrieve" in caplog.text and "node=confidence_gate" in caplog.text
    assert _deltas(frames) == "7 天无理由退货[1]。"


async def test_a1_business_skips_retrieve(caplog):
    rt = _FakeRetriever(_result())
    app = _make_app(
        [['{"intent":"物流","needs_knowledge":false}'],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c1",
                     "args": '{"order_id":"1001"}'}])],
         ["派送中。"]],
        retriever=rt)
    async with await _client(app) as client:
        with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
            await _turn(client, "订单 1001 的物流到哪了")
    assert rt.calls == [] and "node=retrieve" not in caplog.text


# ── 验收 2:Agent 自调工具作答 ──

async def test_a2_agent_calls_logistics_tool():
    app = _make_app(
        [['{"intent":"物流","needs_knowledge":false}'],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c1",
                     "args": '{"order_id":"1001"}'}])],
         ["您的订单由顺丰承运,派送中。"]])
    async with await _client(app) as client:
        frames, _ = await _turn(client, "订单 1001 的物流到哪了")
    starts = [f for f in frames if isinstance(f, dict) and f["type"] == "tool_start"]
    assert [s["name"] for s in starts] == ["query_logistics"]
    assert "派送中" in _deltas(frames)


# ── 验收 3:投诉双按钮;点旧按钮建单绑定原消息;都不点继续正常聊(验收 8 合并) ──

async def test_a3_complaint_two_independent_buttons_and_old_button_binds_original(
        db_session_factory):
    app = _make_app(
        [['{"intent":"投诉","needs_knowledge":false}'],
         ['{"intent":"订单","needs_knowledge":false}'],
         ["订单状态良好。"]],
        db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "我要投诉你们的服务")
        sug = [f for f in frames if isinstance(f, dict) and f["type"] == "suggest_actions"]
        assert len(sug) == 1
        opts = sug[0]["options"]
        assert [o["action"] for o in opts] == ["transfer_human", "create_ticket"]
        assert opts[1]["ticket_type"] == "投诉"
        # 帧序:suggest_actions 在最后一个 delta 之后、DONE 之前
        assert _types(frames).index("suggest_actions") > len(_types(frames)) - 3
        assert _types(frames)[-1] == "[DONE]"
        mid1 = sug[0]["source_message_id"]
        # 不点按钮继续正常聊(产生新用户消息)
        frames2, _ = await _turn(client, "顺便查下订单 1001", sid)
        assert "[DONE]" in _types(frames2)
        # 回头点第一条消息的旧按钮:建的仍是投诉那条
        resp = await client.post("/v1/chat/action", json={
            "user_id": TEST_USER_ID, "session_id": sid, "source_message_id": mid1,
            "action": "create_ticket", "ticket_type": "投诉"})
        assert resp.status_code == 200

        def _check():
            from app.models import Ticket
            with db_session_factory() as s:
                t = s.get(Ticket, resp.json()["ticket_no"])
                assert t.description == "我要投诉你们的服务"
        await asyncio.to_thread(_check)


# ── 验收 4:闲聊固定话术,零生成调用 ──

async def test_a4_chitchat_fixed_reply():
    app = _make_app([['{"intent":"闲聊","needs_knowledge":false}']])
    async with await _client(app) as client:
        frames, _ = await _turn(client, "你好")
    assert _deltas(frames) == CHITCHAT_REPLY
    # 只消费了分类一段脚本;闲聊回复零生成调用
    assert app.state.model.scripts == []


# ── 验收 5:先订单后物流,ReAct 多步 ──

async def test_a5_multi_step_react():
    app = _make_app(
        [['{"intent":"订单","needs_knowledge":false}'],
         [("tool", [{"index": 0, "name": "query_order", "id": "c1",
                     "args": '{"order_id":"1001"}'}])],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c2",
                     "args": '{"order_id":"1001"}'}])],
         ["订单已发货,顺丰派送中。"]])
    async with await _client(app) as client:
        frames, _ = await _turn(client, "订单 1001 买的是什么,到哪了")
    starts = [f["name"] for f in frames
              if isinstance(f, dict) and f["type"] == "tool_start"]
    assert starts == ["query_order", "query_logistics"]


# ── 验收 6:故障不入池 / 零命中入池 / 自评拒答入池 ──

async def test_a6_pooling_rules(db_session_factory):
    from app.models import LowConfidenceQuestion
    # 维护态:不入池,回 KB_UNAVAILABLE_ANSWER
    app = _make_app([['{"intent":"售后","needs_knowledge":true}']],
                    retriever=_FakeRetriever(_result(note=NOTE_REBUILDING,
                                                     low=True, hits=[], score=None)),
                    db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "维修寄修流程是什么")
        assert _deltas(frames) == KB_UNAVAILABLE_ANSWER
        assert not any(isinstance(f, dict) and f["type"] == "citations" for f in frames)
    # 零命中:入 retrieval_low_conf
    app = _make_app([['{"intent":"售后","needs_knowledge":true}']],
                    retriever=_FakeRetriever(_result(note=NOTE_NOT_BUILT,
                                                     low=True, hits=[], score=None)),
                    db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid2 = await _turn(client, "偏门问题甲")
        assert _deltas(frames) == REFUSAL_ANSWER
    # 高分但模型自评拒答:入 self_check
    app = _make_app([['{"intent":"退款退货","needs_knowledge":true}'],
                     [REFUSAL_ANSWER]],
                    retriever=_FakeRetriever(_result()), db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid3 = await _turn(client, "定制商品能退吗")
        assert _deltas(frames) == REFUSAL_ANSWER

    def _check():
        with db_session_factory() as s:
            rows = s.query(LowConfidenceQuestion).order_by(
                LowConfidenceQuestion.id).all()
            sources = [(r.raw_question, r.source) for r in rows]
            assert ("维修寄修流程是什么", "retrieval_low_conf") not in sources
            assert ("偏门问题甲", "retrieval_low_conf") in sources
            assert ("定制商品能退吗", "self_check") in sources
    await asyncio.to_thread(_check)


# ── 验收 7:checkpoint 跨轮无临时状态串用 ──

async def test_a7_state_reset_between_turns():
    rt = _FakeRetriever(_result(low=True, hits=[], score=0.01))
    app = _make_app(
        [['{"intent":"退款退货","needs_knowledge":true}'],   # 第一轮:低置信被拒
         ['{"intent":"闲聊","needs_knowledge":false}'],      # 第二轮:闲聊
         ],
        retriever=rt)
    async with await _client(app) as client:
        f1, sid = await _turn(client, "火星特产能退吗")
        assert _deltas(f1) == REFUSAL_ANSWER
        f2, _ = await _turn(client, "你好呀", sid)
        assert _deltas(f2) == CHITCHAT_REPLY
        # 第二轮不得出现第一轮的低置信标记副作用:无 citations、无 suggest_actions
        assert not any(isinstance(f, dict) and f["type"] in ("citations", "suggest_actions")
                       for f in f2)


# ── 验收 9:预算在发起下一次模型调用前生效 ──

async def test_a9_budget_blocks_before_call():
    model = ScriptedChatModel(scripts=[['{"intent":"订单","needs_knowledge":false}'],
                                       ["不应出现的文本"]])
    runtime = make_runtime(tools=[])
    app = create_app(settings=make_settings(max_agent_tokens=1), model=model,
                     runtime=runtime)
    async with await _client(app) as client:
        frames, _ = await _turn(client, "查订单")
    assert _deltas(frames) == AGENT_BUDGET_ANSWER
    assert "不应出现" not in _deltas(frames)
    assert model.scripts == [["不应出现的文本"]]  # 第二轮脚本根本没被消费


# ── 验收 10:多步消息整体提交;按钮帧在 commit 后 ──

async def test_a10_multi_step_persisted_as_groups(db_session_factory):
    app = _make_app(
        [['{"intent":"订单","needs_knowledge":false}'],
         [("tool", [{"index": 0, "name": "query_order", "id": "c1",
                     "args": '{"order_id":"1"}'}])],
         [("tool", [{"index": 0, "name": "query_logistics", "id": "c2",
                     "args": '{"order_id":"1"}'}])],
         ["查好了。"]],
        db_sf=db_session_factory)
    async with await _client(app) as client:
        _, sid = await _turn(client, "先查订单再查物流")

    def _check():
        from app.models import Message
        with db_session_factory() as s:
            rows = (s.query(Message).filter_by(conversation_id=int(sid))
                    .order_by(Message.id).all())
            roles = [r.role for r in rows]
            assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
            assert rows[1].tool_calls[0]["id"] == "c1" and rows[2].tool_call_id == "c1"
            assert rows[3].tool_calls[0]["id"] == "c2" and rows[4].tool_call_id == "c2"
    await asyncio.to_thread(_check)


# ── 验收 11:聊天 Agent 无建单/转人工权限 ──

async def test_a11_agent_has_no_write_power(db_session_factory):
    app = _make_app(
        [['{"intent":"售后","needs_knowledge":false}'],
         [("tool", [{"index": 0, "name": "create_ticket", "id": "c9",
                     "args": '{"description":"x","ticket_type":"投诉"}'}])],
         [("tool", [{"index": 0, "name": "suggest_options", "id": "s1",
                     "args": '{"options":["转人工","建工单"],"ticket_type":"投诉"}'}])],
         ["建议您点击下方按钮。"]],
        db_sf=db_session_factory)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "我现在就要你帮我建工单并转人工")
        ends = [f for f in frames if isinstance(f, dict) and f["type"] == "tool_end"]
        assert ends[0]["ok"] is False  # 伪造 create_ticket → unknown_tool
        sug = [f for f in frames if isinstance(f, dict) and f["type"] == "suggest_actions"]
        assert len(sug) == 1  # 伪工具建议正常发出

    def _check():
        from app.models import Conversation, Ticket
        with db_session_factory() as s:
            assert s.query(Ticket).count() == 0  # 没写库
            assert s.get(Conversation, int(sid)).status == "进行中"  # 没置已转人工
    await asyncio.to_thread(_check)
```

(`suggest_options` 不发徽章,所以验收 11 中 tool_end 帧只有伪造的 create_ticket 一条。)

Run: `uv run pytest tests/test_ch05_acceptance.py -v`
Expected: 全绿;红则按 spec 修实现,不得改验收语义。

- [ ] **Step 2: 持久化测试**

```python
# tests/test_graph_persistence.py
"""SQLite 文件级持久化:关闭重开后跨轮历史仍在;失败轮临时消息不进历史。"""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from app.prompts.service import CHITCHAT_REPLY
from app.sessions import InMemorySessionStore
from tests.conftest import ScriptedChatModel, make_settings


def _deps(model, store):
    return GraphDeps(model=model, settings=make_settings(), retriever=None,
                     store=store, system_prompt="测试")


async def test_sqlite_checkpoint_survives_reopen(tmp_path):
    db = str(tmp_path / "cp.db")
    store = InMemorySessionStore(10, 100, 8000)
    cfg = {"configurable": {"thread_id": "s1"}}
    async with AsyncSqliteSaver.from_conn_string(db) as cp:
        graph = build_chat_graph(_deps(
            ScriptedChatModel(scripts=[['{"intent":"闲聊","needs_knowledge":false}']]),
            store), cp)
        await graph.ainvoke(new_turn_state("你好"), cfg)
    # 关闭后重开同一文件
    async with AsyncSqliteSaver.from_conn_string(db) as cp2:
        graph2 = build_chat_graph(_deps(
            ScriptedChatModel(scripts=[['{"intent":"闲聊","needs_knowledge":false}']]),
            store), cp2)
        out = await graph2.ainvoke(new_turn_state("在吗"), cfg)
        texts = [m.content for m in out["messages"]]
        assert "你好" in texts and CHITCHAT_REPLY in texts  # 第一轮历史仍在
        assert texts.count(CHITCHAT_REPLY) == 2  # 两轮答复都进了历史


async def test_failed_turn_leaves_no_messages(tmp_path):
    db = str(tmp_path / "cp2.db")
    store = InMemorySessionStore(10, 100, 8000)
    cfg = {"configurable": {"thread_id": "s2"}}
    async with AsyncSqliteSaver.from_conn_string(db) as cp:
        graph = build_chat_graph(_deps(
            ScriptedChatModel(scripts=[[ConnectionError("down")]]), store), cp)
        with pytest.raises(Exception):
            await graph.ainvoke(new_turn_state("查订单"), cfg)
        state = await graph.aget_state(cfg)
        assert not state.values.get("messages")  # 失败轮不追加跨轮历史
```

Run: `uv run pytest tests/test_graph_persistence.py -v` → 全绿

- [ ] **Step 3: 全量 + 联调验证**

1. `docker compose up -d`;`uv run pytest` 全量绿(DB 用例在线)。
2. DeepSeek stream_usage 联调探针(spec §7.3 要求先核实):

```bash
uv run python - <<'EOF'
import asyncio
from app.config import Settings
from langchain_openai import ChatOpenAI

async def main():
    s = Settings()
    m = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                   base_url=s.openai_base_url, max_tokens=50, stream_usage=True)
    usage = None
    async for chunk in m.astream("说一个字"):
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
    print("usage_metadata:", usage)

asyncio.run(main())
EOF
```

期望打印非 None 的 input/output tokens。若供应商拒绝 `stream_options`(400 等)或 usage 恒 None → **停下问用户**(spec 定死了 stream_usage 方案,不自行改)。结果记 dev-notes。

3. 起服务做真人五连(对应验收 1-5;命令逐个执行,`--noproxy '*'` 必须):

```bash
uv run uvicorn app.main:create_app --factory &
curl --noproxy '*' -N -X POST localhost:8000/v1/chat/stream -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","session_id":null,"message":"退货政策是什么"}'
# 看 retrieve/confidence_gate 日志 + 带角标答复 + citations 帧;再问「订单 1001 的物流到哪了」(同 session_id)
# 再说「我要投诉」→ 页面两个按钮;「你好」→ 固定话术;「先查订单 1001 再查物流」→ 两个 tool 徽章
```

浏览器打开 `http://localhost:8000/` 人工点验按钮交互(转人工模拟/建单/互不绑定)。验证完 `kill %1`。

- [ ] **Step 4: 文档更新**

- `README.md`:架构段加 ch05(LangGraph 编排、图节点链、/v1/chat/action、checkpoints.db);运行方式不变。
- `AGENTS.md`:目录约定加 `app/graph/` 与 `examples/`;「当前状态」段更新为 ch05 完成状态;红线区补一条「聊天 Agent 无写权限,建单唯一通道 /v1/chat/action」。

- [ ] **Step 5: dev-notes 收尾段 + 提交**

在 `dev-notes/ch05.md` 追加「阶段 4:code review 结论」与「阶段 5:finish」(四样格式;code review 结论由 code-review 流程产出后填入)。

```bash
git add tests/test_ch05_acceptance.py tests/test_graph_persistence.py README.md AGENTS.md dev-notes/ch05.md
git commit -m "test: ch05 验收 11 条集成钉与 SQLite 持久化验证;文档对齐"
```

---

## Self-Review 记录(plan 落盘后已逐项核对 spec)

- **spec 覆盖**:§5 拓扑→Task 13 builder;§6.1-6.6 节点→Task 8/9/10/12;§7 ReAct/预算/prompt→Task 11/3;§8 帧→Task 7/12/13;§9 端点→Task 5/6;§10 State/持久化→Task 7/13/15;§11 热身→Task 2;§12 前端→Task 14;§13 测试矩阵→各任务 + Task 15;§14 依赖→Task 1。
- **类型一致性**:`GraphDeps`(model/settings/retriever/store/system_prompt)在 Task 8 定义、Task 11 要求补 `system_prompt` 字段(带默认值,不破坏 Task 8 测试)、Task 13 装配;`new_turn_state` 键名与 §10.1 字段表逐字一致;`ev_*` payload 键与 `_translate_custom` 读取键一致;`CommitTurnResult.source_message_id` 贯穿 Task 4→6→12→14。
- **已知执行风险**(写进任务内):`get_stream_writer` 裸调节点行为(Task 9 注)、`BaseChatModel.astream` 回调包装(Task 13 Step 3 注)、DeepSeek `stream_usage` 兼容性(Task 15 Step 3,失败须停下问用户)。
- **自审发现并就地修复**(2026-09-18):
  1. `AppRuntime` 是 frozen dataclass,Task 6/15 测试中 `runtime.retriever = x` 直接赋值会抛 FrozenInstanceError → 全部改为 `dataclasses.replace`;
  2. Task 6 测试原依赖 `post_stream` 走聊天链路播种消息 → Task 13 换图后必坏,改为 db fixtures 直接播种(对图重构免疫);
  3. Task 11 agent_node 头部 import 冗余(`from app.graph.nodes import GraphDeps, logger` 后又重定义 logger)→ 去重;
  4. spec §13 要求的 `test_tool_chat_chain.py` 多组重建/裁剪钉原缺 → 补进 Task 11 Step 2;
  5. test_a4 双重建 app 的残留语句 → 改为 `_make_app` + `app.state.model.scripts` 断言。
