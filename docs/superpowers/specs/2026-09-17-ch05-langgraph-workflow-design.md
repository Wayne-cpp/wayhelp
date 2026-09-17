# ch05 设计:LangGraph Workflow 编排骨架 + 主力 ReAct Agent

- 日期:2026-09-17
- 状态:已获用户批准(brainstorm 定稿)
- 前置:ch01~ch04(见 dev-notes/),当前 HEAD `b889480`,工作区干净

## 1. 背景与目标

把现有聊天主链路从手写「两次调用」编排升级为生产级架构:**LangGraph Workflow 确定性编排做骨架,主力 Agent 作为骨架里的核心节点**。本章同时承担「祛魅」教学任务:先不用框架手写最裸的 Agent 循环,看清 Agent = 带工具的循环,再重构进 LangGraph。

本章明确不做:意图识别/指代消解正式版、上下文管理策略升级、MCP 接入、数据飞轮入库管道。

## 2. 现状摘要(设计依据)

- 现有链路 `app/services/chat_service.py`:手写「两次调用」——第一次 `bind_tools().astream` + 执行工具,第二次绑工具但不执行(防标记语法裸文本泄漏);`RetrievalTrace` 桥接检索硬闸门(low_confidence → 直接 `REFUSAL_ANSWER`)。
- 业务工具 `app/tools/business.py` `build_tools(session_factory, conversation_id, retriever, settings) -> TurnToolset`:`query_order` / `query_product` / `query_logistics`(mock JSON,不触库)、`query_faq`(检索)、`create_ticket`(写 tickets 表,同事务把 `conversations.status` 置「已转人工」)。执行层 `ToolExecutor` 提供超时/重试/shield 语义。
- 检索 `app/knowledge/retriever.py` `KnowledgeRetriever.search(...) -> RetrievalResult`,带 `confidence_score` / `low_confidence`(分策略冻结阈值,默认 hybrid_rerank);`assemble_evidence` 组装证据。
- 会话:MySQL `conversations` / `messages`;`session_id = str(conversations.id)`;`SessionStore` 协议(create/exists/snapshot/commit_turn),低置信问题入池 `LowConfidenceQuestion` 与消息同事务。
- 前端 `app/static/chat.html`(922 行原生 JS):fetch + ReadableStream 消费 SSE,反馈组件(👍/👎 一次锁定)可复用为按钮形态。
- 依赖:langchain-core 1.6.2 / langchain-openai 1.6.1;**langgraph 未装**。
- SSE 帧协议(`app/routers/chat.py`):`session` / `delta` / `tool_start` / `tool_end` / `citations` / `error` / `[DONE]`。

## 3. 已确认的设计决策(brainstorm 问答定案)

| # | 问题 | 定案 |
|---|------|------|
| D1 | 新旧链路关系 | **原地替换**:`/v1/chat/stream` 内部换成图,SSE 协议不变只加新帧;旧编排专属契约随代码退役 |
| D2 | State 持久化 | **AsyncSqliteSaver**(langgraph-checkpoint-sqlite),落 `data/` 下文件;MySQL 仍做账本 |
| D3 | ReAct 实现 | **手写循环节点**,与热身裸循环同构;不用 create_react_agent 黑盒 |
| D4 | 建工单通道 | **结构化动作端点** `POST /v1/chat/action`,确定性调工具,不过模型 |
| D5 | 热身产物 | **留档** `examples/bare_agent.py` 可运行示例 + 单测 |

## 4. 总体架构与模块布局

新增 `app/graph/` 包:

- `state.py` — `ChatGraphState` TypedDict
- `nodes.py` — resolve_reference / classify_intent / retrieve / confidence_gate / complaint_reply / chitchat_reply / gate_fallback / log 节点
- `agent_node.py` — 主力 Agent 手写 ReAct 循环节点
- `builder.py` — StateGraph 装配 + `compile(checkpointer=...)`

其他改动:

- `examples/bare_agent.py` — 热身留档(见 §11)
- `app/prompts/intent.py` — `INTENT_PROMPT`;`app/prompts/service.py` 加 `COMPLAINT_REPLY` / `CHITCHAT_REPLY` 常量并修订与 ReAct 冲突的条款(见 §7)
- `app/routers/chat.py` — 新增 `POST /v1/chat/action`;`app/schemas.py` 加 `ChatActionRequest`
- `app/services/chat_service.py` — `stream()` 内部换成驱动图;prepare 三段语义(建/验会话 → 每会话锁 → finally 释锁)保留在路由/服务边界,`aclose` 释锁契约不动
- `app/tools/business.py` — 从 `create_ticket` 抽出纯写库函数 `write_ticket()`(见 §9)
- `app/static/chat.html` — suggest_actions 按钮组 + 转人工模拟 + 建工单调用
- `app/main.py` — lifespan 装配图与 checkpointer
- `app/config.py` — 新增 `max_agent_steps`(默认 8)、`max_agent_tokens`(默认 20000)

## 5. 图拓扑与分流规则

```
START → resolve_reference → classify_intent → <条件边:route>
  ├─ "knowledge"  → retrieve → confidence_gate
  │     ├─ 证据够  → main_agent ───────────┐
  │     └─ 证据弱  → gate_fallback ────────┤
  ├─ "business"  → main_agent ────────────┤
  ├─ "complaint" → complaint_reply ───────┤
  └─ "chitchat"  → chitchat_reply ────────┤
                                          ▼
                                     log → END
```

分流规则**写死在代码里**(常量 dict,条件边函数只查表):

| 意图(classify_intent 输出) | 出口 route | 路径 |
|---|---|---|
| 商品咨询、退款退货 | `knowledge` | 强制预检索 → 置信度闸 → (过)Agent / (卡)固定拒答 |
| 物流、订单、售后 | `business` | 不预检索,直接进 Agent 自调工具 |
| 投诉 | `complaint` | 安抚话术 + suggest_actions,不进 Agent |
| 闲聊 | `chitchat` | 固定话术,零模型生成调用 |

每轮用户消息独立走全图;追问重新分类、重新检索(指代消解透传的已知限制,正式版下一章)。

## 6. 节点契约

### 6.1 resolve_reference(最简版)

原样透传:`resolved_query = 用户原句`。留扩展位,不做任何模型调用。

### 6.2 classify_intent(最简版)

- `INTENT_PROMPT`:简单 prompt,要求只输出 JSON `{"intent": "物流|订单|商品咨询|退款退货|售后|投诉|闲聊"}`,不看会话历史。
- 解析:提取 JSON、校验七类枚举;**解析失败/非法值兜底为「售后」(business 出口,fail-open 进 Agent)**,不用固定话术搪塞潜在真实问题。
- 每轮一次模型调用,开销接受(闲聊省下的抵消)。

### 6.3 retrieve(强制预检索)

- 调 `KnowledgeRetriever.search(resolved_query)`(默认策略 hybrid_rerank,走现有 Query 理解管道),`RetrievalResult` 原样写入 State。
- 节点进出打 INFO 日志(logger `wayhelp.graph`,含节点名与关键字段)——验收「日志里能看到强制检索节点被走到」由此满足,测试用 caplog 断言。
- 仅 knowledge 出口有此节点;business 出口在图上就没有这条边(不是跳过,是不经过)。

### 6.4 confidence_gate(最简版)

- 位置:检索之后、进 Agent 之前(Agent 流式吐字,答完再判就晚了)。
- 判定:直接消费 `RetrievalResult.low_confidence`(ch04 冻结的分策略阈值)。
- 证据弱 → `gate_fallback`:`REFUSAL_ANSWER` 固定话术进 messages;State 置 `low_conf_source="retrieval_low_conf"`,由 log 节点复用现有入池机制把原始问题记入 LowConfidenceQuestion(同事务)——即「记下来留给数据飞轮」,飞轮管道本身本章不做。
- 证据够 → `assemble_evidence` 结果写入 `state.evidence`,进 Agent。
- 已知继承限制:reranker 降级到 hybrid 时该策略不可闸(ch04 已记录,正式置信度检查留可观测章)。

### 6.5 complaint_reply / chitchat_reply

- `complaint_reply`:`COMPLAINT_REPLY` 安抚话术进 messages(草案:「非常抱歉给您带来了不好的体验,您的问题我们十分重视。您可以转接人工客服,或让我为您建立工单跟进处理。」)+ `suggested_actions = [转人工, 建工单(ticket_type="投诉")]`。
- `chitchat_reply`:`CHITCHAT_REPLY` 固定话术(草案:「您好,我是客服小蜜,可以帮您查订单、查物流、介绍商品和退换货政策,有什么可以帮您的吗?」),零模型调用。
- 两条路径的答复都以一次性 delta 帧发出,前端无感知差异。

### 6.6 log(日志记录)

- 所有终态路径汇入;调 `store.commit_turn`(用户消息 + 助手答复 + 可选低置信入池,单事务,沿用 shield 语义)。
- 同时输出本轮节点轨迹日志(intent/route/gate 结果/steps/tokens)。
- `suggested_actions` 不落入 messages 表(UI 亲和层,非对话内容)。

## 7. 主力 Agent(手写 ReAct,从 bare loop 重构)

节点内显式循环,与 `examples/bare_agent.py` 同一结构:

```
messages = 历史 + [系统补充消息(知识类时注入 evidence + 引用角标协议)]
loop:
  1. model.bind_tools(tools).astream(messages) → token 直出 SSE;聚合 tool_call_chunks
  2. 无 tool_calls → 收敛(内容可以是追问用户,天然支持「缺信息就追问」)
  3. 有 → ToolExecutor 逐个执行(超时/重试/shield 语义不变)→ ToolMessage 喂回,steps+1
  4. steps ≥ max_agent_steps(8)或累计 usage_metadata tokens ≥ max_agent_tokens(20000)
     → 输出预算耗尽兜底话术,收敛
```

细则:

- **工具集** = 第 2 章业务工具 `query_order` / `query_product` / `query_logistics` / `create_ticket`。`query_faq` 退出聊天工具集(知识类已由预检索覆盖;工具本身保留,评估链路在用)。本章不新写业务工具。
- **编排伪工具** `suggest_options(options: list["转人工","建工单"])`(参数用中文标签,节点映射为 §8 的 action id:「转人工」→`transfer_human`,「建工单」→`create_ticket` 并带 `ticket_type`):Agent 认为合适时调用 = 只发建议信号;loop 识别后写 `state.suggested_actions` 并提示模型收尾,零副作用、不进 ToolExecutor 写路径、不算业务工具。建议可只给一个也可两个都给;后端不自动执行任何动作。
- **护栏**:单次响应 tool_calls 上限 5(沿用);每轮 `create_ticket` ≤ 1(沿用);空最终文本兜底 `FALLBACK_ANSWER`(沿用)。
- **流式**:token 经 `stream_mode="messages"` 从节点内流出,按 `langgraph_node` 元数据过滤只取 main_agent;结构化事件(tool_start/tool_end/citations/suggest_actions)经 `get_stream_writer` + `stream_mode="custom"` 流出。
- **citations**:知识类且答复引用证据时,沿现有 `[n]` 角标协议发 `citations` 帧。
- **系统 prompt 修订**:`SERVICE_SYSTEM_PROMPT` 现行条款「每轮最多一次工具调用」与 ReAct 多步冲突,修订为多步语义(允许按中间结果连续调多个工具,单轮上限由 max_agent_steps 约束);其余条款(不编造/引用角标/拒答口径等)保留。`tests/test_prompts.py` 契约断言同步更新,改后跑全量 pytest。

## 8. SSE 协议增量

现有帧不变,新增一种:

```
suggest_actions: {"options": [
  {"action": "transfer_human", "label": "转人工"},
  {"action": "create_ticket", "label": "建工单", "ticket_type": "投诉"}
]}
```

- `options` 长度 1~2,可只给一个;`ticket_type` 仅 create_ticket 选项携带。
- 帧封装方式(事件名/data 行格式)沿用现有 `chat.py` 的实现风格。
- 触发源:complaint_reply 节点(必给两个),或 main_agent 经 `suggest_options` 伪工具(视情况给一个或两个)。

## 9. 动作端点(建工单)

`POST /v1/chat/action`:

- 请求 `ChatActionRequest`:`{user_id, session_id, action: "create_ticket", ticket_type: "售后|投诉|咨询"}`(action 枚举本章只有一个值,留扩展)。
- 校验:会话存在且属于 user_id(404 语义同现有聊天接口);参数非法 422。
- 行为:从 `create_ticket` 抽出纯写库函数 `write_ticket(session_factory, conversation_id, description, ticket_type) -> ticket_no`(只 INSERT tickets 行);端点调它,**不改 `conversations.status`**——建工单与转人工是两回事,互不绑定。description 取该会话最近一条用户消息,取不到用固定兜底文案「用户通过快捷操作请求建单」。
- 响应 `{ticket_no, status: "待处理"}`。
- Agent 工具路径的 `create_ticket` 维持 ch02 旧行为(同事务置「已转人工」,有测试钉),内部改为调 `write_ticket` + 置状态,行为不变。
- 「转人工」不进此端点:本章纯前端模拟,不接真人系统、不做任何后端动作。

## 10. 会话状态与持久化

- State 用 `add_messages` reducer 累积 messages;`thread_id = session_id`。
- checkpointer = `AsyncSqliteSaver`(langgraph-checkpoint-sqlite,含 aiosqlite),库文件 `data/checkpoints.db`(与 Milvus Lite 同风格,确认 .gitignore 覆盖);在 `create_app` lifespan 建/关,单 worker 约束不变。
- 职责分离:checkpoint = 图工作记忆(跨轮上下文),MySQL = 账本(会话/消息/低置信池,log 节点照旧写)。
- 已知限制(写进文档):checkpoint 文件丢失/删除时退化为「无历史新会话」,不回填 MySQL 历史;两库不做对账。

## 11. 热身留档:examples/bare_agent.py

- 无框架最裸 Agent 循环:`model.bind_tools(tools)` + `invoke` + 手写「有 tool_calls 就执行并喂回、没有就收敛」的 for 循环 + max_steps。
- 工具复用真实只读业务工具(query_order / query_logistics,mock JSON 不触库),证明「同一个 Agent,框架只是壳」。
- 可 `uv run python examples/bare_agent.py` 直接运行:脚本自包含一个脚本化假模型(不依赖 tests/),默认走假模型演示;带开关可切真实模型。
- `tests/test_bare_agent.py` 用 FakeStreamModel 钉:一次收敛、工具结果喂回、max_steps 兜底。
- 正式开发时把同一循环结构重构进 `agent_node.py`(invoke→astream、加 token 预算、加事件流出)。

## 12. 前端改动(chat.html,Vibe 例外照旧)

- 新增 suggest_actions 帧处理:在该条 assistant 气泡下渲染**各自独立的按钮**(复用 addFeedback 的组件形态;锁定粒度为「被点的那个按钮自己锁定防重」,另一个按钮不受影响,不用反馈组件的整组锁定)。
- 点「转人工」:纯前端渲染「已转接人工客服」系统提示 + 客服小猫问候气泡(「您好,我是客服小猫,请问有什么可以帮您的?」);不发任何请求。
- 点「建工单」:`POST /v1/chat/action`,成功后渲染「已为您创建工单 T…」;失败给错误提示。
- 两按钮互不绑定;用户都不点、继续发消息 = 普通对话正常走(后端无任何动作)。
- `test_chat_page.py` 补字符串断言(两个按钮文案、小猫问候、action 调用),人工点验照旧。

## 13. 测试策略

新增:

- `tests/test_bare_agent.py` — 裸循环三态(收敛/喂回/兜底)
- `tests/test_graph_state.py` — 分流表七类意图→四出口参数化断言
- `tests/test_graph_nodes.py` — 意图解析(合法/兜底)、置信度闸过与不过、投诉/闲聊固定话术、log 节点入池
- `tests/test_agent_node.py` — 一步收敛、订单→物流多步(步骤数与喂回顺序)、步数/token 预算耗尽、追问收敛、suggest_options 信号
- `tests/test_chat_action.py` — 建工单写 tickets 行、不动 conv.status、404/422

改造(图语义重写,保留可保留的断言):

- `test_chat_api.py` HTTP/SSE 契约尽量不动(帧序/错误码/aclose 释锁)
- `test_chat_service.py` / `test_orchestration.py` / `test_chat_api_tools.py` 按图重写;旧编排专属契约(「第二次调用绑工具但不执行 tool_calls」、两次调用次数钉)随旧代码退役
- `test_prompts.py` 契约断言随 §7 条款修订更新

验收 → 测试映射:

| 验收标准 | 验证方式 |
|---|---|
| 1. 政策类问题日志可见强制检索节点 | 集成测试 caplog 断言 retrieve 节点日志 + State 经 retrieve |
| 2. 「订单 1001 的物流到哪了」Agent 自调工具 | SSE 集成:tool_start query_logistics,答复含 mock 物流数据 |
| 3. 投诉出两独立按钮,转人工纯前端,建工单写库,都不点正常聊 | suggest_actions 帧断言 + 前端字符串断言 + action 端点写库断言 + 次轮正常对话断言 |
| 4. 闲聊固定话术 | delta 帧 = CHITCHAT_REPLY;断言除意图识别外无生成调用 |
| 5. 先订单后物流的复杂问题 ReAct 多步 | tool_start 帧序列 [query_order, query_logistics],agent_steps ≥ 2 |

测试基建沿用:FakeStreamModel、假 retriever、dbfixtures(Docker MySQL);checkpointer 测试用内存版 AsyncSqliteSaver(`:memory:`)。全量 `uv run pytest` 绿才交付。

## 14. 依赖变更

- `+ langgraph`(1.0.x,Context7 已核 1.0.8 API:StateGraph/add_conditional_edges/compile(checkpointer)/get_stream_writer/stream_mode)
- `+ langgraph-checkpoint-sqlite`(AsyncSqliteSaver,含 aiosqlite)
- 实现前按用户要求再以 Context7 核对:AsyncSqliteSaver 生命周期用法、`add_messages` reducer、astream 多模式组合,版本对不上就停下问,不自行换方案。

## 15. 风险与已知限制

1. AsyncSqliteSaver 接入 FastAPI lifespan 的连接管理(启停对称,测试隔离)。
2. 手写 ReAct 边界(空回复/工具异常/预算耗尽/伪工具误用)需测试全覆盖。
3. 旧服务层测试改造量集中在 test_chat_service / test_orchestration / test_chat_api_tools 三个文件。
4. 意图识别每轮 +1 次模型调用(延迟/成本),正式版再优化。
5. 指代消解透传 → 代词追问可能误分类(下一章正式版解决)。
6. checkpoint 与 MySQL 双写无对账;checkpoint 丢失退化为新会话(§10)。
7. reranker 降级到 hybrid 时闸失效(ch04 继承,可观测章处理)。
