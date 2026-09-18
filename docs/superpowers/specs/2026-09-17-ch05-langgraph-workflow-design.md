# ch05 设计:LangGraph Workflow 编排骨架 + 主力 ReAct Agent

- 日期:2026-09-17
- 状态:已按审阅意见修订定稿;用户于 2026-09-17 确认 Agent 禁止直接建单、增加 needs_knowledge 分流
- 前置:ch01~ch04(见 dev-notes/);原设计调研基线 `b889480`,本轮修订基线 `75ba316`

## 1. 背景与目标

把现有聊天主链路从手写「两次调用」编排升级为生产级架构:**LangGraph Workflow 确定性编排做骨架,主力 Agent 作为骨架里的核心节点**。本章同时承担「祛魅」教学任务:先不用框架手写最裸的 Agent 循环,看清 Agent = 带工具的循环,再重构进 LangGraph。

本章明确不做:意图识别/指代消解正式版、上下文管理策略升级、MCP 接入、数据飞轮入库管道。现有输入预算、完整工具组裁剪、检索异常分类与拒答入池契约须迁移保留,不属于策略升级。

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
| D4 | 建工单通道 | **结构化动作端点** `POST /v1/chat/action`,用户点击后确定性写库,不过模型;聊天 Agent 无建单权限 |
| D5 | 热身产物 | **留档** `examples/bare_agent.py` 可运行示例 + 单测 |
| D6 | 知识分流 | 七类 intent + `needs_knowledge` 布尔标记;政策/知识与混合问题强制预检索,纯业务数据查询直接进 Agent;分类解析失败先检索 |

本轮修订以 D4/D6 替代原设计中的“聊天 Agent 保留 create_ticket”“只按七类意图分流”及“解析失败直接进业务 Agent”。旧 create_ticket 仅保留兼容实现及工具级测试,不注册进新聊天图;query_faq 也不向聊天 Agent 暴露。

## 4. 总体架构与模块布局

新增 `app/graph/` 包:

- `state.py` — `ChatGraphState` TypedDict
- `nodes.py` — resolve_reference / classify_intent / retrieve / confidence_gate / complaint_reply / chitchat_reply / gate_fallback / log 节点
- `agent_node.py` — 主力 Agent 手写 ReAct 循环节点
- `builder.py` — StateGraph 装配 + `compile(checkpointer=...)`

其他改动:

- `examples/bare_agent.py` — 热身留档(见 §11)
- `app/prompts/intent.py` — `INTENT_PROMPT`;`app/prompts/service.py` 加 `COMPLAINT_REPLY` / `CHITCHAT_REPLY` / `KB_UNAVAILABLE_ANSWER` / `AGENT_BUDGET_ANSWER` 常量并修订与 ReAct 冲突的条款(见 §7)
- `app/routers/chat.py` — 新增 `POST /v1/chat/action`;`app/schemas.py` 加 `ChatActionRequest`
- `app/services/chat_service.py` — `stream()` 内部换成驱动图;prepare 三段语义(建/验会话 → 每会话锁 → finally 释锁)保留在路由/服务边界,`aclose` 释锁契约不动;每轮显式重置临时 State
- `app/sessions.py` / `app/store_db.py` — `commit_turn` 返回本轮用户消息 ID(见 §6.6),并把 `validate_turn` 扩展为可校验多个完整工具组;内存 store 同步实现协议
- `app/chains/tool_chat_chain.py` — 复用并适配现有完整 turn 裁剪与工具上下文预算,覆盖每次 ReAct 模型调用
- `app/tools/business.py` — 从 `create_ticket` 抽出纯写库函数 `write_ticket()`(见 §9);新聊天装配只向模型和执行注册表暴露三种只读业务工具,不直接复用完整旧 toolset
- `app/static/chat.html` — suggest_actions 按钮组 + 转人工模拟 + 建工单调用
- `app/main.py` — lifespan 装配图与 checkpointer,聊天模型显式开启流式 usage
- `app/config.py` — 新增 `max_agent_steps`(默认 8)、`max_agent_tokens`(默认 20000)

## 5. 图拓扑与分流规则

```
START → resolve_reference → classify_intent → <条件边:route>
  ├─ "knowledge"  → retrieve → confidence_gate
  │     ├─ 证据够  → main_agent ───────────┐
  │     └─ 证据弱/服务不可用 → gate_fallback┤
  ├─ "business"  → main_agent ────────────┤
  ├─ "complaint" → complaint_reply ───────┤
  └─ "chitchat"  → chitchat_reply ────────┤
                                          ▼
                                     log → END
```

分流规则**写死在代码里**:常量 dict 以 `(intent, needs_knowledge)` 为键,覆盖七类意图 × 两个布尔值的 14 种组合,条件边只查表,模型不直接输出 route。

| 意图(classify_intent 输出) | needs_knowledge | 出口 route | 路径 |
|---|---|---|---|
| 任意七类意图 | true | `knowledge` | 强制预检索 → 置信度闸 → (过)Agent / (卡)固定回复 |
| 退款退货 | false | `knowledge` | 保守覆盖:现有业务工具不能办理退款退货,仍须先检索政策 |
| 物流、订单、商品咨询、售后 | false | `business` | 纯业务数据查询/收集信息/动作建议,不预检索,进 Agent |
| 投诉 | false | `complaint` | 纯投诉安抚 + suggest_actions,不进 Agent |
| 闲聊 | false | `chitchat` | 固定话术,节点自身零模型调用 |

- `needs_knowledge=true` 优先于意图名称:保修流程、运费规则、订单取消规则、商品规格,以及“查订单 + 解释政策”的混合问题都先检索。即使主意图为投诉/闲聊,带知识需求也走 knowledge;纯投诉/纯闲聊保留原固定回复分支。
- 商品价格/库存/是否在售由 query_product 提供,归商品咨询 + false;产品功能/规格与政策归商品咨询 + true。业务工具不支持的写操作只可追问或建议按钮,不能臆造已办理。
- 混合问题用完整 resolved_query 检索,复用现有 Query 理解的子查询支路;过闸后 Agent 可继续查订单/物流。低置信或检索不可用时整轮走固定回复,本章不先输出其中的业务部分。
- 每轮重新分类,仅 knowledge 分支重新检索;指代消解透传的限制保留。解析失败按 §6.2 归入 knowledge,不得绕过闸门。

## 6. 节点契约

### 6.1 resolve_reference(最简版)

原样透传:`resolved_query = 用户原句`。留扩展位,不做任何模型调用。

### 6.2 classify_intent(最简版)

- `INTENT_PROMPT`:要求只输出含 intent 与 needs_knowledge 的 JSON,例如 `{"intent":"售后","needs_knowledge":true}`。intent 限七类枚举,needs_knowledge 必须是 JSON 布尔值,不能把字符串 "false" 或缺失字段当 false;不看会话历史。
- 分类边界:涉及规则、流程、保修、运费、商品规格或任一知识子问时 needs_knowledge=true;只查具体订单/物流/商品价格库存、收集工具参数、提议动作或纯投诉/纯闲聊时为 false。退款退货要求 true,路由表额外防御误报 false。情绪词不抹掉问题中的政策需求。
- 解析失败、字段缺失、非法枚举/类型时统一兜底为 `{"intent":"售后","needs_knowledge":true}`,记录解析失败日志,随后经 retrieve/confidence_gate;不得退回 business。模型请求异常仍按 upstream_error 结束,不把网络故障当合法分类结果。
- 每轮一次模型调用,开销接受(闲聊省下的抵消)。

分类 prompt 与路由验收至少包含:

| 用户问题 | intent | needs_knowledge | route |
|---|---|---|---|
| 订单 1001 的物流到哪了 | 物流 | false | business |
| 保温杯还有库存吗 | 商品咨询 | false | business |
| 维修寄修流程是什么 | 售后 | true | knowledge |
| 未发货订单如何取消 | 订单 | true | knowledge |
| 订单 1001 到哪了,退货运费谁承担 | 物流 | true | knowledge |
| 我要投诉你们的服务 | 投诉 | false | complaint |
| 我要投诉,并问一下退货运费规则 | 投诉 | true | knowledge |
| 你好 | 闲聊 | false | chitchat |

表中 intent 用于固定测试样例,混合问题实际可有其他合法主意图;知识需求标记与最终 route 必须符合规则。

### 6.3 retrieve(强制预检索)

- 调 `KnowledgeRetriever.search(resolved_query)`(默认策略 hybrid_rerank,走现有 Query 理解管道);同步检索在线程中运行,保留 `knowledge_tool_timeout_seconds` 总等待上限,本层不额外重试。`RetrievalResult` 所有字段转换为可序列化的数据快照写入 State,不得把连接、retriever 或模型实例存入 checkpoint。
- 先识别检索服务状态,再判断置信度:`NOTE_UNCONFIGURED` / `NOTE_REBUILDING` / `NOTE_REBUILD_REQUIRED` 分别归为 `kb_unconfigured` / `kb_rebuilding` / `kb_rebuild_required`;超时及检索异常归为 `kb_unavailable`。这些状态置 `retrieval_status="unavailable"`,即使原结果的 `low_confidence=True` 也不作为知识缺口。
- `NOTE_NOT_BUILT` 沿用现有“无可用知识”语义,参与低置信判定;reranker 降级但仍有检索结果的情形继续按 effective_strategy 的阈值判定。
- 节点进出打 INFO 日志(logger `wayhelp.graph`,含节点名与关键字段)——验收「日志里能看到强制检索节点被走到」由此满足,测试用 caplog 断言。
- 仅 knowledge 出口有此节点;business 出口在图上就没有这条边(不是跳过,是不经过)。

### 6.4 confidence_gate(最简版)

- 位置:检索之后、进 Agent 之前(Agent 流式吐字,答完再判就晚了)。
- 判定顺序:先检查 `retrieval_status`;仅检索服务正常时消费 `RetrievalResult.low_confidence`(ch04 冻结的分策略阈值)。
- 服务不可用 → `gate_fallback`:输出 `KB_UNAVAILABLE_ANSWER="知识库暂时不可用,请稍后重试。"`,保留机器错误码用于日志,不进 Agent、不发 citations、不入低置信池。
- 证据弱 → `gate_fallback`:输出 `REFUSAL_ANSWER`,置 `retrieval_status="low_confidence"`、`low_conf_source="retrieval_low_conf"`;reason 保留 requested_strategy / effective_strategy / top1 / threshold / note。由 log 节点把原始问题记入 LowConfidenceQuestion(同事务)。
- 证据够 → 置 `retrieval_status="ok"`,按现有 `rerank_top_n`、`max_tool_result_chars` 及组装开销调用 `assemble_evidence`,结果写入 `state.evidence`,进 Agent。因证据预算裁剪得到空列表时仍适用模型自评拒答契约(§6.6),不能以高检索分替代证据。
- 固定答复先写入 `final_text` 和本轮 `turn_messages`,经 custom 流发送一次 delta;只有 log 提交成功后才追加到跨轮 `messages`。
- 已知继承限制:reranker 降级到 hybrid 时该策略不可闸(ch04 已记录,正式置信度检查留可观测章)。

### 6.5 complaint_reply / chitchat_reply

- `complaint_reply`:`COMPLAINT_REPLY="非常抱歉给您带来了不好的体验,您的问题我们十分重视。您可以选择转接人工客服,或点击下方按钮建立工单跟进处理。"`;`suggested_actions = [转人工, 建工单(ticket_type="投诉")]`。
- `chitchat_reply`:`CHITCHAT_REPLY="您好,我是客服小蜜,可以帮您查订单、查物流、介绍商品和退换货政策,有什么可以帮您的吗?"`,节点自身不调模型。
- 两条路径都设置 `final_text` 和本轮 `turn_messages`,通过 custom 流发送一次性 delta;跨轮 messages 仅由 log 追加。按钮在提交成功后发送(§8),不在该节点提前发送可点击动作。

### 6.6 log(日志记录)

- 所有成功终态(含固定拒答、服务不可用说明、预算兜底)汇入;调 `store.commit_turn`(当前用户消息 + 本轮各组 AI tool_calls/ToolMessage + 最终助手答复 + 可选低置信入池,单事务,沿用 shield 语义)。模型异常、非法工具调用、输入/输出超限等失败路径发 error 后结束,不提交、不发 `[DONE]`。
- 保留两类拒答入池:`retrieval_low_conf` 按 §6.4 写入;`retrieval_status="ok"` 且 `final_text.strip() == REFUSAL_ANSWER` 时写 `source="self_check"`,reason 含当前 evidence 的 ref_no/chunk_id。原始问题取 `raw_query`,每轮至多一条;服务故障、预算耗尽及普通业务答复不入池,所有拒答不发 citations。
- `commit_turn(...) -> CommitTurnResult(source_message_id: str)`:DB store 在同一事务内 flush 本轮用户行取得 `messages.id`,仅在提交成功后返回其十进制字符串。内存 store 为用户消息分配稳定 ID,维持同一返回协议;调用方不得提交后另查“最近一条”来猜 ID。本章无需新增 MySQL 字段。
- 提交成功后,log 把完整 `turn_messages` 追加到 `messages`(add_messages),写入 `source_message_id` 并发送 citations(如有)、suggest_actions(如有)。图正常结束后服务发 `[DONE]`;提交失败不得发按钮或成功终帧。
- `validate_turn` 必须逐组校验多步 ReAct 的 AI tool_calls → ToolMessage 一一配对,完成一组后清空组内状态再校验下一组;不能沿用当前只接受一个工具组的实现。历史消息不重复落库,临时 SystemMessage 不落入消息表。
- 同时输出本轮节点轨迹日志(intent/needs_knowledge/route/gate 结果/steps/tokens)。
- `suggested_actions` 的 UI 帧不另写成对话消息;模型发起的 suggest_options 工具申请与确认 ToolMessage 仍随完整 turn 落库,用于保持消息配对,不代表用户已点击或动作已执行。

## 7. 主力 Agent(手写 ReAct,从 bare loop 重构)

节点内显式循环,与 `examples/bare_agent.py` 同一结构:

```text
turn_messages = [HumanMessage(raw_query)]
loop:
  1. 检查模型调用次数上限;从历史 + turn_messages 构建并裁剪本次上下文
  2. 检查本轮剩余 token 预算,预留本次估算输入 + max_output_tokens
  3. steps += 1;model.bind_tools(tools).astream(context) → token 直出 SSE
     聚合 tool_call_chunks 和本次 usage,记录本次消耗
  4. 无 tool_calls → 文本作为 final_text;若空则输出 FALLBACK_ANSWER;收敛
  5. 有 tool_calls → 校验整组,追加 AIMessage;逐个执行只读工具/处理伪工具
     为每个 call 追加匹配的 ToolMessage,再进入下一次循环
  6. 无法再发起模型调用时 → 追加 AGENT_BUDGET_ANSWER,收敛
```

### 7.1 工具、消息与流式

- **工具集** = 第 2 章只读业务工具 `query_order` / `query_product` / `query_logistics`,另绑定零副作用的 `suggest_options`。`create_ticket` 与 `query_faq` 均退出聊天模型绑定列表和 ToolExecutor 注册表;query_faq 保留供评估链路使用,旧 create_ticket 保留兼容实现。本章不新写业务工具。
- **编排伪工具** `suggest_options(options, ticket_type=None)`:options 为 1~2 个不重复的中文枚举标签「转人工」/「建工单」;含「建工单」时 ticket_type 必填且限「售后」/「投诉」/「咨询」,不含时必须省略。节点映射为 §8 的 action id;只更新本轮 `suggested_actions`,返回匹配 tool_call_id 的 ToolMessage 提示模型收尾,不执行任何动作。非法参数返回错误 ToolMessage 且不产生建议;多次合法建议以最后一组替换,log 最多发送一帧。不发伪工具的 tool_start/tool_end 徽章。
- **护栏**:单次响应 tool_calls 上限 5(含伪工具),保留调用 ID 合法性/组内唯一性校验;聊天 Agent 的建单执行次数必须为 0。即使模型伪造 create_ticket 调用也只返回 unknown_tool,不得动态找回旧工具执行;用户口头要求建单/转人工同样只能产生建议按钮。空最终文本兜底 `FALLBACK_ANSWER`。
- **流式**:使用 `stream_mode=["messages", "custom"]`;模型 token 按 `langgraph_node` 过滤只取 main_agent,分类/Query 理解的文本不外发。工具事件和固定回复 delta 经 `get_stream_writer` 流出;citations/suggest_actions 延迟至 log 提交后发出。
- **消息完整性**:每次带 tool_calls 的 AIMessage 都要跟齐 ToolMessage,包括伪工具与工具错误。到达预算时不留下悬空 tool_calls;中间文本保存在对应 AIMessage,最终文本另存 `final_text`,避免把上一轮/中间文本误判为最终答复。
- **citations**:本轮 route 为 knowledge、retrieval_status 为 ok、最终答复确实引用当前 evidence 且不是固定拒答/预算兜底时才发送。沿现有 `[n]` 协议传完整且顺序稳定的 evidence 列表,不筛掉未引用条目再重编号。
- **系统 prompt 修订**:把第 7 条“调用 create_ticket 并宣布已转人工”改为“调用 suggest_options,请用户点击对应按钮”;把第 9 条改为多步只读工具语义,删除 query_faq 调用要求。第 5 条及引用协议改为区分业务工具数据和 Workflow 注入的 evidence:政策/规格结论只能依赖本轮 evidence 并标角标,业务事实依赖只读工具结果;business 路径没有 evidence 不应导致拒绝正常工具数据答复,也不能据此编造政策。保留不编造、多子问完整作答、REFUSAL_ANSWER 逐字口径及负面知识禁令,同步更新 `tests/test_prompts.py`,实现后跑全量 pytest。
- **动作表述**:聊天模型不得声称“已建工单/已转人工”,也不靠对话历史推断按钮是否已执行。建单成功提示仅由动作端点结果驱动前端展示,模拟转人工仅由前端点击驱动。同步修订 `FALLBACK_ANSWER="抱歉,暂时没有查到相关信息。您可以换个说法,或请求人工帮助并选择相应按钮。"`。

### 7.2 输入上下文预算

- 每次模型调用前复用 `build_messages_with_tools` / `fit_tool_context` 的裁剪规则并适配 BaseMessage 历史:先删除最旧的完整历史 turn,不拆 AI tool_calls/ToolMessage 配对;保留 SERVICE_SYSTEM_PROMPT、当前 HumanMessage、本轮完整工具往返和本轮 evidence 补充消息。
- `max_input_tokens` 与 `max_agent_tokens` 是两个限制:前者约束单次调用输入,后者约束本轮累计消耗。输入估算覆盖实际传给模型的消息(含 evidence)及绑定工具 schema 的开销,不能只检查用户原句。
- 当前系统 prompt + 用户输入本身已超预算时,沿用 prepare 阶段拒绝语义;删除全部历史仍放不下本轮证据/工具结果时,发 `error(code="tool_context_too_long")` 并结束,不调用模型、不丢当前问题或只保留半个工具组。
- 保留整轮 `max_message_chars` 输出限制及 `finish_reason="length"` → `output_too_long` 语义;固定兜底也计入可见输出长度。预算兜底属于成功回复,上下文/输出超限属于失败,两者不得混用。

### 7.3 步数与累计 token 预算

- `max_agent_steps=8` 指 main_agent 内最多发起 8 次模型调用(含最终纯文本调用);一次模型返回多个工具仍算一个 step。分类和 Query 理解调用不计入该计数/该 token 预算,本章不是全请求成本限额。
- 聊天模型显式设置 `stream_usage=True`;消费到流结束,包括 content 为空的 usage chunk。本次完整响应的 `input_tokens + output_tokens` 只结算一次,不把累计 usage 重复加总。`agent_tokens` 累加各次输入与输出,包括重传历史与工具 schema 的成本。
- 发起调用前按输入估算 + `max_output_tokens` 预留预算;剩余预算不足则直接输出 `AGENT_BUDGET_ANSWER="本轮处理已达到上限,请缩小问题范围后重试。"`,不再调用模型。收到有效 usage 后用真实消耗替换本次预留;未返回或返回非法 usage 时保留预留作为估算消耗,不得按 0 计。
- 日志与 State 的 `token_accounting` 标明 usage/estimated/mixed。估算使用现有 token 估算器并覆盖工具 schema;这是请求准入与循环停止预算,不宣称在供应商缺少 usage 时等于实际账单硬上限。真实结算达到/超过上限后停止后续模型调用;已产出的正常最终答复直接结束,不再拼接矛盾兜底。
- 若模型返回工具申请后达到最后一步/累计上限,仍为本次合法只读调用和伪工具生成完整结果组,随后追加预算兜底,不发起第 9 次模型调用。只读工具继续遵守 ToolExecutor 的超时/重试契约;写 shield 仅用于动作端点及保留的旧工具兼容实现。
- 供应商拒绝 `stream_options` 时按上游错误处理并记录配置原因,不偷偷重发可能已执行的整轮请求;实现联调时须验证当前 DeepSeek 兼容端点,如确需关闭 usage 开关则先核实,缺失 usage 的估算分支仍必须可用。

## 8. SSE 协议增量

现有帧不变,新增一种(实际 SSE 仍为 `data: {JSON}\n\n`):

```json
{"type": "suggest_actions", "source_message_id": "123", "options": [
  {"action": "transfer_human", "label": "转人工"},
  {"action": "create_ticket", "label": "建工单", "ticket_type": "投诉"}
]}
```

- `options` 长度 1~2,可只给一个;`ticket_type` 仅 create_ticket 选项携带。
- 帧封装方式(事件名/data 行格式)沿用现有 `chat.py` 的实现风格。
- 触发源:complaint_reply 节点(必给两个),或 main_agent 经 `suggest_options` 伪工具(视情况给一个或两个)。
- `source_message_id` 是产生本组按钮的那一轮用户消息 ID,只以字符串传输,避免 JavaScript BIGINT 精度损失;不是会话 ID 或“当前最新消息”。同一轮至多发一帧,在最后一个 delta 与提交成功之后、`[DONE]` 之前由 log 发出。
- 顺序:`session` → delta/tool_start/tool_end → 提交 → citations(如有) → suggest_actions(如有) → `[DONE]`。非模型固定回复及 Agent 预算/空文本兜底也必须显式走 custom delta,不能依赖 messages 模式自动产生 token。

## 9. 动作端点(建工单)

`POST /v1/chat/action`:

- 请求 `ChatActionRequest`:`{user_id, session_id, source_message_id, action: "create_ticket", ticket_type: "售后|投诉|咨询"}`(action 枚举本章只有一个值,留扩展)。`source_message_id` 是必填的正十进制 ID 字符串,受数据库 ID 范围约束。
- 校验:会话存在且属于 user_id,source_message_id 对应消息属于该会话且 role 为 user;任一不满足返回 404(不泄露其他会话内容),参数非法返回 422。消息缺失时不得改用最新消息或固定描述建单。
- 行为:在同一个数据库 Session 中完成归属/消息查询,从指定用户消息取 description,调用 `write_ticket(db_session, conversation_id, description, ticket_type) -> ticket_no`。helper 只添加 Ticket 并按需 flush,不自行打开 Session、commit 或 rollback;事务由调用者提交,失败整体回滚。动作端点**不改 `conversations.status`**。
- description 使用被绑定用户消息的完整文本,长度受已有聊天输入上限约束;无需经过旧工具的 2000 字参数限制,也不调用模型摘要。同步数据库工作在线程内完成,端点沿用写操作 shield:取消时等事务落地再结束,不自动重试写操作。
- 响应 `{ticket_no, status: "待处理"}`。
- 旧 `create_ticket` 工具保留为兼容实现,但不注册进新聊天图。它在自己创建的同一个 Session 中调用 `write_ticket` + 置「已转人工」+ 一次 commit,任何一步失败整体回滚;旧工具级契约继续验证,不能据此恢复聊天 Agent 的写权限。
- 「转人工」不进此端点:本章纯前端模拟,不接真人系统、不做任何后端动作。

## 10. 会话状态与持久化

- State 只有 `messages` 使用 `add_messages` reducer;`thread_id = session_id`。同一 thread 的新调用会继承未覆盖字段,因此“走一遍全图”不等于“新建 State”。

### 10.1 State 字段与生命周期

| 字段 | 类型/取值 | 每轮入口与生命周期 |
|---|---|---|
| messages | `Annotated[list[BaseMessage], add_messages]` | 仅保留 log 已完成的完整对话历史;新轮不重传全量历史,也不直接把当前 HumanMessage 追加到此字段 |
| raw_query / resolved_query | `str` | raw_query 写当前原句;resolved_query 重置为空,由 resolve_reference 赋值 |
| intent / route | 七类意图或 None / 四出口或 None | 每轮重置 None,由分类与确定性路由赋值 |
| needs_knowledge | bool 或 None | 每轮 None;分类成功写布尔值,解析失败写 true,路由前不得为 None |
| retrieval_result | 检索结果全字段数据快照或 None | 每轮 None;包含 hits/query_plan 的可序列化值,不含运行时依赖 |
| retrieval_status / retrieval_error_code | `not_run/ok/low_confidence/unavailable` / str 或 None | 每轮 `not_run` / None,由 retrieve/confidence_gate 写入 |
| evidence | `list[dict]` | 每轮 `[]`,仅来自本轮检索结果 |
| low_conf_source / low_conf_reason | `retrieval_low_conf/self_check/None` / dict 或 None | 每轮 None;按 §6.4/§6.6 判定,不得沿用上一轮 |
| suggested_actions | `list[dict]` | 每轮 `[]`,保存本轮候选动作,log 成功后才发送 |
| agent_steps / agent_tokens | `int` | 每轮 0;分别记录已发起模型调用次数和本轮计费/估算消耗 |
| token_accounting | `none/usage/estimated/mixed` | 每轮 none,区分真实 usage 与缺失时估算(§7) |
| final_text / turn_messages | `str` / `list[BaseMessage]` | 每轮空文本 / `[]`;节点构造当前用户及本轮助手/工具消息,log 提交后追加历史 |
| source_message_id | str 或 None | 每轮 None,仅由 commit_turn 成功结果赋值 |
| node_trace | `list[dict]` | 每轮 `[]`,只记录本轮轨迹;不做跨轮追加 reducer |

- 服务在持有会话锁时通过同一个入口函数构造上述**全部临时字段**的初值,作为每次图调用的显式输入;TypedDict 注解本身不提供默认值。所有分支使用同一入口,不依赖某条分支恰好覆盖旧值。
- Agent 模型上下文从 `messages` 历史 + 当前 HumanMessage + 本轮工具往返临时构建;系统 prompt 与检索补充消息仅供本次模型调用使用,不累积进历史。每次调用按 §7 裁剪该副本,不删除账本历史。
- 固定回复路径与 Agent 路径都只提交当前 `turn_messages`;未到 log 就出错/取消的临时消息不追加到跨轮 messages。下一轮以新输入重新走全图,不自动恢复或重放失败轮的 log/写操作。

### 10.2 存储与故障边界

- checkpointer = `AsyncSqliteSaver`(langgraph-checkpoint-sqlite,含 aiosqlite),库文件 `data/checkpoints.db`(与 Milvus Lite 同风格,确认 .gitignore 覆盖 SQLite 主库及旁文件);在 `create_app` lifespan 的 `async with AsyncSqliteSaver.from_conn_string(...)` 内装配图并服务,退出时关闭,单 worker 约束不变。
- 职责分离:checkpoint = 图工作记忆(跨轮上下文),MySQL = 账本(会话/消息/低置信池,log 节点照旧写)。
- 已知限制(写进文档):checkpoint 文件丢失/删除时退化为「无历史新会话」,不回填 MySQL 历史;两库不做对账。
- MySQL 提交成功与后续 checkpoint 保存之间不存在跨库原子事务;checkpoint 保存失败不回滚已提交账本,不自动重试 log。日志记录 session_id 与 source_message_id 供排查;本章不提供断点重放/恰好一次提交保证。

## 11. 热身留档:examples/bare_agent.py

- 无框架最裸 Agent 循环:`model.bind_tools(tools)` + `invoke` + 手写「有 tool_calls 就执行并喂回、没有就收敛」的 for 循环 + max_steps。
- 工具复用真实只读业务工具(query_order / query_logistics,mock JSON 不触库),证明「同一个 Agent,框架只是壳」。
- 可 `uv run python examples/bare_agent.py` 直接运行:脚本自包含一个脚本化假模型(不依赖 tests/),默认走假模型演示;带开关可切真实模型。
- `tests/test_bare_agent.py` 用具有同步 `invoke` 接口的脚本假模型钉:一次收敛、工具结果喂回、max_steps 兜底;不把仅有 `astream` 的现有 FakeStreamModel 直接用于同步示例。
- 正式开发时把同一循环结构重构进 `agent_node.py`(invoke→astream、加 token 预算、加事件流出)。

## 12. 前端改动(chat.html,Vibe 例外照旧)

- 新增 suggest_actions 帧处理:在该条 assistant 气泡下渲染**各自独立的按钮**(复用 addFeedback 的组件形态;锁定粒度为「被点的那个按钮自己锁定防重」,另一个按钮不受影响,不用反馈组件的整组锁定)。
- 点「转人工」:纯前端渲染「已转接人工客服」系统提示 + 客服小猫问候气泡(「您好,我是客服小猫,请问有什么可以帮您的?」);不发任何请求。
- 点「建工单」:请求携带该气泡捕获的 user_id/session_id/source_message_id/ticket_type,不得读取随后变化的全局 sessionId 或“最近问题”;成功后渲染「已为您创建工单 T…」,失败给错误提示且不自动重试。
- 两按钮互不绑定;用户都不点、继续发消息 = 普通对话正常走(后端无任何动作)。
- `test_chat_page.py` 补字符串断言(两个按钮文案、小猫问候、action 调用),人工点验照旧。

## 13. 测试策略

新增:

- `tests/test_bare_agent.py` — 裸循环三态(收敛/喂回/兜底)
- `tests/test_graph_state.py` — 七类 intent × 两个 needs_knowledge 的 14 种组合参数化断言;同一 thread 的“低置信→正常业务”“投诉→闲聊”“知识 A→知识 B”“失败→新轮”均重置临时字段(含 needs_knowledge),保留已提交历史;不同 thread 隔离
- `tests/test_graph_nodes.py` — 意图/布尔值解析(含字符串 false、缺字段等非法情况),解析失败必经 retrieve/gate;§6.2 政策/纯业务/混合问题的路由与调用路径;置信度闸过与不过、三种配置/维护态及超时故障不入池、正常零命中入池、高分但模型拒答入 self_check、纯投诉/闲聊/故障固定 delta
- `tests/test_agent_node.py` — 一步收敛、订单→物流多步、追问收敛;8 次调用边界、累计 usage 一次结算、空内容 usage chunk、缺失/非法 usage 使用非零预留、调用前余额不足;多轮及工具结果超输入预算、工具组完整性、suggest_options 信号;绑定列表与执行注册表均无 create_ticket/query_faq,伪造 create_ticket 调用也不写库
- `tests/test_chat_action.py` — 绑定消息建单且不动 conv.status;继续聊天后点击旧按钮仍取原问题;消息属于其他会话/非 user/不存在返回 404,缺 ID 或非法参数返回 422;事务失败不留工单,取消等待事务结束
- `tests/test_graph_persistence.py` — SQLite 文件关闭重开仍保留完整历史;失败轮临时消息不进入后续模型历史;checkpointer 关闭与测试隔离。仅 `:memory:` 测试不足以验证文件持久化

改造(图语义重写,保留可保留的断言):

- `test_chat_api.py` HTTP/SSE 契约尽量不动(帧序/错误码/aclose 释锁)
- `test_chat_service.py` / `test_orchestration.py` / `test_chat_api_tools.py` 按图重写;旧编排专属契约(「第二次调用绑工具但不执行 tool_calls」、两次调用次数钉)随旧代码退役
- `test_prompts.py` 契约断言随 §7 条款修订更新
- `test_sessions.py` / `test_store_db.py` / `test_tool_chat_chain.py` — commit 返回当前用户消息 ID,多组工具往返可整体提交/重建/裁剪,不匹配 ID 仍拒绝,历史不重复落库
- `test_tools.py` — 兼容 create_ticket 的 helper 与状态更新同事务,故障注入验证整体回滚;新动作端点仅建单
- `test_spec11_safety.py` — 保留取消/提交/释锁及上下文预算的安全性质;旧两次调用次数改为图语义。原“聊天中建单后模型失败/取消仍保留工单”场景退役,改验聊天零建单副作用、动作端点写事务取消完成性;旧写工具的 shield 仍由 executor/tool 测试覆盖

验收 → 测试映射:

| 验收标准 | 验证方式 |
|---|---|
| 1. 政策类及混合问题日志可见强制检索节点 | 保修、订单取消、运费/物流混合样例 caplog 断言 retrieve;纯业务数据查询断言不检索;解析失败必须检索 |
| 2. 「订单 1001 的物流到哪了」Agent 自调工具 | SSE 集成:tool_start query_logistics,答复含 mock 物流数据 |
| 3. 纯投诉出两独立按钮,转人工纯前端,建工单写库,都不点正常聊 | suggest_actions 帧断言 + 前端字符串断言 + action 端点写库断言 + 次轮正常对话断言;带政策问题的投诉另验 knowledge 分支 |
| 4. 闲聊固定话术 | delta 帧 = CHITCHAT_REPLY;断言除意图识别外无生成调用 |
| 5. 先订单后物流的复杂问题 ReAct 多步 | tool_start 帧序列 [query_order, query_logistics],agent_steps ≥ 2 |
| 6. 不把服务维护和模型自评拒答混为一谈 | 三种维护/配置态、超时均不入池;零命中 retrieval_low_conf;高分后固定拒答 self_check,reason 含本轮证据引用 |
| 7. checkpoint 跨轮无临时状态串用 | 低置信/投诉/知识轮后切换分支,断言 evidence/actions/入池标记/计数重置,完整历史仍在 |
| 8. 原气泡按钮只为原问题建单 | 先投诉再查订单,点击投诉按钮,工单描述仍为投诉;ID 越权/缺失拒绝 |
| 9. 预算在发起下一次模型调用前生效 | 缺 usage、最后一步、长 evidence/工具结果三类测试;不能靠删当前问题、断开工具组或计零通过 |
| 10. 多步消息和按钮关联整体提交 | 两组工具调用均落库;commit 返回同轮 user 行 ID;失败无按钮/无 DONE,成功按钮在 commit 后 |
| 11. 聊天 Agent 无建单与转人工权限 | 明确要求人工/建单和伪造 create_ticket 两类输入均不写 tickets、不改 conv.status;只有 action 请求写工单,且不置已转人工 |

测试基建:假 retriever、dbfixtures(Docker MySQL);节点逻辑测试可适配现有 FakeStreamModel,但 graph messages 流集成测试必须使用能触发 LangChain 回调的假模型,覆盖 token 真正从图流出及内部调用过滤。checkpointer 主要用 AsyncSqliteSaver(`:memory:`),文件恢复测试用临时目录中的 SQLite 文件。实现交付前全量 `uv run pytest` 绿;付费 RAG 评估不属于本章自动运行的单测。

## 14. 依赖变更

- `+ langgraph`(1.0.x,Context7 已核 1.0.8 API:StateGraph/add_conditional_edges/compile(checkpointer)/get_stream_writer/stream_mode)
- `+ langgraph-checkpoint-sqlite`(AsyncSqliteSaver,含 aiosqlite)
- 实现前按用户要求再以 Context7 核对:AsyncSqliteSaver 生命周期用法、`add_messages` reducer、astream 多模式组合,版本对不上就停下问,不自行换方案。
- 本轮修订已核:同 thread 未覆盖字段会延续;AsyncSqliteSaver 使用异步上下文管理器管理连接;ChatOpenAI 需显式开启 stream_usage。参考 [checkpoint-sqlite 1.0.8 文档](https://github.com/langchain-ai/langgraph/blob/1.0.8/libs/checkpoint-sqlite/README.md) 与 [ChatOpenAI usage 文档](https://docs.langchain.com/oss/python/integrations/chat/openai)。

## 15. 风险与已知限制

1. AsyncSqliteSaver 接入 FastAPI lifespan 的连接管理(启停对称,测试隔离)。
2. 手写 ReAct 边界(空回复/工具异常/预算耗尽/伪工具误用)需测试全覆盖。
3. 旧服务层测试改造量集中在 test_chat_service / test_orchestration / test_chat_api_tools 三个文件。
4. 意图识别每轮 +1 次模型调用(延迟/成本),正式版再优化。
5. 指代消解透传 → 代词追问可能误分类(下一章正式版解决)。
6. checkpoint 与 MySQL 双写无对账;checkpoint 丢失退化为新会话(§10)。
7. reranker 降级到 hybrid 时闸失效(ch04 继承,可观测章处理)。
8. usage 缺失时累计 token 使用估算,不保证等于供应商实收;max_agent_steps 与每次输入/输出预算仍独立生效。
9. 按钮单次点击锁定只防当前页面重复操作,动作端点本章不承诺网络重试/跨页面请求幂等;前端不自动重试建单。
10. needs_knowledge 仍由最简分类模型判断,不宣称完全消除误分类;解析失败的保守检索可能使纯业务问题被拒答,混合问题过不了知识闸时整轮拒答,正式意图识别留下一章。
