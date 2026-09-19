# ch06 正式版分流器 设计文档

日期：2026-09-19
状态：brainstorm 定稿，待用户评审
前作：ch05 LangGraph Workflow(`docs/superpowers/specs/2026-09-17-ch05-langgraph-workflow-design.md`)

## §1 背景与目标

ch05 把聊天主链路切到 LangGraph 图，但分流器仍是半成品：`resolve_reference` 是原样透传的占位节点（nodes.py 注释自承「正式指代消解是后续章节的事」),`classify_intent` 不看历史、无「其他」兜底类、置信度不输出；退款/售后没有确定性子流程，业务工具全 mock 随机出数。

本章把分流器做成正式版，目标：

1. 指代消解 + Query 归一化改写（一次 LLM 调用，完整问句原样透传）
2. LLM 意图识别：七类业务意图 + 「其他」兜底，强制 JSON `{intent, confidence}`，四件套 prompt
3. 退款退货/售后走确定性子流程：先拿订单 → Query 扩写（`{queries: [...]}`，仅该场景）→ 强制检索 policy → 主力 Agent 只判「这一单能不能退」
4. 缺订单号不让模型猜：图内 `interrupt()` 挂起，前端弹订单选择器，点选后 `Command(resume=...)` 从挂起点继续
5. 退款单：固定类目下拉，复用 tickets 表落地

本章不做：微调/BERT 意图分类、跨会话记忆、双模型 cascade（只留配置项）、真实订单表（用确定性 mock)。

## §2 现状摘要

- 图拓扑（ch05):`START → resolve_reference(透传占位) → classify_intent → [route_by_intent 查 ROUTE_TABLE]`，出口 knowledge(retrieve→confidence_gate→main_agent/gate_fallback)、business(main_agent)、complaint、chitchat，全图唯一提交点 log。
- `INTENTS` 七类无「其他」;`ROUTE_TABLE` 按 `(intent, needs_knowledge)` 14 组合写死，`("退款退货", False)` 保守走 knowledge;「售后」+needs_knowledge=False 走 business 不检索。
- `INTENT_PROMPT` 输出 `{intent, needs_knowledge}`;`parse_intent_output` 失败兜底 `("售后", True)`。
- 检索：`retriever.search(query, scope=...)` 四策略混合 + 硅基流动重排；scope 白名单含 `policy`(returns-policy.md 派生）；置信闸分策略阈值。
- Query 改写已有先例：`app/knowledge/query_understanding.py` 的 `plan_query`(retriever 内部管道，本章不动它，子流程扩写另起 `EXPAND_PROMPT`)。
- 工具：query_order/query_product/query_logistics 全 random mock；无订单表（01-ddl.sql 头注「商品、订单、物流走工具内 mock，不建表」)。
- SSE:custom writer → chat_service 翻译 → routers 序列化 → 前端手解 SSE(fetch+ReadableStream)；帧序 `session → delta/tool_* → citations? → suggest_actions? → [DONE]`。
- checkpoint:AsyncSqliteSaver(./data/checkpoints.db),thread_id=session_id；图内现无任何 interrupt/Command。已核实环境装 langgraph 1.2.11,`from langgraph.types import interrupt, Command` 可用。
- 测试：FakeStreamModel / ScriptedChatModel 脚本化假模型；节点测试经编译小图驱动；evals/probe_intent.py + intent_samples.jsonl(8 条单句）为真模型意图基线。
- interrupt 官方语义（Context7 核实，docs.langchain.com/oss/python/langgraph/interrupts):resume 后**节点从头重跑**,`interrupt()` 调用点返回 resume 值；必须同 thread_id;pending interrupt 可经 `aget_state` 的 tasks 检查；`Command(resume=...)` 是唯一应作为输入的 Command。

## §3 已确认设计决策（brainstorm 定案）

| # | 决策 | 结论 | 备注 |
|---|---|---|---|
| D1 | 订单数据源 | 确定性 mock 订单服务（固定数据按 user 分桶），不建订单表 | 不碰 DDL 红线；query_order 工具改读它，子流程与 Agent 工具同源 |
| D2 | 意图 JSON 字段 | 只 `intent` + `confidence`;**删除 needs_knowledge 维度**，路由改纯 intent 映射 | 用户定死「字段就两个」 |
| D3 | 缺订单号回传架构 | **LangGraph interrupt/resume**（图内挂起，Command 恢复） | 用户纠偏：否决了「点选=发新消息」的简单方案 |
| D4 | 退款单落地 | 复用 tickets 表；`/v1/chat/action` 扩展 `create_refund`;ticket_type=「售后」,description 结构化 | 零 DDL 变更 |
| D5 | 意图模型 | 只用主模型（可配 `intent_model_name`)，双模型 cascade 不实现 | 成本吃紧时再做降级路 |
| D6 | 图拓扑 | 方案 A：顺向流水线加深，interrupt 只出现在 refund_prepare 一个节点 | 否决合并理解层（B）与独立子图（C) |
| D7 | 指代消解与改写 | 一次 LLM 调用合并完成（understand_query 节点），完整问句原样透传 | 用户原话「和指代消解一并处理」 |
| D8 | Query 扩写范围 | 仅退款退货/售后子流程；检索侧现查现用，入库侧不拆存 | 简单 FAQ 不扩 |
| D9 | 「其他」类 | 八类之一，固定 FALLBACK_ANSWER 兜底，零模型调用 | 拿不准就归它 |
| D10 | 子流程检索置信 | 复用现有置信评估，低置信/不可用 → gate_fallback | 与整轮预检索行为一致 |

## §4 模块布局

```
app/graph/
  state.py          # +intent_confidence/order_context/expanded_queries/understanding_degraded; INTENTS+「其他」; ROUTE_TABLE 纯映射
  nodes.py          # understand_query(替换 resolve_reference) / classify_intent(新契约) / refund_prepare / refund_policy / other_fallback
  builder.py        # 新边与路由映射
  events.py         # +ev_order_selector
app/prompts/
  understand.py     # 新 UNDERSTAND_PROMPT
  intent.py         # 重写 INTENT_PROMPT(八类、intent+confidence、few-shot、其他兜底)
  expand.py         # 新 EXPAND_PROMPT
app/services/
  orders.py         # 新 确定性 mock 订单服务 list_orders/get_order
  chat_service.py   # +resume 通道(Command 驱动)、_translate_custom 新分支
app/routers/chat.py # +POST /v1/chat/resume;action 帧序列化不变
app/schemas.py      # +ChatResumeRequest;ChatActionRequest 加 create_refund/order_id/refund_reason
app/tools/business.py # query_order 改读 orders 服务
app/static/chat.html  # 订单选择器卡片 + 退款表单(Vibe 例外)
evals/
  intent_samples.jsonl  # 扩充:八类+边界+多轮指代
  probe_intent.py       # 适配新 JSON,支持多轮
  probe_router_multi.py # 新 多轮剧本 probe(验收 1/3)
tests/                  # 见 §13
```

## §5 图拓扑

```
START → understand_query → classify_intent → [route_by_intent 纯 intent 映射]
  business  (物流, 订单)        → main_agent
  knowledge (商品咨询)          → retrieve → confidence_gate → main_agent / gate_fallback
  refund    (退款退货, 售后)    → refund_prepare → refund_policy → main_agent
                                  refund_prepare 缺订单号 → 发 order_selector 帧 → interrupt() 挂起
                                  refund_policy 低置信/不可用 → gate_fallback
  complaint (投诉)              → complaint_reply
  chitchat  (闲聊)              → chitchat_reply
  other     (其他)              → other_fallback
main_agent / gate_fallback / complaint_reply / chitchat_reply / other_fallback → log → END
```

路由表（替换 14 组合）:

```python
ROUTE_TABLE = {
    "物流": "business", "订单": "business",
    "商品咨询": "knowledge",
    "退款退货": "refund", "售后": "refund",
    "投诉": "complaint", "闲聊": "chitchat",
    "其他": "other",
}
```

挂起轮形态：order_selector 帧发出后 SSE 流正常收尾（无 delta、无 citations/suggest_actions、直接 [DONE]),log 不提交，会话锁释放。resume 请求驱动同一 thread 继续，走完子流程后 log 正常提交：**用户原始问句 + 助手答复作为一轮落库**,validate_turn(user 开头 assistant 结尾）不变式保持。挂起期间用户直接发新消息 → 走 new_turn_state 全新轮；旧挂起态不再恢复（实现期用测试钉死 LangGraph 实际语义：新输入从 START 起跑后，旧 interrupt 的 resume 请求应 409)。

## §6 节点契约

### 6.1 understand_query（替换 resolve_reference)

- 输入：state.messages(checkpoint 历史，最近 N=6 条）+ raw_query;LLM 一次 `ainvoke`(UNDERSTAND_PROMPT)。
- 输出：`resolved_query`；解析失败/模型异常 → `resolved_query = raw_query` 且 `understanding_degraded=True`，不炸轮（模型异常不发 upstream_error，与 classify 的 TurnAbort 不同——理解层降级代价只是回到透传）。
- 历史为空时跳过 LLM 调用，直接透传（首轮零成本）。

### 6.2 classify_intent（改契约）

- 输入：仅 `resolved_query`（历史已由 understand 消化）。
- LLM `ainvoke`(INTENT_PROMPT)→ `parse_intent_output` → `(intent, confidence)`;intent 白名单八类；confidence 非法（非数/越界）置 None 不拒判。
- 解析失败兜底 `("其他", None)` → other_fallback；模型异常 → upstream_error + TurnAbortError（同现状）。
- confidence 本章只记录进 node_trace / state,不参与路由（D5)。

### 6.3 refund_prepare

- 从 `resolved_query` 确定性提取订单号：正则匹配 mock 订单集合（不让模型猜，D1/D3)。
- 命中 → `order_context = get_order(oid)` 快照，直通 refund_policy。
- 未命中 → writer 发 `ev_order_selector(list_orders(user_id))` → `selected = interrupt({"type": "order_selector"})` 挂起；resume 后节点重跑，`interrupt()` 返回 `{"order_id": ...}`，校验属于该用户 → 写 order_context 继续；非法订单号 → 落 other_fallback（防伪造 resume)。
- user_id 来源：GraphDeps 注入的 store 会话上下文（实现期定具体传递方式：state 新增不入库的上下文字段或 deps 闭包，plan 阶段定）。

### 6.4 refund_policy

- LLM `ainvoke`(EXPAND_PROMPT，输入 resolved_query + order_context 摘要）→ `parse_expand_output` → queries(2~4 条；解析失败 → `[resolved_query]` 单条）。
- 多条查询 `asyncio.gather` 并发 `asyncio.to_thread(retriever.search, q, scope="policy")`，按 chunk_id 去重合并（保最高分），合成 RetrievalResult 走现有置信评估；写 `expanded_queries`。
- 低置信/不可用 → gate_fallback（复用现有固定话术与 error_code 语义）；过闸 → evidence = 政策条款 + 「订单上下文」块。

### 6.5 other_fallback

- 固定 FALLBACK_ANSWER delta，零模型调用，→ log。

### 6.6 main_agent（零结构改动）

- evidence 渲染增加「订单上下文」块（有 order_context 时）；系统提示不变；判断「这一单能不能退」；可经 suggest_options 建议「申请退款」。

### 6.7 log（小改）

- 挂起轮不到达 log（无提交）；resume 轮到达时按现契约提交；suggest_actions 选项新增 `{"action": "refund_form", "order_id": ...}` 的透传（action 白名单 +1)。

## §7 Prompt 契约

惯例沿用：字符串常量、`.replace()` 渲染（字面 JSON 花括号）、正则提 JSON + 校验 + 结构化降级。

1. `UNDERSTAND_PROMPT`(app/prompts/understand.py)：输入最近 N 轮历史 + 当前问句；输出单行 JSON `{"resolved_query": "..."}`；规则：已完整/指代明确 → 原样透传；有指代 → 补全为独立可懂问句；口语模糊 → 归一标准问法。
2. `INTENT_PROMPT`（重写）:① 八类枚举选择题（七类定义 + 「其他」：拿不准/跨类/无法判断时选它）;② 强制 JSON 就 `{intent, confidence}`;③ 边界 few-shot（投诉 vs 售后、退款 vs 售后、闲聊 vs 其他、怪问题→其他）;④ 删除 needs_knowledge 全部规则。
3. `EXPAND_PROMPT`(app/prompts/expand.py)：仅子流程；输入 resolved_query + 订单摘要；输出 `{"queries": [...]}` 2~4 条、侧重点不同（能不能退/时效/运费/流程）。

解析函数：`parse_understand_output` / `parse_expand_output` 新增，`parse_intent_output` 改签名为 `(text) -> (intent, confidence)`；风格对齐现有 parse_plan。test_prompts.py 更新 + 新增契约钉（八类枚举齐、两字段、`.replace` 可渲染）。

## §8 SSE 协议增量

新帧（四层同改：events.py → chat_service._translate_custom → routers 序列化 → chat.html 分发）:

```json
{"type": "order_selector", "orders": [{"order_id": "1001", "product": "保温杯", "amount": 89.0, "status": "已完成", "created_at": "..."}]}
```

帧序：挂起轮 `session → order_selector → [DONE]`;resume 轮 `session → delta → citations? → suggest_actions? → [DONE]`（与现序一致）。suggest_actions 的 options 元素新增 `{"action": "refund_form", "label": "申请退款", "order_id": "..."}`。

## §9 端点

- `POST /v1/chat/resume`（新）：请求 `{user_id, session_id, order_id}`(ChatResumeRequest)。校验：会话归属（404 同现有语义）→ `graph.aget_state` 确认 pending interrupt 存在（无 → 409)→ 订单属于该用户（404/409)→ 会话锁 → `astream(Command(resume={"order_id": oid}), config, stream_mode=["messages","custom"])`。响应为标准 SSE 流。
- `POST /v1/chat/action`（扩展）:`action` Literal + `"create_refund"`；请求 + `order_id: str`、`refund_reason: Literal["质量问题","商品破损","发错货","未收到货","拍错或多拍","七天无理由"]`。落 tickets:ticket_type=「售后」,description=`退款单:订单 {order_id},原因:{refund_reason}`；沿用归属校验（404 不泄露）、不改 conv.status、响应 `{ticket_no, status}`。

## §10 State 字段生命周期

新增（全部进 new_turn_state 重置 + test_graph_state 钉）:

| 字段 | 类型 | 写入者 | 说明 |
|---|---|---|---|
| `intent_confidence` | float \| None | classify_intent | 只记录不路由 |
| `order_context` | dict \| None | refund_prepare | 订单快照，进 evidence |
| `expanded_queries` | list[str] | refund_policy | 扩写结果（含单条兜底） |
| `understanding_degraded` | bool | understand_query | 透传降级标记 |

既有 `resolved_query` 语义不变（现在是真消解结果）。`messages` 契约不变：log 提交成功后才追加。挂起轮 state 冻结在 checkpoint;resume 重跑 refund_prepare 时其上游字段（resolved_query/intent 等）在 checkpoint 中保持，不重置——**new_turn_state 只在全新轮入口调用，resume 路径传 Command 不传 dict**。

## §11 mock 订单服务

`app/services/orders.py`：模块级固定数据，按 user_id 哈希分桶或直接每 user 同一组演示单（plan 阶段定），3~4 单覆盖：7 天内可退 / 超期不可退 / 定制或生鲜不支持退货 / 在途。API:`list_orders(user_id) -> list[OrderInfo]`、`get_order(order_id) -> OrderInfo | None`(OrderInfo: order_id/product/amount/status/created_at/delivered_at/returnable_note)。`query_order` 工具改读 get_order（保持返回 JSON 结构兼容现有 Agent 消费）。测试 fixtures 直接复用该模块数据。

## §12 前端改动（Vibe 例外）

不进硬契约，效果约定：订单卡渲染在聊天流内（订单号/商品/金额/状态/时间，CSS token 体系，DOM 构建禁 innerHTML)，点选 → 本地 hint「已选择订单 ×××」→ POST /v1/chat/resume 继续读流；「申请退款」按钮 → 弹层表单（订单号回填、原因下拉固定类目）→ create_refund → 提示工单号。断言方式不变：test_chat_page 加 needle + 人工点验（验收 4)。

## §13 测试策略

新增（pytest,fake model):
- test_graph_nodes:understand_query（消解/透传/降级/首轮跳过）、classify_intent 新契约、refund_prepare（直通/挂起发帧）、refund_policy（合并去重/gate 分支）、other_fallback;parse_* 三组参数化。
- test_graph_state:INTENTS 八类、ROUTE_TABLE 纯映射八行、新字段重置钉。
- test_refund_flow（新，ScriptedChatModel + InMemorySaver/SQLite)：缺号→order_selector 帧→[DONE]→resume→子流程走完→log 落库一轮；挂起后发新消息语义钉；resume 409/404;create_refund 全路径。
- test_chat_api / test_chat_action:resume 端点、action 扩展（枚举校验、归属 404)。
- test_chat_page：订单卡/resume/退款表单 needle。

改造：conftest 两个 fake model 的罐头 ainvoke 按 prompt 分派（understand/classify 罐头）、test_prompts、test_ch05_acceptance 中受路由表/占位节点改动影响的断言。

evals（真模型手跑，不进 pytest):
- intent_samples.jsonl 扩充至八类+边界+多轮（带 history);probe_intent.py 适配。
- probe_router_multi.py：多轮剧本（物流→退款→物流；「这个能退吗」)，逐轮打印消解/意图/路由/子流程。

验收映射：

| 验收 | 验证方式 |
|---|---|
| 1 多轮意图+指代全对 | evals/probe_router_multi.py 真模型跑剧本 + 多轮样例入 intent_samples |
| 2 JSON 稳定可解析、怪问题落「其他」 | probe_intent 边界样例 + pytest parse 参数化 + classify 兜底钉 |
| 3 「这个能退吗」补全指代→子流程拿订单和政策 | 集成测试断言帧序与 evidence 内容 + probe 逐轮输出人工核 |
| 4 浏览器缺号弹订单卡、点选走完 | 人工点验 + resume 端点集成测试 + chat.html needle |

门槛：全量 pytest 绿（Docker 在线）;evals probe 真模型跑一轮，命中率记 dev-notes。

## §14 依赖与配置

无新增包。新增配置项：`intent_model_name`（默认 None=复用主模型）、`understand_history_turns`（默认 6)、`refund_expand_enabled`（默认 True，降级开关）。LangGraph interrupt/Command 用法以 Context7 拉取的 1.x 官方文档为准（§2 末）。改 prompts 后跑全量 pytest(AGENTS.md 红线）。

## §15 风险

- 挂起态语义：新输入覆盖 pending interrupt 的精确行为在文档中无逐字说明 → 实现期先写语义探测测试钉死，再定 resume 409 边界。
- refund_prepare 重跑幂等：interrupt 前的 writer 发帧在重跑时不应重复发——interrupt() 调用点之后的代码才带 resume 值，发帧逻辑须放在 interrupt 调用之内或之后跳过（plan 阶段细化，必要时把发帧挪到节点外由驱动层据 aget_state 发）。
- 会话锁与挂起：挂起轮必须在流收尾的 finally 里释锁（现成结构满足）;resume 重新 acquire，长时间挂起不占锁。
- messages 护栏：messages 模式只外发 main_agent 的 AIMessageChunk,refund 子流程无模型流式输出，不冲突。
- 演示数据自洽：mock 订单状态/时间与 returns-policy(7 天无理由）要对得上，避免演示时「能退/不能退」讲反。
