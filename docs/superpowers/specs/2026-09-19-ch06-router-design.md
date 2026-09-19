# ch06 正式版分流器 设计文档

日期：2026-09-19
状态：评审修订定稿；通用 Agent 按需只读检索、通用售后跳过选单已获用户确认
前作：ch05 LangGraph Workflow(`docs/superpowers/specs/2026-09-17-ch05-langgraph-workflow-design.md`)

## §1 背景与目标

ch05 把聊天主链路切到 LangGraph 图，但分流器仍是半成品：`resolve_reference` 是原样透传的占位节点（nodes.py 注释自承「正式指代消解是后续章节的事」),`classify_intent` 不看历史、无「其他」兜底类、置信度不输出；退款/售后没有确定性子流程，业务工具全 mock 随机出数。

本章把分流器做成正式版，目标：

1. 指代消解 + Query 归一化改写（一次 LLM 调用，完整问句原样透传）
2. LLM 意图识别：七类业务意图 + 「其他」兜底，强制 JSON `{intent, confidence}`，四件套 prompt
3. 物流/订单/商品咨询进入通用 Agent，按需调用只读业务工具或知识检索工具；纯业务查询不依赖知识库可用性
4. 退款退货/售后走图内子流程：通用政策/流程问题直接强制检索；具体订单问题先拿订单，再按需扩写并强制检索，主力 Agent 依据订单事实与适用证据回答；不明确的诉求由 Agent 澄清
5. 需要订单却缺号时不让模型猜：图内 `interrupt()` 挂起，前端弹订单选择器，点选后按 interrupt ID 恢复原轮
6. 退款单：固定类目下拉，复用 tickets 表落地

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
- interrupt 官方语义（Context7 核实，[官方文档](https://docs.langchain.com/oss/python/langgraph/interrupts)）：resume 后**节点从头重跑**，`interrupt()` 调用点返回 resume 值；必须同 thread_id；pending interrupt 可经 `aget_state` 的 tasks 检查，支持按 interrupt ID 传 `Command(resume={interrupt_id: value})`。
- 已在本地 langgraph 1.2.11 用 InMemorySaver 小图复核（2026-09-19)：挂起时传新 dict 会从 START 开始新轮并产生新的 interrupt ID；**不匹配的 resume ID 被静默忽略**(LangGraph 不报错、不执行，图保持 pending)，不绑定 ID 的裸 resume 值会作用于当前 pending interrupt；对已完成的图重复 resume 不报错，astream 会回放当前最终 state(stream_mode="values" 下有非空 chunk)。因此 HTTP 409 必须由服务端在会话锁内按挂起实例校验，不能依赖 LangGraph 自动拒绝。

## §3 已确认设计决策（含本轮评审修订）

| # | 决策 | 结论 | 备注 |
|---|---|---|---|
| D1 | 订单数据源 | 确定性 mock 订单服务，订单按 user 隔离，不建订单表 | 统一 `get_order(user_id, order_id)`；直通、resume、只读工具和退款建单共用归属检查 |
| D2 | 意图 JSON 字段 | 只 `intent` + `confidence`;**删除 needs_knowledge 维度**，路由改纯 intent 映射 | 用户定死「字段就两个」 |
| D3 | 缺订单号回传架构 | **LangGraph interrupt/resume**（图内挂起，Command 恢复） | 用户纠偏：否决了「点选=发新消息」的简单方案 |
| D4 | 退款单落地 | 复用 tickets 表；`/v1/chat/action` 扩展 `create_refund`;ticket_type=「售后」,description 结构化 | 零 DDL 变更 |
| D5 | 意图模型 | 只用主模型（可配 `intent_model_name`)，双模型 cascade 不实现 | 成本吃紧时再做降级路 |
| D6 | 图拓扑 | 顺向流水线；退款分支先判断是否需要订单，interrupt 只出现在 refund_prepare | 通用问题绕过订单准备；保留独立理解层，不引入子图 |
| D7 | 指代消解与改写 | 一次 LLM 调用合并完成（understand_query 节点），完整问句原样透传 | 用户原话「和指代消解一并处理」 |
| D8 | Query 扩写范围 | 仅退款退货/售后的具体订单分支；检索侧现查现用，入库侧不拆存 | 通用政策/流程问题与其他简单 FAQ 不扩 |
| D9 | 「其他」类 | 八类之一，固定 FALLBACK_ANSWER 兜底，零模型调用 | 拿不准就归它 |
| D10 | 子流程检索置信 | 多查询按 ID 合并候选后统一重排，再用现有置信信号；低置信/不可用 → gate_fallback | 统一重排失败回退 base_query 的独立结果，禁止混用不同策略分数 |
| D11 | 通用业务分支 | 物流/订单/商品咨询 → main_agent，允许按需调用图专用只读 query_faq | 用户已确认；本章扩展 ch05 只读白名单，纯库存查询无需检索，Agent 无写权限仍保持 |
| D12 | 售后订单需求 | refund_scope 判断 general / order_specific / clarify；仅 order_specific 进入 refund_prepare | 用户已确认通用问题跳过选单；判断发生在执行分支，不给 intent JSON 增加字段 |

评审修订补充：订单卡绑定 interrupt ID，pending 校验与恢复共用会话锁；已确认订单在同一 thread 内保存为 `active_order`；订单事实与知识引用分别渲染；退款建议只绑定服务端已校验的订单。

## §4 模块布局

```
app/graph/
  state.py          # +intent_confidence/refund_mode/order_context/active_order/expanded_queries/understanding_degraded; 生命周期见 §10
  nodes.py          # understand_query / classify_intent / refund_scope / refund_prepare / refund_policy / other_fallback
  builder.py        # 新边与路由映射
  events.py         # +ev_order_selector
  agent_node.py     # 通用分支只读检索与置信处理、退款建议、订单上下文渲染
app/prompts/
  understand.py     # 新 UNDERSTAND_PROMPT
  intent.py         # 重写 INTENT_PROMPT(八类、intent+confidence、few-shot、其他兜底)
  refund.py         # 新 REFUND_SCOPE_PROMPT，仅判断执行分支是否需要订单
  expand.py         # 新 EXPAND_PROMPT
  service.py        # 知识证据与订单事实分开约束、退款建议只读语义
app/knowledge/
  retriever.py      # 候选统一重排的公开接口；search/plan_query 的既有单查询语义不变
app/services/
  orders.py         # 新 确定性 mock 订单服务 list_orders/get_order
  chat_service.py   # +resume 通道、锁内 pending 校验、挂起后的 OrderSelectorEvent
app/routers/chat.py # +POST /v1/chat/resume;action 帧序列化不变
app/schemas.py      # +ChatResumeRequest;ChatActionRequest 加 create_refund/order_id/refund_reason
app/tools/business.py # 用户范围的订单/物流工具；独立构建图专用只读 query_faq
app/static/chat.html  # 订单选择器卡片 + 退款表单(Vibe 例外)
evals/
  intent_samples.jsonl  # 扩充:八类+边界+多轮指代
  probe_intent.py       # 适配新 JSON,支持多轮
  probe_router_multi.py # 新 多轮剧本 probe(验收 1/3)
tests/                  # 见 §13
```

## §5 图拓扑

分类输出仍只有 intent/confidence，首层路由仅查 intent。通用分支按需检索；退款/售后分支先判断是否需要订单，再由图强制检索。refund_scope 是执行分支内的判断，不恢复 needs_knowledge 分类维度。

```
START → understand_query → classify_intent → [route_by_intent 纯 intent 映射]
  business  (物流, 订单, 商品咨询) → main_agent（可按需调用只读 query_faq）
  refund    (退款退货, 售后)      → refund_scope
      general                   → refund_policy → main_agent / gate_fallback
      order_specific            → refund_prepare → refund_policy → main_agent / gate_fallback
                                    缺号时 interrupt() 挂起 → 驱动层发 order_selector
      clarify                   → main_agent（仅澄清诉求）
  complaint (投诉)              → complaint_reply
  chitchat  (闲聊)              → chitchat_reply
  other     (其他)              → other_fallback
main_agent / gate_fallback / complaint_reply / chitchat_reply / other_fallback → log → END
```

路由表（替换 14 组合）:

```python
ROUTE_TABLE = {
    "物流": "business", "订单": "business",
    "商品咨询": "business",
    "退款退货": "refund", "售后": "refund",
    "投诉": "complaint", "闲聊": "chitchat",
    "其他": "other",
}
```

refund_policy 的条件边按 retrieval_status 选择 main_agent 或 gate_fallback；refund_prepare 无可选订单或恢复后的防御性校验失败时转 other_fallback。原 knowledge 顶层出口退出本章主图；已有 retrieve/confidence_gate 的检索状态映射、证据组装和闸门逻辑提取复用，不能因换入口丢失错误码与低置信记录。

| 问题 | intent / 执行分支 | 预期能力 |
|---|---|---|
| 保温杯还有库存吗 | 商品咨询 / business | query_product；不调用 query_faq，知识库不可用也可查询 |
| 未发货订单如何取消 | 订单 / business | query_faq 取得 FAQ 中的取消规则后回答 |
| 自动饮水机 MH-W20 的水箱容量是多少 | 商品咨询 / business | query_faq 取得 product-specs 中的 2L 规格后回答 |
| 维修寄修流程是什么 | 售后 / general | 不选单，强制检索售后手册 |
| 如何申请退款 | 退款退货 / general | 不选单，强制检索 FAQ 中的申请步骤 |
| 这个能退吗 / 我要申请退款 | 退款退货 / order_specific | 查已确认订单；缺号时选单，再查适用政策 |
| 这台设备还在保修期吗 | 售后 / order_specific | 查订单签收信息与保修政策 |

「投诉」限纯投诉情绪；含明确业务诉求时按该业务归类。明确的退货/保修个案与其他业务混合时优先走 refund；其余可识别主诉求按主意图，保留完整问句供检索/工具处理。只有无法确定主诉求才落「其他」，不能把所有多子问一律拒答。

挂起轮形态：图在 refund_prepare 挂起后，驱动层在仍持有会话锁时读取 pending interrupt，发送绑定其 ID 的 order_selector，随后 `[DONE]` 并释锁。此轮无 delta、citations 或 suggest_actions，不到达 log。resume 驱动同一 thread 继续，完成后将**用户原始问句 + 工具往返（如有）+ 助手答复作为一轮落库**，保持 validate_turn 不变式。

挂起期间用户发新消息，入口仍先获取同一会话锁，再用 new_turn_state 从 START 开始新轮，旧卡作废。即使新轮也挂起，旧卡携带的 interrupt ID 仍须返回 409；判断依据是当前 pending ID 是否匹配，不能只看「是否存在任意 pending」。resume 不接受 checkpoint_id，不允许客户端恢复旧 checkpoint。

## §6 节点契约

### 6.1 understand_query（替换 resolve_reference)

- 输入：state.messages 中最近 N=6 个完整对话轮（包含轮内成对工具消息）+ raw_query + 同会话 `active_order` 的已校验订单摘要；按 token 预算裁剪最旧完整轮。N 的单位统一为轮，不能直接切最后 N 条消息。
- `active_order` 只辅助明确的指代，如「这单」「刚才选的订单」；完整问题原样透传，新出现的订单或商品优先。存在多个候选或指代不明时保留未消解内容，不能猜订单号。理解输出中的订单号必须可追溯到用户原文或已校验的会话订单，凭空新增则按解析失败降级。
- 输出：`resolved_query`；解析失败/模型异常 → `resolved_query = raw_query` 且 `understanding_degraded=True`，不炸轮（模型异常不发 upstream_error，与 classify 的 TurnAbort 不同——理解层降级代价只是回到透传）。
- 历史与 active_order 均为空时跳过 LLM 调用，直接透传（首轮零成本）；其余情况一次 `ainvoke(UNDERSTAND_PROMPT)`。

### 6.2 classify_intent（改契约）

- 输入：仅 `resolved_query`（历史已由 understand 消化）。
- LLM `ainvoke`(INTENT_PROMPT)→ `parse_intent_output` → `(intent, confidence)`;intent 白名单八类；confidence 非法（非数/越界）置 None 不拒判。
- 解析失败兜底 `("其他", None)` → other_fallback；模型异常 → upstream_error + TurnAbortError（同现状）。
- confidence 本章只记录进 node_trace / state,不参与路由（D5)。

### 6.3 refund_scope / refund_prepare

**refund_scope** 只在 refund 分支调用一次主模型，以 raw_query/resolved_query 判断处理该诉求是否需要具体订单事实，输出单行 `{"mode": "general|order_specific|clarify"}`，经枚举校验写入 refund_mode；此 JSON 与 intent 分类契约独立：

- general：询问通用规则、步骤或故障自查，不需要某笔订单数据，例如「维修寄修流程是什么」「如何申请退款」「饮水机不出水怎么检查」。即使已有 active_order，也不因此要求选单。
- order_specific：要求判断某单资格、期限、个案状态或发起该单退款，例如「这单能退吗」「还在保修期吗」「我要申请退款」。没有订单号不改变该模式，而是在 refund_prepare 选单。
- clarify：无法确定请求是什么；解析失败或模型异常也降级到 clarify 并记 node_trace。直接进入 main_agent，附加「仅提出简短澄清问题」的执行约束，不查询订单、不检索、不猜选单需求。本节点不向用户追问。

refund_scope 位于 interrupt 上游，resume 不重跑它。**refund_prepare** 只接收 order_specific：

- 订单服务统一暴露 `get_order(user_id, order_id)`。用户原文中的显式订单号优先；resolved_query 中补出的订单号须通过 §6.1 来源校验。唯一候选且属于该用户时，将实时查询的快照写入 `order_context`，再进入 refund_policy。
- 缺号、多候选或显式订单不存在/不属于用户时，调用 `interrupt({"type": "order_selector", "orders": list_orders(user_id)})`；不回显他人订单详情，也不偷偷改用旧 active_order。没有可选订单时转 other_fallback，避免挂起在空卡上。
- resume 重跑节点，`interrupt()` 返回 `{"order_id": ...}` 后再次执行用户范围查询；通过才写 order_context，防御性校验失败转 other_fallback。HTTP 层已在驱动前按 §9 返回可识别的 404/409。
- 节点不发送 order_selector，也不在 interrupt 前写账本或修改 active_order。驱动层在图挂起后读取 `tasks[].interrupts`，以运行时分配的 ID 和 interrupt.value.orders 生成帧；恢复成功的流中不重复发送选择器。
- user_id 由 ChatService 校验会话归属后逐请求放入 `configurable.user_id`，与 thread_id 一同传给普通调用和 resume。节点从 config 读取，只读工具由当前调用闭包绑定；共享 GraphDeps 不保存可变的「当前用户」，模型参数不能指定或覆盖用户。

### 6.4 refund_policy

- general 与 order_specific 都由此节点强制检索并过置信闸。general 无需订单，base_query=resolved_query；order_specific 将已校验订单的商品/类别事实附加到 resolved_query 形成 base_query，避免选单后仍用「这个能退吗」作为无对象的检索问题。附加内容仅用于检索，raw_query 落库口径不变。
- 检索统一使用 `scope=None`，覆盖 policy、after_sales_manual、faq 及其他适用资料；不把全部退款/售后问题限制在 policy。必须根据证据内容回答当前诉求：个案退货/保修资格需要适用条款，寄修步骤可引用售后手册，申请步骤可引用 FAQ；相关分数高不等于条款适用于该商品。
- general 只查 base_query，不调用 EXPAND_PROMPT。仅 order_specific、`refund_expand_enabled=True` 且策略为 hybrid_rerank 时，一次 `ainvoke(EXPAND_PROMPT)` 得到 2~4 个扩写问句；去空、去重后以 base_query 为第一条，总计最多 4 条（至多 3 条新增问句）。解析失败、模型异常或开关关闭时回到 `[base_query]`；dense/bm25/hybrid 保留原单查询路径及各自阈值，不为了扩写隐式切换策略。expanded_queries 记录实际执行的列表。
- 多查询以 `asyncio.gather` 并发 `asyncio.to_thread(retriever.search, q, scope=None, deadline=deadline)`，保留 base_query 的完整 RetrievalResult 作为独立回退结果。整次检索/合并重排共用 knowledge_tool_timeout_seconds 总等待上限及 deadline，节点层不另起重试循环。
- 各查询结果只贡献候选正文，按 chunk_id 去重，稳定排序后最多包含 `4 * rerank_top_n` 个候选；**不比较、取最大值或混合这些结果的 score/confidence_threshold**，因为其中可能同时存在 hybrid_rerank 与降级 hybrid。
- 通过 retriever 的公开 `rerank_candidates(query, hits, deadline)`，用 base_query 对候选集合统一重排；节点不直接操作其私有客户端。重排成功后构造单一 hybrid_rerank RetrievalResult，按现有 confidence_from_scores/rerank_confidence_signal 与 rerank_min_score 判闸，再按既有 evidence 数量和字符预算组装。订单事实仅在 Agent 输入处单独附加。
- 统一重排失败时使用 base_query 回退结果的完整策略、分数、阈值与 note，追加可观测的扩写降级原因；不把 RRF 分数装成 rerank 分数。无新增有效候选时直接使用该回退结果。
- base_query 检索异常/超时或遇到知识库配置、维护不可用状态时走 unavailable；任一查询发现重建/需重建状态也按维护不可用处理。仅扩写支路的临时异常可丢弃该支路并记录原因；普通无命中/低分候选不等于服务故障。总等待超时统一 kb_unavailable，不使用之后才返回的线程结果。
- 最终低置信/不可用 → gate_fallback，保留现有 error_code、固定话术及低置信入池语义；过闸只将知识条款写入 evidence。订单上下文不能代替缺失政策证据。

### 6.5 other_fallback

- 固定 FALLBACK_ANSWER delta，零模型调用，→ log。

### 6.6 main_agent（保留 ReAct 循环，补齐输入与建议契约）

**只读工具与检索：**

- business 分支绑定 query_order/query_product/query_logistics/query_faq + suggest_options；refund 已过图内检索的分支绑定三种业务工具 + suggest_options，不再绑定 query_faq；refund_mode=clarify 时不绑定工具，只澄清诉求。其他固定回复分支不进入 Agent。
- 图专用 `query_faq()` 无模型参数，以本轮完整 resolved_query 和 `scope=None` 调用现有 retriever.search；模型只决定是否需要检索，不缩窄子问、不指定用户或过滤范围。保留检索器内部 plan_query，通用分支不使用本章 EXPAND_PROMPT。现有旧版/评估用带参数 query_faq 保持兼容，不能为恢复该工具而重新调用包含 create_ticket 的旧 build_tools。
- 每轮首次 query_faq 执行后缓存其完整检索结果；再次合法调用返回相同结果和编号，不重复外部检索。工具白名单、参数拒绝及本轮缓存均在当前调用内构建，不跨用户/thread 共享。
- 本轮局部检索状态沿用 retrieval_result/retrieval_status/retrieval_error_code/evidence/low_conf_source/low_conf_reason。成功时先按 assemble_evidence 分配引用号，将同一 evidence 返回 ToolMessage 并用于后续模型上下文；节点结束时回写 State，供 log 引用和入池。query_faq 作为真实只读工具发送 tool_start/tool_end，suggest_options 仍不产生工具徽章。
- query_faq 使用 knowledge_tool_timeout_seconds/deadline 和现有检索异常/维护状态映射。低置信、无命中或不可用时不给模型可引用的候选正文；先为当前同一组 tool_calls 补齐匹配 ToolMessage，再直接以 REFUSAL_ANSWER 或 KB_UNAVAILABLE_ANSWER 固定收尾，不再调用模型。低置信记录入池，不可用不入池；不得把某个失败结果的正文与上一轮引用混用。
- 纯订单/物流/库存查询无需 query_faq，也不因知识库不可用而被拒答；一旦本轮主动请求知识且未过闸，按上述固定话术结束整轮。通用政策/规格问题的检索触发由系统提示和 Agent 决策保障，强制图检索保证仅适用于 refund 分支；评估须覆盖漏调用工具的真实模型情况。

**模型上下文与建议动作：**

- `evidence` 保持知识 Evidence 列表的原结构；`order_context` 单独渲染为「本轮已校验订单事实」，包含查询时刻，供 Agent 将政策用于当前订单。订单块不生成 ref_no/chunk_id，不混入知识引用或低置信原因中的 evidence_refs。
- 模型上下文包含 resolved_query 的理解结果，同时保留 raw_query 作为唯一用户消息落库；订单事实、理解结果及本轮证据都纳入上下文预算。系统提示区分订单事实与政策依据，不把订单状态当成支持退款的政策。
- query_order/query_logistics 成功后，由处理器保留已校验的订单快照；仅用户本轮明确指定或明确指代的唯一订单可成为 order_context。模型自行猜到一个属于用户的订单号，不足以建立跨轮焦点或退款按钮。
- `suggest_options(options, ticket_type=None)` 的标签扩展为「转人工」「建工单」「申请退款」，仍限 1~2 个不重复标签。只有包含「建工单」时 ticket_type 才必填且允许出现，旧组合保持兼容。
- 模型侧不增加 order_id 参数。包含「申请退款」时，处理函数要求本轮已有通过归属校验的 order_context，并从中绑定订单号，生成 `{"action": "refund_form", "label": "申请退款", "order_id": ...}`。缺上下文、重复标签、非法 ticket_type 或模型擅传订单号均返回错误 ToolMessage，整组不产生新建议。
- 退款建议仅代表打开申请表，不代表审批通过或已经退款；用户提交后写的是待处理工单。create_ticket/create_refund 始终不进入 Agent 工具注册表，伪造调用只返回 unknown_tool。

### 6.7 log（小改）

- 挂起轮不到达 log；完成轮按现契约单次提交。提交成功后才能发 citations、suggest_actions，以及把本轮已确认订单更新为同 thread 的 active_order（生命周期见 §10）。失败或取消而未完成提交的轮次不建立新的订单焦点。
- `_should_cite` 不再限定 route 必须为 knowledge：本轮 retrieval_status=ok、知识 evidence 非空、最终答复实际引用其中有效 ref_no，且不是固定拒答/预算兜底时才发 citations。business 的工具检索与 refund 的图检索适用同一规则；只发完整知识 evidence 列表，保持编号与顺序，订单事实不作为知识引用发送。
- suggest_actions 透传已经校验、绑定订单的 refund_form，使用本轮提交返回的 source_message_id。退款低置信/不可用分支仍沿用现有入池区分，不因新增订单块产生伪知识引用。

## §7 Prompt 契约

惯例沿用：字符串常量、`.replace()` 渲染（字面 JSON 花括号）、正则提 JSON + 校验 + 结构化降级。

1. `UNDERSTAND_PROMPT`(app/prompts/understand.py)：输入最近 N 个完整轮历史 + 已校验 active_order 摘要 + 当前问句；输出单行 JSON `{"resolved_query": "..."}`。已完整/指代明确时原样透传；有明确指代时补全为独立问句；订单歧义保持未消解，不虚构订单号。
2. `INTENT_PROMPT`（重写）：① 八类枚举选择题（七类定义 + 无法确定主诉求时的「其他」）；② 强制 JSON 只有 `{intent, confidence}`；③ 边界 few-shot（纯投诉 vs 带业务诉求、退款 vs 售后、闲聊 vs 其他、库存 vs 规格）；④ 删除 needs_knowledge 全部规则，按 §5 确定混合问题主意图，保留完整问题。
3. `REFUND_SCOPE_PROMPT`(app/prompts/refund.py)：只输出 `{"mode": "general|order_specific|clarify"}`，说明 §6.3 的定义和边界，包含「如何申请退款」与「我要申请退款」的成对样例；不追问用户、不生成订单号、不改变 intent。
4. `EXPAND_PROMPT`(app/prompts/expand.py)：仅具体订单子流程；输入 resolved_query + 已校验订单摘要；输出 `{"queries": [...]}` 2~4 条、侧重点不同。扩写围绕当前诉求：退款可覆盖资格/时效/运费/流程，维修可覆盖保修/寄修/费用，不把维修问题统一扩写成退款。
5. `SERVICE_SYSTEM_PROMPT`：区分纯业务事实与政策/流程/规格结论；business 缺少当前证据时先调用 query_faq，检索前只可给简短过程说明，refund 使用图注入的证据。仅有适用证据才回答知识结论并标引用；仅澄清分支只提出问题。订单事实与知识证据分别渲染，申请退款只是打开人工审核表单。保留 REFUSAL_ANSWER 逐字契约和不承诺个案退款到账日等约束。

解析函数：新增 `parse_understand_output` / `parse_refund_scope_output` / `parse_expand_output`；`parse_intent_output` 改为 `(text) -> (intent, confidence)`。各自校验对象类型及所需字段，降级分支按节点契约，风格对齐现有 parse_plan。test_prompts.py 更新八类、分类两字段、执行模式三枚举与 `.replace` 渲染契约。

## §8 SSE 协议增量

新帧（四层同改：events.py → chat_service 驱动/事件类型 → routers 序列化 → chat.html 分发）：

```json
{"type": "order_selector", "interrupt_id": "当前挂起实例的ID", "orders": [{"order_id": "当前用户的订单号", "product": "保温杯", "amount": 89.0, "status": "已完成", "created_at": "..."}]}
```

order_selector 由驱动层在挂起后通过事件工厂构造，包含 interrupt_id 的类型需贯穿服务事件、路由序列化与前端。它不依赖节点在 interrupt 前发送 custom 事件；其余 custom 事件仍沿原翻译链路。

帧序：挂起轮 `session → order_selector → [DONE]`；resume 轮 `session → delta/tool_* → citations? → suggest_actions? → [DONE]`。成功恢复不重发订单卡。suggest_actions 的 options 元素新增 `{"action": "refund_form", "label": "申请退款", "order_id": "..."}`。

messages 流新增模型调用级白名单：仅 main_agent 的对用户输出模型调用带 `chat_visible` tag，驱动同时核验 node=main_agent、该 tag 存在、消息为 AIMessageChunk 三个条件。tag 不加在整个节点、图或共享 deps.model 上，避免 query_faq 内部 plan_query 调用继承可见标记。内部改写即使发生在 main_agent 节点内，也不外发 JSON/token。该方式依据 [LangGraph 按模型调用 tags 过滤流的官方文档](https://docs.langchain.com/oss/python/langgraph/streaming)，需真实回调假模型的集成测试验证。

## §9 端点

### 9.1 POST /v1/chat/resume

请求 `{user_id, session_id, interrupt_id, order_id}`（ChatResumeRequest）；ID 字段必须是非空字符串，user/session 的格式约束沿用现有请求。

顺序固定：会话归属检查 → 获取与 `/stream` 相同的会话锁 → `graph.aget_state` → 核验恰有一个 refund_prepare/order_selector pending 且 ID 等于请求 interrupt_id → `get_order(user_id, order_id)` → 核验订单仍在该卡的候选列表 → 创建 SSE 响应并驱动：

```python
graph.astream(
    Command(resume={interrupt_id: {"order_id": order_id}}),
    config,  # thread_id=session_id，user_id 来自已校验的请求身份
    stream_mode=["messages", "custom"],
)
```

- 会话归属失败、订单不存在或属于他人：404，统一响应不泄露详情。
- 无 pending、ID 不匹配、pending 节点/类型不符、订单不在卡片候选内：409。旧卡、已恢复卡、重复点击，以及新轮再次挂起时的旧卡均适用。
- 校验在返回 StreamingResponse 前完成，因此错误是真正的 HTTP 404/409；pending 校验与图恢复之间保持持锁。校验异常、等待取消、流未迭代即关闭、流中断及正常完成都必须幂等释锁。
- 并发 resume 中只能一个消费该挂起实例；第二个取得锁后重新读状态并返回 409，不能复用排队前的 snapshot。新消息与 resume 按取得同一锁的顺序串行执行。

### 9.2 POST /v1/chat/action

以 action 区分请求契约，共用 user_id/session_id/source_message_id：

- `create_ticket`：继续必填原 ticket_type，原请求无需提供任何退款字段。
- `create_refund`：必填 order_id 和 `refund_reason: Literal["质量问题","商品破损","发错货","未收到货","拍错或多拍","七天无理由"]`；ticket_type 由服务端固定为「售后」，客户端可省略，显式提供时只允许「售后」。字段缺失、原因非法、相互冲突的参数返回 422。

退款建单依次校验会话归属、source_message_id 是本会话的用户消息、`get_order(user_id, order_id)` 成功；任何归属/不存在问题返回相同的 404。订单号不因来自前端回填就获得信任，也不使用当前 active_order 覆盖用户提交的订单。此端点允许对属于用户的订单提交人工审核申请，不把 Agent 的建议作为退款资格证明。

沿用现有同事务查询消息与写 ticket、取消时等待写事务完成的契约。写 tickets：ticket_type=「售后」，description=`退款单:订单 {order_id},原因:{refund_reason}`；不改 conv.status，返回 `{ticket_no, status: "待处理"}`。用户原始消息的归属验证继续保留；不新增订单表或退款审批副作用。

## §10 State 字段生命周期

本轮临时字段（全部进 new_turn_state 重置 + test_graph_state 钉）：

| 字段 | 类型 | 写入者 | 说明 |
|---|---|---|---|
| `intent_confidence` | float \| None | classify_intent | 只记录不路由 |
| `refund_mode` | Literal["general", "order_specific", "clarify"] \| None | refund_scope | 每轮重置；只影响 refund 执行分支，不进入 intent JSON |
| `order_context` | dict \| None | refund_prepare / query_order、query_logistics 的处理器 | 当前轮已校验订单快照，独立于 evidence；多订单不自动确立唯一焦点 |
| `expanded_queries` | list[str] | refund_policy | 扩写结果（含单条兜底） |
| `understanding_degraded` | bool | understand_query | 透传降级标记 |

既有 `resolved_query` 语义不变（现在是真消解结果）。retrieval_* / evidence / low_conf_* 仍逐轮重置，写入者为 refund_policy 或 business 内的 query_faq 处理逻辑；缓存只在当前 main_agent 调用内存活。移除 needs_knowledge 的 State 字段与日志引用，不能留下失效维度。

`messages` 契约不变：log 提交成功后才追加。挂起轮 state 冻结在 checkpoint；resume 重跑 refund_prepare 时 resolved_query/intent/refund_mode 等上游字段保持，不重置——**new_turn_state 只在全新轮入口调用，resume 路径传 Command 不传 dict**。

跨轮字段新增 `active_order: {order_id: str, source_message_id: str} | None`，只保存同一 thread 已完成轮中用户明确指定或点选、且归属校验通过的唯一订单。log 提交成功后更新，new_turn_state 不重置；不依赖助手是否复述订单号，也不依赖前端 hint 进入 messages。

每轮使用 active_order 前重新通过用户范围查询获取快照；失效时清空，不复用旧状态/时间。它只辅助明确指代，不覆盖用户本轮的显式订单号，不把未确认候选或模型猜测变成焦点。明确切换到新订单且本轮完成后替换焦点；新会话不继承，SQLite 关闭重开仍可在原 thread 恢复。退款按钮固定绑定生成时的订单，之后焦点变化不改变旧按钮订单。

## §11 mock 订单服务

`app/services/orders.py`：固定订单模板按 user_id 确定性实例化，公开订单号带用户命名空间，不同用户的订单号不重合。模板可复用，订单归属不能共享；不采用「所有用户都能查同一组订单」的替代方案。

API：`list_orders(user_id) -> list[OrderInfo]`、`get_order(user_id, order_id) -> OrderInfo | None`。不存在与非本人订单均返回 None。OrderInfo 包含 order_id/product/amount/status/created_at/delivered_at/returnable_note；以可注入的统一演示时钟生成日期，当前轮快照带查询时刻，测试冻结时钟。覆盖 7 天内普通商品、超过无理由期限、定制或生鲜、在途；「超期/特殊类目」仅表示无理由退货受限，质量问题仍以政策为准。

query_order 逐调用绑定经会话校验的 user_id，模型仍只传 order_id；返回 JSON 保留原有字段。query_logistics 同样先通过用户范围查询再产生物流演示结果，避免另一个只读入口绕过订单归属；物流状态须与订单签收状态一致。子流程直通、resume、Agent 查询和 create_refund 全部调用同一服务。测试使用至少两个用户、各自不同订单，同时覆盖正向与交叉访问。

## §12 前端改动（Vibe 例外）

界面走 Vibe 例外，交互契约仍须验收：订单卡以 DOM 构建（禁 innerHTML），显示订单号/商品/金额/状态/时间。卡片捕获生成时的 user_id、session_id、interrupt_id；点选后携带这三个值与 order_id 调 resume，不能从全局当前会话猜身份。提交中禁用按钮；成功后消费卡片，409 显示「该选择已失效」，新消息发出时使旧卡失效。前端禁用只是体验措施，服务端仍执行全部并发与归属校验。

恢复输出追加到原挂起轮的助手区域，不额外制造一条用户消息；「已选择订单 ×××」是本地提示，跨轮记忆由 active_order 保证。「申请退款」打开订单号只读回填、原因固定下拉的表单；提交捕获的 session_id/source_message_id/order_id，成功仅提示待处理工单号，不展示审批通过。断言采用 test_chat_page needle + 浏览器人工点验。

## §13 测试策略

新增与扩展（pytest，fake model；不调用付费模型）：

- test_graph_nodes：understand_query 消解/完整句透传/降级/首轮跳过，按完整轮取历史与订单号来源校验；classify_intent 八类两字段；refund_scope 三模式、解析/模型失败转澄清、已有 active_order 不把通用问句变成个案；refund_prepare 唯一订单直通、多候选/无号挂起、空订单列表、跨用户订单；other_fallback 与 parse_* 参数化。
- test_graph_state：INTENTS/ROUTE_TABLE 八行，商品咨询走 business，needs_knowledge 移除；refund_mode 等临时字段重置，active_order 同 thread 保留、新 thread 隔离、显式换单优先、失效订单清空、未完成轮不更新焦点。
- test_refund_flow（ScriptedChatModel + InMemorySaver/SQLite）：缺号挂起只发 session/order_selector/[DONE] 且账本无半轮；按 ID resume 只提交一轮且不重发卡；挂起后新消息完成及再次挂起两种情形，旧 ID 都返回 409；关闭重开 SQLite 文件后同 ID 可恢复。
- test_chat_api / test_chat_service：两个并发 resume 仅一个成功，后者在锁内重新校验并返回 409；与新消息竞争按锁顺序执行；错误在 SSE 响应建立前返回；404/409、等待取消、校验异常、从未消费的流和断流均验证释锁。
- test_orders / test_tools：至少两个用户，直通、resume、query_order、query_logistics、create_refund 的交叉访问均失败；模型不能覆盖绑定 user_id；日期时钟冻结且订单/物流状态一致。
- test_chat_action：旧 create_ticket 原请求保持兼容；create_refund 固定原因、所需字段、ticket_type 冲突；用户/会话/消息/订单任一归属错误均 404；正确订单和原因写入待处理 ticket，不改变 conv.status；取消写事务仍保持原完成性。
- test_agent_node：按 route/refund_mode 检查真实绑定列表与执行注册表；仅 business 提供 query_faq，clarify 无工具。query_faq 无参且使用完整 resolved_query、同轮重复调用复用结果/编号、下一轮重新检索；低置信/不可用补齐整组 ToolMessage 后固定收尾，不再调用模型。新 suggest_options 标签、长度/重复/ticket_type 校验，服务端绑定订单，缺订单或伪造 order_id 不出按钮；伪造 create_ticket/create_refund 均不写库；上下文各块纳入预算。
- test_graph_nodes / test_retriever：混合 hybrid_rerank/hybrid 候选只按 ID 合并后统一重排，原分数不用于跨策略比较；统一重排失败使用 base_query 完整结果；部分扩写异常、维护状态、总超时、关闭扩写、非 rerank 策略、全空候选均有确定的闸门结果。
- test_graph_nodes / test_chat_service：business 工具检索与 refund 图检索在高置信且引用有效 ref_no 时都于提交后发 citations；订单块不进入 citations/evidence_refs；低置信、不可用、预算兜底不发引用；两种检索入口的低置信记录都按原语义入池。
- test_router_coverage：用 §5 表中问题验证完整链路，包含知识库不可用时纯库存仍可答、取消规则先检索、通用寄修/退款步骤不进入 refund_prepare 且不调用扩写、个案缺号才出选择器、澄清分支无工具/无订单卡。通用问题在无历史焦点时零订单服务调用；已有焦点可做有效性复核，但仍不得要求选单。fake retriever 返回不同 scope 的证据，确认售后手册与 FAQ 未被过滤掉。
- test_chat_service：用真实 LangChain 回调的 ScriptedChatModel 同时产生 main_agent 可见输出与 query_faq 内部改写输出，断言 SSE 只外发带 chat_visible 的 AIMessageChunk；分类、模式判断、扩写及内部改写的 JSON 全部不可见。
- test_chat_page：卡片捕获 session/interrupt ID、提交中禁用、409 失效提示、原轮恢复、退款表单及固定订单的 needle；浏览器人工点验按 §12。

改造：conftest 两个 fake model 的罐头 ainvoke 按 understand/classify/refund_scope/expand prompt 分派，流式假模型适配调用级 config/tags；test_prompts 更新渲染与契约钉；test_ch05_acceptance 更新拓扑和只读工具白名单断言，保持写工具不可达、提交/取消/预算护栏及原业务样例。实现完成时同步 AGENTS.md 的章节状态与只读工具名单；本次 spec 修订不提前宣称新运行行为已上线。

evals（真模型手跑，不进 pytest):
- intent_samples.jsonl 扩充至八类+边界+多轮（带 history）；保留 §5 的库存/规格/取消/维修问题及含投诉情绪的业务问句，probe_intent.py 适配两字段。
- probe_router_multi.py：多轮剧本（物流→退款→物流；「这个能退吗」），逐轮打印消解/意图/路由/refund_mode/工具调用/active_order。必须另含「原问句无订单号→点选→助手答复不复述订单号→这单到哪了」，以及「如何申请退款→我要申请退款」的模式切换；通用知识问题单独统计是否先调用 query_faq，避免仅分类命中却实际无证据回答。

验收映射：

| 验收 | 验证方式 |
|---|---|
| 1 多轮意图+指代全对 | evals/probe_router_multi.py 真模型跑剧本 + 多轮样例入 intent_samples |
| 2 JSON 稳定可解析、怪问题落「其他」 | probe_intent 边界样例 + pytest parse 参数化 + classify 兜底钉 |
| 3 「这个能退吗」补全指代→子流程拿订单和政策 | 集成测试断言帧序与 evidence 内容 + probe 逐轮输出人工核 |
| 4 浏览器缺号弹订单卡、点选走完 | 人工点验 + resume 端点集成测试 + chat.html needle |
| 5 旧卡/重复点击/并发请求不误恢复 | 新轮再次挂起 + 并发 resume + 锁释放集成测试 |
| 6 订单隔离与点选后的跨轮记忆 | 两用户交叉访问 + 不复述订单号的后续指代 + SQLite 文件恢复 |
| 7 多查询降级不混用分数 | 混合 effective_strategy、统一重排失败、超时和部分失败的假检索测试 |
| 8 退款建议与引用完整 | 伪工具参数/订单绑定 + 提交后的 citations/suggest_actions 帧序 + action 写库 |
| 9 原业务覆盖保留，纯库存不依赖知识库 | §5 样例的集成测试 + 真模型 query_faq 触发率记录 |
| 10 通用售后不选单，具体订单才选单 | refund_scope 三模式与降级测试 + 寄修/退款步骤的跨 scope 证据测试 |

实现交付门槛：全量 pytest 绿（Docker 在线）；真模型 probe 跑一轮，§5 核心样例及订单切换剧本逐例核验，分类/执行模式/知识工具触发与最终答复结果分别记 dev-notes。失败样例应修复并重跑，不能只记录总体命中率就视为通过。

## §14 依赖与配置

无新增包。新增配置项：`intent_model_name`（默认 None=复用主模型）、`understand_history_turns`（默认 6 个完整轮）、`refund_expand_enabled`（默认 True；具体生效策略与降级规则见 §6.4）。LangGraph interrupt/Command 用法以 Context7 拉取的 1.x 官方文档和本地 1.2.11 探针为依据（§2）；升级后重跑语义测试。实现阶段改 prompts 后跑全量 pytest（AGENTS.md 红线）。

## §15 风险

- 挂起态语义依赖运行库版本：现已在 1.2.11 验证新 dict 从 START 起跑，但应用仍必须自己绑定 interrupt ID、在锁内验证，保留 InMemorySaver 与 SQLite 两套回归。
- refund_prepare 重跑：订单服务只读，节点不发卡；驱动层只在当前图调用结束且仍 pending 时发一次带 ID 的选择器。这样不依赖 interrupt 内部未公开的 resume 标记。
- 会话锁与挂起：整个 pending 校验/恢复持锁，挂起流和所有异常出口均释锁；长时间等待用户选择不占锁。前端禁用按钮不能替代后端校验。
- 检索成本：扩写只在指定子流程和 hybrid_rerank 下生效，最多 4 路查询及一次最终重排；共享总超时。统一重排的结果分布可能变化，真模型 probe 应同时记录拒答与答复正确性，不把沿用阈值等同于已重新校准。
- 通用分支的知识工具调用由 Agent 决定，存在漏调用风险；用系统提示、完整问题输入、返回前置信处理和真模型场景验收约束。退款/售后的已识别知识诉求仍由图保证检索；不能把两条路径都描述为图强制检索。
- refund_scope 只判断订单数据是否必要，不负责需求追问；不明确时交给主力 Agent 澄清。每次新轮重新判断，已有订单焦点不能让通用问题强制选单。
- messages 护栏：新增 query_faq 后，仅按 main_agent 节点名过滤已不足以隔离内部模型输出；按 §8 同时过滤可见调用 tag 与 AIMessageChunk，固定回复和工具帧仍走原 custom 通道。
- 演示数据自洽：mock 订单状态/时间与 returns-policy(7 天无理由）要对得上，避免演示时「能退/不能退」讲反。
