# ch07 · 会话上下文管理(三层历史 + 异步摘要)设计

日期:2026-09-22 · 状态:已评审(brainstorm 四节逐节确认)

## §1 定位与范围

把 ch01 沿用至今的「简单裁剪」(`fit_tool_context` 丢最旧轮 / `_history_turns` 按轮数截)升级为正经的会话上下文管理:三层历史(原文层 / 截短层 / 摘要层)+ 双锚点降级 + 后台异步分段摘要 + token 预算从模型窗口倒推 + 上下文全量可观测 + 前端多会话侧栏。

**只管当前会话。不跨会话、不存用户画像。** 语义检索捞历史、主题重要度留关键事实两层不实现;跨会话长期记忆不实现。

## §2 术语与三层定义

历史按离当前轮的远近分三层,边界由两个消息 id 锚点表达(存 `conversations` 行,**降级只挪 id、不搬数据**):

- `id ≤ summary_upto_msg_id`:已进摘要,不再进滑窗
- `summary_upto_msg_id < id ≤ layer1_from_msg_id`:**层 2**,渲染成截短形态
- `id > layer1_from_msg_id`:**层 1**,原样进上下文
- NULL 初态:`summary_upto=NULL` 无摘要;`layer1_from=NULL` 全部层 1

**数据流(已定)**:完整历史(含 ToolMessage)的唯一读源是 LangGraph checkpoint 里的 `state["messages"]`;锚点与摘要投影的权威存储是 MySQL `conversations` 行;分层组装每轮点查一次锚点,按 id 切 state 里的消息。log 节点 commit 时把行 id 盖进消息的 `additional_kwargs["db_id"]` 再交给 `add_messages` reducer。checkpoint 丢失 → 锚点悬空 → 退化为「摘要投影 + 空历史」,不回填(沿用 ch05 契约,且比现状强:梗概仍在)。

## §3 DDL 与 ORM

- `sql/ch07-ddl.sql`(已备好,本章冻结):`conversations` 加 `summary TEXT`(最近几段梗概拼成的投影)/ `summary_upto_msg_id` / `layer1_from_msg_id` 三列;新建 `conversation_summaries(id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at)`,`UNIQUE(conversation_id, seq)`,**一段一行只追加,不回炉重压**(同一事实只经历一次有损压缩)。
- 头部注释修正:该文件实际两步合一,首段注释里「另一个锚点和分段摘要表在 ch07-layers.sql」是过时表述,实现时改写为单文件说明(`sql/` 与 `db/init/` 两侧同步改,保持逐字节一致)。
- 接线三处:`db/init/05-ddl.sql`(与 `sql/ch07-ddl.sql` 逐字节一致);`tests/test_ddl_sync.py` 的 `DDL_PAIRS` 加 `("05", "ch07")`;`tests/dbfixtures.py` 的 `DDL_PATHS` 加 `05-ddl.sql`、`TABLES` 在 `conversations` 前插 `conversation_summaries`。
- `app/models.py`:`Conversation` 加三列镜像;新增 `ConversationSummary` ORM。
- 老库(非 docker 首启)需手工 `mysql wayhelp < sql/ch07-ddl.sql`,README 运行节补一句。

## §4 token 估算口径(唯一一把尺)

新 `app/services/token_budget.py`:

```
estimate_tokens(text)  = CJK字符数×1 + ceil(其余字符数/4)     # CJK 区间:一-鿿、㐀-䶿、　-〿、-￯
estimate_message(msg)  = estimate_tokens(content) + 4         # 每条 +4 结构开销;AIMessage 另加 tool_calls JSON 估算
estimate_messages(msgs) = Σ estimate_message
```

对 deepseek 是轻微高估(安全方向)。**预算、自检、截断、触发全用这一把尺**;聊天链路里的 `count_tokens_approximately` 全部替换(它对中文按 ÷4 估算,低估约 4 倍,与新预算混用净效果是反的)。extract/mining 等离线链路不在预算关键路径,保留旧尺(§13 列明)。

## §5 预算公式与配置

从模型窗口倒推,无写死常量:

```
PEAK   = MAX_AGENT_STEPS × (TOOL_RESULT_MAX_TOKENS + 50)      # 本轮 ReAct 瞬时峰值:每步一份工具结果+调用开销
FIXED  = SYS_TOKENS + RERANK_TOP_K×300 + 650 + SAFETY_MARGIN_TOKENS
         # SYS_TOKENS = 启动时实测(系统提示+最大工具集 schema,用 §4 尺)
         # RERANK_TOP_K×300 = 证据预留;650 = 梗概投影预留(最近 3 段 × 约 217)
AVAIL  = MODEL_CONTEXT_WINDOW − MAX_OUTPUT_TOKENS − MAX_USER_INPUT_TOKENS − PEAK − FIXED
BUDGET = min(HISTORY_TARGET_TURNS × STEADY_TOKENS_PER_TURN, AVAIL)   # 「想留住的」与「匀得出来的」取小
层1预算 = floor(BUDGET × 0.7);层2预算 = BUDGET − 层1预算
```

**验收配置复算(写测试钉死)**:`MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5`,SYS_TOKENS 实测 1901(`SERVICE_SYSTEM_PROMPT` 1092 + 5 工具 schema 809,langchain_core 1.6.2 实测):FIXED = 1901+1500+650+550 = 4601;AVAIL = 18000−2000−2000−3750−4601 = **5649(≈滑窗 5650)**;层1 = **3954**;层2 = **1695**。`SAFETY_MARGIN_TOKENS=550` 是配平项,冻结为默认值。默认窗口(65536)下 AVAIL ≈ 4.6 万 > 20×300,BUDGET 取 6000,20 轮不触发任何降级/摘要(验收 3)。

**启动自检**(lifespan):算一遍 BUDGET 并 `logger.info` 打印全部分量(可观测);若 `BUDGET < STEADY_TOKENS_PER_TURN`(一轮都装不下)→ `logger.critical("上下文预算不足 …")` 报警,**不阻断启动**。替代 `main.py:121` 的 system-prompt 探针(原 raise 行为退役);`main.py:126` 的 extract 探针保留。

配置变更:

| 配置 | 默认 | 变化 |
|---|---|---|
| `MODEL_CONTEXT_WINDOW` | 65536 | 新增 |
| `MAX_USER_INPUT_TOKENS` | 2000 | 新增;用户输入校验换 §4 尺、只量输入本身(不含 system) |
| `TOOL_RESULT_MAX_TOKENS` | 1200 | 新增;工具结果截断从字符换成 token 口径(ToolExecutor) |
| `RERANK_TOP_K` | 10 | `rerank_top_n` 改名(语义不变,默认值不变;retriever 截断、evidence `max_items`、候选上限 `4×` 全部随之改名);进 FIXED 证据预留 |
| `HISTORY_TARGET_TURNS` | 20 | 新增 |
| `STEADY_TOKENS_PER_TURN` | 300 | 新增 |
| `SUMMARY_MAX_CHARS` | 200 | 新增,单段梗概上限 |
| `SAFETY_MARGIN_TOKENS` | 550 | 新增,配平冻结值 |
| `max_input_tokens` | — | 聊天链路停用(prepare 早闸移交 `MAX_USER_INPUT_TOKENS`;agent 预算移交公式);extract/mining 保留同名同义 |
| `max_tool_result_chars` | — | envelope 落库职责消亡;保留字段,职责收窄为证据组装字符预算(`assemble_evidence` 的 `budget_chars`) |
| `understand_history_turns` | — | 退役(分层滑窗接管,ch06 M1 backlog 在此闭环) |

## §6 SessionStore 协议扩展

`CommitTurnResult` 加 `message_ids: list[str]`(本轮各行 id,与入库顺序对齐;`source_message_id` 语义不变)。

协议新增(全部 async;`DbSessionStore` SQL 实现,`InMemorySessionStore` 内存实现供图测试):

```
get_context_meta(session_id) -> ContextMeta(summary: str|None, summary_upto: int|None, layer1_from: int|None)
list_conversations(user_id) -> list[ConversationItem]   # 新在前(updated_at DESC, id DESC);preview=首条 user 消息前 40 字;summarized=summary IS NOT NULL
list_messages(session_id, user_id) -> list[StoredMessage] | None   # 归属不符 None(404 不泄露)
move_layer1_from(session_id, new_id) -> bool            # 只许前移(new > 旧);并发兜底
fetch_span_texts(session_id, after_id, upto_id) -> list[tuple[int, str, str]]  # (id, role, content),user/assistant only
append_summary(session_id, from_id, upto_id, content) -> int       # INSERT seq=max+1;同事务更新 summary 投影(最近 3 段拼接)与 summary_upto=upto_id;返回 seq
```

`commit_turn` 行为变更:**`_to_stored` 不再产出 tool 行**(tool 结果不落消息表);`validate_turn` 同步改:中间 assistant 仍须带 tool_calls,但不再要求 tool 行一一配对,出现 `role=="tool"` 直接拒(防御)。InMemory 实现:每条消息分配递增 id(store 级计数器),锚点/摘要用 per-session dict 支持,使图集成测试不需要 DB。

## §7 三层组装器 `app/services/context_layers.py`

纯函数核心(DB 读由调用方注入):

```
build_layered_view(messages, meta: ContextMeta, settings) -> LayeredView
LayeredView: layer2_rendered: list[BaseMessage]  # 截短形态,保留原 role
             layer1: list[BaseMessage]           # 原文
             summary: str | None                 # 投影全文
             l1_tokens/l2_tokens/summary_tokens: int   # §4 尺估算
             span_meta: (l1_from, l2_from 等观测字段)
```

渲染规则:层 2 内用户原话不动;assistant 答复只留开头 `LAYER2_HEAD_CHARS=50` 字 + 「…」(命名常量,进 `token_budget.py`);ToolMessage 大块结果换成一行标识 `[工具结果已省略: <name>]`;tool 组配对不断(截短不改结构,`_tool_groups_complete` 语义可复用)。id 归属:顺序扫描,无 `db_id` 的消息(理论上只有异常路径残留)继承最近已盖章 id 的区间。understand/classify 的滑窗文本渲染(「用户:/助手:」行)里,ToolMessage 无论层级一律渲染为同一行标识,不展开。

**降级**(log 节点 commit 后,同步、纳秒级):对「db_id > layer1_from」的原文段(含本轮)做 `trim_messages(strategy="last", max_tokens=层1预算, token_counter=§4尺, start_on="human", include_system=False, allow_partial=False)`;有丢弃 → `layer1_from` 前移到被保留首条消息的前一条 id(整轮为界,tool 组不跨层劈开),`move_layer1_from` 落库,日志 `层1 降级`。组装本身永远是纯读:锚点滞后一轮也安全,trim 兜底保证本次调用不超预算。边界情形:单轮就超层 1 预算 → 层 1 可为空,层 2 兜住上一轮,下轮摘要接住,系统不崩。

## §8 图改造

- `ChatGraphState` 加 `history_block: str`(临时字段,`new_turn_state` 重置为 "")。
- `understand_query`:`meta = await store.get_context_meta(thread_id)` → `build_layered_view` → 渲染「摘要行 + 分层滑窗(层2截短+层1原文,用户:/助手: 行)」写 `state["history_block"]`;**每轮打 `history_ctx` 日志**(含无历史轮);短路条件改为「滑窗空 且 无摘要 且 无 active_summary」才透传不调模型。`_history_turns`/`_render_history` 删除。
- `classify_intent`:`INTENT_PROMPT` 加 `{history_block}` 占位,注入同一份共享渲染(「指代消解/意图识别共用的历史」)。prompt 契约变化,`probe_intent` 需人工重跑校准(计划里列为手动步骤,烧额度不进 pytest)。
- `main_agent`(business / refund 过闸 / clarify):拼装顺序定死——
  ```
  [0] SystemMessage(人设+红线,逐字不变;工具定义走 bind_tools,每轮一致)
  […] 层2 截短渲染 + 层1 原文(时间序、原 role)
  [k] HumanMessage 当前用户句
  [k+1] HumanMessage「【对话背景】梗概投影 + 【检索证据/订单事实】」合成一条(都为空则不出;永不占 system)
  [k+2…] 本轮 ReAct 步消息(AI/Tool)
  ```
  每步调模型前打 `model_ctx`。meta 在节点入口读一次,整步复用。总守卫:估算总量 > `MODEL_CONTEXT_WINDOW − MAX_OUTPUT_TOKENS` → 返回 None → 现有 `tool_context_too_long` 兜底帧。`build_agent_context` 重写;`fit_tool_context`/`build_messages_with_tools`/`_drop_broken_head` 退役。
- `log` 节点:`commit_turn`(无 tool 行)→ 拿回 `message_ids` → **盖 `db_id` 进 turn_messages 副本再交 reducer** → 评估降级(§7)→ 层 2 渲染态估算 > 层 2 预算 → `ensure_future` 起摘要任务(§9),不阻塞 SSE 收尾。
- interrupt 挂起轮不 commit、不进 messages,天然不碰锚点。

## §9 摘要任务 `app/services/summarizer.py`

- **触发**:log 节点 commit 后,层 2 渲染态 token 估算 > 层 2 预算(触发看用量不数条数:tool 结果不落消息表,条数看不见它涨)。
- **批次**:当前整个层 2 区间 `(summary_upto, layer1_from]` 压成一段;完成即 `summary_upto` 追到层 1 起点,层 2 清空重攒。
- **数据源**:`fetch_span_texts` 读 messages 表该区间 user/assistant 原文。
- **prompt 契约**(新 `app/prompts/summary.py`):只提炼事实与诉求(问过哪款商品、报过的订单号/手机号、明确诉求、还没解决的问题);对话里没出现的一个字不许编;寒暄闲聊不留;≤ `SUMMARY_MAX_CHARS`;旧投影只作背景传入帮助连贯,输出只覆盖新区间,**不合并不重写旧段**。输出超限直接截断 + 日志。
- **异步**:`ensure_future` + `shield`(同 chat_service 写事务模式);模块级 in-flight 集合按 conversation_id 防重入(单 worker 内成立);skip:in-flight / 区间为空;失败 `logger.exception`,锚点不动,下轮自然重试。任务与下一轮并发安全:批次边界在任务开始时点查捕获,`append_summary` 只前移不回退。
- **prompt 验证例外**(纯 Prompt 产出):`evals/` 加标注样例 jsonl(对话 + 必含事实清单 + 禁含闲聊清单)与 `probe_summary.py`,真模型跑一遍人工核对;不进 pytest。

## §10 日志契约(`log/app.log`)

`main.py` 加 FileHandler(`log/app.log`,utf-8,追加;目录自动创建;`log/` 入 `.gitignore`),保留 stderr。单行 JSON,字面钉如下(grep 友好,验收 4 直接看):

- `history_ctx {"session":…,"summary":…,"window":[逐条],"est":…}` — understand 渲染,每轮必打(闲聊/投诉兜底轮也有)。
- `model_ctx {"session":…,"step":…,"budget":{"l1":…,"l2":…},"counts":{"l1":…,"l2":…},"est":{"l1":…,"l2":…,"summary":…,"total":…},"summary":"投影全文","layer2":[渲染逐条],"layer1":[原文逐条]}` — 主力 Agent 每步调模型前,原样不截。
- `层1 降级 session=… <旧>→<新>`(旧为 None 显示 `-`)。
- `summary trigger session=… 层2 约 N token > 预算 M` / `summary start session=… 区间 (a, b]` / `summary done session=… 第N段 覆盖 (a, b] 耗时 Xms` / `summary skip session=… reason=…` / `summary failed session=…`(exception 堆栈)。

## §11 只读接口(新 `app/routers/conversations.py`,`/api` 前缀)

- `GET /api/conversations?user_id=…` → `[{id, status, created_at, updated_at, preview, summarized}]`,新在前。
- `GET /api/conversations/{id}/messages?user_id=…` → `[{id, role, content, created_at}]` 按 id 升序;归属不符 404(复用现有 AppError→HTTP 映射,不泄露)。
- 响应模型进 `app/schemas.py`;无写动作,不改 SSE 协议,不加新帧类型。

## §12 前端侧栏(`app/static/chat.html`,Vibe Coding 例外)

- 左栏 `#sidebar` + `#conv-list`:加载 `GET /api/conversations`,渲染首问预览 + 时间 + 「已摘要」标记;**加载失败 catch 隐藏侧栏,聊天不受影响**(静默降级)。
- 点击 `.conv-item`:`GET …/messages` → 清聊天区回放 user/assistant 纯文本气泡(tool 徽章/citations/suggest 按钮不回放——数据已不存,定取舍)→ 切 sessionId 接着聊(checkpoint 在,上下文不断);活跃项高亮。
- 「+ 新对话」:sessionId 置 null 清屏,旧会话留侧栏;首轮 session 帧回来把新会话插到列表顶部。
- `tests/test_chat_page.py` 字符串断言钉:sidebar DOM id、两个 fetch URL、catch 降级路径。

## §13 退役与行为变更清单(存量改写)

1. `_to_stored` 不产 tool 行;`validate_turn` 改 §6;`test_store_db`/`test_graph_nodes` 的 envelope 断言改写;`tool_envelope.wrap` 失去聊天链路调用方(模块保留,mining 等不涉)。
2. `fit_tool_context`/`build_messages_with_tools`/`_drop_broken_head`/`_validate` 退役;`test_tool_chat_chain.py` 按新语义重写。
3. prepare 早闸:`check_input_budget(system, input, max_input_tokens)` → `check_user_input(input, MAX_USER_INPUT_TOKENS)`(§4 尺、只量输入);`max_message_chars` 保留为输出/落库字符闸。
4. `understand_history_turns` 退役(M1 闭环)。
5. `main.py:121` system-prompt 探针 → §5 启动自检(critical 不 raise);extract 探针保留。
6. `rerank_top_n` → `rerank_top_k` 改名(默认 10 不变);波及 retriever/nodes/business/test_config/evals 脚本,逐一改。
7. `INTENT_PROMPT` 加 `{history_block}`(probe_intent 人工重跑)。
8. AGENTS.md(落库口径、新配置、log/app.log、当前状态)与 README(运行节:老库手工 apply ch07 DDL)收尾更新。

## §14 测试策略

- TDD 先行:`token_budget`(口径/公式/5649·3954·1695 复算钉/自检阈值)、`context_layers`(三层切分/截短渲染/tool 标识/NULL 与悬空锚点/单轮超预算)、`summarizer`(触发/skip/追加/锚点前移/投影三段/失败不挪锚点,conftest 假模型)、store 两实现(commit 无 tool 行+返回 ids、五个新方法)、图集成(盖章进 checkpoint、降级持久化、摘要触发不阻塞 SSE、history_block 共享)、API 契约(404/排序/preview/summarized)、前端字符串断言。
- 例外(按要求替代 TDD):摘要 prompt → 标注样例 + `probe_summary.py` 真模型验证;验收 1/2/3 → `evals/probe_context.py` 20+ 轮真模型长跑(烧额度手跑);验收 4 → grep 日志;验收 5 → 人工点验。
- 收尾全量 `uv run pytest`(530+ 必须全绿,项目红线)。

## §15 验收映射

| # | 验收 | 验证方式 |
|---|---|---|
| 1 | 20+ 轮 token 不爆 | `probe_context.py` 默认窗口长跑 + 全量 pytest |
| 2 | 演示配置触发降级→摘要级联,梗概答对早期订单 | 演示配置跑 `probe_context.py`(剧本含早期订单号+尾问「最开始那个订单后来怎么说」),grep 日志看 `层1 降级`/`summary trigger`/`summary done`,人工核对末答 |
| 3 | 默认窗口 20 轮零降级零摘要 | 同一脚本默认配置,grep 断言无 `层1 降级`/`summary trigger` |
| 4 | 日志可见摘要痕迹且不阻塞回复 | `log/app.log` grep `model_ctx`/`history_ctx`/summary 生命周期;probe 计时末轮 SSE 时长不受摘要影响 |
| 5 | 侧栏多会话对照、切回回载、接着聊 | 人工点验(截图实证,沿 ch06 验收惯例) |

## §16 风险与红线

- 单 worker:内存 in-flight、会话锁、SQLite checkpoint、FileHandler 全部在该前提下成立,不引入新跨进程状态。
- checkpoint 悬空退化路径要有测试钉(锚点 id 在 state 找不到 → 摘要投影 + 空历史)。
- 摘要任务不得持会话锁、不得阻塞 SSE;失败只留日志不挪锚点。
- 改 prompt/检索后跑全量 pytest 再交付(项目红线);`probe_intent`/`probe_summary`/`probe_context` 烧额度,手跑不进 pytest。
- 起服红线不变:`no_proxy=127.0.0.1,localhost`、访问本机 `curl --noproxy '*'`。
- 库 API 核实:本会话 Context7 MCP 不可用(宿主未加载),已改用 .venv 内安装包源码核实——`trim_messages` 签名(langchain_core 1.6.2:`messages/max_tokens/token_counter/strategy/allow_partial/end_on/start_on/include_system/text_splitter`)、`add_messages`、`interrupt`/`Command`、SQLAlchemy 用法均以仓库现役代码为准。若后续 Context7 可用,实现阶段抽查复核一遍。
