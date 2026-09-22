# ch07 · 会话上下文管理(三层历史 + 异步摘要)设计

日期:2026-09-22 · 状态:已评审(brainstorm 四节逐节确认; review 修订版)

## §1 定位与范围

把 ch01 沿用至今的「简单裁剪」(`fit_tool_context` 丢最旧轮 / `_history_turns` 按轮数截)升级为正经的会话上下文管理:三层历史(原文层 / 截短层 / 摘要层)+ 双锚点降级 + 后台异步分段摘要 + token 预算从模型窗口倒推 + 上下文全量可观测 + 前端多会话侧栏。

**只管当前会话。不跨会话、不存用户画像。** 语义检索捞历史、主题重要度留关键事实两层不实现;跨会话长期记忆不实现。

## §2 术语与三层定义

历史按离当前轮的远近分三层,边界由两个消息 id 锚点表达(存 `conversations` 行,**降级只挪 id、不搬数据**):

- `id ≤ summary_upto_msg_id`:已进摘要,不再进滑窗
- `summary_upto_msg_id < id ≤ layer1_from_msg_id`:**层 2**,渲染成截短形态
- `id > layer1_from_msg_id`:**层 1**,原样进上下文
- NULL 初态:`summary_upto=NULL` 无摘要;`layer1_from=NULL` 全部层 1

`layer1_from=NULL` 按负无穷解释;第一次降级无旧锚点也允许前移。所有锚点在应用层使用正十进制 `int`,写入 `additional_kwargs["db_id"]` 前后都必须显式转换,禁止把字符串和整数混作可比较值。

**数据流(已定)**:完整历史(含 ToolMessage)的唯一读源是 LangGraph checkpoint 里的 `state["messages"]`;锚点与摘要投影的权威存储是 MySQL `conversations` 行;分层组装每轮点查一次锚点,按消息上的 `db_id` 切 state 里的消息。ToolMessage 不单独落库,但属于前一个带 `tool_calls` 的 assistant 消息组,继承该组的 `db_id` 区间归属。

log 节点提交成功后,只给真正落库的 checkpoint 消息盖 `db_id`;“旧 checkpoint 历史 + 本轮盖章消息”只用于本轮降级/摘要计算,向 `add_messages` reducer 返回的仍然**只有本轮新增消息**,避免旧历史重复追加。不能只用提交前的 `state["messages"]` 做本轮降级或摘要触发。checkpoint 丢失或无法对齐 → 退化为「摘要投影 + 空历史」,不回填。

### 旧 checkpoint 兼容

ch06 已存在的 checkpoint 可能没有 `db_id`,或仍对应旧版落库的 ToolMessage。首次读取会话上下文时执行一次只读对齐:

1. 按 checkpoint 顺序读取 MySQL `messages`(包括历史 tool 行),对 user/assistant 比较 `role + content + tool_calls`;对 tool 只比较 `role + tool_call_id`,忽略 envelope 与模型原始结果文本差异。
2. 新版数据库没有 tool 行时,先从 checkpoint 序列过滤 ToolMessage,再与持久化 user/assistant 序列对齐。
3. 全序列可唯一对齐时,为对应 checkpoint 消息补盖 DB id,并让 ToolMessage 继承其所在 assistant tool-call 组的 id;对齐结果只写 checkpoint,不改历史表。
4. 任一条无法唯一对齐、顺序不一致或出现重复候选时,记录 `checkpoint_reconcile_failed`,本轮按“摘要投影 + 空历史”继续,不得猜测 ID。

对齐函数独立成纯函数并有旧版 tool 行、无 tool 行、内容重复、末尾未完成轮四类测试;正常新轮不再重复执行全量对齐。

## §3 DDL 与 ORM

- `sql/ch07-ddl.sql`(已备好,本章冻结):`conversations` 加 `summary TEXT`(摘要投影)/ `summary_upto_msg_id` / `layer1_from_msg_id` 三列;新建 `conversation_summaries(id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at)`,`UNIQUE(conversation_id, seq)`,**一段一行只追加,不回炉重压**。
- `conversation_summaries.conversation_id` 必须有 `FOREIGN KEY ... REFERENCES conversations(id)`、`idx_conv_upto` 和 `ON DELETE CASCADE`(或等价的清理策略);摘要行不能悬空。
- 头部注释修正:该文件实际两步合一,首段注释里「另一个锚点和分段摘要表在 ch07-layers.sql」是过时表述,实现时改写为单文件说明(`sql/` 与 `db/init/` 两侧同步改,保持逐字节一致)。
- 接线三处:`db/init/05-ddl.sql`(与 `sql/ch07-ddl.sql` 逐字节一致);`tests/test_ddl_sync.py` 的 `DDL_PAIRS` 加 `("05", "ch07")`;`tests/dbfixtures.py` 的 `DDL_PATHS` 加 `05-ddl.sql`、`TABLES` 在 `conversations` 前插 `conversation_summaries`。
- `app/models.py`:`Conversation` 加三列镜像;新增 `ConversationSummary` ORM,`conversation_id` 建关系/外键镜像。
- `app/db.py` 新增 `check_ch07_tables`(至少校验三列、`conversation_summaries` 和关键列),生产启动在 `check_ch04_tables` 后调用;缺失时直接给出 `mysql wayhelp < sql/ch07-ddl.sql` 升级提示,不得等到首轮请求才报错。
- 老库(非 docker 首启)需手工 `mysql wayhelp < sql/ch07-ddl.sql`,README 运行节补一句。

## §4 token 估算口径(唯一一把尺)

新 `app/services/token_budget.py`:

```
estimate_tokens(text)  = CJK字符数×1 + ceil(其余字符数/4)
estimate_message(msg) = estimate_tokens(content or "") + 4
                         + estimate_tokens(JSON(tool_calls))  # AIMessage 有 tool_calls 时
estimate_messages(msgs) = Σ estimate_message
```

CJK 区间明确为 `U+4E00–U+9FFF`、`U+3400–U+4DBF`、`U+3000–U+303F`、`U+FF00–U+FFEF`;其他字符按每 4 个 1 token 向上取整。`content=None` 按空串计,ToolMessage 的标识文本按实际渲染文本计。

对 deepseek 是轻微高估(安全方向)。**预算、自检、截断、触发全用这一把尺**;聊天链路里的 `count_tokens_approximately` 全部替换(它对中文按 ÷4 估算,低估约 4 倍,与新预算混用净效果是反的)。extract/mining 等离线链路不在预算关键路径,保留旧尺(§13 列明)。

## §5 预算公式与配置

从模型窗口倒推,不依赖散落的聊天常量:

```
PEAK   = MAX_AGENT_STEPS × (TOOL_RESULT_MAX_TOKENS + 50)
FIXED  = SYS_TOKENS + RERANK_TOP_K×300 + SUMMARY_PROJECTION_TOKENS + SAFETY_MARGIN_TOKENS
AVAIL  = MODEL_CONTEXT_WINDOW − MAX_OUTPUT_TOKENS − MAX_USER_INPUT_TOKENS − PEAK − FIXED
BUDGET = min(HISTORY_TARGET_TURNS × STEADY_TOKENS_PER_TURN, max(0, AVAIL))
层1预算 = floor(BUDGET × 0.7);层2预算 = BUDGET − 层1预算
```

`SYS_TOKENS` 是启动时按 §4 实测的“系统提示 + 当前分支最大工具集 schema”;`RERANK_TOP_K×300` 是证据预留;`SUMMARY_PROJECTION_TOKENS` 是摘要投影预留。`max_agent_tokens` **保留**为本轮 ReAct 实际/估算消耗上限,与上下文输入预算独立;不能用它替代 `BUDGET`。

**验收配置复算(写测试钉死)**:`MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 SYS_TOKENS=1901 SUMMARY_PROJECTION_TOKENS=650 SAFETY_MARGIN_TOKENS=550`:FIXED = 1901+1500+650+550 = 4601;AVAIL = 18000−2000−2000−3750−4601 = **5649**;层1 = **3954**;层2 = **1695**。默认窗口(65536)下按同一公式复算并记录日志,不能在测试里写死“4.6 万”近似值;默认 `BUDGET=min(20×300, AVAIL)=6000` 时,验收脚本的 20 轮应不触发降级/摘要。

**启动自检**(lifespan):算一遍 BUDGET 并 `logger.info` 打印全部分量;若 `BUDGET < STEADY_TOKENS_PER_TURN`(一轮都装不下)→ `logger.critical("上下文预算不足 …")` 报警,**不阻断启动**。替代 `main.py:121` 的 system-prompt 探针(原 raise 行为退役);`main.py:126` 的 extract 探针保留。

配置变更:

| 配置 | 默认 | 变化 |
|---|---:|---|
| `model_context_window` | 65536 | 新增 |
| `max_user_input_tokens` | 2000 | 新增;用户输入校验换 §4 尺,只量输入本身 |
| `tool_result_max_tokens` | 1200 | 新增;工具结果截断从字符换成 token 口径 |
| `summary_projection_tokens` | 650 | 新增;摘要投影预算,不再固定最近 3 段 |
| `rerank_top_k` | 10 | `rerank_top_n` 改名,语义和默认值不变 |
| `history_target_turns` | 20 | 新增 |
| `steady_tokens_per_turn` | 300 | 新增 |
| `summary_max_chars` | 200 | 新增,单段梗概上限 |
| `safety_margin_tokens` | 550 | 新增,配平冻结值 |
| `max_input_tokens` | — | 聊天链路停用;extract/mining 保留同名同义 |
| `max_tool_result_chars` | — | 聊天 envelope 落库职责消亡;保留为证据组装字符预算 |
| `understand_history_turns` | — | 退役,分层滑窗接管 |

环境变量沿项目现有大小写不敏感的 Settings 约定,文档与代码统一使用 snake_case 字段名。

## §6 SessionStore 协议扩展

内部 `StoredMessage` 继续表示待落库行,不携带数据库字段;新增只读 DTO:

```
ConversationMessage(id: str, role: Literal["user","assistant"], content: str,
                   tool_calls: list[dict] | None, created_at: datetime)
PersistedMessageRecord(id: str, role: Literal["user","assistant","tool"],
                       content: str | None, tool_calls: list[dict] | None,
                       tool_call_id: str | None, created_at: datetime)
ConversationItem(id: str, status: str, created_at: datetime, updated_at: datetime,
                 preview: str | None, summarized: bool)
ContextMeta(summary: str | None, summary_upto: int | None, layer1_from: int | None)
```

`CommitTurnResult` 加 `message_ids: list[str]`(本轮真正落库行 id,与 `stored` 顺序对齐;`source_message_id` 仍为第一条 user 行 id)。`_to_stored` 不再只返回 list,而返回 `PersistedTurn(stored, checkpoint_indexes)`,其中 `checkpoint_indexes` 是每个 `stored` 行在 `turn_messages` 中的原始索引;log 盖章只能按这张映射表做,禁止按整轮 zip。

协议新增(全部 async;`DbSessionStore` SQL 实现,`InMemorySessionStore` 内存实现供图测试):

```
get_context_meta(session_id, user_id) -> ContextMeta
list_conversations(user_id) -> list[ConversationItem]   # 新在前(updated_at DESC, id DESC)
list_messages(session_id, user_id) -> list[ConversationMessage] | None
list_checkpoint_records(session_id, user_id) -> list[PersistedMessageRecord] | None
move_layer1_from(session_id, user_id, new_id) -> bool
fetch_span_texts(session_id, after_id, upto_id) -> list[tuple[int, str, str]]
append_summary(session_id, from_id, upto_id, content) -> SummaryAppendResult
```

`list_messages` 是前端回放 DTO:只返回 user/assistant 且 `content` 非空的行,过滤 tool 行和 `content=NULL` 的工具调用 assistant;`list_checkpoint_records` 供旧 checkpoint 对齐,可读 legacy tool 行。所有归属不符返回 `None`,不泄露 404。

`commit_turn` 行为变更:**`_to_stored` 不再产出 tool 行**(tool 结果不落消息表,但带 `tool_calls` 的 assistant 行仍落);`validate_turn` 中间 assistant 仍须带 tool_calls,不再要求 tool 一一配对,出现 `role=="tool"` 直接拒。`InMemorySessionStore` 为每个持久化行分配递增 id,`message_ids` 逐行返回;锚点/摘要用 per-session dict 支持,使图集成测试不需要 DB。

`move_layer1_from` 以事务条件更新保证只前移,且 `new_id` 必须属于该会话并落在完整 turn 边界;NULL 视为负无穷。`SummaryAppendResult` 至少包含 `seq: int | None`、`applied: bool`、`reason: str | None`;`append_summary` 在同一事务内对会话行加锁,把 NULL `summary_upto` 规范化为 0,校验 `summary_upto == from_id` 且 `upto_id <= 当前 layer1_from`,再按 `MAX(seq)+1` 插入并更新投影/锚点。条件不满足返回 `applied=False`,不得覆盖新摘要。

## §7 三层组装器 `app/services/context_layers.py`

纯函数核心(DB 读由调用方注入):

```
build_layered_view(messages, meta: ContextMeta, settings) -> LayeredView
LayeredView: layer2_rendered: list[BaseMessage]
             layer1: list[BaseMessage]
             summary: str | None
             l1_tokens/l2_tokens/summary_tokens: int
             span_meta: dict
```

组装前先执行 `reconcile_checkpoint_ids`(仅对发现缺失 `db_id` 的旧 checkpoint 执行,规则见 §2)。无法对齐时返回 `LayeredView(summary=meta.summary, layer1=[], layer2_rendered=[], span_meta={"reconcile":"failed"})`。

渲染规则:层 2 内用户原话不动;assistant 答复只留开头 `LAYER2_HEAD_CHARS=50` 字 + 「…」;ToolMessage 大块结果换成一行标识 `[工具结果已省略: <name>]`;tool 组配对不断,截短不改结构。ToolMessage 继承所属 assistant tool-call 组的 `db_id`,不作为独立消息 id。understand/classify 的滑窗文本渲染(「用户:/助手:」行)里,ToolMessage 无论层级一律渲染为同一行标识,不展开。

**降级**(log 节点 commit 后,同步完成):以 `pre_turn_messages = state["messages"]` 与 `stamped_turn_messages` 合并后的完整列表为输入,对 `db_id > layer1_from` 的原文段(含本轮)做 `trim_messages(strategy="last", max_tokens=层1预算, token_counter=§4尺, start_on="human", include_system=False, allow_partial=False)`;有丢弃 → `layer1_from` 前移到被保留首条消息前一个持久化 id(整轮为界,tool 组不跨层劈开),`move_layer1_from` 落库。若单轮就超预算,层1可为空,但必须保留本轮 user/assistant 的可落库事实并由层2/摘要在下轮接住,系统不崩。

## §8 图改造

- `ChatGraphState` 加 `history_block: str`、`history_layer2: list[BaseMessage]`、`history_layer1: list[BaseMessage]`、`history_summary: str | None`(均为临时字段,`new_turn_state` 重置为空);这样 understand/classify/main_agent 复用同一份已组装视图,本轮只点查一次锚点。
- 图节点入口的 `state["messages"]` 明确定义为本轮开始前的 checkpoint 历史;当前用户只在 `turn_messages` 中出现一次,不能被同时拼入 history。
- `understand_query`:`meta = await store.get_context_meta(thread_id,user_id)` → 对旧 checkpoint 执行对齐 → `build_layered_view` → 渲染「摘要行 + 分层滑窗(层2截短+层1原文,用户:/助手: 行)」写 `state["history_block"]`;每轮打 `history_ctx` 日志(含无历史轮);短路条件改为「滑窗空 且 无摘要 且 无 active_summary」才透传不调模型。`_history_turns`/`_render_history` 删除。
- `classify_intent`: `INTENT_PROMPT` 加 `{history_block}` 占位,注入同一份共享渲染(指代消解/意图识别共用的历史)。
- `main_agent`(business / refund 过闸 / clarify):直接消费 state 中同一份 `history_layer2/history_layer1/history_summary`,按同一份 `LayeredView` 拼装:
  ```
  [0] SystemMessage(人设+红线,逐字不变;工具定义走 bind_tools,每轮一致)
  […] 层2截短渲染 + 层1原文(时间序、原 role)
  [k] HumanMessage 当前用户句
  [k+1] HumanMessage「【对话背景】梗概投影 + 【检索证据/订单事实】」合成一条(都为空则不出)
  [k+2…] 本轮 ReAct 步消息(AI/Tool)
  ```
  每步调模型前打 `model_ctx`。meta 在节点入口读一次,整步复用。估算总量 > `model_context_window - max_output_tokens` → 返回 None → 现有 `tool_context_too_long` 兜底帧。`build_agent_context` 重写;`fit_tool_context`/`build_messages_with_tools`/`_drop_broken_head` 退役。
- `log`:先将 `_to_stored` 返回的 `PersistedTurn` 交给 `commit_turn`;成功后按 `message_ids ↔ checkpoint_indexes` 盖 `db_id` 到本轮消息副本,用“旧历史 + 盖章本轮”计算降级/摘要,但返回给 `messages` reducer 的只有盖章本轮消息。随后评估层1降级;层2渲染态 token > 层2预算时 `ensure_future` 起摘要任务,不阻塞 SSE 收尾。
- interrupt 挂起轮不 commit、不进 checkpoint messages,天然不碰锚点。

## §9 摘要任务 `app/services/summarizer.py`

- **触发**:log 节点 commit 与层1降级完成后,层2渲染态 token 估算 > 层2预算;触发看用量不数条数。
- **批次**:任务开始时捕获当前 `(summary_upto, layer1_from]`;若区间为空 skip。完成后只把该区间推进为摘要,不吞并后来新增消息。
- **数据源**:`fetch_span_texts` 读 messages 表该区间 user/assistant 原文;工具调用 assistant 的 `content=NULL` 跳过,不得把 NULL 拼进 prompt。
- **prompt 契约**(新 `app/prompts/summary.py`):只提炼事实与诉求(问过哪款商品、报过的订单号/手机号、明确诉求、还没解决的问题);对话里没出现的一个字不许编;寒暄闲聊不留;≤ `summary_max_chars`;旧投影只作背景传入帮助连贯,输出只覆盖新区间,**不合并不重写旧段**。输出超限直接截断 + 日志。
- **投影策略(方案 A)**:`conversation_summaries` 每段永久保留;`conversations.summary` 不再固定最近 3 段。每次 append 在 `summary_projection_tokens` 预算内按 `seq` 升序为**所有已有段**分配可用额度:能完整放入则保留全文,超预算则对该段做仅用于投影的 head+tail 截短并加省略标记,不改 `content` 原文;每段至少贡献有限的可识别片段。该函数纯函数化并测试“4 段以上仍无最近 3 段硬上限、顺序稳定、预算不超”。
- **异步**:`ensure_future` + `shield`;运行时持有按 conversation_id 防重入的 in-flight 集合(单 worker 内成立),应用关闭时由 task registry 统一取消/等待。skip:in-flight / 区间为空;失败 `logger.exception`,锚点不动,下轮自然重试。
- **并发安全**:任务提交摘要时必须在同一事务中锁定 conversation 行,校验 `summary_upto` 仍等于任务捕获的 `from_id` 且 `upto_id <= 当前 layer1_from`;不满足返回 `applied=False`,不改 projection/锚点。`move_layer1_from` 与摘要更新都只允许单调前移。
- **prompt 验证例外**: `evals/` 加标注样例 jsonl(对话 + 必含事实清单 + 禁含闲聊清单)与 `probe_summary.py`,真模型跑一遍人工核对;不进 pytest。

## §10 日志契约(`log/app.log`)

`main.py` 加幂等的 FileHandler(`log/app.log`,utf-8,追加;目录自动创建;`log/` 入 `.gitignore`),保留 stderr;测试多次 `create_app` 不得重复挂 handler。所有上下文事件使用单行 JSON,字段值必须可 JSON 序列化:

- `history_ctx {"session":…,"summary":…,"window":[{"role":…,"content":…,"db_id":…}],"est":{"l1":…,"l2":…,"summary":…}}` — understand 渲染,每轮必打(闲聊/投诉兜底轮也有)。
- `model_ctx {"session":…,"step":…,"budget":{"l1":…,"l2":…},"counts":{"l1":…,"l2":…},"est":{"l1":…,"l2":…,"summary":…,"total":…},"summary":"投影全文","layer2":[…],"layer1":[…]}` — 主力 Agent 每步调模型前,原样不截。
- `context_reconcile_failed session=… reason=…`、`层1 降级 session=… <旧>→<新>`(旧为 None 显示 `-`)。
- `summary trigger session=… 层2 约 N token > 预算 M` / `summary start session=… 区间 (a, b]` / `summary done session=… 第N段 覆盖 (a, b] 耗时 Xms` / `summary skip session=… reason=…` / `summary failed session=…`(exception 堆栈)。

## §11 只读接口(新 `app/routers/conversations.py`,`/api` 前缀)

- `GET /api/conversations?user_id=…` → `list[ConversationItem]`,新在前;ID、时间分别序列化为十进制字符串和 ISO 8601。
- `GET /api/conversations/{id}/messages?user_id=…` → `list[ConversationMessage]` 按 id 升序;只返回 user/assistant 非空文本,归属不符 404(复用现有 AppError→HTTP 映射,不泄露)。
- 响应模型进 `app/schemas.py`;查询只读,不改 SSE 协议,不加新帧类型。用户 ID 复用现有 UUID 校验,会话 ID 复用十进制/UUID schema 但生产 DB 只接受合法十进制。

## §12 前端侧栏(`app/static/chat.html`,Vibe Coding 例外)

- 左栏 `#sidebar` + `#conv-list`:加载 `GET /api/conversations?user_id=…`,渲染首问预览 + 时间 + 「已摘要」标记;**加载失败 catch 隐藏侧栏,聊天不受影响**(静默降级)。
- 点击 `.conv-item`: `GET /api/conversations/{id}/messages?user_id=…` → 清聊天区回放返回的 user/assistant 纯文本气泡(tool 徽章/citations/suggest 按钮不回放)→ 切 `sessionId` 接着聊(checkpoint 在,上下文不断);活跃项高亮。回放完成前锁发送区,失败时恢复当前会话 UI 并显示轻量错误。
- 「+ 新对话」:sessionId 置 null 清屏,旧会话留侧栏;首轮 session 帧回来把新会话插到列表顶部,并以 API 列表数据刷新该项的 preview/updated_at。
- `tests/test_chat_page.py` 字符串断言钉:sidebar DOM id、两个带 query 参数的 fetch URL、catch 降级路径、回放只建立 user/assistant 气泡。

## §13 退役与行为变更清单(存量改写)

1. `_to_stored` 不产 tool 行但返回 `PersistedTurn`;`validate_turn` 改 §6;`test_store_db`/`test_graph_nodes` 的 envelope 断言改写;`tool_envelope.wrap` 失去聊天链路调用方(模块保留,mining 等不涉)。
2. `fit_tool_context`/`build_messages_with_tools`/`_drop_broken_head`/`_validate` 退役;`test_tool_chat_chain.py` 按新语义重写。
3. prepare 早闸:`check_input_budget(system, input, max_input_tokens)` → `check_user_input(input, max_user_input_tokens)`(§4 尺、只量输入);`max_message_chars` 保留为字符闸。
4. `understand_history_turns` 退役。
5. `main.py:121` system-prompt 探针 → §5 启动自检(critical 不 raise);extract 探针保留;新增 `check_ch07_tables`。
6. `rerank_top_n` → `rerank_top_k` 改名(默认 10 不变);波及 retriever/nodes/business/test_config/evals 脚本,逐一改。
7. `INTENT_PROMPT` 加 `{history_block}`;`probe_intent` 人工重跑。
8. `AGENTS.md`(落库口径、新配置、log/app.log、旧 checkpoint 对齐、当前状态)与 README(运行节:老库手工 apply ch07 DDL)收尾更新。

## §14 测试策略

- TDD 先行:`token_budget`(口径/公式/5649·3954·1695 复算钉/自检阈值/`max_agent_tokens` 分离)、`context_layers`(三层切分/截短渲染/tool 标识/NULL 与悬空锚点/单轮超预算/旧 checkpoint 对齐失败)、`summarizer`(触发/skip/追加/投影所有段分配/锚点前移/CAS 失败/失败不挪锚点,conftest 假模型)、store 两实现(逐行 ids、无 tool 行、五个新方法、归属与并发条件更新)、图集成(旧历史+本轮降级、盖章进 checkpoint、摘要触发不阻塞 SSE、history_block 共享)、API DTO 契约(404/排序/preview/summarized/ISO 时间)、前端字符串断言。
- 数据库测试新增:ch07 DDL 字节同步、外键级联、启动缺 schema 失败提示、摘要条件更新在并发下不回退。
- 例外(按要求替代 TDD):摘要 prompt → 标注样例 + `probe_summary.py` 真模型验证;验收 1/2/3 → `evals/probe_context.py` 20+ 轮真模型长跑(烧额度手跑);验收 4 → grep 日志;验收 5 → 人工点验。
- 收尾全量 `uv run pytest`(530+ 必须全绿,项目红线)。

## §15 验收映射

| # | 验收 | 验证方式 |
|---|---|---|
| 1 | 20+ 轮 token 不爆 | `probe_context.py` 默认窗口长跑 + 全量 pytest |
| 2 | 演示配置触发降级→摘要级联,梗概答对早期订单 | 演示配置跑 `probe_context.py`(剧本含早期订单号+尾问「最开始那个订单后来怎么说」),grep 日志看 `层1 降级`/`summary trigger`/`summary done`,人工核对末答;测试同时证明投影不受“最近 3 段”硬上限限制 |
| 3 | 默认窗口 20 轮零降级零摘要 | 同一脚本默认配置,grep 断言无 `层1 降级`/`summary trigger` |
| 4 | 日志可见摘要痕迹且不阻塞回复 | `log/app.log` grep `model_ctx`/`history_ctx`/summary 生命周期;probe 计时末轮 SSE 时长不受摘要影响 |
| 5 | 侧栏多会话对照、切回回载、接着聊 | 人工点验(截图实证,沿 ch06 验收惯例) |

## §16 风险与红线

- 单 worker:内存 in-flight、会话锁、SQLite checkpoint、FileHandler 全部在该前提下成立;数据库 CAS 仍是摘要/锚点正确性的最终保障。
- checkpoint 悬空或旧 checkpoint 对齐失败退化路径要有测试钉(锚点 id 在 state 找不到 → 摘要投影 + 空历史);不得猜测历史 ID。
- 摘要任务不得持会话锁、不得阻塞 SSE;失败只留日志不挪锚点;应用关闭要统一取消/等待后台任务。
- 改 prompt/检索后跑全量 pytest 再交付(项目红线);`probe_intent`/`probe_summary`/`probe_context` 烧额度,手跑不进 pytest。
- 起服红线不变:`no_proxy=127.0.0.1,localhost`、访问本机 `curl --noproxy '*'`。
- 库 API 核实:本会话 Context7 MCP 不可用(宿主未加载),已改用 .venv 内安装包源码核实;实现阶段若 Context7 可用再抽查复核一遍。
