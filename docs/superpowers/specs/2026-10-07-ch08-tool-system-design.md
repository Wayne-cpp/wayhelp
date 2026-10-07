# ch08 · 即插即用工具系统 · 设计定稿

日期:2026-10-07
状态:已按 Superpowers brainstorming 流程分节确认(§1–§6 全部用户 ok)

## 0. 背景与目标

把客服系统的工具层从几个写死的内置工具,升级成即插即用的工具系统:统一工具目录(内置 + MCP)、统一执行引擎(JSON Schema 校验 / 权限闸 / 超时重试 / 格式化)、统一审计落表(tool_audit_logs)、两个业务 MCP Server(物流 / 售后)动态接入、create_ticket 带前端预览确认流。

**本章不做**:Skill 机制(纯生态概念);仓储等更多外部系统;docker-compose 托管 MCP Server(手起进程即可);DB 持久化工具表 + 注册 HTTP API(方案 C,已否)。

**技术选型(定死)**:MCP Server 用官方 MCP Python SDK,Streamable HTTP 传输;MCP Client 用 langchain-mcp-adapters 的 MultiServerMCPClient。两依赖均需新增进 pyproject。

**方案裁定**:进程内全局工具目录 + 每轮工具面装配(方案 A)。「不重启服务」只约束 MCP 侧;内置新工具是新 Python 文件,启动时扫描登记(验收 1 不要求免重启)。

## 1. 总体架构与工具目录

### 1.1 ToolCatalog(进程级,挂 app.state)

进程内唯一工具表,比照 retriever/kb_store 挂载。条目字段:

| 字段 | 说明 |
|---|---|
| name | 工具名,全目录唯一(重名拒注,沿用现有 ToolRegistry 语义) |
| description | 用途描述(模型可见) |
| args_schema | JSON Schema 参数定义(模型可见 + 执行前校验依据) |
| source | `builtin` / `mcp`;MCP 条目另带 `mcp_server` 名 |
| permission | `read` / `write`。MCP 工具分类只认本地策略表(默认 read),Server 自报 annotations 一律不看 |
| ownership | `none` / `order`(需订单归属,执行前过本地订单服务) |
| timeout / retries | 超时秒数与重试上限,可被 `TOOL_TIMEOUT_OVERRIDES` 按名覆盖 |
| formatter | 结果格式化器:只挑回答用得上的字段、内部枚举码翻人话、JSON 中文不转义 |
| callable / dispatch | 内置→本地函数;MCP→`client.session(server)` 内 `call_tool` |

- 内置条目:启动时扫描 `app/tools/builtin/` 登记,运行期不变。
- MCP 条目:每轮装配时 `get_tools()` 现问现拿现合成,**不持久化**。
- 目录对消费者只读;每轮产出的「工具面」= 目录子集 + 本轮绑定。

### 1.2 内置工具注册与迁移

- `app/tools/builtin/*.py` 内 `@register_builtin(...)` 装饰器自登记;启动时扫描导入。新工具 = 新文件 + 装饰器,零核心代码改动(验收 1)。
- 迁移:`query_order`、`query_product`、`query_faq`(无参、同轮缓存、置信闸行为不变)迁入 builtin 登记模式;`query_logistics` 从内置清单**下线**,挪给物流 MCP Server,不留重名。
- `suggest_options` 伪工具维持现状(本地校验器,不进引擎)。
- 旧 `build_tools`(business.py)依旧禁回挂到图;`create_ticket` 以新写工具身份进入目录(§5)。

### 1.3 TurnContext(每轮显式上下文)

替代现有闭包绑定,数据类透传:`user_id` / `conversation_id` / `resolved_query` / `retriever` / `settings` / `session_factory`(审计用)。装配工具面与 executor.execute 都收它。消除「目录全局、绑定每轮」的矛盾。

### 1.4 分支矩阵(不变)

| 分支 | 工具面 |
|---|---|
| business | 全部只读(内置 + MCP)+ `create_ticket` + suggest_options |
| refund 过闸 | query_order / query_product / 物流·售后 MCP(无 query_faq)+ suggest_options |
| clarify | 不绑工具 |

### 1.5 启动接线

`app/main.py:create_app` 生产运行时构建处:扫描 builtin 登记进 ToolCatalog → 建 MultiServerMCPClient 单例 → 挂 app.state → GraphDeps 加字段透传给 main_agent。启动预算自检(SYS_TOKENS 实测)的工具面改走目录装配,口径不变。

## 2. 执行引擎(升级 app/tools/executor.py)

所有工具调用(内置 / MCP / 写)同走 `ToolExecutor.execute(call, ctx)`,固定六步:

1. **查目录**:名不在本轮清单 → `unknown_tool`(伪造 create_ticket 走这条,现状保留)。
2. **JSON Schema 校验**:类型不对 / 必填缺失 / 枚举越界 → 拦下,错误说明包成工具结果回灌模型(`{"error": "参数校验失败: ..."}`),审计记「校验拦下」,不重试。
3. **权限闸**:条目 `write` → 不直接执行,转 §5 确认流;未经确认的写调用直接拒绝,审计记「权限拒绝」。MCP 读写分类只认本地目录登记。
4. **归属把门**:条目 `ownership=order` → 本地订单服务 `get_order(ctx.user_id, order_id)` 为 None → 回「订单不存在或不属于当前用户」(业务空结果,不重试,**不发 MCP 调用**)。
5. **分发执行**:内置 → 本地函数;MCP → `client.session(server)` + `call_tool`。超时默认 `tool_timeout_seconds=5`,可按名覆盖;重试只给网络抖动/连接异常(指数退避,上限 `tool_max_retries=2`);**写操作不自动重试**(shield 保护,超时按「已发未必未成」);业务空结果不重试。
6. **结果格式化**:条目 formatter 挑字段、枚举码翻人话、`json.dumps(ensure_ascii=False)`、过长走现有 `truncate_tool_result` token 截断。

**错误三分类**(回灌措辞按此分):
- 参数不合法 → 校验失败说明,引导模型重组调用或追问用户;
- 查询落空 → 「没查到」如实说,不算故障;
- 真故障(超时 / 连接断 / Server 报错) → 「服务暂时不可用」类措辞,审计记「失败/超时」+ 错误说明。

**写路径超时**:写操作也设超时(`TOOL_WRITE_TIMEOUT_SECONDS=10`);超时 → 记「超时」、`retry_count=0`,回灌模型「提交超时、结果未知、请勿重复提交」类措辞。

现有 `ToolExecutionRecord` 保留,作为审计数据源。`ToolRegistry` 被 ToolCatalog 吸收(重名拒注语义保留)。

## 3. 审计留痕(tool_audit_logs)

DDL 即 `sql/ch08-ddl.sql`(已给定,不改):不挂外键,conversation_id 普通索引;status 中文 ENUM(成功/失败/超时/校验拦下/权限拒绝)。

**落库时机**:每次 execute 终结落一条(五状态含 write 确认流的「取消」);一个 tool_call 一条,retry_count 记实际重试次数。

**字段映射**:

| 列 | 来源 |
|---|---|
| conversation_id | `ctx.conversation_id`(图内 = thread_id 数字化);无会话上下文 NULL |
| tool_call_id | 模型申请单 `call["id"]` |
| tool_name / tool_source / mcp_server | 目录条目(builtin → mcp_server=NULL) |
| arguments | 模型原始入参 JSON(写操作记确认前原始申请) |
| result_summary | 格式化后结果截断(`AUDIT_RESULT_MAX_CHARS=2000`,超长截断标注) |
| status / error_message | 引擎分诊;内部异常 message 不落,落分诊后说明(防泄密) |
| retry_count / duration_ms | 引擎实测(duration 含重试全程) |

**写入方式**:同步 Session + `asyncio.to_thread` + `asyncio.shield`(照 `_create_ticket_sync` 模式);引擎构造时注入 session_factory。

**红线**:审计写失败不拦工具执行(try/except 兜底只打 error 日志);不挂外键,指向已删会话照写。

**DDL 配套**:`db/init/06-ddl.sql` 逐字节复制;`test_ddl_sync.py` DDL_PAIRS 加 `("06","ch08")`;`dbfixtures.py` DDL_PATHS/TABLES 同步;`app/db.py` 加 `check_ch08_tables` 启动探针(同构 check_ch04/07);`app/models.py` 加 ToolAuditLog ORM。

## 4. MCP 接入

### 4.1 两个业务 Server(`app/mcp_servers/`,各自独立进程)

官方 MCP Python SDK,FastMCP,Streamable HTTP。mock 照 ch02 做法:随机生成,不接真实系统、不建表、不认用户(归属在客户端把门)。生成按 order_id 种子稳定(同一单多次查结果一致,比 ch02 的纯随机更适合演示),但不读订单服务、不与订单状态联动(风险记 §7)。

| Server | 默认端口 | 工具 | mock 内容 |
|---|---|---|---|
| logistics | :8101 | `query_logistics(order_id)` | 公司/运单号/轨迹条,按 order_id 种子化生成 |
| after_sales | :8102 | `query_warranty(order_id)`、`query_return_progress(order_id)` | 在保/过保、退货进度状态机节点,种子化生成 |

启动:`uv run python -m app.mcp_servers.logistics` / `.after_sales`,端口走 .env;README/Makefile 补说明。

### 4.2 Client 装配

`MultiServerMCPClient` 全局单例(app.state),连接配置从 settings 读(server 名 → URL)。每轮装配工具面时 `get_tools()` 现问现拿,转成本地目录条目(name/description/args_schema 照录;source=mcp + server 名;permission 默认 read;三个工具标 `ownership=order`)。

- 某 Server 连不上:发现超时(`MCP_DISCOVERY_TIMEOUT_SECONDS=2`)即跳过记 warning,聊天照常,工具本轮缺席——MCP 故障不拖死主链路。
- 验收 3 闭环:Server 侧加工具 → 重启该 Server → 下一轮 get_tools() 带回 → Agent 可用;主服务代码不动、进程不重启。
- 归属把门在客户端(§2 第 4 步)。

### 4.3 依赖

pyproject 新增 `mcp` 与 `langchain-mcp-adapters`;`uv sync` 更新锁。API 用法一律以 Context7 查到的官方文档为准(已查:MultiServerMCPClient 每次调用新建 session,天然无重连负担;get_tools() 支持按 server 过滤)。

## 5. 建工单确认流

### 5.1 触发面

business 分支工具面新增 `create_ticket(ticket_type, description)`,permission=write。系统提示词加规矩:只有客户明确要求建工单时才发起;description 必填,信息不够先追问,不许瞎编(硬调残缺参数会被 §2 校验拦下回灌)。`test_prompts.py` 契约同步。

### 5.2 图内动线(照 ch06 订单选择器模式)

```
main_agent 发现 create_ticket 调用
  → 不执行;同步骤排它前面的只读调用照常执行;write 及其后的调用压下
  → state.pending_ticket = {tool_call_id, ticket_type, description}
  → 条件边 → ticket_confirm 节点
ticket_confirm:
  → interrupt({"type":"ticket_preview","tool_call_id","ticket_type","description"})
  → 驱动层扫 pending interrupt → SSE 帧 {"type":"ticket_preview","interrupt_id",...}(同 order_selector 链路)
resume:
  → confirm:用 checkpoint 快照参数经引擎写路径(shield/不重试)落 tickets 表
            → 审计「成功」→ ToolMessage(含工单号)进 messages
            → 不改 conv.status(与动作端点 _create_ticket_sync 口径一致)
  → cancel:不执行 → 审计「权限拒绝」(arguments=原始申请)→ ToolMessage「用户已取消」
  → 两条路 → 回 main_agent,模型带工具结果组织答复(confirm 答复带工单号)
```

**不在 main_agent 循环里直接 interrupt()**:LangGraph resume 从头重跑节点,main_agent 重跑 = 模型重调 + token 重流;独立节点与 refund_prepare 同构,重跑零代价。

### 5.3 resume 契约扩展

`POST /v1/chat/resume` 复用不新开端点。请求体放宽:`order_id` 变可选,新增可选 `decision: "confirm"|"cancel"`。服务端锁内校验照旧,按 pending interrupt 的 type 分派:

- order_selector → 必须带 order_id(旧契约逐字不变);
- ticket_preview → 必须带合法 decision;
- 卡片失效 / 字段不对 → 409;归属错误 → 404(不泄露)。

**确认执行的参数只信 checkpoint 快照**,客户端字段一律不采用(防篡改)。

### 5.4 前端(Vibe Coding 例外)

预览卡片渲染工单类型 + 问题描述,「确认提交」「取消」两按钮;捕获式身份(user_id/session_id/interrupt_id)、全组 disable、409 提示「该选择已失效」、发新消息前端先失效、历史回载不带卡——全套照搬订单卡模式;TOOL_LABELS 补中文标签(含 MCP 三工具)。ch05 投诉流程 suggest_actions 按钮建单路径不动(点按钮本身就是确认)。`test_chat_page.py` 补字符串断言。

### 5.5 状态与路由

`ChatGraphState` 加 `pending_ticket`(每轮 `new_turn_state` 重置);main_agent → ticket_confirm 条件边;confirm/cancel 后 ticket_confirm → main_agent 边。ToolMessage 进 messages 走现有 add_messages;落库仍只收 user/assistant 行(validate_turn 口径不变)。

## 6. 配置、测试与验收映射

### 6.1 新增配置(.env.example 注释列约定)

| 配置 | 默认 | 用途 |
|---|---|---|
| MCP_LOGISTICS_URL / MCP_AFTER_SALES_URL | http://127.0.0.1:8101/mcp / :8102/mcp | Server 地址 |
| MCP_DISCOVERY_TIMEOUT_SECONDS | 2 | 每轮发现超时即跳过 |
| TOOL_WRITE_TIMEOUT_SECONDS | 10 | 写路径超时(超时≠没执行,绝不重试) |
| TOOL_TIMEOUT_OVERRIDES | '{}' | JSON 按名覆盖超时,运维 knob + 验收 6 故障注入 |
| AUDIT_RESULT_MAX_CHARS | 2000 | result_summary 截断 |

### 6.2 测试策略(全量 pytest 须绿,现 576 条)

- 引擎单测(照 test_executor.py 夹具扩展):校验拦下 / 归属把门 / 重试只对暂时故障 / 写不重试 / 格式化中文不转义 / 五状态审计各一条 / 审计写失败不拦执行 / 写超时 retry_count=0。
- MCP 集成:真起两 Server 进程跑通三工具;Server 宕机降级;Server 侧加工具重起后被发现(验收 3 自动化)。
- 确认流:帧序 `session → ticket_preview → [DONE]`;confirm 全链(tickets 落行 + 答复带工单号 + 审计成功);cancel(无新行 + 审计权限拒绝);409/404;SQLite 重开 resume 仍活(照 test_resume_survives_sqlite_reopen 先例)。
- 前端:test_chat_page.py 预览卡 needle;prompt 契约同步。

### 6.3 验收 → 演示映射

| # | 验收 | 演示方式 |
|---|---|---|
| 1 | 新写工具零核心改动可用 | 现场新写小工具(如 query_store_hours):一个文件 + 装饰器,重启服,对话里用上 |
| 2 | 双 Server 独立进程,物流经 MCP | 两进程起来,问物流轨迹,Agent 经 MCP 查答 |
| 3 | Server 侧加工具只重启该 Server | after_sales 加演示工具重启,主服务不动,再问即用 |
| 4 | 追问补齐 → 预览卡 → 确认落表带单号 | 聊天「帮我建个工单」→ 追问 → 卡片 → 确认 → tickets 落行 + 答复带单号 |
| 5 | 取消 → 未建 + 审计权限拒绝 | 同上到卡片点取消,查 tool_audit_logs |
| 6 | 超时重试与写不重试 | `TOOL_TIMEOUT_OVERRIDES={"query_product":0.001}` → 超时 + retry_count=2 + 耗时;`{"create_ticket":0.001}` → 确认流提交 → 超时 + retry_count=0(顺带演示「超时未必没执行」:tickets 可能仍落行) |

## 7. 风险与注意

- **SSE 协议新增一帧**(ticket_preview),全链四处(chat_service 事件 + _pending_selector_events 分支 + routers/chat.py 翻译 + chat.html 分发)与订单卡同构,漏一处卡片不出。
- **resume schema 放宽**不能削弱旧契约:order_selector 路径逐字保持,有钉。
- **每轮 get_tools() 成本**:localhost 两次 list_tools 毫秒级;必须带超时与降级,防 MCP 故障拖聊天。
- **写路径超时语义**:超时后工具结果措辞必须阻止模型立刻重试(「请勿重复提交」),答复建议用户查工单列表确认。
- **审计与 messages 口径**:tool 结果行不落 messages 表的规矩不变;审计表是独立台账,不回灌上下文。
- **MCP 轨迹与订单状态可能不自洽**:Server 按 ch02 做法只认 order_id 种子、不读订单服务,物流轨迹的进度可能与 query_order 报的订单状态对不上(如订单「待发货」轨迹已到派送)。用户已拍板 ch02 做法;若演示在意,增强路径是归属把门时已拿到订单快照,把 status/created_at 作为模型不可见的注入参数带给 Server——本章不做。
