# ch08 · 即插即用工具系统 · 设计修订稿

日期:2026-10-07
状态:已纳入用户裁决与实现规则,待用户确认

## 0. 背景与目标

把客服系统的工具层从几个写死的内置工具,升级成即插即用的工具系统:统一工具目录(内置 + MCP)、统一执行引擎(JSON Schema 校验 / 权限闸 / 超时重试 / 格式化)、统一审计落表(tool_audit_logs)、两个业务 MCP Server(物流 / 售后)动态接入、create_ticket 带前端预览确认流。

**本章不做**:Skill 机制(纯生态概念);仓储等更多外部系统;docker-compose 托管 MCP Server(手起进程即可);DB 持久化工具表 + 注册 HTTP API(方案 C,已否)。

**技术选型(定死)**:MCP Server 用官方 MCP Python SDK v1 系列的 FastMCP API,Streamable HTTP 传输;MCP Client 用 langchain-mcp-adapters 的 MultiServerMCPClient。两依赖均需新增进 pyproject 并锁定兼容主版本;实现不得混用 v2 `MCPServer` API。Adapter transport 字面量以锁定版本官方类型定义为准,并由真实 Server 的 list-tools 冒烟测试钉住 `/mcp` 端点。

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
| permission | `read` / `write` / `deny`。MCP 工具分类只认本地 Server 能力策略,Server 自报 annotations 一律不看;未知 Server 默认 `deny` |
| ownership | `none` / `order`(需订单归属,执行前过本地订单服务);MCP 的订单参数名由本地策略声明 |
| timeout / retries | 只读工具默认 `TOOL_TIMEOUT_SECONDS` / `TOOL_MAX_RETRIES`;按名超时可被 `TOOL_TIMEOUT_OVERRIDES` 覆盖;写工具使用 `TOOL_WRITE_TIMEOUT_SECONDS` 且重试上限恒为 0 |
| budget_priority | 工具 schema 超出本轮预算时的保留优先级(数值越大越优先保留);同级按稳定 name 排序 |
| formatter | 结果格式化器:只挑回答用得上的字段、内部枚举码翻人话、JSON 中文不转义 |
| callable / dispatch | 内置→本地函数;MCP→`client.session(server)` 内 `call_tool` |

- 内置条目:启动时扫描 `app/tools/builtin/` 登记,运行期不变。
- MCP 条目:每轮装配时 `get_tools()` 现问现拿现合成,**不持久化**。
- 目录对消费者只读;每轮产出的「工具面」= 目录子集 + 本轮绑定。
- logistics / after_sales 在本地策略中声明为只读 Server,只接受两种 schema 形态:仅含必填字符串 `order_id` 的工具继承 `ownership=order`,零参数工具继承 `ownership=none`;其他参数形态、无法证明归属或疑似写入的工具不进入工具面。其他未知 Server 或无匹配策略的 MCP 工具默认拒绝,不把 Server 自报 annotations 当授权。
- 动态发现遇到全局重名时拒绝该新条目并记 warning,不能按发现顺序覆盖既有工具。

### 1.2 内置工具注册与迁移

- `app/tools/builtin/*.py` 内 `@register_builtin(...)` 装饰器自登记;启动时扫描导入。新工具 = 新文件 + 装饰器,零核心代码改动(验收 1)。
- 迁移:`query_order`、`query_product`、`query_faq`(无参、同轮缓存、置信闸行为不变)迁入 builtin 登记模式;`query_logistics` 从内置清单**下线**,挪给物流 MCP Server,不留重名。
- `suggest_options` 伪工具维持现状(本地校验器,不进引擎)。
- 旧 `build_tools`(business.py)依旧禁回挂到图;`create_ticket` 以新写工具身份进入目录(§5)。

### 1.3 TurnContext(每轮显式上下文)

替代现有闭包绑定,数据类透传:`user_id` / `conversation_id` / `resolved_query` / `retriever` / `settings` / `session_factory`(审计用)。`conversation_id` 只取已确认的 MySQL conversation id;不能把任意 `thread_id` 字符串强转成会话 id,非数字或无法归属时传 `NULL`。装配工具面与 executor.execute 都收它。消除「目录全局、绑定每轮」的矛盾。

### 1.4 分支矩阵(不变)

| 分支 | 工具面 |
|---|---|
| business | 全部只读(内置 + MCP)+ `create_ticket` + suggest_options |
| refund 过闸 | query_order / query_product / 物流·售后 MCP(无 query_faq)+ suggest_options |
| clarify | 不绑工具 |

### 1.5 启动接线

`app/main.py:create_app` 生产运行时构建处:扫描 builtin 登记进 ToolCatalog → 建 MultiServerMCPClient 单例 → 挂 app.state → GraphDeps 加字段透传给 main_agent。MCP client 在应用 lifespan 中按其 API 做关闭/释放。启动预算自检只作基线;每轮实际工具面装配完成后,按该轮工具 schema 重新 `measure_sys_tokens` / `compute_budget`(可按工具签名缓存),不能继续复用不含动态工具的固定预算。动态工具导致预算不足时,按 `budget_priority` 数值从低到高、同级按 name 稳定剔除可选工具并记 warning;核心内置工具仍装不下时走既有安全错误路径,不得带着超预算工具面调用模型。

## 2. 执行引擎(升级 app/tools/executor.py)

所有模型发起的工具调用(内置 / MCP / 写)同走 `ToolExecutor.execute(call, ctx)`,固定六步;确认后的写入走独立的 `execute_confirmed(pending, ctx)` 路径,不能把 `confirmed` 当作模型可控参数传回普通 execute,避免权限闸绕回确认流:

1. **查目录**:名不在本轮清单 → `unknown_tool`(伪造 create_ticket 走这条,现状保留)。
2. **JSON Schema 校验**:类型不对 / 必填缺失 / 枚举越界 → 拦下,错误说明包成工具结果回灌模型(`{"error": "参数校验失败: ..."}`),审计记「校验拦下」,不重试。
3. **权限闸**:条目 `write` → 不直接执行,生成带 checkpoint 快照的 pending write 转 §5 确认流;未经确认的写调用不分发。MCP 读写分类只认本地能力策略;`deny` 条目直接回权限拒绝。
4. **归属把门**:条目 `ownership=order` → 本地订单服务 `get_order(ctx.user_id, order_id)` 为 None → 回「订单不存在或不属于当前用户」(业务空结果,不重试,**不发 MCP 调用**)。
5. **分发执行**:内置 → 本地函数;MCP → 在 `async with client.session(server) as session` 内 `call_tool`。`CallToolResult.isError=true` 统一归为 Server 失败,不把错误正文当业务结果回灌;只有网络抖动/连接异常才按策略重试。有效超时按以下优先级解析:`TOOL_TIMEOUT_OVERRIDES[tool_name]`(存在时) → 写工具的 `TOOL_WRITE_TIMEOUT_SECONDS` → 只读工具的 `TOOL_TIMEOUT_SECONDS`;按名覆盖只改变等待期限,不能给写工具打开重试。业务空结果不重试。写操作只由 `execute_confirmed` 分发,不自动重试。
6. **结果格式化**:条目 formatter 挑字段、枚举码翻人话、`json.dumps(ensure_ascii=False)`;MCP `isError` 先转为脱敏失败结果;过长走现有 `truncate_tool_result` token 截断。

**错误三分类**(回灌措辞按此分):
- 参数不合法 → 校验失败说明,引导模型重组调用或追问用户;
- 查询落空 → 「没查到」如实说,不算故障;
- 真故障(超时 / 连接断 / Server 报错) → 「服务暂时不可用」类措辞,审计记「失败/超时」+ 错误说明。

**写路径超时**:写操作默认超时(`TOOL_WRITE_TIMEOUT_SECONDS=10`),但遵守上一步的按名覆盖优先级;等待达到有效期限即向模型返回「提交超时、结果未知、请勿重复提交」类工具结果,`retry_count=0`,并在返回前记一条「超时」审计。底层 shielded 写任务继续执行,不得因请求断开或会话锁释放而被取消;任务由 `app.state` 应用级 pending-task 集合强引用,完成后移除。后台最终成功时只写幂等表/业务表并打结构化日志,不为同一 `tool_call_id` 再追加第二条审计记录;原审计行仍保留「超时」作为调用方观察到的终态。若首次执行在崩溃窗口内尚未留下审计,内部恢复最多补一条成功终态,不重复记同一 `tool_call_id`;应用优雅关闭时先等待该集合中的任务结束再释放 session factory;硬终止仍按结果未知处理,后续确认依靠幂等键重放。

**工具调用批次契约**:同一条模型响应最多允许一个 `write` call,且该 call 必须是该批次最后一个 call。写调用后仍有其他 call,或同批次出现多个写调用,整批按 `invalid_tool_call` 拦下且任何工具都不执行;批次内每个 call 均记「校验拦下」审计。对满足该契约的批次,写调用之前的只读 call 可照常执行;写调用本身进入确认流,不在本轮执行。这样中断前后始终能为该条 `AIMessage.tool_calls` 补齐确定的 `ToolMessage`。

现有 `ToolExecutionRecord` 保留并补齐 `status / source / mcp_server / result_summary` 等审计所需字段,作为审计数据源。`ToolRegistry` 被 ToolCatalog 吸收(重名拒注语义保留)。`execute` 产生的 pending write 在确认或取消时只落一条最终审计记录,不得因预览和执行各落一条。

## 3. 审计留痕(tool_audit_logs)

DDL 包含 `tool_audit_logs` 与 `tool_write_idempotency`;不改 ch02 的 `tickets` 表结构。审计表不挂外键,conversation_id 普通索引;status 中文 ENUM(成功/失败/超时/校验拦下/权限拒绝)。幂等表以唯一 key 映射到已创建的 ticket_no,和 `tickets` 插入处于同一 MySQL 事务,不挂外键。

**落库时机**:每次只读 `execute` 或确认后的 `execute_confirmed` 终结落一条(五状态含 write 确认流的「取消」);预览和 pending write 不单独落行;一个 tool_call 一条,retry_count 记实际重试次数。

**字段映射**:

| 列 | 来源 |
|---|---|
| conversation_id | `ctx.conversation_id`(已确认的 MySQL conversation id);无会话上下文 NULL,不得从任意 `thread_id` 字符串推导 |
| tool_call_id | 模型申请单 `call["id"]` |
| tool_name / tool_source / mcp_server | 目录条目(builtin → mcp_server=NULL) |
| arguments | 模型原始入参 JSON(写操作记确认前原始申请) |
| result_summary | 格式化后结果截断(`AUDIT_RESULT_MAX_CHARS=2000`,超长截断标注) |
| status / error_message | 引擎分诊;内部异常 message 不落,落分诊后说明(防泄密) |
| retry_count / duration_ms | 引擎实测(duration 含重试全程) |

**终态映射**:参数类型/必填/枚举错误、非法 tool-call 批次 → `校验拦下`;未知工具、MCP `isError`、未分类异常、重试耗尽、幂等键参数摘要冲突 → `失败`;超时 → `超时`;未授权 Server/工具、用户取消写入 → `权限拒绝`;订单不存在或不属于用户但工具正常返回业务空结果 → `成功`(result_summary 记脱敏空结果)。用户确认前的预览不是执行终态,暂不写审计行;confirm/cancel 各写一条。幂等摘要冲突只记录脱敏的内部一致性错误,不把摘要内容写入 `error_message`。

**写操作幂等表**:`tool_write_idempotency` 列为 `idempotency_key CHAR(64) PRIMARY KEY`(SHA-256(`conversation_id + ":" + tool_call_id`)),`arguments_sha256 CHAR(64) NOT NULL`,`ticket_no VARCHAR(32) NOT NULL`,`created_at DATETIME NOT NULL`。`conversation_id` 与 `tool_call_id` 均必须是非空、已校验的值;参数摘要按规范化 JSON(键排序、UTF-8、无空白)计算,避免同一参数不同序列化得到不同摘要。写确认必须带已确认的非空 `ctx.conversation_id`;缺失时在进入业务事务前按「校验拦下」终止。首次 confirm 在一个 Session/事务内插入 ticket 与幂等映射后一起 commit。`execute_confirmed` 在真正插入前先查该 key:相同 key 且参数摘要相同时读取并返回原 `ticket_no`,不再插入;参数摘要不同则拒绝并记「失败」,不执行第二次写入。并发唯一键冲突后回滚当前事务、读取已提交映射并按同一规则返回或拒绝。该表不存 description 原文。公开 `resume` 始终先在会话锁内校验 pending interrupt;已消费或不存在时直接 409,不因命中幂等表改成成功。只有 checkpoint 尚未推进的服务内部恢复或唯一键并发窗口可以读取幂等表收敛结果。

**写入方式**:同步 Session + `asyncio.to_thread` + `asyncio.shield`(照 `_create_ticket_sync` 模式);引擎构造时注入 session_factory。审计写入本身也受 shield 保护,但审计写失败不回滚工具业务事务;后台写任务的异常必须被应用级 task callback 消费并写 error 日志,不能产生未取得的 task exception。

**红线**:审计写失败不拦工具执行(try/except 兜底只打 error 日志);不挂外键,指向已删会话照写。

**DDL 配套**:`sql/ch08-ddl.sql` 与 `db/init/06-ddl.sql` 同时建审计表和幂等表并逐字节一致;`test_ddl_sync.py` DDL_PAIRS 加 `("06","ch08")`;`dbfixtures.py` DDL_PATHS/TABLES 同步并清理幂等表;`app/db.py` 的 `check_ch08_tables` 同时检查两张表(同构 check_ch04/07);`app/models.py` 加 ToolAuditLog 与 ToolWriteIdempotency ORM。

## 4. MCP 接入

### 4.1 两个业务 Server(`app/mcp_servers/`,各自独立进程)

官方 MCP Python SDK v1 系列的 FastMCP API,Streamable HTTP。Server 启动参数、HTTP 路径和 adapter transport 字面量必须以锁定依赖版本的官方类型定义为准,并由两条真实 Server 的 list-tools 冒烟测试覆盖。mock 照 ch02 做法:随机生成,不接真实系统、不建表、不认用户(归属在客户端把门)。生成按 order_id 种子稳定(同一单多次查结果一致,比 ch02 的纯随机更适合演示),但不读订单服务、不与订单状态联动(风险记 §7)。

| Server | 默认端口 | 工具 | mock 内容 |
|---|---|---|---|
| logistics | :8101 | `query_logistics(order_id)` | 公司/运单号/轨迹条,按 order_id 种子化生成 |
| after_sales | :8102 | `query_warranty(order_id)`、`query_return_progress(order_id)` | 在保/过保、退货进度状态机节点,种子化生成 |

启动:`uv run python -m app.mcp_servers.logistics` / `.after_sales`,端口走 .env;README/Makefile 补说明。

### 4.2 Client 装配

`MultiServerMCPClient` 全局单例(app.state),连接配置从 settings 读(server 名 → URL)。每轮装配工具面时按 Server 分别调用 `get_tools(server_name=...)`,每个调用单独套 `MCP_DISCOVERY_TIMEOUT_SECONDS` 并独立捕获异常,再合并成功 Server 的结果;不得用一次无 server 过滤的聚合调用让一个坏 Server 拖掉其他 Server。转换为本地目录条目(name/description/args_schema 照录;source=mcp + server 名;permission/ownership 由本地能力策略决定)。实际分发使用 adapter session 的异步上下文管理器,不能把 `client.session(server)` 当作普通同步对象调用。

- 某 Server 连不上:发现超时(`MCP_DISCOVERY_TIMEOUT_SECONDS=2`)即跳过记 warning,聊天照常,该 Server 工具本轮缺席;其他 Server 工具仍可用——MCP 故障不拖死主链路。
- 验收 3 闭环:Server 侧加工具 → 重启该 Server → 下一轮 get_tools() 带回 → Agent 可用;主服务代码不动、进程不重启。
- 归属把门在客户端(§2 第 4 步)。

### 4.3 依赖

pyproject 新增 `mcp>=1,<2` 与兼容的 `langchain-mcp-adapters` 版本;`uv sync` 更新锁。Server 统一使用 v1 `from mcp.server.fastmcp import FastMCP` API 与 `streamable-http` 传输;Client 按锁定 adapter 版本使用 `get_tools(server_name=...)` 和 `async with client.session(server) as session`。API 用法以锁定版本官方文档为准,并用 list-tools / call-tool 集成测试固定实际行为。

## 5. 建工单确认流

### 5.1 触发面

business 分支工具面新增 `create_ticket(ticket_type, description)`,permission=write。系统提示词加规矩:只有客户明确要求建工单时才发起;description 必填,信息不够先追问,不许瞎编(硬调残缺参数会被 §2 校验拦下回灌)。`test_prompts.py` 契约同步。

### 5.2 图内动线(照 ch06 订单选择器模式)

```
main_agent 发现 create_ticket 调用
  → 先校验「最多一个写调用且写调用最后」批次契约
  → 不执行写调用;同批次排在它前面的只读调用照常执行
  → state.pending_ticket = {tool_call_id, ticket_type, description, original_args, call_batch_digest}
  → 条件边 → ticket_confirm 节点
ticket_confirm:
  → interrupt({"type":"ticket_preview","tool_call_id","ticket_type","description"})
  → 驱动层扫 pending interrupt → SSE 帧 {"type":"ticket_preview","interrupt_id",...}(同 order_selector 链路)
resume:
  → confirm:只用 checkpoint 快照参数调用 `execute_confirmed`(客户端字段一律不信),经写路径(shield/不重试)落 tickets 表与幂等表
            → 审计「成功」→ ToolMessage(含工单号)进 messages
            → 不改 conv.status(与动作端点 _create_ticket_sync 口径一致)
  → cancel:不执行 → 审计「权限拒绝」(arguments=原始申请)→ ToolMessage「用户已取消」
  → 两条路 → 回 main_agent,模型带工具结果组织答复(confirm 答复带工单号)
```

**不在 main_agent 循环里直接 interrupt()**:LangGraph resume 从头重跑节点,main_agent 重跑 = 模型重调 + token 重流;独立节点与 refund_prepare 同构,重跑零代价。`ticket_confirm` 在恢复前先为挂起 AIMessage 的写 call 补一条最终 ToolMessage;该写 call 是批次最后一项,因此不存在“后续 call 无结果”的半闭合上下文。确认/取消都只能消费一次同一 `interrupt_id`。

### 5.3 resume 契约扩展

`POST /v1/chat/resume` 复用不新开端点。请求体放宽:`order_id` 变可选,新增可选 `decision: "confirm"|"cancel"`。服务端锁内校验照旧,按 pending interrupt 的 type 分派:

- order_selector → 必须带 order_id(旧契约逐字不变);
- ticket_preview → 必须带合法 decision;
- 卡片失效 / 字段不对 → 409;归属错误 → 404(不泄露)。

**确认执行的参数只信 checkpoint 快照**,客户端字段一律不采用(防篡改)。`decision` 只决定 confirm/cancel,不能携带或覆盖 `ticket_type / description / tool_call_id`。已消费或不存在 pending 的 `interrupt_id` 继续返回 409,即使幂等表已有该写调用记录;幂等表只覆盖 checkpoint 尚未推进的恢复窗口,不把旧卡变成可重放的公共接口。

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
| TOOL_TIMEOUT_OVERRIDES | '{}' | JSON 按名覆盖读写工具的等待期限,优先级见 §2,运维 knob + 验收 6 故障注入 |
| AUDIT_RESULT_MAX_CHARS | 2000 | result_summary 截断 |

### 6.2 测试策略(全量 pytest 须绿,现 576 条)

- 引擎单测(照 test_executor.py 夹具扩展):校验拦下 / 归属把门 / 重试只对暂时故障 / 写不重试 / `CallToolResult.isError` 脱敏失败 / 格式化中文不转义 / 五状态审计各一条 / 审计写失败不拦执行 / 写超时 `retry_count=0` 且立即返回 / 后台任务最终完成只保留一条超时审计。
- MCP 集成:真起两 Server 进程跑通三工具;单个 Server 宕机时另一个仍可用;Server 侧增加符合本地只读策略的工具后重起,下一轮被发现(验收 3 自动化);未知 Server、写能力或不符合 schema 的工具不进入工具面。
- 动态预算:同一模型在不同 MCP 工具面下按实际 schema 重新计算预算;工具面过大时先按能力优先级裁剪,不带超预算 schema 调模型。
- 确认流:帧序 `session → ticket_preview → [DONE]`;confirm 全链(tickets 与幂等表同事务落行 + 答复带工单号 + 审计成功);cancel(无新 tickets/幂等行 + 审计权限拒绝);409/404;SQLite 重开 resume 仍活(照 test_resume_survives_sqlite_reopen 先例)。
- 幂等与并发:已消费 `interrupt_id` 的公开重放仍为 409;参数摘要篡改拒绝;并发 confirm 只有一个事务建票,另一请求在锁内重新校验后按旧契约 409;模拟“业务事务已提交、checkpoint 尚未推进”后服务内部恢复读取同一映射、不重复建票。
- tool-call 批次:写调用必须最后且最多一个;非法批次零副作用;确认/取消后 AIMessage 的每个 tool_call_id 均有对应 ToolMessage。
- 前端:test_chat_page.py 预览卡 needle;prompt 契约同步。

### 6.3 验收 → 演示映射

| # | 验收 | 演示方式 |
|---|---|---|
| 1 | 新写工具零核心改动可用 | 现场新写小工具(如 query_store_hours):一个文件 + 装饰器,重启服,对话里用上 |
| 2 | 双 Server 独立进程,物流经 MCP | 两进程起来,问物流轨迹,Agent 经 MCP 查答 |
| 3 | Server 侧加符合本地只读策略的工具只重启该 Server | after_sales 加演示工具重启,主服务不动,下一轮发现并可用;不符合策略的工具保持缺席 |
| 4 | 追问补齐 → 预览卡 → 确认落表带单号 | 聊天「帮我建个工单」→ 追问 → 卡片 → 确认 → tickets 落行 + 答复带单号 |
| 5 | 取消 → 未建 + 审计权限拒绝 | 同上到卡片点取消,查 tool_audit_logs |
| 6 | 超时重试与写不重试 | `TOOL_TIMEOUT_OVERRIDES={"query_product":0.001}` → 超时 + retry_count=2 + 耗时;`{"create_ticket":0.001}` → 确认流提交 → 按名覆盖期限到期立即返回未知 + retry_count=0,后台可能仍落行但同一 key 重放只返回原工单(未覆盖时默认 10 秒) |

## 7. 风险与注意

- **SSE 协议新增一帧**(ticket_preview),全链四处(chat_service 事件 + _pending_selector_events 分支 + routers/chat.py 翻译 + chat.html 分发)与订单卡同构,漏一处卡片不出。
- **resume schema 放宽**不能削弱旧契约:order_selector 路径逐字保持,有钉。
- **每轮 get_tools() 成本**:localhost 两次 list_tools 毫秒级;必须按 Server 独立超时与降级,防一个 MCP 故障拖聊天,并按动态 schema 重算预算。
- **动态工具权限**:Server 自报 annotations 不提供授权;只有本地已知 Server 的只读能力策略可让新工具进入工具面,不符合 schema 或试图声明写能力的工具保持拒绝。
- **写路径超时语义**:超时后工具结果措辞必须阻止模型立刻重试(「请勿重复提交」);后台任务继续并受应用级集合跟踪,审计保留超时观察结果,幂等表负责后续确认/重放收敛。
- **超时后的用户可见性**:公开 `resume` 仍遵守已消费 `interrupt_id` → 409,后台任务完成不会补发 SSE;本章不新增工单状态查询接口,因此超时答复必须明确告知“结果未知、请勿重复提交”,由现有人工/运营渠道核验。内部恢复只使用幂等表,不把旧卡重新开放给浏览器。
- **审计与 messages 口径**:tool 结果行不落 messages 表的规矩不变;审计表是独立台账,不回灌上下文。
- **MCP 轨迹与订单状态可能不自洽**:Server 按 ch02 做法只认 order_id 种子、不读订单服务,物流轨迹的进度可能与 query_order 报的订单状态对不上(如订单「待发货」轨迹已到派送)。用户已拍板 ch02 做法;若演示在意,增强路径是归属把门时已拿到订单快照,把 status/created_at 作为模型不可见的注入参数带给 Server——本章不做。
