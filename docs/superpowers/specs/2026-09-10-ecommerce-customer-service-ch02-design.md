# 电商智能客服 Ch02: Function Calling 工具链 - 设计文档

日期: 2026-09-10
状态: 复审修订稿(review 问题已并入,等待用户最终确认)

## 1. 目标与范围

在 Ch01 纯对话系统上接入 **Function Calling 工具链**: 用户在 SSE 聊天页提问,模型自己决定是否调用工具;若调用,服务执行工具并把结果回灌模型,最终回答继续流式输出。

交付五项能力:

1. 项目基建: FastAPI + SQLAlchemy 分层骨架;Docker 起 MySQL;创建 faq / conversations / messages / tickets 四张表,只给 faq 灌 seed,其余三表由运行时写入
2. 五个 LangChain `@tool` 业务工具: `query_order` / `query_product` / `query_logistics`(工具内随机生成演示数据,不接真实接口、不建表)、`query_faq`(SQL LIKE 查 faq 表)、`create_ticket`(写 tickets 表并更新会话状态)
3. 工具基础设施: 注册管理、参数 Schema 校验、执行错误处理、只读工具超时重试、工具结果回灌模型
4. 工具链接入现有 SSE 聊天: 模型选工具 -> 执行 -> 回灌收敛;推送工具状态帧;完整工具轨迹落 conversations/messages 表;聊天页显示工具徽章
5. 只做一个工具阶段: 第一次模型调用可申请 1 至 `MAX_TOOL_CALLS_PER_TURN` 个工具,按顺序执行;第二次模型调用绑定工具以承接结构化通道,但不再执行任何 tool_calls,必须直接收敛

技术栈(定死): Python 3.12(uv) + FastAPI + SQLAlchemy 2.x + MySQL 8(Docker) + LangChain(`langchain-core` / `langchain-openai`) `@tool` 装饰器。依赖版本写入并提交 `uv.lock`。

明确不做: 多轮自动循环的 Agent Loop;向量检索、RAG;真实电商/物流系统对接;正式身份认证。Ch02 仍只支持单进程运行。浏览器匿名 `user_id` 只是会话关联标识,不是认证凭证或安全边界。

## 2. 架构(方案 A: 手写单轮编排)

在 Ch01 四个模块(路由 / ChatService / chain / Store)上增加数据和工具模块:

- **数据模块**(`app/db.py` + `app/models.py`): SQLAlchemy 2.0 同步 engine + `sessionmaker`;ORM 模型与用户手写 DDL 逐字段对齐
- **工具模块**(`app/tools/`): `business.py` 定义五个 `@tool` 与每轮工具工厂;`executor.py` 提供 `ToolRegistry` 与工具执行器
- **组合入口**(`app/main.py`): `create_app` 接收可选的 `AppRuntime`;生产路径在未注入 runtime 时构造 DB Store 和真实工具,测试注入内存 Store 与 fake/空工具集

单轮编排手写实现(不引入 LangGraph / AgentExecutor): 第一次 `bind_tools` 调用聚合 tool calls,有则执行回灌;第二次调用绑定工具以承接结构化通道,但不再执行任何 tool_calls;模型仍试图调用则丢弃并以固定话术兜底;落库时剥离未执行的 tool_calls。落选方案: B(LangGraph `create_react_agent`,为单轮引入整个 LangGraph 不值,且 Agent Loop 是本章排除项)、C(LangChain `AgentExecutor` 限轮,当前官方主路径已转向 `create_agent`)。

`ChatService` 继续作为聊天流程的主要接口,集中处理会话锁、模型编排、输出校验和提交分界。chain 模块只负责消息构造、预算与模型调用;路由只编码 SSE;工具执行器只处理一次工具调用。调用方无需理解这些实现细节。

聊天页工具徽章按用户指定走 Vibe Coding;其依赖的匿名用户协议与 SSE 帧契约仍以本 spec 为准,后端实现照常测试。

## 3. 项目结构(在 Ch01 基础上新增/调整)

```text
docker-compose.yml           # mysql:8.4,挂载 db/init,健康检查,端口见 §9
db/init/01-ddl.sql           # 用户手写 sql/ch02-ddl.sql 原样复制到此
db/init/02-seed.sql          # faq 测试数据
app/
  db.py                      # engine + sessionmaker + ping;Session 在线程内创建/销毁
  models.py                  # Faq / Conversation / Message / Ticket ORM
  sessions.py                # SessionStore / StoredMessage / InMemorySessionStore
  store_db.py                # DbSessionStore
  tools/business.py          # 五个 @tool + build_tools(session_factory, conversation_id)
  tools/executor.py          # ToolRegistry + ToolExecutor
  chains/tool_chat_chain.py  # 消息预算、chunk 聚合、单轮工具编排辅助函数
  services/chat_service.py   # 会话锁、工具事件、整轮消息提交
  routers/chat.py            # session / delta / tool_start / tool_end / error SSE
  static/chat.html           # 匿名 user_id + 工具徽章(Vibe Coding)
  main.py                    # AppRuntime + create_app;供 uvicorn --factory 调用
tests/                       # Ch01 行为回归 + 新增 DB/工具/编排测试
dev-notes/ch02.md            # 过程留痕
```

新增依赖: `sqlalchemy>=2.0`、`pymysql`、`cryptography`(MySQL 8 默认认证插件需要)。不引入 Alembic: DDL 文件经 `/docker-entrypoint-initdb.d/` 首启执行,是建表唯一事实源;ORM 模型须与之逐字段一致。后续章节改表时再评估迁移工具。

## 4. 数据库、用户与会话标识契约

以用户手写 `db/init/01-ddl.sql` 为准,本章不改动其字段定义:

- **conversations**: `id` BIGINT UNSIGNED 自增主键;`user_id` VARCHAR(64) NOT NULL;`status` ENUM('进行中','已转人工','已结束') 默认 '进行中';`created_at` / `updated_at`
- **messages**: `id` 自增主键;`conversation_id` 外键;`role` ENUM('user','assistant','tool');`content` TEXT 可空;`tool_calls` JSON 可空;`tool_call_id` VARCHAR(64) 可空;`created_at`
- **faq**: `id` 自增主键;`question` VARCHAR(512);`answer` TEXT;`category` VARCHAR(64)
- **tickets**: `ticket_no` VARCHAR(32) 主键;`conversation_id` 外键;`description` TEXT;`ticket_type` ENUM('售后','投诉','咨询');`status` ENUM('待处理','已处理') 默认 '待处理';`created_at`

聊天请求扩展为:

```json
{"user_id":"浏览器匿名 UUID","session_id":"可选会话 ID","message":"用户输入"}
```

- 首次打开页面时用 `crypto.randomUUID()` 生成匿名 UUID,保存到 `localStorage` 的 `wayhelp_user_id`;以后每次请求都原样携带
- `user_id` 必须是规范 UUID 字符串,服务端保存到新建的 conversation;它只用于关联同一浏览器发起的会话
- 已有 session 必须同时匹配 `session_id` 与 `user_id`;不存在或不匹配统一返回 404 `session_not_found`,但该检查不构成身份认证
- `session_id` 在请求层接受两种规范形式: 无前导零的正十进制字符串(DB adapter)或规范 UUID 字符串(内存 adapter)。具体 Store 只识别自己创建的形式,另一种合法形式按不存在处理
- 服务端仍是 session id 的唯一创建者;客户端不能指定一个尚不存在的新 id

**seed(02-seed.sql)**: faq 至少 8 条,分类覆盖退货/换货/发票/运费/账户等;必须含可命中“退货政策”的条目;全表不出现“邮费”二字,保证验收中的 LIKE 漏召回如实发生。seed 只灌 faq;conversations/messages/tickets 由运行时产生。

**启动健康检查**: 生产 runtime 构造时执行 `SELECT 1`;失败则应用创建立即失败并指明数据库连接问题,公开文本与日志均不含密码或完整连接串。注入测试 runtime 时不创建或 ping 生产 DB。

## 5. 会话存储、消息重建与并发

### 5.1 SessionStore 接口

Ch02 保留 `create / exists / snapshot / commit_turn` 四个方法,扩展参数和语义:

- `create(user_id) -> session_id`: 创建会话并绑定匿名用户
- `exists(session_id, user_id) -> bool`: 同时核对会话与匿名用户
- `snapshot(session_id) -> list[StoredMessage]`: 返回不可变的已提交历史快照
- `commit_turn(session_id, messages)`: 在一个事务中提交本轮完整有序消息列表

`StoredMessage` 包含 `role`、可空 `content`、可选 `tool_calls`、可选 `tool_call_id`。一次合法 turn 必须满足:

1. 第一条是当前 user 消息,最后一条是非空 assistant 最终回答
2. 无工具时只有 user + assistant
3. 有工具时顺序为 user + assistant(tool_calls) + 与每个 call 一一对应的 tool 消息 + assistant
4. 每个 tool call id 非空且不超过 64 字符;每个 id 恰好对应一条 ToolMessage,无重复或孤儿结果

`DbSessionStore.commit_turn` 在单个数据库事务中插入全部 messages,并显式更新 `conversations.updated_at`;任一 SQL 失败则整体回滚。`InMemorySessionStore` 同样以完整 turn 为单位追加和淘汰,不能再按固定 user/assistant 两条删除。`MAX_MESSAGES_PER_SESSION` 不再要求偶数;内存 Store 始终保留最新完整 turn,并按完整 turn 从旧到新淘汰,因此实际消息上界是 `max(MAX_MESSAGES_PER_SESSION, MAX_TOOL_CALLS_PER_TURN + 3)`。DB Store 本章不做容量回收。

### 5.2 工具结果持久化格式

DDL 没有独立的工具名与执行状态字段,因此 `messages.content` 保存可重建的 JSON envelope:

```json
{"v":1,"ok":true,"content":"工具返回的字符串"}
{"v":1,"ok":false,"error_code":"invalid_args","message":"脱敏错误摘要"}
```

- assistant 行的 `tool_calls` 保存规范列表,每项只含 `name / args / id / type`
- tool 行通过 `tool_call_id` 在前一条 assistant 的 `tool_calls` 中找到工具名;通过 envelope 的 `ok` 恢复 `ToolMessage.status`
- envelope 格式不合法、tool call 对不上或缺失结果视为持久数据损坏,以内部错误结束请求,不得把非法消息发给上游模型

### 5.3 历史裁剪与第二次调用预算

DB 按 `created_at, id` 升序读取。历史先按完整 turn 分组,再沿用 Ch01 的 `trim_messages` 参数。裁剪后执行结构校验: user 起始、最终 assistant 收尾、assistant(tool_calls) 与其全部 ToolMessage 同在或同去;若头部留下半个 turn,继续丢弃整个最旧 turn并重新计数,直至合法。

第一次模型调用前保留 system 与当前 human。工具执行后,第二次模型调用用 `system + 已裁剪历史 + 当前 human + AIMessage(tool_calls) + 全部 ToolMessage` 重新计算预算;先继续删除最旧完整历史 turn。当前 turn 的工具组不可拆分,若删除全部历史后仍超出 `MAX_INPUT_TOKENS`,发送 `tool_context_too_long` error 帧,不提交本轮 messages。已经独立提交的工单按 §6.4 保留。

### 5.4 同步 Session 与取消语义

每个同步 DB 操作作为一个整体放入 `asyncio.to_thread`;`Session` 必须在线程函数内部创建、提交/回滚并关闭,不能跨线程或跨工具调用共享。

最终 `commit_turn` 在线程任务中执行并由 `asyncio.shield` 保护。若客户端在提交期间取消,服务必须等待该线程任务结束后才释放 session 锁: 事务成功则 turn 已提交,即使客户端没收到 `[DONE]`;事务失败则没有任何本轮 message。不能在后台提交尚未结束时释放锁或宣称未提交。

### 5.5 会话锁

锁不能只在本进程创建新 session 时建立,因为 DB 会话会跨应用重启存在。`ChatService` 使用 `SessionLockRegistry.acquire(session_id)` 懒创建锁;registry 对持有者和等待者计数,最后一个使用者释放后删除条目。同一 session 的“读历史 -> 两次模型调用/工具执行 -> 提交”全程串行,不同 session 可并发。该保证只覆盖本章单进程部署。

## 6. 工具模块

### 6.1 五个业务工具

| 工具 | 参数 | 行为 |
|---|---|---|
| `query_order` | `order_id: str` | 随机生成订单状态、金额、商品名、下单时间;返回 JSON 字符串 |
| `query_product` | `product_name: str` | 随机生成价格、库存、是否在售;返回 JSON 字符串 |
| `query_logistics` | `order_id: str` | 随机生成物流公司、运单号、2-4 条物流节点;返回 JSON 字符串 |
| `query_faq` | `keyword: str` | 参数化 SQL 查询 question/answer,按 id 升序取前 5 条;`%` 和 `_` 按普通字符转义;无结果返回明确文本 |
| `create_ticket` | `description: str, ticket_type: Literal['售后','投诉','咨询']` | 创建工单并把 conversation 状态更新为“已转人工”;返回工单号 |

工具参数去除首尾空白后必须非空,并通过 `Annotated` + Pydantic 固定长度: `order_id` 1-64 字符、`product_name` 1-128 字符、`keyword` 1-128 字符、`description` 1-2000 字符。`ticket_type` 必须使用 `Literal` 或显式枚举 schema,不能写成无约束的 `str`。`conversation_id` 由每轮工具工厂闭包注入,不暴露给模型。

三个 mock 工具不接真实接口、不建表;随机结果每次调用可不同,单次调用内自洽即可。`query_faq` 与 `create_ticket` 每次调用各在线程内打开并关闭自己的 Session。

`build_tools(session_factory, conversation_id) -> list[BaseTool]` 每轮构造绑定该会话的工具实例。

### 6.2 注册与协议校验

`ToolRegistry` 有序注册工具,name 唯一,重复注册立即报错;对外提供按名查找和 `bind_tools` 视图。编排与执行器共享同一 registry。

完整聚合第一次模型响应后再读取 `tool_calls`:

- 第一次响应 `finish_reason=length`: 发送 `output_too_long` error 帧,不执行任何工具,不提交 turn
- 超过 `MAX_TOOL_CALLS_PER_TURN`、同一响应多次申请 `create_ticket`、存在 `invalid_tool_calls`、call id 重复/缺失/过长或结构无法解析: 发送 `invalid_tool_call` error 帧,不执行任何工具,不提交 turn
- 工具名未注册: 为该 call 生成 `unknown_tool` error ToolMessage,继续第二次模型调用
- 缺字段、类型错误、枚举越界: 生成 `invalid_args` error ToolMessage,不重试
- 每个合法或业务失败的 parsed tool call 都必须产生恰好一条同 id 的 ToolMessage

### 6.3 只读工具超时与重试

`query_order / query_product / query_logistics / query_faq` 使用统一执行策略:

- 单次尝试由 `asyncio.wait_for` 限制 `TOOL_TIMEOUT_SECONDS`(默认 5 秒)
- 超时和 DB 连接/网络异常最多额外重试 `TOOL_MAX_RETRIES` 次(默认 2,即最多 3 次尝试);第 n 次重试前等待 `0.5 * 2^(n-1)` 秒
- 参数错误、未知工具、SQL 约束等业务错误不重试
- 重试耗尽后返回 `status="error"` 的 ToolMessage;公开内容只含稳定错误码和脱敏摘要
- 被 `wait_for` 取消的线程可能继续运行,因此这套策略只允许用于无副作用的查询工具;每次尝试均持有独立 Session并最终关闭

每条工具结果 envelope 的序列化长度不超过 `MAX_TOOL_RESULT_CHARS`;超出时在保持合法 JSON envelope 的前提下截断 `content` 并标记 `truncated=true`。

执行器产出内存 `ToolExecutionRecord(name, args, ok, duration_ms, retry_count, error_type)` 供 SSE 和脱敏日志使用。四表 DDL 没有执行记录表,本章不持久化 duration/retry_count。

### 6.4 `create_ticket` 的独立事务

`create_ticket` 是写工具,不使用 `wait_for` 包裹后台线程,也不做应用层自动重试。数据库连接、读写和锁等待使用 engine/driver 的有限超时。工具线程必须执行到明确的 commit 或 rollback 后,执行器才能返回结果并释放会话锁。

工单号格式为 `T` + UTC 时间 `YYYYMMDDHHMMSS` + 12 位随机十六进制,总长 27。一次事务同时插入 ticket 并把 conversation 状态改为“已转人工”;两者同成同败。主键冲突或其他约束异常回滚且不重试。

工单事务与本轮 messages 事务相互独立。工单成功后,即使客户端取消、第二次模型调用失败或最终 messages 提交失败,工单与“已转人工”状态仍保留。对应 `tool_end(ok=true)` 只表示工单事务成功,不表示聊天 turn 已提交。

## 7. 单轮编排与 SSE 契约

`ChatService.stream` 流程:

1. 校验 `user_id / session_id / message`,创建或核对 session,取得 session 锁;读取已提交历史并构造第一次调用消息
2. 调用 `model.bind_tools(registry.tools).astream(messages)`,边接收边聚合 `AIMessageChunk`;文本 chunk 立即作为 delta 转发,tool call chunks 只聚合不对外暴露
3. 聚合完成后:
   - 无 tool calls: 校验第一次回答并提交 user + assistant,发送 `[DONE]`
   - 有 tool calls: 保存完整 `AIMessage(content + tool_calls)`;按顺序为每个 call 发送 ToolStartEvent、执行工具、发送 ToolEndEvent,构造对应 ToolMessage
4. 重新执行 §5.3 的第二次输入预算;第二次调用绑定工具以承接结构化通道,但不再执行任何 tool_calls;文本 chunk 继续转发为 delta;若流结束时无任何文本(模型只给了 tool_calls),推送固定兜底话术并作为最终回答;落库时剥离未执行的 tool_calls
5. 校验两次调用累计对用户可见文本、第二次 finish_reason 与最终回答;通过后一次性提交 user + assistant(tool_calls) + tool messages + final assistant,再发送 `[DONE]`

如果第一次响应同时包含文本和工具调用,前置文本已经流给用户并保存在 assistant(tool_calls) 行的 content 中;第二次回答接在同一前端回答气泡中。`MAX_MESSAGE_CHARS` 同时限制单条持久化消息和本轮累计可见 assistant 文本;超限发 error 帧且不提交 messages。

SSE 新增帧:

```text
data: {"type":"tool_start","tool_call_id":"call_abc","name":"query_logistics","args":{"order_id":"1001"}}\n\n
data: {"type":"tool_end","tool_call_id":"call_abc","name":"query_logistics","ok":true,"summary":"结果前 80 个 Unicode 字符"}\n\n
```

- session 始终是首帧;tool_start 在实际工具函数运行前发送;tool_end 在该调用得出最终结果后发送
- `args` 是模型给出的已解析 JSON 参数;`summary` 最多 80 个 Unicode 字符;失败时只给稳定、脱敏摘要
- 无工具时顺序为 `session -> delta(first)* -> [DONE]`
- 有工具时顺序为 `session -> delta(first)* -> (tool_start -> tool_end)* -> delta(second)* -> [DONE]`;第一次调用没有文本时自然从 tool_start 开始
- error 帧可以出现在已发送的 delta/tool 帧之后;error 后不发 `[DONE]`
- 前端在当前 assistant 气泡中为每个 `tool_call_id` 建一个徽章: start 为执行中,end 后变成功/失败;前置 delta 不影响徽章挂载
- 除 §6.4 已独立提交的工单外,任何 messages 提交前错误或取消都不保存半个 turn

## 8. System Prompt 增补

在 Ch01 五条约束后追加:

- 查询订单/商品/物流/常见问题时优先使用对应工具,不得编造查询结果
- 工具返回错误或无结果时如实说明,并询问或引导用户转人工;不得声称已经完成转人工
- 用户明确要求人工或问题超出工具能力时调用 `create_ticket`;只有收到成功结果后才能说“已转人工”并告知工单号
- 订单/商品/物流为演示数据的事实不主动强调,但不得承诺其与现实系统一致

## 9. 配置、应用工厂与 Docker

`.env` 新增:

```dotenv
DATABASE_URL=mysql+pymysql://wayhelp:wayhelp@127.0.0.1:3306/wayhelp?charset=utf8mb4
TEST_ADMIN_DATABASE_URL=mysql+pymysql://root:root-password@127.0.0.1:3306/mysql?charset=utf8mb4
TEST_DATABASE_URL=mysql+pymysql://root:root-password@127.0.0.1:3306/wayhelp_test?charset=utf8mb4
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
MAX_TOOL_CALLS_PER_TURN=5
MAX_TOOL_RESULT_CHARS=4000
```

`DATABASE_URL` 在 Ch02 Settings 中为非空必填。测试 Settings 使用无真实凭据的占位 URL,并注入 `AppRuntime`;只有生产 runtime 构造路径才连接和 ping `DATABASE_URL`。`TOOL_TIMEOUT_SECONDS / MAX_TOOL_CALLS_PER_TURN` 为正数,`MAX_TOOL_RESULT_CHARS >= 256`,`TOOL_MAX_RETRIES` 为大于等于 0 的整数。移除 Ch01 对 `MAX_MESSAGES_PER_SESSION` 必须为偶数的限制。

`AppRuntime` 只暴露 `store` 与 `toolset_factory` 两个依赖。`create_app(settings=None, model=None, runtime=None)` 在 runtime 为空时创建 engine、ping、DbSessionStore 和真实工具工厂;测试注入 runtime 时不触碰生产 DB。Ch01 测试保留原行为断言,统一修改测试 fixture/工厂调用来注入 InMemory Store 与空工具集;不要求测试源码逐字不动。

应用以 factory 模式启动,避免导入模块时创建生产连接:

```bash
uv run uvicorn app.main:create_app --factory
```

`docker-compose.yml` 使用 mysql:8.4,挂载 `./db/init` 到 `/docker-entrypoint-initdb.d/`,配置 `mysqladmin ping` 健康检查、`MYSQL_ROOT_HOST=%` 并映射 `3306:3306`。测试 fixture 先用 `TEST_ADMIN_DATABASE_URL` 连接已有的 `mysql` 库并创建/删除 `wayhelp_test`,再用 `TEST_DATABASE_URL` 执行同一 `01-ddl.sql` 和业务测试;每个测试清空数据。应用账号不需要 CREATE DATABASE 权限。若本机端口改为 3307,同时修改三个 URL 示例。

## 10. 错误处理与公开语义

- DB ping 失败: 应用创建失败,错误文本脱敏
- `user_id/session_id` 格式非法: 422;session 不存在或 user 不匹配: 404
- 第一次模型响应含非法、重复建单或过多 tool calls: `invalid_tool_call` error 帧,不执行工具
- 工具名未知、参数非法、查询超时/连接失败: error ToolMessage 回灌,流程继续第二次模型调用
- 工具失败时只引导转人工;未成功创建 ticket 时不得回答“已转人工”
- `create_ticket` 事务成功后发生后续失败: ticket 和 conversation 状态保留,messages 不提交
- 第二次调用输入仍超预算: `tool_context_too_long` error 帧
- 运行期 DB 会话读写异常: SSE 开始后用 error 帧收尾;开始前返回 500
- 最终 `commit_turn` 期间取消: 等事务结束再释放锁;是否保留以数据库事务结果为准
- 所有公开错误不含 SQL、连接串、堆栈或 SDK 原始异常;日志只记异常类型、稳定错误码和必要标识

创建 session 后首轮失败会留下空 conversation,沿用 Ch01 语义;客户端已收到 session 帧时可以用同一 id 重试。

## 11. 测试与验证

**自动测试(pytest):**

- 配置/runtime: 生产路径缺 DB 或 ping 失败会启动失败;注入 runtime 不连接 DB;`TOOL_MAX_RETRIES=0` 合法;`MAX_MESSAGES_PER_SESSION` 可为奇数
- 用户/session: 浏览器 user_id 请求字段;数字与 UUID 两种 session 格式;user 不匹配返回 404;DB 会话模拟应用重启后仍能懒创建锁并继续对话
- ORM/store: 四表 CRUD;DbSessionStore 四方法;完整 turn 原子提交/回滚;工具 envelope 落库与重建;损坏或孤儿工具轨迹被拒绝;按完整 turn 裁剪
- 线程/取消: Session 在线程内创建关闭;取消期间 commit 完成前不释放锁;commit 成功后断开仍保留 turn
- 五工具: FAQ 命中与“邮费”漏召回;LIKE wildcard 按字面匹配;`create_ticket` 的 Literal schema 拒绝越界值;建单与会话状态同事务;三个 mock 返回可解析 JSON
- 执行器: 只读工具超时重试次数与退避;业务异常不重试;未知工具生成同 id ToolMessage;结果截断仍是合法 envelope;写工具不经过 `wait_for`/自动重试
- 编排: 无工具路径只有一次模型调用;有工具路径第二次绑定工具但不执行其返回的 tool_calls(模型仍试图调用则丢弃并以固定话术兜底);多个工具串行且一一回灌;非法/过多 tool calls 不执行;第一次前置文本 + 工具帧 + 第二次文本顺序;第二次预算不足受控失败
- 副作用分界: 工单成功后模拟第二次模型失败/客户端取消,断言 ticket 与“已转人工”保留而 messages 不提交
- SSE/页面: 新帧 JSON 转义、80 字符摘要、徽章状态;Ch01 的 session/delta/error/[DONE]、并发、取消和结构化提取行为回归

**评估/标注类:**

- SQL 直接断言“退货政策”命中、“邮费”漏召回
- 真实模型人工核对工具选择、失败措辞和只有建单成功后才声称已转人工

**真实模型浏览器验收:**

1. 问“订单 1001 的物流到哪了” -> 选择 query_logistics,显示成功徽章,按返回结果作答
2. 问“退货政策是什么” -> query_faq 命中并作答
3. 问“邮费是多少” -> query_faq 未找到,如实说明并引导转人工,记录漏召回
4. 问“帮我转人工处理” -> create_ticket 成功,显示工单号;SQL 抽查 ticket 存在且 conversation 为“已转人工”

**验证顺序:**

```bash
docker compose up -d
docker compose ps
uv run pytest
uv run uvicorn app.main:create_app --factory
```

浏览器打开 `/`;验收完成后可用 `docker compose down` 停止容器。

## 12. 验收标准映射

| 目标 | 验收证据 |
|---|---|
| 模型自选工具 + 徽章 | 编排/SSE/页面测试;浏览器验收 1 |
| query_faq 命中与预期漏召回 | 工具/SQL 测试;浏览器验收 2、3 |
| 聊天记录落表 | store 测试断言 `3 + N` 条工具 turn 消息;验收后 SQL 抽查 |
| 单轮收敛 | 编排测试断言第二次调用绑定的 tool_calls 不被执行、模型仍试图调用时以固定话术兜底 |
| 超时重试不复制写副作用 | 执行器测试断言只读重试、写工具不自动重试 |
| 工单与转人工状态一致 | 事务测试 + 后续失败测试 + 浏览器验收 4 |
| 持久会话并发安全 | 重启后锁懒创建测试 + 同 session 串行测试 |
| 两次调用均满足预算 | 工具结果截断和第二次预算测试 |

## 13. API 核对来源

2026-09-10 经 Context7 与仓库锁定环境(`langchain-core==1.6.2`)核对:

- [LangChain 模型与工具调用](https://docs.langchain.com/oss/python/langchain/models): `bind_tools`;流式 `AIMessageChunk` 可相加聚合;完整 chunk 的 `tool_calls` 提供 name/args/id
- [LangChain streaming](https://docs.langchain.com/oss/python/langchain/streaming): 流中可能分别出现文本与 `tool_call_chunks`,必须聚合后读取完整 parsed tool calls
- [LangChain ToolMessage](https://docs.langchain.com/oss/python/langchain/messages): `tool_call_id` 必须与 AIMessage call id 对应;当前版本 `status` 支持 `success/error`;以完整 ToolCall 调用工具可直接得到 ToolMessage
- [SQLAlchemy Session 基础](https://docs.sqlalchemy.org/en/20/orm/session_basics.html): Session 是可变、有状态对象,不能跨线程或并发任务共享;每个线程使用独立 Session
- [SQLAlchemy 事务管理](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html): context manager 管理 commit/rollback/close 生命周期
- [SQLAlchemy MySQL/PyMySQL](https://docs.sqlalchemy.org/en/20/dialects/mysql.html): `mysql+pymysql://user:pass@host/db?charset=utf8mb4`

实现计划阶段仍须以最终 `uv.lock` 复核导入路径和方法签名;若加依赖导致 LangChain 主版本变化,先更新本节与相应契约再实现。
