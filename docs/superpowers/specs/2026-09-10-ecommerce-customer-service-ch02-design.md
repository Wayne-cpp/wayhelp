# 电商智能客服 Ch02:Function Calling 工具链 - 设计文档

日期:2026-09-10
状态:定稿(用户手写 DDL 已并入,见 §4)

## 1. 目标与范围

在 Ch01 纯对话系统上接入 **Function Calling 工具链**:用户在 SSE 聊天页问一句,模型自己决定调哪个工具,工具结果回灌后组织最终回答,最终回答仍逐 token 流式输出。

交付五项能力:

1. 项目基建:FastAPI + SQLAlchemy 分层骨架;Docker 起 MySQL;四张表(faq / conversations / messages / tickets)灌测试数据
2. 五个 LangChain `@tool` 业务工具:`query_order` / `query_product` / `query_logistics`(工具内随机生成演示数据,不接真实接口、不建表)、`query_faq`(SQL LIKE 查 faq 表)、`create_ticket`(写 tickets 表)
3. 工具基础设施:注册管理、参数 Schema 校验、执行错误处理、超时重试;工具结果回灌模型
4. 工具链接入现有 SSE 聊天:模型定工具 → 执行 → 回灌收敛;工具执行段先推状态帧;聊天记录(含工具调用与结果)落 conversations/messages 表;聊天页气泡显示工具轨迹徽章
5. 只做单轮调用:模型调一次工具就收敛

技术栈(定死):Python 3.12(uv)+ FastAPI + SQLAlchemy 2.x + MySQL 8(Docker)+ LangChain(`langchain-core` / `langchain-openai`)`@tool` 装饰器。依赖版本写入并提交 `uv.lock`。

明确不做:多轮自动循环的 Agent Loop;向量检索、RAG;真实电商/物流系统对接;身份认证。Ch02 仍只支持单进程运行。

## 2. 架构(方案 A:手写单轮编排)

在 Ch01 四层(路由 / ChatService / chain / Store)上增加两层:

- **数据层**(`app/db.py` + `app/models.py`):SQLAlchemy 2.0 同步 engine + `sessionmaker`;ORM 模型与用户手写 DDL 逐字段对齐。同步 Session 经 `asyncio.to_thread` 包进异步流程,不引入 async 驱动
- **工具层**(`app/tools/`):`business.py` 定义五个 `@tool` 与每轮工具工厂;`executor.py` 提供 ToolRegistry 注册管理与超时重试执行

单轮编排手写实现(不引入 LangGraph / AgentExecutor):第一次 `bind_tools` 调用聚合 tool_calls,有则执行回灌,第二次调用不绑工具物理保证收敛。落选方案:B(LangGraph `create_react_agent`,为单轮引入整个 LangGraph 不值,且 Agent Loop 恰是本章排除项)、C(LangChain `AgentExecutor` 限轮,官方已边缘化)。

聊天页工具徽章按用户指定走 Vibe Coding,不套 brainstorm/TDD/code review 流程;但其依赖的 SSE 帧契约(§7)仍属本 spec,后端实现照常 TDD。

## 3. 项目结构(在 Ch01 基础上新增)

```
docker-compose.yml           # mysql:8.4,挂载 db/init,健康检查,端口见 §9
db/init/01-ddl.sql           # 用户手写 ch02-ddl.sql 原样落此(建表唯一事实源)
db/init/02-seed.sql          # faq 测试数据(本 spec §4)
app/
  db.py                      # create_engine + sessionmaker + 启动 ping
  models.py                  # Faq / Conversation / Message / Ticket ORM
  store_db.py                # DbSessionStore:SessionStore 协议的 MySQL 实现
  tools/business.py          # 五个 @tool + build_tools(session_factory, conversation_id)
  tools/executor.py          # ToolRegistry + 超时重试执行器
  chains/tool_chat_chain.py  # 单轮编排:两次模型调用的消息组装
  services/chat_service.py   # 扩展:工具事件、整轮消息序列提交
  routers/chat.py            # 扩展:tool_start / tool_end SSE 帧
  static/chat.html           # 工具徽章(Vibe Coding)
tests/                       # 新增 DB/工具/编排测试;Ch01 的 59 个测试不动
dev-notes/ch02.md            # 过程留痕
```

新增依赖:`sqlalchemy>=2.0`、`pymysql`、`cryptography`(MySQL 8 默认认证插件需要)。不引入 Alembic:DDL 文件经 `/docker-entrypoint-initdb.d/` 首启执行,是建表唯一事实源,ORM 模型须与之逐字段一致;后续章节改表再评估迁移工具。

## 4. 数据库与表契约

以用户手写 `db/init/01-ddl.sql` 为准(本章不得改动其字段定义),要点:

- **conversations**:`id` BIGINT UNSIGNED 自增主键;`user_id` VARCHAR(64);`status` ENUM('进行中','已转人工','已结束') 默认 '进行中';`created_at` / `updated_at`
- **messages**:`id` 自增主键;`conversation_id` 外键;`role` ENUM('user','assistant','tool');`content` TEXT 可空(assistant 纯工具调用时为空);`tool_calls` JSON 可空(assistant 行的工具调用申请单);`tool_call_id` VARCHAR(64) 可空(tool 行对号入座);`created_at`
- **faq**:`id` 自增主键;`question` VARCHAR(512);`answer` TEXT;`category` VARCHAR(64)
- **tickets**:`ticket_no` VARCHAR(32) 主键;`conversation_id` 外键;`description` TEXT;`ticket_type` ENUM('售后','投诉','咨询');`status` ENUM('待处理','已处理') 默认 '待处理';`created_at`

**会话 id 协议变更**:Ch01 的 `session_id` 是 UUID 字符串;Ch02 的 DB 会话使用自增 id,API 中以**十进制数字字符串**透传。`ChatStreamRequest.session_id` 校验由 UUID 放宽为「非空数字字符串或 null」;前端原样透传无感知。内存 Store 继续用 UUID,两实现的 id 形态各自内部自洽。

**seed(02-seed.sql)**:faq ≥ 8 条,分类覆盖 退货/换货/发票/运费/账户 等;必须含可命中「退货政策」的条目(验收 2);**全表不出现「邮费」二字**(运费条目用「运费」措辞),保证验收 3 的 LIKE 漏召回如实发生,该漏召回是预期结果,记入 dev-notes 留待下一章升级。seed 只灌 faq;conversations/messages/tickets 由运行时产生,不预置。

**启动健康检查**:`create_app` 时对 DB 做 `SELECT 1` ping,失败则应用创建立即失败并指明是数据库连接问题(脱敏,不含密码)。运行期 DB 异常按 §10 处理。

## 5. 会话存储:DbSessionStore 与协议扩展

Ch01 的 `SessionStore` 协议(`create / exists / snapshot / commit_turn`)保留,`InMemorySessionStore` 及其测试不动。Ch02 扩展协议语义:

- `commit_turn` 由「提交一对 user/assistant」扩展为「**一次性提交本轮完整消息序列**」:user 消息、可选的 assistant 工具申请消息(content 可空 + tool_calls JSON)、每条 tool 结果消息(content + tool_call_id)、最终 assistant 消息。签名改为接收有序消息列表;`StoredMessage` 扩展可选字段 `tool_calls`、`tool_call_id`。内存实现同步扩展,保存扩展字段,与 DB 实现保持一致的 snapshot/commit 语义
- `snapshot` 返回的历史须能重建模型上下文:DB 实现按 created_at/id 升序读取,role=tool 的行重建为 `ToolMessage`,带 tool_calls 的 assistant 行重建为 `AIMessage(tool_calls=...)`,使下一轮模型能看到完整工具轨迹。裁剪策略沿用 Ch01(`trim_messages` 参数不变),但 tool 消息不参与裁剪起点约束——若裁剪把 tool_calls 申请与其结果拆散(孤儿 ToolMessage 会被上游拒绝),须把挂起的 tool 组整体丢弃;实现为:裁剪后以「assistant(tool_calls) 与其全部 ToolMessage 同在或同去」为校验规则,违反则从头部继续丢弃直至合法
- 容量语义:Ch01 的 MAX_SESSIONS / MAX_MESSAGES_PER_SESSION 上限只约束内存实现;DB 实现不设进程内上限,本章不回收会话
- 生产路径(`create_app` 默认)使用 DbSessionStore;内存实现继续服务现有单测与无 DB 的本地起服场景由测试注入选择

**原子提交语义沿用 Ch01**:只在最终回答校验通过后一次性提交整个 turn;提交前异常/取消不留半个 turn(包括不产生任何 messages 行);所有退出路径释放 session 锁。锁表沿用 Ch01 的进程内 `asyncio.Lock` 字典,键为会话 id 字符串。

## 6. 工具层

### 6.1 五个业务工具(`@tool` 装饰器)

| 工具 | 参数 | 行为 |
|---|---|---|
| `query_order` | `order_id: str` | 随机生成:订单状态(待付款/待发货/已发货/已完成)、金额、商品名、下单时间;返回 JSON 字符串 |
| `query_product` | `product_name: str` | 随机生成:价格、库存、是否在售;返回 JSON 字符串 |
| `query_logistics` | `order_id: str` | 随机生成:物流公司、运单号、2-4 条物流节点(时间+描述);返回 JSON 字符串 |
| `query_faq` | `keyword: str` | `SELECT ... WHERE question LIKE %keyword% OR answer LIKE %keyword%` 查 faq 表,按 id 升序取前 5 条;无结果返回明确的「未找到」文本(不是空串) |
| `create_ticket` | `description: str, ticket_type: str` | 写 tickets 表,返回工单号;`ticket_type` 仅允许 '售后'/'投诉'/'咨询'(schema 枚举约束);`conversation_id` 由工具工厂闭包注入,**不对模型暴露** |

三个 mock 工具不接真实接口、不建表;随机结果每次调用可不同,单次调用内自洽即可(如同一单号在工具描述里说明「演示数据」由 system prompt 层面不承诺真实性——见 §8)。

`build_tools(session_factory, conversation_id) -> list[BaseTool]` 每轮请求构造一批绑定该会话的工具实例;query_faq / create_ticket 经闭包拿到 session_factory 与 conversation_id,三个 mock 工具无 DB 依赖。

### 6.2 注册管理

`ToolRegistry`:有序注册(name 唯一,重复注册立即报错)、按名查找、`bind_tools` 所需的工具列表视图。编排层与执行器共用同一 Registry,不允许两处各持一份工具清单。

### 6.3 参数校验

`@tool` 由函数签名与 docstring 生成 args schema(pydantic);模型给出的非法参数(缺字段、类型错、枚举越界)在执行入口被校验拦截,生成 `status=error` 的 ToolMessage 回灌(内容含「参数错误」与期望格式摘要),**不重试**。模型据此在最终回答中说明或自行纠正后给出回答——本章只允许这一次工具调用,不给它二次调用机会。

### 6.4 超时与重试

执行器语义:

- 单次执行超时 `TOOL_TIMEOUT_SECONDS`(默认 5s,`asyncio.wait_for` 包裹;同步 DB 操作用 `asyncio.to_thread` 落入线程)
- 可重试异常(超时、DB 连接/网络类)最多重试 `TOOL_MAX_RETRIES` 次(默认 2,即共 3 次尝试),退避 0.5s、1s;业务异常(参数校验、SQL 约束冲突等)不重试
- 重试耗尽或业务异常 → `status=error` ToolMessage 回灌,内容脱敏(不含 SQL 文本、连接串、堆栈);编排继续走第二次模型调用,由模型组织「查询失败,已为你转人工」式回答,不得让工具失败直接终止整条流
- 执行器对每次调用产出执行记录(name、args、ok、耗时、重试次数)供落库与日志;日志脱敏只记异常类型

## 7. 单轮编排与 SSE 契约

`ChatService.stream` 扩展后的流程(prepare 的锁/裁剪/预算语义不变):

1. 读历史快照重建消息(含工具轨迹,§5),拼 system + history + 当前 human
2. **第一次调用**:`model.bind_tools(registry.tools).astream(messages)`,逐 chunk 聚合(`AIMessageChunk` 相加合并 tool_call_chunks)
   - 若聚合结果**无 tool_calls**:其 content 增量已经在流式到达,直接作为 delta 事件转发(纯聊天路径与 Ch01 行为一致,不多一次调用)
   - 若**有 tool_calls**:逐个 tool_call:发 ToolStartEvent → 执行器执行(§6.4)→ 发 ToolEndEvent → 拼 `AIMessage(tool_calls=...)` + 每条 `ToolMessage`。边界约定:模型发 tool_calls 时通常不带文本;若工具调用出现前已有文本增量被转发,不撤回已发增量,工具流程照常继续
3. **第二次调用**:`model.astream(完整消息含工具结果)`,**不绑工具**(物理保证单轮收敛),content 增量逐 token 转发为 delta
4. 最终回答沿用 Ch01 校验(字符上限、finish_reason=length、空响应);通过后 `commit_turn` 一次性提交完整消息序列(§5),发 `[DONE]`

并发安全:同一 session 串行由既有锁保证;tool_calls 中多个工具按模型给出的顺序逐个执行,不并行(本章简化,保持日志/帧顺序确定)。

SSE 帧契约(向后兼容,session / delta / error / [DONE] 不变),新增:

```text
data: {"type":"tool_start","tool_call_id":"call_abc","name":"query_logistics","args":{"order_id":"1001"}}\n\n
data: {"type":"tool_end","tool_call_id":"call_abc","name":"query_logistics","ok":true,"summary":"...结果前 80 字符"}\n\n
```

- tool_start 在对应工具执行前发送,tool_end 在执行结束后发送;`ok=false` 时 summary 为脱敏错误摘要
- args 原样转发模型给出的参数(已是 JSON);summary 截断至 80 字符
- 帧顺序固定:session → (tool_start/tool_end)×N → delta×M → [DONE];任何 error 帧后不再发 [DONE]
- Ch01 的原子提交分界不变:提交前异常/取消不入库;tool 帧已发不代表已提交

## 8. System Prompt 增补

在 Ch01 的五条约束上增补工具使用规则(追加,不改写原文):

- 有工具可用时,查订单/商品/物流/常见问题优先调工具,不得编造查询结果
- 工具返回错误或无结果时,如实说明并引导转人工,不得假装查到
- 用户明确要求人工、或问题超出工具能力时,调 `create_ticket`;创建后告知工单号
- 订单/商品/物流为演示数据的事实不对用户强调(演示系统约定),但也不得承诺其与现实一致

## 9. 配置与 Docker

`.env` 新增:

```dotenv
DATABASE_URL=mysql+pymysql://wayhelp:wayhelp@127.0.0.1:3306/wayhelp?charset=utf8mb4
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
```

`DATABASE_URL` 必填(Ch02 起应用创建依赖 DB ping);`.env.example` 同步更新占位值。两个工具配置为正数且设默认值。

`docker-compose.yml`:mysql:8.4 容器,root 密码与 wayhelp 库/用户经环境变量固定在 compose 文件内(本地开发演示,非生产密钥管理);`./db/init` 挂载至 `/docker-entrypoint-initdb.d/`;`healthcheck: mysqladmin ping`;端口映射 `3306:3306`(若本机占用则改 `3307:3306` 并同步 .env.example 注释)。测试库 `wayhelp_test` 由测试 fixture 在同一容器内创建,不占 initdb。

## 10. 错误处理(在 Ch01 基础上新增)

- DB ping 失败 → 应用创建失败,指明数据库连接问题(脱敏)
- 工具参数非法 → error ToolMessage 回灌,流程继续(§6.3)
- 工具超时/连接失败且重试耗尽 → error ToolMessage 回灌,流程继续(§6.4)
- 运行期 DB 会话读写异常(非工具路径) → 按上游错误同等对待:SSE 已开始则 error 帧收尾不提交;开始前 500
- 会话 id 格式非法(非数字串) → 422;不存在 → 404(沿用 Ch01 语义)
- 所有公开错误不含 SQL、连接串、堆栈;日志只记异常类型

## 11. 测试与验证(TDD 变体)

**可单测部分(pytest,DB 用 Docker MySQL 测试库):**

- 测试基建:fixture 负责 `wayhelp_test` 库的创建、四表结构(执行 01-ddl.sql)与逐测试清表;测试需要 Docker 在线,不可用时该组测试显式报错退出而非跳过静默
- ORM/store:四表 CRUD;DbSessionStore 的 create/exists/snapshot/commit_turn;工具轨迹(assistant tool_calls + tool 结果)的落库与重建;裁剪后孤儿 tool 组整体丢弃
- 五工具:query_faq 命中与漏召回(「邮费」查不到「运费」条目,断言返回「未找到」文本)、create_ticket 落库且 ticket_type 枚举被 schema 拦截、三个 mock 工具返回可解析 JSON
- 执行器:超时重试次数与退避、业务异常不重试、error ToolMessage 内容脱敏
- 编排:fake model 发 tool_calls → 断言 SSE 帧序列(session → tool_start → tool_end → delta → [DONE])与落库消息序列(user/assistant+tool_calls/tool/assistant);无 tool_calls 时帧序列与 Ch01 一致;第二次调用断言未绑工具
- SSE 路由:新帧 JSON 转义与顺序;Ch01 既有契约回归(59 个测试不动)

**评估/标注类(替代 TDD 的部分):**

- faq seed 即标注样例:验收 2 的「退货政策」必须命中、验收 3 的「邮费」必须漏召回,用 SQL 直接断言这两条事实(脚本或测试均可)
- System Prompt 增补的工具规则,真实模型验收时人工核对(浏览器三条)

**真实模型验收(浏览器,Vibe Coding 页面):**

1. 问「订单 1001 的物流到哪了」→ 模型选中 query_logistics,气泡带工具徽章,按返回结果作答
2. 问「退货政策是什么」→ query_faq 命中并作答
3. 问「邮费是多少」→ query_faq 查不到,漏召回如实发生,记录进 dev-notes 留待下一章

**验证命令:** `uv run pytest`;`docker compose up -d` 后起服 `uv run uvicorn app.main:app`,浏览器打开 `/`。

## 12. 验收标准映射

| 目标 | 验收证据 |
|---|---|
| 模型自选工具 + 徽章 | 编排测试断言帧序列;浏览器验收 1 |
| query_faq 命中 | 工具单测 + 浏览器验收 2 |
| 关键词漏召回(预期) | 工具单测断言「未找到」+ 浏览器验收 3,归档 dev-notes |
| 聊天记录落表 | store 测试断言四行消息序列;验收后 SQL 抽查 |
| 单轮收敛 | 编排测试断言第二次调用未绑工具 |
| 超时重试/错误处理 | 执行器测试 |

## 13. API 核对来源

2026-09-10 经 Context7 核对(实现时结合 `uv.lock` 选定版本再次确认):

- [LangChain 工具调用:bind_tools 与 tool_calls 结构](https://docs.langchain.com/oss/python/langchain/models)(`@tool` 定义 → `bind_tools` → `response.tool_calls` 为含 name/args/id 的字典列表;`tool.invoke(tool_call)` 直接返回 ToolMessage;手动执行循环模式)
- [LangChain ToolMessage](https://docs.langchain.com/oss/python/langchain/messages)(content / tool_call_id / name 三要素,tool_call_id 必须与 AIMessage 中的调用 id 一致)
- [SQLAlchemy 2.0 ORM:DeclarativeBase + mapped_column](https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html) 与 [sessionmaker](https://docs.sqlalchemy.org/en/20/orm/session_api.html)
- [SQLAlchemy MySQL/PyMySQL 连接串](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)(`mysql+pymysql://user:pass@host/db?charset=utf8mb4`)

待实现计划阶段补充核对:`astream` 下 `AIMessageChunk` 的 tool_call_chunks 聚合法;`@tool` 在 langchain-core 1.x 的导入路径与 args_schema 行为;`asyncio.to_thread` 与 SQLAlchemy Session 的线程边界(Session 不跨线程共享,线程内创建使用销毁)。
