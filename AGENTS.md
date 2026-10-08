# AGENTS.md — wayhelp

## 定位
电商智能客服演示项目:FastAPI + SSE 流式聊天,LangGraph Workflow 图编排聊天主链路(指代消解 understand → 八类意图分流 → 退款/售后确定性子流程(缺单 interrupt 挂起选单 + 扩写多查询强制检索 + 置信闸)→ 手写 ReAct 主力 Agent(business 分支按需 query_faq)),Milvus Lite 四策略混合检索(dense / bm25 / hybrid / hybrid_rerank)+ 硅基流动重排,deepseek 生成与裁判。分工:MySQL 是账本(知识块/会话/台账),Milvus 是货架(向量),SQLite(checkpoints.db)是图工作记忆。

## 怎么跑
- `uv sync`;`cp .env.example .env` 填 OPENAI_* 与 EMBEDDING_API_KEY
- `docker compose up -d`(MySQL,首启自动建表)→ `uv run uvicorn --factory app.main:create_app`
- 可选:起业务 MCP Server——`make mcp-logistics` / `make mcp-after-sales`(或 `uv run python -m app.mcp_servers.logistics_server` / `after_sales_server`),.env 配 MCP_LOGISTICS_URL / MCP_AFTER_SALES_URL 后图内可用物流/售后工具;每轮动态发现,主服不重启(见 README「MCP Server」节)
- `uv run pytest`(616 条,需 Docker 在线);页面:`/` 聊天、`/kb` 知识库、`/rag-eval` 评估
- 细节以 README.md 为准;逐章开发实录在 dev-notes/

## 目录与约定
- app/{routers,services,knowledge,prompts,tools,static,graph} 分层自明(app/graph/ 是 LangGraph 聊天图,ch05 搭骨架、ch06 正式版分流器:state / nodes / agent_node / builder / events);examples/ 是 ch05 热身留档脚本(bare_agent.py);knowledge_docs/ 是知识库唯一源;evals/ 评估与真模型 probe(烧额度,手跑不进 pytest);`sql/chXX-ddl.sql` 与 `db/init/` 同名 DDL 必须逐字节一致(有测试钉,ch08 对 = sql/ch08-ddl.sql ↔ db/init/06-ddl.sql)
- ch08 工具系统模块:`app/tools/catalog.py` 进程内工具目录 + MCP 能力策略(ToolSpec / ToolCatalog / TurnContext / classify_mcp_tool);`app/tools/executor.py` 统一执行引擎(校验/权限/归属/超时重试/格式化六步 + execute_confirmed 写确认 + 批次契约 + 审计出口);`app/tools/builtin/` 内置工具装饰器自登记(business / faq / ticket 三文件);`app/services/tool_audit.py` 审计落库(五终态,shield,失败只记日志);`app/services/mcp_gateway.py` MCP client 网关(每轮按 Server 独立超时发现 + call_tool 分发);`app/mcp_servers/` 两个独立进程业务 MCP Server(物流 / 售后,FastMCP streamable-http + 按 order_id 种子化的 mock 数据)
- Conventional Commits;改 prompt / 检索 / 知识库后跑全量 pytest 再交付

## 红线与坑(都踩过)
- 本机代理会 502:访问本机服务一律 `curl --noproxy '*'`
- 单 Uvicorn worker(内存作业注册表 + Milvus Lite 文件独占);pkill/pgrep -f 模式会匹配自身 shell 命令行——`[u]vicorn` 方括号技巧只在命令行不含该字面量时有效,复合命令里嵌着模式照样自匹配(已三次踩坑);进程判断以端口监听为准(`ss -tlnp | grep :8000`)
- 改 knowledge_docs 已导入文档:ingest_docs 拒绝覆盖(不覆盖旧知识),唯一路径 `POST /kb/api/rebuild`(服内执行);ingest/mine 等直写 `./data` 库的 CLI 须先停服
- 评估(make eval-rag 或 /rag-eval 重跑)约 25 分钟/轮、烧 API 额度;产物 rag_eval.json 与时间戳归档均不入库
- faith_cases 台账:status 取中文值 未解决/已解决/无需解决;「已解决」=确认真编造且已修(复发自动打回未解决),「无需解决」=裁判误判;`eval_id` 列存 case 编号(如 E40),不是 run_id
- 改 app/prompts/service.py 条款后:test_prompts.py 有契约断言(REFUSAL_ANSWER 逐字内嵌等),跑全量验证
- chat.html 走 Vibe Coding 例外(无 TDD),前端逻辑靠 test_chat_page 字符串断言 + 人工点验;dev-notes/ch04.md 含用户本人手写段落,改动前先 `git diff` 看清归属
- 建工单双通道(ch08):①动作端点 `POST /v1/chat/action`(create_ticket / create_refund 双动作契约,前端按钮点按即确认,校验会话/消息/订单归属、绑定 source_message_id、不改 conv.status);②图内 create_ticket 确认流——模型发起写调用 → agent_node park 成 pending_ticket(批次契约:一条回复最多一个写且必须在最后)→ ticket_preview interrupt 出预览卡 → resume decision(confirm/cancel)→ execute_confirmed 只信 checkpoint 快照参数 / deny_write 落权限拒绝;写超时按「已发未必未成」返回未知且绝不重试,tool_write_idempotency 收敛重复提交;退款单仍只走动作端点;「转人工」是纯前端模拟,后端零动作
- 工具面(ch08 即插即用):进程级 ToolCatalog(挂 app.state,启动扫描 `app/tools/builtin/` 装饰器自登记)——内置 query_order(绑定当前 user_id,非本人/不存在回 error JSON)/ query_product / 无参 query_faq(以本轮 resolved_query 检索,同轮缓存)/ create_ticket(写,park 确认流)+ suggest_options 伪工具;business 分支全量内置 + MCP 工具,refund 过闸分支只绑 query_order / query_product + MCP(不绑 faq 与写),clarify 分支无工具;query_logistics 内置已下线,物流查询由 MCP Server 接管;MCP 条目每轮发现现合成,与既有工具重名拒进面;旧 `build_tools`(business.py:120,捆绑 create_ticket)只剩 legacy 接线(AppRuntime.toolset_factory),**禁止回挂到图**
- MCP 权限只认本地能力策略(catalog.classify_mcp_tool):已知只读 Server 仅 {logistics, after_sales},schema 只放行两种形态——零参数 / 仅必填字符串 order_id;未知 Server、额外或写意味参数、形态不符一律拒进工具面(默认 deny),Server 自报 annotations 一律不看
- 审计双红线:tool_audit_logs 写失败不许拦工具执行(只打 error 日志不抛);审计表与写幂等表均不挂外键(会话删除不级联;审计行 conversation_id 可空,挂起前的调用也记账)
- `POST /v1/chat/resume` 契约:会话锁内自校验 pending interrupt 并按类型分派——order_selector 须 order_id(候选卡内 + 用户归属),ticket_preview 须 decision(confirm/cancel);归属错误 404 不泄露、卡片失效/字段不对/重放一律 409(幂等表有记录也不放行,它只供内部恢复);resume 传 langgraph `Command` 不传 dict;新消息或再次挂起后旧卡立即 409
- 订单 mock 服务(app/services/orders.py)是订单数据唯一入口:get_order / list_orders 按用户命名空间实例化,不存在与不属于该用户均返回 None;**不建订单表、不动任何 DDL**
- 意图 JSON 就 `{"intent","confidence"}` 两字段;needs_knowledge 维度已整体移除(state / prompt / parse / 日志 / probe 不留失效引用)
- SSE delta 三条件过滤:langgraph_node==main_agent + metadata.tags 含 chat_visible + AIMessageChunk;understand / classify / refund_scope / expand 等节点内部模型调用不外发
- 全量 pytest 的 DB 依赖:dbfixtures 用 .env 的 TEST_ADMIN_DATABASE_URL(root@127.0.0.1 TCP)建 wayhelp_test 库;若本机是原生 mysqld 而非 Docker,root 走 auth_socket 会 1698 拒连、fixture 入口直接 exit(3)——要么起 docker compose,要么给 root 开 TCP 密码授权(wayhelp 账号只有 wayhelp 库权限,不够)
- uvicorn 起服:shell 有代理变量(http_proxy→7890 等)且无 no_proxy 时,milvus-lite 内嵌 gRPC 会被劫持进代理(GOAWAY)→ 启动契约探针误判 rebuild_required、检索整轮不可用;起服命令前加 `no_proxy=127.0.0.1,localhost`(curl 访问本机服务仍须另加 --noproxy '*')
- ch07 起账本行契约:messages 表只收 user/assistant 行,tool 结果行不落库(`sessions.validate_turn` 拒收,turn 中间只许带 tool_calls 的 assistant 行);工具调用轨迹仍在 assistant.tool_calls,工具结果只在 token 截断后进模型上下文
- 上下文 token 唯一估算尺 `app/services/token_budget.py`(estimate_tokens / estimate_messages / compute_budget 窗口倒推):一切预算、层1降级、摘要触发的判断只准用这把尺,不许另立字符数/条数启发式;演示配置验收数字 AVAIL=5667、层1=3966、层2=1701 有测试钉死
- 上下文锚点/摘要全部事务 CAS 单调前移(layer1_from 只前不回退;摘要段 CAS 追加 + in-flight 防重入 + 失败不挪锚点);绕过 store 的 CAS 方法直改 conversation_summaries / 上下文 meta 一律禁止
- log/app.log 由 main.py 的 FileHandler 幂等挂载(重复 create_app 不重复追加 handler);model_ctx(main_agent 每次模型调用:两层预算/条数/估算/逐条消息)与 history_ctx(understand/classify 分层历史读出:摘要/滑窗/估算)是单行 JSON 日志,grep 友好

## 当前状态(2026-10-08)
- ch08 完成:即插即用工具系统——统一 ToolCatalog(内置 `app/tools/builtin/` 装饰器自登记 + MCP 每轮发现现合成)→ 每轮 ToolFace 工具面 + 统一执行引擎 `app/tools/executor.py`(JSON Schema 校验/权限闸/归属把门/超时重试/格式化六步 + execute_confirmed 写确认 + 批次契约:一条回复最多一个写且必须在最后);tool_audit_logs 全量审计(五终态,写失败不拦执行、不挂外键);tool_write_idempotency 写幂等(超时「已发未必未成」,绝不重试);query_logistics 内置下线,物流/售后由两个独立进程 MCP Server 接管(`app/mcp_servers/`,FastMCP streamable-http,McpGateway 按 Server 独立超时发现,权限只认本地两形态能力策略,未知 Server 默认拒);create_ticket 走 ticket_preview 预览卡确认流(resume decision confirm/cancel,只信 checkpoint 快照参数);每轮动态预算按 budget_priority 裁剪(装不下核心经 abort 不带超预算面调模型);前端工单预览卡(确认/取消);新增配置 MCP_LOGISTICS_URL / MCP_AFTER_SALES_URL / MCP_DISCOVERY_TIMEOUT_SECONDS / TOOL_WRITE_TIMEOUT_SECONDS / TOOL_TIMEOUT_OVERRIDES / AUDIT_RESULT_MAX_CHARS(.env.example 注释列);评审 backlog 已闭环(死 import/魔数常量/依赖上限/并发确认钉);全量 pytest **617 passed**
- 2026-09-28:ch07 人工验收五条全过(20+轮稳定/演示窗口级联全现/默认零降级/日志全量/侧栏四步点验)+ Task 16 P1 修复——中断轮被新消息取代后用户消息双丢(MySQL+模型上下文),prepare 先落库用户消息、log 幂等补 assistant 行;语义边界记档:被取代轮不进活会话即时上下文,走账本→摘要投影回模型视野(浏览器实证:摘要投影答出 A1001);全量 pytest **576 passed**
- ch07 完成:会话上下文三层管理——历史按「层1 原文滑窗(预算 70%)/ 层2 截短投影(assistant 答复留开头)/ 摘要多段投影」拼装,understand/classify 共享分层 history_block,旧 checkpoint 只读对齐 db_id(对齐失败整轮放弃不猜);log 节点盖章 db_id → 层1 降级锚点前移 → 层2 超预算 ensure_future 异步摘要;输入 token 闸(tool_context_too_long)+ 启动预算自检(装不下稳态开销打 critical 不阻断)+ 工具结果 token 截断;只读接口 GET /api/conversations 与 /{id}/messages;聊天页会话侧栏(列表/切换/历史回载,失联静默降级);DDL conversation_summaries 外键级联(db/init/05-ddl.sql)。probe 真模型实测:摘要标注 3/3、意图 24/24、默认窗口 20 轮无降级零级联、演示 13000 窗口降级+摘要级联全现;新增配置 MODEL_CONTEXT_WINDOW / MAX_USER_INPUT_TOKENS / TOOL_RESULT_MAX_TOKENS / SUMMARY_PROJECTION_TOKENS / HISTORY_TARGET_TURNS / STEADY_TOKENS_PER_TURN / SUMMARY_MAX_CHARS / SAFETY_MARGIN_TOKENS(.env.example 注释列),RERANK_TOP_N 全仓改名 RERANK_TOP_K,UNDERSTAND_HISTORY_TURNS 退役;全量 pytest **570 passed**
- 2026-09-21:ch06 评审 Minor backlog 闭环——M3 规则 13 限定「可用工具含 query_faq 时」/ M4 §5 表第 3/5/7 行补字面钉 / M5 chat_visible 节点内钉(每步 astream 带 tag)/ M6 nodes.py 中部 import 归位;M1(历史 token 裁剪)/M2(订单 NS 碰撞)/M7(工具选择非确定)按评审标注记档不修;全量 pytest **530 passed**;**验收 4 浏览器人工点验四项全过**(订单卡/退款表单/旧卡失效/引用弹层,截图实证),ch06 全部验收关闭
- ch06 完成:正式版分流器上线——understand_query 指代消解(历史完整轮 + active_order 辅助 + 订单号来源校验)/ classify_intent 八类两字段(纯 intent ROUTE_TABLE)/ refund 子流程(refund_scope 三分支 → refund_prepare 缺单 interrupt 挂起订单选单 → refund_policy 扩写+多查询统一重排+置信闸)/ POST /v1/chat/resume(锁内校验 404/409)/ create_refund 动作 / 前端 order_selector 订单卡 + refund_form 退款表单;probe 真模型验证 24/24 intent + 3 多轮剧本 0 失败;R2 milvus_store 首连竞态加锁修复;全量 pytest **520 passed**
- ch05 完成:聊天主链路切换 LangGraph 图驱动(SSE 协议不变 + suggest_actions 帧),spec §13 验收 11 条集成钉 + SQLite checkpoint 文件级持久化验证全绿,全量 pytest 443 passed;ch04 的 R7 忠实度 1.0 / 覆盖 0.970 / 库外拒答 1.0 成果保留
- 运行入口不变(docker compose up -d → uv run uvicorn app.main:create_app --factory);图工作记忆落 data/checkpoints.db(.gitignore 的 data/ 规则已覆盖);DeepSeek 端点 stream_usage 已联调核实(流式 usage 正常返回)
