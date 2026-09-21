# AGENTS.md — wayhelp

## 定位
电商智能客服演示项目:FastAPI + SSE 流式聊天,LangGraph Workflow 图编排聊天主链路(指代消解 understand → 八类意图分流 → 退款/售后确定性子流程(缺单 interrupt 挂起选单 + 扩写多查询强制检索 + 置信闸)→ 手写 ReAct 主力 Agent(business 分支按需 query_faq)),Milvus Lite 四策略混合检索(dense / bm25 / hybrid / hybrid_rerank)+ 硅基流动重排,deepseek 生成与裁判。分工:MySQL 是账本(知识块/会话/台账),Milvus 是货架(向量),SQLite(checkpoints.db)是图工作记忆。

## 怎么跑
- `uv sync`;`cp .env.example .env` 填 OPENAI_* 与 EMBEDDING_API_KEY
- `docker compose up -d`(MySQL,首启自动建表)→ `uv run uvicorn --factory app.main:create_app`
- `uv run pytest`(530 条,需 Docker 在线);页面:`/` 聊天、`/kb` 知识库、`/rag-eval` 评估
- 细节以 README.md 为准;逐章开发实录在 dev-notes/

## 目录与约定
- app/{routers,services,knowledge,prompts,tools,static,graph} 分层自明(app/graph/ 是 LangGraph 聊天图,ch05 搭骨架、ch06 正式版分流器:state / nodes / agent_node / builder / events);examples/ 是 ch05 热身留档脚本(bare_agent.py);knowledge_docs/ 是知识库唯一源;evals/ 评估与真模型 probe(烧额度,手跑不进 pytest);`sql/chXX-ddl.sql` 与 `db/init/` 同名 DDL 必须逐字节一致(有测试钉)
- Conventional Commits;改 prompt / 检索 / 知识库后跑全量 pytest 再交付

## 红线与坑(都踩过)
- 本机代理会 502:访问本机服务一律 `curl --noproxy '*'`
- 单 Uvicorn worker(内存作业注册表 + Milvus Lite 文件独占);pkill 模式会匹配自身 shell 命令行,用 `pgrep -f '[u]vicorn --factory'` 方括号技巧
- 改 knowledge_docs 已导入文档:ingest_docs 拒绝覆盖(不覆盖旧知识),唯一路径 `POST /kb/api/rebuild`(服内执行);ingest/mine 等直写 `./data` 库的 CLI 须先停服
- 评估(make eval-rag 或 /rag-eval 重跑)约 25 分钟/轮、烧 API 额度;产物 rag_eval.json 与时间戳归档均不入库
- faith_cases 台账:status 取中文值 未解决/已解决/无需解决;「已解决」=确认真编造且已修(复发自动打回未解决),「无需解决」=裁判误判;`eval_id` 列存 case 编号(如 E40),不是 run_id
- 改 app/prompts/service.py 条款后:test_prompts.py 有契约断言(REFUSAL_ANSWER 逐字内嵌等),跑全量验证
- chat.html 走 Vibe Coding 例外(无 TDD),前端逻辑靠 test_chat_page 字符串断言 + 人工点验;dev-notes/ch04.md 含用户本人手写段落,改动前先 `git diff` 看清归属
- 聊天 Agent 无写权限:图专用 `build_graph_tools`(app/tools/business.py)只产只读工具——query_order / query_logistics 绑定当前 user_id(非本人/不存在回 error JSON)、query_product、无参 query_faq(以本轮 resolved_query 检索,同轮缓存)+ suggest_options 伪工具;business 分支四只读全绑,refund 过闸分支不绑 query_faq;旧 `build_tools`(business.py:120)捆绑 create_ticket,**禁止回挂到图**;模型伪造 create_ticket / create_refund 只回 unknown_tool 不写库;建工单/退款单唯一通道 `POST /v1/chat/action`(create_ticket / create_refund 双动作契约,校验会话/消息/订单归属、绑定 source_message_id、不改 conv.status);「转人工」是纯前端模拟,后端零动作
- `POST /v1/chat/resume` 契约:会话锁内自校验 pending interrupt(interrupt_id 匹配 + 订单在候选卡内 + 用户归属),归属错误 404 不泄露、卡片失效/不匹配 409;resume 传 langgraph `Command` 不传 dict;新消息或再次挂起后旧卡立即 409
- 订单 mock 服务(app/services/orders.py)是订单数据唯一入口:get_order / list_orders 按用户命名空间实例化,不存在与不属于该用户均返回 None;**不建订单表、不动任何 DDL**
- 意图 JSON 就 `{"intent","confidence"}` 两字段;needs_knowledge 维度已整体移除(state / prompt / parse / 日志 / probe 不留失效引用)
- SSE delta 三条件过滤:langgraph_node==main_agent + metadata.tags 含 chat_visible + AIMessageChunk;understand / classify / refund_scope / expand 等节点内部模型调用不外发
- 全量 pytest 的 DB 依赖:dbfixtures 用 .env 的 TEST_ADMIN_DATABASE_URL(root@127.0.0.1 TCP)建 wayhelp_test 库;若本机是原生 mysqld 而非 Docker,root 走 auth_socket 会 1698 拒连、fixture 入口直接 exit(3)——要么起 docker compose,要么给 root 开 TCP 密码授权(wayhelp 账号只有 wayhelp 库权限,不够)
- uvicorn 起服:shell 有代理变量(http_proxy→7890 等)且无 no_proxy 时,milvus-lite 内嵌 gRPC 会被劫持进代理(GOAWAY)→ 启动契约探针误判 rebuild_required、检索整轮不可用;起服命令前加 `no_proxy=127.0.0.1,localhost`(curl 访问本机服务仍须另加 --noproxy '*')

## 当前状态(2026-09-21)
- 2026-09-21:ch06 评审 Minor backlog 闭环——M3 规则 13 限定「可用工具含 query_faq 时」/ M4 §5 表第 3/5/7 行补字面钉 / M5 chat_visible 节点内钉(每步 astream 带 tag)/ M6 nodes.py 中部 import 归位;M1(历史 token 裁剪)/M2(订单 NS 碰撞)/M7(工具选择非确定)按评审标注记档不修;全量 pytest **530 passed**;**验收 4 浏览器人工点验四项全过**(订单卡/退款表单/旧卡失效/引用弹层,截图实证),ch06 全部验收关闭
- ch06 完成:正式版分流器上线——understand_query 指代消解(历史完整轮 + active_order 辅助 + 订单号来源校验)/ classify_intent 八类两字段(纯 intent ROUTE_TABLE)/ refund 子流程(refund_scope 三分支 → refund_prepare 缺单 interrupt 挂起订单选单 → refund_policy 扩写+多查询统一重排+置信闸)/ POST /v1/chat/resume(锁内校验 404/409)/ create_refund 动作 / 前端 order_selector 订单卡 + refund_form 退款表单;probe 真模型验证 24/24 intent + 3 多轮剧本 0 失败;R2 milvus_store 首连竞态加锁修复;全量 pytest **520 passed**
- ch05 完成:聊天主链路切换 LangGraph 图驱动(SSE 协议不变 + suggest_actions 帧),spec §13 验收 11 条集成钉 + SQLite checkpoint 文件级持久化验证全绿,全量 pytest 443 passed;ch04 的 R7 忠实度 1.0 / 覆盖 0.970 / 库外拒答 1.0 成果保留
- 运行入口不变(docker compose up -d → uv run uvicorn app.main:create_app --factory);图工作记忆落 data/checkpoints.db(.gitignore 的 data/ 规则已覆盖);DeepSeek 端点 stream_usage 已联调核实(流式 usage 正常返回)
