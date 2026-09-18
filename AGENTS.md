# AGENTS.md — wayhelp

## 定位
电商智能客服演示项目:FastAPI + SSE 流式聊天,LangChain 工具调用,Milvus Lite 四策略混合检索(dense / bm25 / hybrid / hybrid_rerank)+ 硅基流动重排,deepseek 生成与裁判。分工:MySQL 是账本(知识块/会话/台账),Milvus 是货架(向量)。

## 怎么跑
- `uv sync`;`cp .env.example .env` 填 OPENAI_* 与 EMBEDDING_API_KEY
- `docker compose up -d`(MySQL,首启自动建表)→ `uv run uvicorn --factory app.main:create_app`
- `uv run pytest`(439 条,需 Docker 在线);页面:`/` 聊天、`/kb` 知识库、`/rag-eval` 评估
- 细节以 README.md 为准;逐章开发实录在 dev-notes/

## 目录与约定
- app/{routers,services,knowledge,prompts,tools,static,graph} 分层自明(app/graph/ 是 ch05 LangGraph 图:state / nodes / agent_node / builder);examples/ 是 ch05 热身留档脚本(bare_agent.py);knowledge_docs/ 是知识库唯一源;evals/ 评估;`sql/chXX-ddl.sql` 与 `db/init/` 同名 DDL 必须逐字节一致(有测试钉)
- Conventional Commits;改 prompt / 检索 / 知识库后跑全量 pytest 再交付

## 红线与坑(都踩过)
- 本机代理会 502:访问本机服务一律 `curl --noproxy '*'`
- 单 Uvicorn worker(内存作业注册表 + Milvus Lite 文件独占);pkill 模式会匹配自身 shell 命令行,用 `pgrep -f '[u]vicorn --factory'` 方括号技巧
- 改 knowledge_docs 已导入文档:ingest_docs 拒绝覆盖(不覆盖旧知识),唯一路径 `POST /kb/api/rebuild`(服内执行);ingest/mine 等直写 `./data` 库的 CLI 须先停服
- 评估(make eval-rag 或 /rag-eval 重跑)约 25 分钟/轮、烧 API 额度;产物 rag_eval.json 与时间戳归档均不入库
- faith_cases 台账:status 取中文值 未解决/已解决/无需解决;「已解决」=确认真编造且已修(复发自动打回未解决),「无需解决」=裁判误判;`eval_id` 列存 case 编号(如 E40),不是 run_id
- 改 app/prompts/service.py 条款后:test_prompts.py 有契约断言(REFUSAL_ANSWER 逐字内嵌等),跑全量验证
- chat.html 走 Vibe Coding 例外(无 TDD),前端逻辑靠 test_chat_page 字符串断言 + 人工点验;dev-notes/ch04.md 含用户本人手写段落,改动前先 `git diff` 看清归属
- 聊天 Agent 无写权限:工具绑定与执行注册表只有 query_order / query_product / query_logistics + suggest_options 伪工具,模型伪造 create_ticket 只回 unknown_tool 不写库;建工单唯一通道 `POST /v1/chat/action`(校验会话/消息归属、绑定 source_message_id、不改 conv.status);「转人工」是纯前端模拟,后端零动作

## 当前状态(2026-09-18)
- ch05 完成:聊天主链路切换 LangGraph 图驱动(SSE 协议不变 + suggest_actions 帧),spec §13 验收 11 条集成钉 + SQLite checkpoint 文件级持久化验证全绿,全量 pytest 439 passed;ch04 的 R7 忠实度 1.0 / 覆盖 0.970 / 库外拒答 1.0 成果保留
- 运行入口不变(docker compose up -d → uv run uvicorn app.main:create_app --factory);图工作记忆落 data/checkpoints.db(.gitignore 的 data/ 规则已覆盖);DeepSeek 端点 stream_usage 已联调核实(流式 usage 正常返回)
