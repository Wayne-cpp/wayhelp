# AGENTS.md — wayhelp

## 定位
电商智能客服演示项目:FastAPI + SSE 流式聊天,LangChain 工具调用,Milvus Lite 四策略混合检索(dense / bm25 / hybrid / hybrid_rerank)+ 硅基流动重排,deepseek 生成与裁判。分工:MySQL 是账本(知识块/会话/台账),Milvus 是货架(向量)。

## 怎么跑
- `uv sync`;`cp .env.example .env` 填 OPENAI_* 与 EMBEDDING_API_KEY
- `docker compose up -d`(MySQL,首启自动建表)→ `uv run uvicorn --factory app.main:create_app`
- `uv run pytest`(365 条,需 Docker 在线);页面:`/` 聊天、`/kb` 知识库、`/rag-eval` 评估
- 细节以 README.md 为准;逐章开发实录在 dev-notes/

## 目录与约定
- app/{routers,services,knowledge,prompts,tools,static} 分层自明;knowledge_docs/ 是知识库唯一源;evals/ 评估;`sql/chXX-ddl.sql` 与 `db/init/` 同名 DDL 必须逐字节一致(有测试钉)
- Conventional Commits;改 prompt / 检索 / 知识库后跑全量 pytest 再交付

## 红线与坑(都踩过)
- 本机代理会 502:访问本机服务一律 `curl --noproxy '*'`
- 单 Uvicorn worker(内存作业注册表 + Milvus Lite 文件独占);pkill 模式会匹配自身 shell 命令行,用 `pgrep -f '[u]vicorn --factory'` 方括号技巧
- 改 knowledge_docs 已导入文档:ingest_docs 拒绝覆盖(不覆盖旧知识),唯一路径 `POST /kb/api/rebuild`(服内执行);ingest/mine 等直写 `./data` 库的 CLI 须先停服
- 评估(make eval-rag 或 /rag-eval 重跑)约 25 分钟/轮、烧 API 额度;产物 rag_eval.json 与时间戳归档均不入库
- faith_cases 台账:status 取中文值 未解决/已解决/无需解决;「已解决」=确认真编造且已修(复发自动打回未解决),「无需解决」=裁判误判;`eval_id` 列存 case 编号(如 E40),不是 run_id
- 改 app/prompts/service.py 条款后:test_prompts.py 有契约断言(REFUSAL_ANSWER 逐字内嵌等),跑全量验证
- chat.html 走 Vibe Coding 例外(无 TDD),前端逻辑靠 test_chat_page 字符串断言 + 人工点验;dev-notes/ch04.md 含用户本人手写段落,改动前先 `git diff` 看清归属

## 当前状态(2026-09-17)
- ch04 完成:R7 评估忠实度 1.0(首次零编造)/ 覆盖 0.970 / 库外拒答 1.0,门槛全过;台账 19 条全部裁决(已解决 10 / 无需解决 9)
- 工作区有一批未提交优化(prompt 第 12/13 条、多意图 sub_queries 支路、评估链路修复、知识库 4 份文档修订),提交前 `git status` 逐项核对,chat.html 与 dev-notes/ch04.md 的用户改动勿裹挟
