# wayhelp — 电商智能客服(ch02 Function Calling 工具链)

SSE 流式客服聊天 + 模型自选工具:用户问一句,后端走「模型定工具 → 执行 → 结果回灌 → 收敛作答」,回答逐 token 吐出,聊天气泡带工具轨迹徽章。单轮最多调一次工具。

## 环境
- `uv sync`(自动建 Python 3.12 虚拟环境)
- `cp .env.example .env` 并填写 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME

## 运行
```bash
docker compose up -d          # 启动 MySQL(首启自动建表 faq/conversations/messages/tickets + 灌 faq seed)
uv run pytest                 # 测试 136 条(DB 用例需 Docker 在线)
uv run uvicorn app.main:create_app --factory   # 起服
# 浏览器打开 http://127.0.0.1:8000/
```

## 工具链(ch02 新增)
- LangChain `@tool` 五个业务工具(`app/tools/business.py`):`query_order` / `query_product` / `query_logistics`(演示用随机数据,不接真实接口)、`query_faq`(SQL LIKE 查 faq 表)、`create_ticket`(写 tickets 表)
- 基础设施:注册管理、参数 Schema 校验、错误处理、超时重试(`TOOL_TIMEOUT_SECONDS` / `TOOL_MAX_RETRIES`),工具结果截断后回灌模型
- 聊天页:工具执行前推状态帧,气泡上方显示工具徽章;含工具调用与结果的完整流水落 conversations/messages 表

## 验证(eval 脚本)
- uv run python evals/run_provider_smoke.py
- uv run python evals/run_extract_eval.py

## 手动验收

# 1. 流式对话
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好,我想咨询退货"}'

# 2. 多轮上下文(把上一响应的 session_id 带回来)
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<上一步的 session_id>","message":"我刚才问的是什么?"}'

# 3. 结构化提取
curl -X POST http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"订单 20260828012 显示签收但没收到,我不要货了,直接退钱"}'

## 浏览器验收(ch02)
- 「订单 1001 的物流到哪了」→ 模型选中 query_logistics,气泡带工具徽章,按工具返回作答
- 「退货政策是什么」→ query_faq 命中 faq 表并作答
- 「邮费是多少」→ query_faq 关键词查不到,如实说明(漏召回为预期结果,留待下一步升级 RAG)
- 「转人工」→ create_ticket 建工单,作答引用工单号

## 知识库与向量语义检索(ch03 新增)

- 知识文档建库:`uv run python -m app.jobs.ingest_docs [knowledge_docs/]`
- 对话挖知识:`uv run python -m app.jobs.mine_qa`
- 召回评估:`uv run python evals/run_knowledge_eval.py`
- 新增环境变量:EMBEDDING_BASE_URL / EMBEDDING_API_KEY(硅基流动,必填后两个命令才可运行)/ EMBEDDING_MODEL(BAAI/bge-m3)/ MILVUS_URI(./data/milvus_lite.db)等,见 .env.example

### 数据库初始化与升级

- 全新环境:`docker compose up -d` 首启自动执行 db/init 全部 DDL(含 ch03)
- 已有 ch02 数据卷的升级(不得删卷):`docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch03-ddl.sql`

### 运行约束

- Milvus Lite 本地库独占打开:跑建库/挖矿/评估前先停在线服务(uvicorn),完成后再起;服务单 worker
- 不并行运行两个知识任务;缺 EMBEDDING_API_KEY 时在线检索降级为「知识检索未配置」,服务正常启动
