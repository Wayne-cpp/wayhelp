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
