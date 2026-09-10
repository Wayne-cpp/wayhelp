# wayhelp — 电商智能客服(ch01 纯对话)

## 环境
- `uv sync`(自动建 Python 3.12 虚拟环境)
- `cp .env.example .env` 并填写 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME

## 运行
```bash
docker compose up -d          # 启动 MySQL(首启自动建表+灌 faq seed)
uv run pytest                 # 测试(DB 用例需 Docker 在线)
uv run uvicorn app.main:create_app --factory   # 起服
# 浏览器打开 http://127.0.0.1:8000/
```

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
