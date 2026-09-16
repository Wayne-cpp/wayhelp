# wayhelp — 电商智能客服

SSE 流式客服聊天 + 模型自选工具 + 向量知识库:用户问一句,后端走「模型定工具 → 执行 → 结果回灌 → 收敛作答」,回答逐 token 吐出,聊天气泡带工具轨迹徽章;ch03 起叠加 Milvus 向量语义检索,FAQ/政策类问题先查知识库;ch04 起升级四策略混合检索与重排,回答带 [n] 引用角标可回原文,证据不足固定话术拒答并入低置信池。单轮工具调用上限 MAX_TOOL_CALLS_PER_TURN(默认 5,create_ticket 单轮限一次)。

## 环境
- `uv sync`(自动建 Python 3.12 虚拟环境)
- `cp .env.example .env` 并填写 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME

## 运行
```bash
docker compose up -d          # 启动 MySQL(首启自动建表 faq/conversations/messages/tickets + 灌 faq seed)
uv run pytest                 # 测试 317 条(DB 用例需 Docker 在线)
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
- 「邮费是多少」→ query_faq 关键词查不到,如实说明(漏召回为预期结果,留待下一步升级 RAG;ch03 已升级为向量语义检索,该问题现在能召回运费说明,见下文 ch03 节)
- 「转人工」→ create_ticket 建工单,作答引用工单号

## 知识库与向量语义检索(ch03 新增)

- 知识文档建库:`uv run python -m app.jobs.ingest_docs [knowledge_docs/]`
- 对话挖知识:`uv run python -m app.jobs.mine_qa`
- 召回评估:`uv run python evals/run_knowledge_eval.py`(加 `--dump-corpus` 建评估语料并打印可标注块)
- 新增环境变量:EMBEDDING_BASE_URL / EMBEDDING_API_KEY(硅基流动;上面三个命令都要先填它,在线服务缺它降级为「知识检索未配置」但不拒启动)/ EMBEDDING_MODEL(BAAI/bge-m3)/ MILVUS_URI(./data/milvus_lite.db)等,见 .env.example

### 数据库初始化与升级

- 全新环境:`docker compose up -d` 首启自动执行 db/init 全部 DDL(含 ch03)
- 已有 ch02 数据卷的升级(不得删卷):`docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch03-ddl.sql`

### 运行约束

- 知识库管理页:浏览器打开 `http://127.0.0.1:8000/kb`(聊天页页脚有入口),日常建库/差异预览/向量化/挖掘/选择性重建/检索自测都在页面上完成,作业在 uvicorn 进程内复用同一 Milvus 连接,全局互斥
- Milvus Lite 本地库独占打开(文件锁互斥):服务运行(uvicorn)时请勿另跑知识库 CLI(ingest_docs / mine_qa / 召回评估),要用 CLI 先停服务;CLI 保留用于离线/脚本场景,服务单 worker
- 不并行运行两个知识任务(页面侧已互斥);缺 EMBEDDING_API_KEY 时在线检索降级为「知识检索未配置」,服务正常启动;/kb 的建库在该情况下只做切块入库留 pending,向量化/挖掘报 embedding_not_configured

## 四策略混合检索与引用(ch04 新增)

- 检索策略四选一(`KNOWLEDGE_STRATEGY`,默认 `hybrid_rerank`):
  - `dense`:向量语义检索(ch03 既有,BAAI/bge-m3)
  - `bm25`:Milvus Lite 全文检索(jieba 中文分词,型号/关键词类问题强项)
  - `hybrid`:dense + bm25 双路召回,RRF 融合
  - `hybrid_rerank`:hybrid 候选(前 RETRIEVAL_CANDIDATE_K 条)交硅基流动重排模型(BAAI/bge-reranker-v2-m3)精排取 RERANK_TOP_N;缺重排密钥时自动降级 hybrid
- 查询改写与拒答:QUERY_REWRITE_ENABLED(默认开)先经模型生成检索查询;命中低于策略阈值即判 low_confidence,服务端不再作答,直接回固定拒答话术并把问题入 `low_confidence_questions` 表
- 聊天页:回答中的 [1] 角标可点,弹层显示原文与章节路径;👍/👎 满意度反馈(本地 localStorage 锁定);/kb 页提供「重建索引」全量重置入口与策略/范围检索自测

### 数据库初始化与升级

- 全新环境:`docker compose up -d` 首启自动执行 db/init 全部 DDL(含 ch04)
- 已有 ch03 数据卷的升级(不得删卷):`docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch04-ddl.sql`(新增 low_confidence_questions / faith_cases 两表),再起服到 /kb 点「重建索引」——ch04 语料已换血且 Milvus 集合 schema 变更,必须全量重建;服务启动时检测到旧 schema 会显示 rebuild_required 状态

### 验证(eval 脚本)

- 语料完整性校验(离线,无需 key):`uv run python evals/validate_corpus.py`
- 四策略对比 + 生成段 Faithfulness 评估(300 case):`uv run python evals/run_retrieval_compare.py`(可加 `--max-d-pass 0.10` 调 D 桶误通过上限);产物落 `evals/results/{时间戳}_compare.json` 与 `.md`,报告含各策略 SR@5/10、CH@5/10、MRR@10、D 桶拒答正确率、误拒率与冻结阈值;某策略在 D 约束下无可行阈值时该臂 ungated 仅观测(threshold=null,有命中即过闸),评估不中断
- ch03 的 `evals/run_knowledge_eval.py` 已弃用:ch04 语料换血后旧标注失效,该脚本仅供指标函数复用

### 新增依赖与环境变量

- jieba(新增第三方依赖,Milvus Lite BM25 的 JiebaAnalyzer 中文分词必需),`uv sync` 自动安装
- 新增环境变量(默认值见 .env.example):RERANK_BASE_URL / RERANK_API_KEY(留空回退 EMBEDDING_API_KEY)/ RERANK_MODEL、RETRIEVAL_CANDIDATE_K、RERANK_TOP_N、RERANK_MIN_SCORE / BM25_MIN_SCORE / HYBRID_MIN_SCORE(策略阈值,评估冻结值见 .env.example 注释)、KNOWLEDGE_STRATEGY、QUERY_REWRITE_ENABLED、KNOWLEDGE_TOOL_TIMEOUT_SECONDS

### 运行约束(沿用 ch03)

- 服务单 worker;Milvus Lite 本地库独占打开:服务运行时勿另跑知识库 CLI,跑评估先停服务
- 评估前置:EMBEDDING_API_KEY 必填(重排密钥回退同 key);生成段还需 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME;主库须已执行 ch04 DDL(faith_cases 写主业务库)
