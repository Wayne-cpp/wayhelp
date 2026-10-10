# wayhelp — 电商智能客服

SSE 流式客服聊天 + 模型自选工具 + 向量知识库:用户问一句,后端走「模型定工具 → 执行 → 结果回灌 → 收敛作答」,回答逐 token 吐出,聊天气泡带工具轨迹徽章;ch03 起叠加 Milvus 向量语义检索,FAQ/政策类问题先查知识库;ch04 起升级四策略混合检索与重排,回答带 [n] 引用角标可回原文,证据不足固定话术拒答并入低置信池;ch05 起聊天主链路由 LangGraph Workflow 图编排,投诉/建工单改为前端独立按钮 + 动作端点;ch06 起图换正式版分流器(指代消解 → 八类意图纯 intent 路由 → 退款/售后确定性子流程:缺订单号 interrupt 挂起出订单卡、选单 resume 续跑、扩写多查询统一重排 + 置信闸),订单数据为按用户命名空间实例化的确定性 mock。ch07 起会话历史按三层窗口管理(摘要投影 + 层2 截短 + 层1 原文滑窗,超预算后台异步摘要),聊天页带会话侧栏(多会话切换/历史回载)。ch08 起工具层即插即用:内置工具装饰器自登记,物流/售后查询拆到两个独立进程 MCP Server(每轮动态发现,主服不重启),建工单走前端预览卡确认流,全量工具调用落审计台账。单轮工具调用上限 MAX_TOOL_CALLS_PER_TURN(默认 5)。

## 环境
- `uv sync`(自动建 Python 3.12 虚拟环境)
- `cp .env.example .env` 并填写 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME

## 运行
```bash
docker compose up -d          # 启动 MySQL(首启自动建表 faq/conversations/messages/tickets + 灌 faq seed)
# 已有 ch06 数据卷的老库升级(不得删卷;docker 首启自动含 db/init/05-ddl.sql,新装可跳过):
docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch07-ddl.sql
# 已有 ch07 数据卷的老库升级(不得删卷;docker 首启自动含 db/init/06-ddl.sql,新装可跳过):
docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch08-ddl.sql
uv run pytest                 # 测试 738 条(DB 用例需 Docker 在线)
uv run uvicorn app.main:create_app --factory   # 起服
# 浏览器打开 http://127.0.0.1:8000/
```

## 工具链(ch02 新增)
- LangChain `@tool` 五个业务工具(`app/tools/business.py`):`query_order` / `query_product` / `query_logistics`(演示用随机数据,不接真实接口)、`query_faq`(SQL LIKE 查 faq 表)、`create_ticket`(写 tickets 表)(ch08 起:工具迁 `app/tools/builtin/` 装饰器自登记 + MCP 每轮动态发现,query_logistics 内置下线改由物流 MCP Server 接管,create_ticket 改走预览卡确认流,见下文 MCP Server 节)
- 基础设施:注册管理、参数 Schema 校验、错误处理、超时重试(`TOOL_TIMEOUT_SECONDS` / `TOOL_MAX_RETRIES`),工具结果截断后回灌模型
- 聊天页:工具执行前推状态帧,气泡上方显示工具徽章;含工具调用的完整流水落 conversations/messages 表(ch07 起 tool 结果行不再落表,assistant 行仍带 tool_calls,见下文 ch07 节)

## 验证(eval 脚本)
- uv run python evals/run_provider_smoke.py
- uv run python evals/run_extract_eval.py

## 手动验收

```bash
# 本机有代理时访问本机服务一律加 --noproxy '*';user_id 必填且必须是 UUID
# (11111111-… 是演示账号,mock 订单为 1111-1001..1004)

# 1. 流式对话
curl --noproxy '*' -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","message":"你好,我想咨询退货"}'

# 2. 多轮上下文(把上一响应的 session_id 带回来)
curl --noproxy '*' -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","session_id":"<上一步的 session_id>","message":"我刚才问的是什么?"}'

# 3. 结构化提取
curl --noproxy '*' -X POST http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"订单 20260828012 显示签收但没收到,我不要货了,直接退钱"}'

# 4. 退款选单(ch06):不带订单号问 → 返回 order_selector 帧(订单卡)
curl --noproxy '*' -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","message":"这个能退吗"}'

# 5. 选单后续跑(帧里的 interrupt_id;重放同 ID 预期 409)
curl --noproxy '*' -N -X POST http://127.0.0.1:8000/v1/chat/resume \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","session_id":"…","interrupt_id":"…","order_id":"1111-1001"}'

# 6. 退款申请动作(suggest_actions 帧回传的 source_message_id;建工单同端点 action=create_ticket)
curl --noproxy '*' -X POST http://127.0.0.1:8000/v1/chat/action \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","session_id":"…","action":"create_refund","source_message_id":"…","order_id":"1111-1001","refund_reason":"七天无理由"}'
```

## 浏览器验收(ch02)
- 「订单 1001 的物流到哪了」→ 模型选中 query_logistics,气泡带工具徽章,按工具返回作答
- 「退货政策是什么」→ query_faq 命中 faq 表并作答
- 「邮费是多少」→ query_faq 关键词查不到,如实说明(漏召回为预期结果,留待下一步升级 RAG;ch03 已升级为向量语义检索,该问题现在能召回运费说明,见下文 ch03 节)
- 「转人工」→ create_ticket 建工单,作答引用工单号(ch05 起改为安抚话术+前端「转人工/建工单」独立按钮,不再由模型主动建单,见下文 ch05 节)

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
  - `hybrid_rerank`:hybrid 候选(前 RETRIEVAL_CANDIDATE_K 条)交硅基流动重排模型(BAAI/bge-reranker-v2-m3)精排取 RERANK_TOP_K;缺重排密钥时自动降级 hybrid
- 查询改写与拒答:QUERY_REWRITE_ENABLED(默认开)先经模型生成检索查询;命中低于策略阈值即判 low_confidence,服务端不再作答,直接回固定拒答话术并把问题入 `low_confidence_questions` 表
- 聊天页:回答中的 [1] 角标可点,弹层显示原文与章节路径;👍/👎 满意度反馈(本地 localStorage 锁定);/kb 页提供「重建索引」全量重置入口与策略/范围检索自测

### 数据库初始化与升级

- 全新环境:`docker compose up -d` 首启自动执行 db/init 全部 DDL(含 ch04)
- 已有 ch03 数据卷的升级(不得删卷):`docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch04-ddl.sql`(新增 low_confidence_questions / faith_cases 两表),再起服到 /kb 点「重建索引」——ch04 语料已换血且 Milvus 集合 schema 变更,必须全量重建;服务启动时检测到旧 schema 会显示 rebuild_required 状态

### 验证(eval 脚本)

- 语料完整性校验(离线,无需 key):`uv run python evals/validate_corpus.py`
- 四策略对比 + 生成段 Faithfulness 评估(300 case):`uv run python evals/run_retrieval_compare.py`(可加 `--max-d-pass 0.10` 调 D 桶误通过上限);产物落 `evals/results/{时间戳}_compare.json` 与 `.md`,报告含各策略 SR@5/10、CH@5/10、MRR@10、D 桶拒答正确率、误拒率与冻结阈值;某策略在 D 约束下无可行阈值时该臂 ungated 仅观测(threshold=null,有命中即过闸),评估不中断
- `make eval-rag`:等价于上面的四策略评估命令;跑完产出 `evals/results/rag_eval.json`,后台 `/rag-eval` 页读这份产物。评估产物(滚动报告 + 时间戳归档)均不入库

### RAG 评估页与编造个案台账

- `http://127.0.0.1:8000/rag-eval`:报告头 / 四项 KPI / 检索质量(四策略 × 总体+四桶,MRR / Recall@5 / 证据覆盖度切换)/ 生成质量 / 完整数据汇总表 / 编造个案台账 / 就地重跑(日志窗,跑完自动重新取数)
- 台账三状态:`未解决` → 人工裁决为「已解决」(确认是真编造且根因已修)或「无需解决」(裁判误判),处置必填一句说明;已解决的个案若复发会被自动打回未解决
- 两个幻觉率口径:本轮裁判判出率 = 上线策略 fabricated/judged;人工确认率 = 本轮判出中已标「已解决」的比例(个案证据快照须与当前报告同 run_id,否则该轮不计算)
- 经验值:deepseek-v4-flash 当裁判的误杀率约五成(19 条历史个案人工裁决出 9 条误判),判出率虚高,以人工确认率为准;一轮评估约 25 分钟、烧 API 额度
- 部署契约:本应用按**单 Uvicorn worker** 运行(内存作业注册表与 Milvus Lite 独占均不支持多 worker);不要配置 `--workers >1`。
- ch03 的 `evals/run_knowledge_eval.py` 已弃用:ch04 语料换血后旧标注失效,该脚本仅供指标函数复用

### 新增依赖与环境变量

- jieba(新增第三方依赖,Milvus Lite BM25 的 JiebaAnalyzer 中文分词必需),`uv sync` 自动安装
- 新增环境变量(默认值见 .env.example):RERANK_BASE_URL / RERANK_API_KEY(留空回退 EMBEDDING_API_KEY)/ RERANK_MODEL、RETRIEVAL_CANDIDATE_K、RERANK_TOP_N(ch07 起改名 RERANK_TOP_K)、RERANK_MIN_SCORE / BM25_MIN_SCORE / HYBRID_MIN_SCORE(策略阈值,评估冻结值见 .env.example 注释)、KNOWLEDGE_STRATEGY、QUERY_REWRITE_ENABLED、KNOWLEDGE_TOOL_TIMEOUT_SECONDS

### 运行约束(沿用 ch03)

- 服务单 worker;Milvus Lite 本地库独占打开(文件锁):ingest_docs / mine_qa 等直写 `./data` 库的 CLI 须先停服;`make eval-rag` 用独立临时 Milvus,与在线服务并存无冲突;/rag-eval 就地重跑在 uvicorn 进程内执行,无需停服
- 修改 `knowledge_docs/` 已导入的文档后,ingest_docs 会因「不覆盖旧知识」拒绝——改库唯一路径:/kb 页「重建索引」或 `POST /kb/api/rebuild`(服内执行,预检评估集 GT 覆盖,全量重置)
- 评估前置:EMBEDDING_API_KEY 必填(重排密钥回退同 key);生成段还需 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME;主库须已执行 ch04 DDL(faith_cases 写主业务库)

## LangGraph 图编排与动作端点(ch05 新增)

- 聊天主链路改为 LangGraph Workflow 图驱动(`app/graph/`),SSE 帧协议不变,新增 `suggest_actions` 按钮。图节点链:`START → resolve_reference → classify_intent`,按写死的路由表 `(intent, needs_knowledge)` 分四出口——knowledge(`retrieve → confidence_gate` 过闸进 Agent / 卡闸固定话术)、business(直接进 Agent)、complaint(固定安抚 + 转人工/建工单双按钮)、chitchat(固定话术零模型);四出口全汇 `log` 节点单事务提交(消息、低置信入池、按钮帧都在提交成功后才发)(ch06 起该路由表作废,换正式版分流器,见下文 ch06 节)
- 主力 Agent(`app/graph/agent_node.py`)是图内手写 ReAct 多步循环:绑定三只读业务工具 + `suggest_options` 伪工具,步数与累计 token 预算在每次模型调用前检查;聊天 Agent 无任何写权限
- 建工单唯一通道:`POST /v1/chat/action`(校验会话与消息归属、绑定产生按钮的那条用户消息,不改会话状态);「转人工」为纯前端模拟
- 跨轮上下文持久化:AsyncSqliteSaver 落 `data/checkpoints.db`(图工作记忆),MySQL 仍为账本;checkpoint 丢失退化为无历史新会话,不回填
- 新增环境变量(默认值见 .env.example):MAX_AGENT_STEPS / MAX_AGENT_TOKENS(主力 Agent 步数上限与累计 token 预算)/ CHECKPOINT_DB_PATH(图工作记忆 SQLite 路径)
- 验收:`uv run pytest tests/test_ch05_acceptance.py tests/test_graph_persistence.py -v`(spec §13 验收 11 条集成钉 + SQLite 文件级持久化)

## 正式版分流器与退款子流程(ch06 新增)

- 图拓扑:`START → understand_query(指代消解:最近 6 个完整轮历史 + active_order 已校验订单摘要辅助,输出订单号须可溯源,失败降级透传不炸轮)→ classify_intent(八类意图,JSON 就 {intent, confidence} 两字段,路由只查写死的 ROUTE_TABLE)`;出口五路:business(物流/订单/商品咨询 → 主力 Agent 按需 query_faq)、refund(退款退货/售后 → refund_scope 三分支)、complaint / chitchat / other(固定节点零模型)
- refund 子流程:refund_scope 判 general(不选单,直接强制检索)/ order_specific(先 refund_prepare:订单号在问句且唯一 → 直通;缺号或多候选 → `interrupt()` 挂起,驱动层发 order_selector 订单卡;`POST /v1/chat/resume` 按 interrupt_id 续跑,锁内校验归属,错归属 404 不泄露、卡片失效/重放 409)/ clarify(进 Agent 只澄清);refund_policy 强制检索(个案 + 开关 + hybrid_rerank 时扩写至多 4 查询并发、只按 chunk_id 合并后统一重排,扩写支路异常丢弃记 note、base 异常才整轮不可用),过闸进 Agent,卡闸固定话术
- 订单数据:`app/services/orders.py` 确定性 mock(订单号 {用户命名空间}-{序号},如 1111-1001..1004;get_order/list_orders 归属校验,非本人/不存在均 None;不建订单表、不动 DDL)
- 聊天 Agent 仍无写权限:图专用 `build_graph_tools` 只产只读工具(query_order/query_logistics 绑当前用户、query_product、无参 query_faq 以本轮 resolved_query 检索且同轮缓存)+ suggest_options 伪工具(擅传 order_id 等未知键回错误);退款单走 `POST /v1/chat/action` 的 create_refund 动作(六类固定原因,落 tickets 表,状态待处理;create_ticket 同端点);模型伪造 create_ticket/create_refund 只回 unknown_tool 不写库(ch08 起:`build_graph_tools` 退役,工具面改由 ToolCatalog 每轮装配 + 统一执行引擎把门,create_ticket 经预览卡确认流进图;退款单仍只走动作端点,见下文 MCP Server 节)
- 前端:order_selector 帧渲染订单卡组(点选 → POST /resume,答复追加到原挂起轮;旧卡 disable/失效 409 提示);suggest_actions 的 refund_form 选项出「申请退款」按钮 → 退款表单(订单号服务端绑定只读回填、原因六类下拉)
- SSE delta 三条件过滤:langgraph_node==main_agent + metadata.tags 含 chat_visible + AIMessageChunk,understand/classify/refund_scope/expand 等节点内部模型调用不外发;citations 帧在 log 提交成功后、DONE 前发,判定看整轮累计可见文本(带工具调用步骤的角标也能出卡)
- 新增环境变量(config.py 默认值,.env.example 未列):UNDERSTAND_HISTORY_TURNS=6(指代消解历史轮数;ch07 已退役,历史用量改由三层预算公式接管)/ REFUND_EXPAND_ENABLED=true(退款个案扩写开关)/ INTENT_MODEL_NAME=None(reserved 未接线,意图双模型 cascade 明确不做)
- 运行坑:shell 有代理变量时起服须带 `no_proxy=127.0.0.1,localhost`,否则 milvus-lite 内嵌 gRPC 被劫持进代理、启动契约探针误判 rebuild_required(检索整轮不可用)
- 验收:`uv run pytest tests/test_refund_flow.py tests/test_router_coverage.py -v`(挂起/resume/落库一轮 + spec §5 样例表全链路字面钉);真模型 probe(烧额度、手跑不进 pytest):`uv run python evals/probe_intent.py`(24 单/多轮意图样例)、`uv run python evals/probe_router_multi.py`(3 个多轮剧本全链路)

## 会话上下文管理(ch07 新增)

- 三层历史:模型上下文按「层1 原文滑窗(预算 70%)→ 层2 截短投影(assistant 答复只留开头)→ 摘要多段投影(【对话背景】)」拼装;understand/classify 共享同一份分层历史渲染;ch05/ch06 旧 checkpoint 会话只读对齐 db_id,对齐失败历史整轮放弃、不猜 ID
- 降级与摘要:log 节点落库后先给本轮消息盖章 db_id,层1 超预算把锚点 layer1_from 单调前移(整轮为界,tool 组不跨层劈开);层2 渲染 token 超预算时后台异步起摘要任务(ensure_future,不阻塞 SSE 收尾;CAS 追加、in-flight 防重入、失败不挪锚点);工具结果按 token 截断(TOOL_RESULT_MAX_TOKENS)
- 输入闸与自检:拼装超出窗口预算回固定话术 tool_context_too_long;启动时按真实工具面实测 SYS_TOKENS 打全分量预算日志,装不下一轮稳态开销打 critical(不阻断启动)
- 会话侧栏:聊天页左侧多会话列表(切换 / 历史回载),数据来自只读接口 `GET /api/conversations` 与 `GET /api/conversations/{id}/messages`;接口失联静默降级,不阻塞聊天
- 日志(log/app.log,FileHandler 幂等挂):`history_ctx`(understand/classify 每轮读出分层历史:摘要 + 滑窗逐条 role/content/db_id + 分层 token 估算)与 `model_ctx`(main_agent 每次模型调用:层1/层2 预算、条数、估算与逐条消息)均为单行 JSON,grep 友好;另有 `层1 降级`、`summary trigger/start/done/skip/failed` 运维事件线
- 数据库:conversations/messages 账本——prepare 先落库本轮用户消息(中断挂起轮不丢问句),log 节点幂等补 assistant 行(user_row_id 防重插);新增 conversation_summaries 表(会话外键级联删除);老库升级见「运行」节(sql/ch07-ddl.sql,docker 首启自动含 db/init/05-ddl.sql)
- 新增环境变量(默认值见 .env.example 注释):MODEL_CONTEXT_WINDOW(65536)/ MAX_USER_INPUT_TOKENS / TOOL_RESULT_MAX_TOKENS / SUMMARY_PROJECTION_TOKENS / HISTORY_TARGET_TURNS / STEADY_TOKENS_PER_TURN / SUMMARY_MAX_CHARS / SAFETY_MARGIN_TOKENS;RERANK_TOP_N 全仓改名 RERANK_TOP_K,UNDERSTAND_HISTORY_TURNS 退役
- probe(烧额度、手跑不进 pytest):`uv run python evals/probe_summary.py`(摘要标注样例)、`uv run python evals/probe_context.py`(20+ 轮长跑:默认窗口零降级零摘要 / 小窗口触发层1 降级 + 摘要级联)

## MCP Server(ch08 新增)

- 工具层即插即用:内置工具在 `app/tools/builtin/` 装饰器自登记(query_order / query_product / query_faq / create_ticket);物流与售后查询拆到两个独立进程业务 MCP Server,主服每轮动态发现、发现即用
- 起 Server(各自独立进程,默认端口 8101 / 8102,可用 `WAYHELP_MCP_LOGISTICS_PORT` / `WAYHELP_MCP_AFTER_SALES_PORT` 覆盖):

```bash
uv run python -m app.mcp_servers.logistics_server    # 物流:query_logistics
uv run python -m app.mcp_servers.after_sales_server  # 售后:query_warranty / query_return_progress
# 或等价 make 目标:
make mcp-logistics       # 起物流 MCP Server(:8101)
make mcp-after-sales     # 起售后 MCP Server(:8102)
```

- 主服接入:`.env` 配 `MCP_LOGISTICS_URL=http://127.0.0.1:8101/mcp` 与 `MCP_AFTER_SALES_URL=http://127.0.0.1:8102/mcp`(留空 = 不接入);其余新变量 MCP_DISCOVERY_TIMEOUT_SECONDS / TOOL_WRITE_TIMEOUT_SECONDS / TOOL_TIMEOUT_OVERRIDES / AUDIT_RESULT_MAX_CHARS 见 .env.example 注释
- 动态发现,主服不重启:每轮装配工具面时按 Server 现问现拿(get_tools),单个 Server 超时(默认 2 秒)跳过不拖聊天;Server 重启只影响当轮发现,下一轮自然恢复,加新工具同样无需动主服;只放行「零参数 / 仅必填字符串 order_id」两种形态的工具,未知 Server 默认拒绝(权限只认本地能力策略,不看 Server 自报)
- 建工单双通道:前端按钮走 `POST /v1/chat/action`(点按即确认);图内 create_ticket 走确认流——模型发起 → ticket_preview 预览卡 → `POST /v1/chat/resume` 带 decision(confirm / cancel),确认才真正建单、取消不建(参数只信服务端快照)
- 全量工具调用(内置 / MCP / 写确认)同走统一执行引擎,落 tool_audit_logs 审计台账;写超时按「已发未必未成」处理且绝不重试,tool_write_idempotency 收敛重复提交
- 红线:本机有代理时访问本机服务一律 `curl --noproxy '*'`;起 uvicorn 与 MCP Server 前加 `no_proxy=127.0.0.1,localhost`

## 可观测性与数据飞轮(ch09 新增)

- Langfuse 自托管可观测性:`docker compose up -d` 除 MySQL 外一并起 Langfuse v3 服务组(web / worker / ClickHouse / Postgres / Valkey / MinIO,web 暴露 127.0.0.1:3000);首次访问 `http://localhost:3000` 建站建项目,把 public/secret key 填进 `.env`(LANGFUSE_ENABLED=true + 两把密钥;未配齐 = 完全不挂回调,系统行为与现状一致)。trace 以逻辑轮为单位一棵树:LangChain 回调覆盖图内全部模型调用,classify 出意图即写入 trace metadata/tags;检索 / 工具 / MCP 在固定边界补 SDK span(无 ambient 上下文不产 span——评估/脚本直跑不会刷孤儿 trace);resume 与服务重启后从 checkpoint 重建 trace 上下文,消耗归原意图不进 unknown。注意 GET 类 API 数据可能滞后 ~10 分钟
- 成本统计:`GET /api/stats/cost-by-intent?days=7|30` 按 trace metadata.intent 聚合各意图 token/成本(observation 按 id 去重、每轮 1 request、优先 Langfuse costDetails、缺失按 MODEL_INPUT/OUTPUT_PRICE_PER_MTOK 折算,单价皆 0 时 cost_available=false);/rag-eval 页新增「成本」区块(分组柱状图)与「趋势」区块(eval_runs 历史 recall@10/MRR/忠实率三线,语料/评估集/置信闸版本变化即断线新起一组)
- 数据飞轮:三入口落池(检索低置信 / 生成自评缺知识 / 👎 反馈经 turn_committed 锚定精确回捞)带召回片段快照 → lifespan 单 worker 异步「模型标准化 → 与待审队列查重合并(occurrence_count 累加,失败指数退避,到限转 failed 等人工重试)」→ review_queue 待审 → 人工审核;通过即冻结答案与 FAQ 块、向量化写回知识库(`review:<id>` 来源,对线上检索生效),驳回留档不可重开
- /review 审核页(`http://127.0.0.1:8000/review`):状态 tab(待审/写入中/通过/驳回)+ 处理失败视图;详情抽屉展示归并的用户原话与当轮召回快照(原文+得分),辅助判断「真缺知识」还是「有但没检到」;待审可编辑核准答案后通过,写入中展示冻结答案/最近错误并可重试;「立即处理」按钮主动唤醒飞轮 worker
- 评估定时:EVAL_SCHEDULE_ENABLED(默认 true)/ EVAL_SCHEDULE_HOUR(默认 3)/ EVAL_SCHEDULE_TIMEZONE(默认 Asia/Shanghai),每日本地时区到点自动跑 eval-rag(烧额度,可关);跑完经 run_id 幂等落 eval_runs 表供趋势图
- 置信闸升级:hybrid_rerank 主臂改三信号合成 evidence_confidence(Top1 分/有效证据数/Top1-Top2 分差),参数由 `uv run python evals/run_retrieval_compare.py --calibrate-evidence` 校准冻结(2026-10-09:top1=0.7/count=0.3/margin=0.0,min_effective=0.01,threshold=0.15,d_pass=0.100/pass=0.992;产物 evals/calibration/evidence_confidence.json);降级臂与未校准时仍走旧 top1 口径

### 数据库初始化与升级

- 全新环境:`docker compose up -d` 首启自动执行 db/init 全部 DDL(含 ch09 的 07-ddl.sql:review_queue / chat_feedback / eval_runs 三新表 + low_confidence_questions 八新列)
- 已有 ch08 数据卷的升级(不得删卷):`docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch09-ddl.sql`(docker 首启自动含 db/init/07-ddl.sql,新装可跳过)

### 运行约束

- 红线(同 milvus-lite):shell 有代理变量时,起服与访问本机 Langfuse 都要 `no_proxy=127.0.0.1,localhost`(SDK 走 HTTP 会被劫持进代理);curl 本机服务仍加 `--noproxy '*'`
- 审核通过(写回知识库)与建库/向量化/挖掘/选择性重建/全量重建/手工录入共用一把进程级知识库作业锁:锁忙返回 409 job_busy,审核项不变,稍后重试;写入中 = 人工已核准、允许发布的冻结内容,不代表「未发布」
- 选择性重建只删 `knowledge_docs/` 来源的块,保留 `review:<id>` 审核来源与其他来源;全量重建会先清空再从 knowledge_docs 重灌并重放全部「通过/写入中」审核的冻结 FAQ 块
- 飞轮 probe(烧额度、手跑不进 pytest):`uv run python evals/probe_flywheel.py`(标准化/查重标注样例)
