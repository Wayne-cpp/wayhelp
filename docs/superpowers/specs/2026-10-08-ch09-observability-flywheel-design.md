# ch09 · 可观测性与数据飞轮 · 设计文档

日期:2026-10-08
状态:已获用户逐节确认(§1~§6 全部 OK)

## 1. 背景与目标

给客服系统"装上眼睛"(Langfuse 链路追踪 + 按意图的 token 成本统计),再把答不上的问题变成改进燃料(低置信度问题池 → 标准化/查重 → 人工审核 → 写回知识库的数据飞轮),外加自动化评估流水线(定期跑 ch04 评估集,分数落 eval_runs 连成趋势)。

技术选型定死:Langfuse 开源自部署,链路数据不出自家服务器。

## 2. 范围

**做**(对应需求 1~6):

1. Langfuse 接入:docker-compose 并入自托管服务组;图编译后挂一次 LangChain 回调;每条请求完整 trace 树(节点 prompt / 工具调用 / 检索结果 / token / 耗时)。
2. Cost Control:意图打进 trace 元数据,按意图聚合 token 花销,一页可看。
3. 数据飞轮三入口落池:检索证据低(正式版置信闸)/ 生成自评不足(沿用 ch04)/ 用户 👎(新接后端);落池存召回片段快照 Top N(没走检索的入口为 NULL)。
4. 飞轮流水线:标准化(口语→FAQ 式问题+示例答案)→ 模型查重(同义合并、累加次数)→ review_queue 待审;落池后自动异步触发。
5. 评估流水线:复用 ch04 评估集与指标(Recall@K / MRR / Faithfulness),每天凌晨定时跑,分数落 eval_runs,/rag-eval 页看趋势。
6. /review 审核后台页:列表 + 详情(用户原话 + 召回快照)+ 通过(走 ch03 落库写知识库)/ 驳回。

**不做**:低置信度问题按主题归类的微调分类器(下一步的事)。

## 3. 验收标准(用户原文)

1. Langfuse 里能点开任意一条请求,看到完整链路。
2. 问一个知识库没有的问题,拿到兜底话术,且该问题出现在待审队列里;审核页详情能看到用户原话和当时的召回片段。
3. 审核页点通过后,同一个问题再问就能答对——飞轮转完一整圈。
4. 聊天页对一条回答点 👎,这条用户问题落进问题池,标准化查重后出现在待审队列里。
5. 能拿出按意图汇总的 token 花销统计,哪类意图最烧钱看得出来。
6. 评估流水线至少跑两轮,能看到指标趋势对比。

## 4. 关键决策(岔路选型,用户确认 AAA)

| 岔路 | 选定 | 否决项 |
|---|---|---|
| Langfuse 挂载 | LangChain CallbackHandler,astream config 挂一次 | 手动 OTel 埋点(漏埋风险);只挂 main_agent(trace 缺枝) |
| 👎 快照回捞 | LangGraph checkpoint `aget_state_history` 读当轮 state | messages 加快照列(破 messages 表契约);不存(违反需求原文) |
| 成本统计 | 调 Langfuse public API 现算,不加新表 | 本地 usage 表(双写不一致);只看 Langfuse UI(拿不出一页统计) |
| Langfuse 部署 | 并入项目 docker-compose | 官方 compose 单独起;已有实例 |
| 评估定时 | 每天凌晨,lifespan asyncio 周期任务 | 每周;只手动 |
| 飞轮触发 | 落池后 ensure_future 自动跑 + 手动按钮兜底 | 只手动;定时批量 |

## 5. 系统设计

### 5.1 Langfuse 可观测性(§1)

**部署**:docker-compose.yml 追加 Langfuse v3 服务组:langfuse-web、langfuse-worker、ClickHouse、Postgres(Langfuse 专用,不复用项目 MySQL)、Redis/Valkey,加数据卷与 healthcheck,web 暴露 3000 端口。`docker compose up -d` 一条命令起齐。

**配置**(.env.example 新增 ch09 节块,照 ch07/ch08 注释惯例):
- `LANGFUSE_ENABLED`(默认 false)/ `LANGFUSE_HOST`(默认 http://localhost:3000)/ `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`(默认空串 + `has_langfuse_key()` 帮手,照 config.py 现有惯例)。
- 未配齐(ENABLED=false 或密钥空)→ 完全不挂回调,系统行为与现状一致。

**接入点**:`app/services/chat_service.py` 的 `ChatService.stream()` 构造 astream config 处追加 `callbacks=[langfuse_handler]` 与 `metadata={"conversation_id": ..., "user_id": ...}`;`/v1/chat/resume` 的 resume 路径同样处理。`build_chat_graph` 签名不变。

- trace 以 turn 为单位一棵树;LangChain 回调自动覆盖全图节点内 ChatOpenAI 调用(understand/classify/refund_scope/expand/main_agent/摘要),带 prompt/completion/usage/耗时;检索与工具调用以 span 进树。
- `stream_usage=True` 已在 ch05 联调验证,token 消耗自动进 trace。
- **intent 元数据**:classify_intent 节点出结果后回写当前 trace 的 metadata 与 tags(`intent`、`confidence`)。具体 SDK 调用以 Context7 查到的 langfuse Python SDK v3 文档为准(禁止凭记忆写 API)。
- **代理红线**:Langfuse 本地端点与 milvus-lite 同款坑——SDK 走 HTTP,起服命令与 compose 中确保 `no_proxy` 含 localhost/127.0.0.1;README/AGENTS.md 补充说明。

**成本统计**:
- `GET /api/stats/cost-by-intent?days=7|30`:服务端调 Langfuse public API 拉 traces(intent metadata + usage),按意图聚合 `requests / total_tokens / cost`。DeepSeek 单价做配置(`MODEL_INPUT_PRICE_PER_MTOK` / `MODEL_OUTPUT_PRICE_PER_MTOK`,默认可填 0 表示只统计 token)。
- /rag-eval 页新增「成本」区块:分组柱状图(抄现有检索四策略图模式),展示各意图 token 花销与占比。

### 5.2 正式版 evidence_confidence 置信闸(§2)

新模块 `app/knowledge/evidence_confidence.py`:

- 输入:合并重排后的 `RetrievalResult`(hybrid_rerank)。
- 信号:① Top1 相关性分;② 有效证据数量(score ≥ 有效下限的条数);③ Top1−Top2 分差(仅一条时记满分)。
- 输出:0~1 合成单分;权重与阈值**由校准决定,不拍脑袋**。

**校准与冻结**:扩展 `evals/run_retrieval_compare.py` 校准段,在 calibration split(奇数 id,150 条)上扫阈值——沿用现有约束风格:D_absent 误放行率 ≤ 0.10 前提下最大化通过率。冻结进 `settings.rerank_evidence_min_confidence`,注释写明校准日期/指标(照 0.0553 现有钉法)。评估脚本输出信号有效性报告。

**闸位不动,两处共用**:
- `refund_policy` → `route_after_gate`(nodes.py:459)换新判定。
- business 分支 query_faq 工具(app/tools/builtin/faq.py:60-65)同一判定函数。
- 拦下 → 兜底话术不变(REFUSAL_ANSWER)→ 落池(带快照)。
- 仅 `effective_strategy=="hybrid_rerank"` 时用新分;降级到 dense/bm25/hybrid 维持现有阈值,行为不变。

### 5.3 三入口落池 + 召回快照(§3)

- `LowConfidenceRecord`(app/sessions.py:18)加 `retrieved_chunks: list[dict] | None` 字段。
- `_build_low_conf`(app/graph/nodes.py:531)从 `state["retrieval_result"]["hits"]` 截 Top N(`LOW_CONF_SNAPSHOT_TOP_N`,默认 3),每条存 `{chunk_id, score, section_path, question, answer}`,answer 截 500 字符。
  - retrieval_low_conf / self_check:state 里本有 hits,直接截(self_check 是 retrieval ok 后模型拒答,快照同样有价值)。
- **user_feedback(👎 接后端)**:
  - 新端点 `POST /v1/chat/feedback`,body `{conversation_id, reply_index, sentiment: "up"|"down"}`。
  - `down` 时:按会话内第 N 条 assistant 消息定位该轮 → `graph.aget_state_history(thread_id)` 回捞当轮 state 的 `retrieval_result` → 有 hits 截快照;`retrieval_status=="not_run"`(闲聊/业务数据类)或 checkpoint 缺失/对不上 → `retrieved_chunks=NULL`,静默降级不报错("尽力回捞")。
  - 落池 `source='user_feedback'`(ch04 ENUM 已预留),reason 记锚点(reply_index + assistant message_id)。
  - 落池成功后触发飞轮(§5.4)。
  - 校验:会话归属(404 不泄露)、reply_index 越界/重复反馈幂等。
- 前端 chat.html `addFeedback` 改为真发请求;localStorage 保留做按钮锁定与刷新恢复。
- DB 层:`store_db._commit_sync` 写 `retrieved_chunks` 新列;`InMemorySessionStore` 同步扩展(同成同败有测试钉)。

### 5.4 飞轮流水线(§4)

新模块 `app/services/flywheel.py`:`process_pending(settings, session_factory, model) -> dict`。

- 处理对象:`matched_review_id IS NULL` 的 low_confidence_questions 行(批量上限 50/轮)。
- **标准化**:每条一次模型调用,口语原话 → `{normalized_question, suggested_answer}` JSON。prompt 惯例照 EXPAND_PROMPT/JUDGE_PROMPT,输出契约两个字检。
- **查重**:把标准化问题与 review_queue 全部「待审」行拼进一个 prompt,模型判是否同义。命中 → `occurrence_count += 1`、lcq.matched_review_id = 命中行 id;未命中 → 新建 review_queue 行。待审行上限 200 保护(超出按 updated_at 最近活跃截取,注释记档)。
- **触发**:log 节点落池后 `ensure_future` 自动跑;模块级 asyncio.Lock 防重入,运行中又落新池 → dirty 标记,跑完再扫。审核页「立即处理」按钮 `POST /api/flywheel/run` 兜底。
- **容错**:每条 lcq 独立事务;模型输出解析失败 → 跳过该行下轮再试,记日志,不堵队列。
- 纯 Prompt 任务(标准化/查重)不走单测断言字面,用 evals/ 下标注样例集手跑验证(不进 pytest,不烧 CI 额度)。

### 5.5 评估流水线(§5)

- **落表**:JobRunner 跑完 eval-rag 后(无论触发方式)回调解析 `evals/results/rag_eval.json` → 插 eval_runs 行:`triggered_by`('定时'/'手动')、`dataset_size`(test 条数)、`metrics` JSON `{recall_at_k, mrr, faithfulness, coverage, sr10, ...}`。手动按钮路径与定时路径共用同一回调,triggered_by 由触发入口传入。
- **定时**:lifespan 挂 asyncio 周期任务;`EVAL_SCHEDULE_ENABLED`(默认 true)+ `EVAL_SCHEDULE_HOUR`(默认 3,本地时间)。起服计算下次触发点,sleep 到点触发 `job_runner.run("eval-rag", triggered_by="定时")`。单 worker 内存任务,不引 APScheduler。
- **趋势**:`GET /api/eval-runs` 返回历史轮次;/rag-eval 页新增「趋势」区块,自绘 SVG 折线(recall/mrr/faithfulness 三条),按时间排列,下滑一眼可见。

### 5.6 审核后台页(§6)

**/review 新页**(app/main.py FileResponse 模式,app/static/review.html;前端走 Vibe Coding 例外,不套 brainstorm/TDD/code review):

- 列表:状态 tab(待审/通过/驳回)+ 每行:标准化问题、出现次数、AI 示例答案摘要、入队时间。
- 详情抽屉:归并的 lcq 用户原话列表 + 每条 retrieved_chunks 召回快照(原文+得分)。审核人对着片段判断"真缺这块,还是有但没检到"。
- 操作:
  - 通过:核准答案可编辑(默认取 AI 示例答案)→ 走 ch03 `kb_admin.manual_ingest`(category="审核补充",content_type="faq",vectorize=True)→ `review_status='通过'` + 存 `approved_answer`。下次同问题即可检索命中(验收 3)。
  - 驳回:`review_status='驳回'`。
- API(router 薄壳 + asyncio.to_thread 包同步 service,错误契约 `{"error":{"code","message"}}`,照 kb.py 惯例):
  - `GET /api/review?status=&page=`
  - `GET /api/review/{id}/detail`(含归并原话 + 快照)
  - `POST /api/review/{id}/approve`(body: approved_answer)
  - `POST /api/review/{id}/reject`
  - `POST /api/flywheel/run`(立即处理)

## 6. 数据模型变更

以 `sql/ch09-ddl.sql` 为准(已提供,逐字节复制到 `db/init/07-ddl.sql`):

- 新建 `review_queue`(标准化问题/AI 示例答案/occurrence_count/review_status 中文 ENUM/approved_answer)。
- 新建 `eval_runs`(triggered_by 中文 ENUM/dataset_size/metrics JSON/created_at)。
- `low_confidence_questions` ALTER 加 `retrieved_chunks JSON`(AFTER reason)+ `matched_review_id`(FK → review_queue.id ON DELETE SET NULL)。

配套惯例(都有测试钉):
1. `tests/test_ddl_sync.py` 映射加 `("07","ch09")`。
2. `app/db.py` 加 `check_ch09_tables`,`main.py` 启动校验挂载。
3. `app/models.py` 加 ReviewQueue / EvalRuns 两个 Mapped 类,LowConfidenceQuestion 补两列,逐列对齐 DDL(中文 ENUM 用 `Enum(..., name=...)`)。
4. `tests/dbfixtures.py` 的 DDL_PATHS 与 TABLES(先子后父)追加。

## 7. 配置项汇总(.env.example ch09 节块)

| 变量 | 默认 | 说明 |
|---|---|---|
| LANGFUSE_ENABLED | false | 总开关,未配齐密钥自动不挂 |
| LANGFUSE_HOST | http://localhost:3000 | 自托管端点 |
| LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY | 空 | Langfuse 项目密钥 |
| MODEL_INPUT_PRICE_PER_MTOK / MODEL_OUTPUT_PRICE_PER_MTOK | 0 | 成本折算单价(0=只统计 token) |
| RERANK_EVIDENCE_MIN_CONFIDENCE | 校准值 | 正式闸阈值(校准后冻结,注释写来源) |
| LOW_CONF_SNAPSHOT_TOP_N | 3 | 落池快照条数 |
| FLYWHEEL_BATCH_SIZE | 50 | 飞轮单轮处理上限 |
| REVIEW_QUEUE_MATCH_LIMIT | 200 | 查重候选上限保护 |
| EVAL_SCHEDULE_ENABLED | true | 评估定时开关 |
| EVAL_SCHEDULE_HOUR | 3 | 每天本地时间几点跑 |

## 8. 测试策略

非前端全程 TDD(红→绿→重构);纯 Prompt 任务用标注样例集验证替代单测;前端 Vibe 例外。

- DDL/模型:逐字节同步测试映射、check_ch09_tables、models 列对齐。
- 置信闸:合成 RetrievalResult 单测(三信号、边界、降级策略不变);校准脚本改动跑一轮校准验证阈值合理。
- 落池快照:三入口集成测试;👎 端点(归属 404、越界、幂等、checkpoint 回捞命中/降级 NULL)。
- 飞轮:标准化/查重用 evals/ 标注样例(新文件,手跑不进 pytest);合并幂等(同义两条→一行 count=2);事务独立性。
- 评估:eval_runs 落表单测(metrics JSON 结构、triggered_by);定时任务单测(下次触发点计算、开关)。
- Langfuse:enabled/disabled 两态挂载测试;intent metadata 写入测试(handler mock)。
- 成本统计:Langfuse API 响应 mock 的聚合单测。
- 前端:chat.html 反馈字符串断言照旧;review 页人工点验。
- 验收 1~6:落成集成测试/验收脚本,交付前全量 pytest + 人工浏览器点验。

## 9. 风险与红线

- Langfuse v3 容器组较重(ClickHouse 等 ~1GB 内存),compose 里注释说明。
- SDK 版本与项目 langchain-core 兼容性:动手前 Context7 查 langfuse 官方文档核实 callback 用法与版本约束。
- 代理红线:本机访问 Langfuse / Langfuse SDK 访问 localhost 都需 no_proxy,起服命令照 AGENTS.md 现有惯例。
- 评估烧额度:定时每天一轮是用户拍板;EVAL_SCHEDULE_ENABLED 可关。
- checkpoint 回捞依赖 SQLite checkpoints.db 仍在且未清理;缺失时静默降级 NULL。
- review_queue 通过落库走 manual_ingest,复用「先插行拿 id → vectorize_pending」双写幂等,不直写 Milvus(check_consistency 钉两库主键相等)。
