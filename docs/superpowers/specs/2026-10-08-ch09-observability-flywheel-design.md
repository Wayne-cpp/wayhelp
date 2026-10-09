# ch09 · 可观测性与数据飞轮 · 设计文档

日期:2026-10-09(审查修订)
状态:审查修订完成;评估口径与写入状态已由用户确认,待实施

## 1. 背景与目标

给客服系统"装上眼睛"(Langfuse 链路追踪 + 按意图的 token 成本统计),再把答不上的问题变成改进燃料(低置信度问题池 → 标准化/查重 → 人工审核 → 写回知识库的数据飞轮),外加自动化评估流水线(定期跑 ch04 评估集,分数落 eval_runs 连成趋势)。

技术选型定死:Langfuse 开源自部署,链路数据不出自家服务器。

## 2. 范围

**做**(对应需求 1~6):

1. Langfuse 接入:docker-compose 并入自托管服务组;每次图调用统一注入一次请求级 LangChain 回调,并为原生检索/工具边界补固定 span;每条请求完整 trace 树(节点 prompt / 工具调用 / 检索结果 / token / 耗时)。
2. Cost Control:意图打进 trace 元数据,按意图聚合 token 花销,一页可看。
3. 数据飞轮三入口落池:检索证据低(正式版置信闸)/ 生成自评不足(沿用 ch04)/ 用户 👎(新接后端);落池存召回片段快照 Top N(没走检索的入口为 NULL)，同时保存指代消解后的问题和稳定轮次锚点。
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

## 4. 关键决策(用户确认)

| 岔路 | 选定 | 否决项 |
|---|---|---|
| Langfuse 挂载 | LangChain CallbackHandler 覆盖图内模型调用 + SDK 固定边界 span 覆盖原生检索/工具/MCP | 全面手写 OTel(漏埋风险);只挂 main_agent(trace 缺枝) |
| 👎 快照回捞 | LangGraph checkpoint `aget_state_history` 按完成轮专用锚点精确匹配 | 在累积 messages 中查包含关系(会串轮);messages 加快照列(破 messages 表契约) |
| 成本统计 | 调 Langfuse public API 现算,不加新表 | 本地 usage 表(双写不一致);只看 Langfuse UI(拿不出一页统计) |
| Langfuse 部署 | 并入项目 docker-compose | 官方 compose 单独起;已有实例 |
| 评估定时 | 每天凌晨,lifespan asyncio 周期任务 | 每周;只手动 |
| 飞轮触发 | lifespan 单 worker,落池通知 + 到期定时唤醒 + 手动按钮兜底 | 每次落池临时建 task(退避到期无唤醒);只手动 |
| 夜间评估语料 | 固定 `knowledge_docs` 基线,飞轮闭环另做独立验收 | 线上快照(补库会改变 D_absent 标注,趋势不可直接比较) |
| 审核写入后驳回 | 进入“写入中”即冻结核准答案,失败仅可重试,驳回返回 409 | 允许驳回并实现 MySQL/Milvus 清理与恢复 |

## 5. 系统设计

### 5.1 Langfuse 可观测性(§1)

**部署**:docker-compose.yml 追加 Langfuse v3 服务组:langfuse-web、langfuse-worker、ClickHouse、Postgres(Langfuse 专用,不复用项目 MySQL)、Redis/Valkey、S3 兼容对象存储(MinIO),加数据卷与 healthcheck,web 暴露 3000 端口。MinIO 启动时创建 event/media bucket,worker 配置 `LANGFUSE_S3_EVENT_*` / `LANGFUSE_S3_MEDIA_*`(以及需要时的 batch-export)并指向该 bucket。`docker compose up -d` 必须能起齐可用的 v3 服务组,不能只启动 web 后再人工补对象存储。

**配置**(.env.example 新增 ch09 节块,照 ch07/ch08 注释惯例):
- `LANGFUSE_ENABLED`(默认 false)/ `LANGFUSE_HOST`(默认 http://localhost:3000)/ `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`(默认空串 + `has_langfuse_key()` 帮手,照 config.py 现有惯例)。
- 自托管依赖的 `LANGFUSE_S3_EVENT_BUCKET`、`LANGFUSE_S3_ENDPOINT`、`LANGFUSE_S3_ACCESS_KEY_ID`、`LANGFUSE_S3_SECRET_ACCESS_KEY` 等配置只供 compose worker 使用;默认值指向 compose 内 MinIO,密码通过 `.env` 覆盖,不得把默认密钥用于生产。
- 未配齐(ENABLED=false 或密钥空)→ 完全不挂回调,系统行为与现状一致。

**接入点**:`app/services/chat_service.py` 的 `ChatService.stream()` 构造 astream config 处追加 `callbacks=[langfuse_handler]` 与 `metadata={"conversation_id": ..., "user_id": ..., "turn_id": ..., "intent": ..., "confidence": ...}`;`/v1/chat/resume` 的 resume 路径同样处理。`build_chat_graph` 签名不变。trace 配置由一个工厂按请求创建,关闭时 flush;禁用或未配齐密钥时不创建 handler。

- trace 以逻辑 turn 为单位一棵树;LangChain 回调自动覆盖全图节点内 ChatOpenAI 调用(understand/classify/refund_scope/expand/main_agent/摘要),带 prompt/completion/usage/耗时。
- 当前检索器和 `ToolExecutor`/MCP gateway 是普通 Python 调用,不会因为挂了 CallbackHandler 自动出现 observation。必须在固定边界补 SDK span(不引入全局 OTel):`retrieval.search` 记录 requested/effective strategy、命中数、置信分、耗时和截断后的结果摘要;`tool.execute` / `mcp.call` 记录工具名、server、状态、重试次数、耗时和截断结果。参数按现有审计脱敏/截断规则处理,不得把密钥或未截断的大型 payload 发到 Langfuse。
- 每条 trace 的验收必须同时看到模型 generation、检索 span、工具/MCP span(未发生的分支不要求伪造 span);普通闲聊只应有实际发生的节点。
- `stream_usage=True` 已在 ch05 联调验证,token 消耗自动进 trace。
- **intent 元数据**:classify_intent 节点出结果后回写当前逻辑 turn 的 metadata 与 tags(`intent`、`confidence`,后者取现有 `intent_confidence`,不新增另一套置信字段)。逻辑 turn_id 使用稳定的 turn_message_id;trace 工厂为首次调用生成 trace_id 并随本轮 state 持久化,新轮重置、resume 保留。resume 不会再次经过 classify,所以必须从 checkpoint 的 intent/intent_confidence/turn_message_id/trace_id 显式重建 trace context 并注入原意图,不能依赖 handler 的内存缓存。退款选单 resume、工单确认 resume 的模型/工具消耗都必须关联到原 trace 并归到原意图,服务重启后的 resume 也不得进入 unknown。
- 具体 SDK 调用以 Context7 查到的 Langfuse Python SDK v3 文档为准(禁止凭记忆写 API);实现时锁定兼容的 `langfuse` 版本并记录在 `pyproject.toml`。
- **代理红线**:Langfuse 本地端点与 milvus-lite 同款坑——SDK 走 HTTP,起服命令与 compose 中确保 `no_proxy` 含 localhost/127.0.0.1;README/AGENTS.md 补充说明。

**成本统计**:
- `GET /api/stats/cost-by-intent?days=7|30`:服务端按 self-hosted v3 兼容文档读取带 `intent` 的 trace/observation,不能把 `trace.list` 的 observation ID 列表当作 usage。按页读取 generation observations(或使用等价的 metrics API),以 `observation_id` 去重;每个逻辑 turn 计 1 个 request,`total_tokens` 为该 turn 下 generation observations 的 `input + output`(缺一项时使用 `total`,三者都缺则记入缺失计数且不静默当 0)。检索/工具 span 不计入模型 token。优先使用 Langfuse 提供的 input/output cost;成本缺失时按 `MODEL_INPUT_PRICE_PER_MTOK` / `MODEL_OUTPUT_PRICE_PER_MTOK` 以 token/1,000,000 折算,单价均为 0 时 cost=0 且返回 `cost_available=false`。无意图的 trace 聚到 `unknown`,响应包含 `missing_usage_observations`、`incomplete` 和时间窗。
- /rag-eval 页新增「成本」区块:分组柱状图(抄现有检索四策略图模式),展示各意图 token 花销与占比。

### 5.2 正式版 evidence_confidence 置信闸(§2)

新模块 `app/knowledge/evidence_confidence.py`:

- 输入:合并重排后的 `RetrievalResult`(hybrid_rerank)。
- 信号:① Top1 相关性分;② 有效证据数量(score ≥ 有效下限的条数);③ Top1−Top2 分差(仅一条时记满分)。
- 输出:0~1 合成单分;权重、有效分下限和阈值**由校准决定,不拍脑袋**。函数必须同时返回 `confidence_score`、`confidence_threshold`、`low_confidence` 和 `effective_strategy`,供 state、evidence、落池原因共同使用。

**校准与冻结**:扩展 `evals/run_retrieval_compare.py --calibrate-evidence` 校准段,在 calibration split(奇数 id,150 条)上扫权重/有效分下限/阈值——沿用现有约束风格:D_absent 误放行率 ≤ 0.10 前提下最大化通过率。候选与原始报告写入被忽略的 `evals/results/`;经评审的冻结产物写入 **`evals/calibration/evidence_confidence.json` 并纳入版本控制**,内容含 `version`、校准日期、语料/模型标识、权重、有效分下限、阈值、D_absent 误放行率和通过率。冻结值写入 settings(`RERANK_EVIDENCE_MIN_CONFIDENCE` 及对应权重/版本),注释写明校准日期/指标(照 0.0553 现有钉法)。实现交付必须包含真实校准产物,本 spec 不预填未经测量的权重。

日常 `eval-rag` 回归**不重新选择**信号或阈值:读取已冻结 artifact,用生产同一 `evidence_confidence()` 及 `effective_strategy` 降级规则生成 test 指标;artifact 缺失、版本不匹配或生产配置与 artifact 不一致时整轮失败且不发布新报告。报告记录 `evidence_confidence_version`、实际权重/阈值和各策略降级计数;校准模式与夜间回归模式分别测试。

**闸位不动,两处共用**:
- `refund_policy` 在构造 state 字段时调用新判定,再由 `route_after_gate` 分流;`query_faq` 也调用同一判定函数。必须先更新 `confidence_score`、`confidence_threshold`、`low_confidence`、`retrieval_status`、`evidence` 和快照,再进入 route/落池,不能只替换 route 条件。
- 两处都覆盖“旧分判低而新分判可用”和“旧分判可用而新分判低”的回归测试,并验证降级到 dense/bm25/hybrid 时仍使用旧阈值。
- 拦下 → 兜底话术不变(REFUSAL_ANSWER)→ 落池(带快照)。
- 仅 `effective_strategy=="hybrid_rerank"` 时用新分;降级到 dense/bm25/hybrid 维持现有阈值,行为不变。

### 5.3 三入口落池 + 召回快照(§3)

- `LowConfidenceRecord`(app/sessions.py:18)加 `retrieved_chunks: list[dict] | None`、`resolved_question: str | None`、`turn_message_id: int | None` 字段。`raw_question` 仍保留用户原话;`resolved_question` 来自当轮 `resolved_query`,用于离线标准化;`turn_message_id` 是稳定的用户轮次锚点,等于 prepare 已提交的用户行 id,与既有 `source_message_id` 含义一致。
- **完成轮锚点**:ChatGraphState 新增 `turn_message_id` / `final_assistant_message_id`。`new_turn_state` 设置本轮用户行 id 并清空最终回答 id;resume 保留本轮锚点。log 提交成功后必须同时返回带 `db_id` 的 `turn_messages`、累积 `messages` 和两个锚点,其中最终回答 id 取本轮最后一条已落库 assistant 行。当前 log 仅回写累积 `messages`,需补齐本轮字段,不能只在局部变量盖章。
- `_build_low_conf`(app/graph/nodes.py:531)从 `state["retrieval_result"]["hits"]` 截 Top N(`LOW_CONF_SNAPSHOT_TOP_N`,默认 3),每条存 `{chunk_id, score, section_path, question, answer}`,answer 截 500 字符。
  - retrieval_low_conf / self_check:state 里本有 hits,直接截(self_check 是 retrieval ok 后模型拒答,快照同样有价值)。
- **user_feedback(👎 接后端)**:
  - 新端点 `POST /v1/chat/feedback`,body `{user_id, conversation_id, assistant_message_id, sentiment: "up"|"down"}`。正常聊天与 resume 在 log checkpoint 成功写入、图正常完成后,由 ChatService 发 SSE `turn_committed` 帧 `{conversation_id, turn_message_id, assistant_message_id}` 再发 done;挂起/失败轮不发该帧。所有 API/SSE 消息 id 用十进制字符串传输,避免前端 BIGINT 精度丢失,禁止自行计算序号。
  - **先校验账本**:两个 sentiment 均先校验 `user_id` 对会话归属(统一 404),再读该会话 messages。合法最终回答是用户行之后、下一用户行之前,最后一条有正文且无 tool_calls 的 assistant 行;中间工具调用 assistant、跨会话/不存在 id 返回 409。对应用户行提供可靠的 `turn_message_id` 和 `raw_question`;未完成的挂起轮没有可反馈回答。不能因 checkpoint 缺失跳过此校验。
  - **再尽力回捞**:`down` 遍历 checkpoint history,仅接受 `state.final_assistant_message_id == 请求 id` 且 `state.turn_message_id == 账本用户 id` 的完成轮;再校验已盖章 `turn_messages` 的首尾 id。**禁止在跨轮累积 `state.messages` 中按包含 db_id 定位**,后续每轮都会含旧消息。匹配后取当轮 `resolved_query`/`retrieval_result`;没走检索或 checkpoint 缺失/旧格式/锚点不符时 `retrieved_chunks=NULL`,缺失解析问题时 `resolved_question=NULL`,仍用已验证的用户原话落池。
  - 落池 `source='user_feedback'`(ch04 ENUM 已预留),reason 记 `assistant_message_id`、`turn_message_id` 和回捞状态(命中/未检索/缺失/不匹配)。复用 Top N 与 answer 500 字符截断规则。
  - **持久幂等**:新增 `chat_feedback`,两个 sentiment 都写入,唯一键为 **`(conversation_id, assistant_message_id)`**,不包含 sentiment。`down` 的反馈行与 lcq 在同一事务提交,反馈行以 `low_confidence_question_id` 指向本次 lcq;`up` 不建 lcq。重复同向请求返回原记录,反向返回 409;并发唯一键冲突后读取既有记录判定,不能留下多余 lcq 或半条反馈。成功提交后才通知飞轮(§5.4)。
- 前端 chat.html `addFeedback` 改为真发请求,收到 `turn_committed` 才启用本轮按钮;失败请求不锁按钮。历史 messages API 补 `feedback_eligible` 与已持久化 sentiment,只给最终回答显示按钮。localStorage 仅作显示缓存,服务端状态为准,刷新/换设备/重启仍保持幂等。
- DB 层:`store_db._commit_sync` 同事务写入上述 LowConfidenceRecord 字段;`InMemorySessionStore` 同步扩展锚点/反馈契约(同成同败有测试钉)。消息表仍不存检索快照或 tool 结果。

### 5.4 飞轮流水线(§4)

新模块 `app/services/flywheel.py`:lifespan 管理一个异步 worker;内部 `async process_pending(settings, session_factory, model) -> dict` 处理一批,只能由该 worker 调用。模型调用异步 await,同步 DB 操作用 `asyncio.to_thread`,接口只通知 worker,不在请求内等待模型。

- **状态/处理对象**:lcq 新增 `process_status=pending|processed|failed`。只处理 `process_status='pending' AND matched_review_id IS NULL AND (next_attempt_at IS NULL OR next_attempt_at <= 数据库当前时间)` 的行,按 `id ASC` 取 `FLYWHEEL_BATCH_SIZE`(默认 50)。完成一批后继续扫直到没有到期行,超过 50 条也会排空;成功归并同时写 `processed` 与 `matched_review_id`,失败终态 `failed` 不再自动选择。review_queue 本章没有硬删除入口,正常运行中 processed 行必须有归并目标。
- **标准化**:每条一次模型调用,输入同时包含 `resolved_question`(有则为主输入,缺失则用原话)与 `raw_question`(语气/原话参考),输出 `{normalized_question, suggested_answer}` JSON。prompt 惯例照 EXPAND_PROMPT/JUDGE_PROMPT,输出契约严格校验。
- **查重**:单 worker 防重入,先读“待审”候选快照再调用模型,不跨模型调用持有 DB 行锁。候选上限 `REVIEW_QUEUE_MATCH_LIMIT`(默认 200),超出按 `updated_at DESC, id DESC` 截取。模型命中 id 必须来自该候选集;写回事务锁住当前 lcq 并复核仍为 pending/未归并,再锁目标 review 复核仍为待审。目标状态变化则重新读取候选查重,不能归到写入中/通过/驳回。命中计数 `+1` 或新建 review、lcq 指针及 processed 状态在同一事务提交,成功时清空 next_attempt_at/last_error,保留 attempt_count 作历史记录;已处理的重放直接跳过,不会重复累加。
- **触发/唤醒**:worker 启动即扫描数据库,包含尚未到期的重试行。log 或反馈事务提交后设置应用级 `asyncio.Event`;「立即处理」按钮同样设置 Event 并返回 202,不绕过退避或另建处理 task。无到期行时查最早 `next_attempt_at`,等待 Event 或该到期时间,到时自动再扫;没有未来行时只等 Event。先清 Event 再扫描/计算等待,确保等待建立期间的新通知不会丢失。队列与重试时间以 DB 为准,Event 只加速唤醒,聊天请求不等待飞轮。
- **容错/重试**:每条 lcq 独立事务。模型输出解析失败或上游异常 → `attempt_count += 1`,保存截断的 `last_error`,退避秒数 `min(FLYWHEEL_RETRY_MAX_SECONDS, FLYWHEEL_RETRY_BASE_SECONDS * 2 ** (attempt_count - 1))`,用 DB 时钟写 `next_attempt_at`,跳过并处理后续行。达到 `FLYWHEEL_MAX_ATTEMPTS`(默认 5)时转 failed、清空 next_attempt_at,在审核页“处理失败”视图显示原话、错误和次数。人工重试仅接受 failed 行,以行锁 CAS 改为 pending、计数归零并清错误/重试时间,提交后通知 worker;重复重试返回 409。
- **生命周期**:服务启动恢复积压并按未来到期时间唤醒;关闭时 cancel/wait worker,取消不计作模型失败,已开始的 DB 事务须等提交/回滚完成再释放资源。worker 自身 DB/循环异常记录日志并按有上限的退避恢复,不能退出后静默停止,也不能忙循环。单 Uvicorn worker 约束保留。
- 纯 Prompt 任务(标准化/查重)不走单测断言字面,用 evals/ 下标注样例集手跑验证(不进 pytest,不烧 CI 额度)。

### 5.5 评估流水线(§5)

- **语料口径(用户确认)**:保留当前 evaluator 的隔离 MySQL/Milvus 构建方式,只从 `knowledge_docs` 建固定基线,不导入线上 review 补库或手工块。生产 `evidence_confidence()` 与冻结参数共用,语料保持隔离。报告 meta 及 `eval_runs` 记录 `corpus_mode='knowledge_docs_baseline'`、`corpus_version`(按相对路径稳定排序,汇总实际导入文档的路径/内容 SHA-256)、`dataset_version`(评估集文件原始内容 SHA-256);趋势只连接语料/评估集/置信闸版本相同的轮次,版本变化显示为新组。固定基线分数不代表线上补库效果;验收 2~4 使用独立测试知识库/会话验证“缺知识拒答 → 入池待审 → 通过 → 再问答对”,不改 ch04 的 D_absent 标注或混入基线趋势。
- **落表**:JobRunner 跑完 eval-rag 后,仅在读取到新的、契约校验通过的 `meta.run_id` 时调用一次 `on_report_published(report, triggered_by)`;回调在 `rag_eval.json` 原子替换成功后执行,失败不插半条。以 `run_id` 唯一幂等,重复回调不重复插入。写入 `eval_runs`: `run_id`、`triggered_by`('定时'/'手动')、`dataset_size`(`meta.test_cases`)、meta 的 corpus_mode/corpus_version/dataset_version、`metrics` JSON(见 §6)和 `created_at`。`JobRunner.run(name, *, triggered_by)` 记录触发来源,手动路由传“手动”,调度器传“定时”。
- **定时**:lifespan 创建一个可取消的 asyncio task;`EVAL_SCHEDULE_ENABLED`(默认 true)+ `EVAL_SCHEDULE_HOUR`(默认 3)+ `EVAL_SCHEDULE_TIMEZONE`(默认 `Asia/Shanghai`)。用 timezone-aware `zoneinfo` 计算下一次本地 03:00,DST 重复时同一自然日只运行一次;服务关闭先 cancel/wait 该 task,任务异常记录日志后计算下一天,不把异常传播到 lifespan。单 worker 内存任务,不引 APScheduler。
- **趋势**:`GET /api/eval-runs` 返回历史轮次;/rag-eval 页新增「趋势」区块,自绘 SVG 折线(recall/mrr/faithfulness 三条),按时间排列,下滑一眼可见。

### 5.6 审核后台页(§6)

**/review 新页**(app/main.py FileResponse 模式,app/static/review.html;前端走 Vibe Coding 例外,不套 brainstorm/TDD/code review):

- 列表:状态 tab(待审/写入中/通过/驳回)+ 每行:标准化问题、出现次数、AI 示例答案摘要、入队时间;“写入中”行展示核准答案、最近写入错误与重试入口,不再提供编辑/驳回。另设 lcq“处理失败”视图,展示尚未生成审核项的失败原话与重试入口。
- 详情抽屉:归并的 lcq 用户原话列表 + 每条 retrieved_chunks 召回快照(原文+得分)。审核人对着片段判断"真缺这块,还是有但没检到"。
- 操作:
  - **通过/写入**:待审时核准答案可编辑(默认取 AI 示例答案),提交给 `review_service.approve(review_id, approved_answer)`。适配层必须先获得 `kb_admin` 与 ingest/vectorize/reset/rebuild 共用的进程级知识库作业锁,再取得 review 行锁,锁顺序固定为 **知识库锁 → review 行锁**。知识库忙返回 409 `job_busy`,不改变审核项;不能另建一把不互斥的审核锁,也不能在已持锁时嵌套调用再次取锁的管理入口。
  - **冻结事务**:取得锁后复核待审,将 `review_status='写入中'` 与 `approved_answer`、新建/复用的 pending chunks 及 `knowledge_chunk_ids` 在同一 MySQL 事务提交。知识块显式设置 `source_doc='review:<id>'`、`content_type='faq'`、`category='审核补充'`,问题取 normalized_question;复用 ch03 FAQ 切分逻辑,稳定的 `chunk_index` 从 1 起,利用既有唯一键 `(source_doc, chunk_index)` 防重复并维护 prev/next 指针。`knowledge_chunk_ids` 可由 source_doc 重建,不以旧 id 盲目复用其他来源的块。不能把不支持 category/content_type 的 `manual_ingest` 直接当作该契约。
  - **向量化/完成**:冻结事务提交后释放 DB 行锁,继续持有知识库锁,复用 `vectorize_pending` 的 embedding/upsert/pending→done 流程;它扫描全库 pending,因此“写入中”必须已代表人工核准、允许发布的冻结内容。所有本 review 块 done 且 MySQL/Milvus 对应向量确认存在后,短事务以行锁 CAS 将“写入中”改为“通过”,写 `approved_at` 并清 `last_write_error`,再释放知识库锁。失败则保留写入中与冻结答案/块,保存截断的 last_write_error 并返回错误;不回退待审,部分向量可能已经可检索,UI 不能把写入中解释为“未发布”。
  - **重试/幂等**:写入中只允许省略 approved_answer 或提交与冻结值相同的答案,复用本 review 块补齐向量化/最终状态;不同答案、对驳回项 approve 返回 409。已通过时同样拒绝答案变更,相同请求返回原 knowledge_chunk_ids。并发 approve 由共用知识库锁串行(竞争者先得 409,之后可重试),review 行锁/CAS 保证审核与驳回竞争只有一种结果;外部 embedding 调用不持有 DB 行锁。成功后再问同问题可命中(验收 3)。
  - **驳回(用户确认)**:只允许行锁/CAS `待审 → 驳回`,此时尚未创建本 review 知识块;重复驳回幂等。写入中/通过的驳回一律 409,不删除块,不改变已冻结答案。无重开或硬删除审核项的入口。
- API(router 薄壳,同步审核/DB service 用 asyncio.to_thread;flywheel 异步入口直接通知 worker。错误契约 `{"error":{"code","message"}}`,照 kb.py 惯例):
  - `GET /api/review?status=&page=`
  - `GET /api/review/{id}/detail`(含归并原话 + 快照)
  - `POST /api/review/{id}/approve`(body: approved_answer;首次通过必填非空,写入中重试可省略)
  - `POST /api/review/{id}/reject`
  - `POST /api/flywheel/run`(202 通知处理到期 pending)
  - `GET /api/flywheel/failures?page=`(失败 lcq 列表与错误)
  - `POST /api/flywheel/questions/{id}/retry`(仅 failed → pending,202)

**知识库维护/恢复**:review_queue 的冻结问题与答案是审核知识的恢复源,此来源与 knowledge_docs 并存,实施时同步更新 README/AGENTS.md 的“唯一源”说明。

- 所有知识库写入口共用上述作业锁,覆盖建库、向量化、挖掘、reset、full rebuild 与审核写回;当前未取锁的 `manual_ingest` 也必须纳入互斥,覆盖其 MySQL 插块至可选向量化全过程。只读 preview/search 不取写锁,CLI 直写仍遵守先停服的现有红线。
- 选择性 reset 只删除 `knowledge_docs/` 来源命名空间下的文档块(按 `resolve_source_doc` 的规范路径识别,包含已移除文件的旧块),必须保留 `review:<id>` 与其他来源;不能沿用 `source_doc IS NOT NULL` 广删条件。
- full rebuild 仍先预检,进入 rebuilding 后清知识表/向量集合,再以同一知识库锁重灌 knowledge_docs,重建全部“通过”与“写入中” review 的冻结 FAQ 块。按新主键刷新 knowledge_chunk_ids 与 prev/next 指针,不删除 review_queue 或 lcq/反馈。重放/向量化后复核每项块,写入中可收敛为通过;已通过项保留原 approved_at。
- 重放函数接受已持知识库锁的调用,不递归取锁;失败保持 `rebuild_required`,保留审核恢复源,下次 rebuild 可继续重建。只有审核来源全部恢复且 pending 归零、指针完整、MySQL/Milvus 主键一致性终验通过后才能置 READY。approve 在知识库状态不是 READY 时返回 409,由 rebuild 恢复全局一致性。

## 6. 数据模型变更

本次设计配套 `sql/ch09-ddl.sql`;实施时逐字节复制到 `db/init/07-ddl.sql` 并纳入启动/测试路径,本次不执行 DDL 或提前启用部署脚本:

- 新建 `review_queue`(normalized_question/ai_suggested_answer/occurrence_count/review_status 中文 ENUM **待审、写入中、通过、驳回**/approved_answer/approved_at/knowledge_chunk_ids JSON/last_write_error/created_at/updated_at)。写入中和通过必须有 approved_answer;approved_at 只在完成通过后设置。knowledge_chunk_ids 为恢复索引,可按 source_doc 刷新。
- 新建 `chat_feedback`(conversation_id/assistant_message_id/turn_message_id/sentiment up|down/low_confidence_question_id/created_at)。唯一键 `(conversation_id, assistant_message_id)`;会话/两类消息/关联 lcq 外键均沿用账本默认 RESTRICT,本章没有删除反馈入口。up 的 lcq 指针为 NULL,down 与指向的 lcq 同事务创建;服务校验关联对象属于同一会话/轮次,不依赖单列外键替代归属校验。
- 新建 `eval_runs`(`run_id VARCHAR(64)` 唯一、triggered_by 中文 ENUM/dataset_size/corpus_mode/corpus_version/dataset_version/metrics JSON/created_at)。两个 version 为 SHA-256 十六进制内容摘要,metrics 固定保存 `{online_strategy, recall_at_10, mrr, faithfulness, coverage, sr10, quality_passed, evidence_confidence_version}`;缺失指标存 null,不得用 0 伪造。
- `low_confidence_questions` ALTER 加 `retrieved_chunks JSON`(AFTER reason)+ `resolved_question TEXT`+ `turn_message_id BIGINT UNSIGNED`(历史/非会话来源可 NULL,不挂新消息外键)+ `matched_review_id`(FK → review_queue.id ON DELETE SET NULL)+ `process_status ENUM('pending','processed','failed') DEFAULT 'pending'`+ `attempt_count INT UNSIGNED DEFAULT 0`+ `next_attempt_at DATETIME`+ `last_error TEXT`。旧行自动成为 pending;索引 `(process_status, next_attempt_at, id)` 支持到期处理/失败列表。review 不硬删;若维护脚本删除导致 processed 指针为空,必须显式修复为 pending 再恢复处理。

实施时补齐下列配套与测试:

1. `tests/test_ddl_sync.py` 映射加 `("07","ch09")`。
2. `app/db.py` 加 `check_ch09_tables`,`main.py` 启动校验挂载。
3. `app/models.py` 加 ReviewQueue / ChatFeedback / EvalRun 三个 Mapped 类,LowConfidenceQuestion 补齐上述全部新列,逐列对齐 DDL(ENUM 用 `Enum(..., name=...)`,含 pending/processed/failed)。
4. `tests/dbfixtures.py` 的 DDL_PATHS 与 TABLES(先 chat_feedback,再 low_confidence_questions,再 review_queue 等父表)追加;审核补充知识的 source 标识与 reset/full rebuild 重放契约一并钉住。

## 7. 配置项汇总(.env.example ch09 节块)

| 变量 | 默认 | 说明 |
|---|---|---|
| LANGFUSE_ENABLED | false | 总开关,未配齐密钥自动不挂 |
| LANGFUSE_HOST | http://localhost:3000 | 自托管端点 |
| LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY | 空 | Langfuse 项目密钥 |
| LANGFUSE_S3_* | compose 内 MinIO 默认值 | v3 event/media blob storage,生产改为独立密钥与 bucket |
| MODEL_INPUT_PRICE_PER_MTOK / MODEL_OUTPUT_PRICE_PER_MTOK | 0 | 成本折算单价(0=只统计 token) |
| RERANK_EVIDENCE_MIN_CONFIDENCE | 校准值 | 正式闸阈值(校准后冻结,注释写来源) |
| EVIDENCE_CONFIDENCE_VERSION | 校准 artifact 版本 | 线上与夜间评估必须一致 |
| LOW_CONF_SNAPSHOT_TOP_N | 3 | 落池快照条数 |
| FLYWHEEL_BATCH_SIZE | 50 | 飞轮单轮处理上限 |
| REVIEW_QUEUE_MATCH_LIMIT | 200 | 查重候选上限保护 |
| FLYWHEEL_RETRY_BASE_SECONDS | 60 | 模型/解析失败的指数退避基数 |
| FLYWHEEL_RETRY_MAX_SECONDS | 3600 | 自动重试与 worker 异常恢复的退避上限 |
| FLYWHEEL_MAX_ATTEMPTS | 5 | 连续失败次数上限,到限转 failed 等人工重试 |
| EVAL_SCHEDULE_ENABLED | true | 评估定时开关 |
| EVAL_SCHEDULE_HOUR | 3 | 每天本地时间几点跑 |
| EVAL_SCHEDULE_TIMEZONE | Asia/Shanghai | 定时任务使用的 IANA 时区 |

## 8. 测试策略

非前端全程 TDD(红→绿→重构);纯 Prompt 任务用标注样例集验证替代单测;前端 Vibe 例外。

- DDL/模型:逐字节同步测试映射、check_ch09_tables、models 列对齐。
- 置信闸:合成 RetrievalResult 单测(三信号、边界、降级策略不变);校准 artifact 形状/版本/参数冻结测试及干净检出可读取;夜间回归拒绝隐式重新校准;旧低→新可用、新可用→新低两处闸位测试。
- 落池快照:三入口集成测试;反馈归属 404、非法/跨会话/中间工具 assistant 的 409;两轮不同检索快照后反馈旧回答仍命中旧快照;新轮显式清最终回答锚点;正常/resume 完成的 SSE 锚点与 checkpoint 一致,挂起轮不发 turn_committed;checkpoint 缺失/旧格式仍按账本原话落池 NULL 快照。
- 反馈幂等:up/down 同向并发、反向冲突、重启与历史回载保持原状态;down 插 lcq 或反馈失败时整体回滚,不产生重复 lcq;BIGINT 字符串与前端只给最终回答反馈。
- 飞轮:标准化/查重用 evals/ 标注样例(新文件,手跑不进 pytest);fake model/fake clock 钉同义两条→一行 count=2、事务重放不重复累加、候选审核状态变化后复查;失败不堵后续;超过 50 条排空;**无人新落池时退避到期自动重试**,重启时只有未来行也能唤醒;Event 等待竞争不丢通知;到限失败可见/人工重试 CAS;关闭取消及 worker 异常恢复。
- 审核写回:重复/并发 approve 无重复 chunks、知识库忙 409 无状态变更、approve 与 reset/rebuild/vectorize/manual_ingest 互斥;冻结事务原子性;向量化失败保留写入中/冻结答案且可重试;失败后 reject=409,后续全库 vectorize 只发布已核准内容;approve/reject 竞争只有一种结果;不同答案重试=409。
- 知识库恢复:reset 保留 review 来源并清理已删除文档旧块;full rebuild 重放通过/写入中、刷新 chunk ids/指针、保留 approved_at、最终仍可检索;重放失败不置 READY,下次 rebuild 能恢复;核准前的待审/驳回从不重放。
- 评估:eval_runs 落表单测(run_id 幂等、metrics/语料版本、triggered_by);固定基线不加载线上审核块、D_absent 标注不变、异版本趋势不连线;飞轮闭环独立集成验收;定时任务单测(时区、DST 同日去重、取消、开关、异常后续调度);新报告发布后回调只执行一次。
- Langfuse:enabled/disabled 两态挂载测试;模型、retrieval、tool/MCP span mock;resume 两类路径 intent 归属测试。
- 成本统计:分页 observations/metrics 响应 mock 的聚合单测(usage 缺失、去重、unknown、单价折算、请求数口径)。
- 前端:chat.html 反馈字符串断言照旧;review 页待审/写入中/通过/驳回与处理失败重试人工点验。
- 验收 1~6:落成集成测试/验收脚本,交付前全量 pytest + 人工浏览器点验。

## 9. 风险与红线

- Langfuse v3 容器组较重(ClickHouse 等 ~1GB 内存),compose 里注释说明。
- Langfuse v3 还依赖对象存储;MinIO 数据卷、bucket 和 worker 环境变量必须与 web/worker 一起验收。
- SDK 版本与项目 langchain-core 兼容性:动手前 Context7 查 langfuse 官方文档核实 callback 用法与版本约束。
- 代理红线:本机访问 Langfuse / Langfuse SDK 访问 localhost 都需 no_proxy,起服命令照 AGENTS.md 现有惯例。
- 评估烧额度:定时每天一轮是用户拍板;EVAL_SCHEDULE_ENABLED 可关。
- checkpoint 回捞依赖 SQLite checkpoints.db 仍在且未清理;缺失时静默降级 NULL,但消息/最终回答/对应用户原话必须经账本校验。累积 messages 的包含关系不能证明当轮检索快照。
- 审核核准后以 review_queue 为可恢复源;写入中表示可能部分发布,仅可重试,不得驳回或修改答案。reset 保留 review 来源,full rebuild 在同一知识库锁下重放通过/写入中并刷新主键;不能只用 review 行锁保护跨库写入。
- 夜间趋势只衡量固定 knowledge_docs 基线;飞轮补库效果以独立闭环验收证明。校准冻结产物必须随仓库提交,不能放在忽略的 evals/results/。
- cost API 的 usage 字段可能分散在 observation 而非 trace list;实现必须遵循 self-hosted v3 兼容 API 的分页/字段组契约,缺失 usage 要可见地标记 incomplete。
