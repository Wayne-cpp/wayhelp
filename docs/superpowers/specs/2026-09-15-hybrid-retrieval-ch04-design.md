# 电商智能客服 Ch04: RAG 进阶 · 混合检索 + 重排 + 生成质量控制 + 评估体系 - 设计文档

日期: 2026-09-15
前置: ch03(向量语义检索)已上线;Milvus 集合 `knowledge` 现为两列(id + vector),原文权威源在 MySQL `knowledge_chunks`。
本文档是与用户对齐后的实施依据;技术选型为用户定死:Milvus 原生 BM25 + hybrid_search RRF 融合 + bge-reranker-v2-m3 重排。
2026-09-15 评审修订:补齐评估集/语料一致性、确定性拒答、重试降级、scope 过滤、引用 trace、分片校准、存量 DDL 升级与重建维护状态等契约。

## 1. 目标与范围

把检索质量做上一个台阶:混合检索(dense + BM25,RRF 融合)、bge-reranker-v2-m3 精排、元数据过滤、Query 理解、生成质量控制(引用/拒答/低置信度池)、四策略评估体系,以及聊天页配套(可点引用角标 + 满意度反馈)。

验收标准(用户原话):
1. 四策略对比报告能跑出数字。
2. 问带具体型号的问题,BM25 那路能命中。
3. 答案引用编号能定位回原文,聊天页上点引用能看到来源原文。
4. 问知识库没有的内容,得到明确拒答,且问题进了低置信度池。

本章不做(用户原话):指代消解、多轮改写。
也不做:faith_cases 处置界面(只落表,处置用 SQL);满意度反馈的后端落库(纯前端采集,留入口)。

## 2. 架构与选型(已与用户逐项确认)

| 决策点 | 结论 |
|---|---|
| 集成方式 | **retriever 内部升级**:`KnowledgeRetriever` 内部变四级管道,策略参数化,评估与在线同一代码路径 |
| 重排接入 | **硅基流动远程 rerank API**(`POST /v1/rerank`,model=`BAAI/bge-reranker-v2-m3`),复用 embedding 同套凭证面 |
| 部署形态 | **留在 Milvus Lite**(锁定 3.2.1 已核实支持 BM25/sparse/hybrid_search/RRF/Jieba);不切 Standalone |
| 集合迁移 | **同名重建**:Lite 不支持 schema 变更,drop 旧 `knowledge` 集合 → 新 schema 建同名集合 → 从 MySQL 重灌 |
| 重建作业语义 | **全量重置**:清空 MySQL 全部 knowledge 块(含挖掘块)+ drop 集合 + 重新 ingest 当前 knowledge_docs/ + 向量化;挖掘进度一并清 |
| 元数据过滤入口 | `query_faq` 工具加可选 `scope` 参数,只允许有限知识范围枚举;不再把现有 `category` 误作商品品类 |
| 拒答入池范围 | 检索低置信 + 生成自评不足**两种都入池**,靠 `source` 列区分;`user_feedback` 枚举留后续章节 |
| Query 改写模型 | 复用主对话模型(OPENAI_BASE_URL/MODEL_NAME),失败降级原问法直查 |
| 评估集 | 用户提供 `evals/retrieval_compare.txt`(300 条 CSV,已就位);loader 按此格式写 |
| 评估分片 | 每桶奇数编号 30 条为 calibration,偶数编号 30 条为 test;阈值只看 calibration,正式验收只看 test |
| 知识文档 | 以 300 条评估集为业务需求基线;四份 md 除补 frontmatter 外,还须补齐评估正例要求但当前缺失的业务知识 |
| Rerank 恢复 | reranker 内部最多重试 2 次;耗尽后降级 hybrid,不交给 ToolExecutor 重跑整条检索 |
| 重建可用性 | 破坏性步骤前完成全部预检;重建期间进入显式维护状态,终验通过后恢复 |
| DDL 升级 | 全新库走 db/init;已有数据卷显式执行 ch04 DDL,应用启动时校验两张表存在 |
| faith_cases 界面 | 本章不做,只落表 + 报告 |

## 3. 项目结构(新增/改动)

```
app/knowledge/
  query_understanding.py   新增:改写 + 同义词扩展(主模型,纯函数式封装,可注入假模型)
  reranker.py              新增:硅基流动 rerank 客户端(httpx,内部超时/重试 + 降级结果)
  retriever.py             重构:四级管道 + RetrievalResult + 策略参数 + scope 过滤
  milvus_store.py          扩展:五列 schema、双路 search、hybrid_search、重建支持
  ingest.py                小改:upsert 行带 text/scope,scope 统一派生
app/services/kb_admin.py   扩展:rebuild_index 预检 + 维护状态 + 全量重置
app/tools/business.py      改:query_faq 加 scope 可选参 + 出参带 ref_no + RetrievalTrace
app/services/chat_service.py 扩展:TurnToolset、citations SSE、后端强制拒答、原子入池
app/prompts/service.py     扩展:引用协议 + 拒答话术 + 负面知识禁令
app/prompts/               新增:query_understanding / faithfulness judge 提示词
app/models.py              新增:LowConfidenceQuestion / FaithCase ORM
app/routers/chat.py        小改:citations 帧序列化分支
app/routers/kb.py          小改:rebuild 接口
app/schemas.py             小改:probe strategy/scope 参数
app/sessions.py            扩展:commit_turn 可携带低置信记录
app/store_db.py            扩展:消息 + 低置信记录同事务提交
app/tool_envelope.py       扩展:v2 metadata 保存引用快照,兼容读取 v1
app/main.py                改:主模型注入 retriever、TurnToolset 装配、启动 DDL 校验
db/init/04-ddl.sql         新增(与 sql/ch04-ddl.sql 逐字节一致)
sql/ch04-ddl.sql           注释修订(两种拒答都入池;bucket 含 D_absent),结构不动
evals/run_retrieval_compare.py  新增:四策略对比 + 生成段 Faithfulness
evals/retrieval_compare.txt     评估集(用户提供,300 条;修正 D3 冲突文案)
app/static/chat.html       引用角标可点 + 👍/👎 反馈(vibe,不套流程)
app/static/kb.html         重建索引按钮(vibe)
knowledge_docs/*.md        补 frontmatter + 评估集要求的缺失知识
README.md                  补 ch04 存量 DDL 升级、重建维护说明与评估命令
```

## 4. 数据契约

### 4.1 Milvus 集合 `knowledge`(新 schema,同名重建)

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 主键,auto_id=False | 与 MySQL chunk id 1:1(第一不变量不变) |
| `vector` | FLOAT_VECTOR(1024),COSINE | dense 语义向量 |
| `text` | VARCHAR(max_length 4096),enable_analyzer=True | BM25 语料 = `vector_text` 三格拼接(与向量同源,保证两路语料一致) |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出列;SPARSE_INVERTED_INDEX,metric BM25 |
| `scope` | VARCHAR(32) | 有限知识范围枚举,供参数化元数据过滤 |

- BM25 函数:`Function(name="bm25_fn", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])`,`schema.add_function(...)`。
- 分词器:Lite 用 Jieba 中文分词;参数形态(`{"tokenizer": "jieba"}` vs `{"type": "chinese"}`)实现期以 Context7 + `run_analyzer` 实测核销(脆点 C1)。
- `scope` 只允许 `faq` / `policy` / `product_spec` / `after_sales_manual` / `qa_mined` / `manual`;由 `content_type + source_doc` 通过单一纯函数派生。`product-specs.md` 与 `after-sales-manual.md` 分别映射专用 scope,其他手工录入为 `manual`。MySQL 现有 `category` 继续表示文档/章节分类,不改语义,不作为过滤参数。
- `scope` 过滤必须使用 `expr_params`,不得把模型传入值直接拼进表达式。工具层先用 Literal/枚举拒绝未知 scope。
- `ensure_collection` 契约校验升级:存在集合必须是五列新 schema(字段名/类型齐、主键 INT64、dim=1024、函数与索引齐),旧两列 schema 报「需重建索引」,**不自动重建**(沿用 ch03 原则)。
- `MilvusKnowledgeStore` 方法扩展:`search_dense`(现 search 改名/参数化)、`search_bm25(data=[文本])`、`hybrid_search(dense_vec, bm25_text, scope)`;全部走既有 `_ensure_loaded` 入口。

### 4.2 MySQL `low_confidence_questions`(用户 DDL,结构不动)

原话、来源会话、入池入口(source ENUM)、判不能原因、时间。**注释修订**:本章两种拒答都入池(`retrieval_low_conf` / `self_check`),`user_feedback` 留后续章节。

入池记录不是旁路 best-effort 写入。`SessionStore.commit_turn` 接受可选低置信记录,DB adapter 在同一事务内写本轮 messages、更新 conversation、插入 `low_confidence_questions`;任一失败整体回滚。内存 adapter 保存等价记录供测试断言。

### 4.3 MySQL `faith_cases`(用户 DDL,结构不动)

编造个案台账:一题一行(uk_eval_id),重判更新快照 + seen_count+1;已解决复发退回「未解决」并清空 resolution。`citations` JSON 存当轮**实际交给生成模型**的证据全集快照(角标 `[ref_no]` = 列表序号),不得存预算裁剪前的候选。bucket 注释补 D_absent。

检索语料和临时 Milvus 仍使用独立评估库;`faith_cases` 明确写入 `DATABASE_URL` 对应的主业务库,跨评估运行累计 `seen_count`。评估脚本只在启动时校验主库 ch04 表,不得创建/删除主库表,也不得修改主库 knowledge 数据。

### 4.4 DDL 双轨同步

新增 `db/init/04-ddl.sql`,与 `sql/ch04-ddl.sql` 逐字节一致;`test_ddl_sync.py` 扩展;`dbfixtures` 的 TABLES/DDL_PATHS 加入两新表(先子后父)。ORM 模型与 DDL 一一对应。

- 全新 MySQL volume 由 `db/init/04-ddl.sql` 首启建表。
- 已有数据卷必须显式执行 `docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch04-ddl.sql`;README 同步记录,不得通过删卷升级。
- 生产 runtime 启动时只读校验 `low_confidence_questions` / `faith_cases` 及关键列存在;缺失时启动失败并给出上述升级提示。应用不调用 `create_all`,DDL 仍是唯一 schema 事实源。

### 4.5 评估集契约(用户文件 `evals/retrieval_compare.txt`)

带引号 CSV,表头:`id,桶(bucket),问题(query),期望章节(expect_section),标准要点(expect_points),应拒答(should_refuse)`。300 条:A_policy/B_model/C_colloquial/E_multi 各 60(should_refuse=否)+ D_absent 60(是)。

- 桶即难度梯度,不设独立 difficulty 列。
- **GT 语法**(写死):`+` 表示 AND,`|` 与 `/` 是等价的 OR。正式文法为 `expression := or_group ("+" or_group)*`;`or_group := alias (("|" | "/") alias)*`;`alias := trim 后不为空且不含 "+|/" 的章节名`。因此 OR 先在组内归并,再要求所有组都命中;不支持括号、转义或其他运算符。空白剔除,空组、连续运算符和未知语法直接报错。比如 `MH-W40 + 保修说明 | 质量问题与保修 + 维修寄修` = 必须命中 MH-W40、保修说明/质量问题与保修二者之一、维修寄修三组。
- **GT 匹配**:召回块 `section_path` 包含组内任一章节名即覆盖该组(如「MH-LP100」命中「智能猫砂盆 Pro(型号 MH-LP100)」所在路径)。一个块可以覆盖多个组。
- `expect_points`(`|` 分隔要点)供生成段裁判使用。
- loader 覆盖断言:五桶齐、id 唯一、每桶 60 条、奇偶各 30 条、每个 AND 组至少有一个 OR 别名命中语料 section_path;D 桶 expect_section/expect_points 必空且 should_refuse=是,其余桶 should_refuse=否。
- 固定分片:每桶奇数编号进入 calibration,偶数编号进入 test。不得根据跑分重新洗牌;报告记录分片规则与各桶数量。

## 5. 检索管道规格(retriever.py 重构)

检索模块外部接口升级为:

```python
KnowledgeRetriever.search(
    query,
    *,
    scope=None,
    strategy=None,
    min_score=None,
    query_plan=None,
) -> RetrievalResult
```

- `QueryPlan={standard_query,synonyms,rewrite_model,degraded,note}`。在线调用不传,模块内部生成;评估先为每题生成一次,四策略复用同一对象。
- `RetrievalResult={hits,requested_strategy,effective_strategy,confidence_score,confidence_threshold,low_confidence,note,query_plan,leg_counts}`。调用方不再从一个含义随策略变化的裸 `score` 猜测拒答状态。
- `KnowledgeHit` 补 `section_path`;其 `score` 明确定义为 effective_strategy 的末级排序分(dense cosine / BM25 / RRF / rerank),只在同一策略内比较。
- `min_score` 只作为测试/评估对当前 effective_strategy 阈值的显式覆盖;不传时按策略读取对应配置。

四级管道:
1. **Query 理解**(query_understanding.py):主模型一次调用产出 JSON `{standard_query, synonyms[]}`;超时/解析失败/未配置 → 原问法直查(降级不阻断)。只作用于检索侧;入库侧不拆存。主模型由 `app/main.py` 显式注入 retriever,测试注入假模型。
2. **双路召回**(各 Top-50;`scope` 非空时两路施加同一个参数化 filter):
   - dense:standard_query embed → `vector` 列;
   - bm25:「standard_query + synonyms 拼接」文本 → `sparse` 列(`data=[文本]`,脆点 C2 实测核销)。
3. **RRF 融合**:`hybrid_search([denseReq, bm25Req], RRFRanker(k=60), limit=50)`。
4. **重排**(reranker.py):候选 50 条文档文本(= text 列同款三格拼接)POST 硅基流动 `/v1/rerank`,`top_n=10`,返回 relevance_score 序。连接/超时/429/HTTP 408/409/5xx 在模块内部按 0.5s/1s 退避最多重试 2 次(共最多 3 次请求);耗尽返回结构化降级结果,由 retriever 使用原 hybrid 排名继续。

策略枚举:`dense` / `bm25` / `hybrid` / `hybrid_rerank`(在线默认 hybrid_rerank,配置 `KNOWLEDGE_STRATEGY`)。

**分策略低置信**:0 命中始终为 low confidence;非空时取 Top-1 末级排序分与 effective_strategy 对应阈值比较。dense / bm25 / hybrid / hybrid_rerank 分别使用 `KNOWLEDGE_MIN_SCORE` / `BM25_MIN_SCORE` / `HYBRID_MIN_SCORE` / `RERANK_MIN_SCORE`。hybrid_rerank 降级后 effective_strategy=hybrid,因此自动改用 HYBRID_MIN_SCORE,不把不同量纲分数混用。

**证据组装**:所有策略先将候选截到最多 10 条,再对实际 N 条使用「奇数 rank 升序 + 偶数 rank 降序」排列;N=10 时恰为 `[1,3,5,7,9,10,8,6,4,2]`,N=6 时为 `[1,3,5,6,4,2]`。`RERANK_TOP_N=10` 只控制 reranker 返回量;证据层自身仍强制 `N <= 10`,包括 dense/bm25/hybrid。规范化对象固定为 `Evidence={ref_no,chunk_id,section_path,question,answer,category}`。组装器逐条序列化完整对象,下一个对象无法完整放进工具结果、第二次模型输入和 envelope 的共同预算时就停止;不得删除字段、截断字段值或截断 JSON。`ref_no=[1..N]` 只按预算裁剪后的最终展示序分配。

**probe**(/kb 检索自测)同步升级:支持选 strategy/scope,返回 requested/effective strategy、各腿命中数、Top-1 分数、对应阈值、是否 low confidence 与降级 note。

## 6. 重建作业与 ingest 改动

`kb_admin.rebuild_index`(作业互斥锁复用,占用即 409 job_busy):
1. **只读预检**:embedding 可用、四份文档 frontmatter/切块/字段长度合法、每个评估正例 GT 组均能命中语料章节、C1/C2 analyzer/BM25 smoke 通过、新 schema 可创建;任一失败不得 drop/清表。
2. 将进程内唯一 `KnowledgeState` 从 `ready` 或 `rebuild_required` 转为 `rebuilding`;此后在线 `query_faq` 返回明确的「知识库正在重建」工具不可用结果,不读取旧库或半成品,也不把它当低置信问题入池。
3. drop 旧集合(存在的话)。drop/create 必须正确复位 store 的 `_loaded` 状态。
4. 清空 `knowledge_chunks` 全表 + `qa_extraction_staging` + `qa_mining_progress`(全量重置,用户确认挖掘块一并清);先解除 knowledge_chunks 自引用或使用同事务内受控的 FK 处理。
5. 按新 schema 建同名集合。
6. 重新 ingest 当前 `knowledge_docs/` 四份文档(补录与修订见 §11)。
7. 向量化全量 pending(upsert 行带 `text`/`scope`)。
8. 终验:无 pending、两库主键集合一致、BM25 型号 smoke 命中(沿用 `_verify` 并扩展)。全部通过才把状态转为 `ready`。

`KnowledgeState` 是 `ready | rebuilding | rebuild_required` 三态枚举,由 admin、retriever 与 `/kb/api/state` 共享同一状态持有者。预检失败不改变进入作业前的状态;进入破坏性步骤后的任何失败都返回 `rebuild_failed` 并转为 `rebuild_required`;同一接口可幂等重跑,再次执行仍先预检再全量重置。进程启动时由集合契约 + `_verify` 初始化为 `ready` 或 `rebuild_required`,不得仅依赖上个进程已丢失的内存标志。读接口 `/kb/api/state` 原样暴露该枚举。

ingest 改动仅限 upsert 数据行多带 `text/scope`;`vector_text` 三格拼接同时是 dense embed 输入、BM25 text 列内容、rerank 候选文档,一处定义三处复用。

## 7. 引用、拒答与低置信度池

### 7.1 引用

- `build_tools` 不再只返回裸 list,而是返回 `TurnToolset={tools,retrieval_trace}`;`ChatService` 与工具闭包持有同一个每轮 `RetrievalTrace={status,result,evidence,error_code}`。`status` 只允许 `not_called | ok | low_confidence | tool_error`;初始为 `not_called`,`query_faq` 完成后只写一次;`result` 仅在 ok/low_confidence 时保存同一个 `RetrievalResult`,`error_code` 仅在 tool_error 时设置。这是引用、低置信判定和持久化的显式 interface,不得靠给函数/list 临时挂属性。
- 后端 `_tool_calls_legal` 强制每轮最多一次 `query_faq`;模型应把完整问题一次传入。其他工具数量仍受 `MAX_TOOL_CALLS_PER_TURN` 约束。
- `query_faq` 入参:`keyword` + 可选 `scope` 枚举。出参(给模型):顶层带 effective_strategy/low_confidence/note,`evidence` 是 §5 定义的最终 `Evidence[]`。原三键契约扩展为结构化对象,契约测试同步更新。
- System Prompt 引用协议:用到证据的句末带 [n];没有证据支撑的话不许说。
- **唯一证据列表**:`RetrievalTrace.evidence: list[Evidence]` 只保存预算裁剪后实际发给模型的最终列表,按组装展示序排列。`query_faq.evidence`、SSE `citations` payload 和 envelope `metadata.citations` 必须逐项逐字段等于该列表;三者不得分别投影、重建或二次裁剪。
- **SSE 新帧 `citations`**:回答流完、`[DONE]` 前原样推送上述最终列表。无成功 query_faq 或后端固定拒答时推空列表/不推帧的选择由前端测试固定为一种;本章采用**不推空帧**。
- **envelope v2**:工具行保存 `{"v":2,"ok":true,"content":"...","metadata":{"citations":[...],"retrieval":{...}}}`;读取端兼容已有 v1。v2 总长度仍受持久化上限约束,content 与 metadata 必须在证据组装时共同预算;envelope 层不得再裁剪列表或生成不完整 JSON。

### 7.2 拒答双闸门

- **检索侧硬闸门**:RetrievalResult 为 low_confidence 时,`query_faq` 仍返回诊断结果并写入 trace,但 ChatService 不发起第二次模型调用,直接输出固定拒答话术。判定来自 0 命中或 effective_strategy 的 Top-1 分数低于该策略阈值。
- **生成侧自评**:只有检索通过才调用模型;prompt 指令「证据不足以完整回答就只回复固定拒答话术」。该路径允许模型决定,但最终文本去除首尾空白后必须与常量完全相等才算 self_check 拒答,不用宽松前缀匹配。
- 固定话术常量化:`REFUSAL_ANSWER = "抱歉,这个问题超出了我目前掌握的资料范围,已为您记录,稍后可转人工客服进一步核实。"`;prompt、后端硬闸门与识别共用同一常量。
- **入池与消息同事务**:
  - 检索硬闸门触发 → `source=retrieval_low_conf`,reason JSON 记录 requested/effective strategy、Top-1 分数、阈值、降级 note;
  - 检索通过但模型返回固定话术 → `source=self_check`,reason JSON 记录当时 evidence ref_no/chunk_id;
  - `SessionStore.commit_turn(messages, low_confidence=...)` 在同一 DB 事务写消息与问题池;conversation_id 使用本轮上下文;user_feedback 不写。
- 重建维护、未配置 embedding、鉴权失败、超时等工具不可用属于系统故障,返回工具错误,不得伪装成 low confidence 或写入问题池。

### 7.3 负面知识禁令(System Prompt 列死)

不承诺:具体订单退款到账、发货、送达的确定日期或结果保证;赔偿金额;知识库未载明的任何承诺类表述。知识库明确记载的通用政策时效可以引用,但必须表述为「政策规定/通常/应在……内」,不得改写成对当前订单的个案保证。知识库没有的,走拒答,不硬编。

## 8. 评估体系(evals/run_retrieval_compare.py)

复用 ch03 评估骨架:独立评估库(TEST_DATABASE_URL 派生)+ 临时 Milvus Lite + `evals/results/{UTC 时间戳}_compare.json`。

- **QueryPlan 冻结**:每个 case 先调用一次 Query 理解并保存 QueryPlan;四策略复用,不得各自改写。报告记录 rewrite_model、standard_query、synonyms、原始响应摘要、degraded/note,使策略差异可复盘。
- **检索段**:四策略 × 全量 case 都跑并留诊断;阈值校准与正式验收严格分片。
  - `SectionRecall@K`:case 内已覆盖 AND 章节组数 / AND 组总数,再做宏平均;
  - `CompleteHit@K`:所有 AND 组均覆盖记 1,否则 0;
  - `MRR@10`:首个匹配任一 GT 组的相关块位次倒数;
  - D_absent 无相关章节,上述检索相关性指标记 N/A,只进入低置信/拒答指标,不得用 0 拉低总 Recall。
- **生成段**(hybrid_rerank 策略跑完整生成,使用预算裁剪后的唯一 citations 列表):
  - **Faithfulness**:主模型当裁判,严格结构化输出 `{verdict: "faithful"|"fabricated", unsupported_claims: [{claim,reason}], cited_refs: [int]}`。解析/字段校验失败最多重试一次;仍失败记 `judge_error`,不计入忠实或编造分母,脚本最终非零退出;
  - **D 桶拒答正确率**(应拒且拒)、**A/B/C/E 误拒率**(不应拒却拒);
  - 编造个案 upsert 主业务库 `faith_cases`(§4.3 语义,judge_model 落列);judge_error 不入台账。
- **分片与阈值**:每桶奇数 30 条 calibration、偶数 30 条 test。dense/bm25/hybrid/hybrid_rerank 各自独立枚举 calibration Top-1 分数相邻区间的阈值,不得跨策略复用分数或阈值。报告为每个策略输出「D 桶误通过率,A/B/C/E 误拒率,pass-adjusted SectionRecall@10」Pareto 候选(被阈值拒答的 case 的 recall 计 0);首次 calibration 后由用户只依据该候选表确认每个策略的 D 桶误通过率约束,选择器在约束内最大化 pass-adjusted SectionRecall@10,并列时依次选择 A/B/C/E 误拒率更低、阈值更小者。约束、候选和选择理由写入报告,选择器写成纯函数并测试。四个阈值冻结后才能执行 test;正式拒答/误拒指标只报告 test,全量 300 条另列 diagnosis,不得称为验收结果。
- **报告**:JSON 全量 + Markdown 摘要落 `evals/results/` 并打印。主表为四策略 × 五桶的 test 指标;附 calibration 阈值、全量诊断、每 case QueryPlan/命中组/分数/拒答/judge 结果与运行配置。
- **硬门槛**:①报告必出数字且 judge_error=0;②B_model test 桶纯 BM25 `SectionRecall@10 ≥ 0.5`;③四个阈值只从 calibration 冻结回写 config/.env.example,不得查看 test 后调参;④ D_absent test 拒答正确率与 A/B/C/E test 误拒率必须出数,具体上线门槛在首次基线报告后由用户确认,不能由脚本偷偷自定。
- 指标纯函数进 `tests/test_eval_metrics.py` 扩展;csv loader 规则(引号/竖线/加号)单测钉死。
- 本评估跑通依赖新语料重建(§6)与可用 rerank key;`RERANK_API_KEY` 为空时可使用 `EMBEDDING_API_KEY`,两者都空则脚本显式报错不静默。主业务库 ch04 表缺失也须在任何远程调用前显式报错。

## 9. 前端(vibe coding 例外,不套 brainstorm/TDD/review)

- **chat.html**:`citations` 帧进 `parseEvents` 分发新分支;正文 [n] 渲染为可点角标,点击弹层显示该 chunk 原文 + section_path(数据全在帧内,不加新 API);沿用 design tokens。
- **反馈**:每条 assistant 气泡左下 👍/👎;点击点亮所选 + 显示「已反馈」+ 一次性锁定;localStorage 持久(键含 session + 消息序号),纯前端采集。
- **kb.html**:「重建索引」按钮走 `runAction` 范式;检索自测卡加策略选择。
- 页面测试沿用字符串断言范式(test_chat_page / test_kb_page 扩展)。

## 10. 配置新增(config.py + .env.example)

| 键 | 默认 | 说明 |
|---|---|---|
| `RERANK_BASE_URL` | `https://api.siliconflow.cn/v1` | 与 embedding 同商 |
| `RERANK_API_KEY` | 空 | 可选覆盖;空时回退 EMBEDDING_API_KEY;回退 key 也空则知识检索不可用 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 定死选型 |
| `RERANK_TIMEOUT_SECONDS` | 5 | 单次 rerank HTTP 请求超时,仍受 20 秒总预算限制 |
| `RERANK_MAX_RETRIES` | 2 | reranker 内部追加重试次数;2 表示最多共请求 3 次 |
| `RETRIEVAL_CANDIDATE_K` | 50 | 双路各召回数 |
| `RERANK_TOP_N` | 10 | 精排出条数 |
| `RERANK_MIN_SCORE` | 0.0(占位) | 拒答阈值,评估校准后冻结回写 |
| `BM25_MIN_SCORE` | 0.0(占位) | BM25 单路拒答阈值,评估校准后冻结回写 |
| `HYBRID_MIN_SCORE` | 0.0(占位) | RRF hybrid 拒答阈值,含 rerank 降级路径 |
| `KNOWLEDGE_STRATEGY` | `hybrid_rerank` | 在线默认策略 |
| `QUERY_REWRITE_ENABLED` | true | Query 理解开关(测试可关) |
| `KNOWLEDGE_TOOL_TIMEOUT_SECONDS` | 20 | query_faq 完整链路总预算;独立于普通工具 5 秒限制 |

`KNOWLEDGE_MIN_SCORE` 继续表示 dense 余弦阈值;hybrid 管道里的 dense 腿不做该阈值截断(Top-50 候选交给 RRF/rerank 裁决)。新语料下四个阈值全部按 §8 calibration 分片复校并记录 dev-notes。

ToolExecutor 对 `query_faq` 使用独立 policy:`timeout=KNOWLEDGE_TOOL_TIMEOUT_SECONDS,max_retries=0`;普通只读工具继续使用 `TOOL_TIMEOUT_SECONDS/TOOL_MAX_RETRIES`。executor 开始执行 `query_faq` 时创建一次基于 monotonic clock 的 deadline;Query 理解、embedding、dense/BM25/hybrid 检索、rerank 的每次请求与退避、结果加载和证据组装都共享并扣减这一个 20 秒总预算。各阶段必须接收剩余时长且不得自行重置计时:Query 理解阶段失败或阶段超时即回退原问法;rerank 在剩余预算内按自身策略重试/降级;任一时刻总预算耗尽即返回工具不可用。executor 不得启动无法取消的整链重复线程。

## 11. 实施前置:评估集与知识文档对齐

四份新 md 缺 frontmatter。在每份顶部补:

```markdown
---
type: faq        # product-faq.md
---
```

type 映射:product-faq→faq;returns-policy→policy;after-sales-manual / product-specs→manual。

评审实测:补 frontmatter 后当前四份文档切出 43 块;至少 17 条 should_refuse=否 的 case 没有任何 GT 章节别名能命中 section_path。**本次确认以 300 条评估集为业务需求基线**,因此原「正文一字不动」约束作废,实施前必须补齐语料:

| 文档 | 至少新增/补强章节 | 必须覆盖的评估事实 |
|---|---|---|
| `product-faq.md` | `会员权益 / 会员等级`、`积分怎么攒`、`积分怎么用`、`可开票类型`、`开票时效`、`运费与包邮` | 银卡 1000 元起/95 折,金卡 5000 元起/9 折/生日猫罐头礼盒;退货扣回积分、双倍积分不计升级;100 积分抵 1 元且单笔最多 30 元;800/2000 积分兑换及对应运费;电子专票资料、企业纸质专票、30 天申请期限/3 个工作日开出;偏远地区不参与普通包邮 |
| `product-specs.md` | `猫窝是否可以机洗` | 布艺猫窝拆垫芯、洗衣袋、轻柔冷水洗;带加热模块猫窝不可机洗、仅局部擦洗 |
| 其余文档 | 按 loader 报告补齐别名或事实 | 每个 AND 组至少一个 section_path 可命中,每个 expect_point 都有权威原文支撑 |

不得为了让 loader 通过而只改标题、不补事实;生成评估需要原文真实承载 expect_points。补录后运行只读 corpus validator,输出 case → GT 组 → 命中 chunk 映射,17 条原失败 case 必须全部转为通过。

评估集 D3「能不能开纸质发票邮寄给我」与 A30/E46 的企业纸质专票规则冲突。D3 改为知识库确实未覆盖的「发票能不能用外币金额开具」,其 bucket/空 GT/should_refuse=是保持不变。除这一处冲突修正外,300 条 case 数量和奇偶分片不变。

## 12. 错误处理

- rerank API:连接/超时/429/HTTP 408/409/5xx → reranker 内部最多重试 2 次;401/403/其他 4xx 零重试。任何失败耗尽后返回降级结果,由 retriever 以 hybrid 排名和 HYBRID_MIN_SCORE 继续;rerank 异常不抛给 ToolExecutor 触发整链重试。
- Query 理解:任何异常 → 原问法直查,note 记录。
- 缺 RERANK_API_KEY 时先尝试 EMBEDDING_API_KEY;两者都空意味着 embedding 也未配置,在线 `query_faq` 返回工具不可用,评估脚本在远程调用前显式报错。key 存在但 rerank 请求最终失败时才按上一条降级 hybrid。
- query_faq 使用独立 20 秒总时限且 executor 不重试;预算从 executor 开始执行该工具起连续计时,覆盖 §10 列出的完整链路,总预算耗尽返回工具不可用,不走拒答、不入低置信池。
- Milvus:未建库/旧 schema/重建失败 → 工具不可用并提示「需重建索引」;重建进行中 → 「知识库正在重建」;读写走 `_ensure_loaded`;Lite 文件锁独占约束不变(停服跑 CLI/评估)。
- SSE:citations 帧推送失败不影响回答主体(try/except 记日志)。
- Faithfulness judge:结构化结果解析失败重试一次;仍失败记 judge_error,报告落盘后脚本非零退出,不得默认判忠实。

## 13. 测试策略与验收

沿用三档:
- **纯单测**:N=1..10 的证据组装(含 N=10 的 `[1,3,5,7,9,10,8,6,4,2]`);第 11 条永不进入证据;按完整对象预算裁剪且 tool/citations/envelope 三份列表逐项逐字段相等;每轮第二个 query_faq 被拒;检索硬闸门不调用第二次模型;生成自评必须精确匹配固定拒答;入池 source 判定矩阵;csv loader(引号内逗号、正式 AND/OR 文法、非法语法、奇偶分片、D 桶断言);SectionRecall/CompleteHit/MRR/四策略独立阈值校准/Faithfulness judge 解析;每 case QueryPlan 四策略复用;query 理解 JSON 解析与降级;rerank 内部重试耗尽返回 hybrid 降级(mock httpx);query_faq executor policy 为独立 20 秒总 deadline/零整链重试,并用 fake clock 验证所有阶段与退避共享同一预算。
- **Milvus Lite 真跑**(tmp 文件 + FakeEmbeddings):五列 schema 建集合含 BM25 function;upsert 自动产出 sparse;BM25 路命中型号文本;hybrid_search RRF 返回;scope 参数化过滤;四策略 Top-1 分别按自身阈值判定;rerank 降级改用 hybrid 阈值;重建预检失败不破坏旧库;重建中读返回维护状态;重建作业全流程;旧两列 schema 报「需重建」。
- **DB 集成**:两新表 CRUD;本轮 messages + LowConfidenceQuestion 同事务提交与故障回滚;faith_cases 写主业务库且 upsert 复发(seen_count/状态退回);启动 schema 校验;dbfixtures 同步;DDL 双轨逐字节一致;envelope v1/v2 均可回放。
- **页面测试**:chat.html 角标/反馈锚点字符串断言;kb.html 重建按钮。
- **数据类任务**:补录语料后先跑 corpus validator,确认所有正例 GT 组与 expect_points 有原文支撑,D3 已替换且仍为 D_absent;再跑 calibration 冻结四阈值,最后只以 test 分片做正式四策略/生成验收。数据和 Prompt 任务以真实评估运行代替伪单测。

验收(对应 §1):
1. 四策略 test 报告出 SectionRecall@5/10、CompleteHit@5/10、MRR@10、拒答/误拒与 Faithfulness 数字,judge_error=0,并附 calibration/全量诊断。
2. B_model test 桶 BM25 SectionRecall@10 ≥ 0.5,具体型号 smoke 可见 BM25 腿命中。
3. 在线问答引用角标可点;模型工具结果、citations SSE、envelope v2 三份证据列表逐项逐字段一致且不超过 10 条,弹层原文与 MySQL 一致。
4. D 类问题由后端低置信硬闸门明确拒答,不调用第二次模型;同一事务内 `low_confidence_questions` 新增 `retrieval_low_conf` 行。检索通过后模型自评拒答则写 `self_check`。
5. rerank 连续失败时回答链路使用 hybrid + HYBRID_MIN_SCORE,报告/trace 可见 effective_strategy=hybrid;知识库重建期间不返回半成品结果。

## 14. 依赖与 Context7 核对清单(动手前核销)

不新增第三方依赖(rerank 用 httpx,已在依赖树)。
Context7 必查:
1. pymilvus `Function`/`FunctionType.BM25`/`schema.add_function`(已核,§4.1);
2. `analyzer_params` Jieba 形态 + `run_analyzer` 实测(C1);
3. `MilvusClient.hybrid_search` + `AnnSearchRequest` + `RRFRanker` 签名(已核基本形;BM25 腿 `data` 传文本串 C2 待实测);
4. sparse 索引 `SPARSE_INVERTED_INDEX` + metric BM25 的 index_params 写法;
5. 硅基流动 `/v1/rerank` 请求/响应格式(web 文档,Context7 无则 FetchURL 官方文档页)。

脆点(先行 smoke):C1 分词器参数形态;C2 BM25 查询 data 形态;C3 Lite 段级 BM25 IDF(segment-local)对小语料分数与候选集的影响(影响 BM25/HYBRID/RERANK 三个阈值校准,不阻塞);C4 pymilvus import 期环境污染隔离沿用 `_load_milvus_client`,新模块不得直接 `from pymilvus import …`。
