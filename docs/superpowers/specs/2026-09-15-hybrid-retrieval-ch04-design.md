# 电商智能客服 Ch04: RAG 进阶 · 混合检索 + 重排 + 生成质量控制 + 评估体系 - 设计文档

日期: 2026-09-15
前置: ch03(向量语义检索)已上线;Milvus 集合 `knowledge` 现为两列(id + vector),原文权威源在 MySQL `knowledge_chunks`。
本文档是与用户对齐后的实施依据;技术选型为用户定死:Milvus 原生 BM25 + hybrid_search RRF 融合 + bge-reranker-v2-m3 重排。

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
| 元数据过滤入口 | `query_faq` 工具加**可选** `category` 参数,模型自行从问法识别品类 |
| 拒答入池范围 | 检索低置信 + 生成自评不足**两种都入池**,靠 `source` 列区分;`user_feedback` 枚举留后续章节 |
| Query 改写模型 | 复用主对话模型(OPENAI_BASE_URL/MODEL_NAME),失败降级原问法直查 |
| 评估集 | 用户提供 `evals/retrieval_compare.txt`(300 条 CSV,已就位);loader 按此格式写 |
| 知识文档 | 用户提供的新四份 md 已就位(喵喵优选宠物用品域),旧三份已删;缺 frontmatter 由我补齐 |
| faith_cases 界面 | 本章不做,只落表 + 报告 |

## 3. 项目结构(新增/改动)

```
app/knowledge/
  query_understanding.py   新增:改写 + 同义词扩展(主模型,纯函数式封装,可注入假模型)
  reranker.py              新增:硅基流动 rerank 客户端(httpx,超时/重试分级)
  retriever.py             重构:四级管道 + 策略参数 + category 过滤
  milvus_store.py          扩展:五列 schema、双路 search、hybrid_search、重建支持
  ingest.py                小改:upsert 行带 text/category
app/services/kb_admin.py   扩展:rebuild_index 作业(全量重置)
app/tools/business.py      小改:query_faq 加 category 可选参 + 出参带 ref_no(出参口径见 §7.1)+ 收集 citations
app/services/chat_service.py 扩展:citations SSE 帧、拒答识别与入池
app/prompts/service.py     扩展:引用协议 + 拒答话术 + 负面知识禁令
app/prompts/               新增:query_understanding / faithfulness judge 提示词
app/models.py              新增:LowConfidenceQuestion / FaithCase ORM
app/routers/chat.py        小改:citations 帧序列化分支
app/routers/kb.py          小改:rebuild 接口
db/init/04-ddl.sql         新增(与 sql/ch04-ddl.sql 逐字节一致)
sql/ch04-ddl.sql           注释修订(两种拒答都入池;bucket 含 D_absent),结构不动
evals/run_retrieval_compare.py  新增:四策略对比 + 生成段 Faithfulness
evals/retrieval_compare.txt     评估集(用户提供,300 条,已就位)
app/static/chat.html       引用角标可点 + 👍/👎 反馈(vibe,不套流程)
app/static/kb.html         重建索引按钮(vibe)
```

## 4. 数据契约

### 4.1 Milvus 集合 `knowledge`(新 schema,同名重建)

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 主键,auto_id=False | 与 MySQL chunk id 1:1(第一不变量不变) |
| `vector` | FLOAT_VECTOR(1024),COSINE | dense 语义向量 |
| `text` | VARCHAR(max_length 4096),enable_analyzer=True | BM25 语料 = `vector_text` 三格拼接(与向量同源,保证两路语料一致) |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出列;SPARSE_INVERTED_INDEX,metric BM25 |
| `category` | VARCHAR(128) | 元数据过滤字段(`category == "..."` expr) |

- BM25 函数:`Function(name="bm25_fn", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])`,`schema.add_function(...)`。
- 分词器:Lite 用 Jieba 中文分词;参数形态(`{"tokenizer": "jieba"}` vs `{"type": "chinese"}`)实现期以 Context7 + `run_analyzer` 实测核销(脆点 C1)。
- `ensure_collection` 契约校验升级:存在集合必须是五列新 schema(字段齐、主键 INT64、dim=1024),旧两列 schema 报「需重建索引」,**不自动重建**(沿用 ch03 原则)。
- `MilvusKnowledgeStore` 方法扩展:`search_dense`(现 search 改名/参数化)、`search_bm25(data=[文本])`、`hybrid_search(dense_vec, bm25_text, category_expr)`;全部走既有 `_ensure_loaded` 入口。

### 4.2 MySQL `low_confidence_questions`(用户 DDL,结构不动)

原话、来源会话、入池入口(source ENUM)、判不能原因、时间。**注释修订**:本章两种拒答都入池(`retrieval_low_conf` / `self_check`),`user_feedback` 留后续章节。

### 4.3 MySQL `faith_cases`(用户 DDL,结构不动)

编造个案台账:一题一行(uk_eval_id),重判更新快照 + seen_count+1;已解决复发退回「未解决」并清空 resolution。`citations` JSON 存当轮 Top-K 证据全集快照(角标 [n] = 列表序号)。bucket 注释补 D_absent。

### 4.4 DDL 双轨同步

新增 `db/init/04-ddl.sql`,与 `sql/ch04-ddl.sql` 逐字节一致;`test_ddl_sync.py` 扩展;`dbfixtures` 的 TABLES/DDL_PATHS 加入两新表(先子后父)。ORM 模型与 DDL 一一对应。

### 4.5 评估集契约(用户文件 `evals/retrieval_compare.txt`)

带引号 CSV,表头:`id,桶(bucket),问题(query),期望章节(expect_section),标准要点(expect_points),应拒答(should_refuse)`。300 条:A_policy/B_model/C_colloquial/E_multi 各 60(should_refuse=否)+ D_absent 60(是)。

- 桶即难度梯度,不设独立 difficulty 列。
- **GT 匹配规则**(写死):`expect_section` 中 `|` = 任一章节命中即相关;`+` = 多个章节取并集;判定 = 召回块 `section_path` **包含**期望章节名(如「MH-LP100」命中「智能猫砂盆 Pro(型号 MH-LP100)」所在路径)。
- `expect_points`(`|` 分隔要点)供生成段裁判使用。
- loader 覆盖断言:五桶齐、id 唯一、每个 expect_section 至少命中语料中一个块的 section_path;D 桶 expect_section 必空。

## 5. 检索管道规格(retriever.py 重构)

`KnowledgeRetriever.search(query, *, category=None, strategy=None, min_score=None) -> (list[KnowledgeHit], note | None)`,契约不变;`KnowledgeHit` 补 `section_path`。

四级管道:
1. **Query 理解**(query_understanding.py):主模型一次调用产出 JSON `{standard_query, synonyms[]}`;超时/解析失败/未配置 → 原问法直查(降级不阻断)。只作用于检索侧;入库侧不拆存。
2. **双路召回**(各 Top-50;`category` 非空时两路 expr 同施):
   - dense:standard_query embed → `vector` 列;
   - bm25:「standard_query + synonyms 拼接」文本 → `sparse` 列(`data=[文本]`,脆点 C2 实测核销)。
3. **RRF 融合**:`hybrid_search([denseReq, bm25Req], RRFRanker(k=60), limit=50)`。
4. **重排**(reranker.py):候选 50 条文档文本(= text 列同款三格拼接)POST 硅基流动 `/v1/rerank`,`top_n=10`,返回 relevance_score 序。

策略枚举:`dense` / `bm25` / `hybrid` / `hybrid_rerank`(在线默认 hybrid_rerank,配置 `KNOWLEDGE_STRATEGY`)。

**证据组装**:Top-10 进 prompt 的排列为 rank 序 `[1,3,5,7,9,10,8,6,4,2]`——最相关钉首尾,不按分数堆中间;引用编号 [1..10] 按此展示序分配。

**probe**(/kb 检索自测)同步升级:支持选策略,返回各腿命中数。

## 6. 重建作业与 ingest 改动

`kb_admin.rebuild_index`(作业互斥锁复用,占用即 409 job_busy):
1. drop 旧集合(存在的话);
2. 清空 `knowledge_chunks` 全表 + `qa_extraction_staging` + `qa_mining_progress`(全量重置,用户确认挖掘块一并清);
3. 按新 schema 建同名集合;
4. 重新 ingest 当前 `knowledge_docs/` 四份文档(先补 frontmatter,见 §11);
5. 向量化全量 pending(upsert 行带 `text`/`category`);
6. 终验:无 pending、两库主键集合一致(沿用 `_verify`)。

ingest 改动仅限 upsert 数据行多带两列;`vector_text` 三格拼接同时是 dense embed 输入、BM25 text 列内容、rerank 候选文档,一处定义三处复用。

## 7. 引用、拒答与低置信度池

### 7.1 引用

- `query_faq` 出参(给模型):每条证据 `{ref_no, question, answer, category}`(section_path 不进模型上下文,省 token;前端展示走 citations 帧)。原三键契约是超集扩展,契约测试同步更新。
- System Prompt 引用协议:用到证据的句末带 [n];没有证据支撑的话不许说。
- **SSE 新帧 `citations`**:回答流完、`[DONE]` 前推 `[{n, chunk_id, section_path, question, answer}]`(本轮 query_faq 全部调用的证据全集,按组装展示序)。收集方式:build_tools 闭包内挂 per-turn collector,tool 执行时写入;ChatService 收尾时推帧。快照顺带写进工具行 envelope JSON 存证(不加新列)。

### 7.2 拒答双闸门

- **检索侧**(确定性):0 命中或 rerank Top-1 relevance_score < `RERANK_MIN_SCORE` → query_faq 结果 JSON 带 `low_confidence: true` + 分数明细,prompt 要求此时只回固定拒答话术。
- **生成侧**(模型自评):prompt 指令「证据不足以回答就**只**回复固定拒答话术」。
- 固定话术常量化(如 `REFUSAL_ANSWER = "抱歉,这个问题超出了我目前掌握的资料范围,已为您记录,稍后可转人工客服进一步核实。"`),prompt 与后端识别共用同一常量(前缀匹配)。
- **入池**(ChatService 收尾,识别到拒答话术后):
  - 工具报了 low_confidence → `source=retrieval_low_conf`,reason=分数明细;
  - 否则 → `source=self_check`,reason=当时证据 ref_no 列表;
  - conversation_id 上下文现成;user_feedback 不写。

### 7.3 负面知识禁令(System Prompt 列死)

不承诺:退款到账的具体时间/日期;发货、送达的具体日期与时效保证;赔偿金额;知识库未载明的任何承诺类表述。知识库没有的,走拒答,不硬编。

## 8. 评估体系(evals/run_retrieval_compare.py)

复用 ch03 评估骨架:独立评估库(TEST_DATABASE_URL 派生)+ 临时 Milvus Lite + `evals/results/{UTC 时间戳}_compare.json`。

- **检索段**:四策略 × 全量 case;指标 **Recall@5、Recall@10、MRR@10**(case 级:任一相关块入 Top-K 即命中;MRR 取首个相关块位次倒数);按桶分桶出表。
- **生成段**(hybrid_rerank 策略跑完整生成):
  - **Faithfulness**:主模型当裁判,对照当轮 citations 快照判「忠实 / 编造 + 理由(编在哪句)」;
  - **D 桶拒答正确率**(应拒且拒)、**A/B/C/E 误拒率**(不应拒却拒);
  - 编造个案 upsert `faith_cases`(§4.3 语义,judge_model 落列)。
- **报告**:JSON 全量 + Markdown 摘要(四策略 × 五桶对比表)落 `evals/results/` 并打印。
- **硬门槛**:①报告必出数字;②B_model 桶纯 BM25 路 Recall@10 ≥ 0.5(校准后冻结,对应验收 2);③RERANK_MIN_SCORE 用本评估集校准(以 D 桶低分、A/B/C/E 通过为原则)后冻结回写 config/.env.example。
- 指标纯函数进 `tests/test_eval_metrics.py` 扩展;csv loader 规则(引号/竖线/加号)单测钉死。
- 本评估跑通依赖新语料重建(§6)与 RERANK_API_KEY;缺 key 时脚本显式报错不静默。

## 9. 前端(vibe coding 例外,不套 brainstorm/TDD/review)

- **chat.html**:`citations` 帧进 `parseEvents` 分发新分支;正文 [n] 渲染为可点角标,点击弹层显示该 chunk 原文 + section_path(数据全在帧内,不加新 API);沿用 design tokens。
- **反馈**:每条 assistant 气泡左下 👍/👎;点击点亮所选 + 显示「已反馈」+ 一次性锁定;localStorage 持久(键含 session + 消息序号),纯前端采集。
- **kb.html**:「重建索引」按钮走 `runAction` 范式;检索自测卡加策略选择。
- 页面测试沿用字符串断言范式(test_chat_page / test_kb_page 扩展)。

## 10. 配置新增(config.py + .env.example)

| 键 | 默认 | 说明 |
|---|---|---|
| `RERANK_BASE_URL` | `https://api.siliconflow.cn/v1` | 与 embedding 同商 |
| `RERANK_API_KEY` | 空 | 空则 hybrid_rerank 降级 hybrid + note |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 定死选型 |
| `RETRIEVAL_CANDIDATE_K` | 50 | 双路各召回数 |
| `RERANK_TOP_N` | 10 | 精排出条数 |
| `RERANK_MIN_SCORE` | 0.0(占位) | 拒答阈值,评估校准后冻结回写 |
| `KNOWLEDGE_STRATEGY` | `hybrid_rerank` | 在线默认策略 |
| `QUERY_REWRITE_ENABLED` | true | Query 理解开关(测试可关) |

`KNOWLEDGE_MIN_SCORE`(dense 余弦阈值):hybrid 管道里 dense 腿**不做** min_score 截断(Top-50 候选交给 RRF/rerank 裁决);该阈值仅继续作用于 strategy=dense 单路形态与 ch03 旧评估。新语料下如需复校,在评估任务里处理并记录 dev-notes。

## 11. 实施前置:文档 frontmatter 补录

四份新 md 缺 frontmatter。在每份顶部补:

```markdown
---
type: faq        # product-faq.md
---
```

type 映射:product-faq→faq;returns-policy→policy;after-sales-manual / product-specs→manual。文档正文一字不动。

## 12. 错误处理

- rerank API:连接/超时/限流/HTTP 408/409/5xx → `RetryableKnowledgeError` 入 executor RETRYABLE 重试;401/403/参数错误零重试;耗尽降级 hybrid 继续作答(证据不带精排序,note 记录)。
- Query 理解:任何异常 → 原问法直查,note 记录。
- 缺 RERANK_API_KEY:启动不拒,在线 hybrid_rerank 降级 hybrid + note;评估脚本显式报错。
- Milvus:未建库/旧 schema → note「需重建索引」;读写走 `_ensure_loaded`;Lite 文件锁独占约束不变(停服跑 CLI/评估)。
- SSE:citations 帧推送失败不影响回答主体(try/except 记日志)。

## 13. 测试策略与验收

沿用三档:
- **纯单测**:证据组装排位 [1,3,5,7,9,10,8,6,4,2];拒答前缀识别;入池 source 判定矩阵;csv loader(引号内逗号、`|`/`+` 规则、D 桶断言);Recall@K/MRR/Faithfulness 解析指标函数;query 理解 JSON 解析与降级;rerank 客户端错误分级(mock httpx)。
- **Milvus Lite 真跑**(tmp 文件 + FakeEmbeddings):五列 schema 建集合含 BM25 function;upsert 自动产出 sparse;BM25 路命中型号文本;hybrid_search RRF 返回;category expr 过滤;重建作业全流程;旧两列 schema 报「需重建」。
- **DB 集成**:两新表 CRUD + faith_cases upsert 复发语义(seen_count/状态退回);dbfixtures 同步;DDL 双轨逐字节一致。
- **页面测试**:chat.html 角标/反馈锚点字符串断言;kb.html 重建按钮。
- **数据类任务**(csv loader 实跑、阈值校准、四策略报告):拿用户评估集实跑一遍验证,代替 TDD。

验收(对应 §1):四策略报告出数字;B 桶 BM25 命中达标;在线问答引用角标可点且弹层原文与 MySQL 一致;D 类问题在线明确拒答且 `low_confidence_questions` 有新行(source 正确)。

## 14. 依赖与 Context7 核对清单(动手前核销)

不新增第三方依赖(rerank 用 httpx,已在依赖树)。
Context7 必查:
1. pymilvus `Function`/`FunctionType.BM25`/`schema.add_function`(已核,§4.1);
2. `analyzer_params` Jieba 形态 + `run_analyzer` 实测(C1);
3. `MilvusClient.hybrid_search` + `AnnSearchRequest` + `RRFRanker` 签名(已核基本形;BM25 腿 `data` 传文本串 C2 待实测);
4. sparse 索引 `SPARSE_INVERTED_INDEX` + metric BM25 的 index_params 写法;
5. 硅基流动 `/v1/rerank` 请求/响应格式(web 文档,Context7 无则 FetchURL 官方文档页)。

脆点(先行 smoke):C1 分词器参数形态;C2 BM25 查询 data 形态;C3 Lite 段级 BM25 IDF(segment-local)对小语料分数分布的影响(影响 RERANK_MIN_SCORE 校准,不阻塞);C4 pymilvus import 期环境污染隔离沿用 `_load_milvus_client`,新模块不得直接 `from pymilvus import …`。
