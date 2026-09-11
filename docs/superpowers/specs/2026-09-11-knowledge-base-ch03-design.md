# 电商智能客服 Ch03: RAG 基础 · 知识库与向量语义检索 - 设计文档

日期: 2026-09-11
状态: 设计定稿(brainstorm 三轮确认,等待用户 spec 评审)

## 1. 目标与范围

把 `query_faq` 的内部实现从**关键词 LIKE 查表**升级为**向量语义检索**,工具的入参出参契约保持不变。围绕检索建一条完整的知识库链路:离线文档建库 + 历史对话挖知识 + 双写落库 + 在线 dense 单路检索。

交付五项能力:

1. **离线建库·文档处理**: 知识文档(退货政策、商品 FAQ、售后手册 Markdown)按标题层级结构感知切分;超长递归切;块间重叠且裁到最近句号;大表格按行切且每块复制表头
2. **离线建库·对话挖知识**: CLI 任务从历史客服对话(conversations/messages 表)分批喂 LLM 抽取问答对,先进 `qa_extraction_staging` 暂存、再整体去重入库
3. **落库结构**: 每条知识带 `category`/`questions`/`answer` 三格,拼一段文本向量化;商品 FAQ 与挖出的问答对 questions 填真实问法;政策手册类 questions 填所在章节标题、category 填上级标题路径;另带章节路径、内容类型、是否关键条款、前后块指针四类元数据,只存不进向量
4. **双写落库**: MySQL `knowledge_chunks` 当原文权威源、Milvus Lite `knowledge` 集合存向量;先写 MySQL 记 pending,再写 Milvus,回填 `vector_id`、状态转 done;按主键幂等,挂了能重跑
5. **在线检索**: 问题向量化后 Milvus 按相似度取 Top-K,替换 `query_faq` 关键词查表实现

技术栈(定死): 嵌入模型 **BGE-M3**(经硅基流动 OpenAI 兼容端点,`BAAI/bge-m3`,1024 维);向量库 **Milvus**(嵌入式 **Milvus Lite**,本地文件,零容器);**MySQL** 当原文权威源。

本章明确不做: 关键词召回、混合检索、重排——只跑 dense 向量单路。

验收标准:

1. 「邮费是多少」这类换说法的问题,现在能召回运费说明并答对(ch02 同题如实漏召回,形成对照)
2. 故意中断建库任务再重跑,漏向量化的块能被捡起补齐

## 2. 架构与选型(方案 B: 裸 pymilvus,已选定)

- **Milvus 访问**: `pymilvus` 的 `MilvusClient` 裸 API。Milvus Lite 即 `MilvusClient(uri="./data/milvus_lite.db")`,集合 schema、upsert、search 全部显式手写——双写的每一步(pending → upsert → 回填 done)可见可控,中断重跑就是 `WHERE vectorize_status='pending'`。
- **嵌入**: 复用现有 `langchain-openai` 的 `OpenAIEmbeddings`,构造参数 `model="BAAI/bge-m3"`、`api_key`、`base_url` 指硅基流动;`embed_documents`(建库批调)/`embed_query`(在线单调)。**必须传 `check_embedding_ctx_length=False`**: 默认走 tiktoken 估算长度,tiktoken 不认识 `BAAI/bge-m3`,第三方端点必踩。
- **落选**: A(langchain-milvus VectorStore 全家桶——双写/回填/幂等要绕框架抽象);C(自抽 VectorStore Protocol——本章只有一个实现,YAGNI)。
- **风格延续**: 与 ch02「手写编排而非 AgentExecutor」一致,核心链路不引框架魔法。

组件边界:

- `chunking.py` 纯函数,无 IO,单测主力
- `embedding.py` 只负责「文本 → 向量」,封装 OpenAIEmbeddings 构造与批调
- `milvus_store.py` 只负责「集合存在性 + upsert + search」,不感知 MySQL
- `ingest.py` / `mining.py` 是流程编排者,组合上述三者 + SQLAlchemy session
- `retriever.py` 是在线检索的唯一入口,`query_faq` 工具闭包调它

## 3. 项目结构(新增部分)

```
knowledge_docs/            # 演示知识文档(本章造,入仓库)
  退货政策.md              # type: policy
  商品FAQ.md               # type: faq(含运费说明:满99包邮/未满8元)
  售后手册.md              # type: manual(含大表格,演示表头复制)
app/
  knowledge/
    __init__.py
    chunking.py            # Markdown 结构感知切分(纯函数)
    embedding.py           # OpenAIEmbeddings 封装(硅基流动 BGE-M3)
    milvus_store.py        # Milvus Lite 集合管理(upsert/search)
    ingest.py              # 建库两阶段流水线(load → vectorize)
    mining.py              # 对话挖知识四子阶段
    retriever.py           # 在线语义检索(query→Top-K→回 MySQL 取原文)
  jobs/
    __init__.py
    ingest_docs.py         # python -m app.jobs.ingest_docs [docs_dir]
    mine_qa.py             # python -m app.jobs.mine_qa
tests/
  test_chunking.py  test_ingest.py  test_mining.py  test_retriever.py  ...
evals/
  knowledge_recall.jsonl   # 标注:换说法问题 → 期望命中的知识要点
  run_knowledge_eval.py    # 真实 embedding + 真实 Milvus Lite 跑召回率
db/init/03-ddl.sql         # 与 sql/ch03-ddl.sql 逐字节一致(initdb 惯例延续)
data/                      # Milvus Lite 本地文件目录(gitignore)
```

`app/models.py` 新增 `KnowledgeChunk`、`QaExtractionStaging` 两个 ORM 模型,字段与 `sql/ch03-ddl.sql` 逐项一致。ch02 的 `faq` 表与 seed 原样保留(`test_seed_recall` 不动);`query_faq` 不再读 faq 表。

## 4. 数据契约

### 4.1 MySQL `knowledge_chunks`(用户手写 DDL,`sql/ch03-ddl.sql`)

三格进向量: `category` + `questions` + `answer`(向量化文本 = 三格按 `"\n"` 拼接)。四类元数据只存不进向量: `section_path` / `content_type` / `is_key_clause` / `prev_chunk_id`+`next_chunk_id`。双写字段: `vector_id`(回填 `str(id)`)、`vectorize_status`(pending/done)。`id` 自增主键,与 Milvus 集合主键对齐。

### 4.2 MySQL `qa_extraction_staging`(挖 QA 离线中转)

`batch_no` 分批追溯;`source_ref` = 会话标识(`conv:<conversation_id>`),判重实现「同一会话只挖一次」;`status`: extracted → kept / discarded。建库完成可清空。

### 4.3 Milvus Lite 集合 `knowledge`

只存两列,原文一律回 MySQL 取(权威源单一):

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64,主键,`auto_id=False` | 直接复用 MySQL chunk id |
| `vector` | FLOAT_VECTOR,dim=1024 | COSINE |

建集合(Context7 已核): `client.create_collection(collection_name="knowledge", dimension=1024, metric_type="COSINE", auto_id=False)`;写入用 **`upsert`**(按主键替换——中断在「Milvus 已写、MySQL 未回填」之间时重跑不会产生同 id 重复向量,幂等靠它兜底);检索 `client.search("knowledge", data=[vec], limit=top_k)`,hit 取 `id`/`distance`。COSINE 下 `distance` 即余弦相似度,**越大越相似**,阈值按 `>=` 过滤。

## 5. 文档切分规格(chunking.py,纯函数)

输入: Markdown 文本;输出: chunk 列表(含三格 + 四类元数据草稿,prev/next 入库后回写)。

**文档头约定(frontmatter)**: 每个 .md 以 `---` 块开头,必填 `type: faq | policy | manual`;缺失或未知值报错不猜。

**结构感知切分**: 按标题层级(`#`/`##`/`###`)切 section,维护标题栈;`section_path` 形如「售后手册 > 退换货 > 退货流程」。前言(H1 之前的正文)并入第一个 section。

**三格填法**:

- `type=faq`: 每个 `##` 问答对一个 chunk,`questions` = 该节标题(真实问法),`category` = 文档 H1 标题
- `type=policy|manual`: `questions` = 所在章节标题(末级),`category` = 上级标题路径(标题栈去掉末级,` > ` 连接;无上级则为文档 H1)

**超长递归切**: section 正文 > `max_chunk_chars`(500) 先按段落(空行分隔)切;单段落仍超长再按句末标点硬切。

**重叠不留半截话**: 相邻块重叠 `chunk_overlap_chars`(80) 字;重叠内容从上一块末尾取,向前裁到最近的句末标点(。!?),裁不到则不重叠。

**大表格按行切**: 连续表格行识别为一个表格;超过 `max_chunk_chars` 按行切,**表头行 + 分隔行复制进每一个表格块**。

**is_key_clause**: 命中关键词清单(「不支持」「不予」「必须」「扣除」「逾期」「无效」)→ 1,否则 0。

**prev/next**: 同一文档内 chunks 按序编号,Phase 1 插入后回写指针(首块 prev=NULL,尾块 next=NULL)。

## 6. 双写流水线(ingest.py)

两阶段,中断重跑天然成立:

**Phase 1 · load(只写 MySQL)**

1. 遍历 docs_dir 的 .md(文件名排序,确定性),逐文档切分
2. 逐条判重: `SELECT 1 WHERE category=? AND questions=? AND answer=?`,存在即跳过(三格内容做幂等键;重跑同一文档全部跳过)
3. 新 chunk `INSERT ... vectorize_status='pending'`,每条提交;同文档插完回写 prev/next

**Phase 2 · vectorize(MySQL → Milvus → 回填)**

1. `SELECT * WHERE vectorize_status='pending' ORDER BY id`,按 32 条一批
2. 每批: 拼三格文本 → `embed_documents` → `milvus upsert`(id = chunk id)→ `UPDATE vector_id=str(id), vectorize_status='done'`,逐批提交
3. embedding 或 Milvus 任一步失败: 该批保持 pending,命令非零退出
4. 首批前校验 `len(vector) == 1024`,不符直接报错(防端点配错模型悄悄写歪维度)

**中断演示路径(验收 2)**: Phase 2 中途 kill → 重跑 `ingest_docs` → Phase 1 判重全跳过 → Phase 2 从 pending 捡起补齐;Milvus 侧 upsert 幂等。

CLI: `python -m app.jobs.ingest_docs [docs_dir]`,默认 `knowledge_docs/`。先跑 Phase 2  resume(捡起历史 pending)再跑 Phase 1+2 正常流程——保证「重跑任何时刻都能补齐」。

## 7. 对话挖知识(mining.py)

`python -m app.jobs.mine_qa`,四个子阶段,重跑幂等:

1. **拉会话**: 遍历 conversations,已在 staging 出现过 `source_ref=conv:<id>` 的跳过(同一会话只挖一次)。对话文本取 user/assistant 行 content 拼「用户:/客服:」,tool 行跳过
2. **分批抽取**: 每 `mining_batch_size`(10) 个会话一批,`batch_no` = UTC 时间戳+序号;用现有 chat model + `with_structured_output` 出 `{items: [{question, answer}]}`(提示词要求抽可复用知识,不要订单号等个体信息);LLM 输出入 staging(`status='extracted'`)。某批失败: 该批不写入,记日志继续下一批
3. **整体去重**(确定性,不引第三个模型): 对 staging 全部 extracted 行——
   - 内部: `question` 规范化(去空白/中英文标点/转小写)后相同,留一余者 `discarded`
   - 对已有知识: 规范化 question 与 `knowledge_chunks.questions` 按行比对,命中 `discarded`
   - 幸存 → `kept`
4. **入库+向量化**: kept → `knowledge_chunks`(`content_type='qa_mined'`、`category='对话挖掘'`、`questions`=抽出的问法、`section_path=NULL`、`is_key_clause=0`、prev/next=NULL;三格判重逻辑同 Phase 1)→ 复用 §6 Phase 2 同一函数向量化补齐

## 8. 在线检索(retriever.py + query_faq 替换)

**契约一字不动**: 入参仍 `keyword`(模型照旧传词/短句);出参仍 `{"results": [{"question","answer","category"}]}`,空结果仍带 `note`。前端徽章、envelope、prompt、SSE 帧全不动。docstring 更新为「查询知识库。参数 keyword 为用户问题或关键词,语义检索返回最相关的前 5 条」。

**链路**: `keyword` → `embed_query` → Milvus `search`(top_k=`knowledge_top_k`=5)→ 按 `distance >= knowledge_min_score` 过滤 → 按命中 id 回 MySQL `SELECT questions, answer, category WHERE id IN (...)`(按相似度排序返回)→ `question` 填 chunk 的 `questions` 字段。

**装配**: `AppRuntime` 新增 `KnowledgeRetriever`(构造: settings + embedding client + MilvusClient 工厂 + session_factory);`build_tools(session_factory, conversation_id, retriever)` 闭包注入。测试注入 fake retriever。

**兜底**: Milvus 文件/集合不存在(`has_collection` 判空)、空库、检索异常 → 返回空结果 note,不炸聊天;只读工具照旧享受 executor 超时重试。

**并发约束**: Milvus Lite 是本地文件单写者——CLI 建库/挖矿与在线服务不同时写;README 写明。

## 9. 配置新增(config.py + .env.example)

| 配置 | 默认 | 说明 |
|---|---|---|
| `embedding_base_url` | `https://api.siliconflow.cn/v1` | 硅基流动 OpenAI 兼容端点 |
| `embedding_api_key` | (必填,无默认) | 用户后填 |
| `embedding_model` | `BAAI/bge-m3` | BGE-M3 |
| `embedding_dim` | `1024` | 建集合与首向量校验用 |
| `milvus_uri` | `./data/milvus_lite.db` | Milvus Lite 文件;测试用 tmp 路径 |
| `knowledge_top_k` | `5` | Top-K |
| `knowledge_min_score` | `0.35` | COSINE 相似度门槛(占位,§11 评估集校准后定终值) |
| `mining_batch_size` | `10` | 每批会话数 |
| `max_chunk_chars` | `500` | 切分上限 |
| `chunk_overlap_chars` | `80` | 块间重叠 |

`milvus_uri` 走本地文件不涉网络;`embedding_api_key` 缺失时建库/挖矿命令显式报错退出(在线检索兜底为空 note)。

## 10. 错误处理

| 场景 | 行为 |
|---|---|
| embedding API 失败(建库/挖矿) | 该批保持 pending,命令非零退出,重跑补齐 |
| embedding API 失败(在线) | executor 超时重试,最终 error envelope(现有机制) |
| LLM 抽取失败(某批) | 该批不入 staging,记日志继续;重跑只补未挖会话 |
| Milvus 集合不存在/空库(在线) | 空结果 note |
| 向量维度 ≠ embedding_dim | 建库首批即报错退出 |
| 文档 frontmatter 缺 type | 报错列出文件名,不猜 |
| CLI 与服务并发写 Milvus Lite | 不防御,README 写明单写者约束 |

## 11. 测试策略与验收

**可单测代码走 TDD**:

- `chunking.py` 纯函数: 层级切分/递归超长切/重叠句号裁剪(不留半截话)/表格表头逐块复制/frontmatter 解析与报错/四类元数据/is_key_clause 关键词命中
- 双写(真实 Docker MySQL + 真实 Milvus Lite tmp 文件 + **fake embedding** 确定性向量): pending→done 流转、vector_id 回填、三格判重跳过、**中断重跑**(Phase 2 注入异常 → 重跑 → 全 done、Milvus 向量数 == MySQL 行数)、维度不符报错
- 挖矿(stub chat model): 抽取入 staging、source_ref 判重(重跑不重复抽)、内部去重留一、对已有知识去重、kept 入库字段正确
- 检索(fake embedding + 真实 Milvus Lite + 真实 MySQL): Top-K、阈值过滤、出参 key 结构不变、集合不存在兜底空 note
- `db/init/03-ddl.sql` 与 `sql/ch03-ddl.sql` 逐字节一致(ch02 惯例断言延续)

**评估集验证(纯 Prompt/数据类任务的 TDD 替代)**: `evals/knowledge_recall.jsonl` 标注「换说法问法 → 期望命中知识要点」(含「邮费是多少→运费说明(满99包邮/8元)」),`evals/run_knowledge_eval.py` 用**真实硅基流动 embedding + 真实 Milvus Lite** 跑召回命中率;`embedding_api_key` 缺失时显式报错退出(延续 ch02「不静默 skip」惯例)。阈值 `knowledge_min_score` 终值由该评估校准。

**演示验收(用户亲测)**:

1. 浏览器问「邮费是多少」→ 召回运费说明并答对(与 ch02 漏召回对照)
2. `ingest_docs` 跑到一半 kill → 重跑 → SQL 查 `vectorize_status` 全 done、Milvus 向量数与 MySQL 行数一致

## 12. 依赖与 Context7 核对清单

- 依赖新增仅 `pymilvus`(>= 2.4,含 Milvus Lite;Linux 平台);`openai` 包已由 `langchain-openai` 传递引入
- 已核(Context7,2026-09-11): `MilvusClient(uri=本地文件)` / `create_collection(collection_name, dimension, metric_type="COSINE", auto_id=False)` / `upsert(collection_name, data)` / `search(collection_name, data, limit)` / hit 的 `id`/`distance` / `OpenAIEmbeddings(model, api_key, base_url)` / `embed_query` / `embed_documents`
- 实现前仍需核: `OpenAIEmbeddings` 的 `check_embedding_ctx_length` 参数当前名;`with_structured_output` 在本项目模型端点的可用方式(ch01 已有先例照用);pymilvus `has_collection` 签名
- 硅基流动端点与 `BAAI/bge-m3` 可用性: 用户填 key 后由 eval 脚本实测确认
