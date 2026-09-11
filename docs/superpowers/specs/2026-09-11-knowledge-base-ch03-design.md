# 电商智能客服 Ch03: RAG 基础 · 知识库与向量语义检索 - 设计文档

日期: 2026-09-11
状态: 评审修订完成(2026-09-11 用户已确认修订方案,待实现)

本次修订落实八项评审意见: QA 来源、独立抽取进度、文档位置幂等、Milvus Lite 依赖、缺 Key 降级、异常重试、切分边界和检索评估。用户已确认新增 `qa_mining_progress` 表,以及 `knowledge_chunks` 的两个来源字段和唯一索引。本次修改范围为 spec;现有 `sql/ch03-ddl.sql` 是待同步的基线,具体 DDL 同步要求见 §4.5。

## 1. 目标与范围

把 `query_faq` 的内部实现从**关键词 LIKE 查表**升级为**向量语义检索**,工具的入参出参契约保持不变。围绕检索建一条完整的知识库链路:离线文档建库 + 历史对话挖知识 + 双写落库 + 在线 dense 单路检索。

交付五项能力:

1. **离线建库·文档处理**: 知识文档(退货政策、商品 FAQ、售后手册 Markdown)按标题层级结构感知切分;超长递归切,必要时按字符硬切;重叠仅取完整句;大表格按行切且每块复制表头,单行无法容纳时报错定位
2. **离线建库·对话挖知识**: CLI 任务从历史客服对话(conversations/messages 表)分批喂 LLM,按会话输出问答对;结果进 `qa_extraction_staging`,抽取成功记录进 `qa_mining_progress`,再整体去重入库;成功但零 QA 也记录进度
3. **落库结构**: 每条知识带 `category`/`questions`/`answer` 三格,拼一段文本向量化;商品 FAQ 与挖出的问答对 questions 填真实问法;政策手册类 questions 填所在章节标题、category 填上级标题路径;另带章节路径、内容类型、是否关键条款、前后块指针四类元数据,以及文档来源与块序号,均不进向量
4. **双写落库**: MySQL `knowledge_chunks` 当原文权威源、Milvus Lite `knowledge` 集合存向量;文档块与指针先在一个 MySQL 事务中提交并记 pending,再写 Milvus,回填 `vector_id`、状态转 done;文档按来源和位置复用 ID,Milvus 按 ID upsert,中断后重跑补齐
5. **在线检索**: 问题向量化后 Milvus 按相似度取 Top-K,替换 `query_faq` 关键词查表实现

技术栈(定死): 嵌入模型 **BGE-M3**(经硅基流动 OpenAI 兼容端点,`BAAI/bge-m3`,1024 维);向量库 **Milvus**(嵌入式 **Milvus Lite**,本地文件,零容器);**MySQL** 当原文权威源。

本章明确不做: 关键词召回、混合检索、重排——只跑 dense 向量单路。

文档生命周期限定为首次导入和原样重跑。同一路径已导入文档的切分结果发生变化时明确报错,不自动覆盖旧知识;文档更新、删除、版本迁移及 Milvus 数据丢失后的全量重建不属于本章恢复承诺。恢复承诺针对同一份源数据、相同切分配置和持久化数据仍在的任务中断。

验收标准:

1. 「邮费是多少」这类换说法的问题,现在能召回运费说明并答对(ch02 同题如实漏召回,形成对照)
2. 故意中断建库任务再重跑,未提交的文档可重新导入,pending 块可补齐向量,前后指针完整;两库有效主键集合一致
3. 固定评估集的独立验收分片达到 Recall@5 ≥ 90%、无答案负例误召回率 ≤ 10%,且「邮费是多少」必须命中(指标与分片规则见 §11)

## 2. 架构与选型(方案 B: 裸 pymilvus,已选定)

- **Milvus 访问**: `pymilvus` 的 `MilvusClient` 裸 API。Milvus Lite 即 `MilvusClient(uri="./data/milvus_lite.db")`,集合 schema、upsert、search 全部显式手写——双写的每一步(pending → upsert → 回填 done)可见可控,中断重跑就是 `WHERE vectorize_status='pending'`。
- **嵌入**: 复用现有 `langchain-openai` 的 `OpenAIEmbeddings`,构造参数 `model="BAAI/bge-m3"`、`api_key`、`base_url` 指硅基流动;`embed_documents`(建库批调)/`embed_query`(在线单调)。**传 `check_embedding_ctx_length=False`**,向兼容端点发送原始文本,避免依赖 OpenAI tokenizer 的模型映射或向第三方端点发送 token ID 数组。在线 embedding 客户端关闭 SDK 自动重试(`max_retries=0`),网络超时不高于单次 `tool_timeout_seconds`,由 executor 统一管理重试(§8)。
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
    mining.py              # 对话挖知识四子阶段 + 独立抽取进度
    retriever.py           # 在线语义检索(query→Top-K→回 MySQL 取原文)
  jobs/
    __init__.py
    ingest_docs.py         # python -m app.jobs.ingest_docs [docs_dir]
    mine_qa.py             # python -m app.jobs.mine_qa
tests/
  test_chunking.py  test_ingest.py  test_mining.py  test_retriever.py  ...
evals/
  knowledge_recall.jsonl   # 正/负例,固定 calibration/test 分片和来源标注
  run_knowledge_eval.py    # 真实 embedding + Milvus Lite + MySQL,校准后独立验收
db/init/03-ddl.sql         # 实现时与按 §4.5 修订后的 sql/ch03-ddl.sql 逐字节一致
data/                      # Milvus Lite 本地文件目录(gitignore)
```

`app/models.py` 新增 `KnowledgeChunk`、`QaExtractionStaging`、`QaMiningProgress` 三个 ORM 模型,字段和约束与按 §4.5 修订后的 DDL 一致。ch02 的 `faq` 表与 seed 原样保留(`test_seed_recall` 不动);`query_faq` 不再读 faq 表。

## 4. 数据契约

### 4.1 MySQL `knowledge_chunks`(基于用户手写 DDL 扩展)

三格进向量: `category` + `questions` + `answer`(向量化文本 = 三格按 `"\n"` 拼接)。四类元数据只存不进向量: `section_path` / `content_type` / `is_key_clause` / `prev_chunk_id`+`next_chunk_id`。双写字段: `vector_id`(回填 `str(id)`)、`vectorize_status`(pending/done)。`id` 自增主键,与 Milvus 集合主键对齐。

新增两个只存不进向量的来源字段:

| 字段/索引 | 类型或约束 | 规则 |
|---|---|---|
| `source_doc` | `VARCHAR(255) NULL`,大小写敏感的路径比较 | 文档来源标识;文档块必填,挖掘 QA 为 NULL |
| `chunk_index` | `INT UNSIGNED NULL` | 同文档从 1 开始的连续序号;挖掘 QA 为 NULL |
| `uk_doc_chunk` | `UNIQUE(source_doc, chunk_index)` | 同一文档位置只允许一个 chunk;不同文档即使三格相同也分别落行 |

`source_doc` 的确定方式: 先解析文件真实路径;仓库内文件使用相对于仓库根的 POSIX 路径(例如 `knowledge_docs/商品FAQ.md`),仓库外文件使用绝对 POSIX 路径。标识不随 CLI 传入目录的层级变化;超过 255 字符报错,不得截断。路径比较保留大小写差异,DDL 使用相应的二进制排序规则。移动源文件会改变来源标识,不视为原样重跑。

文档块两个来源字段必须同时有值;挖掘 QA 两者均为 NULL。`prev_chunk_id` / `next_chunk_id` 只能指向同一 `source_doc` 中相邻 `chunk_index` 的行,不能跨文档共享节点。三格内容不再作为文档的全局幂等键;挖掘 QA 的内容去重单独按 §7 处理。

### 4.2 MySQL `qa_extraction_staging`(挖 QA 离线中转)

保留现有表结构。`batch_no` 分批追溯;本任务写入的 `source_ref` 必须为 `conv:<conversation_id>`,由程序根据校验后的会话 ID 填充,不让模型自行拼接。`status`: extracted → kept / discarded。该表记录 QA 候选和去重结果,不承担会话处理游标。

允许在任务成功完成后人工清理 staging: 先确认没有 extracted 行,且所有 kept 行对应的知识已入库、向量状态为 done。CLI 不自动清表;清理 staging 不得连带清理 §4.3 的进度记录。中断任务尚未完成时保留全部候选,供重跑继续去重和入库。

### 4.3 MySQL `qa_mining_progress`(新增独立抽取进度)

| 字段 | 类型/约束 | 说明 |
|---|---|---|
| `conversation_id` | `BIGINT UNSIGNED NOT NULL PRIMARY KEY` | 成功抽取的会话 ID,与 conversations.id 对应 |
| `batch_no` | `VARCHAR(64) NOT NULL` | 成功抽取所属批次 |
| `qa_count` | `INT UNSIGNED NOT NULL` | 去重前抽出的 QA 数量,允许为 0 |
| `extracted_at` | `DATETIME NOT NULL` | 程序填入 UTC 抽取成功时间 |

一行表示该会话的抽取结果已成功提交,不表示去重或向量化已完成。成功但 `items=[]` 的会话也必须落一行;失败会话不落进度。每批全部 staging 行与对应的进度行在同一 MySQL 事务中提交,任一失败整体回滚。

重跑依据本表跳过已成功提交的会话,包括零 QA 会话;清空 staging 后仍保留此语义。LLM 已返回但事务尚未提交时进程中断,允许重试该次抽取;不承诺外部 LLM 请求物理上只调用一次。

### 4.4 Milvus Lite 集合 `knowledge`

只存两列,原文一律回 MySQL 取(权威源单一):

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64,主键,`auto_id=False` | 直接复用 MySQL chunk id |
| `vector` | FLOAT_VECTOR,dim=1024 | COSINE |

建集合(Context7 已核): `client.create_collection(collection_name="knowledge", dimension=1024, metric_type="COSINE", auto_id=False, enable_dynamic_field=False)`,显式关闭动态字段以保持 id/vector 两列契约;写入用 **`upsert`**(按主键替换——中断在「Milvus 已写、MySQL 未回填」之间时重跑不会产生同 id 重复向量,幂等靠它兜底);检索 `client.search("knowledge", data=[vec], limit=top_k)`,hit 取 `id`/`distance`。COSINE 下 `distance` 即余弦相似度,**越大越相似**,阈值按 `>=` 过滤。

### 4.5 后续 DDL 与 ORM 同步要求

本次先修订 spec,现有 `sql/ch03-ddl.sql` 尚未包含上述变更。实现前必须完成:

1. 在 `knowledge_chunks` 增加 `source_doc`、`chunk_index` 和 `uk_doc_chunk` 唯一索引;来源字段不影响向量化三格或工具出参。
2. 新增 §4.3 的 `qa_mining_progress` 表;`qa_extraction_staging` 保留现有结构。本章新表总数由两张变为三张。
3. 将修订后的完整建表文件同步至 `db/init/03-ddl.sql`,保持逐字节一致,再同步三个 ORM 模型及数据库测试的建表/清理清单。
4. README 区分首次初始化和已有 ch02 MySQL 数据卷的升级: 已有数据卷需显式执行 ch03 DDL,不能只添加 init 文件后重启容器。不得为建新表删除 ch02 数据卷;如已应用旧版 ch03 DDL,先检查现有数据再做增量变更,不得把无来源的文档行静默当成合格的新格式。

## 5. 文档切分规格(chunking.py,纯函数)

输入: Markdown 文本;输出: 有序 chunk 列表(含三格 + 四类元数据草稿)。`source_doc` 由 IO 编排层确定,`chunk_index` 按输出顺序从 1 编号;prev/next 在文档事务内取得 ID 后回写。

**文档头约定(frontmatter)**: 每个 .md 以 `---` 块开头,必填 `type: faq | policy | manual`;缺失或未知值报错不猜。

**结构感知切分**: 按标题层级(`#`/`##`/`###`)切 section,维护标题栈;`section_path` 形如「售后手册 > 退换货 > 退货流程」。文档必须有一个 H1 文档标题;前言(H1 之前的正文)并入第一个可产出知识块的 section。仅有标题而无正文的 section 不单独成块;整个文档无可入库正文时报错。

**三格填法**:

- `type=faq`: 每个 `##` 问答对为一个切分单元,超长时允许产出多个 chunk;这些块的 `questions` 均为该节标题(真实问法),`category` 均为文档 H1 标题。`###` 作为该回答的小标题保留在正文,不另造问法
- `type=policy|manual`: `questions` = 所在章节标题(末级),`category` = 上级标题路径(标题栈去掉末级,` > ` 连接;无上级则为文档 H1)

**长度定义与优先级**: `max_chunk_chars`(500) 是最终 `answer` 的字符上限,按 `len(answer)` 计数,包含重叠、换行及复制的表头/分隔行,不包含 category/questions。先把换行规范化为 `\n`。所有文档块均须满足该上限;完整句优先于可选重叠,单句自身超限时允许硬切。必须保留表格结构的单行无法容纳时明确报错。

**超长递归切**: 先识别标题、段落和连续表格,表格走下述专用规则。普通 section 正文超限先按段落(空行分隔)打包;单段仍超限再按句末标点 `。！？!?` 切句;没有可用标点或单句自身仍超限时按字符硬切,不得无限递归或产生空块。

**重叠仅取完整句**: 只在同一个 section 内相邻的普通文本块之间加入重叠,不跨章节、不与表格块重叠。从上一块末尾选择不超过 `chunk_overlap_chars`(80) 的完整句后缀,保留句末标点;无法形成完整句后缀时不重叠。为重叠和连接换行预留长度预算后再装入新正文;若与下一完整句冲突,先缩短或取消重叠,保持最终 answer 不超限。字符硬切产生的不完整尾句不得当成重叠内容。

**大表格按行切**: 连续 Markdown 表格行识别为一个表格,与前后普通正文分别成块。**表头行 + 分隔行复制进每一个表格块**,把它们和换行计入长度后按整行装入数据;表格块之间不额外重叠数据行。若表头/分隔行本身超限,或加任意单个数据行就超限,报错给出源行号,由 CLI 补充文件名,提示人工缩短该行或拆分源表格;不得截断单元格或静默放宽上限。

**is_key_clause**: 命中关键词清单(「不支持」「不予」「必须」「扣除」「逾期」「无效」)→ 1,否则 0。

**prev/next**: 同一文档内 chunks 按序编号,Phase 1 的文档事务内回写指针(首块 prev=NULL,尾块 next=NULL)。普通正文与表格都参与同一文档序列,不同文档即使内容相同也不共用行。

## 6. 双写流水线(ingest.py)

两阶段;文档位置唯一键、MySQL 文档事务和 Milvus 主键 upsert 分别保证重跑身份、原文/指针原子性和向量写入幂等:

**Phase 1 · load(只写 MySQL)**

1. 按规范化路径排序遍历 docs_dir 下的 .md,逐文档完整切分和校验后,确定 `source_doc` 及连续 `chunk_index`。
2. 每份文档开启一个 MySQL 事务,按 source_doc 读取全部已有行并按 chunk_index 排序。若已存在该来源,其行数、序号、三格和四类元数据中的非指针字段必须与本次结果一致;一致则复用全部 ID。若不一致,明确报错“已导入文档或切分配置发生变化”,回滚该文档事务,不覆盖旧内容、不新增另一套同来源知识。
3. 没有该来源的行时,插入该文档全部 chunks,`vectorize_status='pending'`,通过 flush 获取 ID,此时不提交。不同 source_doc 的相同内容分别插入。
4. 使用全部 ID 按序回写 prev/next(原样重跑也重建指针),最后只提交一次。异常或进程中断时整篇文档事务回滚;已提交的其他文档保留。数据库唯一索引负责兜底同一来源位置的重复插入。

**Phase 2 · vectorize(MySQL → Milvus → 回填)**

1. `SELECT * WHERE vectorize_status='pending' ORDER BY id`,按 32 条一批
2. 每批: 拼三格文本 → `embed_documents` → 校验返回向量数等于输入数、每条维度等于 `embedding_dim=1024` → `milvus upsert`(id = chunk id)→ `UPDATE vector_id=str(id), vectorize_status='done'`,逐批提交。
3. embedding、Milvus 或 MySQL 回填任一步失败: 该批不提交 done,仍为 pending,命令非零退出。Milvus 已写而 MySQL 未提交时,重跑会再次按同 ID upsert,不会生成另一套主键。
4. 首次写入前校验已存在集合的维度/度量/主键契约;不符报错,不得自动删除集合重建。向量长度校验覆盖每一批、每一条,不是只检查首条。

**中断演示路径(验收 2)**: Phase 1 文档事务中途 kill → 该文档没有半套块或指针,重跑重新导入;Phase 2 中途 kill → 重跑从 pending 补齐,已导入文档复用 ID;Milvus 已 upsert、MySQL 未提交 done 的窗口单独验证。

CLI: `python -m app.jobs.ingest_docs [docs_dir]`,默认 `knowledge_docs/`。校验 Key 和库契约后,先跑 Phase 2 resume(捡起历史 pending),再跑 Phase 1+2 正常流程。最终完成标准是 MySQL 无 pending、指针完整、Milvus 有效主键集合与 MySQL chunk 主键集合一致;仅行数相同不足以证明补齐。

## 7. 对话挖知识(mining.py)

`python -m app.jobs.mine_qa`,四个子阶段。启动时先校验 Key,恢复历史 staging 的去重/kept 入库及全部 pending 向量化,完成后再拉新会话。

1. **拉会话**: 按 conversations.id 升序遍历,已存在 `qa_mining_progress.conversation_id` 的跳过。以本次读取的消息快照为输入,按消息 created_at、id 排序,仅取非空的 user/assistant content 拼“用户:/客服:”,跳过 tool 行。成功抽取后同一会话不再自动重挖,后续新增消息不纳入本章“一次抽取”的范围。
2. **分批抽取**: 每批最多 `mining_batch_size`(10) 个会话,`batch_no` = UTC 时间戳+序号。输入显式携带每个 conversation_id 和对应消息边界;使用现有 chat model 的 `with_structured_output`,沿用 `structured_output_method` 与解析错误处理约定。批次同时受模型输入预算约束;超限时缩小批次,单个会话仍超限则记录失败,不得静默截断后标记成功。

   输出 schema:

   ```json
   {"conversations": [{"conversation_id": 123, "items": [{"question": "真实问法", "answer": "可复用答案"}]}]}
   ```

   每个输入会话必须恰好出现一次;允许 `items=[]`,不允许遗漏、重复或编造会话 ID。question/answer 去除首尾空白后必须非空,question 规范化后也不得为空。提示词要求抽取有原文依据的可复用知识,不要订单号等个体信息,不得合并不同会话的事实。

   整批结构校验通过后,在一个 MySQL 事务中插入全部 staging 行(`status='extracted'`,程序填 `source_ref=conv:<id>`)及每个会话的进度行(`qa_count=len(items)`,含零 QA)。LLM、解析或提交失败,整批不留 staging/进度,日志记录批次及会话 ID 后继续下一批;本轮有失败批次时最终非零退出,重跑仅补未成功提交的会话。
3. **整体去重**(确定性,不引第三个模型): 对全部 extracted 行按 staging.id 升序处理。question 规范化为去空白、去中英文标点、转小写。先与已有 `knowledge_chunks.questions` 按行规范化比对,再与 staging 中已 kept 的问法及本次已选幸存问法比对;命中则 discarded,否则 kept。已有知识优先,候选内部保留最早 ID,相同问法但不同答案也按此优先级保留一条。已 kept 行参与判重,避免中断后尚未入库的候选被新批次重复保留。
4. **入库+向量化**: 遍历全部 kept 行,在 `content_type='qa_mined'` 且来源字段均为空的记录中按完整三格精确匹配复用已入库记录,没有则写入 pending chunk。字段为 `content_type='qa_mined'`、`category='对话挖掘'`、`questions`=抽出的问法、`answer`=抽出的答案、`section_path=NULL`、`is_key_clause=0`、`source_doc=NULL`、`chunk_index=NULL`、prev/next=NULL。此处的三格匹配只用于 kept 候选的入库重放,不替代步骤 3 的问法去重。最后复用 §6 Phase 2 向量化补齐。

四个阶段均可重入: staging 和抽取进度保证成功结果不丢,kept 入库重放保证不重复插入同一候选,pending/upsert 保证向量可补齐。只有去重、入库及向量化全部成功后,才满足 §4.2 的人工清理条件。

## 8. 在线检索(retriever.py + query_faq 替换)

**契约一字不动**: 入参仍 `keyword`(模型照旧传词/短句);出参仍 `{"results": [{"question","answer","category"}]}`,空结果仍带 `note`。前端徽章、envelope、prompt、SSE 帧全不动。docstring 更新为「查询知识库。参数 keyword 为用户问题或关键词,语义检索返回最相关的前 5 条」。

**链路**: 先检查配置和 Milvus 文件/集合存在性 → `keyword` → `embed_query` → 校验查询向量维度 → Milvus `search`(top_k=`knowledge_top_k`=5)→ 按 `distance >= knowledge_min_score` 过滤 → 按命中 id 回 MySQL 读取 id、questions、answer、category、source_doc、chunk_index(按相似度排序返回)→ 工具的 `question` 填 chunk 的 `questions` 字段。

retriever 的内部命中记录保留 `chunk_id`、`score`、`source_doc`、`chunk_index` 和三格,供评估按来源标注比对;`query_faq` 闭包仅投影出原有 question/answer/category 字段。评估使用同一次检索的内部记录,不另写一套召回流程,也不向工具出参增加调试字段。

**装配**: `AppRuntime` 新增 `KnowledgeRetriever`(构造: settings + 可选 embedding client + MilvusClient 工厂 + session_factory);`build_tools(session_factory, conversation_id, retriever)` 闭包注入。Key 缺失时构造禁用检索的实例,不构造需要 Key 的 embedding 客户端;测试注入 fake retriever。在线 Milvus 客户端延迟初始化、进程内复用,应用关闭时释放;在线入口不负责创建集合。

**空结果与故障分开处理**:

- Key 缺失: `{"results": [], "note": "知识检索未配置"}`,应用正常启动。
- Milvus 文件不存在: 在构造 MilvusClient 前检查路径并返回“知识库尚未建立”;集合不存在(`has_collection` 判空)、空库或无命中也返回明确的空结果 note。
- embedding 的连接失败、超时、限流,以及明确可恢复的 Milvus 连接/超时: 由知识库适配层转换为统一 `RetryableKnowledgeError`,加入 executor 的 `RETRYABLE`。OpenAI SDK 的 `APIConnectionError`、`APITimeoutError`、`RateLimitError` 不属于现有异常集合,必须显式映射;HTTP 408、409、5xx 同样按暂时故障处理。
- executor 按现有 `tool_timeout_seconds` / `tool_max_retries` 重试;在线 SDK 的 `max_retries=0` 避免叠加重试。耗尽后使用现有 `tool_unavailable` error envelope。
- 鉴权失败、参数错误、模型或向量维度不匹配及未分类的内部异常不重试,交给 executor 生成 `tool_error` error envelope。任何实际调用故障都不得伪装成正常未命中。

上述异常在工具层收敛,聊天仍按现有错误工具结果继续作答,前端徽章、envelope 字段、prompt 和 SSE 帧协议不变。

**运行约束**: 本章采用本地库独占打开的运维约束,README 写明先停在线服务、运行 CLI、完成后再起服务。服务使用单 worker;同一数据集不并行运行两个知识任务或评估任务。此约束同时避免本地库的多进程访问问题和 MySQL 内容去重的并发竞争,不新增任务调度或并发协调组件。

## 9. 配置新增(config.py + .env.example)

| 配置 | 默认 | 说明 |
|---|---|---|
| `embedding_base_url` | `https://api.siliconflow.cn/v1` | 硅基流动 OpenAI 兼容端点 |
| `embedding_api_key` | `""` | Settings 允许缺失;CLI 必填,在线缺失时禁用检索 |
| `embedding_model` | `BAAI/bge-m3` | BGE-M3 |
| `embedding_dim` | `1024` | 集合契约及每条文档/查询向量校验用 |
| `milvus_uri` | `./data/milvus_lite.db` | Milvus Lite 文件;测试用 tmp 路径 |
| `knowledge_top_k` | `5` | Top-K |
| `knowledge_min_score` | `0.35` | 开发初值;§11 校准并冻结后更新发布默认值 |
| `mining_batch_size` | `10` | 每批会话数 |
| `max_chunk_chars` | `500` | 最终文档块 answer 上限,含重叠、表头及换行 |
| `chunk_overlap_chars` | `80` | 普通文本块完整句重叠的上限,允许实际为 0 |

`embedding_api_key` 按去除首尾空白后的值判断是否配置,不得在 Settings 声明为无默认的必填字段,也不回退使用聊天模型的 Key。建库、挖矿和真实评估命令在写库前显式校验,缺失时报错退出;在线返回 §8 的未配置 note。

配置校验: 本章 `embedding_dim` 固定为 1024;`knowledge_top_k`、`mining_batch_size`、`max_chunk_chars` 均为正数;`0 <= chunk_overlap_chars < max_chunk_chars`;`knowledge_min_score` 位于 [-1, 1]。Milvus 使用本地路径模式;在线请求仍会访问远程 embedding API。

## 10. 错误处理

| 场景 | 行为 |
|---|---|
| Key 缺失(在线) | Settings/应用启动成功;空结果 note“知识检索未配置” |
| Key 缺失(建库/挖矿/评估) | 写库前显式报错,非零退出 |
| embedding/Milvus/回填失败(建库/挖矿向量化) | 该批不提交 done,保持 pending,命令非零退出,重跑补齐 |
| 连接、超时、限流或可重试 HTTP 状态(在线) | 映射可重试异常;executor 耗尽重试后 tool_unavailable error envelope |
| 鉴权、参数、维度或未分类内部错误(在线) | 不重试;tool_error error envelope |
| LLM 抽取/结构校验/提交失败(某批) | 该批不留 staging 和进度,日志记录后继续其他批次;整轮非零退出 |
| LLM 成功且某会话零 QA | 写 qa_count=0 的进度,重跑跳过该会话 |
| Milvus 文件/集合不存在、空库或无命中(在线) | 正常空结果 note;文件不存在时不创建客户端 |
| 返回向量数或维度不符、已存在集合契约不符(离线) | 写向量前报错,保留 pending,非零退出 |
| 文档格式错误、无正文、源路径过长或表头+单行超限 | 列出文件名及可用的行号;该文档不入库,非零退出 |
| 已导入文档的切分结果变化 | 报错,该文档事务回滚,不自动覆盖 |
| Phase 1 中断(插入或指针回写期间) | 当前整篇文档事务回滚,重跑重新导入 |
| CLI/服务/评估并发打开同一 Milvus 本地库 | 运维流程要求独占;README 写明停服运行及单 worker,不承诺并发访问 |

## 11. 测试策略与验收

**可单测代码走 TDD**:

- `chunking.py` 纯函数: 层级切分、FAQ 超长答案共享问法、无标点长句硬切、完整句重叠及预算不足时退让、不跨 section 重叠、表头逐块复制、表头+单行超限报错、最终 answer 长度始终不超限、frontmatter/H1/空正文报错、四类元数据和 is_key_clause 关键词命中。
- 双写(真实 Docker MySQL + 真实 Milvus Lite tmp 路径 + **fake embedding** 确定性向量): pending→done、vector_id 回填、文档位置唯一约束、原样重跑 ID 不变、两个文档的相同内容分别落行且各自成链、文档变化拒绝覆盖、每批向量数量/维度校验、已存在集合契约校验。
- 中断恢复: 在文档插入中途及指针回写后但提交前注入失败,确认该文档整体回滚;在 Milvus upsert 前及 upsert 成功/MySQL 提交前注入失败,重跑后确认全 done、前后指针完整、两库有效主键集合相等(去重统计有效 ID,不以可能包含历史版本的累计行数代替)。
- 挖矿(stub chat model): 多会话多 QA 的来源准确、遗漏/重复/未知 ID 拒绝整批、空 QA/空白字段处理、staging 与进度同事务回滚、成功零 QA 重跑不再请求 LLM、完成后清空 staging 仍可凭进度跳过、kept 未入库时的中断恢复、问法去重确定性、kept 入库字段及重放正确。真实 MySQL fixture 包含进度表,各测试独立清理。
- 检索(fake embedding + 真实 Milvus Lite + 真实 MySQL): Top-K、阈值边界、按相似度顺序返回、出参 key 不变、缺文件时不创建文件/客户端、缺集合/空库/无命中 note。通过真实 executor 验证 SDK 连接/超时/限流异常映射及重试次数、耗尽 envelope、鉴权/参数错误零重试;验证缺 Key 时应用能启动且不构造 embedding 客户端。
- CLI 缺 Key、抽取部分批次失败、文档变化和格式错误均非零退出;失败批次不阻止其他可处理抽取批次完成。
- 三个 ORM 模型/来源唯一索引符合修订后 DDL;`db/init/03-ddl.sql` 与 `sql/ch03-ddl.sql` 逐字节一致(ch02 惯例断言延续)。

**评估集验证(纯 Prompt/数据类任务的 TDD 替代)**:

1. `evals/knowledge_recall.jsonl` 至少包含 20 条有答案正例和 20 条无答案负例;正、负例各固定分半到 `calibration` 与 `test`(最低每片 10 正+10 负)。正例覆盖政策、FAQ、手册和换说法问法;负例包含店铺业务范围内但知识库没有答案的问题。相同测试问句不得进入校准集。
2. 每行包含唯一 `id`、`split`、`query`、`answerable`、`relevant_chunks` 和期望答案要点。相关块使用固定语料的 `source_doc + chunk_index` 标注,不依赖 MySQL 自增 ID;负例的相关块列表为空。一个正例可标注多个相关块,允许的同义答案位置必须预先列入标注。
3. 使用只含固定 `knowledge_docs/` 语料的独立评估 MySQL 库和临时 Milvus 库,通过真实 retriever 链路调用**真实硅基流动 embedding + 真实 Milvus Lite + 真实 MySQL 回表**。不混入会变化的历史挖掘 QA。`embedding_api_key` 缺失或上游/建库失败时显式报错,非零退出,不静默 skip,也不把调用错误算成正确拒答。
4. 固定 Top-K=5。正例逐条计算 `Recall@5 = 返回的相关块数 / 标注相关块数`,再对全部正例取宏平均;负例误召回率 = 返回非空 results 的负例数 / 全部负例数。输出每例命中、分数、期望来源及两个汇总指标。
5. 只用 calibration 分片选择 `knowledge_min_score`: 在 COSINE [-1, 1] 范围搜索阈值,先满足负例误召回率 ≤ 10%,再最大化正例 Recall@5,指标相同时取较**低**阈值(见下方勘误)。校准结果未达 Recall@5 ≥ 90% 时同样判失败。冻结阈值后才运行 test 分片,不得根据 test 结果反复调阈值再宣称独立验收。

> **勘误(2026-09-11,用户裁决)**:本条原为「指标相同时取较高阈值」。真实评估实测:并列时取高会把阈值压在正例分数悬崖边上、对未见换说法零泛化余量(test 片 Recall@5 仅 0.70 未达标),而取低(≈对负例最大间隔)达 0.90/0.00 达标。经用户裁决改为取低;验收指标本身(Recall@5 ≥ 90% 且 FPR ≤ 10%、邮费用例必中)不变。
6. test 达标线为 **Recall@5 ≥ 90% 且负例误召回率 ≤ 10%**。固定正例「邮费是多少」放在 test 分片,必须返回包含“满99包邮/未满8元”要点的运费知识;该例不命中时即使总分达标也失败。任一门槛不达标,评估命令非零退出。
7. 将每例结果、模型名称、阈值、语料及标注文件摘要、分片和汇总指标保存至 `evals/results/`。通过后将冻结阈值同步至 Settings 和 `.env.example` 的发布默认值;当前 0.35 只用于开发,不能当成已验证终值。

**演示验收(用户亲测)**:

1. 浏览器问「邮费是多少」→ 召回运费说明并答对(与 ch02 漏召回对照)
2. 停服务后运行 `ingest_docs`,在事务或向量化期间 kill → 重跑 → SQL 查全 done、文档前后指针完整,Milvus 与 MySQL 有效主键集合一致。
3. `mine_qa` 包含至少一个零 QA 会话 → 成功完成后重跑,确认进度记录使它不再调用 LLM;清理已完成的 staging 后再次验证跳过。

## 12. 依赖与 Context7 核对清单

- Milvus 依赖显式声明 **`pymilvus[milvus-lite]`**。不能以“裸 pymilvus 必然附带 Lite”为前提: 本次核对的 PyMilvus 3.0.1 包元数据将 Lite 列为 extra,官方安装文档也要求显式安装该 extra。实现时选择兼容当前 Python/Linux 的版本组合,完成本地库创建、upsert、查询、关闭后重新打开和中断恢复验证,再锁入 `uv.lock` 并记录实际 pymilvus/milvus-lite 版本;不凭本 spec 宣称某版本已经运行验证。
- `openai` 包已由 `langchain-openai` 传递引入;异常映射使用其明确的异常类型。在线 embedding 客户端关闭自动重试,防止与 executor 重试相乘。
- 已核(Context7,2026-09-11): `MilvusClient(uri=本地路径)`、`create_collection`(含 `enable_dynamic_field=False`)、`upsert`、`search` 及 hit 的 id/distance;`OpenAIEmbeddings` 的兼容端点构造、`embed_query`、`embed_documents` 和 `check_embedding_ctx_length=False`;OpenAI SDK 的连接/超时/限流异常及 `max_retries=0`。
- 实现时仍需按最终锁定版本核对: pymilvus `has_collection`/集合 schema 查询及客户端关闭方式、LangChain 超时/重试参数的具体透传、`with_structured_output` 在现有模型端点和新会话分组 schema 下的可用性。涉及接口用法仍先查 Context7,再写实现。
- 硅基流动端点与 `BAAI/bge-m3` 的运行可用性,以及冻结阈值下的效果,由用户填 Key 后执行真实评估确认;SDK 文档核对不能替代该验收。

本轮补充核对来源:

- [Milvus Lite 安装说明](https://github.com/milvus-io/milvus-lite#installation) / [PyMilvus 包元数据](https://pypi.org/pypi/pymilvus/3.0.1/json)
- [OpenAIEmbeddings 非 OpenAI 端点配置](https://reference.langchain.com/python/langchain-openai/embeddings/base/OpenAIEmbeddings)
- [OpenAI Python SDK 错误处理与重试](https://github.com/openai/openai-python#handling-errors)
