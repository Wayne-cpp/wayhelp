# Ch03 知识库与向量语义检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 内部实现从关键词 LIKE 查表升级为 BGE-M3 + Milvus Lite 向量语义检索,含离线文档建库、对话挖知识、MySQL/Milvus 双写幂等,工具入参出参契约不变。

**Architecture:** 新增 `app/knowledge/` 包(chunking 纯函数 / embedding 封装 / milvus_store / ingest 两阶段流水线 / mining 四子阶段 / retriever 在线检索)+ 两个 CLI job;MySQL `knowledge_chunks` 为原文权威源,Milvus Lite 集合 `knowledge` 只存 id+vector;双写按「文档事务 pending → Milvus upsert → 回填 done」幂等,中断重跑补齐。

**Tech Stack:** FastAPI + SQLAlchemy + pymilvus[milvus-lite](Milvus Lite 本地文件)+ langchain-openai OpenAIEmbeddings(硅基流动 `BAAI/bge-m3`,1024 维,COSINE)+ 现有 chat model `with_structured_output`(挖矿)。

**Spec:** `docs/superpowers/specs/2026-09-11-knowledge-base-ch03-design.md`(评审修订稿;计划中的节号引用该文件)

## Global Constraints

- 依赖新增仅 `pymilvus[milvus-lite]`(显式 extra,spec §12);`openai` 由 `langchain-openai` 传递引入,不直写进 pyproject
- `embedding_api_key` 允许为空字符串,按 strip 后判断是否配置;不回退使用聊天模型 Key(spec §9)
- `query_faq` 出参严格保持 `{"results": [{"question","answer","category"}]}`,空结果带 `note`;入参仍 `keyword`(spec §8)
- 在线 embedding 客户端:`check_embedding_ctx_length=False`、`max_retries=0`(spec §2/§8)
- `db/init/03-ddl.sql` 与 `sql/ch03-ddl.sql` 逐字节一致(ch02 惯例;spec §4.5)
- Milvus 集合契约:两列 `id`(INT64 主键,auto_id=False)+ `vector`(FLOAT_VECTOR dim=1024),COSINE,`enable_dynamic_field=False`;写入一律 `upsert`(spec §4.4)
- 向量化文本 = `category + "\n" + questions + "\n" + answer`;其余字段只存不进向量(spec §4.1)
- 涉及库/API 用法先查 Context7 再动手(用户硬性要求);本计划已核:MilvusClient 构造/create_collection/upsert/search/has_collection/describe_collection、OpenAIEmbeddings 构造与 check_embedding_ctx_length
- 测试用真实基础设施(Docker MySQL + Milvus Lite tmp 文件);Docker 离线显式 `pytest.exit`,不静默 skip(ch02 惯例)
- 计划内嵌代码必须可运行;标注「脆点」处允许执行者按锁定版本实测微调,但必须在 commit message 说明
- dev-notes/ch03.md 由编排者在每任务完成后追记,执行者不碰(ch02 约定)

---

### Task 1: 基建 —— DDL 修订、依赖、配置、测试 fixture

**Files:**
- Modify: `sql/ch03-ddl.sql`(全文替换为下方修订版)
- Create: `db/init/03-ddl.sql`(与上逐字节一致)、`tests/test_ddl_sync.py`
- Modify: `pyproject.toml`、`app/config.py`、`.env.example`、`.gitignore`、`tests/dbfixtures.py`、`tests/test_config.py`

**Interfaces:**
- Produces: `Settings` 新增字段 `embedding_base_url/embedding_api_key/embedding_model/embedding_dim/milvus_uri/knowledge_top_k/knowledge_min_score/mining_batch_size/max_chunk_chars/chunk_overlap_chars` 与方法 `has_embedding_key()`;dbfixtures 的 `db_engine`/`db_session_factory` 覆盖三张新表

- [ ] **Step 1: 修订 `sql/ch03-ddl.sql`(全文替换)**

```sql
-- =============================================================
-- ch03 · RAG 基础 · 建表 DDL(2026-09-11 按 spec §4.5 修订)
-- 本章新建三张表:knowledge_chunks(知识库原文权威源)
--   qa_extraction_staging(挖 QA 暂存)/ qa_mining_progress(抽取进度)
-- 向量落 Milvus Lite 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态
-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量
-- 修订:文档块按 source_doc + chunk_index 唯一键复用 ID(uk_doc_chunk);
--   挖掘 QA 两字段为 NULL(MySQL 唯一索引允许多行 NULL);source_doc 大小写敏感(utf8mb4_bin)
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual / qa_mined,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,仅同文档相邻,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,仅同文档相邻,元数据',
  source_doc       VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT '文档来源标识:仓库内相对/仓库外绝对 POSIX 路径,大小写敏感;挖掘 QA 为 NULL',
  chunk_index      INT UNSIGNED    NULL                    COMMENT '同文档从 1 开始的连续序号;挖掘 QA 为 NULL',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_doc_chunk (source_doc, chunk_index),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话标识,形如 conv:<id>,由程序填充,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可人工清空';

CREATE TABLE qa_mining_progress (
  conversation_id BIGINT UNSIGNED NOT NULL COMMENT '成功抽取的会话 ID,与 conversations.id 对应',
  batch_no        VARCHAR(64)     NOT NULL COMMENT '成功抽取所属批次',
  qa_count        INT UNSIGNED    NOT NULL COMMENT '去重前抽出的 QA 数量,允许为 0',
  extracted_at    DATETIME        NOT NULL COMMENT '程序填入的 UTC 抽取成功时间',
  PRIMARY KEY (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话挖知识的独立抽取进度:一行=该会话抽取结果已成功提交,重跑跳过(含零 QA)';
```

约束:语句内部不得出现 `;\n`(dbfixtures `_split_statements` 按 `;\n` 切分);注释行以 `-- ` 开头(fixture 会剔除)。

- [ ] **Step 2: 复制为 `db/init/03-ddl.sql` 并写逐字节一致测试**

```bash
cp sql/ch03-ddl.sql db/init/03-ddl.sql
```

`tests/test_ddl_sync.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ch03_ddl_byte_identical():
    assert (ROOT / "sql" / "ch03-ddl.sql").read_bytes() == (
        ROOT / "db" / "init" / "03-ddl.sql"
    ).read_bytes()


def test_ch02_ddl_byte_identical():
    assert (ROOT / "sql" / "ch02-ddl.sql").read_bytes() == (
        ROOT / "db" / "init" / "01-ddl.sql"
    ).read_bytes()
```

- [ ] **Step 3: 加依赖并验证 Milvus Lite 可安装运行**

```bash
uv add 'pymilvus[milvus-lite]'
uv run python - <<'EOF'
import tempfile
from pathlib import Path
from pymilvus import MilvusClient

uri = str(Path(tempfile.mkdtemp()) / "smoke.db")
client = MilvusClient(uri=uri)
client.create_collection(collection_name="smoke", dimension=4,
                         metric_type="COSINE", auto_id=False,
                         enable_dynamic_field=False)
client.upsert("smoke", [{"id": 1, "vector": [1, 0, 0, 0]},
                        {"id": 2, "vector": [0, 1, 0, 0]}])
hits = client.search("smoke", data=[[0.9, 0.1, 0, 0]], limit=2)
assert hits[0][0]["id"] == 1, hits
client.close()
reopened = MilvusClient(uri=uri)  # 关闭后重开(spec §12 验证项)
assert reopened.has_collection("smoke")
reopened.close()
print("milvus-lite smoke OK")
EOF
```

预期输出 `milvus-lite smoke OK`。失败则按 spec §12 换兼容版本组合重试,并把实际锁定版本记入 commit message。脆点:hit 的访问方式(`hits[0][0]["id"]` 字典式)若与锁定版本不符,以实测为准调整并记录。

- [ ] **Step 4: config.py 新增配置(先写失败测试)**

`tests/test_config.py` 追加:

```python
def test_ch03_defaults():
    s = make_settings()
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.milvus_uri == "./data/milvus_lite.db"
    assert s.knowledge_top_k == 5
    assert s.knowledge_min_score == 0.35
    assert s.mining_batch_size == 10
    assert s.max_chunk_chars == 500
    assert s.chunk_overlap_chars == 80
    assert s.has_embedding_key() is False  # 默认空 key


def test_ch03_overlap_must_be_less_than_chunk():
    with pytest.raises(ValidationError):
        make_settings(max_chunk_chars=80, chunk_overlap_chars=80)


def test_ch03_min_score_range():
    with pytest.raises(ValidationError):
        make_settings(knowledge_min_score=1.5)


def test_ch03_dim_fixed_1024():
    with pytest.raises(ValidationError):
        make_settings(embedding_dim=768)


def test_has_embedding_key_strips_whitespace():
    assert make_settings(embedding_api_key="  sk-x  ").has_embedding_key() is True
    assert make_settings(embedding_api_key="   ").has_embedding_key() is False
```

`app/config.py` 追加(`Literal` 已导入;需补 `model_validator` import):

```python
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: Literal[1024] = 1024
    milvus_uri: str = "./data/milvus_lite.db"
    knowledge_top_k: int = Field(default=5, gt=0)
    knowledge_min_score: float = Field(default=0.35, ge=-1, le=1)
    mining_batch_size: int = Field(default=10, gt=0)
    max_chunk_chars: int = Field(default=500, gt=0)
    chunk_overlap_chars: int = Field(default=80, ge=0)

    @model_validator(mode="after")
    def _overlap_less_than_chunk(self):
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError("chunk_overlap_chars 必须小于 max_chunk_chars")
        return self

    def has_embedding_key(self) -> bool:
        return bool(self.embedding_api_key.strip())
```

运行:`uv run pytest tests/test_config.py -v`,新测试全绿。

- [ ] **Step 5: .env.example 与 .gitignore**

`.env.example` 追加:

```
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
MILVUS_URI=./data/milvus_lite.db
KNOWLEDGE_TOP_K=5
KNOWLEDGE_MIN_SCORE=0.35
MINING_BATCH_SIZE=10
MAX_CHUNK_CHARS=500
CHUNK_OVERLAP_CHARS=80
```

`.gitignore` 追加一行:`data/`

- [ ] **Step 6: dbfixtures 扩展三表 + 全量回归**

`tests/dbfixtures.py` 修改:

```python
DDL_PATHS = [
    Path(__file__).resolve().parent.parent / "db" / "init" / "01-ddl.sql",
    Path(__file__).resolve().parent.parent / "db" / "init" / "03-ddl.sql",
]
TABLES = ("knowledge_chunks", "qa_extraction_staging", "qa_mining_progress",
          "messages", "tickets", "faq", "conversations")  # 先子后父
```

`db_engine` fixture 中重放 DDL 处改为:

```python
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table in TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        for ddl_path in DDL_PATHS:
            for stmt in _split_statements(ddl_path.read_text(encoding="utf-8")):
                conn.execute(text(stmt))
        conn.commit()
```

运行全量:`uv run pytest -x -q`。预期:既有 136 个测试全绿 + 新配置/DDL 测试绿。

- [ ] **Step 7: Commit**

```bash
git add sql/ch03-ddl.sql db/init/03-ddl.sql pyproject.toml uv.lock app/config.py \
        .env.example .gitignore tests/dbfixtures.py tests/test_config.py tests/test_ddl_sync.py
git commit -m "feat(ch03): DDL 修订(来源字段+唯一键+进度表) + pymilvus[milvus-lite] 依赖 + 知识库配置项"
```

---

### Task 2: ORM 模型 —— KnowledgeChunk / QaExtractionStaging / QaMiningProgress

**Files:**
- Modify: `app/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: dbfixtures(T1)
- Produces: `KnowledgeChunk`(字段名与修订后 DDL 一致)、`QaExtractionStaging`、`QaMiningProgress`,供 T5/T6/T7 使用

- [ ] **Step 1: 写失败测试**

`tests/test_models.py` 追加:

```python
def test_knowledge_tables_crud(db_session_factory):
    from app.models import KnowledgeChunk, QaExtractionStaging, QaMiningProgress
    with db_session_factory() as s:
        c1 = KnowledgeChunk(category="售后政策", questions="退货", answer="七天无理由。",
                            section_path="售后政策 > 退货", content_type="policy",
                            is_key_clause=True, source_doc="knowledge_docs/退货政策.md",
                            chunk_index=1, vectorize_status="pending")
        s.add(c1)
        s.flush()  # 先拿 c1.id
        c2 = KnowledgeChunk(category="售后政策", questions="换货", answer="十五天。",
                            section_path="售后政策 > 换货", content_type="policy",
                            is_key_clause=False, prev_chunk_id=c1.id,
                            source_doc="knowledge_docs/退货政策.md", chunk_index=2)
        s.add(c2)
        s.flush()  # 再拿 c2.id 回填 c1.next
        c1.next_chunk_id = c2.id
        s.add(QaExtractionStaging(batch_no="b1", source_ref="conv:1",
                                  question="q", answer="a"))
        s.add(QaMiningProgress(conversation_id=1, batch_no="b1", qa_count=1,
                               extracted_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        s.commit()
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index).all()
        assert [r.chunk_index for r in rows] == [1, 2]
        assert rows[0].next_chunk_id == rows[1].id
        assert rows[1].prev_chunk_id == rows[0].id
        assert rows[0].vectorize_status == "pending"
        assert s.query(QaMiningProgress).count() == 1


def test_source_doc_chunk_index_unique(db_session_factory):
    from app.models import KnowledgeChunk
    from sqlalchemy.exc import IntegrityError
    with db_session_factory() as s:
        s.add(KnowledgeChunk(category="c", questions="q", answer="a",
                             source_doc="d.md", chunk_index=1))
        s.commit()
    with db_session_factory() as s:
        s.add(KnowledgeChunk(category="c2", questions="q2", answer="a2",
                             source_doc="d.md", chunk_index=1))
        with pytest.raises(IntegrityError):
            s.commit()


def test_mined_qa_allows_multiple_null_source(db_session_factory):
    from app.models import KnowledgeChunk
    with db_session_factory() as s:
        for i in range(2):
            s.add(KnowledgeChunk(category="对话挖掘", questions=f"q{i}", answer="a",
                                 content_type="qa_mined", source_doc=None, chunk_index=None))
        s.commit()
        assert s.query(KnowledgeChunk).count() == 2
```

(`test_models.py` 需在文件头补 `from datetime import datetime, timezone`。)

- [ ] **Step 2: 跑测试确认失败(模型未定义)**

- [ ] **Step 3: 实现模型**

`app/models.py` 追加(import 处补 `Boolean`、`INT`、`UniqueConstraint`):

```python
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (UniqueConstraint("source_doc", "chunk_index", name="uk_doc_chunk"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id"), nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id"), nullable=True)
    source_doc: Mapped[str | None] = mapped_column(
        String(255, collation="utf8mb4_bin"), nullable=True)
    chunk_index: Mapped[int | None] = mapped_column(INT(unsigned=True), nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vec_status"), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_status"), default="extracted")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class QaMiningProgress(Base):
    __tablename__ = "qa_mining_progress"

    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    qa_count: Mapped[int] = mapped_column(INT(unsigned=True))
    extracted_at: Mapped[datetime] = mapped_column(DateTime)
```

- [ ] **Step 4: 跑测试确认通过,再跑全量**

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat(ch03): knowledge_chunks/qa_extraction_staging/qa_mining_progress ORM 模型"
```

---

### Task 3: chunking.py —— Markdown 结构感知切分(纯函数)

**Files:**
- Create: `app/knowledge/__init__.py`(空)、`app/knowledge/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces(后续任务依赖的精确签名):
  - `Chunk` dataclass(frozen): `category: str, questions: str, answer: str, section_path: str | None, content_type: str, is_key_clause: bool`
  - `ChunkingError(ValueError)`: 属性 `source: str`、`line_no: int | None`
  - `chunk_document(text: str, *, source: str, max_chars: int, overlap_chars: int) -> list[Chunk]` —— frontmatter 解析 `type`;`chunk_index`/`source_doc` 由 T5 编排层赋

**实现规格(spec §5 落地,全部规则写死)**:

- frontmatter: 首行 `---`,到下一个 `---` 闭合,必填 `type: faq|policy|manual`;缺失/未闭合/未知 type 抛 `ChunkingError`
- 标题栈:`#`/`##`/`###` 维护栈;faq 文档只有 `##` 产生切分单元(`###` 留在正文),policy/manual 三级都切
- 文档必须有 H1(第一个一级标题 = 文档标题);前言(H1 前或与标题夹着无归属正文)并入第一个可产出知识块的 section;仅有标题无正文的 section 不成块;全文无可入库正文报错
- 三格: faq → `questions`=## 标题、`category`=H1;policy/manual → `questions`=末级标题、`category`=栈去末级 ` > ` 连接(仅一级则为 H1)
- `section_path` = 栈 ` > ` 连接
- 长度: 最终 `answer` 含重叠/换行/表头 ≤ `max_chars`(先统一 `\n`)
- 超长递归: 段落打包 → 单段超限按 `。！？!?` 切句 → 单句仍超限按字符硬切;不产生空块
- 重叠: 仅同 section 内相邻**普通文本块**;取上一块末尾 ≤ `overlap_chars` 的完整句后缀(块必须以句末标点结尾,硬切尾不重叠);打包新块时预留 `overlap+1` 预算(section 首块除外),重叠装不进就缩短/取消
- 表格: 连续 `|` 行成一个表格;表头+分隔行复制进每块且计入长度;表头+分隔超限、或加任意单行超限 → `ChunkingError` 带源行号;不截断单元格
- `is_key_clause`: answer 命中 `("不支持","不予","必须","扣除","逾期","无效")`

- [ ] **Step 1: 写失败测试 `tests/test_chunking.py`(全文)**

```python
import pytest

from app.knowledge.chunking import ChunkingError, chunk_document

POLICY_DOC = """---
type: policy
---
# 售后政策

## 退货

七天无理由退货。定制类商品不支持退货。

## 换货

十五天内可申请换货。
"""

FAQ_DOC = """---
type: faq
---
# 商品FAQ

## 运费怎么算

单笔订单实付满 99 元包邮,未满收 8 元基础运费。

## 什么时候发货

工作日 16 点前当天发。
"""


def test_policy_headings_split_and_fields():
    chunks = chunk_document(POLICY_DOC, source="d.md", max_chars=500, overlap_chars=80)
    assert len(chunks) == 2
    c0, c1 = chunks
    assert c0.questions == "退货" and c0.category == "售后政策"
    assert c0.section_path == "售后政策 > 退货"
    assert "七天无理由" in c0.answer
    assert c0.is_key_clause is True  # 命中「不支持」
    assert c1.questions == "换货" and c1.is_key_clause is False
    assert all(c.content_type == "policy" for c in chunks)


def test_faq_questions_from_real_heading():
    chunks = chunk_document(FAQ_DOC, source="f.md", max_chars=500, overlap_chars=80)
    assert [c.questions for c in chunks] == ["运费怎么算", "什么时候发货"]
    assert all(c.category == "商品FAQ" for c in chunks)
    assert all(c.content_type == "faq" for c in chunks)


def test_frontmatter_missing_or_bad():
    with pytest.raises(ChunkingError):
        chunk_document("# 无头\n\n正文。", source="x.md", max_chars=500, overlap_chars=80)
    with pytest.raises(ChunkingError):
        chunk_document("---\ntype: unknown\n---\n# t\n\n正文。",
                       source="x.md", max_chars=500, overlap_chars=80)


def test_missing_h1_and_empty_body():
    with pytest.raises(ChunkingError):
        chunk_document("---\ntype: policy\n---\n## 只有二级\n\n正文。",
                       source="x.md", max_chars=500, overlap_chars=80)
    with pytest.raises(ChunkingError):
        chunk_document("---\ntype: policy\n---\n# 只有标题\n",
                       source="x.md", max_chars=500, overlap_chars=80)


def test_heading_only_section_produces_no_chunk():
    doc = "---\ntype: policy\n---\n# T\n\n## 空节\n\n## 有内容\n\n正文一句。"
    chunks = chunk_document(doc, source="x.md", max_chars=500, overlap_chars=80)
    assert len(chunks) == 1 and chunks[0].questions == "有内容"


def test_long_paragraph_splits_by_sentence_with_overlap():
    body = "第一句很长啊。第二句也不短呢。第三句更长了。"
    doc = f"---\ntype: policy\n---\n# T\n\n## S\n\n{body}"
    chunks = chunk_document(doc, source="x.md", max_chars=20, overlap_chars=10)
    assert len(chunks) == 2
    assert chunks[0].answer == "第一句很长啊。第二句也不短呢。"
    # 重叠 = 上一块末尾 ≤10 的完整句后缀「第二句也不短呢。」
    assert chunks[1].answer == "第二句也不短呢。\n第三句更长了。"
    assert all(len(c.answer) <= 20 for c in chunks)


def test_no_overlap_across_sections():
    doc = ("---\ntype: policy\n---\n# T\n\n## A\n\n甲句结尾在此。"
           "\n\n## B\n\n乙句开头在此。")
    chunks = chunk_document(doc, source="x.md", max_chars=500, overlap_chars=10)
    assert chunks[1].answer == "乙句开头在此。"  # 不跨 section 重叠


def test_hard_cut_when_no_sentence_punctuation():
    body = "无标点" * 30  # 90 字无句末标点
    doc = f"---\ntype: policy\n---\n# T\n\n## S\n\n{body}"
    chunks = chunk_document(doc, source="x.md", max_chars=40, overlap_chars=10)
    assert len(chunks) == 3  # 90/40 硬切
    assert all(len(c.answer) <= 40 for c in chunks)
    assert chunks[1].answer.startswith("无标点")  # 硬切尾不当作重叠内容
    assert chunks[1].answer == body[40:80]


def test_table_header_copied_into_every_piece():
    table = ("| 项目 | 标准 |\n|---|---|\n| 退货时效 | 签收后7天 |\n"
             "| 换货时效 | 签收后15天 |\n| 运费险 | 支持首重 |")
    doc = f"---\ntype: manual\n---\n# 手册\n\n## 时效\n\n{table}"
    chunks = chunk_document(doc, source="m.md", max_chars=40, overlap_chars=10)
    assert len(chunks) == 3
    header = "| 项目 | 标准 |\n|---|---|"
    for c in chunks:
        assert c.answer.startswith(header)
        assert len(c.answer) <= 40
    data_rows = "".join(c.answer for c in chunks)
    assert "退货时效" in data_rows and "换货时效" in data_rows and "运费险" in data_rows


def test_table_row_too_long_errors_with_line_no():
    table = "| 项目 | 标准 |\n|---|---|\n| 超长 | " + "x" * 100 + " |"
    doc = f"---\ntype: manual\n---\n# 手册\n\n## 时效\n\n{table}"
    with pytest.raises(ChunkingError) as exc_info:
        chunk_document(doc, source="m.md", max_chars=40, overlap_chars=10)
    assert exc_info.value.line_no is not None


def test_faq_long_answer_chunks_share_question():
    answer = "第一点说明。第二点说明。第三点说明。"
    doc = f"---\ntype: faq\n---\n# 商品FAQ\n\n## 保修多久\n\n{answer}"
    chunks = chunk_document(doc, source="f.md", max_chars=15, overlap_chars=6)
    assert len(chunks) >= 2
    assert all(c.questions == "保修多久" for c in chunks)


def test_subheading_stays_in_faq_answer():
    doc = ("---\ntype: faq\n---\n# 商品FAQ\n\n## 退货流程\n\n### 第一步\n\n提交申请。"
           "\n\n### 第二步\n\n寄回商品。")
    chunks = chunk_document(doc, source="f.md", max_chars=500, overlap_chars=80)
    assert len(chunks) == 1
    assert "### 第一步" in chunks[0].answer and "### 第二步" in chunks[0].answer
```

- [ ] **Step 2: 跑测试确认全部失败(模块不存在)**

- [ ] **Step 3: 实现 `app/knowledge/chunking.py`(全文)**

```python
"""Markdown 结构感知切分(spec §5)。纯函数,无 IO,单测主力。"""

from dataclasses import dataclass, field

import re


class ChunkingError(ValueError):
    def __init__(self, message: str, source: str, line_no: int | None = None):
        self.source = source
        self.line_no = line_no
        where = f"{source}:{line_no}" if line_no is not None else source
        super().__init__(f"{where}: {message}")


@dataclass(frozen=True)
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool


_KEY_CLAUSE_WORDS = ("不支持", "不予", "必须", "扣除", "逾期", "无效")
_SENTENCE_END = "。！？!?"
_HEADING_RE = re.compile(r"^(#{1,3})\s+(\S.*?)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])")


def _parse_frontmatter(text: str, source: str) -> tuple[str, str]:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ChunkingError("缺少 frontmatter(首行须为 ---)", source)
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ChunkingError("frontmatter 未闭合", source, 1)
    meta = {}
    for ln in lines[1:end]:
        if ":" in ln:
            k, _, v = ln.partition(":")
            meta[k.strip()] = v.strip()
    ctype = meta.get("type", "")
    if ctype not in ("faq", "policy", "manual"):
        raise ChunkingError(f"frontmatter type 缺失或未知: {ctype!r}", source, 1)
    return ctype, "\n".join(lines[end + 1:])


@dataclass
class _Block:
    kind: str      # "para" | "table"
    text: str
    line_no: int   # 首行在传入 text 中的行号(1 起)


@dataclass
class _Section:
    stack: list[str]
    blocks: list[_Block] = field(default_factory=list)


def _parse_sections(body: str, source: str, split_levels: set[int]) -> tuple[str, list[_Section]]:
    """逐行解析。返回 (H1 文档标题, sections)。不在 split_levels 的标题行当正文。"""
    title: str | None = None
    sections: list[_Section] = []
    preamble: list[_Block] = []
    stack: list[str] = []
    cur: _Section | None = None
    para_lines: list[str] = []
    para_start = 0
    table_lines: list[str] = []
    table_start = 0

    def flush_para():
        nonlocal para_lines
        text = "\n".join(para_lines).strip() if para_lines else ""
        if text:
            (cur.blocks if cur else preamble).append(_Block("para", text, para_start))
        para_lines = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            (cur.blocks if cur else preamble).append(
                _Block("table", "\n".join(table_lines), table_start))
            table_lines = []

    for idx, line in enumerate(body.split("\n"), start=1):
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) in split_levels:
            flush_para()
            flush_table()
            level, heading = len(m.group(1)), m.group(2)
            stack = stack[: level - 1] + [heading]
            if level == 1 and title is None:
                title = heading
            cur = _Section(stack=list(stack))
            sections.append(cur)
        elif line.strip().startswith("|"):
            flush_para()
            if not table_lines:
                table_start = idx
            table_lines.append(line.rstrip())
        elif not line.strip():
            flush_para()
            flush_table()
        else:
            flush_table()
            if not para_lines:
                para_start = idx
            para_lines.append(line.rstrip())
    flush_para()
    flush_table()
    if title is None:
        raise ChunkingError("缺少 H1 文档标题", source, 1)
    # 前言并入第一个可产出知识块的 section
    if preamble:
        for sec in sections:
            if any(b.text.strip() for b in sec.blocks):
                sec.blocks = preamble + sec.blocks
                break
        else:
            sections.insert(0, _Section(stack=[title], blocks=preamble))
    return title, sections


def _split_sentences(text: str) -> list[str]:
    return [p for p in _SENTENCE_SPLIT_RE.split(text) if p]


def _hard_cut(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _para_units(text: str, max_chars: int) -> list[str]:
    """段落 → 可装入单元:整段 ≤ max 则整段;否则按句;单句超限硬切。"""
    if len(text) <= max_chars:
        return [text]
    units: list[str] = []
    for sent in _split_sentences(text):
        if len(sent) <= max_chars:
            units.append(sent)
        else:
            units.extend(_hard_cut(sent, max_chars))
    if not units:  # 无任何句末标点:整段视为一句
        units = _hard_cut(text, max_chars)
    return [u for u in units if u]


def _table_chunks(text: str, max_chars: int, source: str, base_line_no: int) -> list[str]:
    lines = text.split("\n")
    if len(lines) < 2:
        raise ChunkingError("表格缺少表头/分隔行", source, base_line_no)
    header = "\n".join(lines[:2])
    if len(header) > max_chars:
        raise ChunkingError("表头+分隔行本身超限,请缩短表头", source, base_line_no)
    chunks: list[str] = []
    cur = header
    for i, row in enumerate(lines[2:], start=3):
        if len(header) + 1 + len(row) > max_chars:
            raise ChunkingError("表头+单个数据行即超限,请缩短该行或拆分源表格",
                                source, base_line_no + i - 1)
        if len(cur) + 1 + len(row) > max_chars:
            chunks.append(cur)
            cur = header
        cur = cur + "\n" + row
    chunks.append(cur)
    return chunks


def _sentence_suffix(text: str, budget: int) -> str:
    """末尾 ≤ budget 的完整句后缀;块尾不是句末标点(硬切尾)则无重叠。"""
    if budget <= 0 or not text or text[-1] not in _SENTENCE_END:
        return ""
    best = ""
    for piece in reversed(_split_sentences(text)):  # 从末尾往前累积完整句
        candidate = piece + best
        if len(candidate) <= budget:
            best = candidate
        else:
            break
    return best


def _emit_answers(blocks: list[_Block], max_chars: int, overlap: int,
                  source: str) -> list[str]:
    """blocks → 有序 answer 列表;文本块打包+完整句重叠,表格块独立成块。"""
    answers: list[tuple[str, str]] = []  # (kind, text)
    cur_text = ""
    cur_capacity = max_chars  # section 首块不预留重叠预算

    def close_text():
        nonlocal cur_text, cur_capacity
        if cur_text:
            answers.append(("text", cur_text))
            cur_text = ""
            cur_capacity = max_chars - (overlap + 1 if overlap > 0 else 0)

    for blk in blocks:
        if blk.kind == "table":
            close_text()
            for piece in _table_chunks(blk.text, max_chars, source, blk.line_no):
                answers.append(("table", piece))
            cur_capacity = max_chars - (overlap + 1 if overlap > 0 else 0)
            continue
        for unit in _para_units(blk.text, max_chars):
            extra = len(unit) if not cur_text else 1 + len(unit)
            if cur_text and len(cur_text) + extra > cur_capacity:
                close_text()
                extra = len(unit)
            cur_text = unit if not cur_text else cur_text + "\n" + unit
    close_text()

    out: list[str] = []
    for i, (kind, text) in enumerate(answers):
        if kind == "text" and i > 0 and answers[i - 1][0] == "text":
            suffix = _sentence_suffix(out[-1], overlap)
            # 与下一完整句冲突(如硬切块满载)时取消重叠,保持 answer 不超限
            if suffix and len(suffix) + 1 + len(text) <= max_chars:
                text = suffix + "\n" + text
        out.append(text)
    return [t for t in out if t.strip()]


def chunk_document(text: str, *, source: str, max_chars: int,
                   overlap_chars: int) -> list[Chunk]:
    text = text.replace("\r\n", "\n")
    ctype, body = _parse_frontmatter(text, source)
    # faq: H1 仅作文档标题、## 产块、### 留正文;policy/manual: 三级都切
    split_levels = {1, 2} if ctype == "faq" else {1, 2, 3}
    title, sections = _parse_sections(body, source, split_levels)
    chunks: list[Chunk] = []
    pending_prefix: list[_Block] = []  # faq 下 H1 直挂正文,并入下一问答块
    for sec in sections:
        blocks = [b for b in sec.blocks if b.text.strip()]
        if ctype == "faq" and len(sec.stack) < 2:
            pending_prefix.extend(blocks)
            continue
        if pending_prefix:
            blocks = pending_prefix + blocks
            pending_prefix = []
        if not blocks:
            continue
        questions = sec.stack[-1]
        if ctype == "faq" or len(sec.stack) == 1:
            category = title
        else:
            category = " > ".join(sec.stack[:-1])
        for answer in _emit_answers(blocks, max_chars, overlap_chars, source):
            chunks.append(Chunk(
                category=category, questions=questions, answer=answer,
                section_path=" > ".join(sec.stack), content_type=ctype,
                is_key_clause=any(w in answer for w in _KEY_CLAUSE_WORDS),
            ))
    if not chunks:
        raise ChunkingError("整个文档无可入库正文", source)
    return chunks
```

备注:`_Block.line_no` 是相对 frontmatter 之后正文的行号;CLI 展示报错时可加 frontmatter 行数偏移,测试只断言非 None。

- [ ] **Step 4: 跑 `uv run pytest tests/test_chunking.py -v` 全绿**

- [ ] **Step 5: Commit**

```bash
git add app/knowledge/__init__.py app/knowledge/chunking.py tests/test_chunking.py
git commit -m "feat(ch03): Markdown 结构感知切分纯函数(层级/递归/完整句重叠/表格表头复制)"
```

---

### Task 4: embedding.py + milvus_store.py —— 嵌入封装与向量集合管理

**Files:**
- Create: `app/knowledge/embedding.py`、`app/knowledge/milvus_store.py`
- Test: `tests/test_milvus_store.py`(真实 Milvus Lite tmp 文件)、`tests/test_embedding.py`(构造契约,不调真实 API)

**Interfaces:**
- Produces:
  - `embedding.build_embeddings(settings: Settings) -> OpenAIEmbeddings`(key 空白抛 `ValueError`)
  - `milvus_store.COLLECTION = "knowledge"`
  - `MilvusKnowledgeStore(uri: str, dim: int)`,方法: `file_exists() -> bool`、`has_collection() -> bool`、`ensure_collection() -> None`(不存在则建,存在则校验契约)、`upsert(rows: list[tuple[int, list[float]]]) -> None`、`search(vector: list[float], top_k: int) -> list[tuple[int, float]]`、`all_ids() -> set[int]`、`close() -> None`

- [ ] **Step 1: 失败测试 `tests/test_milvus_store.py`**

```python
import pytest

from app.knowledge.milvus_store import COLLECTION, MilvusKnowledgeStore


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "test_milvus.db"), dim=4)
    yield s
    s.close()


def test_file_not_exists_initially(store):
    assert store.file_exists() is False


def test_ensure_creates_and_contract_ok(store):
    store.ensure_collection()
    assert store.has_collection() is True
    store.ensure_collection()  # 幂等,重复调用不炸


def test_upsert_search_roundtrip(store):
    store.ensure_collection()
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])])
    hits = store.search([0.9, 0.1, 0.0, 0.0], top_k=2)
    assert hits[0][0] == 1  # 最相似的是 id=1
    assert 0.0 < hits[0][1] <= 1.0
    assert {h[0] for h in hits} == {1, 2}


def test_upsert_same_id_replaces(store):
    store.ensure_collection()
    store.upsert([(1, [1.0, 0.0, 0.0, 0.0])])
    store.upsert([(1, [0.0, 1.0, 0.0, 0.0])])  # 同 id 覆盖,不出重复向量
    assert store.all_ids() == {1}
    hits = store.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert hits[0][0] == 1 and hits[0][1] > 0.9


def test_all_ids_empty_collection(store):
    assert store.all_ids() == set()
    store.ensure_collection()
    assert store.all_ids() == set()


def test_contract_mismatch_dim_rejected(store):
    store.ensure_collection()  # dim=4
    wrong = MilvusKnowledgeStore(store._uri, dim=8)
    with pytest.raises(ValueError, match="维度|dim"):
        wrong.ensure_collection()
    wrong.close()


def test_close_and_reopen(tmp_path):
    uri = str(tmp_path / "reopen.db")
    s1 = MilvusKnowledgeStore(uri, dim=4)
    s1.ensure_collection()
    s1.upsert([(7, [1.0, 0.0, 0.0, 0.0])])
    s1.close()
    s2 = MilvusKnowledgeStore(uri, dim=4)
    assert s2.all_ids() == {7}
    s2.close()
```

- [ ] **Step 2: 失败测试 `tests/test_embedding.py`**

```python
import pytest

from app.knowledge.embedding import build_embeddings
from tests.conftest import make_settings


def test_build_requires_key():
    with pytest.raises(ValueError, match="embedding_api_key"):
        build_embeddings(make_settings(embedding_api_key="  "))


def test_build_params():
    emb = build_embeddings(make_settings(embedding_api_key="sk-test"))
    assert emb.model == "BAAI/bge-m3"
    assert emb.check_embedding_ctx_length is False  # 第三方端点必须关 tiktoken
    assert emb.max_retries == 0                      # 重试由 executor 统一管理
    assert emb.openai_api_base == "https://api.siliconflow.cn/v1"
```

脆点:`OpenAIEmbeddings` 实例的属性名(`openai_api_base` vs `base_url`)随版本变;断言失败时以锁定版本的实际属性名为准调整测试(构造参数名 `model/api_key/base_url/check_embedding_ctx_length/max_retries` 是公开契约,不可调)。

- [ ] **Step 3: 实现 `app/knowledge/embedding.py`**

```python
"""嵌入客户端封装:OpenAIEmbeddings 指硅基流动 BGE-M3(spec §2/§9)。"""

from langchain_openai import OpenAIEmbeddings

from app.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    key = settings.embedding_api_key.strip()
    if not key:
        raise ValueError("embedding_api_key 未配置")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=key,
        base_url=settings.embedding_base_url,
        check_embedding_ctx_length=False,  # 第三方模型 tiktoken 不认识,发原始文本
        max_retries=0,                     # 重试统一由 executor/调用方管理,防叠乘
        request_timeout=settings.tool_timeout_seconds,
    )
```

脆点:超时参数名 `request_timeout`(旧别名 `timeout`);若锁定版本改名,以 Context7/源码为准改这一个参数名并在 commit message 说明。

- [ ] **Step 4: 实现 `app/knowledge/milvus_store.py`**

```python
"""Milvus Lite 集合管理(spec §4.4)。只存 id + vector 两列,原文在 MySQL。"""

from pathlib import Path

from pymilvus import MilvusClient

COLLECTION = "knowledge"


class MilvusKnowledgeStore:
    def __init__(self, uri: str, dim: int):
        self._uri = uri
        self._dim = dim
        self._client: MilvusClient | None = None

    def _cli(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri)
        return self._client

    def file_exists(self) -> bool:
        return Path(self._uri).exists()

    def has_collection(self) -> bool:
        return self._cli().has_collection(COLLECTION)

    def ensure_collection(self) -> None:
        """不存在则按契约创建;已存在则校验主键/维度,不符报错(spec §6 不得自动重建)。"""
        cli = self._cli()
        if not cli.has_collection(COLLECTION):
            Path(self._uri).parent.mkdir(parents=True, exist_ok=True)
            cli.create_collection(collection_name=COLLECTION, dimension=self._dim,
                                  metric_type="COSINE", auto_id=False,
                                  enable_dynamic_field=False)
            return
        info = cli.describe_collection(COLLECTION)
        fields = {f["name"]: f for f in info["fields"]}
        pk = fields.get("id") or {}
        if not pk.get("is_primary"):
            raise ValueError("knowledge 集合契约不符: id 不是主键")
        vec = fields.get("vector") or {}
        actual_dim = (vec.get("params") or {}).get("dim", vec.get("dimension"))
        if actual_dim != self._dim:
            raise ValueError(f"knowledge 集合维度不符: 期望 {self._dim},实际 {actual_dim}")

    def upsert(self, rows: list[tuple[int, list[float]]]) -> None:
        if not rows:
            return
        self._cli().upsert(COLLECTION, [{"id": i, "vector": v} for i, v in rows])

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """返回 [(chunk_id, cosine_similarity)],按相似度降序。"""
        res = self._cli().search(COLLECTION, data=[vector], limit=top_k)
        return [(h["id"], h["distance"]) for h in res[0]]

    def all_ids(self) -> set[int]:
        if not self.has_collection():
            return set()
        rows = self._cli().query(COLLECTION, filter="id >= 0", output_fields=["id"])
        return {r["id"] for r in rows}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
```

脆点(以锁定版本实测为准,测试会钉住行为):① hit 字典式访问 `h["id"]`/`h["distance"]` 若不支持,改 `h.id`/`h.distance`;② `describe_collection` 返回的 fields 结构(list of dict,`params.dim`)若不符,以实测调整取值路径;③ `query(filter="id >= 0")` 在 Milvus Lite 的可用性,若空 filter 语法有差异,改 `filter=""`。

- [ ] **Step 5: 跑测试**

```bash
uv run pytest tests/test_milvus_store.py tests/test_embedding.py -v
```

全绿。若 Milvus Lite 行为与脆点假设不符,按实测修代码(不是修测试断言方向),commit message 记录。

- [ ] **Step 6: Commit**

```bash
git add app/knowledge/embedding.py app/knowledge/milvus_store.py \
        tests/test_embedding.py tests/test_milvus_store.py
git commit -m "feat(ch03): OpenAIEmbeddings 封装(硅基流动 BGE-M3) + Milvus Lite 集合管理"
```

---

### Task 5: ingest.py + ingest_docs CLI —— 双写两阶段流水线

**Files:**
- Create: `app/knowledge/ingest.py`、`app/jobs/__init__.py`(空)、`app/jobs/ingest_docs.py`
- Test: `tests/test_ingest.py`(真实 Docker MySQL + 真实 Milvus Lite tmp + fake embedding)

**Interfaces:**
- Consumes: `chunk_document`(T3)、`build_embeddings`/`MilvusKnowledgeStore`(T4)、`KnowledgeChunk`(T2)、dbfixtures(T1)
- Produces(T6/T9 依赖):
  - `vector_text(category: str, questions: str, answer: str) -> str` —— 三格 `"\n"` 拼接
  - `vectorize_pending(settings, session_factory, embed, store) -> None` —— Phase 2,挖矿复用
  - `resolve_source_doc(path: Path) -> str` —— 仓库内相对 POSIX/仓库外绝对 POSIX,>255 报错
  - `run_ingest(settings, session_factory, embed, store, docs_dir: Path) -> int` —— 0 成功 / 1 失败
  - `IngestError(Exception)`

**流程(spec §6)**: 校验集合契约 → Phase 2 resume(捡历史 pending)→ 逐文档 Phase 1(文档事务)→ Phase 2 → 终验(无 pending、指针完整、两库主键集合一致)。

- [ ] **Step 1: 失败测试 `tests/test_ingest.py`**

```python
import pytest

from app.knowledge.ingest import IngestError, resolve_source_doc, run_ingest, vectorize_pending
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import KnowledgeChunk
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

DOC_A = """---
type: faq
---
# 商品FAQ

## 运费怎么算

满 99 包邮,未满 8 元。

## 发货时间

16 点前当天发。
"""

DOC_B = """---
type: policy
---
# 售后政策

## 退货

七天无理由。
"""


class FakeEmbeddings:
    """确定性假向量:维度 4,内容按文本 hash 可区分。"""
    dim = 4

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    def _v(self, text):
        h = abs(hash(text)) % 100
        return [h / 100.0, (100 - h) / 100.0, 0.5, 0.5]


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "ingest.db"), dim=4)
    yield s
    s.close()


@pytest.fixture()
def settings():
    return make_settings()
```

设计决策(spec §6 的「维度校验」落地口径):`vectorize_pending` 校验 `len(vector) == store 的 dim`(集合维度),配置 `embedding_dim=1024` 用于建集合与在线查询向量校验(T7);离线测试用 dim=4 的 store 绕开真实 1024,真实维度链路由 T9 真实评估覆盖。

```python
def _write_docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text(DOC_A, encoding="utf-8")
    (d / "b.md").write_text(DOC_B, encoding="utf-8")
    return d


def test_ingest_full_pipeline(db_session_factory, store, settings, tmp_path):
    docs = _write_docs(tmp_path)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        assert len(rows) == 3
        assert all(r.vectorize_status == "done" for r in rows)
        assert all(r.vector_id == str(r.id) for r in rows)
        faq_rows = [r for r in rows if r.source_doc.endswith("a.md")]
        assert len(faq_rows) == 2  # 运费 + 发货
        by_idx = {r.chunk_index: r for r in faq_rows}
        assert by_idx[1].next_chunk_id == by_idx[2].id
        assert by_idx[2].prev_chunk_id == by_idx[1].id
        assert by_idx[1].prev_chunk_id is None and by_idx[2].next_chunk_id is None
    assert store.all_ids() == {r.id for r in rows}  # 两库主键集合一致


def test_ingest_rerun_reuses_ids(db_session_factory, store, settings, tmp_path):
    docs = _write_docs(tmp_path)
    run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs)
    with db_session_factory() as s:
        first_ids = [r.id for r in s.query(KnowledgeChunk).order_by(KnowledgeChunk.id)]
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        assert [r.id for r in rows] == first_ids  # 原样重跑 ID 不变、不新增行


def test_ingest_changed_document_rejected(db_session_factory, store, settings, tmp_path):
    docs = _write_docs(tmp_path)
    run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs)
    (docs / "a.md").write_text(DOC_A + "\n## 新问答\n\n新内容。\n", encoding="utf-8")
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 1
    with db_session_factory() as s:  # 旧内容不被覆盖
        assert s.query(KnowledgeChunk).count() == 3


def test_same_content_different_docs_both_stored(db_session_factory, store, settings, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "x.md").write_text(DOC_B, encoding="utf-8")
    (d / "y.md").write_text(DOC_B, encoding="utf-8")
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, d) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 2  # 同内容不同文档分别落行


def test_interrupted_vectorize_resumes(db_session_factory, store, settings, tmp_path, monkeypatch):
    docs = _write_docs(tmp_path)
    real_upsert = store.upsert
    state = {"calls": 0}

    def boom_once(rows):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("simulated crash")
        return real_upsert(rows)

    monkeypatch.setattr(store, "upsert", boom_once)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 1
    with db_session_factory() as s:  # 第一次:全部落 MySQL,向量中断
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 3
    monkeypatch.setattr(store, "upsert", real_upsert)
    assert run_ingest(settings, db_session_factory, FakeEmbeddings(), store, docs) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 0
        ids = {r.id for r in s.query(KnowledgeChunk).all()}
    assert store.all_ids() == ids  # 重跑补齐,主键集合一致


def test_dimension_mismatch_aborts(db_session_factory, store, settings, tmp_path):
    class BadDim(FakeEmbeddings):
        def _v(self, text):
            return [0.1, 0.2]  # 维度 2 ≠ 4
    docs = _write_docs(tmp_path)
    assert run_ingest(settings, db_session_factory, BadDim(), store, docs) == 1
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).filter_by(vectorize_status="done").count() == 0


def test_resolve_source_doc(tmp_path):
    from pathlib import Path
    import app
    repo_root = Path(app.__file__).resolve().parent.parent
    inside = repo_root / "knowledge_docs" / "商品FAQ.md"
    assert resolve_source_doc(inside) == "knowledge_docs/商品FAQ.md"
    outside = tmp_path / "x.md"
    assert resolve_source_doc(outside) == outside.resolve().as_posix()
```

- [ ] **Step 2: 跑测试确认失败(模块不存在)**

- [ ] **Step 3: 实现 `app/knowledge/ingest.py`(全文)**

```python
"""建库两阶段流水线(spec §6):Phase 1 文档事务落 MySQL(pending),
Phase 2 向量化 upsert Milvus 并回填 done。中断重跑幂等。"""

from pathlib import Path

from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.knowledge.chunking import ChunkingError, chunk_document
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import KnowledgeChunk

REPO_ROOT = Path(__file__).resolve().parents[2]
VECTORIZE_BATCH = 32


class IngestError(Exception):
    pass


def vector_text(category: str, questions: str, answer: str) -> str:
    return f"{category}\n{questions}\n{answer}"


def resolve_source_doc(path: Path) -> str:
    real = path.resolve()
    try:
        rel = real.relative_to(REPO_ROOT)
        ident = rel.as_posix()
    except ValueError:
        ident = real.as_posix()
    if len(ident) > 255:
        raise IngestError(f"来源路径超过 255 字符: {ident}")
    return ident


def vectorize_pending(settings: Settings, session_factory: sessionmaker,
                      embed, store: MilvusKnowledgeStore) -> None:
    """Phase 2:pending 批次向量化。任一批失败抛 IngestError,该批保持 pending。"""
    while True:
        with session_factory() as s:
            rows = (s.query(KnowledgeChunk)
                    .filter_by(vectorize_status="pending")
                    .order_by(KnowledgeChunk.id)
                    .limit(VECTORIZE_BATCH).all())
            if not rows:
                return
            payloads = [(r.id, vector_text(r.category, r.questions, r.answer)) for r in rows]
        try:
            vectors = embed.embed_documents([t for _, t in payloads])
            if len(vectors) != len(payloads):
                raise IngestError(
                    f"返回向量数 {len(vectors)} ≠ 输入 {len(payloads)}")
            for v in vectors:
                if len(v) != store._dim:
                    raise IngestError(f"向量维度 {len(v)} ≠ 集合维度 {store._dim}")
            store.upsert(list(zip([i for i, _ in payloads], vectors)))
            with session_factory() as s:
                for chunk_id, _ in payloads:
                    s.query(KnowledgeChunk).filter_by(id=chunk_id).update(
                        {"vector_id": str(chunk_id), "vectorize_status": "done"})
                s.commit()
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError(f"向量化批次失败: {type(exc).__name__}: {exc}") from exc


def _same_chunk(row: KnowledgeChunk, chunk) -> bool:
    return (row.category == chunk.category and row.questions == chunk.questions
            and row.answer == chunk.answer and row.section_path == chunk.section_path
            and row.content_type == chunk.content_type
            and bool(row.is_key_clause) == chunk.is_key_clause)


def _load_document(session_factory: sessionmaker, path: Path, settings: Settings) -> None:
    source_doc = resolve_source_doc(path)
    try:
        chunks = chunk_document(path.read_text(encoding="utf-8"), source=source_doc,
                                max_chars=settings.max_chunk_chars,
                                overlap_chars=settings.chunk_overlap_chars)
    except ChunkingError as exc:
        raise IngestError(str(exc)) from exc
    with session_factory() as s:
        try:
            existing = (s.query(KnowledgeChunk)
                        .filter_by(source_doc=source_doc)
                        .order_by(KnowledgeChunk.chunk_index).all())
            if existing:
                if (len(existing) != len(chunks)
                        or not all(_same_chunk(r, c) for r, c in zip(existing, chunks))):
                    raise IngestError(
                        f"已导入文档或切分配置发生变化: {source_doc}(不覆盖旧知识)")
                rows = existing  # 原样重跑:复用 ID,仍重建指针
            else:
                rows = []
                for i, c in enumerate(chunks, start=1):
                    row = KnowledgeChunk(
                        category=c.category, questions=c.questions, answer=c.answer,
                        section_path=c.section_path, content_type=c.content_type,
                        is_key_clause=c.is_key_clause, source_doc=source_doc,
                        chunk_index=i, vectorize_status="pending")
                    s.add(row)
                    rows.append(row)
                s.flush()  # 拿自增 ID
            for i, row in enumerate(rows):
                row.prev_chunk_id = rows[i - 1].id if i > 0 else None
                row.next_chunk_id = rows[i + 1].id if i + 1 < len(rows) else None
            s.commit()  # 整篇文档只提交一次;中断则整体回滚
        except IngestError:
            s.rollback()
            raise
        except Exception as exc:
            s.rollback()
            raise IngestError(f"文档入库失败 {source_doc}: {type(exc).__name__}") from exc


def _verify(session_factory: sessionmaker, store: MilvusKnowledgeStore) -> None:
    with session_factory() as s:
        pending = s.query(KnowledgeChunk).filter_by(vectorize_status="pending").count()
        if pending:
            raise IngestError(f"仍有 {pending} 个 pending 块未向量化")
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        by_doc: dict[str, list[KnowledgeChunk]] = {}
        for r in rows:
            if r.source_doc is not None:
                by_doc.setdefault(r.source_doc, []).append(r)
        for doc, doc_rows in by_doc.items():
            doc_rows.sort(key=lambda r: r.chunk_index)
            for i, r in enumerate(doc_rows):
                expect_prev = doc_rows[i - 1].id if i > 0 else None
                expect_next = doc_rows[i + 1].id if i + 1 < len(doc_rows) else None
                if r.prev_chunk_id != expect_prev or r.next_chunk_id != expect_next:
                    raise IngestError(f"指针不完整: {doc} chunk_index={r.chunk_index}")
        mysql_ids = {r.id for r in rows}
    milvus_ids = store.all_ids()
    if milvus_ids != mysql_ids:
        raise IngestError(
            f"两库主键集合不一致: 仅 Milvus {sorted(milvus_ids - mysql_ids)},"
            f"仅 MySQL {sorted(mysql_ids - milvus_ids)}")


def run_ingest(settings: Settings, session_factory: sessionmaker, embed,
               store: MilvusKnowledgeStore, docs_dir: Path) -> int:
    try:
        store.ensure_collection()
        vectorize_pending(settings, session_factory, embed, store)  # resume 历史 pending
        paths = sorted(docs_dir.glob("*.md"), key=lambda p: resolve_source_doc(p))
        if not paths:
            raise IngestError(f"目录无 Markdown 文档: {docs_dir}")
        for path in paths:
            _load_document(session_factory, path, settings)
        vectorize_pending(settings, session_factory, embed, store)
        _verify(session_factory, store)
    except IngestError as exc:
        print(f"[ingest] 失败: {exc}")
        return 1
    print("[ingest] 完成: 无 pending,指针完整,两库主键集合一致")
    return 0
```

- [ ] **Step 4: 实现 `app/jobs/ingest_docs.py`**

```python
"""python -m app.jobs.ingest_docs [docs_dir] —— 知识文档建库(spec §6)。"""

import sys
from pathlib import Path

from app.config import Settings
from app.db import make_engine, make_session_factory, ping
from app.knowledge.embedding import build_embeddings
from app.knowledge.ingest import run_ingest
from app.knowledge.milvus_store import MilvusKnowledgeStore


def main(argv: list[str]) -> int:
    docs_dir = Path(argv[1]) if len(argv) > 1 else Path("knowledge_docs")
    settings = Settings()
    if not settings.has_embedding_key():
        print("[ingest] embedding_api_key 未配置,写库前退出", file=sys.stderr)
        return 2
    engine = make_engine(settings.database_url)
    try:
        ping(engine)
    except Exception as exc:
        print(f"[ingest] 数据库不可达: {type(exc).__name__}", file=sys.stderr)
        return 2
    session_factory = make_session_factory(engine)
    store = MilvusKnowledgeStore(settings.milvus_uri, settings.embedding_dim)
    try:
        return run_ingest(settings, session_factory, build_embeddings(settings),
                          store, docs_dir)
    finally:
        store.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 5: 跑 `uv run pytest tests/test_ingest.py -v` 全绿,再全量回归**

- [ ] **Step 6: Commit**

```bash
git add app/knowledge/ingest.py app/jobs/__init__.py app/jobs/ingest_docs.py tests/test_ingest.py
git commit -m "feat(ch03): 建库两阶段流水线(文档事务 pending → upsert → 回填 done,中断重跑幂等)"
```

---

### Task 6: mining.py + mine_qa CLI —— 对话挖知识

**Files:**
- Create: `app/knowledge/mining.py`、`app/jobs/mine_qa.py`
- Test: `tests/test_mining.py`(stub chat model + 真实 Docker MySQL + 真实 Milvus Lite tmp + fake embedding)

**Interfaces:**
- Consumes: `vectorize_pending`(T5)、`MilvusKnowledgeStore`(T4)、三个新模型(T2)
- Produces:
  - `normalize_question(q: str) -> str`
  - `MinedQA / MinedConversation / MiningBatchResult`(pydantic,structured output schema)
  - `run_mining(settings, session_factory, model, embed, store) -> int`(0 成功 / 1 有失败批次 / 2 前置校验失败)
  - `MiningError(Exception)`

**流程(spec §7)**: 校验 Key → `store.ensure_collection()` → 恢复(去重 → kept 入库 → pending 向量化)→ 拉未挖会话(progress 表排除)→ 分批 LLM 抽取(staging+进度同事务)→ 再跑去重/入库/向量化 → 有失败批次则非零退出。

- [ ] **Step 1: 失败测试 `tests/test_mining.py`**

```python
import pytest

from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.mining import (
    MiningBatchResult, MinedConversation, MinedQA, normalize_question, run_mining,
)
from app.models import (
    Conversation, KnowledgeChunk, Message, QaExtractionStaging, QaMiningProgress,
)
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class StubStructuredModel:
    """with_structured_output 返回脚本化 parsed;记录调用次数与输入。"""
    def __init__(self, parsed):
        self._parsed = parsed
        self.calls = 0
        self.received = None

    def with_structured_output(self, schema, method=None, include_raw=False):
        self.calls += 1
        outer = self

        class _R:
            def invoke(self, messages):
                outer.received = messages
                return {"parsed": outer._parsed, "parsing_error": None}

        return _R()


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    def embed_query(self, text):
        return [0.5, 0.5, 0.5, 0.5]


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "mining.db"), dim=4)
    yield s
    s.close()


@pytest.fixture()
def settings():
    return make_settings()


def _mk_conversation(sf, user_texts=("退货怎么弄",), answers=("七天无理由。",)):
    with sf() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        s.flush()
        for i, (u, a) in enumerate(zip(user_texts, answers)):
            s.add(Message(conversation_id=conv.id, role="user", content=u))
            s.add(Message(conversation_id=conv.id, role="assistant", content=a))
        s.commit()
        return conv.id


def _parsed_for(conv_id, qas):
    return MiningBatchResult(conversations=[
        MinedConversation(conversation_id=conv_id,
                          items=[MinedQA(question=q, answer=a) for q, a in qas])
    ])


def test_mining_happy_path(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "支持七天无理由退货")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        chunk = s.query(KnowledgeChunk).one()
        assert chunk.content_type == "qa_mined" and chunk.category == "对话挖掘"
        assert chunk.questions == "退货政策是什么"
        assert chunk.source_doc is None and chunk.chunk_index is None
        assert chunk.vectorize_status == "done"
        prog = s.query(QaMiningProgress).one()
        assert prog.conversation_id == cid and prog.qa_count == 1
        staging = s.query(QaExtractionStaging).one()
        assert staging.source_ref == f"conv:{cid}" and staging.status == "kept"
    assert store.all_ids() == {chunk.id}


def test_mining_rerun_skips_without_llm(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "七天无理由")]))
    run_mining(settings, db_session_factory, model, FakeEmbeddings(), store)
    assert model.calls == 1
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    assert model.calls == 1  # 重跑不再调 LLM
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 1


def test_zero_qa_conversation_progress(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory, ("你好",), ("你好,请问有什么可以帮您?",))
    model = StubStructuredModel(_parsed_for(cid, []))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        prog = s.query(QaMiningProgress).one()
        assert prog.qa_count == 0
        assert s.query(KnowledgeChunk).count() == 0
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    assert model.calls == 1  # 零 QA 会话重跑也跳过


def test_dedup_within_staging(db_session_factory, store, settings):
    c1 = _mk_conversation(db_session_factory, ("邮费多少",), ("满99包邮",))
    c2 = _mk_conversation(db_session_factory, ("邮费多少?",), ("99元包邮",))
    model = StubStructuredModel(MiningBatchResult(conversations=[
        MinedConversation(conversation_id=c1, items=[MinedQA(question="邮费多少", answer="满99包邮")]),
        MinedConversation(conversation_id=c2, items=[MinedQA(question="邮费多少?", answer="99元包邮")]),
    ]))
    settings = make_settings(mining_batch_size=10)  # 两个会话同批
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        kept = s.query(QaExtractionStaging).filter_by(status="kept").all()
        discarded = s.query(QaExtractionStaging).filter_by(status="discarded").all()
        assert len(kept) == 1 and len(discarded) == 1  # 规范化后同问法,留最早 ID
        assert kept[0].question == "邮费多少"
        assert s.query(KnowledgeChunk).count() == 1


def test_dedup_against_existing_knowledge(db_session_factory, store, settings):
    with db_session_factory() as s:
        s.add(KnowledgeChunk(category="退货", questions="退货政策是什么",
                             answer="七天无理由", content_type="faq",
                             source_doc="knowledge_docs/x.md", chunk_index=1,
                             vectorize_status="done", vector_id="0"))
        s.commit()
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么?", "七天")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        assert s.query(QaExtractionStaging).one().status == "discarded"
        assert s.query(KnowledgeChunk).count() == 1


def test_batch_missing_conversation_rejected(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(MiningBatchResult(conversations=[
        MinedConversation(conversation_id=cid + 999, items=[]),  # 编造 ID
    ]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 1
    with db_session_factory() as s:  # 整批不留 staging/进度
        assert s.query(QaExtractionStaging).count() == 0
        assert s.query(QaMiningProgress).count() == 0


def test_blank_question_rejected(db_session_factory, store, settings):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("   ", "答案")]))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 1
    with db_session_factory() as s:
        assert s.query(QaExtractionStaging).count() == 0


def test_recover_kept_after_crash_before_load(db_session_factory, store, settings, monkeypatch):
    cid = _mk_conversation(db_session_factory)
    model = StubStructuredModel(_parsed_for(cid, [("退货政策是什么", "七天无理由")]))
    import app.knowledge.mining as mining_mod
    real_load = mining_mod._load_kept
    monkeypatch.setattr(mining_mod, "_load_kept",
                        lambda sf: (_ for _ in ()).throw(RuntimeError("crash")))
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 1
    with db_session_factory() as s:  # staging/进度已提交,kept 未入库
        assert s.query(QaExtractionStaging).one().status == "kept"
        assert s.query(KnowledgeChunk).count() == 0
    monkeypatch.setattr(mining_mod, "_load_kept", real_load)
    assert run_mining(settings, db_session_factory, model, FakeEmbeddings(), store) == 0
    with db_session_factory() as s:
        assert s.query(KnowledgeChunk).count() == 1  # 重放补齐,不重复插
    assert model.calls == 1


def test_normalize_question():
    assert normalize_question("邮费多少?") == normalize_question(" 邮费 多少？")
    assert normalize_question("ABC") == "abc"
```

- [ ] **Step 2: 跑测试确认失败(模块不存在)**

- [ ] **Step 3: 实现 `app/knowledge/mining.py`(全文)**

```python
"""对话挖知识(spec §7):拉会话 → 分批 LLM 抽取 → 整体去重 → 入库向量化。

幂等设计:qa_mining_progress 记成功抽取(含零 QA);staging 记候选与去重结果;
kept 入库按三格精确匹配重放;向量化复用 ingest.vectorize_pending。
"""

import logging
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.knowledge.ingest import vectorize_pending
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import (
    Conversation, KnowledgeChunk, Message, QaExtractionStaging, QaMiningProgress,
)

logger = logging.getLogger(__name__)


class MiningError(Exception):
    pass


class MinedQA(BaseModel):
    question: str
    answer: str


class MinedConversation(BaseModel):
    conversation_id: int
    items: list[MinedQA]


class MiningBatchResult(BaseModel):
    conversations: list[MinedConversation]


_PUNCT_RE = re.compile(r"[\s，。、！？；：,.!?;:'\"“”‘’（）()【】\[\]<>《》—…·~\-]+")

MINING_SYSTEM = """你是客服知识挖掘器。输入是一批客服对话,每段以「会话 <id>」开头。
从每段对话抽取可复用的客服知识问答对(政策、规则、流程、费用类),
不要订单号、用户个人信息等一次性内容;没有可抽取内容的会话返回空 items。
不得合并不同会话的事实;问答必须有对话原文依据,不得编造。
每个输入会话在输出中恰好出现一次,不得遗漏、重复或编造会话 ID。"""


def normalize_question(q: str) -> str:
    return _PUNCT_RE.sub("", q).lower()


def _dialogue_text(s, conversation_id: int) -> str:
    rows = (s.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at, Message.id).all())
    lines = []
    for m in rows:
        if m.role == "user" and m.content and m.content.strip():
            lines.append(f"用户: {m.content.strip()}")
        elif m.role == "assistant" and m.content and m.content.strip():
            lines.append(f"客服: {m.content.strip()}")
    return "\n".join(lines)


def _extract_batch(settings: Settings, session_factory: sessionmaker, model,
                   conv_ids: list[int], batch_no: str) -> None:
    """一批会话:LLM 抽取 → 结构校验 → staging+进度同事务提交。失败抛 MiningError。"""
    with session_factory() as s:
        parts = [f"会话 {cid}\n{_dialogue_text(s, cid)}" for cid in conv_ids]
    human = "\n\n".join(parts)
    messages = [SystemMessage(content=MINING_SYSTEM), HumanMessage(content=human)]
    if count_tokens_approximately(messages) > settings.max_input_tokens:
        if len(conv_ids) == 1:
            raise MiningError(f"单个会话超输入预算: conv {conv_ids[0]}")
        mid = len(conv_ids) // 2  # 超预算缩批:两半各自递归
        _extract_batch(settings, session_factory, model, conv_ids[:mid], batch_no + "a")
        _extract_batch(settings, session_factory, model, conv_ids[mid:], batch_no + "b")
        return
    structured = model.with_structured_output(
        MiningBatchResult, method=settings.structured_output_method, include_raw=True)
    try:
        result = structured.invoke(messages)
    except Exception as exc:
        raise MiningError(f"LLM 抽取失败: {type(exc).__name__}") from exc
    if result.get("parsing_error") is not None or result.get("parsed") is None:
        raise MiningError("structured output 解析失败")
    parsed: MiningBatchResult = result["parsed"]
    got_ids = [c.conversation_id for c in parsed.conversations]
    if sorted(got_ids) != sorted(conv_ids) or len(set(got_ids)) != len(got_ids):
        raise MiningError(f"输出会话集合不符: 期望 {sorted(conv_ids)},实际 {sorted(got_ids)}")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_factory() as s:
        try:
            for conv in parsed.conversations:
                for item in conv.items:
                    q, a = item.question.strip(), item.answer.strip()
                    if not q or not a or not normalize_question(q):
                        raise MiningError(
                            f"空白 question/answer: conv {conv.conversation_id}")
                    s.add(QaExtractionStaging(
                        batch_no=batch_no, source_ref=f"conv:{conv.conversation_id}",
                        question=q, answer=a, status="extracted"))
                s.add(QaMiningProgress(
                    conversation_id=conv.conversation_id, batch_no=batch_no,
                    qa_count=len(conv.items), extracted_at=now))
            s.commit()
        except MiningError:
            s.rollback()
            raise
        except Exception as exc:
            s.rollback()
            raise MiningError(f"批次提交失败: {type(exc).__name__}") from exc


def _dedup_staging(session_factory: sessionmaker) -> None:
    """整体去重:已有知识优先,候选内部留最早 ID(spec §7 步骤 3)。"""
    with session_factory() as s:
        known: set[str] = set()
        for (questions,) in s.query(KnowledgeChunk.questions).all():
            known.update(normalize_question(line) for line in questions.splitlines())
        rows = (s.query(QaExtractionStaging)
                .filter(QaExtractionStaging.status.in_(["extracted", "kept"]))
                .order_by(QaExtractionStaging.id).all())
        changed = False
        for row in rows:
            key = normalize_question(row.question)
            if key in known:
                if row.status != "discarded":
                    row.status = "discarded"
                    changed = True
            else:
                known.add(key)
                if row.status != "kept":
                    row.status = "kept"
                    changed = True
        if changed:
            s.commit()


def _load_kept(session_factory: sessionmaker) -> None:
    """kept → knowledge_chunks(三格精确匹配重放防重);新行 pending。"""
    with session_factory() as s:
        kept = (s.query(QaExtractionStaging).filter_by(status="kept")
                .order_by(QaExtractionStaging.id).all())
        existing = {
            (r.category, r.questions, r.answer)
            for r in s.query(KnowledgeChunk)
            .filter_by(content_type="qa_mined", source_doc=None).all()
        }
        added = False
        for row in kept:
            triple = ("对话挖掘", row.question, row.answer)
            if triple in existing:
                continue
            s.add(KnowledgeChunk(
                category="对话挖掘", questions=row.question, answer=row.answer,
                section_path=None, content_type="qa_mined", is_key_clause=False,
                source_doc=None, chunk_index=None, vectorize_status="pending"))
            existing.add(triple)
            added = True
        if added:
            s.commit()


def run_mining(settings: Settings, session_factory: sessionmaker, model, embed,
               store: MilvusKnowledgeStore) -> int:
    if not settings.has_embedding_key():
        print("[mining] embedding_api_key 未配置,写库前退出")
        return 2
    failed = False
    store.ensure_collection()
    try:
        # 恢复:历史 staging 的去重/kept 入库/pending 向量化先补齐
        _dedup_staging(session_factory)
        _load_kept(session_factory)
        vectorize_pending(settings, session_factory, embed, store)
    except Exception as exc:
        print(f"[mining] 恢复阶段失败: {type(exc).__name__}: {exc}")
        return 1
    with session_factory() as s:
        done_ids = {r[0] for r in s.query(QaMiningProgress.conversation_id).all()}
        conv_ids = [r[0] for r in s.query(Conversation.id).order_by(Conversation.id).all()
                    if r[0] not in done_ids]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seq = 0
    for i in range(0, len(conv_ids), settings.mining_batch_size):
        batch = conv_ids[i:i + settings.mining_batch_size]
        seq += 1
        batch_no = f"{stamp}-{seq:03d}"
        try:
            _extract_batch(settings, session_factory, model, batch, batch_no)
            logger.info("mining batch %s ok: %d conversations", batch_no, len(batch))
        except MiningError as exc:
            failed = True
            logger.warning("mining batch %s failed (convs=%s): %s", batch_no, batch, exc)
    try:
        _dedup_staging(session_factory)
        _load_kept(session_factory)
        vectorize_pending(settings, session_factory, embed, store)
    except Exception as exc:
        failed = True
        logger.warning("mining finalize failed: %s: %s", type(exc).__name__, exc)
    return 1 if failed else 0
```

注意:`_dedup_staging` 把 kept 也重新纳入判重(中断恢复语义:已 kept 行参与判重,spec §7),状态不变的行不写;这保证「上次 crash 在 kept 未入库」与「新批次与旧 kept 撞问法」都正确。

- [ ] **Step 4: 实现 `app/jobs/mine_qa.py`**

```python
"""python -m app.jobs.mine_qa —— 从历史客服对话挖知识(spec §7)。"""

import sys

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.db import make_engine, make_session_factory, ping
from app.knowledge.embedding import build_embeddings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.mining import run_mining


def main(argv: list[str]) -> int:
    settings = Settings()
    if not settings.has_embedding_key():
        print("[mining] embedding_api_key 未配置,写库前退出", file=sys.stderr)
        return 2
    engine = make_engine(settings.database_url)
    try:
        ping(engine)
    except Exception as exc:
        print(f"[mining] 数据库不可达: {type(exc).__name__}", file=sys.stderr)
        return 2
    model = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key,
                       base_url=settings.openai_base_url,
                       max_tokens=settings.max_output_tokens)
    session_factory = make_session_factory(engine)
    store = MilvusKnowledgeStore(settings.milvus_uri, settings.embedding_dim)
    try:
        return run_mining(settings, session_factory, model,
                          build_embeddings(settings), store)
    finally:
        store.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 5: 跑 `uv run pytest tests/test_mining.py -v` 全绿,再全量回归**

- [ ] **Step 6: Commit**

```bash
git add app/knowledge/mining.py app/jobs/mine_qa.py tests/test_mining.py
git commit -m "feat(ch03): 对话挖知识(分批抽取/staging/整体去重/进度表/重放入库)"
```

---

### Task 7: retriever.py + query_faq 替换 + executor 接线 + main 装配

**Files:**
- Create: `app/knowledge/retriever.py`
- Test: `tests/test_retriever.py`
- Modify: `app/tools/business.py`(query_faq 换实现)、`app/tools/executor.py`(RETRYABLE 加成员)、`app/main.py`(AppRuntime 装配 + lifespan 释放)、`tests/test_tools.py`(query_faq 旧 LIKE 测试替换)、`tests/test_executor.py`(重试分级测试)

**Interfaces:**
- Consumes: `MilvusKnowledgeStore`(T4)、`build_embeddings`(T4)、`KnowledgeChunk`(T2)
- Produces:
  - `RetryableKnowledgeError(Exception)`
  - `KnowledgeHit` dataclass(frozen): `chunk_id: int, score: float, category: str, questions: str, answer: str, source_doc: str | None, chunk_index: int | None`
  - `KnowledgeRetriever(settings, embed=None, store=None, session_factory=None)`:`enabled` 属性、`search(query: str, min_score: float | None = None) -> tuple[list[KnowledgeHit], str | None]`、`close()`
  - `build_tools(session_factory, conversation_id, retriever)`(第三参数新增,契约内调用 `retriever.search(keyword)`)

- [ ] **Step 1: 失败测试 `tests/test_retriever.py`**

```python
import json

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.retriever import (
    KnowledgeRetriever, RetryableKnowledgeError, _as_retryable,
)
from app.models import KnowledgeChunk
from app.tools.business import build_tools
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class FakeEmbeddings:
    """按关键词定向向量:「邮费」→ e1,「登录」→ e2,其余 → e3(与库存向量都不相似)。"""
    def embed_query(self, text):
        if "邮费" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "登录" in text:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]

    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]  # 检索测试不经此路径


@pytest.fixture()
def store(tmp_path):
    s = MilvusKnowledgeStore(str(tmp_path / "retriever.db"), dim=4)
    yield s
    s.close()


def _seed_knowledge(sf, store):
    with sf() as s:
        rows = [
            KnowledgeChunk(category="商品FAQ", questions="运费怎么算",
                           answer="满 99 包邮,未满 8 元。", content_type="faq",
                           source_doc="knowledge_docs/商品FAQ.md", chunk_index=1,
                           vectorize_status="done", vector_id="1"),
            KnowledgeChunk(category="账户", questions="忘记密码",
                           answer="登录页点忘记密码。", content_type="faq",
                           source_doc="knowledge_docs/商品FAQ.md", chunk_index=2,
                           vectorize_status="done", vector_id="2"),
        ]
        s.add_all(rows)
        s.flush()
        ids = [r.id for r in rows]
        s.commit()
    store.ensure_collection()
    store.upsert([(ids[0], [1.0, 0.0, 0.0, 0.0]), (ids[1], [0.0, 1.0, 0.0, 0.0])])
    return ids


def _retriever(settings, store, sf, embed=None):
    return KnowledgeRetriever(settings, embed=embed or FakeEmbeddings(),
                              store=store, session_factory=sf)


def test_search_topk_threshold_order(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.5)
    ids = _seed_knowledge(db_session_factory, store)
    hits, note = _retriever(settings, store, db_session_factory).search("邮费是多少")
    assert note is None
    assert [h.chunk_id for h in hits] == [ids[0]]  # e2 方向被阈值滤掉
    assert hits[0].score >= 0.5
    assert hits[0].source_doc == "knowledge_docs/商品FAQ.md" and hits[0].chunk_index == 1


def test_query_faq_contract_unchanged(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.5)
    _seed_knowledge(db_session_factory, store)
    retriever = _retriever(settings, store, db_session_factory)
    tools = build_tools(db_session_factory, 1, retriever)
    faq = next(t for t in tools if t.name == "query_faq")
    out = json.loads(faq.invoke({"keyword": "邮费是多少"}))
    assert set(out) == {"results"}
    assert set(out["results"][0]) == {"question", "answer", "category"}
    assert out["results"][0]["answer"] == "满 99 包邮,未满 8 元。"
    out2 = json.loads(faq.invoke({"keyword": "登录"}))
    assert out2["results"] and "忘记密码" in out2["results"][0]["question"]


def test_query_faq_no_hit_note(db_session_factory, store):
    settings = make_settings(knowledge_min_score=0.99)
    _seed_knowledge(db_session_factory, store)
    retriever = _retriever(settings, store, db_session_factory)
    faq = next(t for t in build_tools(db_session_factory, 1, retriever)
               if t.name == "query_faq")
    out = json.loads(faq.invoke({"keyword": "完全不沾边的问题xyz"}))
    assert out["results"] == [] and out["note"]


def test_missing_file_not_created(store):
    settings = make_settings()
    r = KnowledgeRetriever(settings, embed=FakeEmbeddings(), store=store,
                           session_factory=None)
    hits, note = r.search("邮费")
    assert hits == [] and note == "知识库尚未建立"
    assert store.file_exists() is False  # 缺文件时不创建文件/客户端


def test_missing_collection(store):
    store._cli()  # 建文件但不建集合
    store.close()
    settings = make_settings()
    r = KnowledgeRetriever(settings, embed=FakeEmbeddings(), store=store,
                           session_factory=None)
    hits, note = r.search("邮费")
    assert hits == [] and note == "知识库尚未建立"


def test_disabled_without_key(db_session_factory, store):
    settings = make_settings(embedding_api_key="")
    r = KnowledgeRetriever(settings, embed=None, store=store,
                           session_factory=db_session_factory)
    assert r.enabled is False
    hits, note = r.search("邮费")
    assert hits == [] and note == "知识检索未配置"


def test_retryable_mapping():
    req = httpx.Request("POST", "http://x/v1/embeddings")
    assert _as_retryable(APIConnectionError(request=req)) is not None
    assert _as_retryable(APITimeoutError(request=req)) is not None
    resp429 = httpx.Response(429, request=req)
    assert _as_retryable(RateLimitError("x", response=resp429, body=None)) is not None
    resp500 = httpx.Response(500, request=req)
    assert _as_retryable(APIStatusError("x", response=resp500, body=None)) is not None
    resp401 = httpx.Response(401, request=req)
    assert _as_retryable(APIStatusError("x", response=resp401, body=None)) is None
    assert _as_retryable(ValueError("bad")) is None


def test_executor_retries_retryable_knowledge_error():
    import asyncio
    from langchain_core.tools import tool
    from app.tools.executor import ToolExecutor, ToolRegistry

    calls = {"n": 0}

    @tool
    def flaky(q: str) -> str:
        """t"""
        calls["n"] += 1
        raise RetryableKnowledgeError("boom")

    ex = ToolExecutor(ToolRegistry([flaky]), timeout_seconds=5, max_retries=2,
                      max_result_chars=4000)
    outcome = asyncio.run(ex.execute({"name": "flaky", "args": {"q": "x"}, "id": "1"}))
    assert outcome.record.ok is False
    assert outcome.record.error_code == "tool_unavailable"
    assert calls["n"] == 3  # 1 + 2 次重试


def test_executor_no_retry_on_plain_error():
    import asyncio
    from langchain_core.tools import tool
    from app.tools.executor import ToolExecutor, ToolRegistry

    calls = {"n": 0}

    @tool
    def broken(q: str) -> str:
        """t"""
        calls["n"] += 1
        raise ValueError("auth failed")

    ex = ToolExecutor(ToolRegistry([broken]), timeout_seconds=5, max_retries=2,
                      max_result_chars=4000)
    outcome = asyncio.run(ex.execute({"name": "broken", "args": {"q": "x"}, "id": "1"}))
    assert outcome.record.error_code == "tool_error"
    assert calls["n"] == 1
```

脆点:openai 异常构造签名(`APIStatusError(message, response=..., body=None)`)随 SDK 版本微调,以锁定版本实测为准。

- [ ] **Step 2: `tests/test_tools.py` 改造**

`build_tools` 第三参数带默认值 `None`,既有调用点(`build_tools(sf, 1)`)**不用改签名**;只需删除 query_faq 的三个 LIKE 时代测试(命中/漏召/通配符转义)及 `_escape_like` 相关断言——该行为已整体迁入 `test_retriever.py`。mock 三工具与 create_ticket 测试原样保留。

- [ ] **Step 3: 实现 `app/knowledge/retriever.py`(全文)**

```python
"""在线语义检索(spec §8):query → embed → Milvus Top-K → 阈值过滤 → 回 MySQL 取原文。

内部命中记录(KnowledgeHit)带 chunk_id/score/source_doc/chunk_index,供评估比对;
query_faq 只投影 question/answer/categories 三键。
"""

import logging
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import Settings
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import KnowledgeChunk

logger = logging.getLogger(__name__)

NOTE_UNCONFIGURED = "知识检索未配置"
NOTE_NOT_BUILT = "知识库尚未建立"


class RetryableKnowledgeError(Exception):
    """可重试的知识检索故障(连接/超时/限流/暂时性 HTTP),由 executor 统一重试。"""


@dataclass(frozen=True)
class KnowledgeHit:
    chunk_id: int
    score: float
    category: str
    questions: str
    answer: str
    source_doc: str | None
    chunk_index: int | None


def _as_retryable(exc: Exception) -> RetryableKnowledgeError | None:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError,
                        TimeoutError, ConnectionError)):
        return RetryableKnowledgeError(type(exc).__name__)
    if isinstance(exc, APIStatusError) and (
            exc.status_code in (408, 409) or exc.status_code >= 500):
        return RetryableKnowledgeError(f"HTTP {exc.status_code}")
    return None


class KnowledgeRetriever:
    def __init__(self, settings: Settings, embed=None,
                 store: MilvusKnowledgeStore | None = None, session_factory=None):
        self._settings = settings
        self._embed = embed
        self._store = store or MilvusKnowledgeStore(settings.milvus_uri,
                                                    settings.embedding_dim)
        self._sf = session_factory

    @property
    def enabled(self) -> bool:
        return self._embed is not None

    def close(self) -> None:
        self._store.close()

    def search(self, query: str,
               min_score: float | None = None) -> tuple[list[KnowledgeHit], str | None]:
        """→ (hits, note)。note 非空 = 降级/未建库;hits 空且 note 空 = 正常无命中。
        min_score 仅供评估覆盖阈值;在线调用不传。"""
        threshold = self._settings.knowledge_min_score if min_score is None else min_score
        if not self.enabled:
            return [], NOTE_UNCONFIGURED
        if not self._store.file_exists():
            return [], NOTE_NOT_BUILT
        try:
            vector = self._embed.embed_query(query)
        except Exception as exc:
            retryable = _as_retryable(exc)
            if retryable is not None:
                raise retryable from exc
            raise
        if len(vector) != self._store._dim:
            raise ValueError(f"查询向量维度 {len(vector)} ≠ 集合维度 {self._store._dim}")
        try:
            if not self._store.has_collection():
                return [], NOTE_NOT_BUILT
            raw = self._store.search(vector, self._settings.knowledge_top_k)
        except Exception as exc:
            retryable = _as_retryable(exc)
            if retryable is not None:
                raise retryable from exc
            raise
        hits = [(i, d) for i, d in raw if d >= threshold]
        if not hits:
            return [], None
        ids = [i for i, _ in hits]
        with self._sf() as s:
            rows = {r.id: r for r in s.query(KnowledgeChunk)
                    .filter(KnowledgeChunk.id.in_(ids)).all()}
        out: list[KnowledgeHit] = []
        for chunk_id, score in hits:  # 保持 Milvus 相似度顺序
            row = rows.get(chunk_id)
            if row is None:
                continue  # 向量在而原文被删:跳过不炸
            out.append(KnowledgeHit(chunk_id, score, row.category, row.questions,
                                    row.answer, row.source_doc, row.chunk_index))
        return out, None
```

- [ ] **Step 4: `app/tools/business.py` query_faq 换实现**

- 删除 `_escape_like` 与 `Faq` import(不再使用)
- `build_tools(session_factory, conversation_id: int, retriever=None)`:`retriever=None` 时 query_faq 直接返回未配置 note(保护旧调用点与 conftest 的 MOCK_TOOLS 路径——MOCK_TOOLS 本就不含 query_faq,此默认仅为防御):

```python
def build_tools(session_factory, conversation_id: int, retriever=None) -> list[BaseTool]:
    """每轮请求构造绑定该会话的工具实例;conversation_id 经闭包注入,不对模型暴露。"""

    @tool
    def query_faq(keyword: Keyword) -> str:
        """查询知识库。参数 keyword 为用户问题或关键词,语义检索返回最相关的前 5 条。"""
        if retriever is None:
            return _json({"results": [], "note": "知识检索未配置"})
        hits, note = retriever.search(keyword)
        if not hits:
            return _json({"results": [], "note": note or "未找到与关键词相关的常见问题"})
        return _json({"results": [
            {"question": h.questions, "answer": h.answer, "category": h.category}
            for h in hits
        ]})

    @tool
    def create_ticket(...):  # 原样保留
        ...

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
```

- [ ] **Step 5: `app/tools/executor.py` 接线**

```python
from app.knowledge.retriever import RetryableKnowledgeError

RETRYABLE = (TimeoutError, asyncio.TimeoutError, OperationalError, ConnectionError,
             RetryableKnowledgeError)
```

- [ ] **Step 6: `app/main.py` 装配**

```python
# AppRuntime 增加可选 retriever 字段
@dataclass(frozen=True)
class AppRuntime:
    store: Any  # SessionStore 协议
    toolset_factory: Callable[[str], list[BaseTool]]
    retriever: Any = None


def _build_production_runtime(settings: Settings) -> AppRuntime:
    try:
        engine = make_engine(settings.database_url)
        ping(engine)
    except Exception as exc:
        raise RuntimeError(f"database ping failed: {type(exc).__name__}") from exc
    session_factory = make_session_factory(engine)
    # 缺 Key 时构造禁用检索的实例,不构造需要 Key 的 embedding 客户端(spec §8)
    embed = build_embeddings(settings) if settings.has_embedding_key() else None
    retriever = KnowledgeRetriever(settings, embed=embed, session_factory=session_factory)

    def toolset_factory(session_id: str) -> list[BaseTool]:
        return build_tools(session_factory, int(session_id), retriever)

    return AppRuntime(store=DbSessionStore(session_factory, settings.max_message_chars),
                      toolset_factory=toolset_factory, retriever=retriever)
```

`create_app` 里:记录 `owns_runtime = runtime is None`;`app = FastAPI(title="wayhelp-ch03", lifespan=...)` 用 lifespan 在关闭时释放:

```python
from contextlib import asynccontextmanager

# create_app 内,runtime 确定之后:
    owns_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if owns_runtime and runtime.retriever is not None:
            runtime.retriever.close()

    app = FastAPI(title="wayhelp-ch03", lifespan=lifespan)
```

(lifespan 是 FastAPI ≥0.93 的标准机制,仓库 fastapi>=0.115;实现时如遇版本差异查 Context7。)

新增应用级测试(`tests/test_chat_api.py` 追加或 `tests/test_retriever.py` 内):

```python
def test_app_starts_without_embedding_key(db_session_factory):
    from app.config import Settings
    from app.main import create_app
    settings = Settings(_env_file=None, **{
        "openai_base_url": "http://test/v1", "openai_api_key": "test-key",
        "model_name": "test-model",
        "database_url": Settings().test_database_url,
        "embedding_api_key": "",
    })
    app = create_app(settings=settings, model=object())  # 生产 runtime 路径
    assert app.state.chat_service is not None  # 缺 Key 也能启动
```

脆点:`model=object()` 无 bind_tools,仅验证启动路径不炸;若 create_app 对 model 有进一步调用再换 FakeStreamModel。

- [ ] **Step 7: 跑测试 + 全量回归**

```bash
uv run pytest tests/test_retriever.py tests/test_tools.py tests/test_executor.py -v
uv run pytest -q
```

预期:全绿(含 ch01/ch02 既有 136+)。

- [ ] **Step 8: Commit**

```bash
git add app/knowledge/retriever.py app/tools/business.py app/tools/executor.py \
        app/main.py tests/test_retriever.py tests/test_tools.py tests/test_executor.py \
        tests/test_chat_api.py
git commit -m "feat(ch03): query_faq 切换向量语义检索(契约不变),RetryableKnowledgeError 进 executor 重试分级"
```

---

### Task 8: knowledge_docs/ 演示知识文档(数据任务)

**Files:**
- Create: `knowledge_docs/退货政策.md`、`knowledge_docs/商品FAQ.md`、`knowledge_docs/售后手册.md`

**说明:** 纯数据任务,不走 TDD;验收方式 = T5 流水线真实跑通 + T9 评估召回命中 + 演示验收 1。文档内容同时是评估语料,标注(T9)引用其 `source_doc + chunk_index`,因此**内容定稿后不得再改**(改了标注即失效)。

- [ ] **Step 1: `knowledge_docs/商品FAQ.md`(type: faq;运费说明是验收 1 的靶子,必须含「满 99 元包邮」与「8 元」要点)**

```markdown
---
type: faq
---
# 商品FAQ

## 运费怎么算

单笔订单实付满 99 元包邮;未满收 8 元基础运费。偏远地区(新疆、西藏等)运费以结算页显示为准。

## 什么时候发货

工作日 16 点前付款的订单当天发出,之后次日发出;预售商品以商品页标注的发货时间为准。

## 保修多久

自营商品自签收之日起保修一年;人为损坏(进液、摔损、私自拆修)不在保修范围内。

## 会员有什么权益

会员享 95 折、每月 3 张免运费券、生日双倍积分;积分可在结算时抵扣,100 积分抵 1 元。

## 支持货到付款吗

部分自营商品支持货到付款,以商品页标识为准;货到付款订单不支持使用优惠券。
```

- [ ] **Step 2: `knowledge_docs/退货政策.md`(type: policy;含关键条款关键词「不支持」「必须」「不予」)**

```markdown
---
type: policy
---
# 退货政策

## 七天无理由退货

签收后 7 天内、商品完好的订单可申请七天无理由退货。退回商品必须保持原包装、配件、赠品齐全,吊牌未拆。运费险订单退货由保险公司承担首重运费。

## 不支持退货的情形

定制类、生鲜类商品不支持七天无理由退货。已激活的软件、虚拟充值类商品不予退货。贴身用品拆封后不支持退货。

## 退款时效

退货签收并验收无误后 1-3 个工作日原路退回;银行卡支付以银行到账时间为准。逾期未到账可联系人工客服核实。

## 退货流程

在「我的订单」找到对应订单,点击「申请售后」选择退货退款,按提示填写原因并提交;审核通过后按系统提供的地址寄回商品。
```

- [ ] **Step 3: `knowledge_docs/售后手册.md`(type: manual;含大表格,行数须使表格总长远超 500 字以演示按行切分+表头复制)**

```markdown
---
type: manual
---
# 售后手册

## 退换货流程

签收后 15 天内可申请换货,在订单页申请售后选择换货;换货商品发出后可在订单中查看新物流。退货审核周期为提交后 48 小时内。

## 运费标准表

| 地区 | 首重费用 | 续重费用 | 时效 |
|---|---|---|---|
| 北京 | 8 元 | 2 元/公斤 | 1-2 天 |
| 上海 | 8 元 | 2 元/公斤 | 1-2 天 |
| 广州 | 8 元 | 2 元/公斤 | 1-2 天 |
| 深圳 | 8 元 | 2 元/公斤 | 1-2 天 |
| 杭州 | 8 元 | 2 元/公斤 | 1-2 天 |
| 南京 | 8 元 | 3 元/公斤 | 2-3 天 |
| 苏州 | 8 元 | 3 元/公斤 | 2-3 天 |
| 成都 | 10 元 | 4 元/公斤 | 2-3 天 |
| 重庆 | 10 元 | 4 元/公斤 | 2-3 天 |
| 武汉 | 10 元 | 4 元/公斤 | 2-3 天 |
| 西安 | 10 元 | 4 元/公斤 | 2-4 天 |
| 天津 | 8 元 | 3 元/公斤 | 1-2 天 |
| 青岛 | 10 元 | 4 元/公斤 | 2-3 天 |
| 济南 | 10 元 | 4 元/公斤 | 2-3 天 |
| 大连 | 12 元 | 5 元/公斤 | 2-4 天 |
| 沈阳 | 12 元 | 5 元/公斤 | 2-4 天 |
| 哈尔滨 | 12 元 | 5 元/公斤 | 3-5 天 |
| 长春 | 12 元 | 5 元/公斤 | 3-5 天 |
| 昆明 | 12 元 | 5 元/公斤 | 3-5 天 |
| 贵阳 | 12 元 | 5 元/公斤 | 3-5 天 |
| 兰州 | 15 元 | 6 元/公斤 | 3-5 天 |
| 西宁 | 15 元 | 6 元/公斤 | 3-5 天 |
| 乌鲁木齐 | 20 元 | 8 元/公斤 | 5-7 天 |
| 拉萨 | 20 元 | 8 元/公斤 | 5-7 天 |
| 呼和浩特 | 15 元 | 6 元/公斤 | 3-5 天 |
| 银川 | 15 元 | 6 元/公斤 | 3-5 天 |
| 南宁 | 12 元 | 5 元/公斤 | 3-5 天 |
| 海口 | 15 元 | 6 元/公斤 | 3-5 天 |
| 福州 | 10 元 | 4 元/公斤 | 2-3 天 |
| 厦门 | 10 元 | 4 元/公斤 | 2-3 天 |

## 关键条款摘录

退货商品如有人为损坏痕迹,仓库有权拒收,相关损失不予承担。到付件必须先联系人工客服确认,擅自发到付件不予签收。退款原路退回,不得要求退至其他账户。
```

- [ ] **Step 4: 用 T5 流水线冒烟验证(数据任务的验证替代 TDD)**

需要 `.env` 已填 `EMBEDDING_API_KEY`(若用户尚未填,本步与 T9 真实运行一并等待,先继续做代码任务,在收尾统一验证):

```bash
docker compose up -d   # 确保 MySQL 在线
# 已有 ch02 数据卷的库首次升级:
docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch03-ddl.sql
uv run python -m app.jobs.ingest_docs
```

预期:退出码 0,输出「完成: 无 pending,指针完整,两库主键集合一致」。抽查:

```bash
docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp --default-character-set=utf8mb4 \
  -e "SELECT id, source_doc, chunk_index, questions, vectorize_status FROM knowledge_chunks;"
```

预期:退货政策/售后手册的块 `questions` 为章节标题、`category` 为上级路径;商品FAQ 的块 `questions` 为真实问法;售后手册运费表被切成多块且每块含表头(查 answer 字段)。

- [ ] **Step 5: Commit**

```bash
git add knowledge_docs/
git commit -m "feat(ch03): 演示知识文档三份(faq/policy/manual,含大表格与关键条款)"
```

---

### Task 9: evals 知识召回评估(评估集验证,替代数据类任务的 TDD)

**Files:**
- Create: `evals/knowledge_recall.jsonl`、`evals/run_knowledge_eval.py`
- Test: `tests/test_eval_metrics.py`(纯指标函数走 TDD;标注文件本身是数据,靠评估运行验证)

**Interfaces:**
- Consumes: `run_ingest`(T5)、`KnowledgeRetriever`(T7)、knowledge_docs(T8)
- Produces: `eval_metric` 纯函数族(见下);评估结果落 `evals/results/<时间戳>.json`;冻结阈值回写 `config.py` / `.env.example`

- [ ] **Step 1: 指标纯函数失败测试 `tests/test_eval_metrics.py`**

```python
from evals.run_knowledge_eval import (
    case_recall, choose_threshold, false_recall_rate, macro_average,
)


def test_case_recall():
    assert case_recall({("d", 1), ("d", 2)}, {("d", 1)}) == 1.0
    assert case_recall(set(), {("d", 1)}) == 0.0
    assert case_recall({("d", 1)}, {("d", 1), ("d", 2)}) == 0.5


def test_macro_average():
    assert macro_average([1.0, 0.5]) == 0.75


def test_false_recall_rate():
    assert false_recall_rate([True, False, False, False]) == 0.25
    assert false_recall_rate([]) == 0.0


def test_choose_threshold_prefers_feasible_then_recall_then_higher():
    # 负例分 0.8/0.6,正例相关块分 0.7:满足 FPR 约束的阈值只剩 > 0.8,召回为 0 也照选
    cases = [
        {"answerable": False, "relevant": set(), "hits": [("n1", 0.8)]},
        {"answerable": False, "relevant": set(), "hits": [("n2", 0.6)]},
        {"answerable": True, "relevant": {("d", 1)}, "hits": [(("d", 1), 0.7)]},
    ]
    t, cal_recall = choose_threshold(cases, max_fpr=0.10)
    assert t > 0.8
    assert cal_recall == 0.0


def test_choose_threshold_maximizes_recall_with_tie_break():
    cases = [
        {"answerable": False, "relevant": set(), "hits": [("n1", 0.3)]},
        {"answerable": True, "relevant": {("d", 1)}, "hits": [(("d", 1), 0.9)]},
        {"answerable": True, "relevant": {("d", 2)}, "hits": [(("d", 2), 0.6)]},
    ]
    t, cal_recall = choose_threshold(cases, max_fpr=0.10)
    assert 0.3 < t <= 0.6  # 两个正例都保住的最高阈值
    assert cal_recall == 1.0


def test_choose_threshold_no_feasible_exits():
    import pytest
    cases = [{"answerable": False, "relevant": set(), "hits": [("n1", 1.0)]}]
    with pytest.raises(SystemExit):
        choose_threshold(cases, max_fpr=0.10)  # 负例满分,任何阈值都误召回
```

`choose_threshold` 返回 `(threshold, calibration_recall)` 两元组;calibration 召回不达 0.90 的判定在 main 流程(spec §11.5),函数本身只做「可行域内最大召回、并列取高」。

- [ ] **Step 2: 实现 `evals/run_knowledge_eval.py`(全文)**

```python
"""知识检索评估(spec §11):真实硅基流动 embedding + 真实 Milvus Lite + 独立 MySQL 评估库。

用法:
  uv run python evals/run_knowledge_eval.py --dump-corpus   # 建评估语料并打印可标注块
  uv run python evals/run_knowledge_eval.py                 # 完整评估:校准冻结阈值 → test 验收
达标线:test 分片 Recall@5 宏平均 ≥ 0.90 且负例误召回率 ≤ 0.10,且 p_youfei 必须命中。
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.knowledge.embedding import build_embeddings
from app.knowledge.ingest import run_ingest
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.knowledge.retriever import KnowledgeRetriever
from app.models import KnowledgeChunk
from tests.dbfixtures import _split_statements

ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).parent / "knowledge_recall.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
EVAL_DB = "wayhelp_eval"
YOUFEI_CASE = "p_youfei"
YOUFEI_POINTS = ["满99元包邮", "8元"]  # 空白规范化后包含判定
MIN_RECALL = 0.90
MAX_FPR = 0.10
TOP_K = 5


def case_recall(returned: set, relevant: set) -> float:
    if not relevant:
        return 0.0
    return len(returned & relevant) / len(relevant)


def macro_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def false_recall_rate(returned_nonempty: list[bool]) -> float:
    return sum(1 for b in returned_nonempty if b) / len(returned_nonempty) if returned_nonempty else 0.0


def choose_threshold(cases: list[dict], max_fpr: float) -> tuple[float, float]:
    """只用 calibration 分片:先满足 FPR ≤ max_fpr,再最大化宏平均召回,并列取高阈值。
    返回 (threshold, calibration_recall);无可行阈值直接 SystemExit。"""
    candidates = {-1.0, 1.0}
    for c in cases:
        candidates.update(score for _, score in c["hits"])
    best_t, best_recall = None, -1.0
    for t in sorted(candidates):
        recalls, negs = [], []
        for c in cases:
            returned = {k for k, s in c["hits"] if s >= t}
            if c["answerable"]:
                recalls.append(case_recall(returned, c["relevant"]))
            else:
                negs.append(bool(returned))
        if false_recall_rate(negs) > max_fpr:
            continue
        recall = macro_average(recalls)
        if recall > best_recall or (recall == best_recall and (best_t is None or t > best_t)):
            best_t, best_recall = t, recall
    if best_t is None:
        raise SystemExit("[eval] 校准失败: 不存在满足负例约束的阈值")
    return best_t, best_recall


def _prepare_eval_db(settings: Settings):
    admin = make_engine(settings.test_admin_database_url)
    with admin.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {EVAL_DB} CHARACTER SET utf8mb4"))
        conn.commit()
    url = settings.test_database_url.replace("/wayhelp_test", f"/{EVAL_DB}")
    engine = make_engine(url)
    ddl = (ROOT / "db" / "init" / "03-ddl.sql").read_text(encoding="utf-8")
    with engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in ("knowledge_chunks", "qa_extraction_staging", "qa_mining_progress"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        for stmt in _split_statements(ddl):
            conn.execute(text(stmt))
        conn.commit()
    return engine


def _build_corpus(settings: Settings, engine, store, embed) -> dict[int, tuple[str, int]]:
    """独立评估库 + 临时 Milvus,只含固定 knowledge_docs 语料(不混入挖掘 QA)。"""
    sf = make_session_factory(engine)
    if run_ingest(settings, sf, embed, store, ROOT / "knowledge_docs") != 0:
        raise SystemExit("[eval] 语料建库失败")
    with sf() as s:
        rows = s.query(KnowledgeChunk).all()
        return {r.id: (r.source_doc, r.chunk_index) for r in rows}


def _load_cases(corpus_keys: set[tuple[str, int]]) -> list[dict]:
    cases = [json.loads(ln) for ln in CASES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case id 重复"
    pos = [c for c in cases if c["answerable"]]
    neg = [c for c in cases if not c["answerable"]]
    assert len(pos) >= 20 and len(neg) >= 20, "正负例各需 ≥ 20"
    for split in ("calibration", "test"):
        assert sum(1 for c in pos if c["split"] == split) >= 10
        assert sum(1 for c in neg if c["split"] == split) >= 10
    for c in cases:
        assert c["split"] in ("calibration", "test")
        if not c["answerable"]:
            assert not c["relevant_chunks"], "负例相关块必须为空"
        for ref in c["relevant_chunks"]:
            key = (ref["source_doc"], ref["chunk_index"])
            assert key in corpus_keys, f"标注引用不存在的块: {key}"
    youfei = [c for c in cases if c["id"] == YOUFEI_CASE]
    assert len(youfei) == 1 and youfei[0]["split"] == "test" and youfei[0]["answerable"]
    return cases


def main(argv: list[str]) -> int:
    settings = Settings()
    if not settings.has_embedding_key():
        print("[eval] embedding_api_key 未配置", file=sys.stderr)
        return 2
    embed = build_embeddings(settings)
    engine = _prepare_eval_db(settings)
    tmp = Path(tempfile.mkdtemp()) / "eval_milvus.db"
    store = MilvusKnowledgeStore(str(tmp), settings.embedding_dim)
    try:
        id_to_key = _build_corpus(settings, engine, store, embed)
        key_to_answer = {}
        sf = make_session_factory(engine)
        with sf() as s:
            for r in s.query(KnowledgeChunk).all():
                key_to_answer[(r.source_doc, r.chunk_index)] = r.answer
        if "--dump-corpus" in argv:
            RESULTS_DIR.mkdir(exist_ok=True)
            snapshot = [{"chunk_id": i, "source_doc": k[0], "chunk_index": k[1]}
                        for i, k in sorted(id_to_key.items())]
            (RESULTS_DIR / "corpus_snapshot.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            for row in snapshot:
                print(row)
            return 0
        cases = _load_cases(set(id_to_key.values()))
        retriever = KnowledgeRetriever(settings, embed=embed, store=store,
                                       session_factory=sf)
        for c in cases:  # 同一次检索链路,min_score=-1 拿原始分,阈值离线施加
            hits, _ = retriever.search(c["query"], min_score=-1.0)
            c["hits"] = [(id_to_key[h.chunk_id], h.score) for h in hits]
        calibration = [c for c in cases if c["split"] == "calibration"]
        threshold, cal_recall = choose_threshold(calibration, MAX_FPR)
        if cal_recall < MIN_RECALL:
            print(f"[eval] 校准失败: calibration 召回 {cal_recall:.3f} < {MIN_RECALL}",
                  file=sys.stderr)
            return 1
        test_cases = [c for c in cases if c["split"] == "test"]
        recalls, negs, per_case = [], [], []
        for c in test_cases:
            returned = {k for k, s in c["hits"] if s >= threshold}
            if c["answerable"]:
                r = case_recall(returned, c["relevant"])
                recalls.append(r)
            else:
                negs.append(bool(returned))
            per_case.append({"id": c["id"], "answerable": c["answerable"],
                             "returned": sorted(str(k) for k in returned),
                             "relevant": sorted(str((x["source_doc"], x["chunk_index"]))
                                                for x in c["relevant_chunks"])})
        recall = macro_average(recalls)
        fpr = false_recall_rate(negs)
        youfei = next(c for c in test_cases if c["id"] == YOUFEI_CASE)
        youfei_hit_keys = {k for k, s in youfei["hits"] if s >= threshold}
        youfei_ok = any(
            all(pt in "".join(key_to_answer[k].split()) for pt in YOUFEI_POINTS)
            for k in youfei_hit_keys if k in key_to_answer)
        ok = recall >= MIN_RECALL and fpr <= MAX_FPR and youfei_ok
        RESULTS_DIR.mkdir(exist_ok=True)
        out = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": settings.embedding_model, "threshold": threshold, "top_k": TOP_K,
            "test_recall_at_5": recall, "test_false_recall_rate": fpr,
            "youfei_hit": youfei_ok, "cases": per_case,
            "cases_file": CASES.name, "corpus": "knowledge_docs/",
        }
        path = RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] threshold={threshold:.3f} recall@5={recall:.3f} "
              f"fpr={fpr:.3f} youfei={youfei_ok} -> {path}")
        if not ok:
            print("[eval] 未达标", file=sys.stderr)
            return 1
        return 0
    finally:
        store.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

脆点:`test_database_url.replace("/wayhelp_test", ...)` 依赖默认库名;若用户改过 TEST_DATABASE_URL,replace 不生效——实现时加断言 `"/wayhelp_test" in url else raise`。

- [ ] **Step 3: 跑指标单测 `uv run pytest tests/test_eval_metrics.py -v` 全绿**

- [ ] **Step 4: 标注 `evals/knowledge_recall.jsonl`(数据任务;先 --dump-corpus 再写标注)**

格式(每行):

```json
{"id": "p01", "split": "calibration", "query": "退货多久内可以申请", "answerable": true, "relevant_chunks": [{"source_doc": "knowledge_docs/退货政策.md", "chunk_index": 1}], "answer_points": ["7天"]}
{"id": "p_youfei", "split": "test", "query": "邮费是多少", "answerable": true, "relevant_chunks": [{"source_doc": "knowledge_docs/商品FAQ.md", "chunk_index": 1}], "answer_points": ["满99元包邮", "8元"]}
{"id": "n01", "split": "calibration", "query": "你们支持微信支付吗", "answerable": false, "relevant_chunks": [], "answer_points": []}
```

规则(spec §11):≥20 正例 + ≥20 负例;正/负各半分 calibration/test(每片 ≥10+10);正例覆盖政策/FAQ/手册与换说法问法;负例为店铺业务范围内但知识库无答案的问题(支付方式、线下门店、客服电话、合作加盟等);同一问句不得跨分片重复;`p_youfei` 固定 test 分片;`relevant_chunks` 引用以 `--dump-corpus` 输出的 `corpus_snapshot.json` 为准(chunk_index 对不上会在加载断言处直接报错)。「邮费是多少」相关块以实际语料为准(商品FAQ 运费块)。

- [ ] **Step 5: 真实运行(需 .env 已填 EMBEDDING_API_KEY;校准冻结阈值)**

```bash
uv run python evals/run_knowledge_eval.py --dump-corpus   # 先出语料快照,核对/修正标注
uv run python evals/run_knowledge_eval.py                 # 完整评估
```

退出码 0 且输出 recall@5/fpr/youfei=True。若校准反复不达标,先查语料与标注(换说法覆盖),再考虑阈值;**不得根据 test 分片结果回调阈值**(spec §11.5)。达标后把冻结阈值写回 `config.py` 的 `knowledge_min_score` 默认值与 `.env.example`,随本任务一起提交。

- [ ] **Step 6: Commit**

```bash
git add evals/knowledge_recall.jsonl evals/run_knowledge_eval.py tests/test_eval_metrics.py \
        evals/results/ app/config.py .env.example
git commit -m "feat(ch03): 知识召回评估集(20+20 分片标注) + 阈值校准冻结流程,test 达标"
```

---

### Task 10: README + 全量回归 + 演示验收

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README 追加 ch03 章节**

内容要点(照 spec §8/§9/§11 写):

```markdown
## Ch03 知识库与向量语义检索

- 知识文档建库: `uv run python -m app.jobs.ingest_docs [knowledge_docs/]`
- 对话挖知识: `uv run python -m app.jobs.mine_qa`
- 召回评估: `uv run python evals/run_knowledge_eval.py`
- 新增环境变量: EMBEDDING_BASE_URL / EMBEDDING_API_KEY(硅基流动,必填后两个命令才可运行)/
  EMBEDDING_MODEL(BAAI/bge-m3)/ MILVUS_URI(./data/milvus_lite.db)等,见 .env.example

### 数据库初始化与升级
- 全新环境: `docker compose up -d` 首启自动执行 db/init 全部 DDL(含 ch03)
- 已有 ch02 数据卷的升级(不得删卷):
  `docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch03-ddl.sql`

### 运行约束
- Milvus Lite 本地库独占打开: 跑建库/挖矿/评估前先停在线服务(uvicorn),完成后再起;服务单 worker
- 不并行运行两个知识任务;缺 EMBEDDING_API_KEY 时在线检索降级为「知识检索未配置」,服务正常启动
```

- [ ] **Step 2: 全量回归**

```bash
uv run pytest -q
```

全绿。预期总数:136(ch02 收官)+ 本章新增(T1~T9 约 45 个)。

- [ ] **Step 3: 演示验收(与用户一起,三条对应 spec §11 演示验收)**

1. 起服 `uv run uvicorn app.main:create_app --factory` → 浏览器问「邮费是多少」→ 召回运费说明并答对(ch02 同题漏召回对照);SQL 抽查落库 tool 行为 query_faq + envelope
2. 停服 → 跑 `ingest_docs` 途中 Ctrl+C/kill → 重跑 → `SELECT vectorize_status, COUNT(*)` 全 done;指针完整;`all_ids` 与 MySQL 主键集合一致(可用 `uv run python -c` 小脚本核对)
3. `mine_qa` 至少含一个零 QA 会话;成功完成后重跑,日志/调用计数确认不再请求 LLM(看 `[mining]` 输出与进度表行数不变)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(ch03): README 知识库章节(命令/升级路径/独占约束/降级行为)"
```

---

## Self-Review 记录(计划落盘前已跑)

- **Spec 覆盖**: §4.1 来源字段/唯一键→T1+T2;§4.3 进度表→T1+T2+T6;§4.4 集合契约→T4;§5 切分→T3;§6 双写→T5;§7 挖矿→T6;§8 检索/降级/异常分级→T7;§9 配置→T1(阈值冻结回写→T9);§10 错误表→各任务错误路径;§11 测试矩阵→T3-T9 测试 + T9 评估 + T10 演示;§12 依赖与核对→T1 Step3 + 各脆点标注。无遗漏。
- **占位扫描**: 无 TBD/TODO;所有测试与实现代码为全文。T9 标注文件是数据任务,给了格式+示例行+规则+加载断言兜底。
- **类型一致性**: `vectorize_pending(settings, session_factory, embed, store)` 在 T5 定义、T6/T9 复用一致;`MilvusKnowledgeStore(uri, dim)` 各任务一致;`KnowledgeRetriever.search` 返回 `(hits, note)` 与 T7 工具/测试/T9 调用一致;`build_tools` 三参数签名 T7 定义与测试一致;`Chunk` 字段与 T5 `_same_chunk` 比较字段一致。
- **遗留脆点**(执行者按锁定版本实测,commit message 说明):pymilvus hit 访问方式/describe_collection 结构/query filter 语法/close 重开;OpenAIEmbeddings 超时参数名与属性名;openai 异常构造签名;FastAPI lifespan;eval 库名 replace 断言。
