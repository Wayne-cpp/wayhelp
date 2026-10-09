# ch09 可观测性与数据飞轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 Langfuse 自托管可观测性(全链路 trace + 按意图成本统计),升级正式版 evidence_confidence 置信闸,三入口落池带召回快照,飞轮流水线(标准化→查重→待审→审核写回知识库),评估流水线定时化并落 eval_runs 看趋势,/review 审核后台页。

**Architecture:** Langfuse 经 LangChain CallbackHandler 按请求注入 astream config(检索/工具/MCP 在固定边界补 SDK span),trace_id/intent 随图 state 持久化支撑 resume 续传;落池快照复用 state.retrieval_result,👎 反馈经 SSE turn_committed 帧锚定 assistant_message_id + checkpoint 精确回捞;飞轮是 lifespan 托管的单 worker(asyncio.Event 唤醒 + DB 退避);审核写回走「冻结事务 → 写入中 → 向量化 → CAS 通过」并与全部知识库写入口共用一把作业锁;评估沿用 JobRunner + 报告发布回调幂等落 eval_runs。

**Tech Stack:** FastAPI / SQLAlchemy 2.x 同步 + PyMySQL / LangGraph 1.x + langchain-core 1.6 / langfuse Python SDK v3(新增依赖)/ Milvus Lite / 硅基流动重排 / deepseek。

**Spec:** `docs/superpowers/specs/2026-10-08-ch09-observability-flywheel-design.md`(GPT 复审修订版,已提交;计划以它为唯一准绳)

## Global Constraints

- **Context7 先行**:凡涉及 langfuse / LangGraph / FastAPI / SQLAlchemy 具体 API 写法,先用 Context7 MCP 查最新文档再动手,禁止凭记忆写(langfuse 关键 API 已查:SDK v3 `from langfuse.langchain import CallbackHandler`,`CallbackHandler(public_key=..., trace_context={"trace_id": ...})`;config metadata 键 `langfuse_session_id/langfuse_user_id/langfuse_trace_name/langfuse_tags`;自定义 span `get_client().start_as_current_observation(as_type="span"|"retriever"|"tool", ...)`;`langfuse.api.trace.list(...)`;v3 自托管需 web+worker+ClickHouse+Postgres+Redis/Valkey+MinIO)。
- **DDL 双写**:`sql/ch09-ddl.sql` 与 `db/init/07-ddl.sql` 逐字节一致(test_ddl_sync 钉);中文 ENUM 前 `SET NAMES utf8mb4`。
- **messages 表契约不破**:只收 user/assistant 行,不存检索快照/tool 结果。
- **单 Uvicorn worker**:内存态(Event/锁/注册表)安全;不引 APScheduler。
- **token 估算唯一尺** `app/services/token_budget.py`;本计划不涉及新估算。
- **代理红线**:Langfuse SDK/MinIO 访问 localhost 需 `no_proxy=127.0.0.1,localhost`;curl 本机服务加 `--noproxy '*'`。
- **不凭拍脑袋定阈值**:置信闸权重/阈值由 `--calibrate-evidence` 校准产出 `evals/calibration/evidence_confidence.json`(纳入版本控制)后冻结进 config 默认值。
- **审计/观测不拦主路**:Langfuse 任何异常只记日志,绝不影响聊天主链路;落池与消息同事务(ch04 语义)保持。
- **错误契约**:HTTP 错误一律 `{"error":{"code","message"}}`;写冲突 409,归属失败 404 不泄露。
- **测试库基建**:DB 测试用 `tests/dbfixtures.py` 的 `db_session_factory` fixture(需 Docker MySQL 在线);Conventional Commits。
- **前端例外**:chat.html / review.html / rag-eval.html 走 Vibe Coding,不写单测,靠 test_chat_page 类字符串断言 + 人工点验。

---

### Task 1: ch09 DDL + ORM 模型 + 启动校验

spec §6 的散文契约落成正式 DDL——用户提供的 `sql/ch09-ddl.sql` 已落后 spec(缺 chat_feedback 整表、review_queue「写入中」与写回列、lcq 六新列、eval_runs 版本列),本任务**整文件重写**。

**Files:**
- Overwrite: `sql/ch09-ddl.sql`
- Create: `db/init/07-ddl.sql`(与上逐字节一致)
- Modify: `tests/test_ddl_sync.py:8`(映射加 `("07","ch09")`)
- Modify: `tests/dbfixtures.py:15-26`(DDL_PATHS 加 07;TABLES 加 chat_feedback/low_confidence_questions/review_queue/eval_runs,先子后父)
- Modify: `app/models.py`(LowConfidenceQuestion 补 8 列;新增 ReviewQueue/ChatFeedback/EvalRun)
- Modify: `app/db.py`(check_ch09_tables)
- Modify: `app/main.py:70-72`(挂载校验)
- Test: `tests/test_ch09_schema.py`(新建)

**Interfaces:**
- Produces(后续任务全部依赖):
  - `ReviewQueue`:`id/normalized_question/ai_suggested_answer/occurrence_count/review_status("待审"|"写入中"|"通过"|"驳回")/approved_answer/approved_at/knowledge_chunk_ids(list[int]|None)/last_write_error/created_at/updated_at`
  - `ChatFeedback`:`id/conversation_id/assistant_message_id/turn_message_id/sentiment("up"|"down")/low_confidence_question_id/created_at`,唯一键 `(conversation_id, assistant_message_id)` name=`uk_cf_conv_msg`
  - `EvalRun`:`id/run_id(unique)/triggered_by("定时"|"手动")/dataset_size/corpus_mode/corpus_version/dataset_version/metrics(dict)/created_at`
  - `LowConfidenceQuestion` 新增:`retrieved_chunks(list|None)/resolved_question(str|None)/turn_message_id(int|None)/matched_review_id(int|None FK→review_queue.id ondelete SET NULL)/process_status("pending"|"processed"|"failed")/attempt_count(int)/next_attempt_at(datetime|None)/last_error(str|None)`
  - `app.db.check_ch09_tables(engine) -> None`(缺表/缺列 RuntimeError 附升级命令)

- [ ] **Step 1: 写失败测试** `tests/test_ch09_schema.py`

```python
"""ch09 DDL/模型契约:表列对齐 DDL、启动校验缺表报错。"""
import pytest
from sqlalchemy import text

from app.db import check_ch09_tables
from app.models import ChatFeedback, EvalRun, LowConfidenceQuestion, ReviewQueue


def test_models_have_ch09_columns():
    lcq = {c.name for c in LowConfidenceQuestion.__table__.columns}
    assert {"retrieved_chunks", "resolved_question", "turn_message_id",
            "matched_review_id", "process_status", "attempt_count",
            "next_attempt_at", "last_error"} <= lcq
    rq = {c.name for c in ReviewQueue.__table__.columns}
    assert {"normalized_question", "ai_suggested_answer", "occurrence_count",
            "review_status", "approved_answer", "approved_at",
            "knowledge_chunk_ids", "last_write_error"} <= rq
    assert {c.name for c in ChatFeedback.__table__.columns} >= {
        "conversation_id", "assistant_message_id", "turn_message_id",
        "sentiment", "low_confidence_question_id"}
    assert {c.name for c in EvalRun.__table__.columns} >= {
        "run_id", "triggered_by", "dataset_size", "corpus_mode",
        "corpus_version", "dataset_version", "metrics"}


def test_check_ch09_tables_passes(db_engine):
    check_ch09_tables(db_engine)  # 不抛即过


def test_check_ch09_tables_fails_when_missing(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("DROP TABLE chat_feedback"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch09"):
            check_ch09_tables(db_engine)
    finally:
        from pathlib import Path
        ddl = (Path(__file__).resolve().parent.parent
               / "db" / "init" / "07-ddl.sql").read_text(encoding="utf-8")
        import tests.dbfixtures as fx
        with db_engine.connect() as conn:
            for stmt in fx._split_statements(ddl):  # 重放 07 恢复现场
                if "chat_feedback" in stmt:
                    conn.execute(text(stmt))
            conn.commit()


def test_review_status_enum_values(db_session_factory):
    with db_session_factory() as s:
        s.add(ReviewQueue(normalized_question="如何申请开发票?"))
        s.commit()
        row = s.query(ReviewQueue).first()
        assert row.review_status == "待审"
        assert row.occurrence_count == 1
```

dbfixtures 的 fixture 在 `tests/conftest.py` 已全局引入(照现有 DB 测试用法,无需 import);若该文件未自动加载,在测试文件顶部 `from tests.dbfixtures import db_engine, db_session_factory  # noqa` 不可行——正确做法:conftest.py 里已有 `pytest_plugins` 或同目录 fixture 自动可见,执行者先 `grep db_session_factory tests/conftest.py tests/*.py | head -5` 确认现有测试如何拿到该 fixture,照抄同款。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch09_schema.py -x -q`
Expected: FAIL(import ReviewQueue 失败 / check_ch09_tables 不存在)

- [ ] **Step 3: 重写 `sql/ch09-ddl.sql` 全量内容**

```sql
-- =============================================================
-- ch09 · 可观测性与数据飞轮 · 建表 DDL
-- 本章新建:review_queue(去重后的知识缺口待审队列,含写回状态机)
--          chat_feedback(👍👎 持久幂等;唯一键 (conversation_id, assistant_message_id))
--          eval_runs(评估轮次,run_id 幂等,语料/评估集内容哈希版本)
-- 并给 ch04 的 low_confidence_questions 加八列:
--   召回快照/指代消解问题/轮次锚点/归并落点/飞轮处理状态与退避
-- 建表顺序:review_queue → ALTER low_confidence_questions → chat_feedback(外键依赖)
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 待审队列:一行 = 一个去重后的知识缺口;查重命中就累加 occurrence_count,不新建行
-- 写回状态机:待审 →(approve 冻结事务)→ 写入中 →(向量化+双库确认)→ 通过
--   写入中即冻结核准答案、可能已部分发布,只允许重试不许驳回/改答案
CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_review_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加,越高越该优先补',
  review_status       ENUM('待审','写入中','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '审核状态;写入中=已冻结核准答案且知识块已建,待向量化确认',
  approved_answer     TEXT            NULL                    COMMENT '通过时冻结的核准答案;写入中/通过必须有值',
  approved_at         DATETIME        NULL                    COMMENT '最终转通过时间;只在完成通过后设置',
  knowledge_chunk_ids JSON            NULL                    COMMENT '本缺口写入的 knowledge_chunks id 列表(恢复索引,可按 source_doc=review:<id> 重建)',
  last_write_error    TEXT            NULL                    COMMENT '写入中阶段最近一次失败(截断),供重试排查',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次入队时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

-- 评估轮次:一行 = 评估流水线跑完的一轮;run_id 幂等,版本不同趋势不连线
CREATE TABLE eval_runs (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  run_id           VARCHAR(64)     NOT NULL                COMMENT '评估报告 meta.run_id,幂等键',
  triggered_by     ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '这轮怎么起的',
  dataset_size     INT UNSIGNED    NOT NULL                COMMENT '本轮 test 条数',
  corpus_mode      VARCHAR(32)     NOT NULL DEFAULT 'knowledge_docs_baseline' COMMENT '语料口径;固定基线,不混线上补库',
  corpus_version   CHAR(64)        NOT NULL                COMMENT '语料内容 SHA-256(相对路径排序汇总)',
  dataset_version  CHAR(64)        NOT NULL                COMMENT '评估集文件内容 SHA-256',
  metrics          JSON            NOT NULL                COMMENT '{online_strategy,recall_at_10,mrr,faithfulness,coverage,sr10,quality_passed,evidence_confidence_version};缺失指标存 null 不用 0 伪造',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '跑完落表时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_run_id (run_id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

-- 原话流水:召回快照 + 归并落点 + 飞轮处理状态
ALTER TABLE low_confidence_questions
  ADD COLUMN retrieved_chunks  JSON            NULL COMMENT '落池时的召回片段快照:Top 几条原文与得分;没走检索为 NULL' AFTER reason,
  ADD COLUMN resolved_question TEXT            NULL COMMENT '指代消解后的问题,飞轮标准化主输入' AFTER retrieved_chunks,
  ADD COLUMN turn_message_id   BIGINT UNSIGNED NULL COMMENT '本轮用户消息行 id(稳定轮次锚点;历史/非会话来源可 NULL,不挂外键)' AFTER resolved_question,
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '查重后归并到的缺口,指向 review_queue.id' AFTER turn_message_id,
  ADD COLUMN process_status    ENUM('pending','processed','failed') NOT NULL DEFAULT 'pending' COMMENT '飞轮处理状态;processed 必有 matched_review_id' AFTER matched_review_id,
  ADD COLUMN attempt_count     INT UNSIGNED    NOT NULL DEFAULT 0 COMMENT '已尝试次数,到 FLYWHEEL_MAX_ATTEMPTS 转 failed' AFTER process_status,
  ADD COLUMN next_attempt_at   DATETIME        NULL COMMENT '下次可自动重试时间(DB 钟);NULL=立即可处理' AFTER attempt_count,
  ADD COLUMN last_error        TEXT            NULL COMMENT '最近一次失败(截断)' AFTER next_attempt_at,
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD KEY idx_process_due (process_status, next_attempt_at, id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id) ON DELETE SET NULL;

-- 反馈持久幂等:两个 sentiment 都落;down 与 lcq 同事务创建并回指
CREATE TABLE chat_feedback (
  id                         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id            BIGINT UNSIGNED NOT NULL COMMENT '所属会话',
  assistant_message_id       BIGINT UNSIGNED NOT NULL COMMENT '被反馈的最终回答行(账本校验过)',
  turn_message_id            BIGINT UNSIGNED NOT NULL COMMENT '该轮用户消息行',
  sentiment                  ENUM('up','down') NOT NULL COMMENT '👍 / 👎',
  low_confidence_question_id BIGINT UNSIGNED NULL COMMENT 'down 时同事务创建的 lcq 行;up 为 NULL',
  created_at                 DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次反馈时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_cf_conv_msg (conversation_id, assistant_message_id),
  KEY idx_cf_lcq (low_confidence_question_id),
  CONSTRAINT fk_cf_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id),
  CONSTRAINT fk_cf_assistant_msg FOREIGN KEY (assistant_message_id) REFERENCES messages (id),
  CONSTRAINT fk_cf_turn_msg FOREIGN KEY (turn_message_id) REFERENCES messages (id),
  CONSTRAINT fk_cf_lcq FOREIGN KEY (low_confidence_question_id) REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='聊天反馈账本:幂等唯一键不含 sentiment,反向冲突 409';
```

- [ ] **Step 4: 复制到 db/init 并接测试基建**

```bash
cp sql/ch09-ddl.sql db/init/07-ddl.sql
```

`tests/test_ddl_sync.py:8` 改为:

```python
DDL_PAIRS = [("01", "ch02"), ("03", "ch03"), ("04", "ch04"), ("05", "ch07"), ("06", "ch08"), ("07", "ch09")]
```

`tests/dbfixtures.py`:DDL_PATHS 列表末尾追加一行 `Path(__file__).resolve().parent.parent / "db" / "init" / "07-ddl.sql",`;TABLES 改为(chat_feedback 引用 lcq/messages,lcq 引用 review_queue,先子后父):

```python
TABLES = ("chat_feedback", "tool_audit_logs", "tool_write_idempotency",
          "faith_cases", "low_confidence_questions", "review_queue", "eval_runs",
          "knowledge_chunks",
          "qa_extraction_staging", "qa_mining_progress",
          "messages", "conversation_summaries", "tickets", "faq",
          "conversations")  # 先子后父
```

- [ ] **Step 5: models.py 加列与新模型**

`LowConfidenceQuestion` 类体内 `created_at` 之前追加:

```python
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    resolved_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_message_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("review_queue.id", ondelete="SET NULL"),
        nullable=True)
    process_status: Mapped[str] = mapped_column(
        Enum("pending", "processed", "failed", name="lcq_process_status"),
        default="pending")
    attempt_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
```

文件末尾追加三个模型(与既有类同风格):

```python
class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), default=1)
    review_status: Mapped[str] = mapped_column(
        Enum("待审", "写入中", "通过", "驳回", name="review_status"), default="待审")
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    knowledge_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    last_write_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())


class ChatFeedback(Base):
    __tablename__ = "chat_feedback"
    __table_args__ = (UniqueConstraint("conversation_id", "assistant_message_id",
                                       name="uk_cf_conv_msg"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"))
    assistant_message_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("messages.id"))
    turn_message_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("messages.id"))
    sentiment: Mapped[str] = mapped_column(Enum("up", "down", name="fb_sentiment"))
    low_confidence_question_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("low_confidence_questions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    triggered_by: Mapped[str] = mapped_column(
        Enum("定时", "手动", name="eval_trigger"), default="定时")
    dataset_size: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    corpus_mode: Mapped[str] = mapped_column(String(32), default="knowledge_docs_baseline")
    corpus_version: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 6: db.py 加 check_ch09_tables 并挂启动**

`app/db.py` 末尾追加:

```python
_CH09_TABLES = ("review_queue", "chat_feedback", "eval_runs")
_CH09_LCQ_COLS = {"retrieved_chunks", "resolved_question", "turn_message_id",
                  "matched_review_id", "process_status", "attempt_count",
                  "next_attempt_at", "last_error"}


def check_ch09_tables(engine) -> None:
    """缺 ch09 表/列启动失败并提示升级命令。"""
    with engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SHOW TABLES"))}
        cols = {r[0] for r in conn.execute(
            text("SHOW COLUMNS FROM low_confidence_questions"))}
    missing = [t for t in _CH09_TABLES if t not in tables]
    if missing or not _CH09_LCQ_COLS <= cols:
        raise RuntimeError(
            f"缺少 ch09 表结构(缺表 {missing},lcq 缺列 "
            f"{sorted(_CH09_LCQ_COLS - cols)}):请先执行 "
            f"mysql wayhelp < sql/ch09-ddl.sql")
```

`app/main.py:18` import 加 `check_ch09_tables`;`app/main.py:72` 之后(`check_ch08_tables(engine)` 下一行)加 `check_ch09_tables(engine)  # 缺 ch09 表/列同样启动失败,提示升级命令`。

- [ ] **Step 7: 跑测试确认通过**

Run: `uv run pytest tests/test_ch09_schema.py tests/test_ddl_sync.py -x -q`
Expected: 全 PASS(dbfixtures 重放 01/03/04/05/06/07)

- [ ] **Step 8: 回归 + Commit**

Run: `uv run pytest -x -q`(全量,617+ 旧测试不能破)
Expected: PASS

```bash
git add sql/ch09-ddl.sql db/init/07-ddl.sql tests/test_ddl_sync.py tests/dbfixtures.py app/models.py app/db.py app/main.py tests/test_ch09_schema.py
git commit -m "feat(ch09): DDL——review_queue/chat_feedback/eval_runs + lcq 快照与飞轮状态列"
```

---

### Task 2: Langfuse 依赖 + compose 服务组 + 配置项

**Files:**
- Modify: `pyproject.toml`(加 langfuse 依赖)
- Modify: `docker-compose.yml`(并入 Langfuse v3 服务组)
- Modify: `app/config.py`(ch09 Langfuse 配置)
- Modify: `.env.example`(ch09 节块开头:Langfuse 段)
- Test: `tests/test_langfuse_config.py`(新建)

**Interfaces:**
- Produces:
  - `Settings.langfuse_enabled: bool = False`、`langfuse_host: str = "http://localhost:3000"`、`langfuse_public_key: str = ""`、`langfuse_secret_key: str = ""`
  - `Settings.has_langfuse_key() -> bool`
  - compose 服务:langfuse-web(127.0.0.1:3000)/langfuse-worker/clickhouse/postgres(langfuse 专用)/valkey/minio(+一次性 init 建 bucket)

- [ ] **Step 1: Context7 + 官方 compose 取证(强制)**

用 Context7 查 `/langfuse/langfuse-python` 确认 SDK v3 最新次版本号与 langchain-core v1 兼容声明;用 FetchURL 拉官方 `https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml` 作为服务定义底本。**版本全部钉精确 tag(不用 latest)**。

- [ ] **Step 2: 写失败测试** `tests/test_langfuse_config.py`

```python
"""ch09 Langfuse 配置契约:默认关闭;密钥齐才视为可用。"""
from app.config import Settings


def test_langfuse_disabled_by_default():
    s = Settings(openai_base_url="http://x", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d")
    assert s.langfuse_enabled is False
    assert s.has_langfuse_key() is False


def test_langfuse_key_detection():
    s = Settings(openai_base_url="http://x", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d",
                 langfuse_enabled=True, langfuse_public_key="pk-lf-x",
                 langfuse_secret_key="sk-lf-x")
    assert s.has_langfuse_key() is True
    assert s.langfuse_host == "http://localhost:3000"


def test_compose_has_langfuse_stack():
    text = open("docker-compose.yml", encoding="utf-8").read()
    for svc in ("langfuse-web:", "langfuse-worker:", "clickhouse:",
                "langfuse-postgres:", "valkey:", "minio:"):
        assert svc in text, svc
    assert "127.0.0.1:3000:3000" in text
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_langfuse_config.py -x -q`
Expected: FAIL(Settings 无 langfuse_enabled)

- [ ] **Step 4: 实现**

`pyproject.toml` dependencies 加 `"langfuse>=3,<4",`(Step 1 核实后的 v3 下限),`uv sync` 落锁。

`app/config.py` 在 ch08 段后追加:

```python
    # ch09 可观测性(spec §5.1):未配齐密钥 = 完全不挂回调,系统行为不变
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
```

`has_embedding_key` 旁加:

```python
    def has_langfuse_key(self) -> bool:
        return bool(self.langfuse_public_key.strip()
                    and self.langfuse_secret_key.strip())
```

`docker-compose.yml` 在 mysql 服务后追加(以 Step 1 官方 compose 为底本改写,服务名加 langfuse 前缀,端口绑 127.0.0.1,卷名 langfuse-* 前缀,全部 healthcheck,web depends_on worker 依赖链照官方;MinIO 加一个 `minio-init` one-shot 服务用 `mc` 建 `langfuse-events`/`langfuse-media` 两个 bucket 并设 `depends_on: minio: condition: service_completed_successfully`;worker 的 `LANGFUSE_S3_EVENT_UPLOAD_*` 系列环境变量指向 minio 服务名)。镜像 tag 钉 Step 1 查到的版本。卷:`langfuse-clickhouse-data`、`langfuse-postgres-data`、`langfuse-minio-data`。

`.env.example` 末尾追加:

```
# ── ch09 可观测性与数据飞轮 ──
# Langfuse 自托管(compose 已并入 langfuse 服务组);密钥在 http://localhost:3000 首次建站后创建
# LANGFUSE_ENABLED=false
# LANGFUSE_HOST=http://localhost:3000
# LANGFUSE_PUBLIC_KEY=
# LANGFUSE_SECRET_KEY=
# 注意:shell 有代理变量时,访问本机 Langfuse 需 no_proxy=127.0.0.1,localhost(同 milvus-lite 红线)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_langfuse_config.py -x -q`
Expected: PASS

- [ ] **Step 6: compose 自检 + Commit**

Run: `docker compose config -q`(语法校验,不启动)
Expected: 无输出退出 0(镜像拉取与起栈留到验收阶段,避免 CI 无网卡死)

```bash
git add pyproject.toml uv.lock docker-compose.yml app/config.py .env.example tests/test_langfuse_config.py
git commit -m "feat(ch09): Langfuse v3 服务组并入 compose + 依赖与配置项"
```

---

### Task 3: Langfuse trace 工厂 + ChatService 挂回调 + intent 元数据 + resume 续传

**Files:**
- Create: `app/services/langfuse_tracing.py`
- Modify: `app/services/chat_service.py:214-257`(stream 注入 config;resume 读 checkpoint 重建)
- Modify: `app/graph/state.py`(加 `trace_id` 字段 + new_turn_state 重置)
- Modify: `app/graph/nodes.py`(classify_intent 回写 intent 元数据;log 节点盖章 trace_id)
- Modify: `app/main.py`(lifespan 关闭时 flush)
- Test: `tests/test_langfuse_tracing.py`(新建)

**Interfaces:**
- Consumes: Task 2 的 settings 字段。
- Produces:
  - `langfuse_tracing.tracing_enabled(settings) -> bool`
  - `langfuse_tracing.new_trace_id() -> str`(uuid4 hex)
  - `langfuse_tracing.build_trace_config(settings, *, conversation_id: str, user_id: str, trace_id: str, intent: str | None = None, intent_confidence: float | None = None) -> dict` — 未启用返回 `{}`;启用返回 `{"callbacks": [handler], "metadata": {...}, "tags": [...]}`
  - `langfuse_tracing.tag_intent(intent: str, confidence: float | None) -> None` — 节点内调用;未启用/无活动 trace 一律 no-op 不抛
  - `langfuse_tracing.flush_langfuse() -> None`
  - state 字段 `trace_id: str | None`(新轮 None,log 盖章,resume 从 checkpoint 读出)

- [ ] **Step 1: 写失败测试** `tests/test_langfuse_tracing.py`

```python
"""ch09 Langfuse 挂载:disabled 零侵入;enabled 注入回调与元数据;resume 意图续传。"""
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.services import langfuse_tracing as lt


def _settings(**kw):
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d")
    base.update(kw)
    return Settings(**base)


def test_disabled_returns_empty_config():
    assert lt.build_trace_config(_settings(), conversation_id="1",
                                 user_id="u", trace_id="t") == {}
    lt.tag_intent("退款退货", 0.9)   # disabled 也不得抛
    lt.flush_langfuse()              # 同上


def test_enabled_builds_handler_and_metadata():
    cfg = lt.build_trace_config(
        _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-t",
                  langfuse_secret_key="sk-lf-t"),
        conversation_id="42", user_id="u1", trace_id="ab" * 16,
        intent="退款退货", intent_confidence=0.92)
    assert len(cfg["callbacks"]) == 1
    md = cfg["metadata"]
    assert md["langfuse_session_id"] == "42"
    assert md["langfuse_user_id"] == "u1"
    assert md["langfuse_trace_name"] == "chat_turn"
    assert md["intent"] == "退款退货" and md["intent_confidence"] == 0.92
    assert "intent:退款退货" in cfg["tags"]


def test_new_turn_state_resets_trace_id():
    from app.graph.state import new_turn_state
    st = new_turn_state("你好", user_db_id=7)
    assert st["trace_id"] is None
    assert st["turn_message_id"] == 7
    assert st["final_assistant_message_id"] is None
```

enabled 用例构造 CallbackHandler 需要 langfuse 包已安装(Task 2);SDK 构造不发起网络。若 SDK 在密钥无效时构造即抛,改用 monkeypatch 替换 `langfuse.langchain.CallbackHandler` 为假类再断言注入形状——执行者按实测选一种,断言语义不变。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_langfuse_tracing.py -x -q`
Expected: FAIL(模块不存在 / state 无字段)

- [ ] **Step 3: 实现 langfuse_tracing.py**

```python
"""ch09:Langfuse(SDK v3)接入。未启用时全部 no-op;任何异常只记日志不拦主路。

SDK 从环境变量读密钥(get_client):enabled 时由 ensure_env 把 settings 值写入
os.environ(进程内,不回写 .env)。trace 属性经 config metadata 键传入
(langfuse_session_id/langfuse_user_id/langfuse_trace_name/langfuse_tags),
resume 续传经 CallbackHandler(trace_context={"trace_id": ...})。
"""

import logging
import os
import uuid
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


def tracing_enabled(settings: Settings) -> bool:
    return settings.langfuse_enabled and settings.has_langfuse_key()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def ensure_env(settings: Settings) -> None:
    """把 settings 映射进 SDK 读的环境变量(幂等,不覆盖已存在的外部设置)。"""
    if not tracing_enabled(settings):
        return
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key.strip())
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key.strip())
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_host.rstrip("/"))


def build_trace_config(settings: Settings, *, conversation_id: str, user_id: str,
                       trace_id: str, intent: str | None = None,
                       intent_confidence: float | None = None) -> dict:
    """每次图调用注入一次的请求级回调配置;未启用 → {}。"""
    if not tracing_enabled(settings):
        return {}
    ensure_env(settings)
    from langfuse.langchain import CallbackHandler  # 延迟 import:disabled 零开销
    handler = CallbackHandler(public_key=settings.langfuse_public_key.strip(),
                              trace_context={"trace_id": trace_id})
    metadata: dict[str, Any] = {
        "langfuse_session_id": str(conversation_id),
        "langfuse_user_id": user_id,
        "langfuse_trace_name": "chat_turn",
        "conversation_id": str(conversation_id),
        "turn_id": trace_id,
    }
    tags = ["chat"]
    if intent is not None:
        metadata["intent"] = intent
        if intent_confidence is not None:
            metadata["intent_confidence"] = intent_confidence
        tags.append(f"intent:{intent}")
    return {"callbacks": [handler], "metadata": metadata, "tags": tags}


def tag_intent(intent: str, confidence: float | None) -> None:
    """classify_intent 出结果后回写当前 trace;无活动 trace/未启用 → no-op。"""
    try:
        from langfuse import get_client
        client = get_client()
        md: dict[str, Any] = {"intent": intent}
        if confidence is not None:
            md["intent_confidence"] = confidence
        client.update_current_trace(tags=[f"intent:{intent}"], metadata=md)
    except Exception:
        logger.debug("langfuse tag_intent skipped", exc_info=True)


def flush_langfuse() -> None:
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        logger.debug("langfuse flush skipped", exc_info=True)
```

**Context7 复核点(执行者必做)**:`update_current_trace` 的确切签名(tags/metadata 参数名);若该版本无此方法,改用它提供的等价「更新当前 trace 属性」API,并在代码注释注明依据的文档 URL。

- [ ] **Step 4: state/chat_service/nodes 接线**

`app/graph/state.py`:`ChatGraphState` 加三字段(注释:ch09 轮次锚点与 trace 续传):

```python
    turn_message_id: int | None            # ch09:本轮用户行 id(prepare 落库盖章)
    final_assistant_message_id: int | None # ch09:本轮最终回答行 id(log 提交后盖章)
    trace_id: str | None                   # ch09:Langfuse trace 续传锚点
```

`new_turn_state` 返回 dict 加:

```python
        "turn_message_id": int(user_db_id) if user_db_id is not None else None,
        "final_assistant_message_id": None,
        "trace_id": None,
```

`app/services/chat_service.py` `stream()`:构造 config 段落(`config = {...}` 之后、`astream` 之前)改为:

```python
            config = {"configurable": {"thread_id": turn.session_id,
                                       "user_id": turn.user_id}}
            # ch09:trace 续传——resume 轮从 checkpoint 拿回原 trace_id/intent,
            # 模型/工具消耗归到原意图,服务重启后 resume 也不进 unknown
            trace_id = None
            intent = confidence = None
            if turn.resume_command is not None:
                st = await self._graph.aget_state(config)
                vals = st.values or {}
                trace_id = vals.get("trace_id")
                intent = vals.get("intent")
                confidence = vals.get("intent_confidence")
            if trace_id is None:
                trace_id = new_trace_id()
            config["configurable"]["trace_id"] = trace_id
            config.update(build_trace_config(
                self._settings, conversation_id=turn.session_id,
                user_id=turn.user_id, trace_id=trace_id,
                intent=intent, intent_confidence=confidence))
```

import 加 `from app.services.langfuse_tracing import build_trace_config, new_trace_id`。

`app/graph/nodes.py` `classify_intent` 在 `route = ROUTE_TABLE[intent]` 之后加:

```python
        tag_intent(intent, confidence)  # ch09:回写 trace 元数据;未启用 no-op
```

import 加 `from app.services.langfuse_tracing import tag_intent`。

`build_log_node` 的 `log_turn`:`out = {...}` 构造后追加盖章:

```python
        tid = (config.get("configurable") or {}).get("trace_id")
        if tid:
            out["trace_id"] = tid  # ch09:trace_id 落 checkpoint,resume 续传
```

`app/main.py` lifespan 关闭段(`await app.state.mcp_gateway.close()` 之前)加:

```python
        from app.services.langfuse_tracing import flush_langfuse
        flush_langfuse()  # ch09:进程退出前投递残余 trace 事件
```

- [ ] **Step 5: 跑测试确认通过 + 回归**

Run: `uv run pytest tests/test_langfuse_tracing.py tests/test_ch05_acceptance.py -x -q`
Expected: PASS(disabled 默认下图行为零变化,既有验收不破)

- [ ] **Step 6: Commit**

```bash
git add app/services/langfuse_tracing.py app/services/chat_service.py app/graph/state.py app/graph/nodes.py app/main.py tests/test_langfuse_tracing.py
git commit -m "feat(ch09): Langfuse 请求级回调注入 + intent 元数据 + resume trace 续传"
```

---

### Task 4: 检索/工具/MCP 边界 span

**Files:**
- Modify: `app/knowledge/retriever.py`(search / rerank_candidates 包 retriever span)
- Modify: `app/tools/executor.py`(execute 包 tool span)
- Modify: `app/services/mcp_gateway.py`(call_tool 包 tool span)
- Test: `tests/test_trace_spans.py`(新建)

**Interfaces:**
- Consumes: Task 3 的 `tracing_enabled`。
- Produces:
  - `langfuse_tracing.observation(as_type: str, name: str, input: dict | None) -> contextmanager` — disabled 时 yield None;enabled 时 `get_client().start_as_current_observation(...)`,yield 观测对象(可 `.update(output=..., metadata=...)`,调用方判 None)
  - span 内容契约:`knowledge.search`/`knowledge.rerank`(retriever;input 含 query 截 500 字符/strategy,output 含 effective_strategy/hits 数/confidence/low/note);`tool.execute`(tool;input 工具名,output 状态/重试/耗时/截断结果);`mcp.call`(tool;input server+工具名,output 状态/耗时)。**不发密钥与未截断大 payload**(沿用 audit_result_max_chars 截断)。

- [ ] **Step 1: 写失败测试**

```python
"""ch09 边界 span:disabled 时三个边界零行为变化且不抛。"""
from app.services.langfuse_tracing import observation
from tests.test_langfuse_tracing import _settings  # 复用构造器


def test_observation_disabled_yields_none():
    with observation("retriever", "knowledge.search", {"query": "x"}) as obs:
        assert obs is None


def test_observation_enabled_mock(monkeypatch):
    import app.services.langfuse_tracing as lt
    calls = []

    class FakeObs:
        def update(self, **kw):
            calls.append(kw)

    class FakeClient:
        def start_as_current_observation(self, as_type, name, input=None):
            calls.append({"as_type": as_type, "name": name, "input": input})
            class Cm:
                def __enter__(self): return FakeObs()
                def __exit__(self, *a): return False
            return Cm()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-t")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-t")
    monkeypatch.setattr(lt, "_get_client", lambda: FakeClient())
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-t",
                  langfuse_secret_key="sk-lf-t")
    with lt.observation("tool", "tool.execute", {"tool": "query_order"}, settings=s) as obs:
        obs.update(output={"status": "成功"})
    assert calls[0]["as_type"] == "tool"
    assert calls[1]["output"]["status"] == "成功"
```

(`observation` 签名带可选 `settings=None`:不传时从 contextvar/全局判断 disabled;为可测性,实现里允许显式传 settings。`_get_client` 是模块内薄封装,便于 monkeypatch。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_trace_spans.py -x -q`
Expected: FAIL(observation 不存在)

- [ ] **Step 3: 实现**

`langfuse_tracing.py` 追加:

```python
from contextlib import contextmanager


def _get_client():
    from langfuse import get_client
    return get_client()


@contextmanager
def observation(as_type: str, name: str, input: dict | None = None,
                settings: Settings | None = None):
    """固定边界 span;disabled/异常一律 yield None 不拦主路。"""
    if settings is not None and not tracing_enabled(settings):
        yield None
        return
    try:
        with _get_client().start_as_current_observation(
                as_type=as_type, name=name, input=input) as obs:
            yield obs
    except Exception:
        logger.debug("langfuse observation %s skipped", name, exc_info=True)
        yield None
```

注意:第二个 `yield None` 只在异常路径到达——contextmanager 里 except 后需 `yield None` 再 return(上面写法对:except 块内 yield)。但这样 with 体在异常时会执行两次?不会——`@contextmanager` 生成器 throw 进 except 后 yield 是合法的恢复路径,调用方拿到 None。**执行者注意**:更稳的写法是 try 里正常 yield,except 里先记日志再 `yield None`,二者取其一,测试两种路径都过即可。

`retriever.py` `search()` 主体包一层(`self._settings` 已有):

```python
        with observation("retriever", "knowledge.search",
                         {"query": query[:500], "strategy": strategy},
                         settings=self._settings) as obs:
            result = self._search_inner(...)   # 原 search 主体改名移入
            if obs is not None:
                obs.update(output={"effective_strategy": result.effective_strategy,
                                   "hits": len(result.hits),
                                   "confidence": result.confidence_score,
                                   "low_confidence": result.low_confidence,
                                   "note": result.note})
            return result
```

`rerank_candidates` 同款(span 名 `knowledge.rerank`)。`executor.py` 统一执行引擎主方法包 `observation("tool", "tool.execute", {"tool": name})`,output 记 `status/retry_count/duration_ms/result[:audit_result_max_chars]`。`mcp_gateway.py` call_tool 包 `observation("tool", "mcp.call", {"server": server, "tool": tool})`。三处改动都要求:disabled(默认)下既有测试全绿——`settings=None` 调用方总是显式传自己持有的 settings,retriever/executor/gateway 都有。

- [ ] **Step 4: 跑测试 + 回归**

Run: `uv run pytest tests/test_trace_spans.py -x -q && uv run pytest tests/ -k "retriever or executor or mcp" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/langfuse_tracing.py app/knowledge/retriever.py app/tools/executor.py app/services/mcp_gateway.py tests/test_trace_spans.py
git commit -m "feat(ch09): 检索/工具/MCP 固定边界 span(未启用零侵入)"
```

---

### Task 5: 落池快照 + 轮次锚点(LowConfidenceRecord/state/log/store)

**Files:**
- Create: `app/services/retrieval_snapshot.py`
- Modify: `app/sessions.py:17-22`(LowConfidenceRecord 加三字段)
- Modify: `app/graph/nodes.py`(_build_low_conf 快照 + log 盖章 final_assistant_message_id/turn_message_id)
- Modify: `app/store_db.py:120-127`(写新列)
- Modify: `app/config.py`(LOW_CONF_SNAPSHOT_TOP_N)
- Test: `tests/test_low_conf_snapshot.py`(新建)

**Interfaces:**
- Produces:
  - `retrieval_snapshot.snapshot_top_chunks(hits: list[dict] | None, top_n: int, *, answer_chars: int = 500) -> list[dict] | None` — hits 元素是 asdict(KnowledgeHit)(键 `questions` 映射为快照键 `question`);空/None → None
  - `LowConfidenceRecord(raw_question, source, reason, conversation_id, retrieved_chunks=None, resolved_question=None, turn_message_id=None)`
  - log 节点 state 输出:`final_assistant_message_id: int`、`turn_message_id: int`(既有 prepared_user_id 口径)
  - `Settings.low_conf_snapshot_top_n: int = 3`

- [ ] **Step 1: 写失败测试**

```python
"""ch09 落池快照:Top N 截取/500 字符截断/三字段随 commit 落库(内存+DB 同语义)。"""
import json

from app.services.retrieval_snapshot import snapshot_top_chunks
from app.sessions import LowConfidenceRecord


def _hit(i, answer="答"):
    return {"chunk_id": i, "score": 0.9 - i * 0.1, "section_path": f"s/{i}",
            "questions": f"问{i}", "answer": answer, "category": "c",
            "source_doc": None, "chunk_index": i}


def test_snapshot_top_n_and_truncation():
    hits = [_hit(i, answer="长" * 600) for i in range(1, 6)]
    snap = snapshot_top_chunks(hits, 3)
    assert [c["chunk_id"] for c in snap] == [1, 2, 3]
    assert snap[0]["question"] == "问1"
    assert len(snap[0]["answer"]) == 500
    assert snapshot_top_chunks([], 3) is None
    assert snapshot_top_chunks(None, 3) is None


def test_build_low_conf_carries_snapshot():
    from app.graph.nodes import _build_low_conf
    state = {"low_conf_source": "retrieval_low_conf", "raw_query": "能开专票吗",
             "resolved_query": "能否开具增值税专用发票",
             "turn_message_id": 77, "retrieval_status": "low_confidence",
             "low_conf_reason": {"top1": 0.01, "threshold": 0.5},
             "final_text": "", "evidence": [],
             "retrieval_result": {"hits": [_hit(1), _hit(2)]}}
    rec = _build_low_conf(state, 42, 3)
    assert rec.source == "retrieval_low_conf"
    assert rec.resolved_question == "能否开具增值税专用发票"
    assert rec.turn_message_id == 77
    assert [c["chunk_id"] for c in rec.retrieved_chunks] == [1, 2]
    json.dumps(rec.retrieved_chunks)  # 必须可 JSON 序列化


def test_commit_turn_writes_snapshot_columns(db_session_factory):
    from app.store_db import DbSessionStore
    from app.models import LowConfidenceQuestion
    import asyncio

    store = DbSessionStore(db_session_factory, 8000)

    async def go():
        sid = await store.create("u1")
        await store.commit_turn(
            sid,
            [__import__("app.sessions", fromlist=["StoredMessage"]).StoredMessage("user", "q"),
             __import__("app.sessions", fromlist=["StoredMessage"]).StoredMessage("assistant", "a")],
            low_confidence=LowConfidenceRecord(
                raw_question="q", source="self_check", reason="{}",
                conversation_id=int(sid),
                retrieved_chunks=[{"chunk_id": 1, "score": 0.5}],
                resolved_question="qq", turn_message_id=None))
        return sid

    asyncio.run(go())
    with db_session_factory() as s:
        row = s.query(LowConfidenceQuestion).first()
        assert row.resolved_question == "qq"
        assert row.retrieved_chunks == [{"chunk_id": 1, "score": 0.5}]
        assert row.process_status == "pending"   # 飞轮默认待处理
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_low_conf_snapshot.py -x -q`
Expected: FAIL(snapshot_top_chunks 不存在)

- [ ] **Step 3: 实现**

`app/services/retrieval_snapshot.py`:

```python
"""ch09:落池召回片段快照(retrieval_result.hits → 审核页展示形态)。"""


def snapshot_top_chunks(hits: list[dict] | None, top_n: int,
                        *, answer_chars: int = 500) -> list[dict] | None:
    """截 Top N;answer 截 answer_chars 防爆。hits 为 asdict(KnowledgeHit) 形态。
    空/None → None(没走检索的入口语义:NULL 而不是空数组)。"""
    if not hits:
        return None
    out = []
    for h in hits[:top_n]:
        out.append({"chunk_id": h.get("chunk_id"), "score": h.get("score"),
                    "section_path": h.get("section_path"),
                    "question": h.get("questions"),
                    "answer": (h.get("answer") or "")[:answer_chars]})
    return out or None
```

`app/config.py` ch09 段加:

```python
    low_conf_snapshot_top_n: int = Field(default=3, gt=0)  # 落池召回快照条数(审核页展示)
```

`app/sessions.py` LowConfidenceRecord 改为:

```python
@dataclass(frozen=True)
class LowConfidenceRecord:
    raw_question: str
    source: str            # "retrieval_low_conf" | "self_check" | "user_feedback"
    reason: str | None
    conversation_id: int | None
    retrieved_chunks: list[dict] | None = None   # ch09:召回快照
    resolved_question: str | None = None         # ch09:指代消解后问题
    turn_message_id: int | None = None           # ch09:轮次锚点
```

`app/graph/nodes.py`:
- import 加 `from app.services.retrieval_snapshot import snapshot_top_chunks`。
- `_build_low_conf(state, cid)` 改签名 `_build_low_conf(state, cid, top_n)`,两分支都补:

```python
    snap = state.get("retrieval_result") or {}
    chunks = snapshot_top_chunks(snap.get("hits"), top_n)
    base = {"raw_question": state["raw_query"], "conversation_id": cid,
            "retrieved_chunks": chunks,
            "resolved_question": state.get("resolved_query") or None,
            "turn_message_id": state.get("turn_message_id")}
```

两个 return 改为 `LowConfidenceRecord(**base, source=..., reason=...)`。log 节点调用处改 `_build_low_conf(state, cid, deps.settings.low_conf_snapshot_top_n)`。
- `log_turn` 的 `out` 构造处补锚点(`final_assistant_message_id` 取本轮最后一行——validate_turn 保证末行 assistant):

```python
        out["final_assistant_message_id"] = int(result.message_ids[-1])
        if prepared_user_id is not None:
            out["turn_message_id"] = prepared_user_id
```

`app/store_db.py` `_commit_sync` 的 LowConfidenceQuestion 构造补:

```python
                    retrieved_chunks=low_confidence.retrieved_chunks,
                    resolved_question=low_confidence.resolved_question,
                    turn_message_id=low_confidence.turn_message_id,
```

内存 store 无需改(dataclass 整体入 `low_confidence` 列表,测试断言字段即可)。

`.env.example` ch09 节块加 `# LOW_CONF_SNAPSHOT_TOP_N=3`。

- [ ] **Step 4: 跑测试 + 回归**

Run: `uv run pytest tests/test_low_conf_snapshot.py -x -q && uv run pytest tests/ -k "low_conf or log or store" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/retrieval_snapshot.py app/sessions.py app/graph/nodes.py app/store_db.py app/config.py .env.example tests/test_low_conf_snapshot.py
git commit -m "feat(ch09): 落池带召回快照/消解问题/轮次锚点,log 盖章最终回答行"
```

---

### Task 6: SSE `turn_committed` 帧

**Files:**
- Modify: `app/services/chat_service.py`(TurnCommittedEvent + stream 发射)
- Modify: `app/routers/chat.py`(_EventStream 序列化)
- Test: `tests/test_turn_committed.py`(新建)

**Interfaces:**
- Consumes: Task 4 的 state 锚点。
- Produces:
  - `TurnCommittedEvent(conversation_id: str, turn_message_id: str, assistant_message_id: str)`(全十进制字符串,BIGINT 防精度丢失)
  - SSE 帧:`{"type": "turn_committed", "conversation_id", "turn_message_id", "assistant_message_id"}` 在 done 之前;挂起轮/失败轮/中断轮不发

- [ ] **Step 1: 写失败测试**

```python
"""ch09 turn_committed:正常完成发帧;挂起轮不发。"""
import asyncio

from app.services.chat_service import ChatService, DoneEvent, TurnCommittedEvent


class _FakeSnapshot:
    def __init__(self, values, tasks=()):
        self.values = values
        self.tasks = tasks


class _FakeGraphOK:
    async def astream(self, *a, **k):
        return
        yield  # pragma: no cover(空流)

    async def aget_state(self, config):
        return _FakeSnapshot({"turn_message_id": 11,
                              "final_assistant_message_id": 12})


class _FakeGraphPending:
    async def astream(self, *a, **k):
        return
        yield  # pragma: no cover

    async def aget_state(self, config):
        intr = type("I", (), {"id": "i1", "value": {"type": "order_selector",
                                                    "orders": []}})()
        task = type("T", (), {"interrupts": [intr]})()
        return _FakeSnapshot({}, [task])


def _service(graph):
    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")
    svc = ChatService(store=None, model=None, settings=s, system_prompt="")
    svc.set_graph(graph)
    return svc


class _Turn:
    session_id = "9"
    user_id = "u"
    user_text = "q"
    lock_key = "9"
    resume_command = None
    released = True   # release_turn 走 locks;绕过:测 service 直调 stream 前替换 release


def _collect(graph):
    svc = _service(graph)
    svc.release_turn = lambda turn: None   # 测试旁路锁注册表
    turn = _Turn()
    return [e for e in asyncio.run(_aiter(svc, turn))]


async def _aiter(svc, turn):
    async for e in svc.stream(turn):
        yield e


def test_committed_frame_emitted_before_done():
    events = _collect(_FakeGraphOK())
    kinds = [type(e).__name__ for e in events]
    assert "TurnCommittedEvent" in kinds
    assert kinds.index("TurnCommittedEvent") < kinds.index("DoneEvent")
    ev = next(e for e in events if isinstance(e, TurnCommittedEvent))
    assert ev.assistant_message_id == "12" and ev.turn_message_id == "11"


def test_no_frame_when_pending_interrupt():
    events = _collect(_FakeGraphPending())
    assert not any(isinstance(e, TurnCommittedEvent) for e in events)
```

注意 `_FakeGraphPending.aget_state` 返回的 tasks 会被 `_pending_selector_events` 消费发 OrderSelectorEvent;`_FakeGraphOK` 的 values 带锚点。若现有 stream 实现细节与本假图假设冲突(如 astream 需要 config 含 callbacks 键),执行者按编译错误修正假图,断言语义不变。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_turn_committed.py -x -q`
Expected: FAIL(TurnCommittedEvent 不存在)

- [ ] **Step 3: 实现**

`chat_service.py`:
- 加事件 dataclass(放 TicketPreviewEvent 后):

```python
@dataclass(frozen=True)
class TurnCommittedEvent:
    """ch09:本轮已落库且图正常完成;前端据 assistant_message_id 启用 👍👎。"""
    conversation_id: str
    turn_message_id: str
    assistant_message_id: str
```

- `ChatEvent` Union 加 `TurnCommittedEvent`。
- `stream()` 尾部(`_pending_selector_events` 循环与 `yield DoneEvent()` 之间)改:

```python
            pending = False
            async for event in self._pending_selector_events(config):
                pending = True
                yield event
            if not pending:  # 挂起轮无最终回答,不发帧(spec §5.3)
                st = await self._graph.aget_state(config)
                vals = st.values or {}
                aid = vals.get("final_assistant_message_id")
                tid = vals.get("turn_message_id")
                if aid and tid:
                    yield TurnCommittedEvent(turn.session_id, str(tid), str(aid))
            yield DoneEvent()
```

`app/routers/chat.py` `_EventStream._body` 加分支(TicketPreviewEvent 之后):

```python
                elif isinstance(event, TurnCommittedEvent):
                    yield _sse({"type": "turn_committed",
                                "conversation_id": event.conversation_id,
                                "turn_message_id": event.turn_message_id,
                                "assistant_message_id": event.assistant_message_id})
```

import 同步加 `TurnCommittedEvent`。

- [ ] **Step 4: 跑测试 + 回归**

Run: `uv run pytest tests/test_turn_committed.py -x -q && uv run pytest tests/ -k "chat" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/chat_service.py app/routers/chat.py tests/test_turn_committed.py
git commit -m "feat(ch09): SSE turn_committed 帧(落库锚点下发,挂起轮不发)"
```

---

### Task 7: 反馈端点 `POST /v1/chat/feedback`(账本校验 + checkpoint 回捞 + 持久幂等)

**Files:**
- Create: `app/services/feedback.py`
- Modify: `app/schemas.py`(ChatFeedbackRequest)
- Modify: `app/routers/chat.py`(端点)
- Modify: `app/errors.py` + `app/main.py`(FeedbackConflictError → 409)
- Test: `tests/test_feedback.py`(新建)

**Interfaces:**
- Consumes: Task 1 ChatFeedback/lcq 模型;Task 4 snapshot_top_chunks;Task 5 帧(前端)。
- Produces:
  - `feedback.submit_feedback(*, settings, session_factory, graph, user_id, conversation_id, assistant_message_id, sentiment, on_pooled: callable | None) -> dict`
    返回 `{"status": "recorded", "feedback_id": str}` 或 `{"status": "duplicate", "feedback_id": str}`
  - 异常:SessionNotFoundError(会话归属/跨会话消息 404)、FeedbackConflictError(409:非最终回答/中间工具行/反向冲突/挂起轮)
  - `FeedbackConflictError(AppError)`,code="feedback_conflict",HTTP 409
  - reason JSON 契约:`{"assistant_message_id": int, "turn_message_id": int, "retrieval": "hit"|"not_run"|"missing"|"mismatch"}`

- [ ] **Step 1: 写失败测试**

```python
"""ch09 反馈:归属 404/非法目标 409/幂等/down 落池+回捞/同事务。"""
import asyncio

import pytest

from app.errors import FeedbackConflictError, SessionNotFoundError
from app.models import ChatFeedback, Conversation, LowConfidenceQuestion, Message
from app.services import feedback


def _seed_turn(sf, user_id="u1"):
    with sf() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        s.flush()
        u = Message(conversation_id=conv.id, role="user", content="怎么退货")
        s.add(u)
        s.flush()
        a = Message(conversation_id=conv.id, role="assistant", content="答复")
        s.add(a)
        s.commit()
        return conv.id, u.id, a.id


class _NoHistoryGraph:
    async def aget_state_history(self, config):
        return
        yield  # pragma: no cover(checkpoint 缺失 → missing)


def _settings():
    from app.config import Settings
    return Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                    database_url="mysql+pymysql://u:p@h/d")


def _submit(sf, graph, **kw):
    args = dict(settings=_settings(), session_factory=sf, graph=graph,
                user_id="u1", on_pooled=None)
    args.update(kw)
    return asyncio.run(feedback.submit_feedback(**args))


def test_down_pools_question_and_feedback_atomically(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    out = _submit(db_session_factory, _NoHistoryGraph(),
                  conversation_id=str(cid), assistant_message_id=str(aid),
                  sentiment="down")
    assert out["status"] == "recorded"
    with db_session_factory() as s:
        fb = s.query(ChatFeedback).first()
        lcq = s.query(LowConfidenceQuestion).first()
        assert fb.sentiment == "down"
        assert fb.low_confidence_question_id == lcq.id
        assert lcq.source == "user_feedback"
        assert lcq.raw_question == "怎么退货"
        assert lcq.retrieved_chunks is None       # checkpoint 缺失 → NULL 仍落池
        assert lcq.process_status == "pending"


def test_ownership_404_and_illegal_target_409(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    with pytest.raises(SessionNotFoundError):
        _submit(db_session_factory, _NoHistoryGraph(), user_id="u2",
                conversation_id=str(cid), assistant_message_id=str(aid),
                sentiment="down")
    with db_session_factory() as s:  # 中间工具调用 assistant 行:409
        mid = Message(conversation_id=cid, role="assistant", content=None,
                      tool_calls=[{"name": "t", "args": {}, "id": "x", "type": "tool_call"}])
        s.add(mid)
        s.commit()
        mid_id = mid.id
    with pytest.raises(FeedbackConflictError):
        _submit(db_session_factory, _NoHistoryGraph(),
                conversation_id=str(cid), assistant_message_id=str(mid_id),
                sentiment="down")


def test_idempotent_same_sentiment_and_conflict_reverse(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    first = _submit(db_session_factory, _NoHistoryGraph(),
                    conversation_id=str(cid), assistant_message_id=str(aid),
                    sentiment="down")
    again = _submit(db_session_factory, _NoHistoryGraph(),
                    conversation_id=str(cid), assistant_message_id=str(aid),
                    sentiment="down")
    assert again["status"] == "duplicate"
    assert again["feedback_id"] == first["feedback_id"]
    with db_session_factory() as s:
        assert s.query(LowConfidenceQuestion).count() == 1   # 不产生重复 lcq
    with pytest.raises(FeedbackConflictError):
        _submit(db_session_factory, _NoHistoryGraph(),
                conversation_id=str(cid), assistant_message_id=str(aid),
                sentiment="up")


def test_up_records_without_lcq(db_session_factory):
    cid, uid, aid = _seed_turn(db_session_factory)
    out = _submit(db_session_factory, _NoHistoryGraph(),
                  conversation_id=str(cid), assistant_message_id=str(aid),
                  sentiment="up")
    assert out["status"] == "recorded"
    with db_session_factory() as s:
        assert s.query(ChatFeedback).count() == 1
        assert s.query(LowConfidenceQuestion).count() == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_feedback.py -x -q`
Expected: FAIL(feedback 模块不存在)

- [ ] **Step 3: 实现 feedback.py**

```python
"""ch09:👍👎 反馈——先校验账本(404/409),再尽力回捞 checkpoint 检索快照,
down 落池与反馈行同事务,唯一键 (conversation_id, assistant_message_id) 持久幂等。"""

import asyncio
import json
import logging

from sqlalchemy.exc import IntegrityError

from app.errors import FeedbackConflictError, SessionNotFoundError
from app.models import ChatFeedback, Conversation, LowConfidenceQuestion, Message
from app.services.retrieval_snapshot import snapshot_top_chunks

logger = logging.getLogger(__name__)


def _validate_turn(s, cid: int, aid: int, user_id: str) -> tuple[int, str]:
    """返回 (turn_message_id, raw_question);非法目标抛 409,归属失败抛 404。"""
    conv = s.get(Conversation, cid)
    if conv is None or conv.user_id != user_id:
        raise SessionNotFoundError("session not found")
    msg = s.get(Message, aid)
    if msg is None or msg.conversation_id != cid:
        raise FeedbackConflictError("message not in conversation")
    if (msg.role != "assistant" or not (msg.content or "").strip()
            or msg.tool_calls):
        raise FeedbackConflictError("not a final answer message")
    prev_user = (s.query(Message.id)
                 .filter(Message.conversation_id == cid, Message.role == "user",
                         Message.id < aid)
                 .order_by(Message.id.desc()).limit(1).first())
    if prev_user is None:
        raise FeedbackConflictError("no preceding user message")
    turn_id = prev_user[0]
    nxt_user = (s.query(Message.id)
                .filter(Message.conversation_id == cid, Message.role == "user",
                        Message.id > aid)
                .order_by(Message.id).limit(1).first())
    q = s.query(Message)
    bound = q.filter(Message.conversation_id == cid, Message.role == "assistant",
                     Message.id > turn_id,
                     Message.content.isnot(None), Message.tool_calls.is_(None))
    bound = bound.filter(Message.id < nxt_user[0]) if nxt_user else bound
    last_final = bound.order_by(Message.id.desc()).first()
    if last_final is None or last_final.id != aid:
        raise FeedbackConflictError("not the final answer of its turn")
    raw = s.get(Message, turn_id)
    return turn_id, raw.content or ""


async def _recover_snapshot(graph, conversation_id: str, aid: int, turn_id: int,
                            top_n: int) -> dict:
    """精确匹配完成轮锚点回捞;任何缺失/异常 → 降级(NULL 快照仍落池)。"""
    config = {"configurable": {"thread_id": str(conversation_id)}}
    try:
        async for snap in graph.aget_state_history(config):
            vals = snap.values or {}
            if (vals.get("final_assistant_message_id") != aid
                    or vals.get("turn_message_id") != turn_id):
                continue
            stamped = [(m.additional_kwargs or {}).get("db_id")
                       for m in (vals.get("turn_messages") or [])]
            stamped = [int(x) for x in stamped if x is not None]
            if not stamped or stamped[0] != turn_id or stamped[-1] != aid:
                return {"chunks": None, "resolved": None, "retrieval": "mismatch"}
            hits = (vals.get("retrieval_result") or {}).get("hits")
            return {"chunks": snapshot_top_chunks(hits, top_n),
                    "resolved": vals.get("resolved_query") or None,
                    "retrieval": "hit" if hits else "not_run"}
    except Exception:
        logger.warning("feedback checkpoint recover failed conv=%s",
                       conversation_id, exc_info=True)
    return {"chunks": None, "resolved": None, "retrieval": "missing"}


async def submit_feedback(*, settings, session_factory, graph, user_id: str,
                          conversation_id: str, assistant_message_id: str,
                          sentiment: str, on_pooled=None) -> dict:
    cid = int(conversation_id)
    aid = int(assistant_message_id)

    def _load():
        with session_factory() as s:
            turn_id, raw = _validate_turn(s, cid, aid, user_id)
            existing = (s.query(ChatFeedback)
                        .filter_by(conversation_id=cid, assistant_message_id=aid)
                        .first())
            return turn_id, raw, existing

    turn_id, raw, existing = await asyncio.to_thread(_load)
    if existing is not None:
        if existing.sentiment == sentiment:
            return {"status": "duplicate", "feedback_id": str(existing.id)}
        raise FeedbackConflictError("feedback already recorded with other sentiment")

    recovered = {"chunks": None, "resolved": None, "retrieval": "missing"}
    if sentiment == "down" and graph is not None:
        recovered = await _recover_snapshot(
            graph, conversation_id, aid, turn_id,
            settings.low_conf_snapshot_top_n)

    def _write():
        with session_factory() as s:
            try:
                lcq = None
                if sentiment == "down":
                    lcq = LowConfidenceQuestion(
                        conversation_id=cid, raw_question=raw, source="user_feedback",
                        reason=json.dumps(
                            {"assistant_message_id": aid, "turn_message_id": turn_id,
                             "retrieval": recovered["retrieval"]}, ensure_ascii=False),
                        retrieved_chunks=recovered["chunks"],
                        resolved_question=recovered["resolved"],
                        turn_message_id=turn_id)
                    s.add(lcq)
                    s.flush()
                fb = ChatFeedback(conversation_id=cid, assistant_message_id=aid,
                                  turn_message_id=turn_id, sentiment=sentiment,
                                  low_confidence_question_id=(
                                      lcq.id if lcq else None))
                s.add(fb)
                s.commit()   # down:lcq 与反馈行同事务,失败整体回滚
                return fb
            except IntegrityError:   # 并发同键:读既有记录判定,不留半条
                s.rollback()
                prior = (s.query(ChatFeedback)
                         .filter_by(conversation_id=cid, assistant_message_id=aid)
                         .first())
                if prior is not None and prior.sentiment == sentiment:
                    return prior
                raise FeedbackConflictError(
                    "feedback already recorded with other sentiment")

    fb = await asyncio.to_thread(_write)
    if fb.created_at and sentiment == "down" and on_pooled is not None:
        on_pooled()   # 提交成功后才通知飞轮(spec §5.4)
    return {"status": "recorded", "feedback_id": str(fb.id)}
```

注意 `_write` 并发撞键返回 prior 时,外层会误报 recorded——处理:`_write` 返回 `(fb, created_flag)`,prior 命中时 `created_flag=False` → 返回 duplicate。执行者修正此细节并补并发测试(fake 并发可用两次调用模拟第二次 IntegrityError 路径,或直接串行两次验证语义)。

`app/errors.py` 加:

```python
class FeedbackConflictError(AppError):
    code = "feedback_conflict"
```

`app/main.py` 加 handler(409),照 ResumeConflictError 同款:

```python
    @app.exception_handler(FeedbackConflictError)
    async def _(request: Request, exc: FeedbackConflictError) -> JSONResponse:
        return JSONResponse(status_code=409,
                            content=_error_body(exc.code, "该反馈目标不可用或已反馈"))
```

`app/schemas.py` 加:

```python
class ChatFeedbackRequest(BaseModel):
    user_id: str
    conversation_id: str
    assistant_message_id: str
    sentiment: Literal["up", "down"]

    @field_validator("user_id")
    @classmethod
    def _user_id(cls, v):
        return _validate_uuid(v)

    @field_validator("conversation_id")
    @classmethod
    def _conversation_id(cls, v):
        return _validate_session_id(v)

    @field_validator("assistant_message_id")
    @classmethod
    def _assistant_message_id(cls, v: str) -> str:
        if not _SOURCE_MSG_ID_RE.match(v):
            raise ValueError("assistant_message_id must be a decimal id string")
        return v
```

`app/routers/chat.py` 加端点:

```python
@router.post("/v1/chat/feedback")
async def chat_feedback(body: ChatFeedbackRequest, request: Request):
    from app.services import feedback
    worker = getattr(request.app.state, "flywheel_worker", None)
    return await feedback.submit_feedback(
        settings=request.app.state.settings,
        session_factory=request.app.state.session_factory,
        graph=request.app.state.chat_service._graph,
        user_id=body.user_id, conversation_id=body.conversation_id,
        assistant_message_id=body.assistant_message_id, sentiment=body.sentiment,
        on_pooled=(worker.notify if worker is not None else None))
```

(`chat_service._graph` 私有读取不雅——给 ChatService 加 `graph` property 返回 `self._graph`,执行者顺手加。)

- [ ] **Step 4: 跑测试 + 回归**

Run: `uv run pytest tests/test_feedback.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/feedback.py app/schemas.py app/routers/chat.py app/errors.py app/main.py app/services/chat_service.py tests/test_feedback.py
git commit -m "feat(ch09): 反馈端点——账本校验+checkpoint 精确回捞+持久幂等落池"
```

---

### Task 8: chat.html 反馈接后端 + 历史消息反馈态(Vibe 例外)

前端任务,不套 TDD;完成后人工点验 + 既有字符串断言回归。

**Files:**
- Modify: `app/static/chat.html`(addFeedback 接 POST /v1/chat/feedback;turn_committed 帧启用按钮;历史回载恢复反馈态)
- Modify: `app/routers/conversations.py` + `app/services/feedback.py`(messages API 补 feedback_eligible/feedback_sentiment)
- Modify: `tests/test_chat_page.py`(字符串断言:fetch 路径/帧类型名)

**Interfaces:**
- Consumes: Task 6 SSE 帧、Task 7 端点。
- Produces:
  - `feedback.annotate_messages(session_factory, conversation_id: int, messages: list[dict]) -> list[dict]` — 每条 assistant 补 `feedback_eligible: bool`(是否当轮最终回答)与 `feedback_sentiment: "up"|"down"|None`(查 chat_feedback)
  - `GET /api/conversations/{id}/messages` 响应消息对象含上述两字段

- [ ] **Step 1: annotate 服务测试(后端部分仍 TDD)** `tests/test_feedback_annotate.py`

```python
def test_annotate_marks_only_final_answers(db_session_factory):
    from tests.test_feedback import _seed_turn
    from app.services import feedback
    import asyncio

    cid, uid, aid = _seed_turn(db_session_factory)
    asyncio.run(feedback.submit_feedback(
        settings=__import__("tests.test_feedback", fromlist=["_settings"])._settings(),
        session_factory=db_session_factory, graph=None, user_id="u1",
        conversation_id=str(cid), assistant_message_id=str(aid),
        sentiment="up", on_pooled=None))
    msgs = [{"id": str(uid), "role": "user", "content": "怎么退货"},
            {"id": str(aid), "role": "assistant", "content": "答复"}]
    out = asyncio.run(asyncio.to_thread(
        feedback.annotate_messages, db_session_factory, cid, msgs))
    assert out[0]["feedback_eligible"] is False
    assert out[1]["feedback_eligible"] is True
    assert out[1]["feedback_sentiment"] == "up"
```

- [ ] **Step 2: 实现 annotate + conversations 路由接线**

`feedback.py` 加(同步函数):

```python
def annotate_messages(session_factory, conversation_id: int,
                      messages: list[dict]) -> list[dict]:
    """给历史消息补 feedback_eligible/feedback_sentiment;无反馈列表演进为全 None。"""
    ids = [int(m["id"]) for m in messages if m["role"] == "assistant"]
    with session_factory() as s:
        rows = (s.query(Message.id).filter(
            Message.conversation_id == conversation_id,
            Message.role == "assistant", Message.content.isnot(None),
            Message.tool_calls.is_(None)).order_by(Message.id).all())
        fbs = {r.assistant_message_id: r.sentiment
               for r in s.query(ChatFeedback)
               .filter_by(conversation_id=conversation_id).all()}
    final_ids = set()
    user_bounds = [int(m["id"]) for m in messages if m["role"] == "user"]
    # 最终回答 = 两个相邻 user 行之间(或末尾)最后一条合法 assistant
    ordered = sorted(int(m["id"]) for m in messages)
    bounds = sorted(user_bounds) + [1 << 62]
    for lo, hi in zip([0, *bounds[:-1]], bounds):
        cands = [i for i, in rows if lo < i < hi]
        if cands:
            final_ids.add(cands[-1])
    out = []
    for m in messages:
        mid = int(m["id"])
        out.append({**m,
                    "feedback_eligible": m["role"] == "assistant" and mid in final_ids,
                    "feedback_sentiment": fbs.get(mid)})
    return out
```

`app/routers/conversations.py` 的 messages 端点在返回前过一遍 annotate(会话归属校验保持原有;service 返回 None 的语义不变)。

- [ ] **Step 3: chat.html 改造(Vibe)**

- SSE 解析加 `turn_committed` 分支:记录 `{turn_message_id, assistant_message_id}` 到当前回复 DOM 的 dataset;`addFeedback` 只在有 assistant_message_id 时渲染按钮。
- 点击:`fetch("/v1/chat/feedback", {method:"POST", body: JSON.stringify({user_id, conversation_id, assistant_message_id, sentiment})})`;成功(200 recorded/duplicate)锁按钮显示「已反馈」;409 显示「已反馈过」并锁;网络失败不锁、允许重试;localStorage 仅做显示缓存,key 改为 `fb:{sessionId}:{assistantMessageId}`。
- 历史回载:按 `feedback_eligible`/`feedback_sentiment` 渲染按钮态。
- resume 完成同样等 turn_committed 再启用。
- 注意 BIGINT:全程字符串持有 id,不 `Number()`。

- [ ] **Step 4: 字符串断言 + 回归**

`tests/test_chat_page.py` 加断言(照现有断言风格):

```python
def test_ch09_feedback_wiring():
    html = (Path(__file__).resolve().parent.parent
            / "app" / "static" / "chat.html").read_text(encoding="utf-8")
    assert "/v1/chat/feedback" in html
    assert "turn_committed" in html
    assert "assistant_message_id" in html
    assert "feedback_eligible" in html
```

Run: `uv run pytest tests/test_chat_page.py tests/test_feedback_annotate.py -x -q`

- [ ] **Step 5: 人工点验 + Commit**

起服点验:正常回复出 👍👎 → 点 👎 后端落池;挂起轮无按钮;刷新后状态保留。

```bash
git add app/static/chat.html app/routers/conversations.py app/services/feedback.py tests/test_chat_page.py tests/test_feedback_annotate.py
git commit -m "feat(ch09): 聊天页 👍👎 接后端,turn_committed 启用按钮,历史回载反馈态"
```

---

### Task 9: 正式版 evidence_confidence 置信闸 + 校准模式

**Files:**
- Create: `app/knowledge/evidence_confidence.py`
- Modify: `app/knowledge/retriever.py:260-268 区域`(hybrid_rerank 置信判定换正式闸;`rerank_candidates` 同款口径)
- Modify: `app/config.py`(冻结参数五 knob + 版本)
- Modify: `evals/run_retrieval_compare.py`(加 `--calibrate-evidence` 模式)
- Create: `evals/calibration/.gitkeep`(目录;产物由校准任务生成)
- Test: `tests/test_evidence_confidence.py`(新建)

**Interfaces:**
- Consumes: RetrievalResult/KnowledgeHit(retriever)。
- Produces:
  - `EvidenceGateParams(frozen dataclass)`: `weight_top1: float, weight_count: float, weight_margin: float, min_effective_score: float, threshold: float, version: str`
  - `evidence_confidence(scores: list[float], params: EvidenceGateParams) -> float | None` — scores 为降序精排分;空 → None
    `score = w_top1*top1 + w_count*min(len([s>=min_effective])/3, 1) + w_margin*(top1 - (second or 0))`;仅一条时 margin 记 top1
  - `evaluate_evidence(hits_scores, params) -> tuple[confidence, low]` 供 retriever 用
  - `Settings.evidence_gate_*`:`evidence_weight_top1/weight_count/weight_margin/min_effective_score/min_confidence/evidence_confidence_version`(默认值 = 占位校准前保守值:1.0/0.0/0.0/0.0/0.0553/"uncalibrated"——即未校准时行为与 top1 信号等价,注释明示「待 --calibrate-evidence 校准后改默认值」)
  - `gate_params_from_settings(settings) -> EvidenceGateParams`

- [ ] **Step 1: 写失败测试**

```python
"""ch09 正式置信闸:三信号合成/边界/与旧 top1 等价退化。"""
from app.knowledge.evidence_confidence import (
    EvidenceGateParams, evidence_confidence, gate_params_from_settings,
)

P = EvidenceGateParams(weight_top1=0.6, weight_count=0.2, weight_margin=0.2,
                       min_effective_score=0.05, threshold=0.4, version="t1")


def test_empty_scores_none():
    assert evidence_confidence([], P) is None


def test_single_hit_margin_full():
    assert evidence_confidence([0.8], P) == 0.6 * 0.8 + 0.2 * (1 / 3) + 0.2 * 0.8


def test_margin_uses_gap():
    a = evidence_confidence([0.8, 0.7], P)
    b = evidence_confidence([0.8, 0.1], P)
    assert b > a   # Top1 相同,分差大者分高


def test_effective_count_capped_at_three():
    a = evidence_confidence([0.8, 0.7, 0.6, 0.5], P)
    b = evidence_confidence([0.8, 0.7, 0.6], P)
    assert a == b   # 有效证据数归一按 3 封顶


def test_gate_params_from_settings_defaults_uncalibrated():
    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")
    p = gate_params_from_settings(s)
    assert p.version == "uncalibrated"   # 校准前保守等价 top1
    assert p.weight_top1 == 1.0
```

- [ ] **Step 2: 跑测试确认失败 → 实现 evidence_confidence.py**

```python
"""ch09 正式版证据置信闸(spec §5.2):Top1 分/有效证据数/Top1-Top2 分差三信号合成。

权重、有效分下限、阈值全部由 evals/run_retrieval_compare.py --calibrate-evidence
在 calibration split 上校准后冻结进 Settings 默认值 + evals/calibration/
evidence_confidence.json(版本控制);本模块不做任何自适应。
"""

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class EvidenceGateParams:
    weight_top1: float
    weight_count: float
    weight_margin: float
    min_effective_score: float
    threshold: float
    version: str


def gate_params_from_settings(settings: Settings) -> EvidenceGateParams:
    return EvidenceGateParams(
        weight_top1=settings.evidence_weight_top1,
        weight_count=settings.evidence_weight_count,
        weight_margin=settings.evidence_weight_margin,
        min_effective_score=settings.evidence_min_effective_score,
        threshold=settings.evidence_min_confidence,
        version=settings.evidence_confidence_version)


def evidence_confidence(scores: list[float], params: EvidenceGateParams) -> float | None:
    """降序精排分 → 0~1 合成置信分;无命中 None。仅一条时 margin 记 top1。"""
    if not scores:
        return None
    top1 = scores[0]
    margin = top1 - scores[1] if len(scores) > 1 else top1
    effective = sum(1 for s in scores if s >= params.min_effective_score)
    return (params.weight_top1 * top1
            + params.weight_count * min(effective / 3.0, 1.0)
            + params.weight_margin * margin)
```

config.py ch09 段加:

```python
    # 正式置信闸冻结参数(spec §5.2):默认 uncalibrated = 退化为旧 top1 信号行为;
    # 校准(uv run python evals/run_retrieval_compare.py --calibrate-evidence)后
    # 把 evals/calibration/evidence_confidence.json 的值同步到此处默认值并注明日期
    evidence_weight_top1: float = Field(default=1.0)
    evidence_weight_count: float = Field(default=0.0)
    evidence_weight_margin: float = Field(default=0.0)
    evidence_min_effective_score: float = Field(default=0.0)
    evidence_min_confidence: float = Field(default=0.0553)
    evidence_confidence_version: str = "uncalibrated"
```

`retriever.py`:把 `_threshold_apply` 段(search 末尾 confidence 计算与 `rerank_candidates` 内同款逻辑)抽成 `_apply_confidence(hits, effective) -> tuple[confidence, threshold, low]`:

```python
    def _apply_confidence(self, hits, effective):
        top_n = self._settings.rerank_top_k
        hits = hits[:top_n]
        scores = [h.score for h in hits]
        if effective == "hybrid_rerank":
            params = gate_params_from_settings(self._settings)
            confidence = evidence_confidence(scores, params)
            threshold = params.threshold
        else:
            confidence = scores[0] if scores else None
            threshold = getattr(self._settings, _THRESHOLD_ATTR[effective])
        low = (confidence is None) or (confidence < threshold)
        return hits, confidence, threshold, low
```

search 与 rerank_candidates 统一改走它(注意 rerank_candidates 现实现若未做阈值截断,保持现状只换 hybrid_rerank 的 confidence 算法——**执行者先读该函数现行实现再动,保持既有测试绿**)。两处闸位(refund_policy 经 search/rerank_candidates、query_faq 经 search)自动共用,满足 spec「两处共用」。

- [ ] **Step 3: 跑测试 + 回归**

Run: `uv run pytest tests/test_evidence_confidence.py -x -q && uv run pytest tests/ -k "retriever or confidence or faq or refund" -q`
Expected: PASS(默认 uncalibrated 参数下旧行为等价,既有闸测试不破)

- [ ] **Step 4: 校准模式(evals,不进 pytest)**

`run_retrieval_compare.py` 加 `--calibrate-evidence` 分支:复用 main 的建库+检索段,只对 calibration split 的 hybrid_rerank 臂收集 `scores` 向量;网格搜索 `weight_top1/weight_count/weight_margin`(0.1 步长归一网格)× `min_effective_score`(0.01~0.2)× `threshold`(0.01 步),约束 D_absent 误放行率 ≤ max_d_pass,最大化通过率;写 `evals/calibration/evidence_confidence.json`:

```json
{"version": "2026-10-09T...Z", "calibrated_at": "...", "corpus": "knowledge_docs/",
 "model": "...", "rerank_model": "...", "weights": {"top1": .., "count": .., "margin": ..},
 "min_effective_score": .., "threshold": .., "d_pass_rate": .., "pass_rate": ..}
```

无可行点 → 退出码 2 且不写产物(**停下来问用户,红线#4**)。此任务步骤为手动执行,计划此处只约束产物契约;跑校准烧额度,放在 Task 16 验收阶段执行一次并把冻结值回填 config 默认值。

- [ ] **Step 5: Commit**

```bash
git add app/knowledge/evidence_confidence.py app/knowledge/retriever.py app/config.py evals/run_retrieval_compare.py tests/test_evidence_confidence.py .env.example
git commit -m "feat(ch09): evidence_confidence 三信号置信闸(默认退化旧行为,待校准冻结)"
```

---

### Task 10: 评估脚本版本化 + 夜间回归读冻结 artifact

**Files:**
- Modify: `evals/run_retrieval_compare.py`(meta 加 corpus_version/dataset_version;hybrid_rerank 闸值口径改 evidence_confidence;校验 artifact 与 settings 一致)
- Modify: `evals/rag_eval_report.py`(build_rag_eval/validate_rag_eval 加 meta 字段 + overall recall_at_10)
- Test: `tests/test_eval_artifact.py`(新建,纯函数级)

**Interfaces:**
- Produces:
  - `run_retrieval_compare.content_sha256(paths: list[Path]) -> str`(相对路径稳定排序,逐文件 sha256 汇总再 sha256)
  - `run_retrieval_compare.load_frozen_evidence(path: Path) -> dict`
  - `run_retrieval_compare.check_evidence_frozen(settings, artifact) -> None`(版本/权重/阈值任一不一致 → SystemExit 非零,不发布报告)
  - rag_eval.json `meta` 新键:`corpus_mode="knowledge_docs_baseline"`、`corpus_version`、`dataset_version`、`evidence_confidence_version`;`retrieval.hybrid_rerank.overall` 加 `recall_at_10`

- [ ] **Step 1: 写失败测试**

```python
"""ch09 评估版本化:内容哈希稳定;冻结 artifact 与 settings 不一致即拒跑。"""
from pathlib import Path

import pytest

from evals.run_retrieval_compare import content_sha256, check_evidence_frozen


def test_content_sha256_stable_and_order_independent(tmp_path):
    (tmp_path / "b.md").write_text("B", encoding="utf-8")
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    h1 = content_sha256([tmp_path / "a.md", tmp_path / "b.md"])
    h2 = content_sha256([tmp_path / "b.md", tmp_path / "a.md"])
    assert h1 == h2 and len(h1) == 64
    (tmp_path / "a.md").write_text("AA", encoding="utf-8")
    assert content_sha256([tmp_path / "a.md", tmp_path / "b.md"]) != h1


def test_frozen_mismatch_rejected():
    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")  # 默认 uncalibrated
    artifact = {"version": "v2026", "weights": {"top1": 0.6, "count": 0.2,
                "margin": 0.2}, "min_effective_score": 0.05, "threshold": 0.4}
    with pytest.raises(SystemExit):
        check_evidence_frozen(s, artifact)
```

- [ ] **Step 2: 实现**

- `content_sha256`:`for p in sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())` 逐文件 `hashlib.sha256(bytes).hexdigest()` 拼行后再 sha256。
- main():`corpus_version = content_sha256(sorted((ROOT/"knowledge_docs").glob("*.md")))`、`dataset_version = sha256(CASES 文件字节)`;artifact = load_frozen_evidence(ROOT/"evals/calibration/evidence_confidence.json");`check_evidence_frozen(settings, artifact)`(uncalibrated 与 artifact 同容忍:两侧 version 都是 "uncalibrated" 且 artifact 缺失?——契约:**artifact 必须存在且与 settings 完全一致;settings 默认 uncalibrated 时若 artifact 缺失则整轮失败**,提示先跑 --calibrate-evidence。即 Task 9 Step 4 的校准是回归前置。测试期不打主流程:check 只在 main() 夜间/手动回归路径执行,--calibrate-evidence 自身不做 check。)
- 闸值口径:`r["gate"]` 对 hybrid_rerank 改为 `evidence_confidence(r["scores"], frozen_params)`,threshold 用 artifact threshold;`confidence_signals` 选拔段保留(观测用),但 test 指标只用冻结口径——日常回归不再重选信号。
- `build_rag_eval` meta 加四键;`validate_rag_eval` 同步放行/要求;`_test_metrics` overall 加 `recall_at_10`(用既有 section_recall_at_k(...,10) 聚合,与 recall5 并列)。

- [ ] **Step 3: 跑测试 + 回归**

Run: `uv run pytest tests/test_eval_artifact.py -x -q && uv run pytest tests/ -k "rag_eval or report" -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add evals/run_retrieval_compare.py evals/rag_eval_report.py tests/test_eval_artifact.py
git commit -m "feat(ch09): 评估版本化——语料/评估集哈希 + 冻结置信闸口径回归"
```

---

### Task 11: eval_runs 落表回调 + JobRunner triggered_by

**Files:**
- Create: `app/services/eval_runs.py`
- Modify: `app/jobs/runner.py`(`run(name, *, triggered_by="手动")` + on_report_published 回调)
- Modify: `app/main.py`(注册回调)
- Modify: `app/routers/jobs.py`(手动路由传「手动」)
- Modify: `app/routers/rag_eval.py` 或新建路由(`GET /api/eval-runs`)
- Test: `tests/test_eval_runs.py`(新建)

**Interfaces:**
- Consumes: Task 1 EvalRun;Task 10 报告 meta 键。
- Produces:
  - `eval_runs.record_run(session_factory, report: dict, triggered_by: str) -> bool`(True=新插入;False=run_id 已存在幂等跳过)
  - `eval_runs.list_runs(session_factory, limit: int = 90) -> list[dict]`
  - `JobRunner.__init__(..., on_report_published=None)`;`run(name, *, triggered_by: str = "手动")`
  - `GET /api/eval-runs` → `{"runs": [{run_id, triggered_by, dataset_size, corpus_version, dataset_version, metrics, created_at}]}`(按 created_at 升序)

- [ ] **Step 1: 写失败测试**

```python
"""ch09 eval_runs 落表:run_id 幂等;metrics 契约;triggered_by 透传。"""
from app.models import EvalRun
from app.services import eval_runs

REPORT = {"meta": {"run_id": "20261009T030000Z-abc", "test_cases": 150,
                   "corpus_mode": "knowledge_docs_baseline",
                   "corpus_version": "c" * 64, "dataset_version": "d" * 64,
                   "evidence_confidence_version": "v1"},
          "retrieval": {"hybrid_rerank": {"overall": {
              "mrr": 0.71, "recall_at_10": 0.82, "evidence_coverage": 0.9,
              "sr10": 0.8}}},
          "generation": {"per_strategy": {"hybrid_rerank": {
              "faithful_rate": 0.9}}},
          "gates": {"passed": True}}


def test_record_run_inserts_and_dedupes(db_session_factory):
    assert eval_runs.record_run(db_session_factory, REPORT, "手动") is True
    assert eval_runs.record_run(db_session_factory, REPORT, "手动") is False
    with db_session_factory() as s:
        rows = s.query(EvalRun).all()
        assert len(rows) == 1
        m = rows[0].metrics
        assert m["recall_at_10"] == 0.82 and m["mrr"] == 0.71
        assert m["faithfulness"] == 0.9 and m["quality_passed"] is True
        assert m["evidence_confidence_version"] == "v1"
        assert rows[0].triggered_by == "手动"
        assert rows[0].dataset_size == 150


def test_record_run_missing_metrics_null_not_zero(db_session_factory):
    bad = {"meta": {**REPORT["meta"], "run_id": "r2"},
           "retrieval": {"hybrid_rerank": {"overall": {}}},
           "generation": {"per_strategy": {"hybrid_rerank": {}}},
           "gates": {}}
    assert eval_runs.record_run(db_session_factory, bad, "定时") is True
    with db_session_factory() as s:
        row = s.query(EvalRun).filter_by(run_id="r2").first()
        assert row.metrics["mrr"] is None and row.metrics["quality_passed"] is False
```

- [ ] **Step 2: 实现 eval_runs.py**

```python
"""ch09:eval_runs 落表——报告发布回调幂等记录一轮评估。"""

from sqlalchemy.exc import IntegrityError

from app.models import EvalRun

_METRIC_KEYS = ("online_strategy", "recall_at_10", "mrr", "faithfulness",
                "coverage", "sr10", "quality_passed", "evidence_confidence_version")


def _metrics_of(report: dict) -> dict:
    overall = ((report.get("retrieval") or {}).get("hybrid_rerank") or {}).get("overall") or {}
    gen = ((report.get("generation") or {}).get("per_strategy")
           or {}).get("hybrid_rerank") or {}
    meta = report.get("meta") or {}
    gates = report.get("gates") or {}
    return {
        "online_strategy": (report.get("generation") or {}).get("online_strategy")
                           or "hybrid_rerank",
        "recall_at_10": overall.get("recall_at_10"),
        "mrr": overall.get("mrr"),
        "faithfulness": gen.get("faithful_rate"),
        "coverage": overall.get("evidence_coverage"),
        "sr10": overall.get("sr10"),
        "quality_passed": bool(gates.get("passed")),
        "evidence_confidence_version": meta.get("evidence_confidence_version"),
    }


def record_run(session_factory, report: dict, triggered_by: str) -> bool:
    """meta.run_id 唯一幂等;缺失指标存 None,不用 0 伪造(spec §6)。"""
    meta = report.get("meta") or {}
    row = EvalRun(run_id=meta["run_id"], triggered_by=triggered_by,
                  dataset_size=int(meta["test_cases"]),
                  corpus_mode=meta.get("corpus_mode", "knowledge_docs_baseline"),
                  corpus_version=meta["corpus_version"],
                  dataset_version=meta["dataset_version"],
                  metrics=_metrics_of(report))
    with session_factory() as s:
        s.add(row)
        try:
            s.commit()
            return True
        except IntegrityError:
            s.rollback()
            return False


def list_runs(session_factory, limit: int = 90) -> list[dict]:
    with session_factory() as s:
        rows = (s.query(EvalRun).order_by(EvalRun.created_at.desc(),
                                          EvalRun.id.desc()).limit(limit).all())
    out = [{"run_id": r.run_id, "triggered_by": r.triggered_by,
            "dataset_size": r.dataset_size, "corpus_mode": r.corpus_mode,
            "corpus_version": r.corpus_version, "dataset_version": r.dataset_version,
            "metrics": r.metrics,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
           for r in rows]
    return out[::-1]  # 升序供趋势图直接画
```

- [ ] **Step 3: JobRunner 回调 + 接线**

`runner.py`:
- `__init__` 加 `on_report_published=None` 存 `self._on_report`。
- `run(self, name, *, triggered_by="手动")`,JobInfo 加字段 `triggered_by: str = "手动"`;_exec 登记时带上。
- `_exec` 在 `job.status = "ok"` 分支(`run_id and run_id != prev_run_id`)内追加:

```python
                if self._on_report is not None:
                    try:
                        self._on_report(report, job.triggered_by)   # 失败只记日志,不翻作业状态
                    except Exception:
                        import logging
                        logging.getLogger(__name__).exception(
                            "on_report_published failed job=%s", job.name)
```

`status()` 返回加 `"triggered_by": job.triggered_by`。

`main.py` 注册处:

```python
    job_runner = JobRunner(log_dir=..., report_loader=_report_loader, cwd=root_dir,
                           on_report_published=(
                               (lambda report, by: eval_runs_service.record_run(
                                   runtime.session_factory, report, by))
                               if runtime.session_factory is not None else None))
```

import `from app.services import eval_runs as eval_runs_service`。

`routers/jobs.py` `job_run` 改 `await runner.run(name, triggered_by="手动")`。

`GET /api/eval-runs`:挂到 `routers/rag_eval.py`(加路由函数,`asyncio.to_thread(eval_runs_service.list_runs, request.app.state.session_factory)`),路径 `/api/eval-runs`(router 无 prefix 则直接 `@router.get("/api/eval-runs")`——先看该 router 是否有 prefix,执行者照实接线)。

- [ ] **Step 4: 测试 + 回归**

`tests/test_eval_runs.py` 再补 JobRunner 回调用例(fake report_loader 返回新 run_id,fake on_report 记录调用;`asyncio.run(runner.run(...))` 后等 `_exec` task 完成—— runner._exec 是 create_task,测试里 `await asyncio.sleep(0.2)` 或直接调用 `runner._exec(job)` 内部逻辑;执行者照 runner 现有测试文件 tests/test_jobs*.py 的同款等待手法):

Run: `uv run pytest tests/test_eval_runs.py -x -q && uv run pytest tests/ -k "jobs or rag_eval" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/eval_runs.py app/jobs/runner.py app/main.py app/routers/jobs.py app/routers/rag_eval.py tests/test_eval_runs.py
git commit -m "feat(ch09): eval_runs 落表回调(run_id 幂等)+ triggered_by 透传"
```

---

### Task 12: 评估定时调度器

**Files:**
- Create: `app/services/eval_scheduler.py`
- Modify: `app/config.py`(EVAL_SCHEDULE_*)
- Modify: `app/main.py`(lifespan 挂 task)
- Test: `tests/test_eval_scheduler.py`(新建)

**Interfaces:**
- Produces:
  - `next_run_at(now: datetime, hour: int, tz: ZoneInfo) -> datetime`(严格大于 now 的下一个本地 hour:00;DST 重复时同一自然日只一次)
  - `eval_scheduler_loop(job_runner, settings, *, sleep=asyncio.sleep) -> None`(可取消;每轮到点 `await job_runner.run("eval-rag", triggered_by="定时")`;异常记日志算下一天,不向 lifespan 传播)
  - `Settings.eval_schedule_enabled: bool = True`、`eval_schedule_hour: int = 3`、`eval_schedule_timezone: str = "Asia/Shanghai"`

- [ ] **Step 1: 写失败测试**

```python
"""ch09 评估定时:下一触发点计算;开关;异常后续调度。"""
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.eval_scheduler import eval_scheduler_loop, next_run_at

TZ = ZoneInfo("Asia/Shanghai")


def test_next_run_at_same_day_and_next_day():
    now = datetime(2026, 10, 9, 2, 30, tzinfo=TZ)
    assert next_run_at(now, 3, TZ) == datetime(2026, 10, 9, 3, 0, tzinfo=TZ)
    now2 = datetime(2026, 10, 9, 3, 0, tzinfo=TZ)
    assert next_run_at(now2, 3, TZ) == datetime(2026, 10, 10, 3, 0, tzinfo=TZ)


def test_loop_triggers_once_then_cancel():
    fired = []

    class FakeRunner:
        async def run(self, name, *, triggered_by="手动"):
            fired.append((name, triggered_by))
            return True

    sleeps = []

    async def fake_sleep(sec):
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d",
                 eval_schedule_enabled=True, eval_schedule_hour=3)
    task = asyncio.get_event_loop().create_task(
        eval_scheduler_loop(FakeRunner(), s, sleep=fake_sleep))
    asyncio.get_event_loop().run_until_complete(task)
    assert fired == [("eval-rag", "定时")]
```

(fake_sleep 第二次即 CancelledError 模拟关闭;loop 第一轮到点触发一次。实现细节:`eval_scheduler_loop` 用注入的 `now()` 也可——为可测性加 `now=lambda: datetime.now(tz)` 参数,执行者按此定型,测试注入固定时钟。)

- [ ] **Step 2: 实现 eval_scheduler.py**

```python
"""ch09:评估定时调度(lifespan 单 task,不引 APScheduler)。"""

import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def next_run_at(now: datetime, hour: int, tz: ZoneInfo) -> datetime:
    """严格大于 now 的下一个本地 hour:00(DST 重复同日只一次)。"""
    local = now.astimezone(tz)
    candidate = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate = (local + timedelta(days=1)).replace(
            hour=hour, minute=0, second=0, microsecond=0)
    return candidate


async def eval_scheduler_loop(job_runner, settings, *,
                              sleep=asyncio.sleep, now=None) -> None:
    tz = ZoneInfo(settings.eval_schedule_timezone)
    clock = now or (lambda: datetime.now(tz))
    while True:
        target = next_run_at(clock(), settings.eval_schedule_hour, tz)
        await sleep(max((target - clock()).total_seconds(), 0.1))
        try:
            await job_runner.run("eval-rag", triggered_by="定时")
        except Exception:
            logger.exception("eval scheduler run failed")  # 不传播,算下一天
```

- [ ] **Step 3: config + lifespan 接线**

config.py ch09 段加:

```python
    eval_schedule_enabled: bool = True          # 每天定时跑 eval-rag(烧额度,可关)
    eval_schedule_hour: int = Field(default=3, ge=0, le=23)
    eval_schedule_timezone: str = "Asia/Shanghai"
```

`main.py` lifespan:`yield` 之前(start 段)加:

```python
        sched_task = None
        if owns_runtime and settings.eval_schedule_enabled:
            from app.services.eval_scheduler import eval_scheduler_loop
            sched_task = asyncio.create_task(
                eval_scheduler_loop(app.state.job_runner, settings))
```

`yield` 之后(关闭段)加:

```python
        if sched_task is not None:
            sched_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sched_task
```

(main.py 顶部已 import asyncio;需补 `import contextlib`——检查现有 import,有则复用。)

`.env.example` ch09 段加 `# EVAL_SCHEDULE_ENABLED=true` / `# EVAL_SCHEDULE_HOUR=3` / `# EVAL_SCHEDULE_TIMEZONE=Asia/Shanghai`。

- [ ] **Step 4: 测试 + 回归 + Commit**

Run: `uv run pytest tests/test_eval_scheduler.py -x -q`

```bash
git add app/services/eval_scheduler.py app/config.py app/main.py .env.example tests/test_eval_scheduler.py
git commit -m "feat(ch09): 评估定时调度(本地时区每日一轮,可关)"
```

---

### Task 13: 飞轮 worker(标准化 + 查重 + 退避 + Event 唤醒)

**Files:**
- Create: `app/prompts/flywheel.py`(STANDARDIZE_PROMPT / DEDUP_PROMPT)
- Create: `app/services/flywheel.py`
- Modify: `app/config.py`(FLYWHEEL_* / REVIEW_QUEUE_MATCH_LIMIT)
- Modify: `app/graph/nodes.py`(GraphDeps 加 flywheel 字段;log 落池后 notify)
- Modify: `app/main.py`(lifespan 装配 worker;app.state.flywheel_worker)
- Test: `tests/test_flywheel.py`(新建)

**Interfaces:**
- Consumes: Task 1 lcq/ReviewQueue 模型;Task 5 落池字段。
- Produces:
  - `parse_standardize_output(text) -> tuple[str, str] | None`(`{"normalized_question","suggested_answer"}`,两键非空字符串否则 None)
  - `parse_dedup_output(text, candidate_ids: set[int]) -> int | None | False`(命中返回 id;`{"matched_id": null}` → None 新建;解析失败/id 不在候选集 → False 视为失败重试)
  - `backoff_seconds(settings, attempt_count) -> float`(`min(max, base * 2 ** (attempt_count - 1))`)
  - `async process_pending(settings, session_factory, model) -> dict`(stats: processed/merged/created/retried/failed)
  - `class FlywheelWorker(settings, session_factory, model)`:`start()` / `notify()` / `aclose()`;`run_now() -> None` 供 API 202(内部 notify)
  - `GraphDeps.flywheel: Any = None`(log 节点落池成功后 `deps.flywheel.notify()`)

- [ ] **Step 1: 写失败测试(解析与退避纯函数 + DB 流程)**

```python
"""ch09 飞轮:解析契约/退避/同义合并累加/新建缺口/失败退避与到限 failed。"""
import asyncio
import json

from app.services.flywheel import (
    backoff_seconds, parse_dedup_output, parse_standardize_output, process_pending,
)
from app.models import LowConfidenceQuestion, ReviewQueue
from app.config import Settings


def _settings(**kw):
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d",
                flywheel_retry_base_seconds=60, flywheel_retry_max_seconds=3600)
    base.update(kw)
    return Settings(**base)


def test_parse_standardize():
    assert parse_standardize_output(
        '{"normalized_question": "如何开发票?", "suggested_answer": "在订单页…"}'
    ) == ("如何开发票?", "在订单页…")
    assert parse_standardize_output("not json") is None
    assert parse_standardize_output('{"normalized_question": ""}') is None


def test_parse_dedup():
    assert parse_dedup_output('{"matched_id": 3}', {3, 5}) == 3
    assert parse_dedup_output('{"matched_id": null}', {3}) is None
    assert parse_dedup_output('{"matched_id": 99}', {3}) is False   # 不在候选集
    assert parse_dedup_output("garbage", {3}) is False


def test_backoff():
    s = _settings()
    assert backoff_seconds(s, 1) == 60
    assert backoff_seconds(s, 2) == 120
    assert backoff_seconds(s, 99) == 3600


class _FakeModel:
    """标准化/查重脚本化应答:按输入内容分流。"""
    def __init__(self, std_map, dedup_id):
        self._std = std_map
        self._dedup_id = dedup_id

    async def ainvoke(self, messages):
        text = messages[0].content
        if "待审队列候选" in text:
            return type("R", (), {"content": json.dumps(
                {"matched_id": self._dedup_id})})()
        q = self._std.get(text, ("标准化问题?", "示例答案。"))
        return type("R", (), {"content": json.dumps(
            {"normalized_question": q[0], "suggested_answer": q[1]})})()


def _seed_lcq(sf, raw="发票咋开啊", resolved="如何开具发票?"):
    with sf() as s:
        row = LowConfidenceQuestion(raw_question=raw, source="self_check",
                                    resolved_question=resolved)
        s.add(row)
        s.commit()
        return row.id


def test_new_gap_creates_review_row(db_session_factory):
    _seed_lcq(db_session_factory)
    model = _FakeModel({}, None)
    stats = asyncio.run(process_pending(_settings(), db_session_factory, model))
    assert stats["created"] == 1
    with db_session_factory() as s:
        rq = s.query(ReviewQueue).first()
        lcq = s.query(LowConfidenceQuestion).first()
        assert rq.normalized_question == "标准化问题?"
        assert rq.occurrence_count == 1 and rq.review_status == "待审"
        assert lcq.process_status == "processed"
        assert lcq.matched_review_id == rq.id


def test_synonym_merges_and_counts(db_session_factory):
    _seed_lcq(db_session_factory)
    _seed_lcq(db_session_factory, raw="开发票在哪里", resolved="在哪里开发票?")
    model = _FakeModel({}, None)
    asyncio.run(process_pending(_settings(), db_session_factory, model))
    with db_session_factory() as s:
        rid = s.query(ReviewQueue).first().id
    model2 = _FakeModel({}, rid)   # 第二条查重命中第一条
    stats = asyncio.run(process_pending(_settings(), db_session_factory, model2))
    assert stats["merged"] == 1
    with db_session_factory() as s:
        assert s.query(ReviewQueue).count() == 1
        assert s.query(ReviewQueue).first().occurrence_count == 2
        assert s.query(LowConfidenceQuestion).filter_by(
            process_status="processed").count() == 2


def test_failure_backoff_and_terminal_failed(db_session_factory):
    _seed_lcq(db_session_factory)

    class BadModel:
        async def ainvoke(self, messages):
            return type("R", (), {"content": "garbage"})()

    s = _settings(flywheel_max_attempts=2)
    asyncio.run(process_pending(s, db_session_factory, BadModel()))
    with db_session_factory() as ss:
        row = ss.query(LowConfidenceQuestion).first()
        assert row.process_status == "pending" and row.attempt_count == 1
        assert row.next_attempt_at is not None and row.last_error
        row.next_attempt_at = None   # 测试直接放行到期的等待
        ss.commit()
    asyncio.run(process_pending(s, db_session_factory, BadModel()))
    with db_session_factory() as ss:
        row = ss.query(LowConfidenceQuestion).first()
        assert row.process_status == "failed" and row.next_attempt_at is None
```

注意 `_FakeModel` 靠 prompt 里是否含「待审队列候选」区分两次调用——**执行者写 prompt 时必须让 DEDUP_PROMPT 含该字面标记**,或改用调用计数分流,测试相应调整。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_flywheel.py -x -q`
Expected: FAIL

- [ ] **Step 3: 实现 prompts + flywheel.py**

`app/prompts/flywheel.py`:

```python
"""ch09 飞轮 prompt:标准化(口语→FAQ 式)与查重(同义归并)。输出契约单行 JSON。"""

STANDARDIZE_PROMPT = """你是电商客服知识库编辑。把用户原话改写成一条标准 FAQ 式问题,并给一条示例答案备查。

指代消解后的问题:{resolved_question}
用户原话:{raw_question}

要求:
- normalized_question:书面化、完整、不含情绪口语,不超过 60 字
- suggested_answer:基于常识的示例答案,仅供审核参考,不超过 200 字
- 只输出一行 JSON:{"normalized_question": "...", "suggested_answer": "..."}"""

DEDUP_PROMPT = """判断「新问题」与待审队列候选是否同一个意思(同义即可,不要求字面相同)。

新问题:{normalized_question}

待审队列候选(格式 id: 问题):
{candidates}

只输出一行 JSON:同义返回 {"matched_id": <候选id>},都不同返回 {"matched_id": null}
matched_id 必须来自上面候选列表,禁止发明 id。"""
```

`app/services/flywheel.py` 要点(完整实现):

```python
"""ch09 飞轮流水线:lcq pending → 标准化 → 查重 → review_queue。

lifespan 单 worker;Event 唤醒 + DB 退避到期自醒;每条 lcq 独立事务;
写回事务行锁复核状态(spec §5.4)。同步 DB 经 asyncio.to_thread,模型异步 await。
"""

import asyncio
import contextlib
import json
import logging
import re
from datetime import timedelta

from langchain_core.messages import HumanMessage
from sqlalchemy import func, or_, select

from app.config import Settings
from app.models import LowConfidenceQuestion, ReviewQueue
from app.prompts.flywheel import DEDUP_PROMPT, STANDARDIZE_PROMPT

logger = logging.getLogger(__name__)


def parse_standardize_output(text) -> tuple[str, str] | None: ...
    # 照 nodes.py parse_intent_output 同款正则抠 JSON;两键非空 str 否则 None

def parse_dedup_output(text, candidate_ids: set[int]):
    # 命中 id 须在 candidate_ids;{"matched_id": null} → None;其余 → False

def backoff_seconds(settings, attempt_count: int) -> float:
    return min(settings.flywheel_retry_max_seconds,
               settings.flywheel_retry_base_seconds * 2 ** (attempt_count - 1))
```

核心 DB 函数(全部同步,to_thread 调用):

```python
def _fetch_due(sf, limit: int) -> list[LowConfidenceQuestion]:
    with sf() as s:
        return list(s.query(LowConfidenceQuestion).filter(
            LowConfidenceQuestion.process_status == "pending",
            LowConfidenceQuestion.matched_review_id.is_(None),
            or_(LowConfidenceQuestion.next_attempt_at.is_(None),
                LowConfidenceQuestion.next_attempt_at <= func.now()),
        ).order_by(LowConfidenceQuestion.id).limit(limit).all())


def _fetch_candidates(sf, limit: int) -> list[ReviewQueue]:
    with sf() as s:
        return list(s.query(ReviewQueue)
                    .filter_by(review_status="待审")
                    .order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
                    .limit(limit).all())


def _commit_match(sf, lcq_id: int, normalized: str, suggested: str,
                  match_id: int | None) -> str:
    """写回:锁 lcq 复核 pending/未归并;命中则锁目标复核仍待审。
    返回 "merged" | "created" | "skipped"(状态已变,重放安全)。"""
    with sf() as s:
        row = s.query(LowConfidenceQuestion).filter_by(id=lcq_id).with_for_update().first()
        if row is None or row.process_status != "pending" \
                or row.matched_review_id is not None:
            return "skipped"
        if match_id is not None:
            target = s.query(ReviewQueue).filter_by(id=match_id).with_for_update().first()
            if target is None or target.review_status != "待审":
                raise RetryDedupError("target state changed")   # 外层重新读候选再判
            target.occurrence_count += 1
            rid = target.id
        else:
            target = ReviewQueue(normalized_question=normalized,
                                 ai_suggested_answer=suggested)
            s.add(target)
            s.flush()
            rid = target.id
        row.matched_review_id = rid
        row.process_status = "processed"
        row.next_attempt_at = None
        row.last_error = None
        s.commit()
        return "merged" if match_id is not None else "created"


def _mark_failure(sf, lcq_id: int, exc: Exception, settings) -> None:
    with sf() as s:
        row = s.query(LowConfidenceQuestion).filter_by(id=lcq_id).with_for_update().first()
        if row is None or row.process_status != "pending":
            return
        row.attempt_count += 1
        row.last_error = f"{type(exc).__name__}: {exc}"[:500]
        if row.attempt_count >= settings.flywheel_max_attempts:
            row.process_status = "failed"
            row.next_attempt_at = None
        else:
            secs = backoff_seconds(settings, row.attempt_count)
            row.next_attempt_at = func.now() + ...  # 用 DB 钟:
        s.commit()
```

退避写库用 DB 钟:`row.next_attempt_at = s.query(func.date_add(func.now(), text(f"INTERVAL {int(secs)} SECOND"))).scalar()` 或更简单 `func.now()` + 应用层读时容忍——**执行者选 SQLAlchemy 可移植写法 `func.dateadd`?MySQL 方言是 `func.date_add(func.now(), text('INTERVAL n SECOND'))`,执行者按此实现并在测试里断言 next_attempt_at > created_at**。

`process_pending` 主循环:

```python
class RetryDedupError(Exception):
    pass


async def process_pending(settings, session_factory, model) -> dict:
    stats = {"processed": 0, "merged": 0, "created": 0, "retried": 0, "failed": 0}
    while True:
        rows = await asyncio.to_thread(_fetch_due, session_factory,
                                       settings.flywheel_batch_size)
        if not rows:
            return stats
        for row in rows:
            try:
                norm = await _standardize(model, row)
                candidates = await asyncio.to_thread(
                    _fetch_candidates, session_factory,
                    settings.review_queue_match_limit)
                match_id = await _dedup(model, norm[0], candidates)
                outcome = await asyncio.to_thread(
                    _commit_match, session_factory, row.id, norm[0], norm[1], match_id)
                stats["processed"] += outcome in ("merged", "created")
                stats[outcome] += 1 if outcome in ("merged", "created") else 0
            except RetryDedupError:
                try:   # 候选状态变了:重读候选重判一次,仍失败走退避
                    candidates = await asyncio.to_thread(
                        _fetch_candidates, session_factory,
                        settings.review_queue_match_limit)
                    match_id = await _dedup(model, norm[0], candidates)
                    outcome = await asyncio.to_thread(
                        _commit_match, session_factory, row.id, norm[0], norm[1],
                        match_id)
                    stats["processed"] += 1
                    stats[outcome] += 1
                except Exception as exc:
                    await _fail(session_factory, settings, row.id, exc, stats)
            except Exception as exc:
                await _fail(session_factory, settings, row.id, exc, stats)
```

`_standardize`:resolved_question 为主(缺失用 raw),`model.ainvoke([HumanMessage(content=prompt)])`,parse 失败 raise ValueError。`_dedup`:候选为空直接 None;否则拼候选文本调用模型,parse_dedup_output False → raise ValueError。

`FlywheelWorker`(spec §5.4 唤醒语义:先清 Event 再扫描;无到期行查最早 next_attempt_at 定时醒;无未来行只等 Event):

```python
class FlywheelWorker:
    def __init__(self, settings, session_factory, model):
        self._settings = settings
        self._sf = session_factory
        self._model = model
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def notify(self) -> None:
        self._event.set()

    def start(self) -> None:
        if self._task is None and self._sf is not None:
            self._task = asyncio.create_task(self._loop())

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def run_now(self) -> None:
        self.notify()

    async def _loop(self) -> None:
        while True:
            self._event.clear()          # 先清再扫,等待建立期间的通知不丢
            try:
                stats = await process_pending(self._settings, self._sf, self._model)
                if any(stats.values()):
                    logger.info("flywheel pass %s", stats)
                wait = await asyncio.to_thread(self._next_due_in)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("flywheel loop error")
                wait = min(self._settings.flywheel_retry_max_seconds, 60.0)
            try:
                if wait is None:
                    await self._event.wait()
                else:
                    await asyncio.wait_for(self._event.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass

    def _next_due_in(self) -> float | None:
        with self._sf() as s:
            nxt = (s.query(func.min(LowConfidenceQuestion.next_attempt_at))
                   .filter(LowConfidenceQuestion.process_status == "pending",
                           LowConfidenceQuestion.next_attempt_at.isnot(None))
                   .scalar())
        if nxt is None:
            return None
        from datetime import datetime
        return max((nxt - datetime.now()).total_seconds(), 0.5)
```

config.py ch09 段加:

```python
    flywheel_batch_size: int = Field(default=50, gt=0)
    review_queue_match_limit: int = Field(default=200, gt=0)
    flywheel_retry_base_seconds: float = Field(default=60, gt=0)
    flywheel_retry_max_seconds: float = Field(default=3600, gt=0)
    flywheel_max_attempts: int = Field(default=5, ge=1)
```

`nodes.py`:`GraphDeps` 加 `flywheel: Any = None  # ch09 飞轮(落池后 notify)`;`log_turn` 在 commit 成功后(`result = await ...` 之后)加:

```python
        if low_conf is not None and deps.flywheel is not None:
            deps.flywheel.notify()   # ch09:落池即唤醒飞轮(Event 加速,DB 为准)
```

`main.py`:装配 `flywheel_worker = FlywheelWorker(settings, runtime.session_factory, model)` → `app.state.flywheel_worker`;GraphDeps 加 `flywheel=flywheel_worker`;lifespan start 段 `flywheel_worker.start()`、关闭段 `await flywheel_worker.aclose()`。

`.env.example` ch09 段加对应五行注释。

- [ ] **Step 4: 测试 + 回归**

Run: `uv run pytest tests/test_flywheel.py -x -q && uv run pytest tests/ -k "log or low_conf" -q`
Expected: PASS

- [ ] **Step 5: 标注样例验证(纯 Prompt 任务,替代单测字面断言)**

Create `evals/flywheel_samples.jsonl`(10~15 条人工标注:`{"raw","resolved","expect_normalized_contains","candidates":[{"id","question"}],"expect_match_id|null"}`),Create `evals/probe_flywheel.py`(真模型跑样例,打印命中率;手跑不进 pytest,照 evals/probe_intent.py 同款骨架)。本步骤只建文件;实跑放 Task 16 验收。

- [ ] **Step 6: Commit**

```bash
git add app/prompts/flywheel.py app/services/flywheel.py app/config.py app/graph/nodes.py app/main.py .env.example tests/test_flywheel.py evals/flywheel_samples.jsonl evals/probe_flywheel.py
git commit -m "feat(ch09): 飞轮 worker——标准化/查重/退避/Event 唤醒"
```

---

### Task 14: 飞轮/审核 API(列表/详情/approve/reject/run/failures/retry)

**Files:**
- Create: `app/services/review_service.py`
- Create: `app/routers/review.py`
- Modify: `app/schemas.py`(ReviewApproveRequest)
- Modify: `app/main.py`(include router)
- Test: `tests/test_review_api.py`(新建)

**Interfaces:**
- Consumes: Task 13 FlywheelWorker。
- Produces(API 契约,错误 `{"error":{"code","message"}}`):
  - `GET /api/review?status=待审|写入中|通过|驳回&page=1` → `{"items":[{id,normalized_question,occurrence_count,ai_suggested_answer,review_status,last_write_error,created_at}],"total":n}`
  - `GET /api/review/{id}/detail` → 上项 + `sources:[{lcq_id,raw_question,resolved_question,source,retrieved_chunks,created_at}]`
  - `POST /api/review/{id}/approve` 在 **Task 15** 才暴露(冻结事务+向量化+全局知识库锁);本任务只做读路径/reject/飞轮三端点
  - `POST /api/review/{id}/reject` → `{"review_status":"驳回"}`;非待审 409
  - `POST /api/flywheel/run` → 202 `{"notified":true}`
  - `GET /api/flywheel/failures?page=1` → `{"items":[{id,raw_question,source,attempt_count,last_error,created_at}],"total":n}`
  - `POST /api/flywheel/questions/{id}/retry` → 202;非 failed 409
  - service 层:`review_service.list_reviews(sf,status,page,page_size)` / `get_review_detail(sf,review_id)` / `reject(sf,review_id)` / `retry_failed_question(sf,lcq_id)`;approve 在 Task 15。

- [ ] **Step 1: 写失败测试(服务+路由两级)**

边界说明:本任务只做读路径 + reject + flywheel run/failures/retry;approve(冻结事务+向量化+知识库锁)整个在 Task 15 实现并暴露端点。

```python
"""ch09 审核读路径与飞轮 API。"""
import asyncio

from app.models import LowConfidenceQuestion, ReviewQueue
from app.services import review_service


def _seed(sf):
    with sf() as s:
        rq = ReviewQueue(normalized_question="如何开发票?", occurrence_count=2,
                         ai_suggested_answer="示例")
        s.add(rq)
        s.flush()
        lcq = LowConfidenceQuestion(raw_question="发票咋开", source="user_feedback",
                                    resolved_question="如何开发票?",
                                    retrieved_chunks=[{"chunk_id": 1, "score": 0.3}],
                                    process_status="processed",
                                    matched_review_id=rq.id)
        s.add(lcq)
        bad = LowConfidenceQuestion(raw_question="坏问题", source="self_check",
                                    process_status="failed", attempt_count=5,
                                    last_error="ValueError: parse")
        s.add(bad)
        s.commit()
        return rq.id, lcq.id, bad.id


def test_list_and_detail(db_session_factory):
    rid, lcq_id, _ = _seed(db_session_factory)
    out = review_service.list_reviews(db_session_factory, "待审", 1, 20)
    assert out["total"] == 1 and out["items"][0]["occurrence_count"] == 2
    detail = review_service.get_review_detail(db_session_factory, rid)
    assert detail["sources"][0]["raw_question"] == "发票咋开"
    assert detail["sources"][0]["retrieved_chunks"] == [{"chunk_id": 1, "score": 0.3}]


def test_reject_only_from_pending(db_session_factory):
    rid, _, _ = _seed(db_session_factory)
    out = review_service.reject(db_session_factory, rid)
    assert out["review_status"] == "驳回"
    import pytest
    from app.services.review_service import ReviewConflictError
    with pytest.raises(ReviewConflictError):
        review_service.reject(db_session_factory, rid)


def test_retry_failed_question_cas(db_session_factory):
    _, _, bad_id = _seed(db_session_factory)
    review_service.retry_failed_question(db_session_factory, bad_id)
    with db_session_factory() as s:
        row = s.query(LowConfidenceQuestion).get(bad_id) if hasattr(
            s.query(LowConfidenceQuestion), "get") else s.get(
            LowConfidenceQuestion, bad_id)
        assert row.process_status == "pending"
        assert row.attempt_count == 0 and row.next_attempt_at is None
        assert row.last_error is None
    import pytest
    from app.services.review_service import ReviewConflictError
    with pytest.raises(ReviewConflictError):   # 非 failed 重复重试 409
        review_service.retry_failed_question(db_session_factory, bad_id)
```

- [ ] **Step 2: 实现 review_service.py(读/reject/retry)+ review.py 路由**

```python
"""ch09 审核队列服务:读路径 + 驳回 + 失败重试;approve 写回在知识库锁任务。"""

from app.models import LowConfidenceQuestion, ReviewQueue

PAGE_SIZE = 20


class ReviewError(Exception):
    code = "review_error"
    status = 500
    def __init__(self, message=None):
        self.message = message or self.code
        super().__init__(self.message)


class ReviewNotFoundError(ReviewError):
    code = "review_not_found"
    status = 404


class ReviewConflictError(ReviewError):
    code = "review_conflict"
    status = 409


def _item(r: ReviewQueue) -> dict:
    return {"id": r.id, "normalized_question": r.normalized_question,
            "occurrence_count": r.occurrence_count,
            "ai_suggested_answer": r.ai_suggested_answer,
            "review_status": r.review_status,
            "approved_answer": r.approved_answer,
            "knowledge_chunk_ids": r.knowledge_chunk_ids,
            "last_write_error": r.last_write_error,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}


def list_reviews(sf, status: str | None, page: int, page_size: int = PAGE_SIZE) -> dict:
    with sf() as s:
        q = s.query(ReviewQueue)
        if status:
            q = q.filter_by(review_status=status)
        total = q.count()
        rows = (q.order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
    return {"items": [_item(r) for r in rows], "total": total}


def get_review_detail(sf, review_id: int) -> dict:
    with sf() as s:
        r = s.get(ReviewQueue, review_id)
        if r is None:
            raise ReviewNotFoundError("review not found")
        srcs = (s.query(LowConfidenceQuestion)
                .filter_by(matched_review_id=review_id)
                .order_by(LowConfidenceQuestion.id).all())
        out = _item(r)
        out["sources"] = [{
            "lcq_id": x.id, "raw_question": x.raw_question,
            "resolved_question": x.resolved_question, "source": x.source,
            "retrieved_chunks": x.retrieved_chunks,
            "created_at": x.created_at.strftime("%Y-%m-%d %H:%M:%S")} for x in srcs]
        return out


def reject(sf, review_id: int) -> dict:
    with sf() as s:
        r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
        if r is None:
            raise ReviewNotFoundError("review not found")
        if r.review_status != "待审":
            raise ReviewConflictError("仅待审可驳回;写入中/通过/驳回不可变更")
        r.review_status = "驳回"
        s.commit()
        return {"review_status": "驳回"}


def retry_failed_question(sf, lcq_id: int) -> None:
    with sf() as s:
        r = s.query(LowConfidenceQuestion).filter_by(id=lcq_id).with_for_update().first()
        if r is None:
            raise ReviewNotFoundError("question not found")
        if r.process_status != "failed":
            raise ReviewConflictError("仅 failed 可人工重试")
        r.process_status = "pending"
        r.attempt_count = 0
        r.next_attempt_at = None
        r.last_error = None
        s.commit()


def list_failed_questions(sf, page: int, page_size: int = PAGE_SIZE) -> dict:
    with sf() as s:
        q = s.query(LowConfidenceQuestion).filter_by(process_status="failed")
        total = q.count()
        rows = (q.order_by(LowConfidenceQuestion.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
    return {"items": [{"id": r.id, "raw_question": r.raw_question,
                       "source": r.source, "attempt_count": r.attempt_count,
                       "last_error": r.last_error,
                       "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                      for r in rows], "total": total}
```

`routers/review.py`:薄壳照 kb.py 惯例(`asyncio.to_thread` 包同步 service;flywheel run/retry 成功后 `request.app.state.flywheel_worker.notify()` 返回 202);错误映射:ReviewNotFoundError 404 / ReviewConflictError 409(main.py 注册 handler,照 KbAdminError 同款——其实可直接复用:让 ReviewError 继承 KbAdminError?**执行者选择**:最简是让 main.py 加 `app.exception_handler(ReviewError)` 用 `exc.status/exc.code`,与 KbAdminError handler 同形)。

`main.py` include_router(review_router)。

- [ ] **Step 3: 测试 + 回归 + Commit**

Run: `uv run pytest tests/test_review_api.py -x -q`

```bash
git add app/services/review_service.py app/routers/review.py app/schemas.py app/main.py tests/test_review_api.py
git commit -m "feat(ch09): 审核读路径/驳回/飞轮触发与失败重试 API"
```

---

### Task 15: 审核写回(冻结事务 + 知识库全局锁 + reset/rebuild 重放)

**Files:**
- Modify: `app/services/kb_admin.py`(_job_lock 公开为 job_lock;manual_ingest 纳入锁;reset 选择性删除改 knowledge_docs 前缀;rebuild 调重放)
- Modify: `app/services/review_service.py`(approve/retry_approve/replay_reviews_locked)
- Test: `tests/test_review_approve.py`(新建)

**Interfaces:**
- Consumes: Task 14 review_service;`kb_admin._job_lock`;`ingest.vectorize_pending`。
- Produces:
  - `kb_admin.job_lock()` contextmanager(公开;原 `_job_lock` 保留别名或直接改名,全部现有调用点同步)
  - `review_service.approve(settings, sf, embed, store, review_id, approved_answer) -> dict`——锁序:知识库锁 → review 行锁;冻结事务(source_doc=`review:<id>`、chunk_index 从 1、prev/next、category="审核补充"、content_type="faq");向量化+CAS 通过;失败留写入中+last_write_error
  - `review_service.replay_reviews_locked(settings, sf, embed, store) -> None`(rebuild 持锁调用,不递归取锁)
  - reset 契约:只删 `knowledge_docs/` 前缀 source_doc

- [ ] **Step 1: 写失败测试**

```python
"""ch09 审核写回:冻结→写入中→通过;并发/重试/驳回约束;reset 保留 review 块。"""
import asyncio

import pytest

from app.models import KnowledgeChunk, ReviewQueue
from app.services import kb_admin, review_service
from app.services.review_service import ReviewConflictError


class _FakeEmbed:
    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _FakeStore:
    def __init__(self):
        self.dim = 1024
        self.ids = set()

    def ensure_collection(self):
        pass

    def upsert(self, payloads):
        self.ids |= {p[0] for p in payloads}

    def all_ids(self):
        return set(self.ids)

    def delete_by_ids(self, ids):
        self.ids -= set(ids)


def _settings():
    from app.config import Settings
    return Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                    database_url="mysql+pymysql://u:p@h/d")


def _seed(sf):
    with sf() as s:
        r = ReviewQueue(normalized_question="如何开发票?",
                        ai_suggested_answer="示例答案")
        s.add(r)
        s.commit()
        return r.id


def test_approve_full_cycle(db_session_factory):
    rid = _seed(db_session_factory)
    store = _FakeStore()
    out = review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                 store, rid, "核准答案正文")
    assert out["review_status"] == "通过"
    with db_session_factory() as s:
        r = s.get(ReviewQueue, rid)
        assert r.approved_answer == "核准答案正文" and r.approved_at is not None
        chunks = s.query(KnowledgeChunk).filter_by(source_doc=f"review:{rid}").all()
        assert len(chunks) >= 1
        assert all(c.vectorize_status == "done" for c in chunks)
        assert chunks[0].content_type == "faq" and chunks[0].category == "审核补充"
        assert r.knowledge_chunk_ids == [c.id for c in chunks]
        assert store.ids >= {c.id for c in chunks}   # 向量已 upsert


def test_approve_conflict_rules(db_session_factory):
    rid = _seed(db_session_factory)
    with pytest.raises(ReviewConflictError):   # 首次通过必须给答案
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid, None)
    review_service.reject(db_session_factory, rid)
    with pytest.raises(ReviewConflictError):   # 驳回不可 approve
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid, "答")
    rid2 = _seed(db_session_factory)
    review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                           _FakeStore(), rid2, "答A")
    with pytest.raises(ReviewConflictError):   # 已通过改答案 409
        review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                               _FakeStore(), rid2, "答B")
    out = review_service.approve(_settings(), db_session_factory, _FakeEmbed(),
                                 _FakeStore(), rid2, "答A")   # 同答案幂等
    assert out["review_status"] == "通过"


def test_reset_preserves_review_chunks(db_session_factory):
    rid = _seed(db_session_factory)
    store = _FakeStore()
    review_service.approve(_settings(), db_session_factory, _FakeEmbed(), store,
                           rid, "核准答案")
    with db_session_factory() as s:   # 塞一个文档来源块
        s.add(KnowledgeChunk(category="c", questions="q", answer="a",
                             source_doc="knowledge_docs/product-faq.md",
                             chunk_index=1, vectorize_status="done"))
        s.commit()
    kb_admin.reset_kb(_settings(), db_session_factory, _FakeEmbed(), store)
    with db_session_factory() as s:
        docs = s.query(KnowledgeChunk).filter(
            KnowledgeChunk.source_doc.like("knowledge_docs/%")).all()
        review_rows = s.query(KnowledgeChunk).filter_by(
            source_doc=f"review:{rid}").all()
    assert docs == []            # 文档块已清(随后 run_ingest 会因目录无文档失败——见下)
    assert review_rows           # review 来源必须保留
```

注意最后这个测试:reset_kb 后半段会 run_ingest 重灌 knowledge_docs——测试库里 knowledge_docs 真实存在会真切块入库(embed 假、store 假没关系,run_ingest 走真目录)。**执行者把 docs_dir 指向 tmp_path 空目录?**不行,空目录 IngestError。对策:测试用 monkeypatch 把 `kb_admin.run_ingest` 换成假函数返回 0,隔离真语料。照此修正测试。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_review_approve.py -x -q`
Expected: FAIL(approve 不存在)

- [ ] **Step 3: 实现**

`kb_admin.py`:
- `_job_lock` → `job_lock`(公开),模块内四处调用点同步改名;保留 `_job_lock = job_lock` 兼容别名?不需要——全仓 grep 改名即可。
- `manual_ingest` 主体包 `with job_lock():`(整个插块+向量化;**注意 vectorize_kb 等已持锁函数不被 manual_ingest 调用,无嵌套**)。
- `reset_kb` 删除过滤器两处(`store.delete_by_ids` 的 ids 查询与 MySQL 删除)从 `source_doc.isnot(None)` 改为 `source_doc.like("knowledge_docs/%")`,注释更新为「只清 knowledge_docs 命名空间;review:/manual:/qa_mined 保留」。
- `rebuild_index`:`run_ingest` 成功后、BM25 smoke 前插入:

```python
            from app.services import review_service
            review_service.replay_reviews_locked(settings, session_factory, embed, store)
```

(`rebuild_index` 已持 job_lock;replay 不取锁。)

`review_service.py` 追加:

```python
def _faq_chunks_for(settings, normalized: str, answer: str):
    """复用 ch03 手工 FAQ 切分(frontmatter faq + chunk_document),source 标签 review。"""
    from app.services.kb_admin import _manual_text
    from app.knowledge.chunking import chunk_document
    return chunk_document(_manual_text("faq", normalized, answer),
                          source="review",
                          max_chars=settings.max_chunk_chars,
                          overlap_chars=settings.chunk_overlap_chars)


def _freeze_chunks(s, settings, review: ReviewQueue, answer: str) -> list[int]:
    """冻结事务内:已有 review:<id> 块复用 id(内容须一致),否则新建;返回 chunk ids。"""
    source_doc = f"review:{review.id}"
    existing = (s.query(KnowledgeChunk).filter_by(source_doc=source_doc)
                .order_by(KnowledgeChunk.chunk_index).all())
    if existing:
        return [c.id for c in existing]
    chunks = _faq_chunks_for(settings, review.normalized_question, answer)
    rows = []
    for i, c in enumerate(chunks, start=1):
        row = KnowledgeChunk(category="审核补充", questions=c.questions,
                             answer=c.answer, section_path=c.section_path,
                             content_type="faq", is_key_clause=c.is_key_clause,
                             source_doc=source_doc, chunk_index=i,
                             vectorize_status="pending")
        s.add(row)
        rows.append(row)
    s.flush()
    for i, row in enumerate(rows):
        row.prev_chunk_id = rows[i - 1].id if i > 0 else None
        row.next_chunk_id = rows[i + 1].id if i + 1 < len(rows) else None
    s.flush()
    return [r.id for r in rows]


def approve(settings, sf, embed, store, review_id: int,
            approved_answer: str | None) -> dict:
    """冻结事务(写入中) → 向量化 → CAS 通过;锁序:知识库锁 → review 行锁。"""
    if embed is None:
        raise ReviewConflictError("未配置 EMBEDDING_API_KEY,无法发布到知识库")
    from app.services.kb_admin import job_lock
    with job_lock():   # 与 build/vectorize/mine/reset/rebuild/manual_ingest 互斥
        with sf() as s:
            r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
            if r is None:
                raise ReviewNotFoundError("review not found")
            if r.review_status == "驳回":
                raise ReviewConflictError("已驳回不可通过")
            if r.review_status == "通过":
                if approved_answer and approved_answer.strip() != r.approved_answer:
                    raise ReviewConflictError("已通过不可改答案")
                return {"review_status": "通过",
                        "knowledge_chunk_ids": r.knowledge_chunk_ids}
            if r.review_status == "写入中":
                if approved_answer and approved_answer.strip() != r.approved_answer:
                    raise ReviewConflictError("写入中答案已冻结,仅可原样重试")
            else:   # 待审 → 冻结
                ans = (approved_answer or "").strip()
                if not ans:
                    raise ReviewConflictError("首次通过必须提供核准答案")
                r.approved_answer = ans
            ids = _freeze_chunks(s, settings, r, r.approved_answer)
            r.knowledge_chunk_ids = ids
            r.review_status = "写入中"
            s.commit()   # 冻结事务:状态+答案+块+索引原子提交
        try:
            store.ensure_collection()
            vectorize_pending(settings, sf, embed, store)   # 扫全库 pending;写入中=已核准
        except Exception as exc:
            _save_write_error(sf, review_id, exc)
            raise ReviewWriteError(f"向量化失败,可重试: {type(exc).__name__}") from exc
        with sf() as s:   # 短事务 CAS:写入中 → 通过
            r = s.query(ReviewQueue).filter_by(id=review_id).with_for_update().first()
            if r is None or r.review_status != "写入中":
                raise ReviewConflictError("状态已变更")
            pending = (s.query(KnowledgeChunk)
                       .filter_by(source_doc=f"review:{review_id}",
                                  vectorize_status="pending").count())
            if pending:
                raise ReviewWriteError("仍有 pending 块,重试")
            r.review_status = "通过"
            r.approved_at = func.now()
            r.last_write_error = None
            s.commit()
            return {"review_status": "通过", "knowledge_chunk_ids": r.knowledge_chunk_ids}
```

补 `ReviewWriteError(ReviewError) code="review_write_failed" status=500`、`_save_write_error`(短事务写 last_write_error 截 500)。`replay_reviews_locked`:遍历 （通过， 写入中） 按 id 升序 → 每条 `_freeze_chunks`(rebuild 后表已清,全是新建)→ 刷新 knowledge_chunk_ids → 统一 `vectorize_pending` → 逐条复核无 pending:写入中 CAS 通过（approved_at=func.now())，通过保留原 approved_at（只刷新 chunk ids)。失败抛出让 rebuild 置 REBUILD_REQUIRED(spec §5.6)。

import 区补 `from sqlalchemy import func`、`from app.knowledge.ingest import vectorize_pending`。

- [ ] **Step 4: 测试 + 回归 + Commit**

Run: `uv run pytest tests/test_review_approve.py tests/test_review_api.py -x -q && uv run pytest tests/ -k "kb or ingest or admin" -q`
Expected: PASS

```bash
git add app/services/kb_admin.py app/services/review_service.py tests/test_review_approve.py
git commit -m "feat(ch09): 审核写回——冻结事务/全局知识库锁/reset 保留/rebuild 重放"
```

---

### Task 16: 成本统计 API + /review 页 + rag-eval 趋势/成本区块 + 验收收尾

**Files:**
- Create: `app/services/cost_stats.py`
- Create: `app/routers/stats.py`
- Modify: `app/main.py`(router + /review 静态页)
- Create: `app/static/review.html`(Vibe)
- Modify: `app/static/rag-eval.html`(趋势 + 成本区块,Vibe)
- Modify: `README.md` + `AGENTS.md`(Langfuse 起法、/review、知识库来源说明修订、评估定时)
- Test: `tests/test_cost_stats.py`(新建)、`tests/test_review_page.py`(字符串断言)

**Interfaces:**
- Produces:
  - `GET /api/stats/cost-by-intent?days=7` → `{"days", "incomplete", "missing_usage_observations", "cost_available", "intents": [{"intent","requests","total_tokens","input_tokens","output_tokens","cost"}], "unknown": {...}|null}`;Langfuse 未启用 503 `langfuse_disabled`
  - `GET /api/eval-runs`(Task 11 已建)供趋势图

- [ ] **Step 1: Context7 复核 Langfuse public API(强制)**

查 `/langfuse/langfuse-python` 或 api reference:`GET /api/public/traces`(分页/时间窗过滤参数名、返回是否含 metadata)与 `GET /api/public/observations`(按 trace_id / type=GENERATION 过滤、usageDetails/costDetails 字段名)。把核实结果写在 cost_stats.py docstring。

- [ ] **Step 2: 写失败测试(httpx MockTransport)**

```python
"""ch09 成本统计:按意图聚合/去重/缺失可见/未启用 503。"""
import pytest

from app.services.cost_stats import CostStatsError, cost_by_intent


def _settings(**kw):
    from app.config import Settings
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d",
                langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk",
                model_input_price_per_mtok=1.0, model_output_price_per_mtok=2.0)
    base.update(kw)
    return Settings(**base)


TRACES = {"data": [
    {"id": "t1", "metadata": {"intent": "退款退货"}},
    {"id": "t2", "metadata": {"intent": "闲聊"}},
    {"id": "t3", "metadata": {}},
], "meta": {"totalPages": 1}}

OBS = {"data": [
    {"id": "o1", "type": "GENERATION",
     "usageDetails": {"input": 100, "output": 50},
     "costDetails": {"input": 0.0001, "output": 0.0001}},
    {"id": "o1", "type": "GENERATION",          # 重复行:按 observation_id 去重
     "usageDetails": {"input": 100, "output": 50},
     "costDetails": {"input": 0.0001, "output": 0.0001}},
    {"id": "o2", "type": "GENERATION", "usageDetails": {}, "costDetails": None},
], "meta": {"totalPages": 1}}


def test_aggregate_by_intent(httpx_mock=None, monkeypatch=None):
    import httpx
    from httpx import MockTransport, Response

    def handler(request):
        if "/api/public/traces" in str(request.url):
            return Response(200, json=TRACES)
        return Response(200, json=OBS)

    transport = MockTransport(handler)
    out = cost_by_intent(_settings(), days=7,
                         client_factory=lambda s: httpx.Client(
                             base_url=s.langfuse_host, transport=transport,
                             auth=(s.langfuse_public_key, s.langfuse_secret_key)))
    by = {i["intent"]: i for i in out["intents"]}
    assert by["退款退货"]["requests"] == 1
    assert by["退款退货"]["total_tokens"] == 150     # 去重后 100+50
    assert out["missing_usage_observations"] == 1   # o2 usage 空:可见标记不静默当 0
    assert out["unknown"]["requests"] == 1          # t3 无意图
    assert out["cost_available"] is True


def test_disabled_raises():
    s = _settings(langfuse_enabled=False)
    with pytest.raises(CostStatsError):
        cost_by_intent(s, days=7)
```

- [ ] **Step 3: 实现 cost_stats.py + stats 路由 + main 接线**

```python
"""ch09 成本统计(spec §5.1):Langfuse public API 分页拉取,按 trace metadata.intent 聚合。

数据源:/api/public/traces(分页)+ /api/public/observations?trace_id=&type=GENERATION。
observation 按 id 去重;usage 缺失计入 missing_usage_observations 不静默当 0;
优先 costDetails,缺失按单价折算(token/1e6);检索/工具 span 不计模型 token。
每逻辑 turn(trace)= 1 个 request。页面上限 MAX_TRACES=200 / MAX_OBS=2000,超限 incomplete=true。
(API 字段名以 Context7 核实为准,见 Step 1 记录。)
"""
```

实现要点:`cost_by_intent(settings, days, *, client_factory=None)`;`from_timestamp = (utcnow - days).isoformat`;trace 翻页至 totalPages 或上限;每 trace 拉 observations 翻页;聚合 dict[intent]。`CostStatsError(code="langfuse_disabled"|"langfuse_unavailable", status=503/502)`。main.py 注册 handler + include stats_router;config.py 加 `model_input_price_per_mtok: float = 0.0` / `model_output_price_per_mtok: float = 0.0`;`.env.example` 注释加两行。

- [ ] **Step 4: /review 页 + rag-eval 趋势/成本(Vibe)**

- `app/static/review.html`:照 kb.html 的 CSS tokens 与 api() 封装;状态 tab(待审/写入中/通过/驳回)+ 失败视图 tab;列表行(标准化问题/次数/示例答案摘要/时间);详情抽屉(归并原话 + retrieved_chunks 逐条原文+得分);通过(可编辑核准答案 textarea,默认 ai_suggested_answer)/驳回/写入中重试按钮;失败视图重试按钮;「立即处理」按钮调 POST /api/flywheel/run。
- `rag-eval.html`:「趋势」区块(SVG 折线 recall_at_10/mrr/faithfulness 三线,数据 GET /api/eval-runs,按 corpus_version+dataset_version+evidence_confidence_version 分组,版本变了断线新起一组)+「成本」区块(GET /api/stats/cost-by-intent?days=7,分组柱状图;未启用显示「Langfuse 未启用」)。
- main.py 加 `@app.get("/review")` FileResponse。
- `tests/test_review_page.py`:字符串断言(`"/api/review"`、`"feedback"`…照 test_chat_page 风格,含 `/api/flywheel/run`、趋势区块 mount id)。

- [ ] **Step 5: 测试 + 全量回归**

Run: `uv run pytest tests/test_cost_stats.py tests/test_review_page.py -x -q && uv run pytest -q`
Expected: 全绿(617+ 旧测试 + 新测试)

- [ ] **Step 6: 校准 + 标注样例 + 验收(烧额度,手动依次执行)**

```bash
# 1. 置信闸校准(产出冻结 artifact;无可行点会退出 2 → 停下来问用户)
uv run python evals/run_retrieval_compare.py --calibrate-evidence
#    把 evals/calibration/evidence_confidence.json 的 weights/threshold/version
#    回填 config.py 默认值(注释注明校准日期与 d_pass/pass_rate),再跑全量 pytest
# 2. 飞轮标注样例
uv run python evals/probe_flywheel.py
# 3. 起栈与起服
docker compose up -d   # 含 Langfuse 组;首次访问 :3000 建站建项目拿密钥填 .env
no_proxy=127.0.0.1,localhost uv run uvicorn --factory app.main:create_app
# 4. 手动跑两轮评估铺趋势(或等定时)
curl --noproxy '*' -X POST localhost:8000/api/jobs/eval-rag/run
# 5. 验收 1~6 逐条点验(Langfuse trace 树/兜底落池待审/审核通过再答对/👎落池/成本统计/趋势)
```

- [ ] **Step 7: 文档收尾 + Commit**

README.md:Langfuse 节(起栈/密钥/no_proxy)、/review 页、评估定时开关、飞轮说明;AGENTS.md:知识库来源修订(knowledge_docs + review:<id> 审核来源并存)、ch09 模块清单、新增配置、当前状态段更新;dev-notes/ch09.md 追记。

```bash
git add -A
git commit -m "feat(ch09): 成本统计 API + /review 审核页 + 评估趋势/成本区块 + 文档"
```

---

## Self-Review 记录(计划作者自查)

- **spec 覆盖**:§5.1→T2/T3/T4/T16;§5.2→T9/T10;§5.3→T5/T6/T7/T8;§5.4→T13/T14;§5.5→T10/T11/T12;§5.6→T14/T15/T16;§6 DDL→T1;§7 配置→各任务内;§8 测试策略→各任务 + T16 Step 6;验收 1~6→T16 Step 6。无缺口。
- **占位符扫描**:无 TBD/TODO;校准数值刻意不预填(spec 明令),以「校准后回填」步骤表达。
- **类型一致性**:snapshot_top_chunks / EvidenceGateParams / submit_feedback / FlywheelWorker.notify / job_lock / record_run / next_run_at 跨任务引用一致;lcq 字段名与 T1 模型一致。
