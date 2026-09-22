# ch07 会话上下文管理(三层历史 + 异步摘要)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把聊天历史从「简单裁剪」升级为三层上下文管理(原文层/截短层/摘要层 + 双锚点降级 + 后台异步分段摘要 + 窗口倒推 token 预算 + 全量可观测),并加前端多会话侧栏。

**Architecture:** 完整历史唯一读源是 LangGraph checkpoint 的 `state["messages"]`(log 节点 commit 后盖 `db_id`);锚点/摘要投影权威存储在 MySQL `conversations`;每轮组装时按锚点把 checkpoint 消息切三层,层 1 超预算同步挪 `layer1_from`,层 2 超预算 `ensure_future` 起摘要任务(CAS 追加 `conversation_summaries` 段并前移 `summary_upto`)。

**Tech Stack:** FastAPI + LangGraph(add_messages/AsyncSqliteSaver)+ LangChain trim_messages(langchain_core 1.6.2)+ SQLAlchemy 1.x 风格 Session + MySQL。

**Spec:** `docs/superpowers/specs/2026-09-22-ch07-session-context-design.md`(评审修订版;本计划 § 号引用该文件)

## Global Constraints

- TDD:先写失败测试再实现;每任务结束 commit(Conventional Commits,`--no-verify` 不使用除非本仓库惯例如此——本仓库此前提交均直接 commit)。
- token 估算唯一尺 `app/services/token_budget.py`:CJK 区间 `U+4E00–U+9FFF`、`U+3400–U+4DBF`、`U+3000–U+303F`、`U+FF00–U+FFEF` 每字符 1 token,其余 `ceil(n/4)`,每条消息 +4;聊天链路禁止再用 `count_tokens_approximately`。
- `sql/ch07-ddl.sql` 与 `db/init/05-ddl.sql` 必须逐字节一致(`tests/test_ddl_sync.py` 钉)。
- 验收数字钉死:演示配置下 AVAIL=5649、层1=3954、层2=1695(spec §5 复算)。
- 配置字段 snake_case,环境变量大写映射(pydantic-settings);`rerank_top_n` 全仓改名 `rerank_top_k`,默认值 10 不变。
- tool 结果不落 `messages` 表;`_to_stored` 只产 user/assistant 行。
- 摘要/锚点更新全部事务 CAS,单调前移不回退;降级只挪 id 不搬数据。
- 单 Uvicorn worker;起服 `no_proxy=127.0.0.1,localhost`;访问本机 `curl --noproxy '*'`。
- 改 prompt/检索后必须全量 `uv run pytest` 绿才交付;probe 脚本烧额度,手跑不进 pytest。
- chat.html 走 Vibe Coding 例外:无 TDD,靠 `tests/test_chat_page.py` 字符串断言 + 人工点验。
- 库 API 以 .venv 安装源码为准(Context7 MCP 本会话不可用):`trim_messages(messages, max_tokens, token_counter, strategy, allow_partial, end_on, start_on, include_system, text_splitter)`。

## Review Focus

1. `content=None` 的 assistant(纯工具调用步)流经渲染/日志/对齐/落库 → 不炸,按空串处理 → 测试在 Task 6。
2. 旧 checkpoint 与 messages 表内容重复(同一句话说了两次)→ 对齐歧义,`reconcile_checkpoint_ids` 返回 None + `context_reconcile_failed`,绝不猜 ID → 测试在 Task 6。
3. 两轮摘要并发/锚点竞争 → `append_summary` CAS 失败返回 `applied=False`,不覆盖投影与锚点 → 测试在 Task 5(DB)。
4. 单轮就超层 1 预算 → 层 1 可为空,系统不崩,下一轮层 2/摘要接住 → 测试在 Task 6、Task 10。
5. 测试多次 `create_app` → FileHandler 幂等,不重复挂、日志不翻倍 → 测试在 Task 11。

## 文件结构

新建:`app/services/token_budget.py`、`app/services/context_layers.py`、`app/services/summarizer.py`、`app/prompts/summary.py`、`app/routers/conversations.py`、`db/init/05-ddl.sql`、`evals/probe_summary.py`、`evals/summary_cases.jsonl`、`evals/probe_context.py`、`tests/test_token_budget.py`、`tests/test_context_layers.py`、`tests/test_summarizer.py`、`tests/test_conversations_api.py`。

修改:`sql/ch07-ddl.sql`(加 FK/CASCADE + 头注)、`app/models.py`、`app/db.py`、`app/config.py`、`app/sessions.py`、`app/store_db.py`、`app/graph/state.py`、`app/graph/nodes.py`、`app/graph/agent_node.py`、`app/chains/tool_chat_chain.py`、`app/chains/chat_chain.py`、`app/tools/executor.py`、`app/prompts/intent.py`、`app/services/chat_service.py`、`app/main.py`、`app/static/chat.html`、`.env.example`、`.gitignore`、`tests/test_ddl_sync.py`、`tests/dbfixtures.py`、`tests/test_config.py`、`tests/conftest.py`(如需)、存量测试若干、`app/knowledge/retriever.py` 等 rerank_top_n 波及点、`AGENTS.md`、`README.md`。

---

### Task 1: DDL 修订、接线、ORM、check_ch07_tables

**Files:**
- Modify: `sql/ch07-ddl.sql`
- Create: `db/init/05-ddl.sql`
- Modify: `tests/test_ddl_sync.py:8`、`tests/dbfixtures.py:15-22`
- Modify: `app/models.py`
- Modify: `app/db.py`(先 Read 看 `check_ch04_tables` 写法,同构实现)
- Modify: `app/main.py:68`(`check_ch04_tables` 后追加调用)
- Test: `tests/test_models.py`、`tests/test_store_db.py`(check 函数用例可放此)

**Interfaces:**
- Produces: `Conversation.summary/summary_upto_msg_id/layer1_from_msg_id` 三列;`ConversationSummary(id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at)`;`check_ch07_tables(engine) -> None`(缺结构 `raise RuntimeError`,提示含 `mysql wayhelp < sql/ch07-ddl.sql`)。

- [ ] **Step 1: 修订 `sql/ch07-ddl.sql`** — 头部注释改为单文件两步合一说明;`conversation_summaries` 加外键与级联。把 CREATE TABLE 段改为:

```sql
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT '第几段,从 1 开始',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的消息区间,闭区间',
  upto_msg_id     BIGINT UNSIGNED NOT NULL,
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id),
  CONSTRAINT fk_summary_conv FOREIGN KEY (conversation_id)
    REFERENCES conversations (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要,一段一行只追加';
```

- [ ] **Step 2: 复制为 `db/init/05-ddl.sql`** — `cp sql/ch07-ddl.sql db/init/05-ddl.sql`;两侧必须逐字节一致。

- [ ] **Step 3: 接线测试基建** — `tests/test_ddl_sync.py:8` 改 `DDL_PAIRS = [("01", "ch02"), ("03", "ch03"), ("04", "ch04"), ("05", "ch07")]`;`tests/dbfixtures.py`:`DDL_PATHS` 追加 `Path(__file__).resolve().parent.parent / "db" / "init" / "05-ddl.sql"`,`TABLES` 改为 `("faith_cases", "low_confidence_questions", "knowledge_chunks", "qa_extraction_staging", "qa_mining_progress", "messages", "conversation_summaries", "tickets", "faq", "conversations")`(先子后父,`conversation_summaries` 在 `conversations` 前)。注意 `05-ddl.sql` 含 ALTER,重放幂等性:dbfixtures 的 `_split_statements` 按 `;\n` 切,ALTER 加列在 DROP 后重放没问题(表是先 DROP 再由 01 重建)。

- [ ] **Step 4: 先跑 DDL 同步测试** — `uv run pytest tests/test_ddl_sync.py -v`,预期全绿(纯文件比较,无需 DB)。

- [ ] **Step 5: ORM 失败测试** — 在 `tests/test_models.py` 追加:

```python
def test_conversation_ch07_columns():
    from app.models import Conversation
    cols = Conversation.__table__.columns
    for name in ("summary", "summary_upto_msg_id", "layer1_from_msg_id"):
        assert name in cols, name
    assert cols["summary"].nullable


def test_conversation_summary_model():
    from app.models import ConversationSummary
    cols = ConversationSummary.__table__.columns
    for name in ("id", "conversation_id", "seq", "from_msg_id", "upto_msg_id",
                 "content", "created_at"):
        assert name in cols, name
    fk = list(cols["conversation_id"].foreign_keys)
    assert fk and fk[0].column.fullname == "conversations.id"
```

- [ ] **Step 6: 实现 ORM** — `app/models.py`:`Conversation` 加 `summary = Column(Text, nullable=True)`、`summary_upto_msg_id = Column(BigInteger, nullable=True)`、`layer1_from_msg_id = Column(BigInteger, nullable=True)`;新增:

```python
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("conversations.id", ondelete="CASCADE"),
                             nullable=False)
    seq = Column(Integer, nullable=False)
    from_msg_id = Column(BigInteger, nullable=False)
    upto_msg_id = Column(BigInteger, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
```

(导入按文件现状对齐:`ForeignKey`、`func` 若未导入则补。)

- [ ] **Step 7: check_ch07_tables 失败测试** — 在 `tests/test_store_db.py` 追加(DB fixture):

```python
def test_check_ch07_tables_ok(db_engine):
    from app.db import check_ch07_tables
    check_ch07_tables(db_engine)  # 不抛即过


def test_check_ch07_tables_missing(db_engine):
    from sqlalchemy import text
    from app.db import check_ch07_tables
    with db_engine.connect() as conn:
        conn.execute(text("ALTER TABLE conversations DROP COLUMN layer1_from_msg_id"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch07-ddl"):
            check_ch07_tables(db_engine)
    finally:
        with db_engine.connect() as conn:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN layer1_from_msg_id "
                              "BIGINT UNSIGNED NULL AFTER summary_upto_msg_id"))
            conn.commit()
```

- [ ] **Step 8: 实现 check_ch07_tables** — Read `app/db.py` 的 `check_ch04_tables`,同构追加:

```python
def check_ch07_tables(engine) -> None:
    """缺 ch07 列/表启动失败并提示升级命令(spec §3)。"""
    with engine.connect() as conn:
        cols = {r[0] for r in conn.execute(text("SHOW COLUMNS FROM conversations"))}
        tables = {r[0] for r in conn.execute(text("SHOW TABLES"))}
    missing = {"summary", "summary_upto_msg_id", "layer1_from_msg_id"} - cols
    if missing or "conversation_summaries" not in tables:
        raise RuntimeError(
            "缺少 ch07 会话上下文表结构,请先执行:mysql wayhelp < sql/ch07-ddl.sql")
```

(`text` 未导入则补。)`app/main.py` 在 `check_ch04_tables(engine)` 下一行加 `check_ch07_tables(engine)`,import 同步补。

- [ ] **Step 9: 跑测试** — `uv run pytest tests/test_models.py tests/test_store_db.py tests/test_ddl_sync.py -v`,全绿(DB 测试需 docker compose up -d)。

- [ ] **Step 10: Commit** — `git add -A && git commit -m "feat(ch07): DDL 接线 + conversation_summaries 外键级联 + check_ch07_tables"`

---

### Task 2: config 变更(新字段 / rerank_top_k 改名 / 退役)

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`
- Modify(改名波及):`app/knowledge/retriever.py`、`app/graph/nodes.py`、`app/tools/business.py`、`evals/run_retrieval_compare.py`、其余 `grep -n "rerank_top_n\|understand_history_turns"` 命中点

**Interfaces:**
- Produces: `Settings.model_context_window=65536`、`max_user_input_tokens=2000`、`tool_result_max_tokens=1200`、`summary_projection_tokens=650`、`rerank_top_k=10`、`history_target_turns=20`、`steady_tokens_per_turn=300`、`summary_max_chars=200`、`safety_margin_tokens=550`;删除 `understand_history_turns`;`max_input_tokens`(extract/mining 专用)与 `max_tool_result_chars`(证据组装字符预算)保留。

- [ ] **Step 1: 失败测试** — `tests/test_config.py` 改:删 `understand_history_turns` 断言;`rerank_top_n` 断言(:140)改 `assert s.retrieval_candidate_k == 50 and s.rerank_top_k == 10`;追加:

```python
def test_ch07_context_defaults():
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")
    assert s.model_context_window == 65536
    assert s.max_user_input_tokens == 2000
    assert s.tool_result_max_tokens == 1200
    assert s.summary_projection_tokens == 650
    assert s.rerank_top_k == 10
    assert s.history_target_turns == 20
    assert s.steady_tokens_per_turn == 300
    assert s.summary_max_chars == 200
    assert s.safety_margin_tokens == 550
    assert not hasattr(s, "understand_history_turns")
    assert not hasattr(s, "rerank_top_n")
```

- [ ] **Step 2: 跑测试确认红** — `uv run pytest tests/test_config.py -v`,FAIL(缺字段)。

- [ ] **Step 3: 实现 config** — `app/config.py` 删除 `understand_history_turns` 行;`rerank_top_n` 字段改名 `rerank_top_k`(默认值/注释保留语义);追加:

```python
    # ch07 会话上下文预算(spec §5)
    model_context_window: int = Field(default=65536, gt=0)
    max_user_input_tokens: int = Field(default=2000, gt=0)
    tool_result_max_tokens: int = Field(default=1200, gt=0)
    summary_projection_tokens: int = Field(default=650, gt=0)
    history_target_turns: int = Field(default=20, gt=0)
    steady_tokens_per_turn: int = Field(default=300, gt=0)
    summary_max_chars: int = Field(default=200, gt=0)
    safety_margin_tokens: int = Field(default=550, ge=0)
```

并给 `max_input_tokens`、`max_tool_result_chars` 改注释:前者「extract/mining 专用,聊天链路不用」;后者「证据组装字符预算(assemble_evidence budget_chars)」。

- [ ] **Step 4: 全仓改名** — `grep -rn "rerank_top_n" app/ evals/ tests/` 逐点改 `rerank_top_k`:`app/knowledge/retriever.py`(4 处)、`app/graph/nodes.py:345,421`、`app/tools/business.py:145,287`、`evals/run_retrieval_compare.py:340`、测试命中点;`grep -rn "understand_history_turns" app/ tests/` 命中点删除(`app/graph/nodes.py:157-158` 的引用在 Task 8 重写,本任务先把 config 引用摘掉让 import 不炸——若 nodes.py 因此报错,先把 `deps.settings.understand_history_turns` 临时替换为字面 `6` 并标注 Task 8 删除)。`.env.example` 追加:

```
# ch07 上下文预算(均有默认值,演示配置见 spec §5)
# MODEL_CONTEXT_WINDOW=65536
# MAX_USER_INPUT_TOKENS=2000
# TOOL_RESULT_MAX_TOKENS=1200
# SUMMARY_PROJECTION_TOKENS=650
# RERANK_TOP_K=10
# HISTORY_TARGET_TURNS=20
# STEADY_TOKENS_PER_TURN=300
# SUMMARY_MAX_CHARS=200
# SAFETY_MARGIN_TOKENS=550
```

- [ ] **Step 5: 跑测试** — `uv run pytest tests/test_config.py -v` 全绿;`uv run python -c "from app.config import Settings"` 不炸。

- [ ] **Step 6: Commit** — `git commit -am "refactor(ch07): 上下文预算配置项 + rerank_top_k 改名 + understand_history_turns 退役"`

---

### Task 3: token_budget(估算尺 / 截断 / 投影分配 / 预算公式)

**Files:**
- Create: `app/services/token_budget.py`
- Test: `tests/test_token_budget.py`

**Interfaces:**
- Consumes: Task 2 的 `Settings` 新字段。
- Produces(后续任务全部依赖):
  - `estimate_tokens(text: str) -> int`
  - `estimate_message(msg) -> int`(BaseMessage;`content=None` 按空串;AIMessage 有 tool_calls 加 JSON 估算)
  - `estimate_messages(msgs) -> int`
  - `truncate_to_tokens(text: str, budget: int) -> str`(二分,结果 `estimate_tokens(结果) <= budget`)
  - `truncate_tool_result(text: str, max_tokens: int) -> str`(超限截断 + `…[已截断]`)
  - `build_summary_projection(segments: list[str], budget_tokens: int) -> str`(方案 A)
  - `measure_sys_tokens(system_prompt: str, tools) -> int`
  - `compute_budget(settings, sys_tokens: int) -> ContextBudget`
  - `ContextBudget(avail, total, layer1, layer2, peak, fixed, sys_tokens, sufficient)`(frozen dataclass)
  - 常量 `LAYER2_HEAD_CHARS=50`、`TOOL_CALL_OVERHEAD_TOKENS=50`、`EVIDENCE_TOKENS_PER_DOC=300`

- [ ] **Step 1: 失败测试** — 写 `tests/test_token_budget.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.token_budget import (
    ContextBudget, build_summary_projection, compute_budget, estimate_message,
    estimate_messages, estimate_tokens, measure_sys_tokens, truncate_to_tokens,
)


def _settings(**over):
    from app.config import Settings
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d",
                model_context_window=18000, max_output_tokens=2000,
                max_user_input_tokens=2000, max_agent_steps=3,
                tool_result_max_tokens=1200, rerank_top_k=5)
    base.update(over)
    return Settings(**base)


def test_estimate_tokens_cjk_and_ascii():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好") == 2          # CJK 1字=1token
    assert estimate_tokens("abcd") == 1          # 其余 ceil(/4)
    assert estimate_tokens("你好abcd") == 3
    assert estimate_tokens("。") == 1            # 中文标点 U+3000–U+303F
    assert estimate_tokens("!") == 1


def test_estimate_message_overhead_and_tool_calls():
    assert estimate_message(HumanMessage(content="你好")) == 2 + 4
    assert estimate_message(AIMessage(content=None)) == 4          # None 按空串
    m = AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "A1"}, "id": "c1", "type": "tool_call"}])
    assert estimate_message(m) > 4
    assert estimate_messages([HumanMessage(content="你好"), ToolMessage(content="x", tool_call_id="c1")]) == 2 + 4 + 1 + 4


def test_truncate_to_tokens():
    text = "汉" * 100
    out = truncate_to_tokens(text, 30)
    assert estimate_tokens(out) <= 30 and len(out) == 30
    assert truncate_to_tokens("短", 30) == "短"


def test_budget_acceptance_numbers():
    b = compute_budget(_settings(), sys_tokens=1901)
    assert (b.peak, b.fixed, b.avail) == (3750, 4601, 5649)
    assert (b.total, b.layer1, b.layer2) == (5649, 3954, 1695)
    assert b.sufficient


def test_budget_min_arm_and_clamp():
    b = compute_budget(_settings(model_context_window=65536), sys_tokens=1901)
    assert b.total == 20 * 300                       # 想留住的轮数 × 稳态 取小
    tiny = compute_budget(_settings(model_context_window=5000), sys_tokens=1901)
    assert tiny.avail >= 0 and not tiny.sufficient   # 一轮装不下 → 自检报警语义


def test_build_summary_projection():
    segs = ["第一段订单A1001要退款", "第二段", "第三段", "第四段"]
    assert build_summary_projection(segs, 10_000) == "\n\n".join(segs)  # 全额
    out = build_summary_projection(segs, 40)       # 超预算:每段等分配截短,段数不丢
    assert out.count("……") >= 1 and estimate_tokens(out) <= 40
    assert len(out.split("\n\n")) == 4             # 方案 A:无「最近 3 段」硬上限
    assert build_summary_projection(segs, 0) == ""


def test_measure_sys_tokens():
    assert measure_sys_tokens("系统", []) == 2 + 4
```

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_token_budget.py -v`,FAIL(模块不存在)。

- [ ] **Step 3: 实现** — 写 `app/services/token_budget.py`:

```python
"""ch07 上下文预算:唯一 token 估算尺 + 从模型窗口倒推历史预算(spec §4/§5)。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F), (0xFF00, 0xFFEF))
PER_MESSAGE_OVERHEAD = 4
TOOL_CALL_OVERHEAD_TOKENS = 50      # ReAct 每步 tool_call JSON 开销(PEAK 公式)
EVIDENCE_TOKENS_PER_DOC = 300       # 证据预留/条
LAYER2_HEAD_CHARS = 50              # 层2 assistant 答复保留开头字数
TRUNCATED_MARK = "…[已截断]"
ELLIPSIS = "……"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text
              if any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES))
    return cjk + math.ceil((len(text) - cjk) / 4)


def estimate_message(msg) -> int:
    content = msg.content if isinstance(getattr(msg, "content", None), str) else ""
    total = estimate_tokens(content) + PER_MESSAGE_OVERHEAD
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        total += estimate_tokens(json.dumps(tool_calls, ensure_ascii=False))
    return total


def estimate_messages(msgs) -> int:
    return sum(estimate_message(m) for m in msgs)


def truncate_to_tokens(text: str, budget: int) -> str:
    """二分截断,保证 estimate_tokens(结果) <= budget。"""
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def truncate_tool_result(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    return truncate_to_tokens(text, max_tokens - estimate_tokens(TRUNCATED_MARK)) + TRUNCATED_MARK


def build_summary_projection(segments: list[str], budget_tokens: int) -> str:
    """方案 A(spec §9):所有段按 seq 序等分配额度;放得下留全文,放不下 head+tail
    截短加省略标记;段数不丢,不改原文。"""
    if not segments or budget_tokens <= 0:
        return ""
    full = "\n\n".join(segments)
    if estimate_tokens(full) <= budget_tokens:
        return full
    per = budget_tokens // len(segments)
    out = []
    for seg in segments:
        if estimate_tokens(seg) <= per:
            out.append(seg)
            continue
        body = max(0, per - estimate_tokens(ELLIPSIS))
        head = truncate_to_tokens(seg, body * 2 // 3)
        tail_budget = body - estimate_tokens(head)
        tail = truncate_to_tokens(seg[::-1], tail_budget)[::-1] if tail_budget > 0 else ""
        out.append(head + ELLIPSIS + tail)
    return "\n\n".join(out)


def measure_sys_tokens(system_prompt: str, tools) -> int:
    """系统提示 + 工具 schema 的估算开销(启动时实测,spec §5 SYS_TOKENS)。"""
    parts = [system_prompt]
    for t in tools:
        parts.append(t.name)
        parts.append(t.description or "")
        try:
            parts.append(json.dumps(t.args_schema.model_json_schema(), ensure_ascii=False))
        except Exception:
            pass
    return estimate_tokens("\n".join(parts)) + PER_MESSAGE_OVERHEAD


@dataclass(frozen=True)
class ContextBudget:
    avail: int
    total: int
    layer1: int
    layer2: int
    peak: int
    fixed: int
    sys_tokens: int
    sufficient: bool      # total >= steady_tokens_per_turn(一轮都装不下 → 启动报警)


def compute_budget(settings, sys_tokens: int) -> ContextBudget:
    peak = settings.max_agent_steps * (settings.tool_result_max_tokens
                                       + TOOL_CALL_OVERHEAD_TOKENS)
    fixed = (sys_tokens + settings.rerank_top_k * EVIDENCE_TOKENS_PER_DOC
             + settings.summary_projection_tokens + settings.safety_margin_tokens)
    avail = max(0, settings.model_context_window - settings.max_output_tokens
                - settings.max_user_input_tokens - peak - fixed)
    total = min(settings.history_target_turns * settings.steady_tokens_per_turn, avail)
    layer1 = int(total * 0.7)
    return ContextBudget(avail=avail, total=total, layer1=layer1, layer2=total - layer1,
                         peak=peak, fixed=fixed, sys_tokens=sys_tokens,
                         sufficient=total >= settings.steady_tokens_per_turn)
```

- [ ] **Step 4: 跑绿** — `uv run pytest tests/test_token_budget.py -v` 全绿。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): token 估算尺与窗口倒推预算公式(验收数字 5649/3954/1695 钉死)"`

---

### Task 4: sessions DTO、validate_turn 变更、InMemory 扩展

**Files:**
- Modify: `app/sessions.py`
- Test: `tests/test_sessions.py`
- Modify: `tests/conftest.py`(`UserBoundMemoryStore` 若与原生归属校验冲突则适配,先 Read :14-28)

**Interfaces:**
- Consumes: `build_summary_projection`(Task 3)。
- Produces:
  - DTO(frozen dataclass):`ContextMeta(summary: str|None, summary_upto: int|None, layer1_from: int|None)`、`ConversationMessage(id: str, role: str, content: str, tool_calls: list[dict]|None, created_at)`、`PersistedMessageRecord(id: str, role: str, content: str|None, tool_calls, tool_call_id, created_at)`、`ConversationItem(id: str, status: str, created_at, updated_at, preview: str|None, summarized: bool)`、`SummaryAppendResult(seq: int|None, applied: bool, reason: str|None)`、`PersistedTurn(stored: list[StoredMessage], checkpoint_indexes: list[int])`
  - `CommitTurnResult(source_message_id: str, message_ids: list[str])`
  - `SessionStore` 新方法签名:`get_context_meta(session_id, user_id) -> ContextMeta`、`list_conversations(user_id) -> list[ConversationItem]`、`list_messages(session_id, user_id) -> list[ConversationMessage] | None`、`list_checkpoint_records(session_id, user_id) -> list[PersistedMessageRecord] | None`、`move_layer1_from(session_id, user_id, new_id: int) -> bool`、`fetch_span_texts(session_id, from_id: int, upto_id: int) -> list[tuple[int, str, str]]`、`append_summary(session_id, from_id: int, upto_id: int, content: str, projection_tokens: int) -> SummaryAppendResult`(projection_tokens 参数是对 spec §6 签名的实现性补充,语义不变)
  - `validate_turn(messages, max_tool_calls)` 新语义:中间 assistant 仍须带 tool_calls;`role=="tool"` 直接 ValueError;不再要求 tool 一一配对。

- [ ] **Step 1: 失败测试** — `tests/test_sessions.py` 追加(现有用例不动):

```python
import pytest

from app.sessions import (
    ContextMeta, ConversationItem, ConversationMessage, InMemorySessionStore,
    PersistedTurn, StoredMessage, validate_turn,
)


def _store() -> InMemorySessionStore:
    return InMemorySessionStore(max_sessions=10, max_messages_per_session=100,
                                max_message_chars=8000, max_tool_calls_per_turn=5)


def _turn() -> list[StoredMessage]:
    return [StoredMessage("user", "你好"),
            StoredMessage("assistant", None, tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("assistant", "在的")]


@pytest.mark.asyncio
async def test_commit_returns_per_row_ids():
    s = _store()
    sid = await s.create("u1")
    r = await s.commit_turn(sid, _turn())
    assert r.source_message_id == r.message_ids[0] and len(r.message_ids) == 3
    assert len(set(r.message_ids)) == 3


def test_validate_turn_rejects_tool_rows():
    bad = [StoredMessage("user", "hi"),
           StoredMessage("assistant", None, tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
           StoredMessage("tool", "{}", tool_call_id="c1"),
           StoredMessage("assistant", "好")]
    with pytest.raises(ValueError, match="tool"):
        validate_turn(bad, 5)
    validate_turn(_turn(), 5)  # 无 tool 行的新形态合法


@pytest.mark.asyncio
async def test_memory_anchor_methods():
    s = _store()
    sid = await s.create("u1")
    assert (await s.get_context_meta(sid, "u1")) == ContextMeta(None, None, None)
    r = await s.commit_turn(sid, _turn())
    last_id = int(r.message_ids[-1])
    assert await s.move_layer1_from(sid, "u1", last_id) is True     # 末条是完整轮边界
    assert await s.move_layer1_from(sid, "u1", last_id) is True     # 幂等(相等不动仍 True?否:不前移 → False)
```

——写测试时把语义定死:`move_layer1_from` 只在 `new_id > 旧值(或旧为 NULL)` 且落完整轮边界时 True,其余 False;上面最后一行应 `is False`,实现后修正测试再跑(红→绿以最终语义为准,把本行写成 False)。

```python
@pytest.mark.asyncio
async def test_memory_summary_append_and_projection():
    s = _store()
    sid = await s.create("u1")
    r = await s.commit_turn(sid, _turn())
    upto = int(r.message_ids[-1])
    assert await s.move_layer1_from(sid, "u1", upto)
    res = await s.append_summary(sid, 0, upto, "用户问过订单A1001退款", 10_000)
    assert res.applied and res.seq == 1
    meta = await s.get_context_meta(sid, "u1")
    assert meta.summary_upto == upto and "A1001" in (meta.summary or "")
    again = await s.append_summary(sid, 0, upto, "重复提交", 10_000)
    assert not again.applied and again.reason                         # CAS:from_id 已前移


@pytest.mark.asyncio
async def test_memory_list_and_span():
    s = _store()
    sid = await s.create("u1")
    await s.commit_turn(sid, _turn())
    convs = await s.list_conversations("u1")
    assert len(convs) == 1 and convs[0].preview == "你好" and convs[0].summarized is False
    msgs = await s.list_messages(sid, "u1")
    assert [m.role for m in msgs] == ["user", "assistant", "assistant"]
    assert await s.list_messages(sid, "other") is None                 # 归属 404
    rows = await s.fetch_span_texts(sid, 0, 10**9)
    assert [r[1] for r in rows] == ["user", "assistant", "assistant"]
    recs = await s.list_checkpoint_records(sid, "u1")
    assert recs is not None and len(recs) == 3
```

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_sessions.py -v`,FAIL(新 DTO/方法不存在;旧 `validate_turn` 对无 tool 行形态报「orphan/不匹配」)。

- [ ] **Step 3: 实现** — `app/sessions.py`:
  - 顶部 import 加 `from datetime import datetime`。
  - `CommitTurnResult` 加 `message_ids: list[str]`(frozen dataclass 第二字段)。
  - 追加 DTO:

```python
@dataclass(frozen=True)
class ContextMeta:
    summary: str | None
    summary_upto: int | None
    layer1_from: int | None


@dataclass(frozen=True)
class ConversationMessage:
    id: str
    role: str
    content: str
    tool_calls: list[dict] | None
    created_at: datetime | None = None


@dataclass(frozen=True)
class PersistedMessageRecord:
    id: str
    role: str
    content: str | None
    tool_calls: list[dict] | None
    tool_call_id: str | None
    created_at: datetime | None = None


@dataclass(frozen=True)
class ConversationItem:
    id: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    preview: str | None
    summarized: bool


@dataclass(frozen=True)
class SummaryAppendResult:
    seq: int | None
    applied: bool
    reason: str | None


@dataclass(frozen=True)
class PersistedTurn:
    stored: list[StoredMessage]
    checkpoint_indexes: list[int]   # stored 行 ↔ turn_messages 原始索引
```

  - `SessionStore` Protocol 追加 7 个新方法签名(同 Interfaces 块)。
  - `validate_turn` 重写中间段:删 `close_group`/tool 配对逻辑,改为:

```python
    for m in messages[1:-1]:
        if m.role == "tool":
            raise ValueError("tool rows are not persisted (ch07)")
        if m.role != "assistant":
            raise ValueError(f"unexpected role in turn middle: {m.role}")
        if not m.tool_calls:
            raise ValueError("middle assistant message without tool calls")
        if len(m.tool_calls) > max_tool_calls:
            raise ValueError("too many tool calls in turn")
        for call in m.tool_calls:
            cid = call.get("id") if isinstance(call, dict) else None
            if not isinstance(cid, str) or not cid or len(cid) > 64:
                raise ValueError("invalid tool call id")
```

  - `InMemorySessionStore`:`__init__` 加 `self._owners: dict[str, str] = {}`、`self._ids: dict[str, list[int]] = {}`、`self._meta: dict[str, ContextMeta] = {}`、`self._segments: dict[str, list[str]] = {}`;`create` 记录 owner/初始化;`exists` 改为 `self._owners.get(session_id) == user_id`(归属生效;`tests/conftest.py` 的 `UserBoundMemoryStore` 若重复校验则保留无害,若冲突把它简化为透传);`commit_turn` 末尾:

```python
        ids = []
        for _ in messages:
            self._msg_seq += 1
            ids.append(str(self._msg_seq))
        self._ids.setdefault(session_id, []).extend(int(i) for i in ids)
        return CommitTurnResult(ids[0], ids)
```

(`_msg_seq` 原语义保留:校验全过后才分配。)删最旧轮时 `self._ids` 同步裁前段。新增方法:

```python
    async def get_context_meta(self, session_id: str, user_id: str) -> ContextMeta:
        return self._meta.get(session_id, ContextMeta(None, None, None))

    async def list_conversations(self, user_id: str) -> list[ConversationItem]:
        out = []
        for sid, owner in self._owners.items():
            if owner != user_id:
                continue
            first_user = next((m for m in self._sessions[sid] if m.role == "user"), None)
            meta = self._meta.get(sid, ContextMeta(None, None, None))
            out.append(ConversationItem(sid, "进行中", None, None,
                                        (first_user.content or "")[:40] if first_user else None,
                                        meta.summary is not None))
        return out[::-1]  # 后建在前

    async def list_messages(self, session_id: str, user_id: str):
        if self._owners.get(session_id) != user_id:
            return None
        return [ConversationMessage(str(i), m.role, m.content, m.tool_calls)
                for i, m in zip(self._ids[session_id], self._sessions[session_id])
                if m.role in ("user", "assistant") and m.content]

    async def list_checkpoint_records(self, session_id: str, user_id: str):
        if self._owners.get(session_id) != user_id:
            return None
        return [PersistedMessageRecord(str(i), m.role, m.content, m.tool_calls, m.tool_call_id)
                for i, m in zip(self._ids[session_id], self._sessions[session_id])]

    def _turn_boundary_ids(self, session_id: str) -> set[int]:
        from app.sessions import _turns  # 模块内直接引用即可
        ids = self._ids.get(session_id, [])
        bounds = set()
        pos = 0
        for t in _turns(self._sessions.get(session_id, [])):
            pos += len(t)
            bounds.add(ids[pos - 1])
        return bounds

    async def move_layer1_from(self, session_id: str, user_id: str, new_id: int) -> bool:
        if self._owners.get(session_id) != user_id:
            return False
        meta = self._meta.get(session_id, ContextMeta(None, None, None))
        old = meta.layer1_from
        if old is not None and new_id <= old:
            return False
        if new_id not in self._ids.get(session_id, []):
            return False
        if new_id not in self._turn_boundary_ids(session_id):
            return False
        self._meta[session_id] = ContextMeta(meta.summary, meta.summary_upto, new_id)
        return True

    async def fetch_span_texts(self, session_id: str, from_id: int, upto_id: int):
        return [(i, m.role, m.content or "")
                for i, m in zip(self._ids.get(session_id, []), self._sessions.get(session_id, []))
                if from_id < i <= upto_id and m.role in ("user", "assistant")]

    async def append_summary(self, session_id: str, from_id: int, upto_id: int,
                             content: str, projection_tokens: int) -> SummaryAppendResult:
        from app.services.token_budget import build_summary_projection
        meta = self._meta.get(session_id, ContextMeta(None, None, None))
        if (meta.summary_upto or 0) != from_id:
            return SummaryAppendResult(None, False, "anchor-moved")
        if meta.layer1_from is None or upto_id > meta.layer1_from:
            return SummaryAppendResult(None, False, "beyond-layer1")
        segs = self._segments.setdefault(session_id, [])
        segs.append(content)
        seq = len(segs)
        projection = build_summary_projection(segs, projection_tokens)
        self._meta[session_id] = ContextMeta(projection, upto_id, meta.layer1_from)
        return SummaryAppendResult(seq, True, None)
```

- [ ] **Step 4: 跑绿 + 回归** — `uv run pytest tests/test_sessions.py -v` 全绿;`uv run pytest tests/test_store_db.py tests/test_chat_service.py -v` 看 `CommitTurnResult`/`validate_turn` 变更的连锁红,按新语义改旧断言(tool 行 envelope 相关旧断言在 Task 10 随 `_to_stored` 一起最终定型,本任务允许暂时标 xfail 并注明,Task 10 必须摘掉)。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): SessionStore 协议扩展 + validate_turn 停收 tool 行 + InMemory 锚点/摘要支持"`

---

### Task 5: DbSessionStore 实现(commit ids / 只读查询 / CAS)

**Files:**
- Modify: `app/store_db.py`
- Test: `tests/test_store_db.py`

**Interfaces:**
- Consumes: Task 4 的 DTO 与协议签名;`build_summary_projection`(Task 3);`ConversationSummary` ORM(Task 1)。
- Produces:`DbSessionStore` 实现协议全部方法;`commit_turn` 返回逐行 `message_ids`,不再接收 tool 行(`validate_turn` 拒)。

- [ ] **Step 1: 失败测试(DB fixture)** — `tests/test_store_db.py` 追加:

```python
import pytest

from app.sessions import ContextMeta, LowConfidenceRecord, StoredMessage

pytestmark = pytest.mark.usefixtures("db_session_factory")


def _turn(user="你好", final="在的"):
    return [StoredMessage("user", user),
            StoredMessage("assistant", None,
                          tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("assistant", final)]


@pytest.mark.asyncio
async def test_commit_turn_ids_and_no_tool_rows(db_session_factory):
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, _turn())
    assert len(r.message_ids) == 3 and r.source_message_id == r.message_ids[0]
    assert [int(i) for i in r.message_ids] == sorted(int(i) for i in r.message_ids)


@pytest.mark.asyncio
async def test_context_meta_and_move_layer1_from(db_session_factory):
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    assert (await store.get_context_meta(sid, "u1")) == ContextMeta(None, None, None)
    r1 = await store.commit_turn(sid, _turn())
    r2 = await store.commit_turn(sid, _turn("第二问", "第二答"))
    last1 = int(r1.message_ids[-1])
    mid = int(r1.message_ids[1])
    assert await store.move_layer1_from(sid, "u1", mid) is False      # 非完整轮边界
    assert await store.move_layer1_from(sid, "u1", last1) is True     # 第一轮末 = 边界
    assert await store.move_layer1_from(sid, "u1", last1) is False    # 不前移
    assert await store.move_layer1_from(sid, "u1", int(r2.message_ids[-1])) is True
    assert await store.move_layer1_from(sid, "other", last1) is False
    meta = await store.get_context_meta(sid, "u1")
    assert meta.layer1_from == int(r2.message_ids[-1])


@pytest.mark.asyncio
async def test_append_summary_cas_and_projection(db_session_factory):
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r1 = await store.commit_turn(sid, _turn())
    upto = int(r1.message_ids[-1])
    bad = await store.append_summary(sid, 0, upto, "段一", 10_000)
    assert not bad.applied and bad.reason == "beyond-layer1"          # layer1_from 未设
    await store.move_layer1_from(sid, "u1", upto)
    ok = await store.append_summary(sid, 0, upto, "用户问过订单A1001退款", 10_000)
    assert ok.applied and ok.seq == 1
    dup = await store.append_summary(sid, 0, upto, "重复", 10_000)
    assert not dup.applied and dup.reason == "anchor-moved"           # CAS 不回退
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary_upto == upto and "A1001" in meta.summary


@pytest.mark.asyncio
async def test_list_conversations_and_messages(db_session_factory):
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid1 = await store.create("u1")
    await store.commit_turn(sid1, _turn())
    sid2 = await store.create("u1")
    convs = await store.list_conversations("u1")
    assert [c.id for c in convs] == [sid2, sid1]                       # 新在前
    assert convs[1].preview == "你好" and convs[1].summarized is False
    msgs = await store.list_messages(sid1, "u1")
    assert [m.role for m in msgs] == ["user", "assistant", "assistant"]
    assert all(m.content for m in msgs)                                # content=NULL 已过滤
    assert await store.list_messages(sid1, "other") is None
    assert await store.list_conversations("other") == []


@pytest.mark.asyncio
async def test_checkpoint_records_and_span(db_session_factory):
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, _turn())
    recs = await store.list_checkpoint_records(sid, "u1")
    assert [x.role for x in recs] == ["user", "assistant", "assistant"]
    rows = await store.fetch_span_texts(sid, 0, int(r.message_ids[1]))
    assert [x[1] for x in rows] == ["user", "assistant"]
    assert await store.list_checkpoint_records(sid, "other") is None


@pytest.mark.asyncio
async def test_summary_cascade_delete(db_session_factory):
    from sqlalchemy import text
    from app.store_db import DbSessionStore
    store = DbSessionStore(db_session_factory, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, _turn())
    upto = int(r.message_ids[-1])
    await store.move_layer1_from(sid, "u1", upto)
    await store.append_summary(sid, 0, upto, "段一", 10_000)
    with db_session_factory() as s:
        s.execute(text("DELETE FROM conversations WHERE id = :cid"), {"cid": int(sid)})
        s.commit()
        left = s.execute(text("SELECT COUNT(*) FROM conversation_summaries")).scalar()
    assert left == 0                                                   # 外键级联
```

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_store_db.py -k "ch07 or ids or meta or summary or list or span or cascade" -v`,FAIL。

- [ ] **Step 3: 实现** — `app/store_db.py` 改造:
  - import 补:`from sqlalchemy import func, select, update`(现状已有 select/update,补 func);`from app.models import Conversation, ConversationSummary, LowConfidenceQuestion, Message`;`from app.sessions import (...新 DTO...)`;`from app.services.token_budget import build_summary_projection`。
  - `_commit_sync` 返回值改 `list[str]`:每 `s.add` 后统一 `s.flush()` 一次(首条 flush 取 id 逻辑保留),收集全部 `str(m.id)`;`commit_turn` 返回 `CommitTurnResult(ids[0], ids)`。
  - 新方法(每个都是 `asyncio.to_thread` 包同步实现,风格同现状):

```python
    async def get_context_meta(self, session_id: str, user_id: str) -> ContextMeta:
        return await asyncio.to_thread(self._get_context_meta_sync, session_id, user_id)

    def _get_context_meta_sync(self, session_id, user_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return ContextMeta(None, None, None)
        with self._sf() as s:
            row = (s.query(Conversation.summary, Conversation.summary_upto_msg_id,
                           Conversation.layer1_from_msg_id)
                   .filter(Conversation.id == cid, Conversation.user_id == user_id)
                   .first())
            if row is None:
                return ContextMeta(None, None, None)
            return ContextMeta(row[0], row[1], row[2])

    async def list_conversations(self, user_id: str):
        return await asyncio.to_thread(self._list_conversations_sync, user_id)

    def _list_conversations_sync(self, user_id):
        with self._sf() as s:
            first_q = (s.query(Message.content)
                       .filter(Message.conversation_id == Conversation.id,
                               Message.role == "user")
                       .order_by(Message.id).limit(1)
                       .correlate(Conversation).scalar_subquery())
            rows = (s.query(Conversation, first_q.label("first_q"))
                    .filter(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                    .all())
            return [ConversationItem(str(c.id), c.status, c.created_at, c.updated_at,
                                     ((fq or "")[:40] or None), c.summary is not None)
                    for c, fq in rows]

    async def list_messages(self, session_id: str, user_id: str):
        return await asyncio.to_thread(self._list_messages_sync, session_id, user_id)

    def _list_messages_sync(self, session_id, user_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return None
        with self._sf() as s:
            owner = (s.query(Conversation.id)
                     .filter(Conversation.id == cid, Conversation.user_id == user_id)
                     .first())
            if owner is None:
                return None
            rows = (s.query(Message)
                    .filter(Message.conversation_id == cid,
                            Message.role.in_(("user", "assistant")),
                            Message.content.isnot(None))
                    .order_by(Message.id).all())
            return [ConversationMessage(str(r.id), r.role, r.content, r.tool_calls,
                                        r.created_at) for r in rows]

    async def list_checkpoint_records(self, session_id: str, user_id: str):
        return await asyncio.to_thread(self._list_checkpoint_records_sync, session_id, user_id)

    def _list_checkpoint_records_sync(self, session_id, user_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return None
        with self._sf() as s:
            owner = (s.query(Conversation.id)
                     .filter(Conversation.id == cid, Conversation.user_id == user_id)
                     .first())
            if owner is None:
                return None
            rows = (s.query(Message).filter(Message.conversation_id == cid)
                    .order_by(Message.id).all())
            return [PersistedMessageRecord(str(r.id), r.role, r.content, r.tool_calls,
                                           r.tool_call_id, r.created_at) for r in rows]

    async def move_layer1_from(self, session_id: str, user_id: str, new_id: int) -> bool:
        return await asyncio.to_thread(self._move_layer1_from_sync, session_id, user_id, new_id)

    def _move_layer1_from_sync(self, session_id, user_id, new_id) -> bool:
        cid = _as_db_id(session_id)
        if cid is None:
            return False
        with self._sf() as s:
            conv = (s.query(Conversation)
                    .filter(Conversation.id == cid, Conversation.user_id == user_id)
                    .with_for_update().first())
            if conv is None:
                return False
            old = conv.layer1_from_msg_id
            if old is not None and new_id <= old:
                return False
            msg = s.get(Message, new_id)
            if msg is None or msg.conversation_id != cid:
                return False
            nxt = (s.query(Message.role)
                   .filter(Message.conversation_id == cid, Message.id > new_id)
                   .order_by(Message.id).limit(1).first())
            if nxt is not None and nxt[0] != "user":
                return False                       # 不在完整轮边界
            conv.layer1_from_msg_id = new_id
            s.commit()
            return True

    async def fetch_span_texts(self, session_id: str, from_id: int, upto_id: int):
        return await asyncio.to_thread(self._fetch_span_texts_sync, session_id, from_id, upto_id)

    def _fetch_span_texts_sync(self, session_id, from_id, upto_id):
        cid = _as_db_id(session_id)
        if cid is None:
            return []
        with self._sf() as s:
            rows = (s.query(Message.id, Message.role, Message.content)
                    .filter(Message.conversation_id == cid,
                            Message.id > from_id, Message.id <= upto_id,
                            Message.role.in_(("user", "assistant")))
                    .order_by(Message.id).all())
            return [(int(r[0]), r[1], r[2] or "") for r in rows]

    async def append_summary(self, session_id: str, from_id: int, upto_id: int,
                             content: str, projection_tokens: int) -> SummaryAppendResult:
        return await asyncio.to_thread(self._append_summary_sync, session_id, from_id,
                                       upto_id, content, projection_tokens)

    def _append_summary_sync(self, session_id, from_id, upto_id, content,
                             projection_tokens) -> SummaryAppendResult:
        cid = _as_db_id(session_id)
        if cid is None:
            return SummaryAppendResult(None, False, "bad-session")
        with self._sf() as s:
            conv = (s.query(Conversation).filter(Conversation.id == cid)
                    .with_for_update().first())
            if conv is None:
                return SummaryAppendResult(None, False, "bad-session")
            if (conv.summary_upto_msg_id or 0) != from_id:
                return SummaryAppendResult(None, False, "anchor-moved")
            if conv.layer1_from_msg_id is None or upto_id > conv.layer1_from_msg_id:
                return SummaryAppendResult(None, False, "beyond-layer1")
            seq = (s.query(func.max(ConversationSummary.seq))
                   .filter(ConversationSummary.conversation_id == cid).scalar() or 0) + 1
            s.add(ConversationSummary(conversation_id=cid, seq=seq, from_msg_id=from_id,
                                      upto_msg_id=upto_id, content=content))
            s.flush()
            segs = [r[0] for r in s.query(ConversationSummary.content)
                    .filter(ConversationSummary.conversation_id == cid)
                    .order_by(ConversationSummary.seq).all()]
            conv.summary = build_summary_projection(segs, projection_tokens)
            conv.summary_upto_msg_id = upto_id
            s.commit()
            return SummaryAppendResult(seq, True, None)
```

  - `snapshot`/`exists` 不动。

- [ ] **Step 4: 跑绿 + 回归** — `uv run pytest tests/test_store_db.py -v` 全绿;旧 envelope 断言按「commit 只收 user/assistant」改写(若 Task 4 已标 xfail,这里定稿)。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): DbSessionStore 逐行 ids + 只读查询 + 摘要/锚点事务 CAS"`

---

### Task 6: context_layers(三层组装 / 渲染 / reconcile / 降级评估)

**Files:**
- Create: `app/services/context_layers.py`
- Test: `tests/test_context_layers.py`

**Interfaces:**
- Consumes: `ContextMeta/PersistedMessageRecord`(Task 4)、`estimate_*/LAYER2_HEAD_CHARS`(Task 3)、`Settings`。
- Produces:
  - `LayeredView(layer2_rendered, layer1, summary, l1_tokens, l2_tokens, summary_tokens, span_meta)`(dataclass)
  - `build_layered_view(messages, meta: ContextMeta, settings) -> LayeredView`
  - `needs_reconcile(messages) -> bool`
  - `reconcile_checkpoint_ids(messages, records: list[PersistedMessageRecord]) -> list | None`(None = 对齐失败)
  - `render_history_text(view: LayeredView) -> str`
  - `evaluate_degrade(merged_messages, layer1_from: int | None, budget_l1: int) -> int | None`
  - `tool_omitted_text(name) -> str`(一行标识统一出处)

- [ ] **Step 1: 失败测试** — 写 `tests/test_context_layers.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.context_layers import (
    build_layered_view, evaluate_degrade, needs_reconcile, reconcile_checkpoint_ids,
    render_history_text, tool_omitted_text,
)
from app.sessions import ContextMeta, PersistedMessageRecord


def _mk(mid, m):
    return m.model_copy(update={"additional_kwargs": {**m.additional_kwargs, "db_id": mid}})


def _settings():
    from app.config import Settings
    return Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                    database_url="mysql+pymysql://u:p@h/d")


def _history():
    out, mid = [], 0
    for i in range(3):
        mid += 1
        out.append(_mk(mid, HumanMessage(content=f"问题{i}")))
        mid += 1
        out.append(_mk(mid, AIMessage(content=f"回答{i}" + "长" * 200)))
    return out  # ids 1..6


def test_three_layer_split_and_render():
    msgs = _history()
    meta = ContextMeta("早期摘要:订单A1001", summary_upto=2, layer1_from=4)
    v = build_layered_view(msgs, meta, _settings())
    assert v.summary == "早期摘要:订单A1001"
    assert [m.content for m in v.layer2_rendered] == ["问题1", "回答1" + "长" * 47 + "…"]
    assert [m.content for m in v.layer1] == ["问题2", "回答2" + "长" * 200]
    assert v.l1_tokens > 0 and v.l2_tokens > 0 and v.summary_tokens > 0


def test_null_anchors_all_layer1_and_none_content_safe():
    msgs = _history() + [_mk(7, AIMessage(content=None, tool_calls=[{"name": "t", "args": {}, "id": "c", "type": "tool_call"}]))]
    v = build_layered_view(msgs, ContextMeta(None, None, None), _settings())
    assert len(v.layer1) == len(msgs) and v.layer2_rendered == [] and v.summary is None
    assert v.l1_tokens > 0  # content=None 不炸


def test_tool_message_inherits_group_id():
    msgs = _history()
    msgs.insert(3, ToolMessage(content="x" * 500, tool_call_id="c1", name="query_order"))
    meta = ContextMeta(None, None, 2)   # 组 id=3(assistant)>2 → 层1;若按 0 会错进层2
    v = build_layered_view(msgs, meta, _settings())
    assert any(isinstance(m, ToolMessage) for m in v.layer1)


def test_layer2_tool_omitted():
    msgs = _history()
    tm = ToolMessage(content="x" * 500, tool_call_id="c1", name="query_order")
    msgs.insert(3, _mk(3, tm))  # 该 tool 自带 id 落层2?否——tool 不落库;这里验证截短形态即可
    v = build_layered_view(msgs, ContextMeta(None, None, 4), _settings())
    texts = [m.content for m in v.layer2_rendered if isinstance(m, ToolMessage)]
    assert texts == [tool_omitted_text("query_order")]


def test_needs_reconcile_and_passthrough():
    assert not needs_reconcile(_history())
    assert needs_reconcile([HumanMessage(content="没盖章")])


def _records(tool_rows: bool):
    recs, mid = [], 0
    for i in range(2):
        mid += 1
        recs.append(PersistedMessageRecord(str(mid), "user", f"问题{i}", None, None))
        mid += 1
        recs.append(PersistedMessageRecord(str(mid), "assistant", f"回答{i}", None, None))
        if tool_rows and i == 0:
            mid += 1
            recs.append(PersistedMessageRecord(str(mid), "tool", "env", None, "c1"))
    return recs


def test_reconcile_legacy_tool_rows_and_new_db():
    msgs = [HumanMessage(content="问题0"), AIMessage(content="回答0"),
            ToolMessage(content="r", tool_call_id="c1", name="t"),
            HumanMessage(content="问题1"), AIMessage(content="回答1")]
    out = reconcile_checkpoint_ids(msgs, _records(tool_rows=True))
    assert out is not None and out[0].additional_kwargs["db_id"] == 1
    assert out[3].additional_kwargs["db_id"] == 4
    out2 = reconcile_checkpoint_ids(msgs, _records(tool_rows=False))
    assert out2 is not None  # 新库无 tool 行:过滤后对齐


def test_reconcile_duplicate_content_fails():
    msgs = [HumanMessage(content="一样"), HumanMessage(content="一样")]
    recs = [PersistedMessageRecord("1", "user", "一样", None, None),
            PersistedMessageRecord("2", "user", "一样", None, None)]
    assert reconcile_checkpoint_ids(msgs, recs) is None  # 歧义,不猜 ID


def test_reconcile_trailing_uncommitted_fails():
    msgs = [HumanMessage(content="问题0"), AIMessage(content="回答0"),
            HumanMessage(content="没落库的问")]
    assert reconcile_checkpoint_ids(msgs, _records(tool_rows=False)) is None


def test_render_history_text():
    msgs = _history()
    meta = ContextMeta("摘要:退款诉求", 2, 4)
    text = render_history_text(build_layered_view(msgs, meta, _settings()))
    assert text.startswith("早期对话摘要:\n摘要:退款诉求")
    assert "用户:问题1" in text and "助手:回答1" in text and "用户:问题2" in text


def test_evaluate_degrade_moves_boundary():
    msgs = _history()  # 3 轮,assistant 各 200+ 字
    boundary = evaluate_degrade(msgs, None, budget_l1=60)
    assert boundary is not None and boundary < 6     # 只保留末尾装进 60 token 的部分
    kept = [m for m in msgs if (m.additional_kwargs.get("db_id") or 0) > boundary]
    assert kept and kept[0].additional_kwargs["db_id"] % 2 == 1  # 切在 user 边界(奇数 id)
    assert evaluate_degrade(msgs, None, budget_l1=10**6) is None


def test_evaluate_degrade_single_turn_over_budget():
    msgs = _history()[-2:]  # 一轮就 200+ 字
    boundary = evaluate_degrade(msgs, None, budget_l1=10)
    assert boundary == 6  # 层1 空:边界推到最末,层2/摘要下轮接住
```

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_context_layers.py -v`,FAIL。

- [ ] **Step 3: 实现** — 写 `app/services/context_layers.py`:

```python
"""ch07 三层历史组装(spec §7):纯函数;DB 读由调用方注入。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import trim_messages

from app.services.token_budget import (
    LAYER2_HEAD_CHARS, estimate_message, estimate_messages, estimate_tokens,
)

TOOL_OMITTED = "[工具结果已省略: {}]"


def tool_omitted_text(name: str | None) -> str:
    return TOOL_OMITTED.format(name or "tool")


@dataclass
class LayeredView:
    layer2_rendered: list[BaseMessage]
    layer1: list[BaseMessage]
    summary: str | None
    l1_tokens: int
    l2_tokens: int
    summary_tokens: int
    span_meta: dict = field(default_factory=dict)


def _db_id(msg) -> int | None:
    v = (getattr(msg, "additional_kwargs", None) or {}).get("db_id")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _effective_ids(messages) -> list[int]:
    """顺序扫描:无 db_id 的消息(未落库的 ToolMessage)继承最近已盖章 id。"""
    out, cur = [], 0
    for m in messages:
        mid = _db_id(m)
        if mid is not None:
            cur = mid
        out.append(cur)
    return out


def _render_layer2(msg) -> BaseMessage:
    if isinstance(msg, ToolMessage):
        return msg.model_copy(update={"content": tool_omitted_text(msg.name)})
    if isinstance(msg, AIMessage):
        content = msg.content if isinstance(msg.content, str) else ""
        if len(content) > LAYER2_HEAD_CHARS:
            return msg.model_copy(
                update={"content": content[:LAYER2_HEAD_CHARS] + "…"})
    return msg  # 用户原话不动


def build_layered_view(messages, meta, settings) -> LayeredView:
    summary_upto = meta.summary_upto or 0
    layer1_from = meta.layer1_from if meta.layer1_from is not None else -1  # NULL=负无穷,全部层1
    layer2, layer1 = [], []
    for m, eff in zip(messages, _effective_ids(messages)):
        if eff <= summary_upto:
            continue
        (layer2 if eff <= layer1_from else layer1).append(m)
    rendered = [_render_layer2(m) for m in layer2]
    return LayeredView(layer2_rendered=rendered, layer1=layer1, summary=meta.summary,
                       l1_tokens=estimate_messages(layer1),
                       l2_tokens=estimate_messages(rendered),
                       summary_tokens=estimate_tokens(meta.summary or ""),
                       span_meta={"summary_upto": meta.summary_upto,
                                  "layer1_from": meta.layer1_from})


def needs_reconcile(messages) -> bool:
    return any(not isinstance(m, ToolMessage) and _db_id(m) is None for m in messages)


def _ua_key_msg(m) -> tuple:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    content = m.content if isinstance(m.content, str) else ""
    calls = json.dumps(getattr(m, "tool_calls", None) or None,
                       ensure_ascii=False, sort_keys=True)
    return (role, content, calls)


def _ua_key_rec(r) -> tuple:
    calls = json.dumps(r.tool_calls or None, ensure_ascii=False, sort_keys=True)
    return (r.role, r.content or "", calls)


def reconcile_checkpoint_ids(messages, records):
    """旧 checkpoint(无 db_id)与 messages 表只读对齐;无法唯一对齐 → None(spec §2)。"""
    if not needs_reconcile(messages):
        return list(messages)
    ua_msgs = [(i, m) for i, m in enumerate(messages) if not isinstance(m, ToolMessage)]
    ua_recs = [r for r in records if r.role in ("user", "assistant")]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    tool_recs = [r for r in records if r.role == "tool"]
    if len({_ua_key_msg(m) for _, m in ua_msgs}) != len(ua_msgs):
        return None  # 重复候选:不猜 ID
    if len({_ua_key_rec(r) for r in ua_recs}) != len(ua_recs):
        return None
    if len(ua_msgs) != len(ua_recs):
        return None  # 末尾未完成轮/长度不一
    out = list(messages)
    for (i, m), r in zip(ua_msgs, ua_recs):
        if _ua_key_msg(m) != _ua_key_rec(r):
            return None  # 顺序不一致
        out[i] = m.model_copy(update={
            "additional_kwargs": {**m.additional_kwargs, "db_id": int(r.id)}})
    if tool_recs:  # legacy tool 行:有序按 tool_call_id 配对;新库无 tool 行则跳过
        if len(tool_msgs) != len(tool_recs):
            return None
        for m, r in zip(tool_msgs, tool_recs):
            if (m.tool_call_id or "") != (r.tool_call_id or ""):
                return None
    return out


def render_history_text(view: LayeredView) -> str:
    parts = []
    if view.summary:
        parts.append("早期对话摘要:\n" + view.summary)
    lines = []
    for m in [*view.layer2_rendered, *view.layer1]:
        if isinstance(m, HumanMessage):
            lines.append(f"用户:{m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(tool_omitted_text(m.name))  # 滑窗文本里一律一行标识
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"助手:{m.content}")
    if lines:
        parts.append("对话历史:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def evaluate_degrade(merged_messages, layer1_from: int | None, budget_l1: int) -> int | None:
    """层1 超预算 → 返回新的 layer1_from(被保留首条前一个持久化 id);否则 None。"""
    base = layer1_from if layer1_from is not None else -1
    span = [m for m, eff in zip(merged_messages, _effective_ids(merged_messages))
            if eff > base]
    if not span or estimate_messages(span) <= budget_l1:
        return None
    kept = trim_messages(span, max_tokens=budget_l1, token_counter=estimate_messages,
                         strategy="last", start_on=HumanMessage,
                         include_system=False, allow_partial=False)
    ids = {id(m): eff for m, eff in zip(span, _effective_ids(span))}
    if not kept:
        return _effective_ids(span)[-1]  # 单轮就超:层1 空,边界推到最末
    first_kept = ids.get(id(kept[0]))
    if first_kept is None:
        return None
    earlier = [eff for eff in ids.values() if eff < first_kept]
    if not earlier:
        return None  # 防御:无更早 id 可退
    return max(earlier)
```

注意:`evaluate_degrade` 里 `ids.get(id(kept[0]))` 依赖 trim_messages 返回原对象引用(langchain_core 1.6.2 `strategy="last"` 返回原消息切片,不复制)——实现时先写一个小断言脚本验证;若返回副本,改为按内容+位置匹配回原 span(把 `kept[0]` 换成在 span 中最后一个等于它的消息)。这一步的验证结果写进 dev-notes。

- [ ] **Step 4: 跑绿** — `uv run pytest tests/test_context_layers.py -v` 全绿。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): 三层组装器 + 旧 checkpoint 只读对齐 + 层1 降级评估"`

---

### Task 7: 摘要 prompt + summarizer 后台任务

**Files:**
- Create: `app/prompts/summary.py`、`app/services/summarizer.py`
- Test: `tests/test_summarizer.py`

**Interfaces:**
- Consumes: `SessionStore` 新方法(Task 4/5)、`Settings.summary_max_chars/summary_projection_tokens`。
- Produces:`SUMMARY_PROMPT`(占位 `{background}`/`{transcript}`/`{max_chars}`);`SummaryRunner(store, model, settings)`:`.maybe_trigger(session_id: str, user_id: str) -> None`(同步返回,内部 ensure_future)、`.aclose() -> None`(取消并等待全部 in-flight)。GraphDeps 在 Task 8 消费。

- [ ] **Step 1: 失败测试** — 写 `tests/test_summarizer.py`:

```python
import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from app.services.summarizer import SummaryRunner
from app.sessions import InMemorySessionStore, StoredMessage


class _FakeModel:
    def __init__(self, text="用户问过订单A1001的退款进度,诉求是尽快退款。"):
        self._text = text
        self.calls = 0

    async def ainvoke(self, msgs, **kw):
        self.calls += 1
        prompt = msgs[0].content
        assert "待压缩对话" in prompt and "订单A1001" in prompt
        return AIMessage(content=self._text)


def _settings():
    from app.config import Settings
    return Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                    database_url="mysql+pymysql://u:p@h/d")


async def _seed(store):
    sid = await store.create("u1")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001怎么了"),
                                      StoredMessage("assistant", "在处理,请稍等")])
    upto = int(r.message_ids[-1])
    await store.move_layer1_from(sid, "u1", upto)   # 全部落入层2
    return sid, upto


@pytest.mark.asyncio
async def test_summary_done_and_projection(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid, upto = await _seed(store)
    runner = SummaryRunner(store, _FakeModel(), _settings())
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        runner.maybe_trigger(sid, "u1")
        await asyncio.sleep(0.05)
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary_upto == upto and "A1001" in (meta.summary or "")
    assert "summary start" in caplog.text and "summary done" in caplog.text
    assert "第1段" in caplog.text
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_skip_empty_span_and_inflight(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u1")                    # 无消息 → 区间为空
    model = _FakeModel()
    runner = SummaryRunner(store, model, _settings())
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        runner.maybe_trigger(sid, "u1")
        await asyncio.sleep(0.05)
    assert model.calls == 0 and "summary skip" in caplog.text
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_cas_failure_keeps_anchor(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid, upto = await _seed(store)
    await store.append_summary(sid, 0, upto, "别人先压完了", 10_000)  # 抢占锚点
    runner = SummaryRunner(store, _FakeModel(), _settings())
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        runner.maybe_trigger(sid, "u1")
        await asyncio.sleep(0.05)
    meta = await store.get_context_meta(sid, "u1")
    assert "别人先压完了" in (meta.summary or "")                      # 未被覆盖
    assert "summary skip" in caplog.text
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_truncates_overflow(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid, upto = await _seed(store)
    runner = SummaryRunner(store, _FakeModel("长" * 500), _settings())
    runner.maybe_trigger(sid, "u1")
    await asyncio.sleep(0.05)
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary is not None and len(meta.summary) <= 200       # summary_max_chars
    await runner.aclose()
```

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_summarizer.py -v`,FAIL。

- [ ] **Step 3: 实现** — 写 `app/prompts/summary.py`:

```python
SUMMARY_PROMPT = """你是客服会话的摘要员。把一段客服与用户的对话压缩成一段事实梗概。

规则:
1. 只提炼事实与诉求:问过哪款商品、报过的订单号/手机号、明确诉求、还没解决的问题。
2. 对话里没出现的内容一个字都不许编。
3. 寒暄、闲聊、客套话一律不留。
4. 输出一段连贯中文,不超过 {max_chars} 字;不分点、不解释、不输出其他内容。

已有背景(仅供理解指代,不要复述,不要改写它):
{background}

待压缩对话:
{transcript}"""
```

写 `app/services/summarizer.py`:

```python
"""ch07 后台摘要任务(spec §9):触发只挪锚点不搬数据;CAS 追加,单调前移。"""

import asyncio
import contextlib
import logging
import time

from langchain_core.messages import HumanMessage

from app.prompts.summary import SUMMARY_PROMPT

logger = logging.getLogger("wayhelp.graph")


class SummaryRunner:
    def __init__(self, store, model, settings):
        self._store = store
        self._model = model
        self._settings = settings
        self._inflight: dict[str, asyncio.Task] = {}

    def maybe_trigger(self, session_id: str, user_id: str) -> None:
        task = self._inflight.get(session_id)
        if task is not None and not task.done():
            logger.info("summary skip session=%s reason=in-flight", session_id)
            return
        self._inflight[session_id] = asyncio.ensure_future(
            self._run_guarded(session_id, user_id))

    async def _run_guarded(self, session_id: str, user_id: str) -> None:
        try:
            await self._run(session_id, user_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("summary failed session=%s", session_id)

    async def _run(self, session_id: str, user_id: str) -> None:
        started = time.monotonic()
        meta = await self._store.get_context_meta(session_id, user_id)
        from_id = meta.summary_upto or 0
        upto = meta.layer1_from
        if upto is None or upto <= from_id:
            logger.info("summary skip session=%s reason=empty-span", session_id)
            return
        rows = [(i, r, c) for i, r, c in
                await self._store.fetch_span_texts(session_id, from_id, upto) if c]
        if not rows:
            logger.info("summary skip session=%s reason=empty-span", session_id)
            return
        logger.info("summary start session=%s 区间 (%d, %d]", session_id, from_id, upto)
        transcript = "\n".join(
            f"{'用户' if r == 'user' else '助手'}:{c}" for _, r, c in rows)
        prompt = (SUMMARY_PROMPT
                  .replace("{background}", meta.summary or "(无)")
                  .replace("{transcript}", transcript)
                  .replace("{max_chars}", str(self._settings.summary_max_chars)))
        resp = await self._model.ainvoke([HumanMessage(content=prompt)])
        content = (resp.content if isinstance(resp.content, str) else "").strip()
        if len(content) > self._settings.summary_max_chars:
            content = content[: self._settings.summary_max_chars]
            logger.info("summary truncated session=%s", session_id)
        result = await self._store.append_summary(
            session_id, from_id, upto, content, self._settings.summary_projection_tokens)
        if not result.applied:
            logger.info("summary skip session=%s reason=%s", session_id, result.reason)
            return
        logger.info("summary done session=%s 第%d段 覆盖 (%d, %d] 耗时 %dms",
                    session_id, result.seq, from_id, upto,
                    int((time.monotonic() - started) * 1000))

    async def aclose(self) -> None:
        for task in self._inflight.values():
            task.cancel()
        for task in self._inflight.values():
            with contextlib.suppress(Exception):
                await task
        self._inflight.clear()
```

- [ ] **Step 4: 跑绿** — `uv run pytest tests/test_summarizer.py -v` 全绿。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): 后台摘要任务(CAS 追加 + in-flight 防重入 + 失败不挪锚点)"`

---

### Task 8: 图 state + understand/classify 共享分层历史

**Files:**
- Modify: `app/graph/state.py`
- Modify: `app/graph/nodes.py`(`understand_query`、`classify_intent`;删 `_history_turns`/`_render_history`)
- Modify: `app/prompts/intent.py`
- Test: `tests/test_graph_nodes.py`、`tests/test_graph_state.py`

**Interfaces:**
- Consumes: `build_layered_view/needs_reconcile/reconcile_checkpoint_ids/render_history_text`(Task 6)、`SummaryRunner`(Task 7)、`ContextBudget`(Task 3)。
- Produces:state 新临时字段 `history_block: str`、`history_layer2: list`、`history_layer1: list`、`history_summary: str | None`;`GraphDeps` 新字段 `context_budget: ContextBudget | None = None`、`summary_runner: Any = None`(Task 9/10 消费);`INTENT_PROMPT` 含 `{history_block}` 占位。

- [ ] **Step 1: 失败测试** — `tests/test_graph_nodes.py` 追加(假模型用 conftest 的 `FakeStreamModel`,它已按 prompt marker 分派指代消解罐头;store 用 `InMemorySessionStore`):

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes import GraphDeps, build_front_nodes
from app.sessions import InMemorySessionStore, StoredMessage


def _deps(store, model, settings):
    return GraphDeps(model=model, settings=settings, retriever=None,
                     store=store, system_prompt="SYS")


@pytest.mark.asyncio
async def test_understand_writes_shared_layered_history(caplog):
    from tests.conftest import FakeStreamModel, make_settings
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u1")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001怎么了"),
                                      StoredMessage("assistant", "在处理")])
    stamped = [HumanMessage(content="订单A1001怎么了",
                            additional_kwargs={"db_id": int(r.message_ids[0])}),
               AIMessage(content="在处理",
                         additional_kwargs={"db_id": int(r.message_ids[1])})]
    deps = _deps(store, FakeStreamModel(), make_settings())
    nodes = build_front_nodes(deps)
    state = {"messages": stamped, "raw_query": "那它呢", "node_trace": [],
             "active_order": None}
    import logging
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        out = await nodes["understand_query"](state, {"configurable": {"thread_id": sid, "user_id": "u1"}})
    assert "订单A1001" in out["history_block"]
    assert out["history_layer1"] and out["history_summary"] is None
    assert "history_ctx" in caplog.text


@pytest.mark.asyncio
async def test_understand_short_circuit_only_when_truly_empty():
    from tests.conftest import FakeStreamModel, make_settings
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u1")
    deps = _deps(store, FakeStreamModel(), make_settings())
    nodes = build_front_nodes(deps)
    state = {"messages": [], "raw_query": "你好", "node_trace": [], "active_order": None}
    out = await nodes["understand_query"](state, {"configurable": {"thread_id": sid, "user_id": "u1"}})
    assert out["resolved_query"] == "你好" and out["history_block"] == ""


@pytest.mark.asyncio
async def test_classify_prompt_carries_history_block():
    from tests.conftest import FakeStreamModel, make_settings
    captured = []

    class _Spy(FakeStreamModel):
        async def ainvoke(self, msgs, **kw):
            captured.append(msgs[0].content)
            return await super().ainvoke(msgs, **kw)

    store = InMemorySessionStore(10, 100, 8000)
    deps = _deps(store, _Spy(), make_settings())
    nodes = build_front_nodes(deps)
    state = {"resolved_query": "退款", "history_block": "对话历史:\n用户:订单A1001怎么了",
             "node_trace": []}
    await nodes["classify_intent"](state)
    assert any("订单A1001" in p for p in captured)
```

`tests/test_graph_state.py` 追加:

```python
def test_new_turn_state_resets_history_fields():
    from app.graph.state import new_turn_state
    s = new_turn_state("q")
    assert s["history_block"] == "" and s["history_layer1"] == []
    assert s["history_layer2"] == [] and s["history_summary"] is None
    assert "messages" not in s  # 跨轮保留契约不变
```

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_graph_nodes.py tests/test_graph_state.py -v`,FAIL。

- [ ] **Step 3: 实现** —
  - `app/graph/state.py`:`ChatGraphState` 加四字段;`new_turn_state` 加 `"history_block": ""`, `"history_layer2": [], "history_layer1": [], "history_summary": None`。
  - `app/prompts/intent.py`:末尾 `用户问题:{query}` 前插入:

```
对话历史(仅供参考,分类对象是最后的用户问题):
{history_block}

用户问题:{query}"""
```

  - `app/graph/nodes.py`:`GraphDeps` 加 `context_budget: Any = None`、`summary_runner: Any = None`;删 `_history_turns`/`_render_history`;`understand_query` 的历史段重写为:

```python
        sid = (config.get("configurable") or {}).get("thread_id", "")
        meta = await deps.store.get_context_meta(sid, user_id)
        messages = list(state.get("messages") or [])
        stamped_back = None
        if needs_reconcile(messages):
            records = await deps.store.list_checkpoint_records(sid, user_id) or []
            reconciled = reconcile_checkpoint_ids(messages, records)
            if reconciled is None:
                logger.info("context_reconcile_failed session=%s reason=align", sid)
                messages = []
            else:
                messages = stamped_back = reconciled
        view = build_layered_view(messages, meta, deps.settings)
        history_block = render_history_text(view)
        logger.info("history_ctx %s", json.dumps({
            "session": sid, "summary": view.summary,
            "window": [{"role": m.type,
                        "content": m.content if isinstance(m.content, str) else None,
                        "db_id": (m.additional_kwargs or {}).get("db_id")}
                       for m in [*view.layer2_rendered, *view.layer1]],
            "est": {"l1": view.l1_tokens, "l2": view.l2_tokens,
                    "summary": view.summary_tokens}}, ensure_ascii=False))
        base = {"history_block": history_block,
                "history_layer2": view.layer2_rendered,
                "history_layer1": view.layer1,
                "history_summary": view.summary}
        if stamped_back is not None:
            base["messages"] = stamped_back  # 同 .id 原地替换,把 db_id 写回 checkpoint
        if not history_block and active_summary is None:
            return {"resolved_query": state["raw_query"], "active_order": active,
                    "node_trace": trace, **base}
        block_parts = []
        if history_block:
            block_parts.append(history_block)
        if active_summary:
            block_parts.append("已确认会话订单:" + active_summary)
        prompt = (UNDERSTAND_PROMPT ...)
        ...
        # 成功/降级两个 return 都带 **base
```

  - `classify_intent`:`prompt = (INTENT_PROMPT.replace("{history_block}", state.get("history_block") or "(无对话历史)").replace("{query}", state["resolved_query"]))`。

- [ ] **Step 4: 跑绿 + 回归** — `uv run pytest tests/test_graph_nodes.py tests/test_graph_state.py -v` 全绿;`uv run pytest tests/test_graph_builder.py tests/test_graph_smoke.py tests/test_ch05_acceptance.py -v` 回归(若 FakeStreamModel 的指代消解罐头匹配不上新 history_block 文案,按其 marker 规则适配)。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): understand/classify 共享分层历史 + history_ctx 日志"`

---

### Task 9: executor token 截断 + main_agent 三层拼装 + tool_chat_chain 退役

**Files:**
- Modify: `app/tools/executor.py`(`max_result_chars` → `max_result_tokens`,截断换 `truncate_tool_result`)
- Modify: `app/graph/agent_node.py`(executor 构造、`build_agent_context` 调用、`model_ctx` 日志、预算换尺)
- Modify: `app/chains/tool_chat_chain.py`(重写 `build_agent_context`;删 `fit_tool_context`/`build_messages_with_tools`/`_drop_broken_head`/`_validate`;`rebuild_messages`/`_tool_groups_complete` 先 `grep -rn` 确认无其他调用方再删,有则保留)
- Test: `tests/test_tool_chat_chain.py`(按新语义重写)、`tests/test_agent_node.py`(适配)、executor 测试文件(`grep -rln "ToolExecutor" tests/` 定位)

**Interfaces:**
- Consumes: `estimate_messages/measure_sys_tokens/truncate_tool_result/ContextBudget`(Task 3)、state 的 `history_layer1/history_layer2/history_summary`(Task 8)。
- Produces:`build_agent_context(system_prompt: str, layer2: list, layer1: list, turn_messages: list, background_text: str | None, guard_tokens: int) -> list | None`;`ToolExecutor(registry, timeout_seconds, max_retries, max_result_tokens, write_tools=..., tool_policies=...)`。

- [ ] **Step 1: 失败测试** — `tests/test_tool_chat_chain.py` 全量重写为:

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.chains.tool_chat_chain import build_agent_context
from app.services.token_budget import estimate_messages


def _turn():
    return [HumanMessage(content="现在这句"),
            AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
            ToolMessage(content="结果", tool_call_id="c1", name="query_order")]


def test_layout_order_and_background_position():
    ctx = build_agent_context("SYS", [HumanMessage(content="层2旧问")],
                              [HumanMessage(content="层1近问"), AIMessage(content="近答")],
                              _turn(), "【对话背景】\n摘要", guard_tokens=10**6)
    roles = [(type(m).__name__, m.content) for m in ctx]
    assert roles[0] == ("SystemMessage", "SYS")
    assert roles[1][1] == "层2旧问" and roles[2][1] == "层1近问"
    assert roles[4] == ("HumanMessage", "现在这句")          # 当前用户句
    assert roles[5] == ("HumanMessage", "【对话背景】\n摘要")  # 合成一条紧跟用户句
    assert isinstance(ctx[6], AIMessage) and isinstance(ctx[7], ToolMessage)


def test_background_omitted_when_empty():
    ctx = build_agent_context("SYS", [], [], _turn(), None, 10**6)
    assert [type(m).__name__ for m in ctx] == ["SystemMessage", "HumanMessage",
                                               "AIMessage", "ToolMessage"]


def test_guard_returns_none():
    assert build_agent_context("SYS" * 1, [], [], _turn(), None, guard_tokens=5) is None
    ctx = build_agent_context("SYS", [], [], _turn(), None,
                              guard_tokens=estimate_messages([SystemMessage(content="SYS"), *_turn()]))
    assert ctx is not None  # 边界值恰好放行
```

executor 测试(定位后)追加:`ToolExecutor(registry, 5, 0, max_result_tokens=20)` 执行一个返回长文本的假工具,断言 ToolMessage 内容以 `…[已截断]` 结尾且 `estimate_tokens(内容) <= 20`。

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_tool_chat_chain.py -v`,FAIL(`build_agent_context` 签名不同)。

- [ ] **Step 3: 实现** —
  - `tool_chat_chain.py` 删除退役函数,保留 `finalize_tool_calls`/`merge_tool_call_chunks`(agent_node 在用)与确认仍有调用方的函数;新写:

```python
def build_agent_context(system_prompt: str, layer2: list, layer1: list,
                        turn_messages: list, background_text: str | None,
                        guard_tokens: int) -> list | None:
    """ch07 拼装(spec §8):[system, *层2, *层1, 当前用户句, 背景?, *ReAct步];
    总量超 guard_tokens → None(调用方发 tool_context_too_long)。"""
    from app.services.token_budget import estimate_messages
    if not turn_messages or not isinstance(turn_messages[0], HumanMessage):
        raise RuntimeError("turn_messages must start with the current human message")
    background = [HumanMessage(content=background_text)] if background_text else []
    base = [SystemMessage(content=system_prompt), *layer2, *layer1,
            turn_messages[0], *background, *turn_messages[1:]]
    return base if estimate_messages(base) <= guard_tokens else None
```

  - `executor.py`:`__init__` 形参 `max_result_chars` 改 `max_result_tokens`,存 `self._max_tokens`;`_success`/`_error` 里 `truncate_content(...)` 换 `truncate_tool_result(content, self._max_tokens)`(import 自 `app.services.token_budget`;`tool_envelope.truncate_content` 不再被引用)。
  - `agent_node.py`:
    - executor 构造第 4 参改 `settings.tool_result_max_tokens`。
    - 节点入口:`budget = deps.context_budget or compute_budget(deps.settings, measure_sys_tokens(deps.system_prompt, tools))`;`schema_est = measure_sys_tokens("", registry.tools)`(替代 `_tool_schema_tokens`,删该函数)。
    - 上下文调用改:

```python
            layer2 = state.get("history_layer2") or []
            layer1 = state.get("history_layer1") or []
            bg_parts = []
            if state.get("history_summary"):
                bg_parts.append("【对话背景】\n" + state["history_summary"])
            if context_extra:
                bg_parts.append(context_extra)
            background_text = "\n\n".join(bg_parts) or None
            context = build_agent_context(deps.system_prompt, layer2, layer1,
                                          turn_messages, background_text,
                                          settings.model_context_window - settings.max_output_tokens)
            if context is None:
                _abort("tool_context_too_long", "工具结果超出上下文预算")
            logger.info("model_ctx %s", json.dumps({
                "session": (config.get("configurable") or {}).get("thread_id", ""),
                "step": steps + 1,
                "budget": {"l1": budget.layer1, "l2": budget.layer2},
                "counts": {"l1": len(layer1), "l2": len(layer2)},
                "est": {"l1": estimate_messages(layer1), "l2": estimate_messages(layer2),
                        "summary": estimate_tokens(state.get("history_summary") or ""),
                        "total": estimate_messages(context)},
                "summary": state.get("history_summary"),
                "layer2": [_ctx_msg(m) for m in layer2],
                "layer1": [_ctx_msg(m) for m in layer1],
            }, ensure_ascii=False))
            est_input = estimate_messages(context) + schema_est
```

    - 文件顶部加 `_ctx_msg(m) = {"role": m.type, "content": m.content if isinstance(m.content, str) else None, "db_id": (m.additional_kwargs or {}).get("db_id")}`(写成函数);import 补 `json`、`estimate_messages/estimate_tokens/compute_budget/measure_sys_tokens`;删 `count_tokens_approximately` 引用(213 行的 reserve 改用 `est_input`)。

- [ ] **Step 4: 跑绿 + 回归** — `uv run pytest tests/test_tool_chat_chain.py tests/test_agent_node.py -v`;`test_agent_node.py` 里直接构造 state 调 main_agent 的用例,历史相关断言改挂 `history_layer1/history_layer2`(缺的按空层处理,行为变化在 dev-notes 记一笔);`uv run pytest tests/test_chat_api_tools.py tests/test_router_coverage.py -v` 回归。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): main_agent 三层拼装与 model_ctx 日志 + 工具结果 token 截断"`

---

### Task 10: log 节点(PersistedTurn / 盖章 / 降级 / 摘要触发)

**Files:**
- Modify: `app/graph/nodes.py`(`_to_stored`、`build_log_node`)
- Test: `tests/test_graph_nodes.py`(单元)、`tests/test_context_maintenance.py`(新建,图集成)

**Interfaces:**
- Consumes:`PersistedTurn/CommitTurnResult.message_ids`(Task 4)、`evaluate_degrade/build_layered_view`(Task 6)、`SummaryRunner`(Task 7)、`ContextBudget`(Task 3)。
- Produces:`_to_stored(turn_messages) -> PersistedTurn`(不再收 settings 参;不产 tool 行);log 节点在 commit 后盖章并按预算降级/触发摘要。

- [ ] **Step 1: 失败测试(单元)** — `tests/test_graph_nodes.py` 追加:

```python
def test_to_stored_persisted_turn_no_tool_rows():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.graph.nodes import _to_stored
    tm = [HumanMessage(content="问"),
          AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}]),
          ToolMessage(content="大结果JSON", tool_call_id="c1", name="query_order"),
          AIMessage(content="答")]
    pack = _to_stored(tm)
    assert [m.role for m in pack.stored] == ["user", "assistant", "assistant"]
    assert pack.checkpoint_indexes == [0, 1, 3]          # ToolMessage 不落库,索引跳 2
    assert pack.stored[1].tool_calls and pack.stored[1].content is None
```

- [ ] **Step 2: 失败测试(图集成)** — 写 `tests/test_context_maintenance.py`(镜像 `tests/test_graph_smoke.py` 的 harness:`build_chat_graph` + `InMemorySaver` + conftest `ScriptedChatModel`;先 Read 该文件对齐构造方式):

```python
import asyncio
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph
from app.graph.nodes import GraphDeps
from app.graph.state import new_turn_state
from app.services.summarizer import SummaryRunner
from app.services.token_budget import compute_budget
from app.sessions import InMemorySessionStore
from tests.conftest import ScriptedChatModel, make_settings


def _tiny_settings():
    # 极小窗口:BUDGET 只够装一两轮 → 强制降级 + 触发摘要
    return make_settings(model_context_window=3000, max_output_tokens=200,
                         max_user_input_tokens=200, max_agent_steps=1,
                         tool_result_max_tokens=100, rerank_top_k=1,
                         history_target_turns=50, steady_tokens_per_turn=10,
                         safety_margin_tokens=0, summary_projection_tokens=200)


@pytest.mark.asyncio
async def test_checkpoint_stamped_and_degrade_and_summary():
    settings = _tiny_settings()
    store = InMemorySessionStore(10, 1000, 8000)
    model = ScriptedChatModel()
    budget = compute_budget(settings, sys_tokens=0)
    runner = SummaryRunner(store, model, settings)
    deps = GraphDeps(model=model, settings=settings, retriever=None, store=store,
                     system_prompt="SYS", context_budget=budget, summary_runner=runner)
    graph = build_chat_graph(deps, InMemorySaver())
    sid = await store.create("u1")
    config = {"configurable": {"thread_id": sid, "user_id": "u1"}}
    for q in ["第一问订单A1001", "第二问", "第三问", "第四问"]:
        async for _ in graph.astream(new_turn_state(q), config,
                                     stream_mode=["custom"]):
            pass
    st = await graph.aget_state(config)
    msgs = st.values["messages"]
    assert msgs and all(
        m.additional_kwargs.get("db_id") for m in msgs
        if m.type in ("human", "ai"))                       # 盖章进 checkpoint
    meta = await store.get_context_meta(sid, "u1")
    assert meta.layer1_from is not None                     # 层1 降级已持久化
    await asyncio.sleep(0.1)                                # 摘要任务后台落地
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary is not None and meta.summary_upto is not None
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_does_not_block_turn():
    settings = _tiny_settings()
    store = InMemorySessionStore(10, 1000, 8000)
    model = ScriptedChatModel()
    runner = SummaryRunner(store, model, settings)
    deps = GraphDeps(model=model, settings=settings, retriever=None, store=store,
                     system_prompt="SYS",
                     context_budget=compute_budget(settings, sys_tokens=0),
                     summary_runner=runner)
    graph = build_chat_graph(deps, InMemorySaver())
    sid = await store.create("u1")
    config = {"configurable": {"thread_id": sid, "user_id": "u1"}}
    async for _ in graph.astream(new_turn_state("触发摘要的一问" + "长" * 300), config,
                                 stream_mode=["custom"]):
        pass
    # astream 返回即本轮结束;此时摘要任务可能尚未完成,但 meta 不因此炸
    meta = await store.get_context_meta(sid, "u1")
    assert meta is not None
    await runner.aclose()
```

- [ ] **Step 3: 跑红** — `uv run pytest tests/test_graph_nodes.py::test_to_stored_persisted_turn_no_tool_rows tests/test_context_maintenance.py -v`,FAIL。

- [ ] **Step 4: 实现** — `app/graph/nodes.py`:
  - `_to_stored` 重写(删 settings 参与 ToolMessage 分支;`wrap` import 若无其他引用一并删):

```python
def _to_stored(turn_messages) -> PersistedTurn:
    """本轮消息 → 落库行 + checkpoint 索引映射(spec §6);tool 结果不落库。"""
    stored: list[StoredMessage] = []
    indexes: list[int] = []
    for i, m in enumerate(turn_messages):
        if isinstance(m, HumanMessage):
            stored.append(StoredMessage("user", m.content))
            indexes.append(i)
        elif isinstance(m, AIMessage):
            stored.append(StoredMessage("assistant", m.content or None,
                                        tool_calls=m.tool_calls or None))
            indexes.append(i)
    return PersistedTurn(stored, indexes)
```

  - `build_log_node` 的 `log_turn` 改造:`stored_pack = _to_stored(state["turn_messages"])`;`commit_turn(sid, stored_pack.stored, ...)`;commit 成功后:

```python
        stamped = list(state["turn_messages"])
        for row_id, msg_idx in zip(result.message_ids, stored_pack.checkpoint_indexes):
            m = stamped[msg_idx]
            stamped[msg_idx] = m.model_copy(update={
                "additional_kwargs": {**m.additional_kwargs, "db_id": int(row_id)}})
        user_id = (config.get("configurable") or {}).get("user_id", "")
        try:  # 上下文维护失败不阻塞本轮回复(锚点下轮再试)
            meta = await deps.store.get_context_meta(sid, user_id)
            budget = deps.context_budget or compute_budget(
                deps.settings, measure_sys_tokens(deps.system_prompt, []))
            merged = [*state["messages"], *stamped]
            new_l1 = evaluate_degrade(merged, meta.layer1_from, budget.layer1)
            if new_l1 is not None:
                moved = await deps.store.move_layer1_from(sid, user_id, new_l1)
                if moved:
                    logger.info("层1 降级 session=%s %s→%d", sid,
                                meta.layer1_from if meta.layer1_from is not None else "-",
                                new_l1)
                    meta = replace(meta, layer1_from=new_l1)
            view = build_layered_view(merged, meta, deps.settings)
            if view.l2_tokens > budget.layer2:
                logger.info("summary trigger session=%s 层2 约 %d token > 预算 %d",
                            sid, view.l2_tokens, budget.layer2)
                if deps.summary_runner is not None:
                    deps.summary_runner.maybe_trigger(sid, user_id)
        except Exception:
            logger.exception("context maintenance failed session=%s", sid)
```

  - `out = {"messages": stamped, ...}`(原 `state["turn_messages"]` 换成 `stamped`;citations/suggest/active_order 逻辑不动)。
  - import 补:`compute_budget/measure_sys_tokens/evaluate_degrade/build_layered_view/PersistedTurn`。

- [ ] **Step 5: 跑绿 + 回归** — `uv run pytest tests/test_graph_nodes.py tests/test_context_maintenance.py -v` 全绿;`uv run pytest tests/test_graph_persistence.py tests/test_refund_flow.py tests/test_ch05_acceptance.py tests/test_chat_api.py -v` 回归(commit 无 tool 行可能影响断言落库行数的用例,按新语义改)。

- [ ] **Step 6: Commit** — `git commit -am "feat(ch07): log 节点盖章 db_id + 层1 降级持久化 + 摘要异步触发"`

---

### Task 11: prepare 早闸 + 启动自检 + FileHandler

**Files:**
- Modify: `app/chains/chat_chain.py`(加 `check_user_input`)、`app/services/chat_service.py:138`
- Modify: `app/main.py`(日志配置、预算自检、探针替换、GraphDeps 接线、lifespan 关闭 runner)
- Modify: `.gitignore`(加 `log/`)
- Test: `tests/test_chat_service.py`、`tests/test_main_boot.py`

**Interfaces:**
- Produces:`check_user_input(text: str, max_tokens: int) -> None`(超 `MessageTooLongError`);启动日志 `context budget …`(全分量)与 critical `上下文预算不足`;`create_app` 多次调用 FileHandler 幂等。

- [ ] **Step 1: 失败测试** — `tests/test_chat_service.py:103-104` 用例改写:

```python
async def test_prepare_user_input_token_gate():
    service, store, _ = make_service([BUSINESS])   # 沿用文件内既有 helper
    with pytest.raises(MessageTooLongError):
        await service.prepare("u", None, "汉" * 2001)   # MAX_USER_INPUT_TOKENS 默认 2000
```

`tests/test_main_boot.py` 追加:

```python
def test_file_handler_idempotent_and_budget_log(caplog):
    import logging
    from app.main import create_app
    from tests.conftest import make_runtime, make_settings
    settings = make_settings()
    create_app(settings=settings, model=object(), runtime=make_runtime(settings))
    create_app(settings=settings, model=object(), runtime=make_runtime(settings))
    root = logging.getLogger()
    fhs = [h for h in root.handlers
           if isinstance(h, logging.FileHandler) and h.baseFilename.endswith("app.log")]
    assert len(fhs) == 1


def test_budget_selfcheck_critical_on_tiny_window(caplog):
    import logging
    from app.main import create_app
    from tests.conftest import make_runtime, make_settings
    tiny = make_settings(model_context_window=100, safety_margin_tokens=0)
    with caplog.at_level(logging.CRITICAL):
        create_app(settings=tiny, model=object(), runtime=make_runtime(tiny))
    assert "上下文预算不足" in caplog.text
```

(`create_app` 的 model 形参若是 object() 会在 bind_tools 处炸——给 conftest 的 `FakeStreamModel()`;`make_runtime` 现状签名先 Read conftest 对齐。)

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_chat_service.py tests/test_main_boot.py -v`,FAIL。

- [ ] **Step 3: 实现** —
  - `chat_chain.py` 追加:

```python
def check_user_input(text: str, max_tokens: int) -> None:
    """ch07:用户输入 token 闸(只量输入本身,§4 尺)。"""
    from app.services.token_budget import estimate_tokens
    if estimate_tokens(text) > max_tokens:
        raise MessageTooLongError("message exceeds MAX_USER_INPUT_TOKENS")
```

  - `chat_service.py:138` 换成 `check_user_input(message, self._settings.max_user_input_tokens)`(import 同步);`check_input_budget` 若全仓仅剩测试引用则连函数带测试一起删。
  - `main.py`:
    - 日志:`create_app` 开头(原 basicConfig 处)改:

```python
    logging.basicConfig()
    log_dir = Path(__file__).resolve().parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    target = str(log_dir / "app.log")
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target
               for h in root.handlers):
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(fh)
    logging.getLogger("wayhelp.graph").setLevel(logging.INFO)
```

    - 预算自检(替换 :121 的 system-prompt 探针;extract 探针 :126-130 保留):

```python
    from app.graph.agent_node import suggest_options
    from app.services.token_budget import compute_budget, measure_sys_tokens
    from app.tools.business import build_graph_tools
    probe_tools = [*build_graph_tools("sys-probe", "", None, settings, with_faq=True).tools,
                   suggest_options]
    context_budget = compute_budget(settings, measure_sys_tokens(SERVICE_SYSTEM_PROMPT, probe_tools))
    logging.getLogger("wayhelp.graph").info(
        "context budget window=%d output=%d user_input=%d peak=%d fixed=%d sys=%d avail=%d total=%d l1=%d l2=%d",
        settings.model_context_window, settings.max_output_tokens,
        settings.max_user_input_tokens, context_budget.peak, context_budget.fixed,
        context_budget.sys_tokens, context_budget.avail, context_budget.total,
        context_budget.layer1, context_budget.layer2)
    if not context_budget.sufficient:
        logging.getLogger("wayhelp.graph").critical(
            "上下文预算不足:total=%d < steady=%d,请调大 MODEL_CONTEXT_WINDOW 或调小固定开销",
            context_budget.total, settings.steady_tokens_per_turn)
```

    - `GraphDeps(...)` 加 `context_budget=context_budget, summary_runner=SummaryRunner(runtime.store, model, settings)`(import `app.services.summarizer`)。
    - lifespan:`await app.state.job_runner.close()` 旁加 `await deps_summary_runner.aclose()`(把 runner 存 `app.state.summary_runner`,lifespan 取);放在 yield 之后、retriever.close 之前的收尾段。
    - `.gitignore` 加一行 `log/`。

- [ ] **Step 4: 跑绿 + 回归** — `uv run pytest tests/test_chat_service.py tests/test_main_boot.py tests/test_lifespan.py tests/test_config.py -v`。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): 输入 token 闸 + 启动预算自检 + log/app.log FileHandler"`

---

### Task 12: 只读接口 GET /api/conversations(+/{id}/messages)

**Files:**
- Modify: `app/schemas.py`
- Create: `app/routers/conversations.py`
- Modify: `app/main.py`(include_router)
- Test: `tests/test_conversations_api.py`

**Interfaces:**
- Consumes:`SessionStore.list_conversations/list_messages`(Task 4/5)。
- Produces:`ConversationItemOut`、`ConversationMessageOut` 两个响应模型;路由挂 `/api`。

- [ ] **Step 1: 失败测试** — 写 `tests/test_conversations_api.py`(harness 镜像 `tests/test_chat_api.py` 的 TestClient 构造,先 Read 对齐):

```python
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.sessions import StoredMessage
from tests.conftest import FakeStreamModel, make_runtime, make_settings


@pytest.fixture()
def client_and_store():
    settings = make_settings()
    runtime = make_runtime(settings)
    app = create_app(settings=settings, model=FakeStreamModel(), runtime=runtime)
    return TestClient(app), runtime.store


def test_conversations_empty(client_and_store):
    client, _ = client_and_store
    r = client.get("/api/conversations", params={"user_id": "u1"})
    assert r.status_code == 200 and r.json() == []


@pytest.mark.asyncio
async def test_conversations_and_messages_flow(client_and_store):
    client, store = client_and_store
    sid = await store.create("u1")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001能退吗"),
                                      StoredMessage("assistant", "可以,政策允许")])
    upto = int(r.message_ids[-1])
    await store.move_layer1_from(sid, "u1", upto)
    await store.append_summary(sid, 0, upto, "用户问订单A1001退款", 10_000)
    lst = client.get("/api/conversations", params={"user_id": "u1"})
    assert lst.status_code == 200
    item = lst.json()[0]
    assert item["id"] == sid and item["preview"] == "订单A1001能退吗"
    assert item["summarized"] is True and "T" in item["created_at"]      # ISO 8601
    msgs = client.get(f"/api/conversations/{sid}/messages", params={"user_id": "u1"})
    assert [m["role"] for m in msgs.json()] == ["user", "assistant"]
    assert msgs.json()[0]["content"] == "订单A1001能退吗"
    nf = client.get(f"/api/conversations/{sid}/messages", params={"user_id": "other"})
    assert nf.status_code == 404
```

(异步 fixture 在同步 TestClient 里写库:用 `asyncio.run` 或把 fixture 改同步包装;实现时按 conftest 现有模式对齐——若 `test_chat_action.py` 已有「先写库再打 API」的先例,照抄其模式。)

- [ ] **Step 2: 跑红** — `uv run pytest tests/test_conversations_api.py -v`,FAIL(404 路由不存在)。

- [ ] **Step 3: 实现** —
  - `app/schemas.py` 追加:

```python
class ConversationItemOut(BaseModel):
    id: str
    status: str
    created_at: str | None
    updated_at: str | None
    preview: str | None
    summarized: bool


class ConversationMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str | None
```

  - 新建 `app/routers/conversations.py`:

```python
"""ch07 会话只读接口(spec §11):侧栏列表 + 历史回放;零写动作。"""

from fastapi import APIRouter, Request

from app.errors import SessionNotFoundError
from app.schemas import ConversationItemOut, ConversationMessageOut

router = APIRouter(prefix="/api", tags=["conversations"])


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


@router.get("/conversations", response_model=list[ConversationItemOut])
async def list_conversations(user_id: str, request: Request):
    items = await request.app.state.store.list_conversations(user_id)
    return [ConversationItemOut(id=i.id, status=i.status, created_at=_iso(i.created_at),
                                updated_at=_iso(i.updated_at), preview=i.preview,
                                summarized=i.summarized) for i in items]


@router.get("/conversations/{session_id}/messages",
            response_model=list[ConversationMessageOut])
async def list_messages(session_id: str, user_id: str, request: Request):
    rows = await request.app.state.store.list_messages(session_id, user_id)
    if rows is None:
        raise SessionNotFoundError("session not found")  # 404 不泄露
    return [ConversationMessageOut(id=r.id, role=r.role, content=r.content,
                                   created_at=_iso(r.created_at)) for r in rows]
```

  - `app/main.py`:import 并 `app.include_router(conversations_router)`;若 404 测试返回 500,Read main.py 尾部异常映射,把 `SessionNotFoundError` 按现有 `AppError` 同款 handler 注册(chat 路由已依赖它,理论上已存在)。

- [ ] **Step 4: 跑绿** — `uv run pytest tests/test_conversations_api.py -v` 全绿。

- [ ] **Step 5: Commit** — `git commit -am "feat(ch07): 会话只读接口 /api/conversations 与历史回放"`

---

### Task 13: 前端会话侧栏(chat.html,Vibe Coding 例外)

**Files:**
- Modify: `app/static/chat.html`
- Test: `tests/test_chat_page.py`(字符串断言)

**Interfaces:**
- Consumes:Task 12 的两个 GET 接口。
- Produces:`#sidebar`/`#conv-list` DOM;`loadConversations()/switchConversation(id)/onNewChat()` JS 函数;静默降级与回放锁。

- [ ] **Step 1: 先 Read 对齐接入点** — `app/static/chat.html` 的 :660-720(布局与「+ 新对话」按钮)、:1290-1345(fetch 与 session 帧处理、新对话 handler),确认:固定演示账号变量名(下称 `USER_ID`,按实际名替换)、消息列表容器 id、session 帧里给 `sessionId` 赋值的位置、现有气泡追加函数名。

- [ ] **Step 2: 失败测试** — `tests/test_chat_page.py` 追加:

```python
def test_chat_page_has_sidebar_and_conversation_api():
    html = _html()  # 沿用文件内既有读取方式
    assert 'id="sidebar"' in html and 'id="conv-list"' in html
    assert "/api/conversations?user_id=" in html
    assert "/api/conversations/${" in html and "/messages?user_id=" in html
    assert "conv-item" in html and "已摘要" in html
    assert "loadConversations" in html and "switchConversation" in html
    assert "display" in html and "catch" in html        # 静默降级路径存在
```

- [ ] **Step 3: 跑红** — `uv run pytest tests/test_chat_page.py -v`,FAIL。

- [ ] **Step 4: 实现** — 在 `chat.html` 主容器外套一层 flex 布局并插入:

```html
<aside id="sidebar">
  <div class="sidebar-head">会话</div>
  <ul id="conv-list"></ul>
</aside>
```

CSS(追加到 `<style>`):

```css
body > .layout { display: flex; align-items: stretch; }
#sidebar { width: 240px; border-right: 1px solid #e5e5e5; overflow-y: auto; padding: 8px; }
#sidebar .sidebar-head { font-weight: 600; margin-bottom: 8px; }
#conv-list { list-style: none; margin: 0; padding: 0; }
#conv-list .conv-item { padding: 8px; border-radius: 6px; cursor: pointer; }
#conv-list .conv-item:hover { background: #f3f3f3; }
#conv-list .conv-item.active { background: #e8f0fe; }
#conv-list .conv-preview { font-size: 13px; overflow: hidden; text-overflow: ellipsis;
                           white-space: nowrap; max-width: 200px; }
#conv-list .conv-meta { font-size: 11px; color: #888; }
#conv-list .conv-badge { font-size: 10px; color: #5b8def; border: 1px solid #5b8def;
                         border-radius: 3px; padding: 0 3px; margin-left: 4px; }
```

JS(追加到 `<script>` 尾部;`USER_ID` 换成页面实际账号变量):

```js
async function loadConversations() {
  try {
    const r = await fetch(`/api/conversations?user_id=${encodeURIComponent(USER_ID)}`);
    if (!r.ok) throw new Error(String(r.status));
    renderConversationList(await r.json());
  } catch (e) {
    document.getElementById('sidebar').style.display = 'none';  // 静默降级,不影响聊天
  }
}

function renderConversationList(items) {
  const ul = document.getElementById('conv-list');
  ul.innerHTML = '';
  for (const it of items) {
    const li = document.createElement('li');
    li.className = 'conv-item' + (String(it.id) === String(sessionId) ? ' active' : '');
    li.dataset.sessionId = it.id;
    const preview = document.createElement('div');
    preview.className = 'conv-preview';
    preview.textContent = it.preview || '(空会话)';
    const meta = document.createElement('div');
    meta.className = 'conv-meta';
    meta.textContent = (it.updated_at || '').slice(0, 16).replace('T', ' ');
    if (it.summarized) {
      const badge = document.createElement('span');
      badge.className = 'conv-badge';
      badge.textContent = '已摘要';
      meta.appendChild(badge);
    }
    li.appendChild(preview);
    li.appendChild(meta);
    li.addEventListener('click', () => switchConversation(it.id));
    ul.appendChild(li);
  }
}

async function switchConversation(id) {
  if (String(id) === String(sessionId)) return;
  setComposerEnabled(false);                       // 回放完成前锁发送区
  try {
    const r = await fetch(`/api/conversations/${id}/messages?user_id=${encodeURIComponent(USER_ID)}`);
    if (!r.ok) throw new Error(String(r.status));
    const msgs = await r.json();
    clearMessageList();
    for (const m of msgs) appendHistoryBubble(m.role, m.content);  // 只建 user/assistant 气泡
    sessionId = id;
    renderConversationListActive(id);
  } catch (e) {
    showToast('历史加载失败,请重试');              // 轻量错误,恢复当前会话 UI
  } finally {
    setComposerEnabled(true);
  }
}

function renderConversationListActive(id) {
  document.querySelectorAll('#conv-list .conv-item').forEach(li => {
    li.classList.toggle('active', String(li.dataset.sessionId) === String(id));
  });
}

function onNewChat() {
  sessionId = null;
  clearMessageList();
  renderConversationListActive(null);
}

function onSessionEstablished(id) {
  const sb = document.getElementById('sidebar');
  if (sb.style.display === 'none') return;
  loadConversations();                             // 新会话落库后以 API 数据刷新列表
}
```

接线(在 Step 1 确认的位置改):
- 三个小 helper 若页面已有等价物(`setComposerEnabled`/`clearMessageList`/`showToast`/`appendHistoryBubble`)就复用;没有则按名实现:`clearMessageList()` 清空消息容器;`appendHistoryBubble(role, content)` 按现有气泡 DOM 结构追加只读气泡(不带 tool 徽章/citations/按钮);`setComposerEnabled(b)` 切换输入框与发送按钮 disabled;`showToast(t)` 复用现有提示或 `console.warn` + 临时浮层。
- 「+ 新对话」按钮原 handler 末尾改调 `onNewChat()`(保留原 sessionId 置空/清屏逻辑时去重,只留一份)。
- session 帧处理里 `sessionId = ...` 赋值后,若是此前为 null(新会话)则调 `onSessionEstablished(sessionId)`。
- `DOMContentLoaded`(或初始化段)调 `loadConversations()`。

- [ ] **Step 5: 跑绿** — `uv run pytest tests/test_chat_page.py -v` 全绿。

- [ ] **Step 6: Commit** — `git commit -am "feat(ch07): 聊天页会话侧栏(列表/切换回载/静默降级)"`

---

### Task 14: probe 脚本与标注样例(烧额度,手跑不进 pytest)

**Files:**
- Create: `evals/summary_cases.jsonl`、`evals/probe_summary.py`、`evals/probe_context.py`

**Interfaces:**
- Consumes:`SUMMARY_PROMPT`(Task 7)、运行中服务(Task 11 后的完整应用)。
- Produces:摘要质量与 20+ 轮长跑的人工验证工具;不进 pytest 收集(文件名 `probe_` 前缀,现有 pytest 配置不收集 evals/)。

- [ ] **Step 1: 标注样例** — 写 `evals/summary_cases.jsonl`(每行一个 case):

```json
{"name": "订单退款", "dialogue": ["用户:你好", "助手:您好,我是客服小蜜,有什么可以帮您?", "用户:订单A1001什么时候发货", "助手:订单A1001已于3月5日发货。", "用户:我要退款", "助手:可以申请退款,请说明原因。", "用户:手机屏幕碎了,还没解决"], "must_include": ["A1001", "退款"], "must_not_include": ["你好", "客套"], "max_len": 200}
{"name": "手机号与未决诉求", "dialogue": ["用户:我手机号13800138000,帮我查下物流", "助手:好的,已为您查询。", "用户:快递三天没动了,我要投诉加急", "助手:已为您记录。"], "must_include": ["13800138000"], "must_not_include": ["好的"], "max_len": 200}
{"name": "纯闲聊", "dialogue": ["用户:今天天气真不错", "助手:是呀,适合出门。", "用户:哈哈是啊"], "must_include": [], "must_not_include": ["天气"], "max_len": 60}
```

- [ ] **Step 2: probe_summary.py** — 先 Read `evals/probe_intent.py` 对齐真实模型客户端构造,然后写:

```python
"""摘要 prompt 标注样例验证(烧额度,手跑):uv run python evals/probe_summary.py"""
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Settings                      # noqa: E402
from app.prompts.summary import SUMMARY_PROMPT        # noqa: E402


def main() -> int:
    from langchain_openai import ChatOpenAI
    s = Settings()
    model = ChatOpenAI(model=s.model_name, api_key=s.openai_api_key,
                       base_url=s.openai_base_url, max_tokens=400)
    cases = [json.loads(x) for x in
             Path(__file__).with_name("summary_cases.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    failures = 0
    for c in cases:
        prompt = (SUMMARY_PROMPT.replace("{background}", "(无)")
                  .replace("{transcript}", "\n".join(c["dialogue"]))
                  .replace("{max_chars}", str(s.summary_max_chars)))
        out = model.invoke([HumanMessage(content=prompt)]).content.strip()
        ok = (all(k in out for k in c["must_include"])
              and all(k not in out for k in c["must_not_include"])
              and len(out) <= c["max_len"])
        print(f"[{'PASS' if ok else 'FAIL'}] {c['name']}: {out}")
        failures += 0 if ok else 1
    print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: probe_context.py** — 先 Read `evals/probe_intent.py` 看既有 probe 的传输方式;若无 HTTP SSE 先例,按下面写(httpx,`trust_env=False` 防代理红线):

```python
"""20+ 轮上下文长跑(烧额度,手跑):
  默认窗口: uv run python evals/probe_context.py --scenario default
  演示配置: 先用演示 env 起服,再 uv run python evals/probe_context.py --scenario demo
起服(演示): MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
  MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
  no_proxy=127.0.0.1,localhost uv run uvicorn --factory app.main:create_app
"""
import argparse
import re
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
USER = "00000000-0000-0000-0000-000000000001"   # 与 chat.html 演示账号一致

SCRIPT = (["我的订单A1001说可以申请退款,帮我看看进度", "退款要多久到账", "好的先这样"]
          + [f"顺便问下保温杯的第{i}个问题:保温多久" for i in range(1, 19)]
          + ["最开始那个订单后来怎么说"])


def chat(client, sid, text):
    with client.stream("POST", f"{BASE}/v1/chat/stream",
                       json={"user_id": USER, "session_id": sid, "message": text},
                       timeout=120) as r:
        sid_out, answer = sid, []
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            import json
            ev = json.loads(payload)
            if ev.get("type") == "session":
                sid_out = ev["session_id"]
            elif ev.get("type") == "delta":
                answer.append(ev["content"])
        return sid_out, "".join(answer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["default", "demo"], required=True)
    ap.add_argument("--log", default="log/app.log")
    args = ap.parse_args()
    log_start = Path(args.log).stat().st_size if Path(args.log).exists() else 0
    sid, t0 = None, time.monotonic()
    with httpx.Client(trust_env=False) as client:   # 代理红线:本机不走代理
        for i, q in enumerate(SCRIPT, 1):
            sid, ans = chat(client, sid, q)
            print(f"[{i:02d}] {q[:24]}… -> {ans[:60]}")
    elapsed = time.monotonic() - t0
    tail = Path(args.log).read_text(encoding="utf-8")[log_start:]
    degraded = "层1 降级" in tail
    triggered = "summary trigger" in tail
    done = "summary done" in tail
    print(f"elapsed={elapsed:.1f}s degraded={degraded} trigger={triggered} done={done}")
    if args.scenario == "default":
        assert not degraded and not triggered and not done, "默认窗口 20 轮不该触发降级/摘要"
    else:
        assert degraded and triggered and done, "演示配置应出现完整级联"
    print("末轮回答(人工核对是否答对早期订单):", ans)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 手动执行(记 dev-notes,不进 pytest)** —
  1. `docker compose up -d`;老库执行 `mysql wayhelp < sql/ch07-ddl.sql`(若首启已含 05 则跳过)。
  2. `uv run python evals/probe_summary.py` → 3/3 PASS(不达标则调 SUMMARY_PROMPT 重跑,记录轮次)。
  3. `uv run python evals/probe_intent.py` → 预期 24/24(INTENT_PROMPT 加了 history_block,校准回归;若掉点,调历史块文案重跑)。
  4. 默认配置起服跑 `probe_context.py --scenario default` → 无降级无摘要;演示配置起服跑 `--scenario demo` → 级联全出现,末轮人工核对答对 A1001 退款诉求。
  5. `grep -c "model_ctx\|history_ctx" log/app.log` 确认每轮可见;掐表末轮 SSE 时长确认摘要未阻塞。

- [ ] **Step 5: Commit** — `git commit -am "test(ch07): 摘要标注样例与 20+ 轮上下文长跑 probe"`

---

### Task 15: 收尾文档与全量回归

**Files:**
- Modify: `AGENTS.md`、`README.md`
- Modify: `dev-notes/ch07.md`(finish 段,由我本人补,不由实施者写)

- [ ] **Step 1: AGENTS.md 更新** — 红线与坑追加:tool 结果不落 messages 表(只有 user/assistant 行);上下文唯一估算尺 `token_budget.py`;锚点 CAS 单调前移;`log/app.log`(FileHandler 幂等);新配置项一组;「当前状态」加 ch07 完成段(finish 时补)。

- [ ] **Step 2: README 更新** — 运行节加:老库升级 `mysql wayhelp < sql/ch07-ddl.sql`(docker 首启自动含);页面能力加「会话侧栏(多会话切换/历史回载)」;日志节加 `log/app.log` 的 model_ctx/history_ctx 说明。

- [ ] **Step 3: 全量回归** — `docker compose up -d && uv run pytest`,530+ 全绿;失败逐项修,不许跳。

- [ ] **Step 4: Commit** — `git commit -am "docs(ch07): 上下文管理章节收尾(AGENTS/README)"`
