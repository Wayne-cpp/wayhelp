# ch08 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把工具层从写死的内置工具升级为即插即用工具系统:统一 ToolCatalog(内置装饰器注册 + MCP 每轮发现)、统一执行引擎(JSON Schema 校验/权限闸/归属把门/超时重试/格式化)、tool_audit_logs 全量审计、两个独立进程业务 MCP Server(物流/售后)、create_ticket 带 LangGraph interrupt 前端预览确认流。

**Architecture:** 进程级 ToolCatalog 挂 app.state;内置工具在 app/tools/builtin/ 下装饰器自登记、启动扫描;MCP 工具每轮经 MultiServerMCPClient 按 Server 独立超时现问现拿,只放行「零参数 / 仅必填字符串 order_id」两种形态;所有调用同走 ToolExecutor 六步管线并落审计;写操作(create_ticket)由 agent_node 检出后 park 成 pending_ticket,经 ticket_confirm 节点 interrupt → 前端预览卡 → resume confirm/cancel → execute_confirmed(幂等表 + 后台任务集合)放行或记权限拒绝。

**Tech Stack:** FastAPI + LangGraph(现有图)、MCP 官方 Python SDK v1(FastMCP,streamable-http)、langchain-mcp-adapters(MultiServerMCPClient)、SQLAlchemy 同步 + asyncio.to_thread、pydantic v2、pytest(裸 async,asyncio_mode=auto)。

**Spec:** docs/superpowers/specs/2026-10-07-ch08-tool-system-design.md(修订稿,论证以它为准)

## Global Constraints

- MCP SDK 锁 `mcp>=1,<2`,只用 v1 `from mcp.server.fastmcp import FastMCP`;不得混用 v2 `mcp.server.mcpserver.MCPServer`。Server 启动参数(host/port 在构造还是 run())与 adapter transport 字面量(`"streamable_http"`)以**安装包类型定义为准**;Task 7/8 的 list-tools 冒烟测试就是钉。
- `sql/ch08-ddl.sql` 与 `db/init/06-ddl.sql` 必须逐字节一致(test_ddl_sync 钉);`tests/dbfixtures.py` 的 DDL_PATHS/TABLES 同步。
- 审计写失败不许拦工具执行;审计表与幂等表均不挂外键。
- 一切 token 估算只用 `app/services/token_budget.py` 的尺;不得另立字符数/条数启发式。
- messages 表只收 user/assistant 行(validate_turn 口径不变);ToolMessage 只进 checkpoint/模型上下文,不落库。
- 归属错误一律 404 不泄露;resume 卡片失效/字段不对 409;LangGraph resume 传 `Command` 不传 dict。
- 会话内归属校验口径不变:订单不存在与非本人均返回「订单不存在或不属于当前用户」。
- 本机访问服务一律 `curl --noproxy '*'`;起 uvicorn 前加 `no_proxy=127.0.0.1,localhost`。
- Conventional Commits;每任务提交不混入 dev-notes;dev-notes 由 master 单独提交。
- 全量 pytest(现 576 条)必须绿;需要 Docker MySQL 在线。
- 改 prompt 后查 `tests/test_prompts.py` 契约断言同步;预算钉 `tests/test_token_budget.py:48-49` 若红按新规重算并同步 AGENTS.md 数字。

## File Structure

**新建:**
- `app/tools/catalog.py` — ToolSpec / ToolCatalog / TurnContext / WriteMeta / classify_mcp_tool / mcp_args_model / canonical_args_sha / idempotency_key_for
- `app/tools/builtin/__init__.py` — `builtin_tool` 装饰器 + `scan_builtin_specs()`
- `app/tools/builtin/business.py` — query_order_fn / query_product_fn
- `app/tools/builtin/faq.py` — query_faq_fn + FaqRetrievalTrace(自 business.py 迁入)
- `app/tools/builtin/ticket.py` — create_ticket_fn(含幂等行同事务)
- `app/services/tool_audit.py` — write_audit / AuditEntry / 五状态常量
- `app/services/mcp_gateway.py` — McpGateway(每轮按 Server 发现 + call_tool 分发)/ McpToolError
- `app/mcp_servers/{__init__,mock_data,logistics_server,after_sales_server}.py` — 两个独立进程 Server + 种子化 mock
- `tests/test_tool_audit.py` / `test_catalog.py` / `test_builtin_tools.py` / `test_mcp_servers.py` / `test_mcp_gateway.py` / `test_ticket_confirm.py`

**重写:**
- `app/tools/executor.py` — ToolFace / ToolExecutor(六步管线 + execute_confirmed)/ PendingWrite / batch_violation / PENDING_WRITE_TASKS
- `tests/test_executor.py` — 新 API 全套夹具

**修改:**
- `sql/ch08-ddl.sql`(+幂等表)、`db/init/06-ddl.sql`(新)、`tests/test_ddl_sync.py`、`tests/dbfixtures.py`
- `app/models.py`(+ToolAuditLog/ToolWriteIdempotency)、`app/db.py`(+check_ch08_tables)、`app/main.py`(启动接线 + lifespan 收尾)
- `app/config.py`(+7 字段)、`.env.example`
- `app/graph/state.py`(+pending_ticket)、`app/graph/nodes.py`(+build_ticket_confirm_node/route_after_agent + GraphDeps 三字段)、`app/graph/builder.py`(接线)、`app/graph/agent_node.py`(目录装配/批次契约/write 检出)
- `app/services/chat_service.py`(+TicketPreviewEvent/prepare_resume 分派/_pending_selector_events 分支)、`app/schemas.py`(ChatResumeRequest)、`app/routers/chat.py`(+帧/resume 传参)
- `app/prompts/service.py`(条款 7 改写 + create_ticket 条款)、`app/static/chat.html`(预览卡,Vibe Coding)、`tests/test_chat_page.py`(needles)
- `app/tools/business.py`(删 build_graph_tools/GraphToolset,FaqRetrievalTrace 迁出;MOCK_TOOLS/build_tools/write_ticket 保留不动)
- `pyproject.toml`(+mcp、langchain-mcp-adapters)、`Makefile`/`README.md`/`AGENTS.md`

---

### Task 1: DDL 双表 + ORM + 启动探针 + fixtures 接线

**Files:**
- Modify: `sql/ch08-ddl.sql`(在现有 tool_audit_logs 之后追加幂等表)
- Create: `db/init/06-ddl.sql`
- Modify: `tests/test_ddl_sync.py:8`、`tests/dbfixtures.py:15-24`
- Modify: `app/models.py`(文件尾追加两个 ORM 类)
- Modify: `app/db.py`(尾追加 check_ch08_tables)、`app/main.py:69-70`(启动链挂探针)
- Test: `tests/test_ch08_ddl.py`(新建)

**Interfaces:**
- Consumes: 现有 `make_engine` / `check_ch04_tables` 同构模式。
- Produces: `ToolAuditLog` / `ToolWriteIdempotency` ORM(Task 3/6 用);`check_ch08_tables(engine) -> None`(main.py 启动链用);测试库含 `tool_audit_logs`、`tool_write_idempotency` 两表(Task 3/6/11 的 DB 测试用)。

- [ ] **Step 1: 写失败测试**

`tests/test_ch08_ddl.py`:
```python
"""ch08 DDL 钉:两表存在、ORM 可写可读、启动探针缺表即炸。"""
import pytest
from sqlalchemy import text

from app.db import check_ch08_tables
from app.models import ToolAuditLog, ToolWriteIdempotency
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_tables_exist(db_engine):
    with db_engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SHOW TABLES"))}
    assert {"tool_audit_logs", "tool_write_idempotency"} <= tables


def test_audit_row_roundtrip(db_session_factory):
    with db_session_factory() as s:
        s.add(ToolAuditLog(
            conversation_id=None, tool_call_id="c1", tool_name="query_order",
            tool_source="builtin", mcp_server=None, arguments={"order_id": "X"},
            result_summary="{}", status="成功", error_message=None,
            retry_count=0, duration_ms=3))
        s.commit()
        row = s.query(ToolAuditLog).filter_by(tool_call_id="c1").one()
        assert row.status == "成功" and row.tool_source == "builtin"


def test_idempotency_row_roundtrip(db_session_factory):
    with db_session_factory() as s:
        s.add(ToolWriteIdempotency(idempotency_key="k" * 64,
                                   arguments_sha256="a" * 64, ticket_no="T1"))
        s.commit()
        row = s.get(ToolWriteIdempotency, "k" * 64)
        assert row.ticket_no == "T1"


def test_startup_probe_raises_when_missing(db_engine):
    with db_engine.connect() as conn:  # 两表都删:重建脚本整份重放才不会 1050
        conn.execute(text("DROP TABLE IF EXISTS tool_write_idempotency"))
        conn.execute(text("DROP TABLE IF EXISTS tool_audit_logs"))
        conn.commit()
    try:
        with pytest.raises(RuntimeError, match="ch08"):
            check_ch08_tables(db_engine)
    finally:
        from tests.dbfixtures import _split_statements, DDL_PATHS
        with db_engine.connect() as c2:
            for stmt in _split_statements(DDL_PATHS[-1].read_text(encoding="utf-8")):
                c2.execute(text(stmt))
            c2.commit()
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_ch08_ddl.py -x -q`
Expected: FAIL(check_ch08_tables 不存在 / 表不存在)

- [ ] **Step 3: 实现**

`sql/ch08-ddl.sql` 在 tool_audit_logs 语句后追加(保持原有内容逐字不动):
```sql

-- 写操作幂等:写超时「已发未必未成」的收敛机制;与 tickets 插入同一事务
-- key = SHA-256(conversation_id + ":" + tool_call_id);不存 description 原文,不挂外键
CREATE TABLE tool_write_idempotency (
  idempotency_key CHAR(64)     NOT NULL                COMMENT '幂等键 SHA-256 hex',
  arguments_sha256 CHAR(64)    NOT NULL                COMMENT '规范化 JSON 参数摘要',
  ticket_no        VARCHAR(32) NOT NULL                COMMENT '已创建工单号',
  created_at       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (idempotency_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='写操作幂等收敛';
```
注意:`tests/dbfixtures.py:_split_statements` 按 `;\n` 切句,文件内不得出现句内分号;两条 CREATE 都以 `;\n` 收尾。

`db/init/06-ddl.sql`:`cp sql/ch08-ddl.sql db/init/06-ddl.sql`,`cmp` 验证一致。

`tests/test_ddl_sync.py:8`:
```python
DDL_PAIRS = [("01", "ch02"), ("03", "ch03"), ("04", "ch04"), ("05", "ch07"), ("06", "ch08")]
```

`tests/dbfixtures.py`:
```python
DDL_PATHS = [
    Path(__file__).resolve().parent.parent / "db" / "init" / "01-ddl.sql",
    Path(__file__).resolve().parent.parent / "db" / "init" / "03-ddl.sql",
    Path(__file__).resolve().parent.parent / "db" / "init" / "04-ddl.sql",
    Path(__file__).resolve().parent.parent / "db" / "init" / "05-ddl.sql",
    Path(__file__).resolve().parent.parent / "db" / "init" / "06-ddl.sql",
]
TABLES = ("tool_audit_logs", "tool_write_idempotency",
          "faith_cases", "low_confidence_questions", "knowledge_chunks",
          "qa_extraction_staging", "qa_mining_progress",
          "messages", "conversation_summaries", "tickets", "faq",
          "conversations")  # 先子后父
```

`app/models.py` 尾追加:
```python
class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp", name="tool_source"))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"))
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(TINYINT(unsigned=True), default=0)
    duration_ms: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolWriteIdempotency(Base):
    __tablename__ = "tool_write_idempotency"

    idempotency_key: Mapped[str] = mapped_column(CHAR(64), primary_key=True)
    arguments_sha256: Mapped[str] = mapped_column(CHAR(64))
    ticket_no: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```
(ORM 类型说明:`retry_count` 用 `TINYINT(unsigned=True)`——`from sqlalchemy.dialects.mysql` 的 import 在现有 `BIGINT, INTEGER` 后追加 `TINYINT`;`idempotency_key`/`arguments_sha256` 用 `String(64)`——DDL 是 CHAR(64),ORM String 与 MySQL CHAR 读写兼容,保持 import 面不变,测试只断言行级行为。)

`app/db.py` 尾追加:
```python
_CH08_TABLES = ("tool_audit_logs", "tool_write_idempotency")


def check_ch08_tables(engine) -> None:
    """缺 ch08 表(审计/幂等)启动失败并提示升级命令。"""
    with engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SHOW TABLES"))}
    missing = [t for t in _CH08_TABLES if t not in tables]
    if missing:
        raise RuntimeError(
            f"缺少 ch08 表 {missing}:请先执行 mysql wayhelp < sql/ch08-ddl.sql")
```

`app/main.py:69-70` 在 `check_ch07_tables(engine)` 后加一行 `check_ch08_tables(engine)` 并更新 import:`from app.db import check_ch04_tables, check_ch07_tables, check_ch08_tables, make_engine, make_session_factory, ping`(以文件现状 import 行为准并入)。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_ch08_ddl.py tests/test_ddl_sync.py -q`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add sql/ch08-ddl.sql db/init/06-ddl.sql tests/test_ddl_sync.py tests/dbfixtures.py app/models.py app/db.py app/main.py tests/test_ch08_ddl.py
git commit -m "feat(ch08): tool_audit_logs + tool_write_idempotency DDL wiring"
```

---

### Task 2: config 新字段 + MCP 依赖 + .env.example

**Files:**
- Modify: `app/config.py:66`(refund_expand_enabled 之后追加)
- Modify: `.env.example`(尾追加注释块)
- Modify: `pyproject.toml`(dependencies 追加两包)
- Test: `tests/test_config.py`(追加用例)

**Interfaces:**
- Produces: `Settings.mcp_logistics_url: str` / `mcp_after_sales_url: str`(空串 = 不配置)/ `mcp_discovery_timeout_seconds: float` / `tool_write_timeout_seconds: float` / `tool_timeout_overrides: dict[str, float]` / `audit_result_max_chars: int`。Task 6/8/12 全量消费。

- [ ] **Step 1: 写失败测试**

`tests/test_config.py` 追加:
```python
def test_ch08_defaults():
    s = Settings(_env_file=None, openai_base_url="http://t", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d")
    assert s.mcp_logistics_url == "" and s.mcp_after_sales_url == ""
    assert s.mcp_discovery_timeout_seconds == 2
    assert s.tool_write_timeout_seconds == 10
    assert s.tool_timeout_overrides == {}
    assert s.audit_result_max_chars == 2000


def test_tool_timeout_overrides_parses_json():
    s = Settings(_env_file=None, openai_base_url="http://t", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d",
                 tool_timeout_overrides='{"query_product": 0.001}')
    assert s.tool_timeout_overrides == {"query_product": 0.001}


def test_tool_timeout_overrides_rejects_bad_json():
    import pytest
    with pytest.raises(Exception):
        Settings(_env_file=None, openai_base_url="http://t", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d",
                 tool_timeout_overrides="{not json")
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_config.py -q`
Expected: 3 条 FAIL(字段不存在)

- [ ] **Step 3: 实现**

`app/config.py` 在 `refund_expand_enabled` 行后追加:
```python
    # ch08 工具系统(spec §6.1)
    mcp_logistics_url: str = ""          # 例 http://127.0.0.1:8101/mcp;空 = 不接入
    mcp_after_sales_url: str = ""        # 例 http://127.0.0.1:8102/mcp;空 = 不接入
    mcp_discovery_timeout_seconds: float = Field(default=2, gt=0)  # 每轮按 Server 发现超时,超时即跳过
    tool_write_timeout_seconds: float = Field(default=10, gt=0)    # 写路径超时(超时≠没执行,绝不重试)
    tool_timeout_overrides: dict[str, float] = Field(default_factory=dict)  # JSON 按名覆盖等待期限(运维 knob/故障注入)
    audit_result_max_chars: int = Field(default=2000, gt=0)        # 审计 result_summary 截断

    @field_validator("tool_timeout_overrides", mode="before")
    @classmethod
    def _parse_overrides(cls, v):
        if isinstance(v, str):
            import json as _json
            raw = _json.loads(v or "{}")
            if not isinstance(raw, dict):
                raise ValueError("TOOL_TIMEOUT_OVERRIDES 必须是 JSON 对象")
            return {str(k): float(x) for k, x in raw.items()}
        return v
```

`.env.example` 尾追加:
```bash
# ── ch08 工具系统 ──
# 两个业务 MCP Server 地址(空 = 不接入;起 server 见 README「MCP Server」节)
# MCP_LOGISTICS_URL=http://127.0.0.1:8101/mcp
# MCP_AFTER_SALES_URL=http://127.0.0.1:8102/mcp
# 每轮工具发现单 Server 超时(秒),超时跳过该 Server 不拖聊天
# MCP_DISCOVERY_TIMEOUT_SECONDS=2
# 写操作等待期限(秒);超时按「已发未必未成」处理且绝不重试
# TOOL_WRITE_TIMEOUT_SECONDS=10
# 按工具名覆盖等待期限(JSON 对象;运维 knob,验收 6 故障注入也用它)
# TOOL_TIMEOUT_OVERRIDES={"query_product":0.001}
# 审计 result_summary 截断字符数
# AUDIT_RESULT_MAX_CHARS=2000
```

`pyproject.toml` dependencies 追加 `"mcp>=1,<2"`、`"langchain-mcp-adapters>=0.1"`,然后 `uv sync` 更新锁。验证:`uv run python -c "from mcp.server.fastmcp import FastMCP; from langchain_mcp_adapters.client import MultiServerMCPClient; import mcp; print(mcp.__doc__)"`。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_config.py -q`
Expected: 全 PASS(含存量用例)

- [ ] **Step 5: Commit**

```bash
git add app/config.py .env.example pyproject.toml uv.lock tests/test_config.py
git commit -m "feat(ch08): settings for MCP/audit/write-timeout + mcp deps"
```

---

### Task 3: 审计服务 app/services/tool_audit.py

**Files:**
- Create: `app/services/tool_audit.py`
- Test: `tests/test_tool_audit.py`

**Interfaces:**
- Consumes: Task 1 的 `ToolAuditLog` ORM、`db_session_factory` fixture。
- Produces:
  - 常量 `AUDIT_SUCCESS/AUDIT_FAILURE/AUDIT_TIMEOUT/AUDIT_INVALID/AUDIT_DENIED`(中文五态)
  - `AuditEntry`(frozen dataclass,字段与 ORM 一一对应)
  - `async write_audit(session_factory, entry: AuditEntry, max_chars: int = 2000) -> None` — shield+to_thread,失败只打 error 日志不抛;result_summary 超 max_chars 截断加 `…[已截断]`。Task 6 引擎唯一审计出口。

- [ ] **Step 1: 写失败测试**

`tests/test_tool_audit.py`:
```python
import logging

from app.services.tool_audit import (
    AUDIT_DENIED, AUDIT_FAILURE, AUDIT_INVALID, AUDIT_SUCCESS, AUDIT_TIMEOUT,
    AuditEntry, write_audit,
)
from app.models import ToolAuditLog
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _entry(**over):
    base = dict(conversation_id=1, tool_call_id="c1", tool_name="query_order",
                tool_source="builtin", mcp_server=None, arguments={"order_id": "X"},
                result_summary="{}", status=AUDIT_SUCCESS, error_message=None,
                retry_count=0, duration_ms=5)
    base.update(over)
    return AuditEntry(**base)


async def test_all_five_statuses_land(db_session_factory):
    for i, st in enumerate((AUDIT_SUCCESS, AUDIT_FAILURE, AUDIT_TIMEOUT,
                            AUDIT_INVALID, AUDIT_DENIED)):
        await write_audit(db_session_factory, _entry(tool_call_id=f"c{i}", status=st))
    with db_session_factory() as s:
        got = {r.tool_call_id: r.status for r in s.query(ToolAuditLog).all()}
    assert got == {f"c{i}": st for i, st in enumerate(
        (AUDIT_SUCCESS, AUDIT_FAILURE, AUDIT_TIMEOUT, AUDIT_INVALID, AUDIT_DENIED))}


async def test_result_summary_truncated(db_session_factory):
    await write_audit(db_session_factory, _entry(result_summary="汉" * 5000), max_chars=100)
    with db_session_factory() as s:
        row = s.query(ToolAuditLog).filter_by(tool_call_id="c1").one()
    assert len(row.result_summary) < 5000
    assert row.result_summary.endswith("…[已截断]")


async def test_audit_failure_never_raises(db_session_factory, caplog):
    class BoomFactory:  # 构造即炸,模拟审计库不可用
        def __call__(self):
            raise RuntimeError("db down")
    with caplog.at_level(logging.ERROR):
        await write_audit(BoomFactory(), _entry())  # 不抛
    assert any("tool_audit" in r.message for r in caplog.records)


async def test_null_conversation_allowed(db_session_factory):
    await write_audit(db_session_factory, _entry(conversation_id=None))
    with db_session_factory() as s:
        row = s.query(ToolAuditLog).filter_by(tool_call_id="c1").one()
    assert row.conversation_id is None
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_tool_audit.py -q`
Expected: ImportError / FAIL

- [ ] **Step 3: 实现**

`app/services/tool_audit.py`:
```python
"""ch08 工具调用审计(spec §3):每次调用终结落一条;写失败不拦工具执行(红线)。"""
import asyncio
import contextlib
import logging
from dataclasses import dataclass

logger = logging.getLogger("wayhelp.tool_audit")

AUDIT_SUCCESS = "成功"
AUDIT_FAILURE = "失败"
AUDIT_TIMEOUT = "超时"
AUDIT_INVALID = "校验拦下"
AUDIT_DENIED = "权限拒绝"

_TRUNC_MARK = "…[已截断]"


@dataclass(frozen=True)
class AuditEntry:
    conversation_id: int | None
    tool_call_id: str | None
    tool_name: str
    tool_source: str          # "builtin" | "mcp"
    mcp_server: str | None
    arguments: dict | None
    result_summary: str | None
    status: str
    error_message: str | None
    retry_count: int
    duration_ms: int | None


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None or len(text) <= max_chars:
        return text
    return text[: max_chars - len(_TRUNC_MARK)] + _TRUNC_MARK


def _write_sync(session_factory, entry: AuditEntry, max_chars: int) -> None:
    from app.models import ToolAuditLog
    with session_factory() as s:
        s.add(ToolAuditLog(
            conversation_id=entry.conversation_id, tool_call_id=entry.tool_call_id,
            tool_name=entry.tool_name, tool_source=entry.tool_source,
            mcp_server=entry.mcp_server, arguments=entry.arguments,
            result_summary=_truncate(entry.result_summary, max_chars),
            status=entry.status, error_message=entry.error_message,
            retry_count=entry.retry_count, duration_ms=entry.duration_ms))
        s.commit()


async def write_audit(session_factory, entry: AuditEntry, max_chars: int = 2000) -> None:
    """shield 保护审计写;任何失败只打 error 日志——审计不得反过来拦工具执行。"""
    if session_factory is None:
        logger.error("tool_audit 无 session_factory,丢弃审计: %s %s", entry.tool_name, entry.status)
        return
    task = asyncio.ensure_future(asyncio.to_thread(_write_sync, session_factory, entry, max_chars))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise
    except Exception as exc:
        logger.error("tool_audit 写失败(不影响工具执行): %s", type(exc).__name__)
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_tool_audit.py -q`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/tool_audit.py tests/test_tool_audit.py
git commit -m "feat(ch08): tool audit service with five terminal statuses"
```

---

### Task 4: ToolCatalog + TurnContext + MCP 能力策略

**Files:**
- Create: `app/tools/catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: 无(纯数据与策略)。
- Produces(Task 5/6/8/9/10 全部依赖,签名逐字钉):
  - `ToolSpec`(frozen):`name: str`、`description: str`、`args_schema: dict`(JSON Schema dict)、`args_model: type[BaseModel]`、`source: str`("builtin"|"mcp")、`permission: str`("read"|"write"|"deny")、`ownership: str = "none"`("none"|"order")、`mcp_server: str | None = None`、`budget_priority: int = 0`、`fn: Callable | None = None`(builtin 本地函数,签名 `(args: dict, ctx: TurnContext) -> str`;写函数多收第三参 `write: WriteMeta`)、`formatter: Callable[[str], str] | None = None`(MCP 结果挑字段后返回 JSON str)
  - `WriteMeta`(frozen):`key: str`、`args_sha256: str`
  - `TurnContext`(mutable dataclass):`user_id: str`、`conversation_id: int | None`、`resolved_query: str`、`retriever: Any`、`settings: Any`、`session_factory: Any = None`、`faq_trace: Any = None`、`order_snapshots: list = field(default_factory=list)`
  - `ToolCatalog`:`register(spec) -> None`(重名 ValueError)、`get(name) -> ToolSpec | None`、`specs() -> list[ToolSpec]`
  - `classify_mcp_tool(server: str, schema: dict) -> tuple[str, str] | None`:本地能力策略(spec §1.1);已知只读 Server = `{"logistics", "after_sales"}`;schema 形态仅两种放行——零参数 → `("read","none")`;properties 恰为 `{"order_id": {"type":"string"}}` 且 required 恰为 `["order_id"]` → `("read","order")`;其他(未知 Server/含写意味参数/形态不符)→ None
  - `mcp_args_model(shape: str) -> type[BaseModel]`:"order" → 单字段 order_id 模型;"none" → 空模型
  - `canonical_args_sha(args: dict) -> str`:规范化 JSON(sort_keys、无空白、UTF-8 不转义)的 SHA-256 hex
  - `idempotency_key_for(conversation_id: int, tool_call_id: str) -> str`:SHA-256(`f"{cid}:{tcid}"`)hex

- [ ] **Step 1: 写失败测试**

`tests/test_catalog.py`:
```python
import pytest
from pydantic import BaseModel, Field

from app.tools.catalog import (
    ToolCatalog, ToolSpec, TurnContext, canonical_args_sha, classify_mcp_tool,
    idempotency_key_for, mcp_args_model,
)


class _Args(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)


def _spec(name="t1", **over):
    base = dict(name=name, description="d", args_schema=_Args.model_json_schema(),
                args_model=_Args, source="builtin", permission="read")
    base.update(over)
    return ToolSpec(**base)


def test_catalog_register_and_duplicate_reject():
    c = ToolCatalog()
    c.register(_spec("a"))
    assert c.get("a").name == "a"
    assert [s.name for s in c.specs()] == ["a"]
    with pytest.raises(ValueError):
        c.register(_spec("a"))
    assert c.get("ghost") is None


def test_turncontext_defaults():
    ctx = TurnContext(user_id="u", conversation_id=None, resolved_query="q",
                      retriever=None, settings=None)
    assert ctx.order_snapshots == [] and ctx.session_factory is None


def test_classify_mcp_tool_two_shapes_only():
    assert classify_mcp_tool("logistics", {"type": "object", "properties": {},
                                           "required": []}) == ("read", "none")
    order_schema = {"type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}
    assert classify_mcp_tool("after_sales", order_schema) == ("read", "order")
    # 未知 Server / 额外参数 / 写意味参数 / order_id 非必填 → 拒
    assert classify_mcp_tool("warehouse", order_schema) is None
    assert classify_mcp_tool("logistics", {"type": "object", "properties": {
        "order_id": {"type": "string"}, "note": {"type": "string"}},
        "required": ["order_id"]}) is None
    assert classify_mcp_tool("logistics", {"type": "object", "properties": {
        "order_id": {"type": "string"}}, "required": []}) is None


def test_mcp_args_model_shapes():
    m = mcp_args_model("order")
    assert m(order_id="X").order_id == "X"
    with pytest.raises(Exception):
        m()
    empty = mcp_args_model("none")
    assert empty().model_dump() == {}


def test_canonical_args_sha_stable_and_key():
    a = canonical_args_sha({"b": 1, "a": "汉"})
    b = canonical_args_sha({"a": "汉", "b": 1})
    assert a == b and len(a) == 64
    assert idempotency_key_for(7, "call-1") == idempotency_key_for(7, "call-1")
    assert idempotency_key_for(7, "call-1") != idempotency_key_for(8, "call-1")
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: ImportError

- [ ] **Step 3: 实现**

`app/tools/catalog.py`:
```python
"""ch08 工具目录(spec §1):进程内唯一工具表;内置启动登记,MCP 每轮发现现合成。
权限/归属分类只认本地策略,Server 自报 annotations 一律不看。"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

Permission = Literal["read", "write", "deny"]
Ownership = Literal["none", "order"]
Source = Literal["builtin", "mcp"]


@dataclass(frozen=True)
class WriteMeta:
    """execute_confirmed 传给写函数的幂等元数据(spec §3)。"""
    key: str
    args_sha256: str


@dataclass
class TurnContext:
    """每轮显式上下文:替代闭包绑定;conversation_id 只取已确认的 MySQL 会话 id。"""
    user_id: str
    conversation_id: int | None
    resolved_query: str
    retriever: Any
    settings: Any
    session_factory: Any = None
    faq_trace: Any = None            # FaqRetrievalTrace(business 分支装配时挂)
    order_snapshots: list = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict                 # JSON Schema dict(模型可见 + 校验依据)
    args_model: type[BaseModel]       # 校验/StructuredTool 载体
    source: Source
    permission: Permission
    ownership: Ownership = "none"
    mcp_server: str | None = None
    budget_priority: int = 0          # 超预算剔除次序;数值越大越优先保留
    fn: Callable | None = None        # builtin:(args, ctx) -> str;写函数多收 write: WriteMeta
    formatter: Callable[[str], str] | None = None  # MCP 结果挑字段 → JSON str


class ToolCatalog:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())


# ── MCP 本地能力策略(spec §1.1/§4.2)──

KNOWN_READONLY_SERVERS = frozenset({"logistics", "after_sales"})


def classify_mcp_tool(server: str, schema: dict) -> tuple[str, str] | None:
    """返回 (permission, ownership);None = 不进工具面。只放行两种形态:
    零参数 → (read, none);仅必填字符串 order_id → (read, order)。"""
    if server not in KNOWN_READONLY_SERVERS:
        return None
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    if not props and not required:
        return ("read", "none")
    if set(props) == {"order_id"} and required == {"order_id"} \
            and (props["order_id"] or {}).get("type") == "string":
        return ("read", "order")
    return None


def mcp_args_model(shape: str) -> type[BaseModel]:
    if shape == "order":
        class _OrderArgs(BaseModel):
            order_id: str = Field(min_length=1, max_length=64, description="订单号")
        return _OrderArgs

    class _EmptyArgs(BaseModel):
        pass
    return _EmptyArgs


def canonical_args_sha(args: dict) -> str:
    raw = json.dumps(args or {}, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def idempotency_key_for(conversation_id: int, tool_call_id: str) -> str:
    return hashlib.sha256(f"{conversation_id}:{tool_call_id}".encode("utf-8")).hexdigest()
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/catalog.py tests/test_catalog.py
git commit -m "feat(ch08): tool catalog, turn context, MCP capability policy"
```

---

### Task 5: builtin 注册包 + 三工具迁移 + create_ticket 登记

**Files:**
- Create: `app/tools/builtin/__init__.py`、`app/tools/builtin/business.py`、`app/tools/builtin/faq.py`、`app/tools/builtin/ticket.py`
- Test: `tests/test_builtin_tools.py`
- Modify: `app/tools/business.py`(本任务只把 `FaqRetrievalTrace` 标记迁出,`build_graph_tools` 的删除在 Task 9 完成——本任务保证新旧并存、全量测试仍绿)

**Interfaces:**
- Consumes: Task 4 的 ToolSpec/TurnContext/WriteMeta;`app.services.orders.get_order`;`app.knowledge.retriever` 的 NOTE_*/assemble_evidence;Task 1 的 `ToolWriteIdempotency`;`app.tools.business.write_ticket`。
- Produces:
  - `builtin_tool(...)` 装饰器:参数 `name/description/args_model/permission/ownership/budget_priority`,挂 `fn.__tool_spec__`
  - `scan_builtin_specs() -> list[ToolSpec]`:import 包内全部模块,收集 `__tool_spec__`
  - `query_order_fn / query_product_fn / query_faq_fn(args, ctx) -> str`(行为与现闭包版逐字一致:归属落空回 error JSON;快照入 `ctx.order_snapshots`;faq 无参、同轮缓存、NOTE_* 三态、`ctx.faq_trace` 写状态)
  - `FaqRetrievalTrace`(faq.py;自 business.py 迁入,字段不变:business.py 保留 `from app.tools.builtin.faq import FaqRetrievalTrace` 兼容 re-export)
  - `create_ticket_fn(args, ctx, write: WriteMeta) -> str`:tickets + 幂等行同一事务,不改 conv.status;`ctx.conversation_id is None` 或会话不存在 → 抛 ValueError(引擎分诊)

- [ ] **Step 1: 写失败测试**

`tests/test_builtin_tools.py`:
```python
import json

from app.config import Settings
from app.tools.builtin import scan_builtin_specs
from app.tools.builtin.faq import FaqRetrievalTrace
from app.tools.catalog import TurnContext, WriteMeta
from tests.conftest import TEST_USER_ID, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _ctx(settings=None, **over):
    base = dict(user_id=TEST_USER_ID, conversation_id=None, resolved_query="q",
                retriever=None, settings=settings or make_settings())
    base.update(over)
    return TurnContext(**base)


def test_scan_finds_all_builtin():
    specs = {s.name: s for s in scan_builtin_specs()}
    assert {"query_order", "query_product", "query_faq", "create_ticket"} <= set(specs)
    assert "query_logistics" not in specs            # 内置已下线(spec §4.1)
    assert specs["create_ticket"].permission == "write"
    assert specs["query_order"].ownership == "order"
    assert specs["query_faq"].args_schema.get("properties", {}) == {}  # 无参


def test_query_order_bound_to_user():
    from app.tools.builtin.business import query_order_fn
    specs = {s.name: s for s in scan_builtin_specs()}
    ctx = _ctx()
    out = json.loads(query_order_fn({"order_id": "1111-1001"}, ctx))
    assert out["status"] == "已完成" and ctx.order_snapshots[0]["order_id"] == "1111-1001"
    miss = json.loads(query_order_fn({"order_id": "9999-1001"}, ctx))
    assert miss == {"error": "订单不存在或不属于当前用户"}


def test_query_product_mock_shape():
    from app.tools.builtin.business import query_product_fn
    out = json.loads(query_product_fn({"product_name": "保温杯"}, _ctx()))
    assert {"product_name", "price", "stock", "on_sale"} <= set(out)


def test_query_faq_caches_and_traces():
    from app.tools.builtin.faq import query_faq_fn
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    trace = FaqRetrievalTrace()
    ctx = _ctx(retriever=_FakeRetriever(_result()), resolved_query="退货政策",
               faq_trace=trace)
    first = json.loads(query_faq_fn({}, ctx))
    second = json.loads(query_faq_fn({}, ctx))
    assert first == second and trace.calls == 1  # 同轮二次走缓存
    assert trace.status == "ok" and first["evidence"]


def test_create_ticket_writes_ticket_and_idempotency_same_tx(db_session_factory):
    from app.tools.builtin.ticket import create_ticket_fn
    from app.models import Ticket, ToolWriteIdempotency, Conversation
    with db_session_factory() as s:
        s.add(Conversation(id=1, user_id=TEST_USER_ID))
        s.commit()
    ctx = _ctx(conversation_id=1, session_factory=db_session_factory)
    meta = WriteMeta(key="k" * 64, args_sha256="a" * 64)
    out = json.loads(create_ticket_fn(
        {"description": "商品有质量问题", "ticket_type": "售后"}, ctx, meta))
    with db_session_factory() as s:
        t = s.get(Ticket, out["ticket_no"])
        assert t is not None and t.ticket_type == "售后"
        assert s.get(Conversation, 1).status == "进行中"  # 不改 conv.status
        idem = s.get(ToolWriteIdempotency, "k" * 64)
        assert idem.ticket_no == out["ticket_no"]


def test_create_ticket_without_conversation_raises():
    from app.tools.builtin.ticket import create_ticket_fn
    import pytest
    with pytest.raises(ValueError):
        create_ticket_fn({"description": "x", "ticket_type": "售后"},
                         _ctx(), WriteMeta(key="k" * 64, args_sha256="a" * 64))
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_builtin_tools.py -q`
Expected: ImportError(app.tools.builtin 不存在)

- [ ] **Step 3: 实现**

`app/tools/builtin/__init__.py`:
```python
"""ch08 内置工具自登记(spec §1.2):装饰器挂元数据,启动扫描收集。
新工具 = 本包内新文件 + @builtin_tool,核心代码零改动。"""
import importlib
import pkgutil

from app.tools.catalog import ToolSpec

_MODULES = ("business", "faq", "ticket")


def builtin_tool(*, name, description, args_model, permission="read",
                 ownership="none", budget_priority=100):
    def deco(fn):
        fn.__tool_spec__ = ToolSpec(
            name=name, description=description,
            args_schema=args_model.model_json_schema(), args_model=args_model,
            source="builtin", permission=permission, ownership=ownership,
            budget_priority=budget_priority, fn=fn)
        return fn
    return deco


def scan_builtin_specs() -> list[ToolSpec]:
    pkg = __name__
    specs = []
    for mod_name in _MODULES:
        mod = importlib.import_module(f"{pkg}.{mod_name}")
        for attr in vars(mod).values():
            spec = getattr(attr, "__tool_spec__", None)
            if spec is not None:
                specs.append(spec)
    return specs
```

`app/tools/builtin/business.py`:
```python
"""ch08 内置业务只读工具:query_order(归属由引擎把门,内部防御性复核)/ query_product。
query_logistics 已下线,物流查询由 MCP Server 接管(spec §4.1)。"""
import json
import random
from dataclasses import asdict
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from app.services import orders as _orders
from app.tools.builtin import builtin_tool


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


class QueryOrderArgs(BaseModel):
    order_id: str = Field(min_length=1, max_length=64, description="订单号")


@builtin_tool(name="query_order",
              description="查询订单信息。参数 order_id 为订单号。返回订单状态、金额、商品名、下单时间。",
              args_model=QueryOrderArgs, ownership="order", budget_priority=100)
def query_order_fn(args: dict, ctx) -> str:
    o = _orders.get_order(ctx.user_id, args["order_id"])
    if o is None:
        return _json({"error": "订单不存在或不属于当前用户"})
    ctx.order_snapshots.append(asdict(o))
    return _json({"order_id": o.order_id, "status": o.status, "amount": o.amount,
                  "product": o.product, "created_at": o.created_at,
                  "delivered_at": o.delivered_at})


class QueryProductArgs(BaseModel):
    product_name: str = Field(min_length=1, max_length=128, description="商品名称关键词")


@builtin_tool(name="query_product",
              description="查询商品信息。参数 product_name 为商品名称关键词。返回价格、库存、是否在售。",
              args_model=QueryProductArgs, budget_priority=100)
def query_product_fn(args: dict, ctx) -> str:
    return _json({
        "product_name": args["product_name"],
        "price": round(random.uniform(9.9, 1999.0), 2),
        "stock": random.randint(0, 500),
        "on_sale": random.random() > 0.1,
    })
```

`app/tools/builtin/faq.py`(`FaqRetrievalTrace` 自 `app/tools/business.py:182-189` 逐字迁入,`query_faq_fn` 行为逐字对齐 `business.py:254-294` 闭包版,`resolved_query/retriever/settings/faq_trace` 改从 ctx 取):
```python
"""ch08 内置 query_faq:无参(以本轮 resolved_query 检索),同轮缓存,置信闸状态写 ctx.faq_trace。"""
import json
from dataclasses import dataclass

from pydantic import BaseModel

from app.knowledge.retriever import (
    NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED, RetrievalResult,
    assemble_evidence,
)
from app.tools.builtin import builtin_tool


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class FaqRetrievalTrace:
    """图专用 query_faq 的单轮显式 interface;首轮调用写一次,后续走缓存。"""
    calls: int = 0
    status: str = "not_called"   # ok | low_confidence | unavailable | tool_error
    result: RetrievalResult | None = None
    evidence: list[dict] | None = None
    error_code: str | None = None
    _cached: str | None = None


class QueryFaqArgs(BaseModel):
    pass


@builtin_tool(name="query_faq",
              description="检索知识库,获取政策/流程/规格/常见问题资料。无需参数,以用户本轮完整问题检索。返回检索策略、低置信标记与证据列表。同轮重复调用返回同一结果。",
              args_model=QueryFaqArgs, budget_priority=100)
def query_faq_fn(args: dict, ctx) -> str:
    trace = ctx.faq_trace
    settings = ctx.settings
    trace.calls += 1
    if trace.calls > 1 and trace._cached is not None:
        return trace._cached
    if ctx.retriever is None:
        trace.status = "unavailable"
        trace.error_code = "kb_unconfigured"
        trace._cached = _json({"evidence": [], "note": "知识检索未配置",
                               "low_confidence": False, "effective_strategy": None})
        return trace._cached
    result = ctx.retriever.search(ctx.resolved_query, scope=None)
    if result.note in (NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED):
        trace.status = "unavailable"
        trace.error_code = (
            "kb_rebuilding" if result.note == NOTE_REBUILDING
            else "kb_rebuild_required" if result.note == NOTE_REBUILD_REQUIRED
            else "kb_unconfigured")
        trace._cached = _json({"evidence": [], "note": result.note,
                               "low_confidence": False,
                               "effective_strategy": result.effective_strategy})
        return trace._cached
    trace.result = result
    if result.low_confidence:
        trace.status = "low_confidence"
        trace._cached = _json({"evidence": [], "note": result.note,
                               "low_confidence": True,
                               "effective_strategy": result.effective_strategy})
        return trace._cached
    evidence = [e.to_dict() for e in assemble_evidence(
        result.hits, max_items=settings.rerank_top_k,
        budget_chars=settings.max_tool_result_chars, overhead_chars=200)]
    trace.status = "ok"
    trace.evidence = evidence
    trace._cached = _json({"effective_strategy": result.effective_strategy,
                           "low_confidence": False, "note": result.note,
                           "evidence": evidence})
    return trace._cached
```

`app/tools/builtin/ticket.py`:
```python
"""ch08 create_ticket(写):tickets 与幂等行同一事务;不改 conv.status(spec §3/§5)。"""
import json
from typing import Literal

from pydantic import BaseModel, Field

from app.models import Conversation, ToolWriteIdempotency
from app.tools.builtin import builtin_tool
from app.tools.business import write_ticket
from app.tools.catalog import WriteMeta


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


class CreateTicketArgs(BaseModel):
    description: str = Field(min_length=1, max_length=2000, description="问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")


@builtin_tool(name="create_ticket",
              description="创建人工工单。仅在顾客明确要求建工单且工单类型、问题描述已核对齐全时调用;调用后需用户在前端预览确认才真正生效。",
              args_model=CreateTicketArgs, permission="write", budget_priority=50)
def create_ticket_fn(args: dict, ctx, write: WriteMeta) -> str:
    if ctx.conversation_id is None or ctx.session_factory is None:
        raise ValueError("conversation context required")
    with ctx.session_factory() as s:
        conv = s.get(Conversation, ctx.conversation_id)
        if conv is None:
            raise ValueError("conversation not found")
        ticket_no = write_ticket(s, ctx.conversation_id,
                                 args["description"], args["ticket_type"])
        s.add(ToolWriteIdempotency(idempotency_key=write.key,
                                   arguments_sha256=write.args_sha256,
                                   ticket_no=ticket_no))
        s.commit()  # 工单与幂等映射同事务;不改 conv.status
    return _json({"ticket_no": ticket_no, "status": "待处理"})
```

`app/tools/business.py:182` 的 `FaqRetrievalTrace` 类定义替换为兼容 re-export(保留 import 位,防旧引用断):
```python
from app.tools.builtin.faq import FaqRetrievalTrace  # noqa: F401  (已迁至 builtin/faq.py)
```
注意:`app/tools/builtin/faq.py` import `app.tools.builtin`(`__init__` 的 builtin_tool),`business.py` 又被 `builtin/ticket.py` import(write_ticket)——无环:`builtin/__init__` 只 import catalog;`builtin/business|faq|ticket` import `builtin/__init__`;`app/tools/business.py` re-export faq 的类(单向)。扫描时 import 顺序按 _MODULES 元组,不触发循环。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_builtin_tools.py -q && uv run pytest -q -k "not eval" --ignore=evals`
Expected: 新测试 PASS;全量不红(FaqRetrievalTrace 搬迁靠 re-export 兼容)

- [ ] **Step 5: Commit**

```bash
git add app/tools/builtin/ tests/test_builtin_tools.py app/tools/business.py
git commit -m "feat(ch08): builtin tool self-registration, migrate read tools, register create_ticket"
```

---

### Task 6: 执行引擎重写(app/tools/executor.py)

**Files:**
- Rewrite: `app/tools/executor.py`
- Rewrite: `tests/test_executor.py`

**Interfaces:**
- Consumes: Task 3 `write_audit/AuditEntry/五态`;Task 4 `ToolSpec/ToolCatalog/TurnContext/WriteMeta/canonical_args_sha/idempotency_key_for`;Task 5 builtin fns;`app.services.token_budget.truncate_tool_result`;`app.services.orders.get_order`。
- Produces(Task 8/9/10 依赖,签名逐字钉):
  - `ToolFace(specs: list[ToolSpec])`:重名 ValueError;`.get(name) -> ToolSpec | None`;`.specs -> list[ToolSpec]`;`.as_langchain_tools() -> list[StructuredTool]`(schema 载体,供 bind_tools/measure_sys_tokens)
  - `PendingWrite`(frozen):`tool_call_id: str`、`name: str`、`args: dict`、`args_sha256: str`
  - `batch_violation(calls: list[dict], face: ToolFace) -> str | None`:多写/写不在最后 → 原因文案;合法 → None
  - `ToolExecutor(face, settings, session_factory=None, mcp=None)`:
    - `async execute(call: dict, ctx: TurnContext) -> ToolOutcome`
    - `async execute_confirmed(pending: PendingWrite, ctx: TurnContext) -> ToolOutcome`
    - `async deny_write(pending: PendingWrite, ctx: TurnContext, reason: str) -> ToolOutcome`
    - `async mark_invalid_batch(calls: list[dict], ctx: TurnContext, reason: str) -> list[ToolMessage]`
  - `ToolOutcome`:`message: ToolMessage`、`record: ToolExecutionRecord`
  - `ToolExecutionRecord`(frozen):`name/args/ok/duration_ms/retry_count/error_type/error_code/source/mcp_server/audit_status`
  - `PENDING_WRITE_TASKS: set[asyncio.Task]`(进程级强引用集合;main.py lifespan 收尾 drain)
  - error_code 全集:`unknown_tool | invalid_args | invalid_batch | permission_denied | tool_unavailable | tool_error | mcp_error | mcp_unavailable | write_timeout | idempotency_conflict`
- 审计状态映射(spec §3 终态映射):ok → 成功(含归属落空业务空结果);`invalid_args/invalid_batch/写缺会话上下文` → 校验拦下;`permission_denied/用户取消` → 权限拒绝;末次异常为超时的 `tool_unavailable`、`write_timeout` → 超时;其余(`unknown_tool`/非超时的 `tool_unavailable`/`tool_error`/`mcp_error`/`idempotency_conflict`)→ 失败
- 有效超时优先级(spec §2):`settings.tool_timeout_overrides[name]` → 写 `settings.tool_write_timeout_seconds` / 读 `settings.tool_timeout_seconds`;`query_faq` 读路径政策档 `settings.knowledge_tool_timeout_seconds`(覆盖优先于政策档)。按名覆盖只改等待期限,不给写工具开重试。

- [ ] **Step 1: 写失败测试(全量重写 tests/test_executor.py)**

```python
"""ch08 执行引擎(spec §2/§3):六步管线 / 超时重试 / 写确认路径 / 批次契约 / 审计。"""
import asyncio
import json
import time

import pytest
from pydantic import BaseModel, Field

from app.models import Ticket, ToolAuditLog, ToolWriteIdempotency, Conversation
from app.tools.catalog import (
    ToolSpec, TurnContext, canonical_args_sha, idempotency_key_for,
)
from app.tools.executor import (
    PendingWrite, ToolExecutor, ToolFace, batch_violation,
)
from tests.conftest import TEST_USER_ID, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class _X(BaseModel):
    x: str = Field(min_length=1, max_length=64)


class _OrderId(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)


def _spec(name, fn, permission="read", ownership="none", source="builtin",
          mcp_server=None, formatter=None, args_model=_X):
    return ToolSpec(name=name, description="d", args_schema=args_model.model_json_schema(),
                    args_model=args_model, source=source, permission=permission,
                    ownership=ownership, mcp_server=mcp_server, fn=fn,
                    formatter=formatter)


def _ctx(sf, **over):
    base = dict(user_id=TEST_USER_ID, conversation_id=None, resolved_query="q",
                retriever=None, settings=make_settings(), session_factory=sf)
    base.update(over)
    return TurnContext(**base)


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def _audits(sf, tool_call_id="c1"):
    with sf() as s:
        return s.query(ToolAuditLog).filter_by(tool_call_id=tool_call_id).all()


# ── 面与契约 ──

def test_face_rejects_duplicate():
    f1 = _spec("a", lambda a, c: "1")
    with pytest.raises(ValueError):
        ToolFace([f1, _spec("a", lambda a, c: "2")])


def test_batch_violation_rules():
    w = _spec("create_ticket", lambda a, c, m: "", permission="write")
    r = _spec("query_order", lambda a, c: "{}")
    face = ToolFace([w, r])
    ok = [_call("query_order", {"x": "1"}, "c1"), _call("create_ticket", {"x": "1"}, "c2")]
    assert batch_violation(ok, face) is None
    two_writes = ok + [_call("create_ticket", {"x": "1"}, "c3")]
    assert batch_violation(two_writes, face) is not None
    write_not_last = [_call("create_ticket", {"x": "1"}, "c1"),
                      _call("query_order", {"x": "1"}, "c2")]
    assert batch_violation(write_not_last, face) is not None


# ── 六步管线 ──

async def test_unknown_tool(db_session_factory):
    ex = ToolExecutor(ToolFace([]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("ghost", {"x": "1"}, "c9"), _ctx(db_session_factory))
    assert out.message.status == "error" and out.record.error_code == "unknown_tool"
    assert _audits(db_session_factory, "c9")[0].status == "失败"


async def test_invalid_args_carries_detail(db_session_factory):
    ex = ToolExecutor(ToolFace([_spec("t", lambda a, c: "ok")]), make_settings(),
                      session_factory=db_session_factory)
    out = await ex.execute(_call("t", {"x": ""}, "c7"), _ctx(db_session_factory))
    assert out.message.status == "error" and out.record.error_code == "invalid_args"
    assert "参数" in out.message.content and "x" in out.message.content  # 校验说明回灌
    assert out.record.retry_count == 0
    assert _audits(db_session_factory, "c7")[0].status == "校验拦下"


async def test_ownership_guard_short_circuits(db_session_factory):
    called = {"n": 0}

    def spy(args, ctx):
        called["n"] += 1
        return "{}"

    spec = _spec("query_order", spy, ownership="order", args_model=_OrderId)
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("query_order", {"order_id": "9999-1001"}),
                           _ctx(db_session_factory))
    assert called["n"] == 0                                   # 归属把门未过,未分发
    assert json.loads(out.message.content) == {"error": "订单不存在或不属于当前用户"}
    assert out.message.status == "success"                    # 业务空结果(spec §3 终态映射)
    assert _audits(db_session_factory)[0].status == "成功"


async def test_read_success_audit_fields(db_session_factory):
    spec = _spec("query_product", lambda a, c: json.dumps({"ok": 1}, ensure_ascii=False))
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("query_product", {"x": "1"}), _ctx(db_session_factory))
    assert out.message.status == "success"
    row = _audits(db_session_factory)[0]
    assert (row.status, row.tool_source, row.mcp_server, row.retry_count) == \
           ("成功", "builtin", None, 0)
    assert row.duration_ms is not None and row.arguments == {"x": "1"}


async def test_timeout_retry_exhausted_audits_timeout(db_session_factory):
    def slow(args, ctx):
        time.sleep(1)
        return "done"

    spec = _spec("slow_tool", slow)
    ex = ToolExecutor(ToolFace([spec]),
                      make_settings(tool_max_retries=1,
                                    tool_timeout_overrides={"slow_tool": 0.05}),
                      session_factory=db_session_factory)
    t0 = time.monotonic()
    out = await ex.execute(_call("slow_tool", {"x": "1"}), _ctx(db_session_factory))
    assert time.monotonic() - t0 < 3
    assert out.record.error_code == "tool_unavailable" and out.record.retry_count == 1
    row = _audits(db_session_factory)[0]
    assert row.status == "超时" and row.retry_count == 1      # 验收 6:重试次数/超时/耗时齐


async def test_retry_eventually_succeeds(db_session_factory):
    calls = {"n": 0}

    def flaky(args, ctx):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("jitter")
        return "ok-after-retry"

    spec = _spec("flaky", flaky)
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("flaky", {"x": "1"}), _ctx(db_session_factory))
    assert out.message.content == "ok-after-retry" and out.record.retry_count == 2
    assert _audits(db_session_factory)[0].status == "成功"


async def test_business_error_not_retried_and_sanitized(db_session_factory):
    def boom(args, ctx):
        raise ValueError("business broke")

    spec = _spec("biz", boom)
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("biz", {"x": "1"}), _ctx(db_session_factory))
    assert out.record.retry_count == 0 and "business broke" not in out.message.content
    assert _audits(db_session_factory)[0].status == "失败"


async def test_formatter_picks_fields(db_session_factory):
    spec = _spec("mcpish", None, source="mcp", mcp_server="logistics",
                 formatter=lambda raw: json.dumps({"company": "顺丰速运"},
                                                  ensure_ascii=False))

    class FakeMcp:
        async def call(self, server, name, args):
            return json.dumps({"company": "顺丰速运", "internal_code": "SF01"})

    ex = ToolExecutor(ToolFace([spec]), make_settings(),
                      session_factory=db_session_factory, mcp=FakeMcp())
    out = await ex.execute(_call("mcpish", {"x": "1"}), _ctx(db_session_factory))
    assert "internal_code" not in out.message.content and "顺丰速运" in out.message.content
    row = _audits(db_session_factory)[0]
    assert (row.tool_source, row.mcp_server) == ("mcp", "logistics")


async def test_write_call_rejected_on_plain_execute(db_session_factory):
    called = {"n": 0}

    def wfn(args, ctx, meta):
        called["n"] += 1
        return "{}"

    spec = _spec("create_ticket", wfn, permission="write")
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("create_ticket", {"x": "1"}), _ctx(db_session_factory))
    assert called["n"] == 0 and out.record.error_code == "permission_denied"
    assert _audits(db_session_factory)[0].status == "权限拒绝"


async def test_mark_invalid_batch_audits_each(db_session_factory):
    ex = ToolExecutor(ToolFace([]), make_settings(), session_factory=db_session_factory)
    msgs = await ex.mark_invalid_batch(
        [_call("a", {"x": "1"}, "c1"), _call("b", {"x": "1"}, "c2")],
        _ctx(db_session_factory), "写操作必须是最后一步调用")
    assert [m.tool_call_id for m in msgs] == ["c1", "c2"]
    assert all(m.status == "error" for m in msgs)
    with db_session_factory() as s:
        rows = s.query(ToolAuditLog).filter(ToolAuditLog.tool_call_id.in_(["c1", "c2"])).all()
    assert {r.status for r in rows} == {"校验拦下"} and len(rows) == 2


# ── execute_confirmed 写确认路径 ──

def _seed_conv(sf, cid=1):
    with sf() as s:
        s.add(Conversation(id=cid, user_id=TEST_USER_ID))
        s.commit()


def _ticket_spec():
    from app.tools.builtin.ticket import create_ticket_fn
    return {s.name: s for s in __import__(
        "app.tools.builtin", fromlist=["scan_builtin_specs"]).scan_builtin_specs()
    }["create_ticket"]


def _pending(args=None):
    args = args or {"description": "商品质量问题", "ticket_type": "售后"}
    return PendingWrite(tool_call_id="tc1", name="create_ticket", args=args,
                        args_sha256=canonical_args_sha(args))


async def test_confirmed_write_success(db_session_factory):
    _seed_conv(db_session_factory)
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    out = await ex.execute_confirmed(_pending(), ctx)
    payload = json.loads(out.message.content)
    assert out.message.status == "success" and payload["ticket_no"].startswith("T")
    with db_session_factory() as s:
        assert s.get(Ticket, payload["ticket_no"]) is not None
        key = idempotency_key_for(1, "tc1")
        assert s.get(ToolWriteIdempotency, key).ticket_no == payload["ticket_no"]
    row = _audits(db_session_factory, "tc1")[0]
    assert row.status == "成功" and row.retry_count == 0


async def test_confirmed_write_idempotent_replay(db_session_factory):
    _seed_conv(db_session_factory)
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    first = json.loads((await ex.execute_confirmed(_pending(), ctx)).message.content)
    again = json.loads((await ex.execute_confirmed(_pending(), ctx)).message.content)
    assert again["ticket_no"] == first["ticket_no"]  # 重放返回原工单,不重复建
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 1


async def test_confirmed_write_args_conflict_rejected(db_session_factory):
    _seed_conv(db_session_factory)
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    await ex.execute_confirmed(_pending(), ctx)
    bad = _pending(args={"description": "被篡改", "ticket_type": "投诉"})
    out = await ex.execute_confirmed(bad, ctx)
    assert out.record.error_code == "idempotency_conflict"
    rows = _audits(db_session_factory, "tc1")
    assert rows[-1].status == "失败" and "被篡改" not in (rows[-1].error_message or "")


async def test_confirmed_write_requires_conversation(db_session_factory):
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute_confirmed(_pending(), _ctx(db_session_factory))  # cid=None
    assert out.record.error_code == "invalid_args"
    assert _audits(db_session_factory, "tc1")[0].status == "校验拦下"


async def test_confirmed_write_timeout_no_retry_background_lands(db_session_factory):
    _seed_conv(db_session_factory)
    import time as _time
    from app.tools.builtin import ticket as ticket_mod
    orig_fn = ticket_mod.create_ticket_fn

    def slow_fn(args, ctx, meta):
        _time.sleep(0.3)
        return orig_fn(args, ctx, meta)

    spec = ToolSpec(**{**_ticket_spec().__dict__, "fn": slow_fn})
    ex = ToolExecutor(ToolFace([spec]),
                      make_settings(tool_timeout_overrides={"create_ticket": 0.05}),
                      session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    out = await ex.execute_confirmed(_pending(), ctx)
    assert out.record.error_code == "write_timeout" and out.record.retry_count == 0
    assert "请勿重复提交" in out.message.content
    rows = _audits(db_session_factory, "tc1")
    assert len(rows) == 1 and rows[0].status == "超时" and rows[0].retry_count == 0
    await asyncio.sleep(0.6)  # 后台 shielded 写最终落地;审计仍只有一条
    with db_session_factory() as s:
        key = idempotency_key_for(1, "tc1")
        assert s.get(ToolWriteIdempotency, key) is not None
    assert len(_audits(db_session_factory, "tc1")) == 1


async def test_deny_write_audits_denied(db_session_factory):
    ex = ToolExecutor(ToolFace([]), make_settings(), session_factory=db_session_factory)
    out = await ex.deny_write(_pending(), _ctx(db_session_factory), "用户取消")
    assert out.message.status == "error" and "取消" in out.message.content
    row = _audits(db_session_factory, "tc1")[0]
    assert row.status == "权限拒绝" and row.arguments["description"] == "商品质量问题"
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_executor.py -q`
Expected: ImportError / FAIL(新 API 不存在;旧测试文件先整体替换为本测试,旧 API 测试随之退役)

注意:`app/graph/agent_node.py` 仍 import 旧 API,本任务 Step 3 替换 executor.py 后 `uv run pytest` 全量会因 agent_node import 失败而红——本任务的「跑绿」只要求 `tests/test_executor.py` 全绿 + `python -c "import app.tools.executor"` 通过;agent_node 适配在 Task 9 完成(计划内顺序依赖,不算返工)。

- [ ] **Step 3: 实现(整体重写 app/tools/executor.py)**

```python
"""ch08 统一执行引擎(spec §2):所有工具调用(内置/MCP/写)同走一处。
六步管线:查目录 → JSON Schema 校验 → 权限闸 → 归属把门 → 分发(超时/重试)→ 格式化。
写操作只由 execute_confirmed 分发(用户确认后),不自动重试;超时按「已发未必未成」。"""
import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError, OperationalError

from app.knowledge.retriever import RetryableKnowledgeError
from app.services.orders import get_order
from app.services.token_budget import truncate_tool_result
from app.services.tool_audit import (
    AUDIT_DENIED, AUDIT_FAILURE, AUDIT_INVALID, AUDIT_SUCCESS, AUDIT_TIMEOUT,
    AuditEntry, write_audit,
)
from app.tools.catalog import (
    ToolSpec, TurnContext, WriteMeta, canonical_args_sha, idempotency_key_for,
)

logger = logging.getLogger("wayhelp.graph")

RETRYABLE = (TimeoutError, asyncio.TimeoutError, OperationalError, ConnectionError,
             RetryableKnowledgeError)

# 写超时后后台仍可能落地:进程级强引用集合,防 GC 回收任务;main.py lifespan 收尾 drain
PENDING_WRITE_TASKS: set[asyncio.Task] = set()


@dataclass(frozen=True)
class PendingWrite:
    """agent_node park 的写调用快照;confirm 时只信它,不信客户端字段。"""
    tool_call_id: str
    name: str
    args: dict
    args_sha256: str


@dataclass(frozen=True)
class ToolExecutionRecord:
    name: str
    args: dict
    ok: bool
    duration_ms: int
    retry_count: int
    error_type: str | None
    error_code: str | None = None
    source: str = "builtin"
    mcp_server: str | None = None
    audit_status: str = AUDIT_SUCCESS


@dataclass(frozen=True)
class ToolOutcome:
    message: ToolMessage
    record: ToolExecutionRecord


def _schema_carrier_func():
    return ""


class ToolFace:
    """每轮工具面:目录子集(含 MCP 发现条目);重名拒(spec §1.1 发现重名不覆盖)。"""

    def __init__(self, specs: list[ToolSpec]):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool name in face")
        self._specs = list(specs)
        self._by_name = {s.name: s for s in self._specs}

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def as_langchain_tools(self) -> list[StructuredTool]:
        return [StructuredTool(name=s.name, description=s.description,
                               args_schema=s.args_model, func=_schema_carrier_func)
                for s in self._specs]


def batch_violation(calls: list[dict], face: ToolFace) -> str | None:
    """批次契约(spec §2):一条模型响应最多一个写调用,且必须是最后一个。"""
    writes = [i for i, c in enumerate(calls)
              if (s := face.get(c.get("name", ""))) and s.permission == "write"]
    if not writes:
        return None
    if len(writes) > 1:
        return "一条回复中最多只能发起一个写操作"
    if writes[0] != len(calls) - 1:
        return "写操作必须是最后一步调用"
    return None


class ToolExecutor:
    def __init__(self, face: ToolFace, settings, session_factory=None, mcp=None):
        self._face = face
        self._settings = settings
        self._session_factory = session_factory
        self._mcp = mcp  # McpGateway | None(Task 8)

    # ── 有效超时:覆盖 > 政策档(faq) > 默认;写另有专档 ──
    def _read_timeout(self, name: str) -> float:
        ov = self._settings.tool_timeout_overrides.get(name)
        if ov is not None:
            return float(ov)
        if name == "query_faq":
            return self._settings.knowledge_tool_timeout_seconds
        return self._settings.tool_timeout_seconds

    def _write_timeout(self, name: str) -> float:
        ov = self._settings.tool_timeout_overrides.get(name)
        if ov is not None:
            return float(ov)
        return self._settings.tool_write_timeout_seconds

    # ── 普通路径(只读;写调用在此一律拒绝)──
    async def execute(self, call: dict, ctx: TurnContext) -> ToolOutcome:
        name = call.get("name", "")
        args = call.get("args") or {}
        call_id = call.get("id") or ""
        started = time.monotonic()
        spec = self._face.get(name)
        if spec is None:
            return await self._finish(None, name, args, call_id, "unknown_tool",
                                      "UnknownTool", started, 0, ctx,
                                      error_text="调用了未注册的工具")
        try:
            spec.args_model(**args)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                               for e in exc.errors()[:3])
            return await self._finish(spec, name, args, call_id, "invalid_args",
                                      "ValidationError", started, 0, ctx,
                                      error_text=f"工具参数不合法: {detail}")
        if spec.permission in ("write", "deny"):
            return await self._finish(spec, name, args, call_id, "permission_denied",
                                      "PermissionDenied", started, 0, ctx,
                                      error_text="没有权限执行该操作")
        if spec.ownership == "order":
            if get_order(ctx.user_id, (args.get("order_id") or "")) is None:
                content = json.dumps({"error": "订单不存在或不属于当前用户"},
                                     ensure_ascii=False)
                return await self._finish(spec, name, args, call_id, None, None,
                                          started, 0, ctx, ok=True, content=content)
        timeout = self._read_timeout(name)
        retries = 0
        max_retries = self._settings.tool_max_retries
        while True:
            try:
                raw = await self._dispatch_read(spec, args, ctx, timeout)
                content = self._format(spec, raw)
                return await self._finish(spec, name, args, call_id, None, None,
                                          started, retries, ctx, ok=True, content=content)
            except RETRYABLE as exc:
                if retries >= max_retries:
                    code = "tool_unavailable"
                    text = "工具暂时不可用(已重试)"
                    return await self._finish(spec, name, args, call_id, code,
                                              type(exc).__name__, started, retries, ctx,
                                              error_text=text,
                                              last_was_timeout=isinstance(
                                                  exc, (TimeoutError, asyncio.TimeoutError)))
                retries += 1
                await asyncio.sleep(0.5 * 2 ** (retries - 1))
            except McpToolError:      # Server 回 isError(业务失败,不重试,脱敏)
                return await self._finish(spec, name, args, call_id, "mcp_error",
                                          "McpToolError", started, retries, ctx,
                                          error_text="外部服务暂时不可用")
            except Exception as exc:
                return await self._finish(spec, name, args, call_id, "tool_error",
                                          type(exc).__name__, started, retries, ctx,
                                          error_text="工具执行失败")

    async def _dispatch_read(self, spec: ToolSpec, args: dict, ctx: TurnContext,
                             timeout: float):
        if spec.source == "mcp":
            if self._mcp is None:
                raise ConnectionError("mcp gateway unavailable")
            return await asyncio.wait_for(
                self._mcp.call(spec.mcp_server, spec.name, args), timeout=timeout)
        return await asyncio.wait_for(asyncio.to_thread(spec.fn, args, ctx), timeout=timeout)

    def _format(self, spec: ToolSpec, raw) -> str:
        content = raw if isinstance(raw, str) else str(raw)
        if spec.formatter is not None:
            content = spec.formatter(content)
        return truncate_tool_result(content, self._settings.tool_result_max_tokens)

    # ── 写确认路径(用户确认后唯一写入口;confirmed 不是模型可控参数)──
    async def execute_confirmed(self, pending: PendingWrite, ctx: TurnContext) -> ToolOutcome:
        started = time.monotonic()
        spec = self._face.get(pending.name)
        if spec is None or spec.permission != "write":
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "permission_denied", "PermissionDenied", started, 0,
                                      ctx, error_text="没有权限执行该操作")
        if ctx.conversation_id is None:
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "invalid_args", "ConversationRequired", started, 0,
                                      ctx, error_text="写操作缺少会话上下文,已拦下")
        key = idempotency_key_for(ctx.conversation_id, pending.tool_call_id)
        found = await asyncio.to_thread(self._idem_lookup, key, pending.args_sha256)
        if found is not None:
            if found == "__conflict__":
                return await self._finish(spec, pending.name, pending.args,
                                          pending.tool_call_id, "idempotency_conflict",
                                          "IdempotencyConflict", started, 0, ctx,
                                          error_text="内部一致性错误,已拦下重复写入")
            content = json.dumps({"ticket_no": found, "status": "待处理",
                                  "note": "重复提交已忽略"}, ensure_ascii=False)
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      None, None, started, 0, ctx, ok=True, content=content)
        meta = WriteMeta(key=key, args_sha256=pending.args_sha256)
        task = asyncio.ensure_future(asyncio.to_thread(spec.fn, pending.args, ctx, meta))
        PENDING_WRITE_TASKS.add(task)
        task.add_done_callback(self._write_task_done)
        try:
            raw = await asyncio.wait_for(asyncio.shield(task),
                                         timeout=self._write_timeout(pending.name))
            content = self._format(spec, raw)
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      None, None, started, 0, ctx, ok=True, content=content)
        except (TimeoutError, asyncio.TimeoutError):
            # 超时≠没执行:立即返回未知;后台任务继续,审计只此一条(spec §3)
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "write_timeout", "TimeoutError", started, 0, ctx,
                                      error_text="提交超时,结果未知,请勿重复提交;"
                                                 "如需要请查询工单列表确认")
        except IntegrityError:
            found = await asyncio.to_thread(self._idem_lookup, key, pending.args_sha256)
            if found is not None and found != "__conflict__":
                content = json.dumps({"ticket_no": found, "status": "待处理",
                                      "note": "重复提交已忽略"}, ensure_ascii=False)
                return await self._finish(spec, pending.name, pending.args,
                                          pending.tool_call_id, None, None, started, 0, ctx,
                                          ok=True, content=content)
            return await self._finish(spec, pending.name, pending.args,
                                      pending.tool_call_id, "idempotency_conflict",
                                      "IdempotencyConflict", started, 0, ctx,
                                      error_text="内部一致性错误,已拦下重复写入")
        except Exception as exc:
            return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                      "tool_error", type(exc).__name__, started, 0, ctx,
                                      error_text="工具执行失败")

    def _idem_lookup(self, key: str, args_sha256: str) -> str | None:
        """返回已建 ticket_no;key 存在但参数摘要不符 → '__conflict__';不存在 → None。
        '__conflict__' 只是标记,任何路径不得把它当 ticket_no 透出。"""
        from app.models import ToolWriteIdempotency
        if self._session_factory is None:
            return None
        with self._session_factory() as s:
            row = s.get(ToolWriteIdempotency, key)
            if row is None:
                return None
            return row.ticket_no if row.arguments_sha256 == args_sha256 else "__conflict__"

    def _write_task_done(self, task: asyncio.Task) -> None:
        PENDING_WRITE_TASKS.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("后台写任务异常(已消费,不再追加审计): %s", type(exc).__name__)
        else:
            logger.info("后台写任务完成(超时返回后落地;不追加第二条审计)")

    # ── 取消写(用户点「取消」):审计权限拒绝,arguments 记原始申请 ──
    async def deny_write(self, pending: PendingWrite, ctx: TurnContext,
                         reason: str) -> ToolOutcome:
        started = time.monotonic()
        spec = self._face.get(pending.name)
        return await self._finish(spec, pending.name, pending.args, pending.tool_call_id,
                                  "permission_denied", "UserCancelled", started, 0, ctx,
                                  error_text=f"用户已取消,本次未执行({reason})")

    # ── 非法批次:零执行,每个 call 一条校验拦下审计 + 错误 ToolMessage 回灌 ──
    async def mark_invalid_batch(self, calls: list[dict], ctx: TurnContext,
                                 reason: str) -> list[ToolMessage]:
        msgs = []
        for call in calls:
            spec = self._face.get(call.get("name", ""))
            out = await self._finish(spec, call.get("name", ""), call.get("args") or {},
                                     call.get("id") or "", "invalid_batch", "InvalidBatch",
                                     time.monotonic(), 0, ctx,
                                     error_text=f"工具调用批次不合法: {reason}")
            msgs.append(out.message)
        return msgs

    # ── 终结:统一产出 ToolOutcome + 落审计(失败不拦执行)──
    async def _finish(self, spec, name, args, call_id, error_code, error_type,
                      started, retries, ctx, *, ok=False, content=None,
                      error_text=None, last_was_timeout=False) -> ToolOutcome:
        duration = int((time.monotonic() - started) * 1000)
        if ok:
            audit_status = AUDIT_SUCCESS
            msg = ToolMessage(content=truncate_tool_result(
                content, self._settings.tool_result_max_tokens),
                tool_call_id=call_id, name=name or "unknown", status="success")
        else:
            if error_code in ("invalid_args", "invalid_batch") \
                    or error_type == "ConversationRequired":
                audit_status = AUDIT_INVALID
            elif error_code == "permission_denied":
                audit_status = AUDIT_DENIED
            elif error_code == "write_timeout" or last_was_timeout:
                audit_status = AUDIT_TIMEOUT
            else:
                audit_status = AUDIT_FAILURE
            msg = ToolMessage(content=error_text or "工具执行失败",
                              tool_call_id=call_id, name=name or "unknown", status="error")
        record = ToolExecutionRecord(
            name, args, ok, duration, retries, error_type, error_code,
            spec.source if spec else "builtin",
            spec.mcp_server if spec else None, audit_status)
        await write_audit(self._session_factory, AuditEntry(
            conversation_id=ctx.conversation_id, tool_call_id=call_id or None,
            tool_name=name or "unknown", tool_source=record.source,
            mcp_server=record.mcp_server, arguments=args or None,
            result_summary=msg.content if ok or error_code in (
                "write_timeout",) else None,
            status=audit_status,
            error_message=None if ok else (error_text or "")[:500],
            retry_count=retries, duration_ms=duration),
            self._settings.audit_result_max_chars)
        return ToolOutcome(msg, record)


class McpToolError(Exception):
    """gateway 抛出:Server 回 CallToolResult.isError=true(业务失败,不重试,脱敏)。"""
```

实现注意:
- `result_summary` 只在成功与写超时时落(写超时可能后台落行,摘要记返回给模型的「未知」措辞);失败类落 error_message。error_message 只记分诊后措辞,不记内部异常 message(防泄密,spec §3)。
- IntegrityError 只在 fn commit 时抛;此时 to_thread 任务已结束,`wait_for(shield(task))` 把 IntegrityError 透传。注意 task 已进入 PENDING_WRITE_TASKS,callback 会再打一次 error 日志(重复但无害,测试中不断言日志)。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_executor.py -q`
Expected: 全 PASS(旧 test_executor.py 已被新文件整体替换)

- [ ] **Step 5: Commit**

```bash
git add app/tools/executor.py tests/test_executor.py
git commit -m "feat(ch08): unified execution engine with audit, write confirm path, batch contract"
```

---

### Task 7: 两个业务 MCP Server(物流 / 售后)

**Files:**
- Create: `app/mcp_servers/__init__.py`(空)
- Create: `app/mcp_servers/mock_data.py`
- Create: `app/mcp_servers/logistics_server.py`
- Create: `app/mcp_servers/after_sales_server.py`
- Test: `tests/test_mcp_servers.py`

**Interfaces:**
- Consumes: Task 2 的 `mcp` 依赖(v1 FastMCP)。
- Produces:
  - `mock_data.logistics_trace(order_id: str) -> dict` / `warranty_status(order_id) -> dict` / `return_progress(order_id) -> dict`:按 order_id SHA 种子稳定生成(同一单多次查一致;不读订单服务、不认用户,spec §4.1)
  - `logistics_server` 模块:工具 `query_logistics(order_id: str) -> dict`;`main()` 入口,`python -m app.mcp_servers.logistics_server` 起服
  - `after_sales_server` 模块:工具 `query_warranty(order_id)` / `query_return_progress(order_id)`;同样 `main()` 入口
  - 端口环境变量:`WAYHELP_MCP_LOGISTICS_PORT`(默认 8101)/ `WAYHELP_MCP_AFTER_SALES_PORT`(默认 8102);HTTP 路径用 SDK 默认 `/mcp`(以安装包类型定义为准,冒烟测试钉)
- **v1 API 钉**:只用 `from mcp.server.fastmcp import FastMCP`;host/port 在 v1 是 `FastMCP(..., host=..., port=...)` 构造参数;`mcp.run(transport="streamable-http")`。若安装包签名不符,以 `uv run python -c "import inspect; from mcp.server.fastmcp import FastMCP; print(inspect.signature(FastMCP.__init__)); print(inspect.signature(FastMCP.run))"` 输出为准调整,transport 字面量不得改成 v2 之外的值。

- [ ] **Step 1: 写失败测试**

`tests/test_mcp_servers.py`:
```python
"""ch08 MCP Server 冒烟(spec §4.3):真起子进程,list-tools + call-tool 钉住
v1 FastMCP API、streamable-http 传输与 /mcp 路径;同时是 adapter transport 字面量的钉。"""
import os
import socket
import subprocess
import sys
import time

import pytest

from app.mcp_servers.mock_data import (
    logistics_trace, return_progress, warranty_status,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} not ready")


@pytest.fixture(scope="module")
def logistics_url():
    port = _free_port()
    env = {**os.environ, "WAYHELP_MCP_LOGISTICS_PORT": str(port),
           "no_proxy": "127.0.0.1,localhost"}
    p = subprocess.Popen([sys.executable, "-m", "app.mcp_servers.logistics_server"],
                         env=env)
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        p.terminate()
        p.wait(10)


@pytest.fixture(scope="module")
def after_sales_url():
    port = _free_port()
    env = {**os.environ, "WAYHELP_MCP_AFTER_SALES_PORT": str(port),
           "no_proxy": "127.0.0.1,localhost"}
    p = subprocess.Popen([sys.executable, "-m", "app.mcp_servers.after_sales_server"],
                         env=env)
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        p.terminate()
        p.wait(10)


def test_mock_data_seeded_stable():
    a, b = logistics_trace("AB12-1001"), logistics_trace("AB12-1001")
    assert a == b and a["company"] and a["tracking_no"] and len(a["traces"]) >= 2
    w = warranty_status("AB12-1001")
    assert w == warranty_status("AB12-1001") and w["in_warranty"] in (True, False)
    r = return_progress("AB12-1001")
    assert r == return_progress("AB12-1001") and r["nodes"]


async def test_logistics_server_list_and_call_tools(logistics_url):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "logistics": {"url": logistics_url, "transport": "streamable_http"}})
    tools = await client.get_tools(server_name="logistics")
    names = {t.name for t in tools}
    assert names == {"query_logistics"}
    t = tools[0]
    schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
    props = schema.get("properties") or {}
    assert set(props) == {"order_id"}
    async with client.session("logistics") as session:
        result = await session.call_tool("query_logistics", {"order_id": "AB12-1001"})
    assert not result.isError
    text = result.content[0].text
    assert "顺丰" in text or "company" in text


async def test_after_sales_server_two_tools(after_sales_url):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "after_sales": {"url": after_sales_url, "transport": "streamable_http"}})
    tools = await client.get_tools(server_name="after_sales")
    assert {t.name for t in tools} == {"query_warranty", "query_return_progress"}
    async with client.session("after_sales") as session:
        r1 = await session.call_tool("query_warranty", {"order_id": "AB12-1001"})
        r2 = await session.call_tool("query_return_progress", {"order_id": "AB12-1001"})
    assert not r1.isError and not r2.isError
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_mcp_servers.py -q`
Expected: ImportError(app.mcp_servers 不存在)

- [ ] **Step 3: 实现**

`app/mcp_servers/mock_data.py`:
```python
"""ch08 两个业务 MCP Server 的 mock 数据(spec §4.1):照 ch02 做法随机生成,
按 order_id 种子稳定(同一单多次查一致);不接真实系统、不建表、不认用户。"""
import hashlib
from datetime import datetime, timedelta

_COMPANIES = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
_TRACE_POOL = ["已揽收", "到达转运中心", "运输中", "派件中", "已签收"]


def _seed(order_id: str) -> int:
    return int(hashlib.sha256(order_id.encode("utf-8")).hexdigest()[:8], 16)


def logistics_trace(order_id: str) -> dict:
    s = _seed(order_id)
    company = _COMPANIES[s % len(_COMPANIES)]
    tracking_no = "SF" + f"{s % 10**10:010d}"
    n = 2 + s % 3                      # 2~4 条轨迹
    base = datetime.now() - timedelta(hours=8 * n)
    traces = [{"time": (base + timedelta(hours=8 * (i + 1))).isoformat(timespec="seconds"),
               "description": _TRACE_POOL[min(i, len(_TRACE_POOL) - 1)]}
              for i in range(n)]
    return {"order_id": order_id, "company": company,
            "tracking_no": tracking_no, "traces": traces}


def warranty_status(order_id: str) -> dict:
    s = _seed(order_id)
    in_warranty = (s % 10) < 6         # 60% 在保
    expire = (datetime.now() + timedelta(days=180 if in_warranty else -30)).date().isoformat()
    return {"order_id": order_id, "in_warranty": in_warranty,
            "warranty_expire": expire,
            "note": "整机保修期内,非人为损坏可免费维修" if in_warranty
                    else "已过保修期,维修需自费"}


def return_progress(order_id: str) -> dict:
    s = _seed(order_id)
    stages = ["退货申请已提交", "商家已同意", "退货物流已揽收", "商家已收货", "退款已到账"]
    n = 1 + s % len(stages)            # 进度停在某一节点
    base = datetime.now() - timedelta(days=n)
    nodes = [{"stage": stages[i],
              "time": (base + timedelta(days=i)).isoformat(timespec="seconds"),
              "done": True} for i in range(n)]
    return {"order_id": order_id, "return_no": "R" + f"{s % 10**8:08d}",
            "current": stages[n - 1], "nodes": nodes}
```

`app/mcp_servers/logistics_server.py`:
```python
"""ch08 物流 MCP Server(独立进程,spec §4.1):query_logistics 接管内置下线的物流查询。
v1 FastMCP + streamable-http;起服:uv run python -m app.mcp_servers.logistics_server"""
import os

from mcp.server.fastmcp import FastMCP

from app.mcp_servers.mock_data import logistics_trace

mcp = FastMCP("logistics", host="127.0.0.1",
              port=int(os.environ.get("WAYHELP_MCP_LOGISTICS_PORT", "8101")))


@mcp.tool()
def query_logistics(order_id: str) -> dict:
    """查询订单物流轨迹。参数 order_id 为订单号。返回物流公司、运单号与物流节点列表。"""
    return logistics_trace(order_id)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
```

`app/mcp_servers/after_sales_server.py`:
```python
"""ch08 售后 MCP Server(独立进程,spec §4.1):查在保 / 查退货进度。
v1 FastMCP + streamable-http;起服:uv run python -m app.mcp_servers.after_sales_server"""
import os

from mcp.server.fastmcp import FastMCP

from app.mcp_servers.mock_data import return_progress, warranty_status

mcp = FastMCP("after_sales", host="127.0.0.1",
              port=int(os.environ.get("WAYHELP_MCP_AFTER_SALES_PORT", "8102")))


@mcp.tool()
def query_warranty(order_id: str) -> dict:
    """查询订单商品是否在保修期内。参数 order_id 为订单号。返回在保状态、保修到期日与说明。"""
    return warranty_status(order_id)


@mcp.tool()
def query_return_progress(order_id: str) -> dict:
    """查询订单退货进度。参数 order_id 为订单号。返回退货单号、当前节点与进度列表。"""
    return return_progress(order_id)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_mcp_servers.py -q`
Expected: 4 PASS。若 v1 签名不符(host/port 不在构造里),按 Step 0 的 inspect 命令输出调整并在此记录实际签名。

- [ ] **Step 5: Commit**

```bash
git add app/mcp_servers/ tests/test_mcp_servers.py
git commit -m "feat(ch08): logistics & after-sales MCP servers with seeded mock data"
```

---

### Task 8: MCP client 网关 + 每轮发现 + main.py 接线

**Files:**
- Create: `app/services/mcp_gateway.py`
- Test: `tests/test_mcp_gateway.py`
- Modify: `app/main.py`(AppRuntime 不动;create_app 内建 catalog/gateway 并挂 app.state + GraphDeps;lifespan 关闭 gateway)

**Interfaces:**
- Consumes: Task 2 config;Task 4 `classify_mcp_tool/mcp_args_model/ToolSpec`;Task 6 `McpToolError`(gateway 从 executor import,不在 gateway 重定义);Task 7 的两个 Server。
- Produces:
  - `McpGateway(settings)`:`configured: bool`;`async discover() -> list[ToolSpec]`(按 Server 独立 wait_for + 独立 try,单个挂不影响另一个;每个工具过 classify_mcp_tool,None 则跳过记 warning;`budget_priority=10`,`permission/ownership` 由策略给);`async call(server, name, args) -> str`(async with session;`result.isError` → 抛 `McpToolError`;否则拼接 text 内容返回);`async close()`
  - main.py:`build_tool_catalog() -> ToolCatalog`(扫描 builtin 注册);`app.state.tool_catalog` / `app.state.mcp_gateway`;GraphDeps 新字段 `catalog` / `mcp_gateway` / `session_factory`(在 Task 9 消费,本任务先挂通)
- discovery 转条目时 args_schema 统一为 dict(adapter 返回 BaseTool 的 args_schema 可能是 pydantic 模型或 dict,两种都归一成 dict)

- [ ] **Step 1: 写失败测试**

`tests/test_mcp_gateway.py`:
```python
"""ch08 MCP 网关(spec §4.2):按 Server 独立超时发现、能力政策过滤、isError 脱敏。"""
import pytest

from app.services.mcp_gateway import McpGateway
from app.tools.executor import McpToolError
from tests.conftest import make_settings
from tests.test_mcp_servers import after_sales_url, logistics_url  # noqa: F401


def _gw(**over):
    return McpGateway(make_settings(**over))


async def test_discover_three_tools(logistics_url, after_sales_url):
    gw = _gw(mcp_logistics_url=logistics_url, mcp_after_sales_url=after_sales_url)
    specs = {s.name: s for s in await gw.discover()}
    assert set(specs) == {"query_logistics", "query_warranty", "query_return_progress"}
    assert all(s.source == "mcp" and s.permission == "read" for s in specs.values())
    assert specs["query_logistics"].mcp_server == "logistics"
    assert specs["query_logistics"].ownership == "order"
    await gw.close()


async def test_one_server_down_other_still_discovered(logistics_url):
    gw = _gw(mcp_logistics_url=logistics_url,
             mcp_after_sales_url="http://127.0.0.1:9/mcp",  # 不可达
             mcp_discovery_timeout_seconds=1)
    specs = await gw.discover()          # 不抛、不等满默认超时
    assert [s.name for s in specs] == ["query_logistics"]
    await gw.close()


async def test_unconfigured_gateway_discovers_nothing():
    gw = _gw()
    assert not gw.configured
    assert await gw.discover() == []


async def test_call_and_iserror(logistics_url):
    gw = _gw(mcp_logistics_url=logistics_url)
    text = await gw.call("logistics", "query_logistics", {"order_id": "AB12-1001"})
    assert "company" in text
    with pytest.raises((McpToolError, Exception)):  # 未知工具:Server 失败统一脱敏
        await gw.call("logistics", "no_such_tool", {})
    await gw.close()
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_mcp_gateway.py -q`
Expected: ImportError

- [ ] **Step 3: 实现**

`app/services/mcp_gateway.py`:
```python
"""ch08 MCP client 网关(spec §4.2):MultiServerMCPClient 全局单例的薄封装。
每轮按 Server 独立超时现问现拿;权限/归属只认本地能力政策;某 Server 挂了跳过记 warning。"""
import asyncio
import json
import logging

from app.tools.catalog import ToolSpec, classify_mcp_tool, mcp_args_model
from app.tools.executor import McpToolError

logger = logging.getLogger("wayhelp.graph")

_MCP_BUDGET_PRIORITY = 10  # 动态工具最低档:超预算先剔(spec §1.5)


class McpGateway:
    def __init__(self, settings):
        self._timeout = settings.mcp_discovery_timeout_seconds
        servers = {}
        if settings.mcp_logistics_url:
            servers["logistics"] = {"url": settings.mcp_logistics_url,
                                    "transport": "streamable_http"}
        if settings.mcp_after_sales_url:
            servers["after_sales"] = {"url": settings.mcp_after_sales_url,
                                      "transport": "streamable_http"}
        self._servers = tuple(servers)
        from langchain_mcp_adapters.client import MultiServerMCPClient
        self._client = MultiServerMCPClient(servers) if servers else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def discover(self) -> list[ToolSpec]:
        """每轮装配时调用;按 Server 独立超时/独立捕获,互不倒逼。"""
        if self._client is None:
            return []
        out: list[ToolSpec] = []
        for server in self._servers:
            try:
                tools = await asyncio.wait_for(
                    self._client.get_tools(server_name=server), timeout=self._timeout)
            except Exception as exc:
                logger.warning("mcp discover %s 失败(本轮跳过): %s",
                               server, type(exc).__name__)
                continue
            for t in tools:
                schema = t.args_schema if isinstance(t.args_schema, dict) \
                    else t.args_schema.model_json_schema()
                verdict = classify_mcp_tool(server, schema)
                if verdict is None:
                    logger.warning("mcp tool %s/%s 不符本地只读策略,拒进工具面",
                                   server, t.name)
                    continue
                permission, ownership = verdict
                out.append(ToolSpec(
                    name=t.name, description=t.description or "",
                    args_schema=schema, args_model=mcp_args_model(ownership),
                    source="mcp", permission=permission, ownership=ownership,
                    mcp_server=server, budget_priority=_MCP_BUDGET_PRIORITY,
                    formatter=_slim_result))
        return out

    async def call(self, server: str, name: str, args: dict) -> str:
        if self._client is None:
            raise ConnectionError("mcp gateway not configured")
        async with self._client.session(server) as session:
            result = await session.call_tool(name, args)
        if getattr(result, "isError", False):
            raise McpToolError(f"{server}/{name} returned isError")
        parts = [getattr(c, "text", "") for c in (result.content or [])]
        return "\n".join(p for p in parts if p)

    async def close(self) -> None:
        # adapter 按调用新建 session,无长连接;保留接口供 lifespan 对称调用
        return None


def _slim_result(raw: str) -> str:
    """MCP 结果格式化(spec §2 第 6 步):JSON 原样透传(种子 mock 字段全是回答用字段),
    只确保是合法 JSON 且中文不转义;非法 JSON 原样返回(截断由引擎统一处理)。"""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (ValueError, TypeError):
        return raw
```

`app/main.py` 改动:
1. `GraphDeps`(nodes.py:47-55)加三字段(在 summary_runner 后):
```python
    catalog: Any = None          # ch08 ToolCatalog(main_agent 工具面装配)
    mcp_gateway: Any = None      # ch08 McpGateway(每轮 MCP 发现;None = 未配置)
    session_factory: Any = None  # ch08 审计/写确认用
```
2. create_app 内(现有 probe_tools 段之后)加:
```python
    # ch08 工具目录与 MCP 网关(spec §1.5):内置启动登记;MCP 每轮现问现拿
    from app.services.mcp_gateway import McpGateway
    from app.tools.builtin import scan_builtin_specs
    from app.tools.catalog import ToolCatalog
    tool_catalog = ToolCatalog()
    for _spec in scan_builtin_specs():
        tool_catalog.register(_spec)
    mcp_gateway = McpGateway(settings)
```
   `deps = GraphDeps(...)` 调用追加 `catalog=tool_catalog, mcp_gateway=mcp_gateway, session_factory=runtime.session_factory`。
   `app.state` 追加 `app.state.tool_catalog = tool_catalog`、`app.state.mcp_gateway = mcp_gateway`。
3. lifespan 收尾(`await app.state.summary_runner.aclose()` 之后)加:
```python
        from app.tools.executor import PENDING_WRITE_TASKS
        if PENDING_WRITE_TASKS:
            await asyncio.gather(*PENDING_WRITE_TASKS, return_exceptions=True)
        await app.state.mcp_gateway.close()
```
   并确保 `import asyncio` 存在(没有就加)。
4. 启动预算自检的 probe_tools 行(main.py:137)暂时保留 build_graph_tools——Task 9 删除该函数时一并改成目录装配;本任务不动它。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest tests/test_mcp_gateway.py -q && uv run pytest tests/test_mcp_servers.py -q`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/mcp_gateway.py tests/test_mcp_gateway.py app/main.py app/graph/nodes.py
git commit -m "feat(ch08): MCP gateway with per-server discovery + app wiring"
```

---

### Task 9: agent_node 改造(目录装配 + 批次契约 + write 检出)

**Files:**
- Modify: `app/graph/agent_node.py`(工具面装配段 :114-131、工具执行段 :311-341)
- Modify: `app/tools/business.py`(删除 `build_graph_tools` / `GraphToolset` / `_deterministic_logistics`;保留 `MOCK_TOOLS`/`build_tools`/`write_ticket`/`RetrievalTrace`/`TurnToolset` 与 FaqRetrievalTrace re-export)
- Modify: `app/main.py:134-139`(启动预算自检 probe 改目录装配)
- Test: `tests/test_agent_tools_face.py`(新建)

**Interfaces:**
- Consumes: Task 4/5/6/8 全部产物;GraphDeps.catalog/mcp_gateway/session_factory(Task 8 已挂)。
- Produces:
  - 分支矩阵(spec §1.4):business → 全部内置 + MCP + suggest_options;refund 过闸 → `query_order`/`query_product` + MCP + suggest_options;clarify → 无工具。MCP 条目与已选重名 → 拒进并记 warning(不按发现顺序覆盖,spec §1.1)
  - `main_agent` 返回新键:`pending_ticket: dict | None`(shape `{"tool_call_id","name","args","args_sha256"}`,checkpoint 可序列化;Task 10 路由消费)
- 行为不变量:faq 闸(`ctx.faq_trace`)/ suggest_options 本地校验 / order_context 焦点(`ctx.order_snapshots`)/ token 三条件过滤 / 预算兜底全部保留。

- [ ] **Step 1: 写失败测试**

`tests/test_agent_tools_face.py`:
```python
"""ch08 工具面装配与批次契约(spec §1.4/§2):business 面含 create_ticket、
物流内置已下线;非法批次(写不在最后)整批拦下、零执行、逐 call 审计、模型恢复收尾。"""
import json

from app.main import create_app
from app.models import ToolAuditLog
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401
from tests.test_ch05_acceptance import _client, _turn, _types


async def test_business_face_has_create_ticket_no_logistics():
    model = ScriptedChatModel(scripts=[
        ['{"intent":"订单","confidence":0.9}'],   # classify(首轮 understand 跳过)
        ["好的。"],                                # main_agent 一步收尾
    ])
    app = create_app(settings=make_settings(), model=model, runtime=make_runtime(tools=[]))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "我的订单怎么样了")
    assert _types(frames) == ["session", "delta", "[DONE]"]
    bound = set(model.bound or [])
    assert {"query_order", "query_product", "query_faq", "create_ticket",
            "suggest_options"} <= bound
    assert "query_logistics" not in bound          # 内置下线,MCP 未配置时缺席(spec §4.1)


async def test_invalid_batch_write_not_last_recovers(db_session_factory):
    scripts = [
        ['{"intent":"订单","confidence":0.9}'],
        [("tool", [{"index": 0, "name": "create_ticket", "id": "c1",
                    "args": '{"description":"商品质量问题","ticket_type":"售后"}'},
                   {"index": 1, "name": "query_order", "id": "c2",
                    "args": '{"order_id":"1111-1001"}'}])],
        ["好的,请问还有什么可以帮您?"],
    ]
    import dataclasses
    runtime = dataclasses.replace(make_runtime(tools=[]), session_factory=db_session_factory)
    app = create_app(settings=make_settings(), model=ScriptedChatModel(scripts=scripts),
                     runtime=runtime)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "帮我建个工单顺便查下订单")
    assert _types(frames) == ["session", "delta", "[DONE]"]  # 零执行:无 tool_start
    with db_session_factory() as s:
        rows = s.query(ToolAuditLog).filter(ToolAuditLog.tool_call_id.in_(["c1", "c2"])).all()
    assert len(rows) == 2 and {r.status for r in rows} == {"校验拦下"}
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_agent_tools_face.py -q`
Expected: FAIL(create_ticket 不在绑定面 / 批次契约未实现)

- [ ] **Step 3: 实现**

`app/graph/agent_node.py` 的 import 段替换:
```python
from app.tools.builtin.faq import FaqRetrievalTrace
from app.tools.catalog import TurnContext, canonical_args_sha
from app.tools.executor import (
    PendingWrite, ToolExecutor, ToolFace, batch_violation,
)
```
(删 `from app.tools.business import build_graph_tools`、`from app.tools.executor import ToolExecutor, ToolRegistry`。)

`main_agent` 工具面装配段(替换现 :114-131):
```python
        sid = (config.get("configurable") or {}).get("thread_id", "")
        faq_trace: FaqRetrievalTrace | None = None
        if refund_mode == "clarify":
            face = ToolFace([])
        else:
            builtin = {s.name: s for s in deps.catalog.specs()} if deps.catalog else {}
            if route == "business":
                picked = list(builtin.values())      # 全量内置(含 create_ticket)
                faq_trace = FaqRetrievalTrace()
            else:  # refund 过闸:query_order/query_product + MCP(无 faq/写)
                picked = [builtin[n] for n in ("query_order", "query_product")
                          if n in builtin]
            seen = {s.name for s in picked}
            mcp_specs = await deps.mcp_gateway.discover() if deps.mcp_gateway else []
            for s in mcp_specs:
                if s.name in seen:
                    logger.warning("mcp tool %s 与既有工具重名,本轮拒进工具面", s.name)
                    continue
                picked.append(s)
                seen.add(s.name)
            face = ToolFace(picked)
        ctx = TurnContext(user_id=user_id,
                          conversation_id=int(sid) if str(sid).isdecimal() else None,
                          resolved_query=state["resolved_query"],
                          retriever=deps.retriever if route == "business" else None,
                          settings=settings, session_factory=deps.session_factory,
                          faq_trace=faq_trace)
        executor = ToolExecutor(face, settings,
                                session_factory=deps.session_factory,
                                mcp=deps.mcp_gateway)
        tools = [*face.as_langchain_tools(), suggest_options] if face.specs else []
        model = deps.model.bind_tools(tools) if tools else deps.model
        budget = deps.context_budget or compute_budget(
            deps.settings, measure_sys_tokens(deps.system_prompt, tools))
        schema_est = measure_sys_tokens("", tools)
```
(`build_agent_context` 的 guard 仍按现状;schema 感知窗口与 budget_priority 裁剪是 Task 12,不在本任务提前做。)

`_faq_ok_fields` / `_effective_order_context` 里的 `toolset` 引用改:
```python
        def _faq_ok_fields() -> dict:
            if faq_trace is None or faq_trace.status != "ok":
                return {}
            return {"retrieval_status": "ok",
                    "retrieval_result": snapshot_retrieval(faq_trace.result),
                    "evidence": faq_trace.evidence or []}

        def _effective_order_context():
            if state.get("order_context"):
                return state["order_context"]
            distinct = {s["order_id"]: s for s in ctx.order_snapshots}
            if len(distinct) == 1:
                oid, snap = next(iter(distinct.items()))
                if oid in state["raw_query"] or oid in state["resolved_query"]:
                    return snap
            return None
```
`_faq_terminal` 的 `ft = toolset.faq_trace` 改 `ft = faq_trace`。

工具执行段(替换现 :311-341,`turn_messages.append(AIMessage(...))` 行保留在原位置):
```python
            violation = batch_violation(calls, face)
            if violation is not None:
                turn_messages.append(AIMessage(content="".join(text_parts), tool_calls=calls))
                msgs = await executor.mark_invalid_batch(calls, ctx, violation)
                turn_messages.extend(msgs)
                trace.append({"node": "main_agent", "invalid_batch": violation})
                continue  # 零执行回灌,让模型重组合法批次(spec §2 批次契约)

            turn_messages.append(AIMessage(content="".join(text_parts), tool_calls=calls))
            faq_terminal: str | None = None
            for call in calls:
                if faq_terminal is not None:
                    turn_messages.append(ToolMessage(
                        content="知识检索未过闸,本调用未执行", tool_call_id=call["id"],
                        name=call["name"], status="error"))
                    continue
                if call["name"] == "suggest_options":
                    msg, actions = _handle_suggest_options(call, state.get("order_context"))
                    if actions is not None:
                        suggested = actions
                    turn_messages.append(msg)
                    continue
                spec = face.get(call["name"])
                if spec is not None and spec.permission == "write":
                    # 批次契约已保证写调用在最后;park 成 pending_ticket 转确认流(spec §5)
                    pending = PendingWrite(tool_call_id=call["id"], name=call["name"],
                                           args=call["args"],
                                           args_sha256=canonical_args_sha(call["args"]))
                    trace.append({"node": "main_agent", "pending_write": call["name"]})
                    return {"turn_messages": turn_messages,
                            "pending_ticket": {"tool_call_id": pending.tool_call_id,
                                               "name": pending.name, "args": pending.args,
                                               "args_sha256": pending.args_sha256},
                            "agent_steps": steps, "agent_tokens": spent,
                            "token_accounting": accounting,
                            "suggested_actions": suggested, "node_trace": trace}
                writer(ev_tool_start(call))
                outcome = await executor.execute(call, ctx)
                logger.info("node=main_agent tool %s ok=%s err=%s audit=%s",
                            outcome.record.name, outcome.record.ok,
                            outcome.record.error_type, outcome.record.audit_status)
                writer(ev_tool_end(call["id"], call["name"], outcome.record.ok,
                                   outcome.message.content[:80]))
                if outcome.record.error_code:
                    outcome.message.additional_kwargs["error_code"] = outcome.record.error_code
                turn_messages.append(outcome.message)
                if call["name"] == "query_faq" and faq_trace is not None:
                    if outcome.record.error_code or faq_trace.status in (
                            "low_confidence", "unavailable", "tool_error"):
                        faq_terminal = ("refusal" if faq_trace.status == "low_confidence"
                                        and not outcome.record.error_code else "unavailable")
            if faq_terminal is not None:
                return _faq_terminal(faq_terminal)
```
注意:原 `if not calls:` 收尾段在 violation 检查**之前**保持不变(无 calls 直接收敛,不查批次)。

`app/tools/business.py`:删 `build_graph_tools` / `GraphToolset` / `_deterministic_logistics` 及其专属 import(`asdict`、`_orders` 若仅它们用);保留 `MOCK_TOOLS`/`build_tools`/`write_ticket`/`RetrievalTrace`/`TurnToolset` 与 FaqRetrievalTrace re-export 行。grep 兜底:`grep -rn "build_graph_tools\|GraphToolset\|_deterministic_logistics" app/ tests/ evals/`,调用点全部改完(main.py probe、agent_node 已在本任务改;evals/ 若有引用,同步改为 `ToolFace(scan_builtin_specs()).as_langchain_tools()` 形态——evals 手跑不进 pytest,但不得留断 import)。

`app/main.py:134-139` probe 段改:
```python
    from app.graph.agent_node import suggest_options
    from app.services.token_budget import compute_budget, measure_sys_tokens
    from app.tools.builtin import scan_builtin_specs
    from app.tools.executor import ToolFace
    probe_tools = [*ToolFace(scan_builtin_specs()).as_langchain_tools(), suggest_options]
```

- [ ] **Step 4: 跑绿(全量)**

Run: `uv run pytest -q`
Expected: 全 PASS(存量图测试的工具面行为不变;test_chat_api_tools 等帧序不变)

- [ ] **Step 5: Commit**

```bash
git add app/graph/agent_node.py app/tools/business.py app/main.py tests/test_agent_tools_face.py
git commit -m "feat(ch08): agent tool face from catalog, batch contract, write parking"
```

---

### Task 10: ticket_confirm 节点 + builder 接线 + ticket_preview SSE 帧

**Files:**
- Modify: `app/graph/state.py:23,55`(+pending_ticket 字段与重置)
- Modify: `app/graph/nodes.py`(+`route_after_agent` / `build_ticket_confirm_node`;import interrupt/TurnContext/ToolFace/ToolExecutor/PendingWrite)
- Modify: `app/graph/builder.py`(+节点/条件边)
- Modify: `app/services/chat_service.py`(+TicketPreviewEvent/ChatEvent Union/_pending_selector_events 分支)
- Modify: `app/routers/chat.py`(+ticket_preview 帧翻译)
- Test: `tests/test_ticket_confirm.py`(新建,本任务先写挂起帧用例)

**Interfaces:**
- Consumes: Task 9 的 `pending_ticket`;Task 6 的 `execute_confirmed/deny_write`。
- Produces:
  - `route_after_agent(state) -> str`:`"ticket_confirm" if state.get("pending_ticket") else "log"`
  - `build_ticket_confirm_node(deps)` → `ticket_confirm(state, config)`:interrupt 载荷 `{"type":"ticket_preview","tool_call_id","ticket_type","description"}`;resume `{"decision":"confirm"|"cancel"}`;confirm → `execute_confirmed`(checkpoint 快照参数,客户端字段一律不信);其他(含缺失)→ `deny_write`;返回 `{"turn_messages": [...], "pending_ticket": None, "node_trace": ...}`
  - SSE 帧:`{"type":"ticket_preview","interrupt_id","ticket_type","description"}`(与 order_selector 同通道:驱动层扫 pending interrupt)
- 图动线:`main_agent --(pending_ticket)--> ticket_confirm --> main_agent --> log`(无 pending 时 main_agent → log 照旧)

- [ ] **Step 1: 写失败测试**

`tests/test_ticket_confirm.py`:
```python
"""ch08 工单预览(spec §5):挂起轮帧序 session → ticket_preview → [DONE];
确认/取消全链在 Task 11 resume 分派落地后补全。"""
from app.main import create_app
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.test_ch05_acceptance import _client, _turn, _types

_SCRIPTS = [
    ['{"intent":"订单","confidence":0.9}'],          # classify(首轮 understand 跳过)
    [("tool", [{"index": 0, "name": "create_ticket", "id": "w1",
                "args": '{"description":"商品有质量问题,要求换货","ticket_type":"售后"}'}])],
    ["已为您提交售后工单,工单号见上方卡片。"],         # Task 11 接通后 main_agent 收尾
]


async def test_write_call_suspends_with_ticket_preview_frame():
    app = create_app(settings=make_settings(),
                     model=ScriptedChatModel(scripts=list(_SCRIPTS)),
                     runtime=make_runtime(tools=[]))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "帮我建个工单")
    assert _types(frames) == ["session", "ticket_preview", "[DONE]"]
    card = frames[1]
    assert card["interrupt_id"]
    assert card["ticket_type"] == "售后"
    assert "质量问题" in card["description"]
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_ticket_confirm.py -q`
Expected: FAIL(ticket_preview 帧不存在;main_agent 直走 log,卡片不出)

- [ ] **Step 3: 实现**

`app/graph/state.py`:`ChatGraphState` 加 `pending_ticket: dict | None  # ch08:park 的写调用快照(ticket_confirm 消费)`;`new_turn_state` 返回 dict 加 `"pending_ticket": None`。

`app/graph/nodes.py` 追加(import 区加 `from langgraph.types import interrupt`——若已有则不重复;`from app.tools.catalog import TurnContext`;`from app.tools.executor import PendingWrite, ToolExecutor, ToolFace`):
```python
# ── ch08 工单确认(spec §5):main_agent park 写调用 → 预览 interrupt → confirm/cancel ──

def route_after_agent(state) -> str:
    return "ticket_confirm" if state.get("pending_ticket") else "log"


def build_ticket_confirm_node(deps: GraphDeps):
    async def ticket_confirm(state, config):
        pending = state["pending_ticket"]
        trace = [*state["node_trace"], {"node": "ticket_confirm"}]
        # 挂起:帧由驱动层在图结束后按 pending interrupt 发射,节点不发 SSE
        selected = interrupt({"type": "ticket_preview",
                              "tool_call_id": pending["tool_call_id"],
                              "ticket_type": pending["args"].get("ticket_type"),
                              "description": pending["args"].get("description")})
        decision = (selected or {}).get("decision")
        sid = config["configurable"]["thread_id"]
        user_id = (config.get("configurable") or {}).get("user_id", "")
        ctx = TurnContext(user_id=user_id,
                          conversation_id=int(sid) if str(sid).isdecimal() else None,
                          resolved_query=state.get("resolved_query", ""),
                          retriever=None, settings=deps.settings,
                          session_factory=deps.session_factory)
        spec = deps.catalog.get(pending["name"]) if deps.catalog else None
        executor = ToolExecutor(ToolFace([spec] if spec else []), deps.settings,
                                session_factory=deps.session_factory,
                                mcp=deps.mcp_gateway)
        pending_write = PendingWrite(tool_call_id=pending["tool_call_id"],
                                     name=pending["name"], args=pending["args"],
                                     args_sha256=pending["args_sha256"])
        if decision == "confirm":
            # 只信 checkpoint 快照参数;客户端 resume 字段一律不采用(spec §5.3)
            outcome = await executor.execute_confirmed(pending_write, ctx)
        else:
            outcome = await executor.deny_write(pending_write, ctx, "用户取消")
        return {"turn_messages": [*state["turn_messages"], outcome.message],
                "pending_ticket": None, "node_trace": trace}

    return ticket_confirm
```

`app/graph/builder.py`:import 加 `build_ticket_confirm_node, route_after_agent`;构图改:
```python
    g.add_node("main_agent", build_agent_node(deps))
    g.add_node("ticket_confirm", build_ticket_confirm_node(deps))
    g.add_node("log", build_log_node(deps))
    ...
    for node in ("gate_fallback", "complaint_reply", "chitchat_reply",
                 "other_fallback"):
        g.add_edge(node, "log")
    g.add_conditional_edges(
        "main_agent", route_after_agent,
        {"ticket_confirm": "ticket_confirm", "log": "log"})
    g.add_edge("ticket_confirm", "main_agent")
    g.add_edge("log", END)
```

`app/services/chat_service.py`:
```python
@dataclass(frozen=True)
class TicketPreviewEvent:
    """ch08(spec §5):图挂起后绑定 interrupt ID 的工单预览帧。"""
    interrupt_id: str
    ticket_type: str
    description: str
```
`ChatEvent` Union 追加 `TicketPreviewEvent`;`_pending_selector_events` 在 order_selector 分支后加:
```python
                if value.get("type") == "ticket_preview":
                    yield TicketPreviewEvent(interrupt_id=intr.id,
                                             ticket_type=value.get("ticket_type") or "",
                                             description=value.get("description") or "")
                    return
```

`app/routers/chat.py`:import 加 `TicketPreviewEvent`;`_EventStream._body` 在 OrderSelectorEvent 分支后加:
```python
                elif isinstance(event, TicketPreviewEvent):
                    yield _sse({"type": "ticket_preview",
                                "interrupt_id": event.interrupt_id,
                                "ticket_type": event.ticket_type,
                                "description": event.description})
```

- [ ] **Step 4: 跑绿(全量)**

Run: `uv run pytest -q`
Expected: 全 PASS(新挂起帧用例通过;存量不受影响——无脚本触发 write 时 main_agent → log 照旧)

- [ ] **Step 5: Commit**

```bash
git add app/graph/state.py app/graph/nodes.py app/graph/builder.py app/services/chat_service.py app/routers/chat.py tests/test_ticket_confirm.py
git commit -m "feat(ch08): ticket_confirm node with preview interrupt + SSE frame"
```

---

### Task 11: resume 契约扩展 + 确认/取消全链

**Files:**
- Modify: `app/schemas.py:75-90`(ChatResumeRequest 放宽)
- Modify: `app/services/chat_service.py:156-186`(prepare_resume 按 interrupt 类型分派)
- Modify: `app/routers/chat.py:102-112`(resume 传 order_id/decision)
- Test: `tests/test_ticket_confirm.py`(追加全链用例)

**Interfaces:**
- Consumes: Task 10 的 ticket_preview 挂起;Task 6 execute_confirmed/deny_write;Task 1 双表。
- Produces: `ChatService.prepare_resume(user_id, session_id, interrupt_id, order_id=None, decision=None)`;契约(spec §5.3):order_selector → 必须 order_id(候选内 + 归属 404,旧契约逐字不变);ticket_preview → 必须 decision ∈ {confirm,cancel};类型不明/卡片失效/字段不对 → 409;归属错误 → 404。
- 铁律:已消费/不存在的 interrupt_id 一律 409,即使幂等表有记录(幂等表只供内部恢复,spec §3/§5.3)。

- [ ] **Step 1: 写失败测试(追加进 tests/test_ticket_confirm.py)**

```python
import json

from app.models import Ticket, ToolAuditLog, ToolWriteIdempotency
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _app(scripts, db_sf=None):
    import dataclasses
    runtime = make_runtime(tools=[])
    if db_sf is not None:
        runtime = dataclasses.replace(runtime, session_factory=db_sf)
    return create_app(settings=make_settings(),
                      model=ScriptedChatModel(scripts=scripts), runtime=runtime)


def _parse(resp):
    return [json.loads(l[5:]) if l.startswith("data:") and l[5:].strip() != "[DONE]"
            else "[DONE]" for l in resp.text.splitlines() if l.strip()]


async def _suspend(client):
    frames, sid = await _turn(client, "帮我建个工单")
    card = frames[1]
    assert card["type"] == "ticket_preview"
    return sid, card["interrupt_id"]


async def test_confirm_lands_ticket_idempotency_audit(db_session_factory):
    app = _app(list(_SCRIPTS), db_sf=db_session_factory)
    async with await _client(app) as client:
        sid, iid = await _suspend(client)
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "confirm"})
        assert resp.status_code == 200
        frames2 = _parse(resp)
        types2 = [f if isinstance(f, str) else f["type"] for f in frames2]
        assert types2[0] == "session" and types2[-1] == "[DONE]"
        assert "ticket_preview" not in types2          # 恢复成功不重发卡
        text = "".join(f.get("content", "") for f in frames2
                       if isinstance(f, dict) and f["type"] == "delta")
        assert "工单" in text
    with db_session_factory() as s:
        tickets = s.query(Ticket).all()
        assert len(tickets) == 1 and tickets[0].ticket_type == "售后"
        idem = s.query(ToolWriteIdempotency).all()
        assert len(idem) == 1 and idem[0].ticket_no == tickets[0].ticket_no
        audits = s.query(ToolAuditLog).filter_by(tool_name="create_ticket").all()
        assert [a.status for a in audits] == ["成功"]
        assert audits[0].conversation_id == int(sid)
        # 旧卡重放 → 409(幂等表有记录也不放行,spec §5.3)
        resp2 = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "confirm"})
        assert resp2.status_code == 409


async def test_cancel_writes_denied_audit_no_ticket(db_session_factory):
    app = _app(list(_SCRIPTS), db_sf=db_session_factory)
    async with await _client(app) as client:
        sid, iid = await _suspend(client)
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "cancel"})
        assert resp.status_code == 200
        _parse(resp)
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 0
        assert s.query(ToolWriteIdempotency).count() == 0
        audits = s.query(ToolAuditLog).filter_by(tool_name="create_ticket").all()
        assert [a.status for a in audits] == ["权限拒绝"]
        assert audits[0].arguments["description"] == "商品有质量问题,要求换货"


async def test_ticket_preview_resume_requires_decision(db_session_factory):
    app = _app(list(_SCRIPTS), db_sf=db_session_factory)
    async with await _client(app) as client:
        sid, iid = await _suspend(client)
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid, "interrupt_id": iid})
        assert resp.status_code == 409          # 字段不对按契约 409(spec §5.3)


async def test_order_selector_contract_unchanged(db_session_factory):
    """旧契约逐字保持:order_selector 必须 order_id;decision 不顶用。"""
    scripts = [
        ['{"intent":"退款退货","confidence":0.9}'],
        ['{"mode":"order_specific"}'],
        ['{"queries":["保温杯退货期限"]}'],
        ["订单 1111-1001 可以退[1]。"],
    ]
    from tests.test_ch05_acceptance import _FakeRetriever, _result
    import dataclasses
    runtime = dataclasses.replace(make_runtime(tools=[]),
                                  retriever=_FakeRetriever(_result()))
    app = create_app(settings=make_settings(),
                     model=ScriptedChatModel(scripts=scripts), runtime=runtime)
    async with await _client(app) as client:
        frames, sid = await _turn(client, "这个能退吗")
        assert _types(frames) == ["session", "order_selector", "[DONE]"]
        iid = frames[1]["interrupt_id"]
        resp = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "decision": "confirm"})   # 缺 order_id → 409
        assert resp.status_code == 409
        resp2 = await client.post("/v1/chat/resume", json={
            "user_id": TEST_USER_ID, "session_id": sid,
            "interrupt_id": iid, "order_id": "1111-1001"}) # 旧路径照常放行
        assert resp2.status_code == 200


async def test_ticket_confirm_survives_sqlite_reopen(db_session_factory, tmp_path):
    """SQLite 文件级持久化钉(照 test_resume_survives_sqlite_reopen 先例):
    挂起 → 关库重开新 saver → confirm 仍放行落票。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command
    from app.graph.builder import build_chat_graph
    from app.graph.nodes import GraphDeps
    from app.graph.state import new_turn_state
    from app.store_db import DbSessionStore
    from app.tools.builtin import scan_builtin_specs
    from app.tools.catalog import ToolCatalog
    catalog = ToolCatalog()
    for s in scan_builtin_specs():
        catalog.register(s)
    settings = make_settings()
    store = DbSessionStore(db_session_factory, settings.max_message_chars)
    sid = await store.create(TEST_USER_ID)
    scripts = [
        ['{"intent":"订单","confidence":0.9}'],
        [("tool", [{"index": 0, "name": "create_ticket", "id": "w1",
                    "args": '{"description":"商品有质量问题,要求换货","ticket_type":"售后"}'}])],
        ["已为您提交售后工单。"],
    ]
    deps = GraphDeps(model=ScriptedChatModel(scripts=scripts), settings=settings,
                     retriever=None, store=store, system_prompt="",
                     catalog=catalog, mcp_gateway=None,
                     session_factory=db_session_factory)
    db = str(tmp_path / "cp.db")
    config = {"configurable": {"thread_id": sid, "user_id": TEST_USER_ID}}
    iid = None
    async with AsyncSqliteSaver.from_conn_string(db) as cp:
        graph = build_chat_graph(deps, cp)
        async for _ in graph.astream(new_turn_state("帮我建个工单"), config,
                                     stream_mode=["custom"]):
            pass
        st = await graph.aget_state(config)
        for task in st.tasks:
            for intr in task.interrupts:
                if (getattr(intr, "value", None) or {}).get("type") == "ticket_preview":
                    iid = intr.id
    assert iid
    async with AsyncSqliteSaver.from_conn_string(db) as cp2:   # 关库重开
        graph2 = build_chat_graph(deps, cp2)
        async for _ in graph2.astream(Command(resume={iid: {"decision": "confirm"}}),
                                      config, stream_mode=["custom"]):
            pass
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 1
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_ticket_confirm.py -q`
Expected: 前 5 条 FAIL(resume 不识别 ticket_preview / decision 字段不存在)

- [ ] **Step 3: 实现**

`app/schemas.py` ChatResumeRequest 替换:
```python
class ChatResumeRequest(BaseModel):
    """ch06(spec §9.1)+ ch08(spec §5.3):按 interrupt ID 恢复挂起轮。
    order_selector 必须带 order_id;ticket_preview 必须带 decision(服务端锁内分派)。"""
    user_id: str
    session_id: str
    interrupt_id: str = Field(min_length=1, max_length=128)
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    decision: Literal["confirm", "cancel"] | None = None

    @field_validator("user_id")
    @classmethod
    def _user_id_must_be_uuid(cls, v: str) -> str:
        return _validate_uuid(v)

    @field_validator("session_id")
    @classmethod
    def _session_id_form(cls, v: str | None) -> str | None:
        return _validate_session_id(v)
```

`app/services/chat_service.py` prepare_resume 替换(签名与分派):
```python
    async def prepare_resume(self, user_id: str, session_id: str,
                             interrupt_id: str, order_id: str | None = None,
                             decision: str | None = None) -> PreparedTurn:
        """ch06(spec §9.1)+ ch08(spec §5.3):锁内核验 pending interrupt 并按类型分派。
        409 必须服务端自校验——LangGraph 对不匹配 resume ID 是静默忽略,不报错。
        已消费/不存在的 interrupt_id 一律 409,不查幂等表(它只供内部恢复)。"""
        from langgraph.types import Command
        from app.services.orders import get_order
        if not await self._store.exists(session_id, user_id):
            raise SessionNotFoundError("session not found")
        await self._locks.acquire(session_id)
        try:
            config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
            st = await self._graph.aget_state(config)
            match = None
            for task in st.tasks:
                for intr in task.interrupts:
                    if intr.id == interrupt_id:
                        match = intr
            value = getattr(match, "value", None) or {}
            itype = value.get("type")
            if match is None or itype not in ("order_selector", "ticket_preview"):
                raise ResumeConflictError("no matching pending interrupt")
            if itype == "order_selector":
                if not order_id:
                    raise ResumeConflictError("order_selector requires order_id")
                candidates = {o.get("order_id") for o in value.get("orders") or []}
                if order_id not in candidates:
                    raise ResumeConflictError("order not in pending candidates")
                if get_order(user_id, order_id) is None:
                    raise SessionNotFoundError("session not found")  # 404 不泄露
                resume_value = {"order_id": order_id}
            else:
                if decision not in ("confirm", "cancel"):
                    raise ResumeConflictError("ticket_preview requires decision")
                resume_value = {"decision": decision}  # 只带决策;参数只信快照
        except Exception:
            self._locks.release(session_id)
            raise
        return PreparedTurn(session_id=session_id, user_text="", lock_key=session_id,
                            user_id=user_id,
                            resume_command=Command(resume={interrupt_id: resume_value}))
```

`app/routers/chat.py` chat_resume:
```python
    prepared = await service.prepare_resume(body.user_id, body.session_id,
                                            body.interrupt_id, body.order_id,
                                            body.decision)
```

- [ ] **Step 4: 跑绿(全量)**

Run: `uv run pytest -q`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py app/services/chat_service.py app/routers/chat.py tests/test_ticket_confirm.py
git commit -m "feat(ch08): resume dispatch by interrupt type, confirm/cancel full chain"
```

---

### Task 12: 每轮动态预算 + budget_priority 裁剪

**Files:**
- Modify: `app/graph/agent_node.py`(预算段;新增模块级 `fit_face_to_budget` 与 `_BUDGET_CACHE`)
- Test: `tests/test_agent_tools_face.py`(追加)

**Interfaces:**
- Consumes: ToolFace/Task 9 装配段;`measure_sys_tokens/compute_budget`;`ContextBudget.sufficient`。
- Produces:
  - `fit_face_to_budget(face, extra_tools, settings, system_prompt, abort) -> tuple[ToolFace, list, ContextBudget, int]`:`tools = [*face.as_langchain_tools(), *extra_tools]`;按 tools 实测 sys_tokens(签名缓存,`_BUDGET_CACHE` 上限 32 条,满即清);`budget.sufficient` 为假 → 按 `(budget_priority, name)` 升序剔 `budget_priority < 100` 的条目(剔一个算一次,逐条记 warning);无可剔仍不足 → `abort("tool_context_too_long", "工具面超出上下文预算")` 抛出,不得带超预算工具面调模型(spec §1.5)。返回 `(face, tools, budget, schema_est)`
  - main_agent 内:`tools/budget/schema_est` 全部由它产出;`build_agent_context` 的 guard 改为 `settings.model_context_window - settings.max_output_tokens - schema_est`(schema 感知窗口,clarify 时 schema_est=0 行为同旧)
- 缓存键 = `tuple(sorted(t.name for t in tools))`(含 suggest_options);sys_tokens 与 budget 一起缓存。

- [ ] **Step 1: 写失败测试**

`tests/test_agent_tools_face.py` 追加:
```python
import pytest
from pydantic import BaseModel, Field

from app.tools.catalog import ToolSpec
from app.tools.executor import ToolFace
from app.graph.agent_node import fit_face_to_budget
from app.tools.builtin import scan_builtin_specs


class _Big(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)


def _fat_mcp(name: str, chars: int) -> ToolSpec:
    return ToolSpec(name=name, description="长" * chars,
                    args_schema=_Big.model_json_schema(), args_model=_Big,
                    source="mcp", permission="read", ownership="order",
                    mcp_server="logistics", budget_priority=10)


def test_fit_trims_low_priority_first():
    core = [s for s in scan_builtin_specs() if s.name in ("query_order", "query_product")]
    ticket = [s for s in scan_builtin_specs() if s.name == "create_ticket"]
    face = ToolFace([*core, *ticket, _fat_mcp("query_logistics", 3000)])
    aborts = []
    new_face, tools, budget, schema_est = fit_face_to_budget(
        face, [], make_settings(), "sys", lambda c, m: aborts.append(c))
    names = [s.name for s in new_face.specs]
    assert "query_logistics" not in names          # 最低优先级先剔
    assert {"query_order", "query_product"} <= set(names)  # 核心必留
    assert not aborts


def test_fit_aborts_when_core_alone_overflows():
    core = [s for s in scan_builtin_specs() if s.name == "query_order"]
    face = ToolFace(core)
    settings = make_settings(model_context_window=120)  # 连核心都装不下

    def _abort(code, msg):
        raise RuntimeError(code)

    with pytest.raises(RuntimeError, match="tool_context_too_long"):
        fit_face_to_budget(face, [], settings, "sys" * 500, _abort)
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest tests/test_agent_tools_face.py -q`
Expected: FAIL(fit_face_to_budget 不存在)

- [ ] **Step 3: 实现**

`app/graph/agent_node.py` 模块级(main_agent 外):
```python
_BUDGET_CACHE: dict[tuple, tuple[int, object]] = {}  # 工具签名 → (sys_tokens, ContextBudget)


def _face_budget(settings, system_prompt: str, tools: list):
    sig = tuple(sorted(t.name for t in tools))
    hit = _BUDGET_CACHE.get(sig)
    if hit is None:
        sys_tokens = measure_sys_tokens(system_prompt, tools)
        hit = (sys_tokens, compute_budget(settings, sys_tokens))
        if len(_BUDGET_CACHE) >= 32:
            _BUDGET_CACHE.clear()
        _BUDGET_CACHE[sig] = hit
    return hit


def fit_face_to_budget(face, extra_tools: list, settings, system_prompt: str, abort):
    """ch08 动态预算(spec §1.5):按本轮实际工具面重算;不足按 budget_priority 升序
    剔非核心条目;核心仍装不下经 abort 抛出,不带超预算工具面调模型。"""
    def _tools() -> list:
        return [*face.as_langchain_tools(), *extra_tools] if face.specs else list(extra_tools)

    tools = _tools()
    if not tools:
        budget = compute_budget(settings, measure_sys_tokens(system_prompt, tools))
        return face, tools, budget, 0
    sys_tokens, budget = _face_budget(settings, system_prompt, tools)
    while not budget.sufficient:
        removable = sorted((s for s in face.specs if s.budget_priority < 100),
                           key=lambda s: (s.budget_priority, s.name))
        if not removable:
            abort("tool_context_too_long", "工具面超出上下文预算")
            raise RuntimeError("abort must raise")  # 防御:abort 契约是抛出不返回
        drop = removable[0]
        logger.warning("tool face 超预算,剔除 %s(priority=%d)",
                       drop.name, drop.budget_priority)
        face = ToolFace([s for s in face.specs if s.name != drop.name])
        tools = _tools()
        sys_tokens, budget = _face_budget(settings, system_prompt, tools)
    return face, tools, budget, sys_tokens
```

main_agent 装配段尾部(Task 9 的 `tools = ...` / `budget = ...` / `schema_est = ...` 三行)替换为:
```python
        def _face_abort(code: str, msg: str):
            writer(ev_error(code, msg))
            raise TurnAbortError(code)

        face, tools, budget, schema_est = fit_face_to_budget(
            face, [suggest_options] if face.specs else [], settings, deps.system_prompt,
            _face_abort)
        model = deps.model.bind_tools(tools) if tools else deps.model
```
注意 schema_est 语义:`fit_face_to_budget` 第四返回值是「仅工具 schema」(`measure_sys_tokens("", tools)`),不是缓存里的 system+tools——`est_input = estimate_messages(context) + schema_est` 行的口径与旧实现逐字一致;缓存的 sys_tokens 只喂 compute_budget。实现里第四返回值写作:
```python
    return face, tools, budget, measure_sys_tokens("", tools)   # 第四值:仅工具 schema
```
`build_agent_context(...)` 调用处 guard 改:
```python
            context = build_agent_context(deps.system_prompt, layer2, layer1,
                                          turn_messages, background_text,
                                          settings.model_context_window
                                          - settings.max_output_tokens - schema_est)
```

- [ ] **Step 4: 跑绿(全量)**

Run: `uv run pytest -q`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/agent_node.py tests/test_agent_tools_face.py
git commit -m "feat(ch08): per-turn dynamic budget with priority-based tool trimming"
```

---

### Task 13: prompt 条款改写 + 契约同步 + 预算钉核查

**Files:**
- Modify: `app/prompts/service.py`(条款 7)
- Modify: `tests/test_prompts.py`(若有条款 7 钉则同步)
- Modify: `tests/test_token_budget.py` + `AGENTS.md`(仅当实测 SYS_TOKENS 变化时)

**Interfaces:**
- Produces: 新条款 7 文案(create_ticket 直接发起 + 预览确认语义);其余条款不动。

- [ ] **Step 1: 核查现状钉**

Run: `grep -n "建工单\|suggest_options" tests/test_prompts.py`;`uv run pytest tests/test_prompts.py -q`
Expected: 确认条款 7 是否被逐字钉住。

- [ ] **Step 2: 改 prompt**

`app/prompts/service.py` 条款 7 替换为:
```
7. 顾客明确要求人工、或问题超出工具能力时,调用 suggest_options 请用户点击对应按钮;顾客明确要求建工单时,先核对工单类型(售后/投诉/咨询)与问题描述是否齐全,不齐先追问、不得编造,齐全后调用 create_ticket——系统会向顾客展示工单预览,顾客确认后才真正建单:确认成功你在答复中如实告知系统返回的工单号,顾客取消则如实说明未建单;你不得虚构工单号,也不得声称已经转人工。
```

- [ ] **Step 3: 同步契约 + 跑绿**

Run: `uv run pytest tests/test_prompts.py tests/test_token_budget.py -q`
Expected: PASS;若条款 7 逐字钉红,按新文案更新钉。
预算钉核查:`uv run uvicorn --factory app.main:create_app`(需 Docker/.env 在线)或跑一条带 runtime 的测试观察 `context budget` 日志行 `sys=` 值;若实测 SYS_TOKENS 与钉的 1901 不同(工具面已变:query_logistics 出、create_ticket 入、条款 7 变长),把 `tests/test_token_budget.py` 的 sys_tokens=1901 与三数钉按 `compute_budget(settings, 新实测值)` 重算更新,并同步 AGENTS.md「验收数字」行;未变则不动。

- [ ] **Step 4: Commit**

```bash
git add app/prompts/service.py tests/test_prompts.py tests/test_token_budget.py AGENTS.md
git commit -m "feat(ch08): prompt rules for create_ticket confirmation flow"
```

---

### Task 14: 前端工单预览卡(Vibe Coding 例外)

**Files:**
- Modify: `app/static/chat.html`
- Modify: `tests/test_chat_page.py`

**Interfaces:**
- SSE 帧 `{"type":"ticket_preview","interrupt_id","ticket_type","description"}`(Task 10 已发);resume POST /v1/chat/resume 正文 `{user_id, session_id, interrupt_id, decision}`(Task 11 已收)。
- 不做 TDD/brainstorm/review(用户明确 Vibe Coding 例外);验收靠 test_chat_page 字符串断言 + Task 15 人工点验。

- [ ] **Step 1: 实现(照订单卡模式,锚点已核)**

- `readChatStream`(chat.html:1286-1331)加分支:`evt.type === "ticket_preview"` → `renderTicketPreview(ui, evt)`,置 `suspended = true`(比照 order_selector 分支写法)。
- 新函数 `renderTicketPreview(ui, evt)`(比照 `renderOrderCards` chat.html:1037-1075):卡片渲染进当前助手纵列,标题「工单预览」,两行字段(工单类型 / 问题描述),按钮组「确认提交」「取消」;闭包捕获 `{user_id, session_id, interrupt_id}`(不得从全局当前会话猜,比照 :1046 注释)。
- 新函数 `decideTicket(captured, decision)`(比照 `pickOrder` :1077):全组 disable → `addSystemHint(decision === "confirm" ? "已确认提交工单" : "已取消建单")` → `setStreaming(true)` → `resumeStream(captured, {decision})`。
- `resumeStream(captured, extra)`(:1088-1104)签名放宽:正文 = `{user_id, session_id, interrupt_id, ...extra}`;订单卡调用点改传 `{order_id}`(行为不变);409 提示沿用「该选择已失效,请重新描述您的问题」。
- 失效清扫:`expireOrderCards()`(:1031)同法加 `expireTicketCards()`(或给两类卡根节点加共同 class 一次扫净),发新消息时调用(:1338 附近);历史回载不带卡(现状语义)。
- `TOOL_LABELS`(:832)补:`query_warranty: "查在保"`、`query_return_progress: "查退货进度"`、`create_ticket: "建工单"`(`query_logistics` 若已有不动)。
- 样式:复用 `.order-cards/.order-card` 既有 design token,可加 `.ticket-card` 微调;模型输出不进 innerHTML 的铁律不破(字段用 textContent)。

- [ ] **Step 2: 断言钉**

`tests/test_chat_page.py` 的 `test_chat_page_order_selector_and_refund_form` 同法加新用例:
```python
def test_chat_page_ticket_preview_card():
    from pathlib import Path
    html = (Path(__file__).parent.parent / "app" / "static" / "chat.html").read_text(encoding="utf-8")
    for needle in ("ticket_preview", "renderTicketPreview", "decideTicket",
                   "确认提交", "取消", "decision", "/v1/chat/resume",
                   "查在保", "查退货进度"):
        assert needle in html
```

- [ ] **Step 3: 跑绿**

Run: `uv run pytest tests/test_chat_page.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/static/chat.html tests/test_chat_page.py
git commit -m "feat(ch08): ticket preview card with confirm/cancel buttons"
```

---

### Task 15: 收尾(Makefile/README/AGENTS.md/全量/验收演练)

**Files:**
- Modify: `Makefile`、`README.md`、`AGENTS.md`

- [ ] **Step 1: Makefile + README**

`Makefile` 追加(先读现有风格,保持一致):
```make
mcp-logistics: ## 起物流 MCP Server(:8101)
	uv run python -m app.mcp_servers.logistics_server

mcp-after-sales: ## 起售后 MCP Server(:8102)
	uv run python -m app.mcp_servers.after_sales_server
```
`README.md` 加「MCP Server」节:两 Server 起法、`.env` 的 `MCP_LOGISTICS_URL` / `MCP_AFTER_SALES_URL`、主服不重启动态发现说明、`curl --noproxy '*'` 红线提醒。

- [ ] **Step 2: AGENTS.md 更新**

- 红线段改写:「建工单唯一通道 POST /v1/chat/action」→ 双通道:动作端点(前端按钮,点按钮即确认)与图内 create_ticket 确认流(模型发起 → ticket_preview interrupt → resume decision);`query_logistics` 内置已下线,物流由 MCP Server 接管;审计双红线(写失败不拦执行/不挂外键);MCP 权限只认本地能力策略(两形态,未知 Server 默认 deny)。
- 目录与约定段:db/init/06-ddl.sql ↔ sql/ch08-ddl.sql 钉;app/tools/{catalog,executor,builtin/}、app/services/{tool_audit,mcp_gateway}.py、app/mcp_servers/ 职责一句话。
- 怎么跑段:MCP Server 起法 + .env 新字段。
- 当前状态段:ch08 摘要 + 测试计数(按 Step 3 实测更新)。

- [ ] **Step 3: 全量验证**

Run: `uv run pytest -q`
Expected: 全 PASS;记录总数。

- [ ] **Step 4: 六条验收演练(人工,交付时逐条过)**

按 spec §6.3:①现场新写 `query_store_hours`(app/tools/builtin/ 新文件 + @builtin_tool,重启服,对话可用);②双 Server 起进程问物流轨迹;③after_sales 加零参数演示工具(如 `after_sales_hotline`),只重启该 Server,再问即用;④「帮我建个工单」→ 追问补齐 → 预览卡 → 确认 → tickets 落行 + 答复带单号;⑤同路点取消 → 无新行 + 审计权限拒绝;⑥`TOOL_TIMEOUT_OVERRIDES={"query_product":0.001}` 起服问商品 → 审计 超时/retry=2/耗时;再 `{"create_ticket":0.001}` 走确认流 → 审计 超时/retry=0(tickets 可能仍落行,演示「超时未必没执行」)。演示输出整理进交付回复。

- [ ] **Step 5: Commit**

```bash
git add Makefile README.md AGENTS.md
git commit -m "docs(ch08): runbook for MCP servers + agent rules for tool system"
```

---

## Self-Review 记录(2026-10-07)

- **Spec 覆盖**:§1 目录/TurnContext/分支矩阵/启动接线 → T4/T5/T8/T9;§1.5 动态预算 → T12;§2 六步/批次契约/execute_confirmed/超时优先级 → T6/T9;§3 审计+幂等 → T1/T3/T6/T11;§4 双 Server/网关/依赖钉 → T2/T7/T8;§5 确认流/resume/前端 → T10/T11/T14;§6 配置/测试/验收 → T2/T6-T12/T15;§7 风险 → SSE 四处 T10 逐一列出、resume 旧契约钉 T11、写超时语义 T6、轨迹不自洽属「本章不做」已在 spec 记档。
- **类型一致**:ToolSpec(T4)= T5 装饰器产物 = T6 ToolFace 元素 = T8 discover 产物;TurnContext(T4)= T5/T6/T9/T10 同签名;PendingWrite(T6)= T9 park → T10 重建;prepare_resume(T11 签名)= routers 调用;fit_face_to_budget(T12)第四返回值在任务内修正为「仅工具 schema」并已写明修正原因。
- **占位扫描**:无 TBD/TODO;所有测试/实现代码块完整;MCP v1 API 的「以安装包类型定义为准」是 spec 明示的钉法(冒烟测试即钉),非占位。
