# Ch02 Function Calling 工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Ch01 纯对话客服上接入 Function Calling 工具链:Docker MySQL 四表、五个 @tool 业务工具、单轮工具编排、SSE 工具帧与落库、聊天页工具徽章。

**Architecture:** FastAPI + SQLAlchemy 2.0 同步 Session(线程内创建销毁,`asyncio.to_thread` 接入异步流程);手写单轮编排(第一次 bind_tools 聚合 tool_calls → 顺序执行 → 第二次不绑工具收敛);DDL 文件经 MySQL initdb 首启执行,是建表唯一事实源;会话存储扩展为可持久化完整工具轨迹。

**Tech Stack:** Python 3.12(uv)、FastAPI 0.141、SQLAlchemy>=2.0、PyMySQL、MySQL 8.4(Docker)、langchain-core 1.6.2 / langchain-openai 1.6.1、pytest 9.1。

**Spec:** docs/superpowers/specs/2026-09-10-ecommerce-customer-service-ch02-design.md

## Global Constraints

- 技术栈定死:FastAPI + SQLAlchemy + MySQL(Docker)+ LangChain `@tool`;发现矛盾或走不通,停下来问用户,不自行换方案
- 动手前用 Context7 核对库接口;已核对记录见 spec §13 与本计划「API 核对记录」
- `db/init/01-ddl.sql` 是用户手写 DDL(`sql/ch02-ddl.sql` 原样复制),本章不得改动其字段定义;ORM 与之逐字段一致
- 四张表:conversations(id BIGINT 自增 / user_id VARCHAR(64) / status 中文枚举) / messages(role ENUM user,assistant,tool / content 可空 / tool_calls JSON / tool_call_id VARCHAR(64)) / faq / tickets(ticket_no 主键 / ticket_type ENUM('售后','投诉','咨询') / status ENUM('待处理','已处理'))
- seed 只灌 faq ≥8 条;含「退货政策」可命中条目;**全表不出现「邮费」二字**(验收 3 的漏召回是预期结果)
- 单轮调用:第二次模型调用不得绑定工具;`MAX_TOOL_CALLS_PER_TURN=5`
- 只有只读工具(query_order/query_product/query_logistics/query_faq)享受超时重试(wait_for + 最多 2 次额外重试,退避 `0.5*2^(n-1)` 秒);`create_ticket` 独立事务、不 wait_for、不自动重试,同事务把 conversation 状态改为「已转人工」
- 工具结果落库格式:tool 行 content = JSON envelope `{"v":1,"ok":bool,"content"|"error_code","message",...}`,envelope 序列化 ≤ `MAX_TOOL_RESULT_CHARS`(默认 4000,≥256),超出截断 content 并标 `truncated=true`
- 原子提交分界:除已独立提交的工单外,任何 messages 提交前错误/取消都不保存半个 turn;最终 commit 用 `asyncio.shield` 保护,取消期间等事务落地再释放锁
- SSE 帧顺序:session → delta(first)* → (tool_start → tool_end)* → delta(second)* → [DONE];error 后不发 [DONE];公开错误脱敏,日志只记异常类型
- 请求协议:`{"user_id": 规范UUID, "session_id": 可选(正十进制数字串|规范UUID), "message": 非空白}`;user 不匹配统一 404
- 聊天页走 Vibe Coding:不套 brainstorm/TDD/code review;但其依赖的 SSE 帧契约与 user_id 协议以 spec 为准,页面静态断言测试保留
- 本章不做:多轮 Agent Loop、向量检索/RAG、真实电商/物流对接、正式身份认证、Alembic
- 每个任务结束跑 `uv run pytest` 保持绿并单独 commit;Ch01 测试行为断言保留,fixture/调用签名允许按本计划适配

## API 核对记录(2026-09-10,已对锁定版本实测)

- `langchain-core==1.6.2` 实测:`from langchain_core.tools import tool`;`tool.invoke({"name","args","id","type":"tool_call"})` 直接返回 `ToolMessage`(status="success", content=工具返回, tool_call_id 对号入座);非法参数抛 `pydantic.ValidationError`;`AIMessageChunk` 支持 `+` 聚合,聚合后 `.tool_calls` 得到 `[{"name","args","id","type"}]`(本计划编排层自行合并 tool_call_chunks,见 T5,不依赖该 `+`)
- Context7:`bind_tools` / 手动执行循环(https://docs.langchain.com/oss/python/langchain/models);`ToolMessage` 三要素 content/tool_call_id/name(https://docs.langchain.com/oss/python/langchain/messages);流中 tool_call_chunks 渐进到达需按 index 聚合(https://docs.langchain.com/oss/python/langchain/streaming)
- SQLAlchemy 2.0:`mysql+pymysql://user:pass@host/db?charset=utf8mb4`(https://docs.sqlalchemy.org/en/20/dialects/mysql.html);`DeclarativeBase` + `Mapped`/`mapped_column`;`sessionmaker(bind=engine)`;Session 不跨线程共享,每线程独立创建/关闭(https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- 启动:`uv run uvicorn app.main:create_app --factory`,删除模块级 `app = create_app()`

---

### Task 1: 依赖、Docker、DDL/seed、Settings、db.py

**Files:**
- Modify: `pyproject.toml`(加依赖)
- Create: `docker-compose.yml`、`db/init/01-ddl.sql`、`db/init/02-seed.sql`
- Modify: `app/config.py`(新增 6 个字段、删偶数校验)
- Create: `app/db.py`
- Modify: `tests/conftest.py`(make_settings 加占位 database_url)
- Modify: `tests/test_config.py`(新增校验用例)

**Interfaces:**
- Consumes: 现有 `Settings`、`make_settings`
- Produces: `Settings.database_url / test_admin_database_url / test_database_url / tool_timeout_seconds / tool_max_retries / max_tool_calls_per_turn / max_tool_result_chars`;`app.db.make_engine(url) -> Engine`、`make_session_factory(engine) -> sessionmaker`、`ping(engine) -> None`

- [ ] **Step 1: 加依赖并锁定**

`pyproject.toml` dependencies 追加:

```toml
  "sqlalchemy>=2.0",
  "pymysql>=1.1",
  "cryptography>=43",
```

Run: `uv sync` —— 通过,`uv.lock` 更新。

- [ ] **Step 2: docker-compose.yml + DDL 落位**

```bash
mkdir -p db/init && cp sql/ch02-ddl.sql db/init/01-ddl.sql
```

`docker-compose.yml`:

```yaml
services:
  mysql:
    image: mysql:8.4
    container_name: wayhelp-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root-password
      MYSQL_DATABASE: wayhelp
      MYSQL_USER: wayhelp
      MYSQL_PASSWORD: wayhelp
      MYSQL_ROOT_HOST: "%"
    ports:
      - "3306:3306"
    volumes:
      - ./db/init:/docker-entrypoint-initdb.d:ro
      - mysql-data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "127.0.0.1", "-proot-password"]
      interval: 5s
      timeout: 3s
      retries: 30

volumes:
  mysql-data:
```

Run: `docker compose up -d && docker compose ps` —— `wayhelp-mysql` healthy。

验证四表已建:

```bash
docker exec wayhelp-mysql mysql -uwayhelp -pwayhelp wayhelp -e "show tables;"
```

预期:conversations / faq / messages / tickets。

- [ ] **Step 3: 02-seed.sql(faq ≥8 条,无「邮费」)**

`db/init/02-seed.sql`:

```sql
-- faq 测试数据;验收 2 要求「退货政策」可命中;全表不得出现「邮费」二字(验收 3 预期漏召回)
INSERT INTO faq (question, answer, category) VALUES
('退货政策是什么', '支持七天无理由退货:签收后 7 天内、商品完好可申请;运费险订单退货由保险公司承担首重运费。定制类、生鲜类商品不支持无理由退货。', '退货'),
('怎么申请退货', '在「我的订单」找到对应订单,点击「申请售后」选择退货退款,按提示填写原因并提交,审核通过后寄回商品。', '退货'),
('退款多久到账', '退货签收并验收无误后 1-3 个工作日原路退回;银行卡支付以银行到账时间为准。', '退货'),
('换货流程怎么走', '签收后 15 天内可申请换货,在订单页申请售后选择换货;换货商品发出后可在订单中查看新物流。', '换货'),
('发票怎么开', '下单时或订单完成后 30 天内可在订单详情页申请电子发票,抬头支持个人与企业;电子发票 24 小时内发送至预留邮箱。', '发票'),
('运费怎么算', '单笔订单实付满 99 元包邮,未满收 8 元基础运费;偏远地区(新疆、西藏等)运费以结算页为准。', '运费'),
('什么时候发货', '工作日 16 点前的订单当天发出,之后次日发出;预售商品以商品页标注的发货时间为准。', '运费'),
('忘记密码怎么办', '登录页点击「忘记密码」,通过绑定手机号接收验证码重置;手机号已停用请联系人工客服核实身份后处理。', '账户'),
('会员有什么权益', '会员享 95 折、每月 3 张免运费券、生日双倍积分;积分可在结算时抵扣,100 积分抵 1 元。', '账户'),
('如何修改收货地址', '未发货订单可在订单详情直接修改;已发货订单请联系人工客服尝试拦截,拦截成功与否以仓库反馈为准。', '账户');
```

注意:已核实无「邮费」字面。重建容器卷使 initdb 重跑(首启才执行):

```bash
docker compose down -v && docker compose up -d
docker exec wayhelp-mysql mysql -uwayhelp -pwayhelp wayhelp -e "select count(*) from faq;"
```

预期:10。

- [ ] **Step 4: Settings 扩展(先写失败测试)**

`tests/test_config.py` 追加:

```python
def test_odd_max_messages_allowed():
    s = make_settings(max_messages_per_session=7)
    assert s.max_messages_per_session == 7


def test_tool_settings_defaults():
    s = make_settings()
    assert s.tool_timeout_seconds == 5
    assert s.tool_max_retries == 2
    assert s.max_tool_calls_per_turn == 5
    assert s.max_tool_result_chars == 4000


def test_tool_max_retries_zero_allowed():
    assert make_settings(tool_max_retries=0).tool_max_retries == 0


def test_bad_tool_settings_rejected():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        make_settings(tool_timeout_seconds=0)
    with pytest.raises(pydantic.ValidationError):
        make_settings(tool_max_retries=-1)
    with pytest.raises(pydantic.ValidationError):
        make_settings(max_tool_result_chars=100)  # < 256
    with pytest.raises(pydantic.ValidationError):
        make_settings(database_url="")
```

`tests/conftest.py` 的 `make_settings` base dict 追加:

```python
        database_url="mysql+pymysql://u:p@127.0.0.1:9/wayhelp",  # 占位,测试注入 runtime 不连接
```

Run: `uv run pytest tests/test_config.py -v` —— 新用例 FAIL(database_url 等字段不存在;奇数校验仍在)。

- [ ] **Step 5: config.py 实现**

替换 `app/config.py` 全文:

```python
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_base_url: str = Field(min_length=1)
    openai_api_key: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    structured_output_method: Literal["json_schema", "json_mode"] = "json_schema"
    max_input_tokens: int = Field(default=2000, gt=0)
    max_output_tokens: int = Field(default=1000, gt=0)
    max_message_chars: int = Field(default=8000, gt=0)
    max_sessions: int = Field(default=1000, gt=0)
    max_messages_per_session: int = Field(default=100, gt=0)
    database_url: str = Field(min_length=1)
    test_admin_database_url: str = Field(
        default="mysql+pymysql://root:root-password@127.0.0.1:3306/mysql?charset=utf8mb4"
    )
    test_database_url: str = Field(
        default="mysql+pymysql://root:root-password@127.0.0.1:3306/wayhelp_test?charset=utf8mb4"
    )
    tool_timeout_seconds: float = Field(default=5, gt=0)
    tool_max_retries: int = Field(default=2, ge=0)
    max_tool_calls_per_turn: int = Field(default=5, gt=0)
    max_tool_result_chars: int = Field(default=4000, ge=256)
```

(偶数校验器 `_must_be_even` 删除。)

Run: `uv run pytest` —— 59 + 新增全绿。

- [ ] **Step 6: app/db.py**

```python
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


def ping(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
```

`tests/test_db.py`(本任务只测 ping 失败路径,连接成功路径在 T2 fixture 中间接覆盖):

```python
import pytest
from sqlalchemy.exc import OperationalError

from app.db import make_engine, ping
from tests.conftest import make_settings


def test_ping_failure_raises():
    engine = make_engine(make_settings().database_url)  # 127.0.0.1:9 不可达
    with pytest.raises(OperationalError):
        ping(engine)
```

Run: `uv run pytest tests/test_db.py -v` —— PASS。

- [ ] **Step 7: .env.example 增补 + 提交**

`.env.example` 追加:

```dotenv
DATABASE_URL=mysql+pymysql://wayhelp:wayhelp@127.0.0.1:3306/wayhelp?charset=utf8mb4
TEST_ADMIN_DATABASE_URL=mysql+pymysql://root:root-password@127.0.0.1:3306/mysql?charset=utf8mb4
TEST_DATABASE_URL=mysql+pymysql://root:root-password@127.0.0.1:3306/wayhelp_test?charset=utf8mb4
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
MAX_TOOL_CALLS_PER_TURN=5
MAX_TOOL_RESULT_CHARS=4000
```

真实 `.env` 同步补 `DATABASE_URL=mysql+pymysql://wayhelp:wayhelp@127.0.0.1:3306/wayhelp?charset=utf8mb4`(不提交)。

```bash
git add -A pyproject.toml uv.lock docker-compose.yml db/ app/config.py app/db.py tests/ .env.example
git commit -m "feat(ch02): deps, docker mysql, ddl/seed, settings, db module"
```

---

### Task 2: ORM models + DB 测试 fixture

**Files:**
- Create: `app/models.py`
- Create: `tests/dbfixtures.py`(独立文件,conftest 不污染无 DB 用例)
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `app.db.make_engine / make_session_factory`、`Settings.test_admin_database_url / test_database_url`
- Produces: `Base`、`Conversation / Message / Faq / Ticket` ORM 类;pytest fixture `db_session_factory`(function 级,自动清表)

- [ ] **Step 1: 写失败测试**

`tests/test_models.py`:

```python
import pytest

from app.models import Conversation, Faq, Message, Ticket
from tests.dbfixtures import db_session_factory  # noqa: F401  (fixture 注册)


async def test_four_tables_crud(db_session_factory):
    def _work(sf):
        with sf() as s:
            conv = Conversation(user_id="u-1")
            s.add(conv)
            s.flush()
            s.add(Message(conversation_id=conv.id, role="user", content="你好"))
            s.add(Faq(question="q", answer="a", category="c"))
            s.add(Ticket(ticket_no="T1", conversation_id=conv.id,
                         description="d", ticket_type="售后"))
            s.commit()
        with sf() as s:
            assert s.query(Conversation).count() == 1
            msg = s.query(Message).one()
            assert msg.role == "user" and msg.conversation_id == conv.id
            assert s.query(Faq).one().category == "c"
            ticket = s.query(Ticket).one()
            assert ticket.status == "待处理"
            conv2 = s.get(Conversation, conv.id)
            assert conv2.status == "进行中"
            # 更新
            conv2.status = "已转人工"
            s.commit()
            assert s.get(Conversation, conv.id).status == "已转人工"

    import asyncio
    await asyncio.to_thread(_work, db_session_factory)
```

(后续 DB 测试均以此模式:同步代码块放 `asyncio.to_thread`。)

Run: `uv run pytest tests/test_models.py -v` —— FAIL(无 app.models / tests.dbfixtures)。

- [ ] **Step 2: tests/dbfixtures.py**

```python
"""Docker MySQL 测试库基建:session 级建库建表,function 级清表。

需要 Docker 在线且 `docker compose up -d` 已启动;连接失败时显式 pytest.exit,
不静默跳过(spec §11)。
"""

from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import Settings
from app.db import make_engine, make_session_factory

DDL_PATH = Path(__file__).resolve().parent.parent / "db" / "init" / "01-ddl.sql"
TABLES = ("messages", "tickets", "faq", "conversations")  # 先子后父


def _split_statements(sql_text: str) -> list[str]:
    # 01-ddl.sql 为用户手写的已知文件,语句内不含分号(已核实);剔除注释行
    statements = []
    for chunk in sql_text.split(";\n"):
        lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


@pytest.fixture(scope="session")
def db_engine():
    settings = Settings()  # 读真实 .env 的 TEST_* 字段
    try:
        admin = make_engine(settings.test_admin_database_url)
        with admin.connect() as conn:
            conn.execute(text("CREATE DATABASE IF NOT EXISTS wayhelp_test "
                              "CHARACTER SET utf8mb4"))
    except Exception as exc:
        pytest.exit(f"DB 测试需要 Docker MySQL 在线(docker compose up -d): "
                    f"{type(exc).__name__}", returncode=3)
    engine = make_engine(settings.test_database_url)
    ddl = DDL_PATH.read_text(encoding="utf-8")
    with engine.connect() as conn:
        for stmt in _split_statements(ddl):
            conn.execute(text(stmt))
        conn.commit()
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session_factory(db_engine):
    sf = make_session_factory(db_engine)
    with db_engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table in TABLES:
            conn.execute(text(f"TRUNCATE TABLE {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.commit()
    return sf
```

注意:fixture 的 DDL 语句切分假定语句内无分号(已核实用户手写 DDL 满足);后续改 DDL 时若引入存储过程等含分号语法,此切分器需同步升级。

- [ ] **Step 3: app/models.py**

```python
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conv_status"), default="进行中"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger(unsigned=True), ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="msg_role"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger(unsigned=True), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger(unsigned=True), ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type")
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"), default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

注意:DDL 的 `ON UPDATE CURRENT_TIMESTAMP` 由 MySQL 服务端完成,ORM `onupdate=func.now()` 只是应用层兜底,不冲突。

- [ ] **Step 4: 跑通**

Run: `docker compose up -d && uv run pytest tests/test_models.py tests/test_db.py -v` —— PASS。
Run: `uv run pytest` —— 全绿。

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/dbfixtures.py tests/test_models.py
git commit -m "feat(ch02): orm models aligned with ddl + docker mysql test fixtures"
```

---

### Task 3: SessionStore 协议扩展 + 内存实现 + DbSessionStore + tool_envelope

**Files:**
- Create: `app/tool_envelope.py`
- Modify: `app/sessions.py`(协议与内存实现重写)
- Create: `app/store_db.py`
- Modify: `app/services/chat_service.py`(最小适配:prepare 加 user_id、commit_turn 改列表、await 化;编排逻辑 T5 才动)
- Modify: `app/routers/chat.py`(透传 user_id;新帧 T6 加)
- Modify: `app/schemas.py`(加 user_id / session_id 双形态校验)
- Modify: `tests/conftest.py`(TEST_USER_ID、FakeStreamModel 不变)、`tests/test_sessions.py`、`tests/test_chat_service.py`、`tests/test_chat_api.py`(适配调用签名,行为断言不动)
- Create: `tests/test_store_db.py`、`tests/test_tool_envelope.py`

**Interfaces:**
- Consumes: T2 的 ORM 与 `db_session_factory` fixture
- Produces:
  - `StoredMessage(role: str, content: str | None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None)`
  - `SessionStore` 协议(全 async):`create(user_id) -> str`、`exists(session_id, user_id) -> bool`、`snapshot(session_id) -> list[StoredMessage]`、`commit_turn(session_id, messages: list[StoredMessage]) -> None`
  - `validate_turn(messages, max_tool_calls) -> None`(sessions.py,非法 turn 抛 ValueError)
  - `tool_envelope.wrap(content, ok, error_code, max_chars) -> str`、`unwrap(text) -> tuple[str, bool]`、`truncate_content(content, ok, error_code, max_chars) -> str`
  - `DbSessionStore(session_factory, max_message_chars)`

- [ ] **Step 1: tool_envelope 失败测试**

`tests/test_tool_envelope.py`:

```python
import json

import pytest

from app.tool_envelope import truncate_content, unwrap, wrap


def test_roundtrip_success():
    text = wrap("结果", True, None, 4000)
    assert json.loads(text)["v"] == 1
    assert unwrap(text) == ("结果", True)


def test_roundtrip_error():
    text = wrap("超时", False, "timeout", 4000)
    payload = json.loads(text)
    assert payload["ok"] is False and payload["error_code"] == "timeout"
    assert unwrap(text) == ("超时", False)


def test_wrap_respects_max_chars():
    text = wrap("x" * 10000, True, None, 300)
    assert len(text) <= 300
    payload = json.loads(text)
    assert payload["truncated"] is True
    content, ok = unwrap(text)
    assert ok is True and len(content) < 10000


def test_truncate_content_matches_wrap():
    truncated = truncate_content("x" * 10000, True, None, 300)
    assert wrap(truncated, True, None, 300) == wrap("x" * 10000, True, None, 300)


def test_unwrap_corrupted_raises():
    with pytest.raises(ValueError):
        unwrap("not-json")
    with pytest.raises(ValueError):
        unwrap('{"v":2,"ok":true,"content":"x"}')  # 版本不符
    with pytest.raises(ValueError):
        unwrap('{"v":1,"content":"x"}')  # 缺 ok
```

Run: `uv run pytest tests/test_tool_envelope.py -v` —— FAIL。

- [ ] **Step 2: app/tool_envelope.py 实现**

```python
import json


def wrap(content: str, ok: bool, error_code: str | None, max_chars: int) -> str:
    """工具结果持久化 envelope;序列化长度 <= max_chars,超出截断 content 并标 truncated。"""
    truncated = False
    while True:
        payload: dict = {"v": 1, "ok": ok}
        if ok:
            payload["content"] = content
        else:
            payload["error_code"] = error_code or "tool_error"
            payload["message"] = content
        if truncated:
            payload["truncated"] = True
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        # 需要截断:保守估算每字符 1 字节(英文)到 3 字节(中文),逐次减半收敛
        truncated = True
        overflow = len(text) - max_chars
        cut = max(1, int(len(content) - max(overflow, len(content) // 10)))
        content = content[:cut]


def truncate_content(content: str, ok: bool, error_code: str | None, max_chars: int) -> str:
    """返回使 wrap(...) 不超长的 content(供回灌模型与落库一致使用)。"""
    # wrap 内部循环保证返回值必 <= max_chars,不能用它判断是否超长;
    # 精确式:空 content 的 envelope(含两个引号)+ json 序列化后的 content 长度,与 wrap 输出逐字节一致
    base = len(wrap("", ok, error_code, max_chars)) - 2

    def fits(candidate: str) -> bool:
        return base + len(json.dumps(candidate, ensure_ascii=False)) <= max_chars

    if fits(content):
        return content
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(content[:mid]):
            lo = mid
        else:
            hi = mid - 1
    return content[:lo]


def unwrap(text: str) -> tuple[str, bool]:
    """envelope -> (content, ok);非法格式视为持久数据损坏,抛 ValueError。"""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("corrupted tool envelope") from exc
    if not isinstance(payload, dict) or payload.get("v") != 1 or "ok" not in payload:
        raise ValueError("corrupted tool envelope")
    ok = bool(payload["ok"])
    body = payload.get("content") if ok else payload.get("message")
    if not isinstance(body, str):
        raise ValueError("corrupted tool envelope")
    return body, ok
```

Run: `uv run pytest tests/test_tool_envelope.py -v` —— PASS。

- [ ] **Step 3: sessions.py 重写(协议 + 校验 + 内存实现)**

替换 `app/sessions.py` 全文:

```python
import uuid
from dataclasses import dataclass
from typing import Protocol

from app.errors import SessionCapacityReachedError


@dataclass(frozen=True)
class StoredMessage:
    role: str  # "user" | "assistant" | "tool"
    content: str | None  # tool 行为 envelope JSON 文本;assistant 纯工具调用时可为 None
    tool_calls: list[dict] | None = None  # 仅 assistant 行: [{"name","args","id","type"}]
    tool_call_id: str | None = None       # 仅 tool 行


class SessionStore(Protocol):
    async def create(self, user_id: str) -> str: ...
    async def exists(self, session_id: str, user_id: str) -> bool: ...
    async def snapshot(self, session_id: str) -> list[StoredMessage]: ...
    async def commit_turn(self, session_id: str, messages: list[StoredMessage]) -> None: ...


def validate_turn(messages: list[StoredMessage], max_tool_calls: int) -> None:
    """spec §5.1 四条;非法抛 ValueError。"""
    if len(messages) < 2:
        raise ValueError("turn must contain at least user + assistant")
    if messages[0].role != "user" or not (messages[0].content or "").strip():
        raise ValueError("turn must start with a non-empty user message")
    last = messages[-1]
    if last.role != "assistant" or not (last.content or "").strip():
        raise ValueError("turn must end with a non-empty assistant message")
    middle = messages[1:-1]
    tool_calls: list[dict] = []
    tool_ids: list[str] = []
    i = 0
    while i < len(middle):
        m = middle[i]
        if m.role == "assistant":
            if tool_calls or tool_ids:
                raise ValueError("assistant tool group must be followed by its tool messages")
            if m.tool_calls:
                if len(m.tool_calls) > max_tool_calls:
                    raise ValueError("too many tool calls in turn")
                tool_calls = list(m.tool_calls)
                for call in tool_calls:
                    cid = call.get("id") if isinstance(call, dict) else None
                    if not isinstance(cid, str) or not cid or len(cid) > 64:
                        raise ValueError("invalid tool call id")
                i += 1
                continue
            raise ValueError("middle assistant message without tool calls")
        if m.role == "tool":
            if not tool_calls:
                raise ValueError("orphan tool message")
            if not m.tool_call_id or len(m.tool_call_id) > 64:
                raise ValueError("tool message missing tool_call_id")
            tool_ids.append(m.tool_call_id)
            i += 1
            continue
        raise ValueError(f"unexpected role in turn middle: {m.role}")
    expected = [c["id"] for c in tool_calls]
    if sorted(expected) != sorted(tool_ids) or len(set(tool_ids)) != len(tool_ids):
        raise ValueError("tool call ids and tool messages must match one-to-one")


def _turns(messages: list[StoredMessage]) -> list[list[StoredMessage]]:
    """按 user 起始切分完整 turn。"""
    turns: list[list[StoredMessage]] = []
    for m in messages:
        if m.role == "user":
            turns.append([m])
        elif turns:
            turns[-1].append(m)
    return [t for t in turns if len(t) >= 2]


class InMemorySessionStore:
    def __init__(self, max_sessions: int, max_messages_per_session: int,
                 max_message_chars: int, max_tool_calls_per_turn: int = 5):
        self._max_sessions = max_sessions
        self._max_messages = max_messages_per_session
        self._max_chars = max_message_chars
        self._max_tool_calls = max_tool_calls_per_turn
        self._sessions: dict[str, list[StoredMessage]] = {}

    async def create(self, user_id: str) -> str:
        if len(self._sessions) >= self._max_sessions:
            raise SessionCapacityReachedError("session capacity reached")
        sid = str(uuid.uuid4())
        self._sessions[sid] = []
        return sid

    async def exists(self, session_id: str, user_id: str) -> bool:
        return session_id in self._sessions  # 内存实现不绑定 user_id(测试用)

    async def snapshot(self, session_id: str) -> list[StoredMessage]:
        return list(self._sessions[session_id])

    async def commit_turn(self, session_id: str, messages: list[StoredMessage]) -> None:
        validate_turn(messages, self._max_tool_calls)
        for m in messages:
            if m.content is not None and len(m.content) > self._max_chars and m.role != "tool":
                raise ValueError("message text exceeds MAX_MESSAGE_CHARS")
        msgs = self._sessions[session_id]
        msgs.extend(messages)
        limit = max(self._max_messages, self._max_tool_calls + 3)
        while len(msgs) > limit:
            turns = _turns(msgs)
            if len(turns) <= 1:
                break  # 始终保留最新完整 turn
            del msgs[: len(turns[0])]
```

注:内存 `exists` 不核对 user_id —— 它只服务测试注入;生产 DB 实现才核对(spec §4)。在 docstring 注明。

- [ ] **Step 4: DbSessionStore 失败测试**

`tests/test_store_db.py`:

```python
import asyncio

import pytest

from app.store_db import DbSessionStore
from app.sessions import StoredMessage
from app.tool_envelope import wrap
from tests.dbfixtures import db_session_factory  # noqa: F401

USER = "11111111-1111-1111-1111-111111111111"


def _turn(sid_msgs=None):
    return [StoredMessage("user", "问"), StoredMessage("assistant", "答")]


async def test_create_exists_snapshot_commit(db_session_factory):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    assert sid.isdigit()
    assert await store.exists(sid, USER) is True
    assert await store.exists(sid, "other-user") is False
    assert await store.snapshot(sid) == []
    await store.commit_turn(sid, _turn())
    snap = await store.snapshot(sid)
    assert [(m.role, m.content) for m in snap] == [("user", "问"), ("assistant", "答")]


async def test_tool_turn_roundtrip(db_session_factory):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    calls = [{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"}]
    envelope = wrap("查到 1 条", True, None, 4000)
    await store.commit_turn(sid, [
        StoredMessage("user", "退货政策?"),
        StoredMessage("assistant", None, tool_calls=calls),
        StoredMessage("tool", envelope, tool_call_id="call_1"),
        StoredMessage("assistant", "退货政策是……"),
    ])
    snap = await store.snapshot(sid)
    assert [m.role for m in snap] == ["user", "assistant", "tool", "assistant"]
    assert snap[1].tool_calls[0]["name"] == "query_faq"
    assert snap[2].tool_call_id == "call_1" and snap[2].content == envelope


async def test_commit_rolls_back_on_failure(db_session_factory, monkeypatch):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)

    from sqlalchemy.orm import Session

    real_flush = Session.flush

    def boom(self):
        if getattr(self, "_boom_armed", False):
            raise RuntimeError("simulated mid-transaction failure")
        return real_flush(self)

    # 第一次 flush(conversations.updated_at 之外的 message 插入)正常,第二次引爆
    state = {"n": 0}

    def counting_flush(self):
        state["n"] += 1
        if state["n"] >= 2:
            raise RuntimeError("boom")
        return real_flush(self)

    monkeypatch.setattr(Session, "flush", counting_flush)
    with pytest.raises(RuntimeError):
        await store.commit_turn(sid, _turn())
    monkeypatch.undo()
    assert await store.snapshot(sid) == []  # 无半个 turn


async def test_invalid_turn_rejected(db_session_factory):
    store = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store.create(USER)
    with pytest.raises(ValueError):
        await store.commit_turn(sid, [StoredMessage("assistant", "答")])
    with pytest.raises(ValueError):
        # 孤儿 tool 消息
        await store.commit_turn(sid, [
            StoredMessage("user", "问"),
            StoredMessage("tool", wrap("x", True, None, 4000), tool_call_id="call_9"),
            StoredMessage("assistant", "答"),
        ])


async def test_survives_restart(db_session_factory):
    """模拟应用重启:新建 DbSessionStore 实例,同一会话仍可读写。"""
    store1 = DbSessionStore(db_session_factory, max_message_chars=8000)
    sid = await store1.create(USER)
    await store1.commit_turn(sid, _turn())
    store2 = DbSessionStore(db_session_factory, max_message_chars=8000)
    assert await store2.exists(sid, USER) is True
    assert len(await store2.snapshot(sid)) == 2
```

Run: `uv run pytest tests/test_store_db.py -v` —— FAIL(无 store_db)。

- [ ] **Step 5: app/store_db.py 实现**

```python
import asyncio
from datetime import datetime

from sqlalchemy import select, update

from app.models import Conversation, Message
from app.sessions import StoredMessage, validate_turn


class DbSessionStore:
    """SessionStore 的 MySQL 实现;所有方法 async,同步 Session 在线程内创建/关闭。"""

    def __init__(self, session_factory, max_message_chars: int):
        self._sf = session_factory
        self._max_chars = max_message_chars

    async def create(self, user_id: str) -> str:
        return await asyncio.to_thread(self._create_sync, user_id)

    def _create_sync(self, user_id: str) -> str:
        with self._sf() as s:
            conv = Conversation(user_id=user_id)
            s.add(conv)
            s.commit()
            return str(conv.id)

    async def exists(self, session_id: str, user_id: str) -> bool:
        return await asyncio.to_thread(self._exists_sync, session_id, user_id)

    def _exists_sync(self, session_id: str, user_id: str) -> bool:
        with self._sf() as s:
            row = s.execute(
                select(Conversation.id).where(
                    Conversation.id == int(session_id),
                    Conversation.user_id == user_id,
                )
            ).first()
            return row is not None

    async def snapshot(self, session_id: str) -> list[StoredMessage]:
        return await asyncio.to_thread(self._snapshot_sync, session_id)

    def _snapshot_sync(self, session_id: str) -> list[StoredMessage]:
        with self._sf() as s:
            rows = (
                s.query(Message)
                .filter(Message.conversation_id == int(session_id))
                .order_by(Message.created_at, Message.id)
                .all()
            )
            return [
                StoredMessage(r.role, r.content, r.tool_calls, r.tool_call_id)
                for r in rows
            ]

    async def commit_turn(self, session_id: str, messages: list[StoredMessage]) -> None:
        validate_turn(messages, max_tool_calls=64)  # DB 侧结构校验;数量上限由编排层把关
        await asyncio.to_thread(self._commit_sync, session_id, messages)

    def _commit_sync(self, session_id: str, messages: list[StoredMessage]) -> None:
        cid = int(session_id)
        with self._sf() as s:
            for m in messages:
                s.add(Message(
                    conversation_id=cid,
                    role=m.role,
                    content=m.content,
                    tool_calls=m.tool_calls,
                    tool_call_id=m.tool_call_id,
                ))
            s.execute(
                update(Conversation)
                .where(Conversation.id == cid)
                .values(updated_at=datetime.now())
            )
            s.commit()  # 任一失败整体回滚(Session 上下文管理器)
```

Run: `uv run pytest tests/test_store_db.py -v` —— PASS。

- [ ] **Step 6: 适配 chat_service / schemas / routers / ch01 测试**

`app/schemas.py` 的 `ChatStreamRequest` 改为:

```python
import re
import uuid

from pydantic import BaseModel, Field, field_validator

_SESSION_ID_RE = re.compile(r"^([1-9]\d{0,18}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")


class ChatStreamRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str

    @field_validator("user_id")
    @classmethod
    def _user_id_must_be_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError, TypeError):
            raise ValueError("user_id must be a canonical UUID string")
        return v

    @field_validator("session_id")
    @classmethod
    def _session_id_form(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _SESSION_ID_RE.match(v):
            raise ValueError("session_id must be a decimal id or canonical UUID string")
        return v

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must contain non-whitespace text")
        return v
```

`app/services/chat_service.py` 最小适配(本任务只保编译与旧行为,编排 T5 重写):

- `prepare(self, user_id: str, session_id: str | None, message: str)`;创建走 `await self._store.create(user_id)`;校验走 `await self._store.exists(sid, user_id)`
- 提交处改:`await self._store.commit_turn(turn.session_id, [StoredMessage("user", turn.user_text), StoredMessage("assistant", full)])`
- `snapshot` 调用点改 `await`

`app/routers/chat.py`:`service.prepare(body.user_id, body.session_id, body.message)`。

ch01 测试适配规则(行为断言一律不动):
- `tests/conftest.py` 加 `TEST_USER_ID = "11111111-1111-1111-1111-111111111111"`
- `test_sessions.py`:`s.commit_turn(sid, "问", "答")` → `await s.commit_turn(sid, [StoredMessage("user", "问"), StoredMessage("assistant", "答")])`;`create()` / `exists()` / `snapshot()` 全部 `await`,`create()` → `create(TEST_USER_ID)`;「按对淘汰」用例改为按完整 turn 淘汰的等价断言;删除/修改「奇数上限报错」相关用例(该限制已移除)
- `test_chat_service.py` / `test_chat_api.py`:`prepare(None, msg)` → `prepare(TEST_USER_ID, None, msg)`;`prepare(uuid.UUID(sid), msg)` → `prepare(TEST_USER_ID, sid, msg)`;请求 JSON 全部加 `"user_id": TEST_USER_ID`;404 用例的 UUID session_id 保留(UUID 形态仍合法,不存在即 404)

Run: `uv run pytest` —— 全绿(含新 DB 测试)。

- [ ] **Step 7: Commit**

```bash
git add app/sessions.py app/store_db.py app/tool_envelope.py app/schemas.py app/services/chat_service.py app/routers/chat.py tests/
git commit -m "feat(ch02): store protocol v2 (user-bound, tool turn), db store, tool envelope"
```

---

### Task 4: 五个业务工具 + ToolRegistry + ToolExecutor

**Files:**
- Create: `app/tools/__init__.py`(空)
- Create: `app/tools/business.py`
- Create: `app/tools/executor.py`
- Create: `tests/test_tools.py`、`tests/test_executor.py`

**Interfaces:**
- Consumes: `db_session_factory` fixture、`app.tool_envelope.truncate_content`
- Produces:
  - `build_tools(session_factory, conversation_id: int) -> list[BaseTool]`(五工具;三个 mock 模块级定义,query_faq/create_ticket 闭包)
  - `MOCK_TOOLS: list[BaseTool]`(query_order/query_product/query_logistics,供测试)
  - `ToolRegistry(tools: list[BaseTool])`,`.tools -> list[BaseTool]`、`.get(name) -> BaseTool | None`;重复 name 构造即 ValueError
  - `ToolExecutor(registry, timeout_seconds, max_retries, max_result_chars)`;`async execute(call: dict) -> ToolOutcome`
  - `ToolOutcome(message: ToolMessage, record: ToolExecutionRecord)`;`ToolExecutionRecord(name, args, ok, duration_ms, retry_count, error_type)`
  - `WRITE_TOOLS = {"create_ticket"}`

- [ ] **Step 1: 工具失败测试**

`tests/test_tools.py`:

```python
import asyncio
import json

import pytest

from app.models import Conversation, Faq, Ticket
from app.tools.business import MOCK_TOOLS, build_tools
from tests.dbfixtures import db_session_factory  # noqa: F401


def _invoke(tool, **args):
    return tool.invoke({"name": tool.name, "args": args, "id": "t1", "type": "tool_call"})


def test_mock_tools_return_parseable_json():
    by_name = {t.name: t for t in MOCK_TOOLS}
    order = json.loads(_invoke(by_name["query_order"], order_id="1001").content)
    assert {"order_id", "status", "amount", "product", "created_at"} <= order.keys()
    product = json.loads(_invoke(by_name["query_product"], product_name="水杯").content)
    assert {"product_name", "price", "stock", "on_sale"} <= product.keys()
    logistics = json.loads(_invoke(by_name["query_logistics"], order_id="1001").content)
    assert {"order_id", "company", "tracking_no", "traces"} <= logistics.keys()
    assert 2 <= len(logistics["traces"]) <= 4


def test_mock_tools_arg_constraints():
    from pydantic import ValidationError
    by_name = {t.name: t for t in MOCK_TOOLS}
    with pytest.raises(ValidationError):
        _invoke(by_name["query_order"], order_id="   ")
    with pytest.raises(ValidationError):
        _invoke(by_name["query_order"], order_id="x" * 65)


async def test_query_faq_hit_and_miss(db_session_factory):
    def _seed(sf):
        with sf() as s:
            s.add_all([
                Faq(question="退货政策是什么", answer="七天无理由", category="退货"),
                Faq(question="运费怎么算", answer="满 99 包邮", category="运费"),
            ])
            s.commit()

    await asyncio.to_thread(_seed, db_session_factory)
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=1)}
    hit = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="退货政策").content))
    # query_faq 返回 JSON 字符串: {"results": [...]}
    assert hit["results"] and "七天无理由" in hit["results"][0]["answer"]
    miss = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="邮费").content))
    assert miss["results"] == [] and "未找到" in miss["note"]


async def test_query_faq_like_wildcards_literal(db_session_factory):
    def _seed(sf):
        with sf() as s:
            from app.models import Faq
            s.add(Faq(question="100% 正品吗", answer="是", category="其他"))
            s.commit()

    await asyncio.to_thread(_seed, db_session_factory)
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=1)}
    # "%" 按字面匹配:keyword="100%" 命中;"%"(裸通配)不应捞全表
    hit = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="100%").content))
    assert len(hit["results"]) == 1
    wild = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["query_faq"], keyword="%").content))
    assert wild["results"] == []


async def test_create_ticket_writes_and_flips_status(db_session_factory):
    def _mk(sf):
        with sf() as s:
            c = Conversation(user_id="u")
            s.add(c)
            s.commit()
            return c.id

    cid = await asyncio.to_thread(_mk, db_session_factory)
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=cid)}
    out = json.loads(await asyncio.to_thread(
        lambda: _invoke(tools["create_ticket"], description="要求人工核实退款",
                        ticket_type="售后").content))
    assert out["ticket_no"].startswith("T") and len(out["ticket_no"]) == 27

    def _check(sf):
        with sf() as s:
            t = s.query(Ticket).one()
            assert t.conversation_id == cid and t.status == "待处理"
            assert s.get(Conversation, cid).status == "已转人工"

    await asyncio.to_thread(_check, db_session_factory)


async def test_create_ticket_rejects_bad_type(db_session_factory):
    from pydantic import ValidationError
    tools = {t.name: t for t in build_tools(db_session_factory, conversation_id=1)}
    with pytest.raises(ValidationError):
        _invoke(tools["create_ticket"], description="x", ticket_type="退款")
```

Run: `uv run pytest tests/test_tools.py -v` —— FAIL。

- [ ] **Step 2: app/tools/business.py 实现**

```python
import json
import random
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import StringConstraints

from app.models import Conversation, Faq, Ticket

OrderId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
ProductName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Keyword = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]

_ORDER_STATUS = ["待付款", "待发货", "已发货", "已完成"]
_COMPANIES = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
_TRACE_TEXT = ["已揽收", "到达转运中心", "派件中", "已签收"]


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
def query_order(order_id: OrderId) -> str:
    """查询订单信息。参数 order_id 为订单号。返回订单状态、金额、商品名、下单时间。"""
    return _json({
        "order_id": order_id,
        "status": random.choice(_ORDER_STATUS),
        "amount": round(random.uniform(9.9, 999.0), 2),
        "product": random.choice(["保温杯", "蓝牙耳机", "机械键盘", "帆布包"]),
        "created_at": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(timespec="seconds"),
    })


@tool
def query_product(product_name: ProductName) -> str:
    """查询商品信息。参数 product_name 为商品名称关键词。返回价格、库存、是否在售。"""
    return _json({
        "product_name": product_name,
        "price": round(random.uniform(9.9, 1999.0), 2),
        "stock": random.randint(0, 500),
        "on_sale": random.random() > 0.1,
    })


@tool
def query_logistics(order_id: OrderId) -> str:
    """查询订单物流。参数 order_id 为订单号。返回物流公司、运单号与物流节点列表。"""
    n = random.randint(2, 4)
    base = datetime.now() - timedelta(hours=n * 8)
    traces = [
        {"time": (base + timedelta(hours=i * 8)).isoformat(timespec="seconds"),
         "description": _TRACE_TEXT[i - 1] if i - 1 < len(_TRACE_TEXT) else "运输中"}
        for i in range(1, n + 1)
    ]
    return _json({
        "order_id": order_id,
        "company": random.choice(_COMPANIES),
        "tracking_no": "SF" + "".join(random.choices("0123456789", k=10)),
        "traces": traces,
    })


MOCK_TOOLS: list[BaseTool] = [query_order, query_product, query_logistics]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_tools(session_factory, conversation_id: int) -> list[BaseTool]:
    """每轮请求构造绑定该会话的工具实例;conversation_id 经闭包注入,不对模型暴露。"""

    @tool
    def query_faq(keyword: Keyword) -> str:
        """查询常见问题库。参数 keyword 为关键词,对问题与答案做模糊匹配,返回最相关的前 5 条。"""
        pattern = f"%{_escape_like(keyword)}%"
        with session_factory() as s:
            rows = (
                s.query(Faq)
                .filter((Faq.question.like(pattern, escape="\\"))
                        | (Faq.answer.like(pattern, escape="\\")))
                .order_by(Faq.id)
                .limit(5)
                .all()
            )
        if not rows:
            return _json({"results": [], "note": "未找到与关键词相关的常见问题"})
        return _json({"results": [
            {"question": r.question, "answer": r.answer, "category": r.category} for r in rows
        ]})

    @tool
    def create_ticket(
        description: Description,
        ticket_type: Literal["售后", "投诉", "咨询"],
    ) -> str:
        """创建人工工单并转接人工客服。参数 description 为问题描述,ticket_type 只能是 售后/投诉/咨询。"""
        ticket_no = "T" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + secrets.token_hex(6)
        with session_factory() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                raise ValueError("conversation not found")
            s.add(Ticket(ticket_no=ticket_no, conversation_id=conversation_id,
                         description=description, ticket_type=ticket_type))
            conv.status = "已转人工"
            s.commit()  # 工单与会话状态同事务,同成同败
        return _json({"ticket_no": ticket_no, "status": "待处理"})

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
```

Run: `uv run pytest tests/test_tools.py -v` —— PASS。

- [ ] **Step 3: 执行器失败测试**

`tests/test_executor.py`:

```python
import asyncio
import time

import pytest
from langchain_core.tools import tool
from pydantic import ValidationError

from app.tools.executor import ToolExecutor, ToolRegistry


@tool
def slow_tool(x: str) -> str:
    """慢工具"""
    time.sleep(10)
    return "done"


@tool
def flaky_tool(x: str) -> str:
    """前两次超时第三次成功"""
    flaky_tool.calls += 1
    if flaky_tool.calls < 3:
        time.sleep(10)
    return "ok-after-retry"


flaky_tool.calls = 0


@tool
def biz_error_tool(x: str) -> str:
    """业务错误"""
    raise ValueError("business broke")


@tool
def write_tool(x: str) -> str:
    """写工具"""
    return "written"


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


async def test_registry_rejects_duplicate():
    with pytest.raises(ValueError):
        ToolRegistry([slow_tool, slow_tool])


async def test_timeout_then_retry_exhausted():
    ex = ToolExecutor(ToolRegistry([slow_tool]), timeout_seconds=0.1, max_retries=1,
                      max_result_chars=4000)
    t0 = time.monotonic()
    outcome = await ex.execute(_call("slow_tool", {"x": "1"}))
    assert outcome.message.status == "error"
    assert outcome.record.retry_count == 1
    assert outcome.record.error_type == "TimeoutError"
    assert time.monotonic() - t0 < 3  # wait_for 生效,不等满 10s


async def test_retry_eventually_succeeds():
    flaky_tool.calls = 0
    ex = ToolExecutor(ToolRegistry([flaky_tool]), timeout_seconds=0.1, max_retries=2,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("flaky_tool", {"x": "1"}))
    assert outcome.message.status == "success"
    assert outcome.message.content == "ok-after-retry"
    assert outcome.record.retry_count == 2


async def test_business_error_not_retried():
    ex = ToolExecutor(ToolRegistry([biz_error_tool]), timeout_seconds=1, max_retries=2,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("biz_error_tool", {"x": "1"}))
    assert outcome.message.status == "error"
    assert outcome.record.retry_count == 0
    assert "business broke" not in outcome.message.content  # 脱敏
    assert outcome.record.error_type == "ValueError"


async def test_unknown_tool():
    ex = ToolExecutor(ToolRegistry([]), timeout_seconds=1, max_retries=0,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("ghost_tool", {"x": "1"}, cid="c9"))
    assert outcome.message.status == "error"
    assert outcome.message.tool_call_id == "c9"
    assert outcome.record.error_type == "UnknownTool"


async def test_invalid_args_becomes_error_tool_message():
    @tool
    def strict_tool(n: int) -> str:
        """严格参数"""
        return str(n)

    ex = ToolExecutor(ToolRegistry([strict_tool]), timeout_seconds=1, max_retries=2,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("strict_tool", {"n": "不是数字"}, cid="c7"))
    assert outcome.message.status == "error"
    assert outcome.message.tool_call_id == "c7"
    assert outcome.record.error_type == "ValidationError"
    assert outcome.record.retry_count == 0


async def test_result_truncated_but_valid_envelope_content():
    @tool
    def big_tool(x: str) -> str:
        """大结果"""
        return "汉" * 5000

    ex = ToolExecutor(ToolRegistry([big_tool]), timeout_seconds=5, max_retries=0,
                      max_result_chars=300)
    outcome = await ex.execute(_call("big_tool", {"x": "1"}))
    assert outcome.message.status == "success"
    from app.tool_envelope import wrap
    assert len(wrap(outcome.message.content, True, None, 300)) <= 300


async def test_write_tool_no_wait_for_no_retry(monkeypatch):
    called = {"wait_for": 0}
    real_wait_for = asyncio.wait_for

    async def spy(coro, timeout):
        called["wait_for"] += 1
        return await real_wait_for(coro, timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy)
    ex = ToolExecutor(ToolRegistry([write_tool]), timeout_seconds=0.001, max_retries=3,
                      max_result_chars=4000, write_tools={"write_tool"})
    outcome = await ex.execute(_call("write_tool", {"x": "1"}))
    assert outcome.message.status == "success"
    assert called["wait_for"] == 0  # 写工具不经过 wait_for
```

Run: `uv run pytest tests/test_executor.py -v` —— FAIL。

- [ ] **Step 4: app/tools/executor.py 实现**

```python
import asyncio
import time
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from app.tool_envelope import truncate_content

RETRYABLE = (TimeoutError, asyncio.TimeoutError, OperationalError, ConnectionError)
WRITE_TOOLS = {"create_ticket"}


@dataclass(frozen=True)
class ToolExecutionRecord:
    name: str
    args: dict
    ok: bool
    duration_ms: int
    retry_count: int
    error_type: str | None


@dataclass(frozen=True)
class ToolOutcome:
    message: ToolMessage
    record: ToolExecutionRecord


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool name")
        self._tools = list(tools)
        self._by_name = dict(zip(names, self._tools))

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def get(self, name: str) -> BaseTool | None:
        return self._by_name.get(name)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds: float, max_retries: int,
                 max_result_chars: int, write_tools: set[str] | None = None):
        self._registry = registry
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_chars = max_result_chars
        self._write_tools = WRITE_TOOLS if write_tools is None else write_tools

    async def execute(self, call: dict) -> ToolOutcome:
        name = call.get("name", "")
        args = call.get("args") or {}
        call_id = call.get("id") or ""
        started = time.monotonic()
        tool = self._registry.get(name)
        if tool is None:
            return self._error(name, args, call_id, "unknown_tool", "UnknownTool", started)
        try:
            tool.args_schema(**args)  # 参数校验先行,失败不重试
        except ValidationError:
            return self._error(name, args, call_id, "invalid_args", "ValidationError", started)
        if name in self._write_tools:
            return await self._run_write(tool, call, started)
        return await self._run_readonly(tool, call, started)

    async def _run_readonly(self, tool: BaseTool, call: dict, started: float) -> ToolOutcome:
        retries = 0
        while True:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, call), timeout=self._timeout
                )
                return self._success(tool, call, result, retries, started)
            except RETRYABLE as exc:
                if retries >= self._max_retries:
                    return self._error(call.get("name", ""), call.get("args") or {},
                                       call.get("id") or "", "tool_unavailable",
                                       type(exc).__name__, started, retries)
                retries += 1
                await asyncio.sleep(0.5 * 2 ** (retries - 1))
            except Exception as exc:  # 业务异常不重试
                return self._error(call.get("name", ""), call.get("args") or {},
                                   call.get("id") or "", "tool_error",
                                   type(exc).__name__, started, retries)

    async def _run_write(self, tool: BaseTool, call: dict, started: float) -> ToolOutcome:
        try:
            result = await asyncio.to_thread(tool.invoke, call)  # 不 wait_for,不重试
            return self._success(tool, call, result, 0, started)
        except Exception as exc:
            return self._error(call.get("name", ""), call.get("args") or {},
                               call.get("id") or "", "tool_error",
                               type(exc).__name__, started, 0)

    def _success(self, tool, call, result, retries, started) -> ToolOutcome:
        content = result.content if isinstance(result, ToolMessage) else str(result)
        content = truncate_content(content, True, None, self._max_chars)
        msg = ToolMessage(content=content, tool_call_id=call.get("id") or "",
                          name=tool.name, status="success")
        return ToolOutcome(msg, ToolExecutionRecord(
            tool.name, call.get("args") or {}, True,
            int((time.monotonic() - started) * 1000), retries, None))

    _ERROR_TEXT = {
        "unknown_tool": "调用了未注册的工具",
        "invalid_args": "工具参数不合法",
        "tool_unavailable": "工具暂时不可用(已重试)",
        "tool_error": "工具执行失败",
    }

    def _error(self, name, args, call_id, code, error_type, started, retries=0) -> ToolOutcome:
        text = truncate_content(self._ERROR_TEXT[code], False, code, self._max_chars)
        msg = ToolMessage(content=text, tool_call_id=call_id, name=name or "unknown",
                          status="error")
        return ToolOutcome(msg, ToolExecutionRecord(
            name, args, False, int((time.monotonic() - started) * 1000), retries, error_type))
```

注意:`asyncio.to_thread(tool.invoke, call)` 内部对 query_faq/create_ticket 即在线程内开/关 Session,满足 spec §5.4。

Run: `uv run pytest tests/test_executor.py -v` —— PASS(慢工具单测用 0.1s 超时,不会真等 10s)。

- [ ] **Step 5: 全量回归 + Commit**

Run: `uv run pytest` —— 全绿。

```bash
git add app/tools/ tests/test_tools.py tests/test_executor.py
git commit -m "feat(ch02): five business tools + registry/executor with timeout-retry"
```

---

### Task 5: 单轮编排(tool_chat_chain + ChatService 重写 + SessionLockRegistry)

**Files:**
- Create: `app/chains/tool_chat_chain.py`
- Modify: `app/services/chat_service.py`(stream 重写为两段式;锁改 SessionLockRegistry)
- Modify: `tests/conftest.py`(FakeStreamModel 支持 tool_call_chunks 与 bind_tools 记录)
- Create: `tests/test_tool_chat_chain.py`、`tests/test_orchestration.py`

**Interfaces:**
- Consumes: T3 store/协议、T4 executor/registry、`build_tools`
- Produces:
  - `rebuild_messages(stored: list[StoredMessage]) -> list[BaseMessage]`(envelope 解开,ToolMessage status 恢复;损坏抛 ValueError)
  - `merge_tool_call_chunks(acc: dict[int, dict], chunks: list[dict]) -> None` 与 `finalize_tool_calls(acc) -> tuple[list[dict], bool]`(返回 (calls, 有无法解析))
  - `build_messages_with_tools(system_prompt, history, current_input, max_input_tokens) -> list[BaseMessage]`(trim + 孤儿 tool 组丢弃 + 结构校验)
  - `ToolStartEvent(tool_call_id, name, args)`、`ToolEndEvent(tool_call_id, name, ok, summary)`(chat_service 事件)
  - `SessionLockRegistry`:`async acquire(session_id) -> asyncio.Lock`、`release(session_id) -> None`
  - `ChatService(store, model, settings, system_prompt, toolset_factory)`

- [ ] **Step 1: conftest FakeStreamModel 扩展**

`tests/conftest.py` 的 FakeChunk/FakeStreamModel 替换为:

```python
class FakeChunk:
    def __init__(self, content: str = "", finish_reason: str | None = None,
                 tool_call_chunks: list[dict] | None = None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.response_metadata = {} if finish_reason is None else {"finish_reason": finish_reason}


class FakeStreamModel:
    """script 元素:str(delta 文本)| Exception | ("finish", reason) | ("tool", [tool_call_chunks...])
    多段脚本用 ("then", next_script) 分隔第二次调用(bind_tools 后第一次、plain 第二次)。"""

    def __init__(self, script):
        self._scripts = [list(script)]
        self.received: list = []
        self.received_tools: list[list | None] = []  # 每次调用绑定的工具名列表或 None

    def bind_tools(self, tools):
        self._bound = [t.name for t in tools]
        return self

    async def astream(self, messages):
        self.received.append(messages)
        self.received_tools.append(getattr(self, "_bound", None))
        self._bound = None
        script = self._scripts.pop(0) if self._scripts else ["(空)"]
        for item in script:
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple) and item[0] == "finish":
                yield FakeChunk("", item[1])
            elif isinstance(item, tuple) and item[0] == "tool":
                yield FakeChunk("", tool_call_chunks=item[1])
            elif isinstance(item, tuple) and item[0] == "then":
                self._scripts.insert(0, list(item[1]))
            else:
                yield FakeChunk(item)
            await asyncio.sleep(0)
```

tool_call_chunks 用例格式(与 LangChain 一致,index 分片聚合):

```python
TOOL_CHUNKS = [
    {"name": "query_logistics", "args": "{\"order_id\": \"10", "id": "call_1", "index": 0},
    {"name": None, "args": "01\"}", "id": None, "index": 0},
]
```

- [ ] **Step 2: tool_chat_chain 失败测试**

`tests/test_tool_chat_chain.py`:

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.chains.tool_chat_chain import (
    build_messages_with_tools,
    finalize_tool_calls,
    merge_tool_call_chunks,
    rebuild_messages,
)
from app.sessions import StoredMessage
from app.tool_envelope import wrap


def test_rebuild_plain_turn():
    msgs = rebuild_messages([StoredMessage("user", "问"), StoredMessage("assistant", "答")])
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage]


def test_rebuild_tool_turn_restores_status():
    stored = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id="c1"),
        StoredMessage("tool", wrap("失败摘要", False, "timeout", 4000), tool_call_id="c2"),
        StoredMessage("assistant", "答复"),
    ]
    # c2 没有对应申请单 -> 对不上即数据损坏
    with pytest.raises(ValueError):
        rebuild_messages(stored)


def test_rebuild_tool_turn_ok():
    stored = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id="c1"),
        StoredMessage("assistant", "答复"),
    ]
    msgs = rebuild_messages(stored)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["id"] == "c1"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].content == "结果"
    assert msgs[2].status == "success" and msgs[2].tool_call_id == "c1"


def test_rebuild_corrupted_envelope_raises():
    with pytest.raises(ValueError):
        rebuild_messages([
            StoredMessage("user", "问"),
            StoredMessage("assistant", None,
                          tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
            StoredMessage("tool", "not-json", tool_call_id="c1"),
            StoredMessage("assistant", "答"),
        ])


def test_merge_and_finalize():
    acc: dict[int, dict] = {}
    merge_tool_call_chunks(acc, [
        {"name": "query_faq", "args": "{\"keyword\": \"退", "id": "call_1", "index": 0},
    ])
    merge_tool_call_chunks(acc, [
        {"name": None, "args": "货\"}", "id": None, "index": 0},
        {"name": "query_order", "args": "{\"order_id\": \"1\"}", "id": "call_2", "index": 1},
    ])
    calls, broken = finalize_tool_calls(acc)
    assert broken is False
    assert calls == [
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"},
        {"name": "query_order", "args": {"order_id": "1"}, "id": "call_2", "type": "tool_call"},
    ]


def test_finalize_broken_json():
    acc: dict[int, dict] = {}
    merge_tool_call_chunks(acc, [
        {"name": "query_faq", "args": "{损坏", "id": "c1", "index": 0},
    ])
    calls, broken = finalize_tool_calls(acc)
    assert broken is True


def test_trim_drops_orphan_tool_group():
    # 历史里工具 turn 完整,但预算只够留最后两条 -> 头部 user 起的整个 turn 被丢
    history = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果很长" * 50, True, None, 4000), tool_call_id="c1"),
        StoredMessage("assistant", "物流答复" * 50),
        StoredMessage("user", "再问"),
        StoredMessage("assistant", "再答"),
    ]
    msgs = build_messages_with_tools("system", history, "现在呢", max_input_tokens=120)
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "再问"
    assert msgs[-1].content == "现在呢"
    # 不得出现孤儿 ToolMessage
    assert not any(isinstance(m, ToolMessage) for m in msgs)


def test_trim_keeps_complete_tool_group_when_budget_allows():
    history = [
        StoredMessage("user", "查物流"),
        StoredMessage("assistant", None,
                      tool_calls=[{"name": "query_logistics", "args": {"order_id": "1"},
                                   "id": "c1", "type": "tool_call"}]),
        StoredMessage("tool", wrap("结果", True, None, 4000), tool_call_id="c1"),
        StoredMessage("assistant", "答复"),
    ]
    msgs = build_messages_with_tools("system", history, "现在呢", max_input_tokens=2000)
    assert any(isinstance(m, ToolMessage) for m in msgs)
    assert msgs[1].content == "查物流"
```

Run: `uv run pytest tests/test_tool_chat_chain.py -v` —— FAIL。

- [ ] **Step 3: app/chains/tool_chat_chain.py 实现**

```python
import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

from app.sessions import StoredMessage
from app.tool_envelope import unwrap


def rebuild_messages(stored: list[StoredMessage]) -> list[BaseMessage]:
    """StoredMessage -> LangChain 消息;tool 行解 envelope 恢复 status;损坏抛 ValueError。"""
    out: list[BaseMessage] = []
    pending_calls: dict[str, dict] = {}
    for m in stored:
        if m.role == "user":
            out.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            calls = m.tool_calls or []
            pending_calls = {c["id"]: c for c in calls}
            out.append(AIMessage(content=m.content or "", tool_calls=calls))
        elif m.role == "tool":
            if m.tool_call_id not in pending_calls:
                raise ValueError("tool message without matching tool call")
            call = pending_calls.pop(m.tool_call_id)
            content, ok = unwrap(m.content or "")
            out.append(ToolMessage(content=content, tool_call_id=m.tool_call_id,
                                   name=call["name"],
                                   status="success" if ok else "error"))
        else:
            raise ValueError(f"unknown stored role: {m.role}")
        if not pending_calls and out and isinstance(out[-1], AIMessage) and m.role == "assistant" and not m.tool_calls:
            pending_calls = {}
    if pending_calls:
        raise ValueError("tool calls without tool results")
    return out


def merge_tool_call_chunks(acc: dict[int, dict], chunks: list[dict]) -> None:
    for ch in chunks:
        idx = ch.get("index", 0)
        slot = acc.setdefault(idx, {"name": None, "args": "", "id": None})
        if ch.get("name"):
            slot["name"] = ch["name"]
        if ch.get("id"):
            slot["id"] = ch["id"]
        if ch.get("args"):
            slot["args"] += ch["args"]


def finalize_tool_calls(acc: dict[int, dict]) -> tuple[list[dict], bool]:
    calls: list[dict] = []
    for idx in sorted(acc):
        slot = acc[idx]
        try:
            args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            return [], True
        if not isinstance(args, dict) or not slot["name"] or not slot["id"]:
            return [], True
        calls.append({"name": slot["name"], "args": args, "id": slot["id"],
                      "type": "tool_call"})
    return calls, False


def _drop_broken_head(messages: list[BaseMessage]) -> list[BaseMessage]:
    """裁剪后修复:首条(非 system)必须是 human;tool 组不完整则从头部丢整个 turn。"""
    rest = list(messages)
    while rest:
        head = rest[0]
        if isinstance(head, HumanMessage) and _tool_groups_complete(rest):
            break
        # 丢弃一个完整 turn:从头部到下一条 HumanMessage 之前
        nxt = next((i for i in range(1, len(rest)) if isinstance(rest[i], HumanMessage)),
                   len(rest))
        rest = rest[nxt:]
    return rest


def _tool_groups_complete(messages: list[BaseMessage]) -> bool:
    i = 0
    while i < len(messages):
        m = messages[i]
        if isinstance(m, AIMessage) and m.tool_calls:
            ids = [c["id"] for c in m.tool_calls]
            j = i + 1
            seen: list[str] = []
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                seen.append(messages[j].tool_call_id)
                j += 1
            if sorted(seen) != sorted(ids):
                return False
            i = j
        else:
            i += 1
    return True


def build_messages_with_tools(system_prompt: str, history: list[StoredMessage],
                              current_input: str, max_input_tokens: int) -> list[BaseMessage]:
    from app.chains.chat_chain import check_input_budget

    check_input_budget(system_prompt, current_input, max_input_tokens)
    system = SystemMessage(content=system_prompt)
    history_msgs = rebuild_messages(history)
    current = HumanMessage(content=current_input)
    trimmed = trim_messages(
        [system, *history_msgs, current],
        max_tokens=max_input_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False,
    )
    fixed = [trimmed[0], *_drop_broken_head(trimmed[1:])]
    _validate(fixed, current_input, max_input_tokens)
    return fixed


def _validate(messages: list[BaseMessage], current_input: str, max_input_tokens: int) -> None:
    if not messages or not isinstance(messages[0], SystemMessage):
        raise RuntimeError("trimmed messages lost the system prompt")
    if not isinstance(messages[-1], HumanMessage) or messages[-1].content != current_input:
        raise RuntimeError("trimmed messages lost the current human message")
    if len(messages) > 1 and not isinstance(messages[1], HumanMessage):
        raise RuntimeError("trimmed messages must start with a human message after system")
    if not _tool_groups_complete(messages[1:]):
        raise RuntimeError("tool group incomplete after repair")
    if count_tokens_approximately(messages) > max_input_tokens:
        raise RuntimeError("trimmed messages still exceed input token budget")


def fit_tool_context(messages: list[BaseMessage], max_input_tokens: int,
                     protected_from: int) -> list[BaseMessage] | None:
    """第二次调用预算。messages = [system, ...历史..., 当前human, AIMessage(tool_calls), ToolMessage...];
    protected_from = 当前 human 的下标,该下标起(含)的尾部不可拆分/删除。
    先丢最旧完整历史 turn;仍超限返回 None(调用方发 tool_context_too_long)。"""
    result = list(messages)
    while count_tokens_approximately(result) > max_input_tokens:
        # 只允许在 [1, protected_from) 区间丢完整历史 turn(system 在 0,不动)
        drop_at = next((i for i in range(1, protected_from)
                        if isinstance(result[i], HumanMessage)), None)
        if drop_at is None:
            return None
        nxt = next((i for i in range(drop_at + 1, protected_from)
                    if isinstance(result[i], HumanMessage)), protected_from)
        del result[drop_at:nxt]
        protected_from -= nxt - drop_at
    return result
```

注:去掉了 Ch01 的 `end_on="human"`(历史尾部可能是 tool/assistant);`start_on="human"` 保留。Ch01 的 `build_chat_messages` 保留给无工具纯聊天?不——Ch02 统一走 `build_messages_with_tools`,`chat_chain.build_chat_messages` 删除,其测试 `test_chat_chain.py` 改为测新函数(用例等价迁移:预算超限 422、保留 system/current、丢最旧 turn)。

- [ ] **Step 4: 编排失败测试**

`tests/test_orchestration.py`:

```python
import pytest

from app.main import create_app  # T6 才改 main;本任务直接用 ChatService 装配
from app.services.chat_service import ChatService
from app.sessions import InMemorySessionStore, StoredMessage
from app.tool_envelope import wrap
from app.tools.business import MOCK_TOOLS
from app.tools.executor import ToolExecutor, ToolRegistry
from tests.conftest import TEST_USER_ID, FakeStreamModel, make_settings

TOOL_CHUNKS = [
    {"name": "query_order", "args": "{\"order_id\": \"10", "id": "call_1", "index": 0},
    {"name": None, "args": "01\"}", "id": None, "index": 0},
]


def make_service(script, tools=None):
    settings = make_settings()
    store = InMemorySessionStore(10, 100, 8000)
    model = FakeStreamModel(script)
    factory_calls = []

    def toolset_factory(sid: str):
        factory_calls.append(sid)
        return tools if tools is not None else MOCK_TOOLS

    service = ChatService(store, model, settings, "system", toolset_factory)
    return service, model, store, factory_calls


async def collect(service, turn):
    return [e async for e in service.stream(turn)]


async def test_no_tool_single_call():
    service, model, store, factory_calls = make_service(["你", "好"], tools=[])
    turn = await service.prepare(TEST_USER_ID, None, "在吗")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    assert types == ["SessionEvent", "DeltaEvent", "DeltaEvent", "DoneEvent"]
    assert model.received_tools == [None]  # 工具集为空时不 bind
    assert factory_calls == [turn.session_id]


async def test_tool_call_full_sequence():
    script = [
        ("tool", TOOL_CHUNKS),
        ("then", ["物流", "在", "路上"]),
    ]
    service, model, store, factory_calls = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "订单 1001 呢")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    assert types == ["SessionEvent", "ToolStartEvent", "ToolEndEvent",
                     "DeltaEvent", "DeltaEvent", "DeltaEvent", "DoneEvent"]
    start = events[1]
    assert start.name == "query_order" and start.tool_call_id == "call_1"
    assert start.args == {"order_id": "1001"}
    end = events[2]
    assert end.ok is True and end.summary
    # 第二次调用未绑工具
    assert model.received_tools[0] == ["query_order", "query_product", "query_logistics"]
    assert model.received_tools[1] is None
    # 落库 4 条:user / assistant(tool_calls) / tool / assistant
    snap = await store.snapshot(turn.session_id)
    assert [m.role for m in snap] == ["user", "assistant", "tool", "assistant"]
    assert snap[1].tool_calls[0]["name"] == "query_order"
    from app.tool_envelope import unwrap
    body, ok2 = unwrap(snap[2].content)
    assert ok2 is True and "order_id" in body
    assert snap[3].content == "物流在路上"


async def test_invalid_tool_call_frame():
    script = [("tool", [{"name": "query_order", "args": "{损坏", "id": "c1", "index": 0}])]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert type(events[-1]).__name__ == "ErrorEvent"
    assert events[-1].code == "invalid_tool_call"
    assert await store.snapshot(turn.session_id) == []


async def test_too_many_tool_calls_rejected():
    chunks = [
        {"name": "query_order", "args": "{\"order_id\": \"1\"}", "id": f"c{i}", "index": i}
        for i in range(6)
    ]
    service, model, store, _ = make_service([("tool", chunks)])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert events[-1].code == "invalid_tool_call"


async def test_duplicate_create_ticket_rejected():
    chunks = [
        {"name": "create_ticket", "args": "{}", "id": "c0", "index": 0},
        {"name": "create_ticket", "args": "{}", "id": "c1", "index": 1},
    ]
    service, model, store, _ = make_service([("tool", chunks)])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert events[-1].code == "invalid_tool_call"


async def test_first_call_length_finish_no_tools_executed():
    script = [("tool", TOOL_CHUNKS), ("finish", "length")]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert events[-1].code == "output_too_long"
    assert await store.snapshot(turn.session_id) == []


async def test_tool_error_still_answers():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def broken(x: str) -> str:
        """总是失败"""
        raise ValueError("db down")

    script = [
        ("tool", [{"name": "broken", "args": "{\"x\": \"1\"}", "id": "c1", "index": 0}]),
        ("then", ["查询失败,转人工"]),
    ]
    service, model, store, _ = make_service(script, tools=[broken])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    assert types == ["SessionEvent", "ToolStartEvent", "ToolEndEvent",
                     "DeltaEvent", "DoneEvent"]
    assert events[2].ok is False
    snap = await store.snapshot(turn.session_id)
    from app.tool_envelope import unwrap
    body, ok = unwrap(snap[2].content)
    assert ok is False and "db down" not in body  # 脱敏


async def test_second_call_context_contains_tool_results():
    script = [("tool", TOOL_CHUNKS), ("then", ["答"])]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "查一下")
    await collect(service, turn)
    second_messages = model.received[1]
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in second_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "call_1"
    ai_with_calls = [m for m in second_messages
                     if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", None)]
    assert ai_with_calls and ai_with_calls[0].tool_calls[0]["id"] == "call_1"


async def test_lock_recreated_lazily_for_existing_session():
    """模拟重启:新 ChatService(锁表为空)对已有 session 仍能串行。"""
    service1, model, store, _ = make_service(["一"])
    turn = await service1.prepare(TEST_USER_ID, None, "hi")
    await collect(service1, turn)
    service2, model2, _, _ = make_service(["二"])
    service2._store = store  # 同一存储,新锁表
    turn2 = await service2.prepare(TEST_USER_ID, turn.session_id, "again")
    events = await collect(service2, turn2)
    assert type(events[-1]).__name__ == "DoneEvent"
    assert model2.received[0]  # 历史被读取
```

Run: `uv run pytest tests/test_orchestration.py -v` —— FAIL。

- [ ] **Step 5: chat_service.py 重写**

替换 `app/services/chat_service.py` 全文:

```python
import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Union

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.chains.tool_chat_chain import (
    build_messages_with_tools,
    finalize_tool_calls,
    fit_tool_context,
    merge_tool_call_chunks,
)
from app.config import Settings
from app.errors import MessageTooLongError, SessionNotFoundError
from app.sessions import SessionStore, StoredMessage
from app.tool_envelope import wrap
from app.tools.executor import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionEvent:
    session_id: str


@dataclass(frozen=True)
class DeltaEvent:
    content: str


@dataclass(frozen=True)
class ToolStartEvent:
    tool_call_id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolEndEvent:
    tool_call_id: str
    name: str
    ok: bool
    summary: str


@dataclass(frozen=True)
class DoneEvent:
    pass


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str


ChatEvent = Union[SessionEvent, DeltaEvent, ToolStartEvent, ToolEndEvent, DoneEvent, ErrorEvent]


class SessionLockRegistry:
    """懒创建 session 锁;持有+等待计数归零后删除条目。"""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    async def acquire(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        self._users[session_id] = self._users.get(session_id, 0) + 1
        await lock.acquire()
        return lock

    def release(self, session_id: str) -> None:
        lock = self._locks[session_id]
        lock.release()
        self._users[session_id] -= 1
        if self._users[session_id] == 0:
            del self._users[session_id]
            del self._locks[session_id]


@dataclass
class PreparedTurn:
    session_id: str
    user_text: str
    messages: list[BaseMessage]
    lock_key: str
    released: bool = False


class ChatService:
    def __init__(self, store: SessionStore, model: Any, settings: Settings,
                 system_prompt: str, toolset_factory: Callable[[str], list] | None = None):
        self._store = store
        self._model = model
        self._settings = settings
        self._system_prompt = system_prompt
        self._toolset_factory = toolset_factory
        self._locks = SessionLockRegistry()

    async def prepare(self, user_id: str, session_id: str | None, message: str) -> PreparedTurn:
        if len(message) > self._settings.max_message_chars:
            raise MessageTooLongError("message exceeds MAX_MESSAGE_CHARS")
        if session_id is None:
            sid = await self._store.create(user_id)
        else:
            sid = session_id
            if not await self._store.exists(sid, user_id):
                raise SessionNotFoundError("session not found")
        await self._locks.acquire(sid)
        try:
            history = await self._store.snapshot(sid)
            messages = build_messages_with_tools(
                self._system_prompt, history, message, self._settings.max_input_tokens
            )
        except Exception:
            self._locks.release(sid)
            raise
        return PreparedTurn(sid, message, messages, sid)

    def release_turn(self, turn: PreparedTurn) -> None:
        if turn.released:
            return
        turn.released = True
        self._locks.release(turn.lock_key)

    async def stream(self, turn: PreparedTurn) -> AsyncIterator[ChatEvent]:
        tools = self._toolset_factory(turn.session_id) if self._toolset_factory else []
        registry = ToolRegistry(tools)
        executor = ToolExecutor(registry, self._settings.tool_timeout_seconds,
                                self._settings.tool_max_retries,
                                self._settings.max_tool_result_chars)
        first_model = self._model.bind_tools(registry.tools) if tools else self._model
        agen = first_model.astream(turn.messages)
        visible_chars = 0
        committed = False
        try:
            yield SessionEvent(turn.session_id)
            text_parts: list[str] = []
            acc: dict[int, dict] = {}
            finish_reason: str | None = None
            try:
                async for chunk in agen:
                    meta = getattr(chunk, "response_metadata", None) or {}
                    if meta.get("finish_reason"):
                        finish_reason = meta["finish_reason"]
                    chunks = getattr(chunk, "tool_call_chunks", None) or []
                    if chunks:
                        merge_tool_call_chunks(acc, chunks)
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if not text:
                        continue
                    visible_chars += len(text)
                    if visible_chars > self._settings.max_message_chars:
                        yield ErrorEvent("output_too_long", "回复超出长度限制")
                        return
                    text_parts.append(text)
                    yield DeltaEvent(text)
            except Exception as exc:
                logger.warning("chat first call upstream error: %s", type(exc).__name__)
                yield ErrorEvent("upstream_error", "上游模型暂时不可用")
                return
            finally:
                with contextlib.suppress(Exception):
                    await agen.aclose()

            if finish_reason == "length":
                yield ErrorEvent("output_too_long", "回复超出长度限制")
                return

            calls, broken = finalize_tool_calls(acc)
            if broken or not self._tool_calls_legal(calls):
                yield ErrorEvent("invalid_tool_call", "工具调用申请不合法")
                return

            tool_messages: list[ToolMessage] = []
            ai_with_calls: AIMessage | None = None
            if calls:
                ai_with_calls = AIMessage(content="".join(text_parts), tool_calls=calls)
                for call in calls:
                    yield ToolStartEvent(call["id"], call["name"], call["args"])
                    outcome = await executor.execute(call)
                    tool_messages.append(outcome.message)
                    logger.info("tool %s ok=%s retries=%d ms=%d err=%s",
                                outcome.record.name, outcome.record.ok,
                                outcome.record.retry_count, outcome.record.duration_ms,
                                outcome.record.error_type)
                    yield ToolEndEvent(call["id"], call["name"], outcome.record.ok,
                                       outcome.message.content[:80])
                second_messages = fit_tool_context(
                    [*turn.messages, ai_with_calls, *tool_messages],
                    self._settings.max_input_tokens,
                    protected_from=len(turn.messages) - 1,  # 当前 human 的下标(turn.messages 末位)
                )
                if second_messages is None:
                    yield ErrorEvent("tool_context_too_long", "工具结果超出上下文预算")
                    return
                final_parts: list[str] = []
                finish2: str | None = None
                agen2 = self._model.astream(second_messages)  # 不绑工具,单轮收敛
                try:
                    async for chunk in agen2:
                        meta = getattr(chunk, "response_metadata", None) or {}
                        if meta.get("finish_reason"):
                            finish2 = meta["finish_reason"]
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if not text:
                            continue
                        visible_chars += len(text)
                        if visible_chars > self._settings.max_message_chars:
                            yield ErrorEvent("output_too_long", "回复超出长度限制")
                            return
                        final_parts.append(text)
                        yield DeltaEvent(text)
                except Exception as exc:
                    logger.warning("chat second call upstream error: %s", type(exc).__name__)
                    yield ErrorEvent("upstream_error", "上游模型暂时不可用")
                    return
                finally:
                    with contextlib.suppress(Exception):
                        await agen2.aclose()
                if finish2 == "length":
                    yield ErrorEvent("output_too_long", "回复超出长度限制")
                    return
                final_text = "".join(final_parts)
            else:
                final_text = "".join(text_parts)

            if not final_text.strip():
                yield ErrorEvent("empty_response", "上游返回了空回复")
                return

            stored = [StoredMessage("user", turn.user_text)]
            if calls:
                stored.append(StoredMessage("assistant", "".join(text_parts) or None,
                                            tool_calls=calls))
                for tm in tool_messages:
                    stored.append(StoredMessage(
                        "tool",
                        wrap(tm.content, tm.status != "error",
                             None if tm.status != "error" else "tool_error",
                             self._settings.max_tool_result_chars),
                        tool_call_id=tm.tool_call_id,
                    ))
            stored.append(StoredMessage("assistant", final_text))
            await asyncio.shield(self._store.commit_turn(turn.session_id, stored))
            committed = True
            yield DoneEvent()
        finally:
            self.release_turn(turn)

    def _tool_calls_legal(self, calls: list[dict]) -> bool:
        if len(calls) > self._settings.max_tool_calls_per_turn:
            return False
        ids = [c["id"] for c in calls]
        if len(set(ids)) != len(ids):
            return False
        if any(not c["id"] or len(c["id"]) > 64 for c in calls):
            return False
        if sum(1 for c in calls if c["name"] == "create_ticket") > 1:
            return False
        return True
```

注:`committed` 标志仅备调试;`asyncio.shield` 保护 commit——客户端取消时 shield 内协程继续跑完(内存 store 立即返回;DB store 在线程中完成事务),`finally` 在 shield 结束后释放锁。取消语义:shield 等待期间若外层被取消,`await shield(...)` 抛 CancelledError 但内层事务继续;release 在 finally 执行时内层可能仍在跑——因此 DB store 的 `commit_turn` 内部用 `asyncio.to_thread`,shield 等待的是协程;取消场景锁在事务线程结束后释放的实现:`commit_turn` 协程体本身完成才标志着事务结束,而 shield 抛出时协程未被取消仍在后台跑。为严格满足「等事务结束再放锁」,实现为:

```python
            commit_task = asyncio.ensure_future(self._store.commit_turn(turn.session_id, stored))
            try:
                await asyncio.shield(commit_task)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await commit_task  # 等事务落地
                raise
            committed = True
            yield DoneEvent()
```

finally 中的 `release_turn` 在 `await commit_task` 之后才会执行(取消重新抛出后 finally 运行,此时任务已结束)。采用此写法替换上面单行 shield。

- [ ] **Step 6: 跑通 + 全量回归**

Run: `uv run pytest tests/test_tool_chat_chain.py tests/test_orchestration.py -v` —— PASS。
Run: `uv run pytest` —— 全绿(test_chat_chain.py 已迁移;test_chat_service.py 的并发/锁断言经 SessionLockRegistry 保持语义)。

注意 `test_chat_service.py` 中引用 `turn.lock.locked()` 的断言改为检查 registry 内部或经由并发行为断言(不直接摸锁对象)。

- [ ] **Step 7: Commit**

```bash
git add app/chains/tool_chat_chain.py app/services/chat_service.py app/chains/chat_chain.py tests/
git commit -m "feat(ch02): single-turn tool orchestration in chat service"
```

---

### Task 6: 路由新帧 + main.py AppRuntime + prompt 增补 + ch01 API 测试适配

**Files:**
- Modify: `app/routers/chat.py`(tool_start/tool_end 帧)
- Modify: `app/main.py`(AppRuntime、--factory、删模块级 app)
- Modify: `app/prompts/service.py`(追加工具规则)
- Modify: `tests/conftest.py`(`make_runtime`、`make_app` 辅助)
- Modify: `tests/test_chat_api.py`、`tests/test_chat_page.py`、`tests/test_extract_api.py`(create_app 调用适配)
- Create: `tests/test_chat_api_tools.py`

**Interfaces:**
- Consumes: T5 全部
- Produces: `AppRuntime(store, toolset_factory)`;`create_app(settings=None, model=None, runtime=None)`;SSE `tool_start` / `tool_end` 帧

- [ ] **Step 1: 失败测试**

`tests/test_chat_api_tools.py`:

```python
import httpx

from app.main import create_app
from tests.conftest import TEST_USER_ID, FakeStreamModel, make_runtime, make_settings
from tests.test_chat_api import parse_frames, post_stream

TOOL_CHUNKS = [
    {"name": "query_order", "args": "{\"order_id\": \"1001\"}", "id": "call_1", "index": 0},
]


def make_app(script):
    return create_app(settings=make_settings(), model=FakeStreamModel(script),
                      runtime=make_runtime())


async def test_tool_frames_over_sse():
    app = make_app([("tool", TOOL_CHUNKS), ("then", ["答", "复"])])
    payload = {"user_id": TEST_USER_ID, "message": "查订单"}
    status, lines = await post_stream(app, payload)
    assert status == 200
    frames = parse_frames(lines)
    assert frames[0]["type"] == "session"
    assert frames[1]["type"] == "tool_start"
    assert frames[1]["name"] == "query_order" and frames[1]["args"] == {"order_id": "1001"}
    assert frames[1]["tool_call_id"] == "call_1"
    assert frames[2]["type"] == "tool_end" and frames[2]["ok"] is True
    assert frames[3]["type"] == "delta" and frames[3]["content"] == "答"
    assert frames[-1] == "[DONE]"


async def test_tool_end_summary_capped_80():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def verbose(x: str) -> str:
        """超长返回"""
        return "长" * 500

    app = create_app(settings=make_settings(), model=FakeStreamModel(
        [("tool", [{"name": "verbose", "args": "{\"x\": \"1\"}", "id": "c1", "index": 0}]),
         ("then", ["答"])]
    ), runtime=make_runtime(tools=[verbose]))
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    frames = parse_frames(lines)
    end = next(f for f in frames if isinstance(f, dict) and f.get("type") == "tool_end")
    assert len(end["summary"]) <= 80


async def test_invalid_user_id_422():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream",
                                 json={"user_id": "not-a-uuid", "message": "hi"})
        assert resp.status_code == 422


async def test_user_mismatch_404():
    app = make_app(["答"])
    _, lines = await post_stream(app, {"user_id": TEST_USER_ID, "message": "hi"})
    sid = parse_frames(lines)[0]["session_id"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": "22222222-2222-2222-2222-222222222222",
            "session_id": sid, "message": "hi"})
        assert resp.status_code == 404


async def test_bad_session_id_form_422():
    app = make_app(["x"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/v1/chat/stream", json={
            "user_id": TEST_USER_ID, "session_id": "abc!!", "message": "hi"})
        assert resp.status_code == 422
```

(`test_tool_frames_over_sse` 里 mock 工具来自 `make_runtime()` 默认空工具集?不行——query_order 未注册会回灌 unknown_tool。`make_runtime()` 默认工具集改为 `MOCK_TOOLS`(无 DB 依赖);空工具集用 `make_runtime(tools=[])`。)

Run: `uv run pytest tests/test_chat_api_tools.py -v` —— FAIL。

- [ ] **Step 2: routers/chat.py 新帧**

`_EventStream._body` 增加分支:

```python
                elif isinstance(event, ToolStartEvent):
                    yield _sse({"type": "tool_start", "tool_call_id": event.tool_call_id,
                                "name": event.name, "args": event.args})
                elif isinstance(event, ToolEndEvent):
                    yield _sse({"type": "tool_end", "tool_call_id": event.tool_call_id,
                                "name": event.name, "ok": event.ok,
                                "summary": event.summary})
```

import 同步更新。

- [ ] **Step 3: main.py AppRuntime 化**

替换 `app/main.py` 中 `create_app` 及模块尾部:

```python
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool

from app.db import make_engine, make_session_factory, ping
from app.store_db import DbSessionStore
from app.tools.business import build_tools


@dataclass(frozen=True)
class AppRuntime:
    store: Any  # SessionStore 协议
    toolset_factory: Callable[[str], list[BaseTool]]


def _build_production_runtime(settings: Settings) -> AppRuntime:
    try:
        engine = make_engine(settings.database_url)
        ping(engine)
    except Exception as exc:
        raise RuntimeError(f"database ping failed: {type(exc).__name__}") from exc
    session_factory = make_session_factory(engine)

    def toolset_factory(session_id: str) -> list[BaseTool]:
        return build_tools(session_factory, int(session_id))

    return AppRuntime(store=DbSessionStore(session_factory, settings.max_message_chars),
                      toolset_factory=toolset_factory)


def create_app(settings: Settings | None = None, model: Any | None = None,
               runtime: AppRuntime | None = None) -> FastAPI:
    settings = settings or Settings()
    if model is None:
        model = ChatOpenAI(...)  # 同 Ch01
    if runtime is None:
        runtime = _build_production_runtime(settings)
    # system prompt / 提取 prompt 预算检查同 Ch01
    service = ChatService(runtime.store, model, settings, SERVICE_SYSTEM_PROMPT,
                          runtime.toolset_factory)
    app = FastAPI(title="wayhelp-ch02")
    app.state.settings = settings
    app.state.model = model
    app.state.store = runtime.store
    app.state.chat_service = service
    ...
```

删除文件末尾的 `app = create_app()`。启动命令改为 `uv run uvicorn app.main:create_app --factory`。

`tests/conftest.py` 追加:

```python
from app.main import AppRuntime
from app.sessions import InMemorySessionStore
from app.tools.business import MOCK_TOOLS

TEST_USER_ID = "11111111-1111-1111-1111-111111111111"


def make_runtime(tools=None, store=None):
    return AppRuntime(
        store=store or InMemorySessionStore(1000, 100, 8000),
        toolset_factory=lambda sid: list(MOCK_TOOLS if tools is None else tools),
    )
```

ch01 测试适配:`test_chat_api.py` / `test_chat_page.py` / `test_extract_api.py` 中 `create_app(settings=..., model=...)` 全部追加 `runtime=make_runtime(tools=[])`(ch01 用例不涉及工具,空工具集让行为与 ch01 一致);`test_chat_api.py` 的请求 JSON 加 `"user_id": TEST_USER_ID`。

- [ ] **Step 4: prompts/service.py 追加**

在第五条后追加:

```python
6. 查询订单、商品、物流、常见问题时,优先调用对应工具,不得编造查询结果。
7. 工具返回错误或未找到时,如实说明并引导顾客转人工;不得声称已经完成转人工。
8. 顾客明确要求人工,或问题超出工具能力时,调用 create_ticket 创建工单;只有收到成功结果后才能说「已转人工」并告知工单号。
9. 订单/商品/物流为演示数据的事实不主动强调,但不得承诺其与现实系统一致。
"""

TOOLS_SYSTEM_ADDENDUM = ""  # 占位不需要,直接并入上文
```

(实现:直接把 6/7/8/9 四条写进 SERVICE_SYSTEM_PROMPT 字符串。)

- [ ] **Step 5: 全量回归 + Commit**

Run: `uv run pytest` —— 全绿(含 ch01 回归)。

```bash
git add app/main.py app/routers/chat.py app/prompts/service.py tests/
git commit -m "feat(ch02): sse tool frames, AppRuntime factory, tool prompt rules"
```

---

### Task 7: 聊天页 user_id + 工具徽章(Vibe Coding)

**Files:**
- Modify: `app/static/chat.html`
- Modify: `tests/test_chat_page.py`(追加静态断言)

**Interfaces:**
- Consumes: T6 的 SSE 帧契约与请求协议
- Produces: 页面发送 `user_id`(localStorage `wayhelp_user_id`);assistant 气泡渲染工具徽章

按用户要求本任务走 Vibe Coding:直接改页面,不套 TDD;仅沿用 ch01 的静态断言模式加最低回归网。

- [ ] **Step 1: 静态断言(先行,防回归)**

`tests/test_chat_page.py` 追加:

```python
async def test_chat_page_tool_badge_hooks():
    from pathlib import Path
    html = (Path(__file__).parent.parent / "app" / "static" / "chat.html").read_text(encoding="utf-8")
    assert "wayhelp_user_id" in html
    assert "crypto.randomUUID" in html
    assert "tool_start" in html and "tool_end" in html
    assert "tool-badge" in html
```

- [ ] **Step 2: 页面改动(chat.html)**

1. user_id:脚本开头加

```javascript
  var userId = localStorage.getItem("wayhelp_user_id");
  if (!userId) {
    userId = crypto.randomUUID();
    localStorage.setItem("wayhelp_user_id", userId);
  }
```

fetch body 改 `JSON.stringify({ user_id: userId, session_id: sessionId, message: text })`。

2. 徽章样式(`<style>` 内,沿用 design token):

```css
  .tool-badges {
    display: flex; flex-wrap: wrap; gap: var(--space-2);
    margin-bottom: var(--space-2);
  }
  .tool-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px;
    font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.03em;
    border-radius: var(--radius-pill);
    background: var(--surface-warm); color: var(--fg-2);
    border: 1px solid var(--border-soft);
  }
  .tool-badge .dot { width: 6px; height: 6px; border-radius: 50%; }
  .tool-badge.running .dot { background: var(--warn); animation: blink 0.9s steps(1) infinite; }
  .tool-badge.ok .dot { background: var(--success); }
  .tool-badge.failed .dot { background: var(--danger); }
```

3. 渲染逻辑:assistant 气泡内 `.msg` 前插 `<div class="tool-badges">`;`tool_start` 时按 tool_call_id 建徽章(`running`,文本 `调用工具 {name}`),`tool_end` 时按 id 找徽章更新为 `ok/failed`(文本 `{name} 成功/失败`)。JS 骨架:

```javascript
  var TOOL_LABELS = { query_order: "查订单", query_product: "查商品",
    query_logistics: "查物流", query_faq: "查常见问题", create_ticket: "创建工单" };

  function addBadge(badgesEl, id, name) {
    var badge = document.createElement("span");
    badge.className = "tool-badge running";
    badge.dataset.toolCallId = id;
    badge.innerHTML = '<span class="dot"></span><span class="label"></span>';
    badge.querySelector(".label").textContent = "调用工具 " + (TOOL_LABELS[name] || name);
    badgesEl.appendChild(badge);
  }

  function settleBadge(badgesEl, id, name, ok) {
    var badge = badgesEl.querySelector('[data-tool-call-id="' + id + '"]');
    if (!badge) return;
    badge.classList.remove("running");
    badge.classList.add(ok ? "ok" : "failed");
    badge.querySelector(".label").textContent =
      (TOOL_LABELS[name] || name) + (ok ? " 成功" : " 失败");
  }
```

`send()` 内在 `addBubble("assistant","")` 后创建 badgesEl 并插入 row 的 `.msg` 之前;parseEvents 回调增加:

```javascript
          } else if (evt.type === "tool_start") {
            addBadge(badgesEl, evt.tool_call_id, evt.name);
            scrollToBottom();
          } else if (evt.type === "tool_end") {
            settleBadge(badgesEl, evt.tool_call_id, evt.name, evt.ok);
            scrollToBottom();
          }
```

建议问题 chips 更新为 `["订单 1001 的物流到哪了", "退货政策是什么", "帮我转人工处理"]`(对齐浏览器验收用例)。

- [ ] **Step 3: 跑静态断言 + 全量**

Run: `uv run pytest tests/test_chat_page.py -v` 与 `uv run pytest` —— 全绿。

- [ ] **Step 4: Commit**

```bash
git add app/static/chat.html tests/test_chat_page.py
git commit -m "feat(ch02): chat page anonymous user_id + tool trace badges"
```

---

### Task 8: seed 召回断言 + 真实模型浏览器验收 + 收尾

**Files:**
- Create: `tests/test_seed_recall.py`
- Modify: `README.md`(启动/验收命令)
- Modify: `dev-notes/ch02.md`(完结段)

**Interfaces:**
- Consumes: 全部前序任务
- Produces: seed 召回断言;验收操作手册

- [ ] **Step 1: seed 召回断言**

`tests/test_seed_recall.py`:

```python
"""faq seed 的标注式断言:验收 2 必须命中,验收 3 必须漏召回(spec §11)。"""

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

SEED_PATH = Path(__file__).resolve().parent.parent / "db" / "init" / "02-seed.sql"


@pytest.fixture()
def seeded_factory(db_engine):
    from app.db import make_session_factory
    with db_engine.connect() as conn:
        for table in ("messages", "tickets", "faq", "conversations"):
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            conn.execute(text(f"TRUNCATE TABLE {table}"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        conn.execute(text(SEED_PATH.read_text(encoding="utf-8")))
        conn.commit()
    return make_session_factory(db_engine)


def _count_like(sf, keyword):
    with sf() as s:
        return s.execute(text(
            "SELECT COUNT(*) FROM faq WHERE question LIKE :p OR answer LIKE :p"
        ), {"p": f"%{keyword}%"}).scalar()


async def test_seed_has_no_youfei_anywhere(db_engine):
    """02-seed.sql 全文件不得出现「邮费」二字。"""
    assert "邮费" not in SEED_PATH.read_text(encoding="utf-8")


async def test_return_policy_hit(seeded_factory):
    assert await asyncio.to_thread(_count_like, seeded_factory, "退货政策") >= 1


async def test_youfei_miss(seeded_factory):
    assert await asyncio.to_thread(_count_like, seeded_factory, "邮费") == 0
```

(02-seed.sql 为单条 INSERT 多行,`conn.execute(text(整文件))` 可行;若拆多语句则按 `;\n` 切分循环执行。)

Run: `uv run pytest tests/test_seed_recall.py -v` —— PASS。

- [ ] **Step 2: README 更新**

README「运行」一节改为:

```bash
docker compose up -d          # 启动 MySQL(首启自动建表+灌 faq seed)
uv run pytest                 # 测试(DB 用例需 Docker 在线)
uv run uvicorn app.main:create_app --factory   # 起服
# 浏览器打开 http://127.0.0.1:8000/
```

- [ ] **Step 3: 全量回归 + 起服验证**

Run: `uv run pytest` —— 全绿。
Run(后台):`uv run uvicorn app.main:create_app --factory`,curl 一帧验证:

```bash
curl -N -X POST localhost:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"11111111-1111-1111-1111-111111111111","message":"订单 1001 的物流到哪了"}'
```

预期帧序列含 tool_start(query_logistics)→ tool_end → delta → [DONE]。

- [ ] **Step 4: 浏览器验收(用户执行,我起服陪同)**

1. 「订单 1001 的物流到哪了」→ query_logistics 徽章成功,按返回作答
2. 「退货政策是什么」→ query_faq 徽章成功并作答
3. 「邮费是多少」→ query_faq 未找到,如实说明并引导转人工(漏召回记 dev-notes)
4. 「帮我转人工处理」→ create_ticket 成功徽章 + 工单号;SQL 抽查:

```bash
docker exec wayhelp-mysql mysql -uwayhelp -pwayhelp wayhelp \
  -e "select ticket_no, status from tickets; select id, status from conversations;"
```

- [ ] **Step 5: dev-notes 完结段 + Commit**

```bash
git add tests/test_seed_recall.py README.md dev-notes/ch02.md
git commit -m "test(ch02): seed recall assertions + readme runbook"
```

---

## Self-Review 结论(计划落盘前已跑)

- **Spec 覆盖**:§4 DDL/seed→T1/T2/T8;§5 store→T3;§6 工具→T4;§7 编排/SSE→T5/T6;§8 prompt→T6;§9 配置/Docker/AppRuntime→T1/T6;§10 错误→各任务错误路径 + T5/T6 测试;§11 测试/验收→T2-T8;徽章→T7。无缺口。
- **类型一致性**:`StoredMessage` 四字段全计划一致;`ChatService(store, model, settings, system_prompt, toolset_factory)` 五参在 T5/T6/main/conftest 一致;`make_runtime(tools=...)` 默认 MOCK_TOOLS 在 T6 测试与 conftest 一致;`wrap(content, ok, error_code, max_chars)` 签名在 T3/T4/T5 一致。
- **已修问题**:T3 初稿 `validate_turn` 未校验 tool_call_id 长度(补 64 字符上限);T5 初稿 shield 单行写法不满足「取消期间等事务落地再放锁」,改 commit_task 两段式;T6 `make_runtime()` 默认空工具集导致 `query_order` unknown_tool,改默认 MOCK_TOOLS。
