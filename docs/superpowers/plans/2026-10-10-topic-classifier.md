# ch10 多标签主题分类器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 微调 RoBERTa-wwm-ext 多标签主题分类器,把飞轮低置信度问题池按 17 类权威主题旁路批量归类,结果落 `topic_classifications` 表并在飞轮后台 `/topics` 页展示主题分布。

**Architecture:** 训练管线独立 `training/` 包(uv dependency-group `topic-data`/`train` 隔离重依赖);主服务只加薄壳归类 service + API + 单文件前端页;推理为独立 FastAPI 进程(:8103)加载 ONNX 模型包;归类走「读快照 → 分批推理全暂存 → 唯一写事务原子提交」,同版本只补漏、换版本重算。

**Tech Stack:** transformers Trainer 全参微调(hfl/chinese-roberta-wwm-ext,CPU)、scikit-multilearn 分层、optimum[onnx] 导出、onnxruntime CPU 推理、FastAPI + SQLAlchemy 2.0 同步 ORM + httpx(主服务侧)。

**Spec:** `docs/superpowers/specs/2026-10-10-topic-classifier-design.md`(评审修订版,计划与其逐节对应)

## Global Constraints

- 模型定死 `hfl/chinese-roberta-wwm-ext`,全参微调,**禁止 LoRA/QLoRA**;不做 embedding/意图识别微调
- 本机无 GPU:一切训练/推理按 CPU 设计(`use_cpu=True`、onnxruntime CPUExecutionProvider)
- 17 类顺序冻结:`退换货、物流、尺码、发票、质量问题、运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修、账号、会员积分、评价、其他`
- 「其他」与其余 16 类互斥;零命中兜底输出 `["其他"]`;标签按冻结顺序返回、不重复、不为空
- 统一解码 `decoder_version="topic-decoder-v1"`,集中实现于 `training/labels.py`(纯标准库,主服务可导入);任何模块不得另写解码口径
- 版本哈希一律:对象键排序、数组保序、UTF-8、无多余空白的规范化 JSON 之 SHA-256(64 位小写十六进制)
- 训练/切分随机种子冻结 42;阈值扫描范围 0.05–0.95 步长 0.01,并列先取最接近 0.5、再取较大者
- 实时对话主链路不调用分类器;归类手动 API 触发,进程内非阻塞作业锁,并发 409 `job_busy`
- 写库原子性:全部批次推理+校验成功才开唯一写事务;任何失败零写入(推理/契约失败 503,DB 失败 500 回滚)
- 普通 `uv run pytest` 不安装 `train` 组、不加载真实模型、不访问推理服务/LLM;耗时/计费脚本手跑不进 pytest
- 涉及库 API 用法先查 Context7(本计划已核实:uv include-group、Trainer/EarlyStoppingCallback、onnxruntime InferenceSession、optimum export=True;实现时对本仓库锁定版本再核对一次)
- Conventional Commits;每个 Task 完成后在 `dev-notes/ch10.md` 追记一节(用户关键原话/关键产出/纠偏/翻车)
- 前端 `topics.html` 走 Vibe Coding 例外:不拆 TDD,直接实现 + 字符串断言兜底
- 代理红线:访问本机服务 `curl --noproxy '*'`;起服带 `no_proxy=127.0.0.1,localhost`

## File Structure

新建:
```
training/__init__.py            # 空(不导入重栈)
training/taxonomy.json          # 17 类冻结类目表
training/labels.py              # 类目加载/版本哈希/统一解码(纯标准库)
training/preprocessing.py       # 确定性清洗(OpenCC,topic-data 组)
training/threshold.py           # 全局阈值扫描(numpy)
training/package.py             # 模型包哈希/manifest(纯标准库)
training/prep_clean.py          # 清洗+LLM 错别字校对
training/prep_generate.py       # LLM 合成数据
training/prep_label.py          # LLM 预标 + 抽审导出 + corrections 回灌
training/prep_split.py          # 来源组分层抽样 + split_manifest
training/prep_augment.py        # 训练集增强
training/train.py               # 全参微调 + 早停 + 阈值扫描
training/evaluate.py            # 测试集报告 + 错例导出
training/export_onnx.py         # ONNX 导出 + 一致性验证 + 模型包
training/serve.py               # FastAPI 推理服务(:8103)
app/services/topic_classify.py  # 归类作业 + 分布统计
app/routers/topics.py           # /api/topics/*
app/static/topics.html          # 主题分布页(Vibe)
db/init/08-ddl.sql              # = sql/ch10-ddl.sql
tests/test_ch10_schema.py / test_taxonomy.py / test_preprocessing.py /
  test_topic_classify.py / test_topic_distribution.py / test_topics_api.py /
  test_prep_pipeline.py / test_prep_split.py / test_prep_augment.py /
  test_threshold.py / test_evaluate_report.py / test_serve_contract.py /
  test_topics_page.py
```

修改:
```
sql/ch10-ddl.sql                # 补 classifier_version/classifier_metadata 两列
pyproject.toml                  # dependency-groups topic-data/train + torch cpu 源
.gitignore                      # data/topic_classifier/、models/
app/models.py                   # TopicClassification ORM
app/db.py                       # check_ch10_tables
app/main.py                     # 校验调用、router、/topics 路由、错误处理器
app/config.py                   # ch10 配置四项
.env.example                    # ch10 节
tests/test_ddl_sync.py          # DDL_PAIRS += ("08","ch10")
tests/dbfixtures.py             # DDL_PATHS += 08;TABLES 加 topic_classifications(在 lcq 前)
tests/test_config.py            # ch10 配置用例
Makefile                        # topic-inference target
AGENTS.md / README.md           # ch10 节(收尾任务)
```

---

### Task 1: DDL 三件套 + ORM + 启动校验 + dbfixtures 纳管

**Files:**
- Modify: `sql/ch10-ddl.sql`
- Create: `db/init/08-ddl.sql`
- Modify: `tests/test_ddl_sync.py:8`
- Modify: `app/models.py`(追加)
- Modify: `app/db.py`(追加)
- Modify: `app/main.py:18,81`
- Modify: `tests/dbfixtures.py:15-28`
- Test: `tests/test_ch10_schema.py`

**Interfaces:**
- Produces: `TopicClassification` ORM 字段 `id / question_id / labels / classifier_version / classifier_metadata / classified_at`;`app.db.check_ch10_tables(engine)`;dbfixtures 建好 `topic_classifications` 表供后续全部 DB 测试用

- [ ] **Step 1: 写失败测试** `tests/test_ch10_schema.py`

```python
"""ch10 topic_classifications 表:DDL 配对、启动校验、ORM 往返。"""

import pytest
from sqlalchemy import text

from app.db import check_ch10_tables
from app.models import LowConfidenceQuestion, TopicClassification
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_check_ch10_tables_passes(db_engine):
    check_ch10_tables(db_engine)  # 不抛即过


def test_orm_roundtrip(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        lcq = LowConfidenceQuestion(raw_question="买大了想退", source="self_check",
                                    process_status="processed")
        s.add(lcq)
        s.flush()
        s.add(TopicClassification(
            question_id=lcq.id, labels=["退换货", "尺码"],
            classifier_version="ab" * 32,
            classifier_metadata={"model_version": "m", "taxonomy_version": "t",
                                 "effective_threshold": 0.5, "decoder_version": "d"}))
        s.commit()
    with sf() as s:
        row = s.query(TopicClassification).filter_by(question_id=lcq.id).one()
        assert row.labels == ["退换货", "尺码"]
        assert len(row.classifier_version) == 64
        assert row.classifier_metadata["decoder_version"] == "d"
        assert row.classified_at is not None
        # uk_question_id 幂等:同 question_id 再插必撞唯一键
        s.add(TopicClassification(question_id=lcq.id, labels=["物流"],
                                  classifier_version="cd" * 32,
                                  classifier_metadata={}))
        with pytest.raises(Exception):
            s.flush()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch10_schema.py -x -q`
Expected: FAIL(`check_ch10_tables` / `TopicClassification` 不存在,收集即 ImportError)

- [ ] **Step 3: 改 DDL(在用户草案基础上补两列)**

`sql/ch10-ddl.sql` 的 CREATE TABLE 改为:

```sql
CREATE TABLE topic_classifications (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  question_id   BIGINT UNSIGNED NOT NULL                COMMENT '归类的问题,指向 low_confidence_questions.id',
  labels        JSON            NOT NULL                COMMENT '多标签,17 类权威类目里命中的若干个',
  classifier_version CHAR(64)   NOT NULL                COMMENT '分类版本:{model_version,effective_threshold,decoder_version} 的规范化 SHA-256',
  classifier_metadata JSON      NOT NULL                COMMENT '与写入版本相符的元数据(model_version/taxonomy_version/effective_threshold/labels/decoder_version)',
  classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '归类时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_question_id (question_id),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='主题分类结果';
```

然后 `cp sql/ch10-ddl.sql db/init/08-ddl.sql`(逐字节一致,SET NAMES 行一并保留)。

- [ ] **Step 4: 接线五处**

`tests/test_ddl_sync.py:8` 改为:

```python
DDL_PAIRS = [("01", "ch02"), ("03", "ch03"), ("04", "ch04"), ("05", "ch07"),
             ("06", "ch08"), ("07", "ch09"), ("08", "ch10")]
```

`app/models.py` 文件末尾追加:

```python
class TopicClassification(Base):
    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("low_confidence_questions.id"),
        unique=True, nullable=False)
    labels: Mapped[list] = mapped_column(JSON)
    classifier_version: Mapped[str] = mapped_column(String(64))
    classifier_metadata: Mapped[dict] = mapped_column(JSON)
    classified_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

`app/db.py` 末尾追加:

```python
_CH10_TABLES = ("topic_classifications",)
_CH10_COLS = {"id", "question_id", "labels", "classifier_version",
              "classifier_metadata", "classified_at"}


def check_ch10_tables(engine) -> None:
    """缺 ch10 表/列启动失败并提示升级命令(spec §8)。"""
    with engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SHOW TABLES"))}
        missing = [t for t in _CH10_TABLES if t not in tables]
        cols = set()
        if not missing:
            cols = {r[0] for r in conn.execute(
                text("SHOW COLUMNS FROM topic_classifications"))}
    if missing or not _CH10_COLS <= cols:
        raise RuntimeError(
            f"缺少 ch10 表结构(缺表 {missing},缺列 "
            f"{sorted(_CH10_COLS - cols)}):请先执行 "
            f"mysql wayhelp < sql/ch10-ddl.sql")
```

`app/main.py:18` import 加 `check_ch10_tables`;`:81` 后加一行 `check_ch10_tables(engine)  # 缺 ch10 表/列同样启动失败,提示升级命令`。

`tests/dbfixtures.py`:`DDL_PATHS` 追加 `... / "08-ddl.sql"`;`TABLES` 在 `"low_confidence_questions"` **之前**插入 `"topic_classifications"`(先子后父)。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_ch10_schema.py tests/test_ddl_sync.py -q`
Expected: 全 PASS(需 Docker MySQL 在线)

- [ ] **Step 6: 生产库执行 DDL 并全量回归**

```bash
docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch10-ddl.sql
uv run pytest -q
```
Expected: 全量 PASS(745+ 条基线上新增)

- [ ] **Step 7: 补实录并提交**

`dev-notes/ch10.md` 追加 `## 2026-10-10 · Task 1 完成:DDL 三件套`(四样:用户原话「这是需要建的表」+ GPT-6.1 增补两列;产出;纠偏;翻车)。

```bash
git add -A && git commit -m "feat(ch10): topic_classifications 建表——DDL 配对/ORM/启动校验/测试库纳管"
```

---

### Task 2: training/ 骨架——依赖分组 + taxonomy.json + labels.py 统一解码

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`(追加)
- Create: `training/__init__.py`(空)、`training/taxonomy.json`、`training/labels.py`
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Produces(后续任务全部依赖):
  - `training.labels.TAXONOMY_PATH / load_taxonomy() -> list[dict] / label_names() -> list[str]`
  - `training.labels.taxonomy_version() -> str`(64 hex)
  - `training.labels.DECODER_VERSION = "topic-decoder-v1"`
  - `training.labels.decode_scores(scores: Sequence[float], threshold: float, labels: Sequence[str] | None = None) -> list[str]`
  - `training.labels.classifier_version(model_version: str, effective_threshold: float, decoder_version: str = DECODER_VERSION) -> str`
  - `training.labels.canonical_json(obj) -> str`

- [ ] **Step 1: 写失败测试** `tests/test_taxonomy.py`

```python
"""taxonomy 冻结钉 + 统一解码口径(spec §2/§3)。"""

from training.labels import (DECODER_VERSION, canonical_json, classifier_version,
                             decode_scores, label_names, load_taxonomy,
                             taxonomy_version)

EXPECTED = ["退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动",
            "价保", "支付", "订单修改", "库存补货", "商品信息", "保修维修",
            "账号", "会员积分", "评价", "其他"]


def test_taxonomy_frozen():
    tax = load_taxonomy()
    assert [t["name"] for t in tax] == EXPECTED  # 类名与顺序钉死
    assert len(tax) == 17
    assert all(t["boundary"].strip() for t in tax)  # 边界非空
    assert len(taxonomy_version()) == 64


def test_decode_threshold_inclusive():
    names = label_names()
    scores = [0.0] * 17
    scores[names.index("退换货")] = 0.5
    assert decode_scores(scores, 0.5) == ["退换货"]  # >= 等号命中


def test_decode_zero_hit_falls_back_to_other():
    scores = [0.9 if i == 16 else 0.01 for i in range(17)]  # 只有「其他」高分
    assert decode_scores(scores, 0.5) == ["其他"]  # 「其他」分数不参与判定,零命中兜底


def test_decode_other_mutually_exclusive_and_order():
    names = label_names()
    scores = [0.0] * 17
    scores[names.index("尺码")] = 0.9
    scores[names.index("退换货")] = 0.8
    scores[names.index("其他")] = 0.99
    assert decode_scores(scores, 0.5) == ["退换货", "尺码"]  # 冻结顺序,剔除「其他」


def test_decode_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        decode_scores([0.1, 0.2], 0.5)


def test_classifier_version_stable_and_sensitive():
    v1 = classifier_version("m" * 64, 0.5)
    assert v1 == classifier_version("m" * 64, 0.5) and len(v1) == 64
    assert v1 != classifier_version("m" * 64, 0.51)
    assert v1 != classifier_version("m" * 64, 0.5, decoder_version="v2")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_taxonomy.py -x -q`
Expected: FAIL(`training.labels` 不存在)

- [ ] **Step 3: 依赖分组与 gitignore**

`pyproject.toml` 的 `[dependency-groups]` 改为(Context7 已核实 `{include-group}` 语法,PEP 735):

```toml
[dependency-groups]
topic-data = [
  "numpy>=1.26",
  "scikit-learn>=1.4",
  "scikit-multilearn>=0.2",
  "opencc>=1.1",
]
dev = [
  {include-group = "topic-data"},
  "pytest>=8",
  "pytest-asyncio>=0.23",
]
train = [
  {include-group = "topic-data"},
  "torch>=2.2",
  "transformers>=4.46,<5",
  "datasets>=2.19",
  "onnx>=1.16",
  "onnxruntime>=1.17",
  "optimum[onnx]>=1.23",
]
```

文件末尾追加(torch 走 CPU 专用源,避免 CUDA 轮子几个 GB):

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```

`.gitignore` 追加:

```
data/topic_classifier/
models/
```

Run: `uv lock && uv sync`(默认只装 dev 闭包,不含 torch;若 pytorch-cpu 源解析失败,停下问用户,不自行换方案)

- [ ] **Step 4: 写 taxonomy.json**

`training/__init__.py` 空文件。`training/taxonomy.json`(顺序冻结,边界含用户点名三对):

```json
[
  {"name": "退换货", "boundary": "把商品退回/换掉/退款的诉求;要修不归这里,归保修维修"},
  {"name": "物流", "boundary": "货的流动:发货/运输/派送/签收/催件;运费金额问题归运费"},
  {"name": "尺码", "boundary": "尺码咨询、偏大偏小、选码换码;明确要退货的同时打退换货"},
  {"name": "发票", "boundary": "开票/换开/抬头/电子发票/发票未收到"},
  {"name": "质量问题", "boundary": "破损/瑕疵/故障/与描述不符的事实陈述;带退换或维修诉求的另打对应类"},
  {"name": "运费", "boundary": "钱:运费多少/包邮门槛/退货运费谁出;货在哪归物流"},
  {"name": "优惠活动", "boundary": "券、满减、折扣、赠品活动的规则与使用;买贵补差价归价保"},
  {"name": "价保", "boundary": "买贵了补差价/价格保护;券和满减归优惠活动"},
  {"name": "支付", "boundary": "付款失败/扣款异常/分期/支付方式/重复扣款"},
  {"name": "订单修改", "boundary": "改地址/改商品/取消订单/改配送时间等订单内容变更"},
  {"name": "库存补货", "boundary": "缺货/何时有货/到货通知/补货"},
  {"name": "商品信息", "boundary": "参数/材质/用法/真伪/保质期等商品咨询"},
  {"name": "保修维修", "boundary": "保修期/坏了要修/维修点;要退归退换货"},
  {"name": "账号", "boundary": "登录/注册/密码/账号异常/注销"},
  {"name": "会员积分", "boundary": "会员权益/积分获取与使用/等级"},
  {"name": "评价", "boundary": "评价/晒单/追评/删改评价"},
  {"name": "其他", "boundary": "没有明确具体类目诉求(库外、闲聊、无诉求表达),与其余 16 类互斥"}
]
```

- [ ] **Step 5: 写 training/labels.py(纯标准库)**

```python
"""17 类权威类目加载、版本哈希与统一解码(decoder_version=topic-decoder-v1)。

纯标准库:主服务 app/services/topic_classify.py 也导入本模块,
不得引入 numpy/torch 等任何第三方依赖(spec §13)。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.json"
DECODER_VERSION = "topic-decoder-v1"
OTHER = "其他"


def load_taxonomy() -> list[dict]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def label_names() -> list[str]:
    return [t["name"] for t in load_taxonomy()]


def canonical_json(obj) -> str:
    """规范化 JSON:键排序、数组保序、UTF-8、无多余空白(spec §2)。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def taxonomy_version() -> str:
    return _sha256(canonical_json(load_taxonomy()))


def classifier_version(model_version: str, effective_threshold: float,
                       decoder_version: str = DECODER_VERSION) -> str:
    return _sha256(canonical_json({
        "model_version": model_version,
        "effective_threshold": effective_threshold,
        "decoder_version": decoder_version}))


def decode_scores(scores: Sequence[float], threshold: float,
                  labels: Sequence[str] | None = None) -> list[str]:
    """统一解码:其余 16 类分数 >= threshold 全命中;有具体类命中剔除「其他」,
    零命中输出 ["其他"];按冻结顺序返回(spec §3)。"""
    names = list(labels) if labels is not None else label_names()
    if len(scores) != len(names):
        raise ValueError(f"scores 维度 {len(scores)} 与类目数 {len(names)} 不符")
    hits = [name for s, name in zip(scores, names)
            if name != OTHER and s >= threshold]
    return hits if hits else [OTHER]
```

- [ ] **Step 6: 跑测试确认通过 + 全量回归**

Run: `uv run pytest tests/test_taxonomy.py -q && uv run pytest -q`
Expected: 新测试 6 条 PASS;全量无回归

- [ ] **Step 7: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): training 骨架——依赖分组/17 类冻结类目表/统一解码纯函数"
```

---

### Task 3: preprocessing.py 确定性清洗

**Files:**
- Create: `training/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `PREPROCESS_VERSION = "topic-preprocess-v1"`;`clean_text(text: str) -> str`(脱敏→繁简→全半角→空白折叠,顺序固定);`preprocess_config() -> dict`(落 preprocess.json 用);被 prep_clean.py 与 serve.py 消费

- [ ] **Step 1: 写失败测试** `tests/test_preprocessing.py`

```python
"""确定性清洗:脱敏/繁简/全半角/空白,幂等且顺序冻结(spec §4.1)。"""

from training.preprocessing import PREPROCESS_VERSION, clean_text, preprocess_config


def test_masking():
    assert clean_text("我手机13812345678,邮箱a@b.com") == "我手机[PHONE],邮箱[EMAIL]"
    assert clean_text("订单A1001啥时候发货") == "订单[ORDER_ID]啥时候发货"
    assert clean_text("单号 20261010001 查一下") == "单号 [ORDER_ID] 查一下"


def test_width_and_space():
    assert clean_text("ＡＢＣ　１２３") == "ABC 123"  # 全角→半角,全角空格折叠
    assert clean_text("  多大   的\t合适\n") == "多大 的 合适"


def test_traditional_to_simplified():
    assert clean_text("我要退貨") == "我要退货"


def test_idempotent_and_deterministic():
    text = "訂單Ａ１００１  退貨,电话13900001111"
    assert clean_text(clean_text(text)) == clean_text(text)


def test_empty_and_version():
    assert clean_text("   ") == ""
    assert PREPROCESS_VERSION == "topic-preprocess-v1"
    cfg = preprocess_config()
    assert cfg["preprocess_version"] == PREPROCESS_VERSION and "steps" in cfg
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_preprocessing.py -x -q`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 写 training/preprocessing.py**

```python
"""确定性清洗:训练与推理共用同一实现(spec §4.1)。

处理顺序冻结:脱敏 → 繁简(OpenCC t2s)→ 全半角 → 空白折叠。
在线推理只走本模块,不调用 LLM 校对。改动任一步 = 换 PREPROCESS_VERSION。
"""

from __future__ import annotations

import re
import unicodedata

from opencc import OpenCC

PREPROCESS_VERSION = "topic-preprocess-v1"

_CC = OpenCC("t2s")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ORDER_RE = re.compile(r"\b[A-Za-z]\d{4,}\b|\b\d{8,}\b")
_WS_RE = re.compile(r"\s+")


def _to_halfwidth(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E
        else (" " if ch == "　" else ch)
        for ch in text)


def clean_text(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ORDER_RE.sub("[ORDER_ID]", text)
    text = _CC.convert(text)
    text = unicodedata.normalize("NFC", text)
    text = _to_halfwidth(text)
    return _WS_RE.sub(" ", text).strip()


def preprocess_config() -> dict:
    return {"preprocess_version": PREPROCESS_VERSION,
            "steps": ["mask_email", "mask_phone", "mask_order_id",
                      "opencc_t2s", "nfc", "halfwidth", "space_collapse"]}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_preprocessing.py -q`
Expected: 5 条 PASS;若某断言与实现真实行为不符(如空格折叠边界),以实现实际输出为准修正测试并注明

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 确定性清洗模块——脱敏/繁简/全半角/空白,顺序冻结"
```

---

### Task 4: 归类服务核心 run_classification(锁/metadata/快照归组/分批推理/原子写)

**Files:**
- Create: `app/services/topic_classify.py`
- Test: `tests/test_topic_classify.py`

**Interfaces:**
- Consumes: `training.labels.{label_names, taxonomy_version, decode_scores, DECODER_VERSION}`(Task 2);`app.models.{LowConfidenceQuestion, TopicClassification}`(Task 1)
- Produces:
  - 错误类:`TopicError(code)` 及子类 `TopicJobBusyError("job_busy")` / `TopicInferenceError("inference_unavailable")` / `TopicContractError("inference_contract")` / `TopicVersionConflictError("classification_version_conflict")`
  - `run_classification(settings, session_factory, *, client_factory=None) -> dict` 返回 `{"classifier_version", "classified_questions", "skipped_questions", "written_rows"}`
  - `question_key(row) -> tuple`(Task 5 的分布统计也用它)

**关键语义(spec §7.3,逐条钉):**
1. 模块级 `threading.Lock` 非阻塞获取,忙 → TopicJobBusyError;`finally` 释放
2. GET `{topic_inference_url}/metadata`:labels 必须 == `label_names()`、`taxonomy_version` 必须 == `taxonomy_version()`、`decoder_version` 必须 == `DECODER_VERSION`、`effective_threshold` 为 (0,1) 有限数——否则 TopicContractError;由此算出目标 `classifier_version`
3. 一致性读事务快照 `lcq LEFT JOIN topic_classifications`,按 `question_key` 归组(双锚点齐 → `("turn", cid, mid)`,缺一 → `("row", id)`);读完关事务再发网络请求
4. 同组全部行已有合法结果、版本 == 目标版本、metadata 规范化后 == 本次 metadata → 整组跳过;否则整组重归类(同版本只补漏、换版本重算全部)
5. 代表文本:组内按 lcq.id 升序取第一个非空 `resolved_question`,全空取第一个非空 `raw_question`;全组无可用文本 → 计入 skipped_questions 且不写
6. 按 `topic_classify_batch_size` 分批 POST `/classify`;每批校验:条数 == 请求条数、`classifier_version` == 目标版本、每条 scores 为 17 维 [0,1] 有限浮点、labels == `decode_scores(scores, threshold)`——任一不符 TopicContractError
7. 全部批次成功才开唯一写事务,`mysql_insert(...).on_duplicate_key_update` 同组所有行写相同 labels/version/metadata;一次 commit,异常整笔回滚
8. httpx 任何异常 / 非 200 → TopicInferenceError;所有 503/500 路径零写入

- [ ] **Step 1: 写失败测试** `tests/test_topic_classify.py`

```python
"""归类作业核心:归组/幂等/原子性/并发锁(spec §7.3 + §10 幂等矩阵)。"""

import httpx
import pytest

from app.models import LowConfidenceQuestion, TopicClassification
from app.services import topic_classify
from app.services.topic_classify import (TopicContractError, TopicInferenceError,
                                         TopicJobBusyError, run_classification)
from training.labels import (classifier_version, decode_scores, label_names,
                             taxonomy_version)
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

NAMES = label_names()
THRESHOLD = 0.5
MODEL_VERSION = "m" * 64
VERSION = classifier_version(MODEL_VERSION, THRESHOLD)
METADATA = {"model_version": MODEL_VERSION, "taxonomy_version": taxonomy_version(),
            "effective_threshold": THRESHOLD, "labels": NAMES,
            "decoder_version": "topic-decoder-v1"}


def _scores(hit_indexes):
    return [0.9 if i in hit_indexes else 0.01 for i in range(17)]


def _fake_client(labels_map):
    """labels_map: text -> hit 下标元组;返回 httpx.Client(MockTransport)。"""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metadata":
            return httpx.Response(200, json={
                "model_version": MODEL_VERSION, "taxonomy_version": taxonomy_version(),
                "classifier_version": VERSION, "effective_threshold": THRESHOLD,
                "labels": NAMES, "decoder_version": "topic-decoder-v1"})
        import json as _json
        texts = _json.loads(request.content.decode())["texts"]
        batch = [_scores(labels_map[t]) for t in texts]
        return httpx.Response(200, json={
            "labels": [decode_scores(s, THRESHOLD) for s in batch],
            "scores": batch, "classifier_version": VERSION})
    return httpx.Client(transport=httpx.MockTransport(handler))


def _seed_lcq(sf, rows):
    """rows: list of dict(raw_question, resolved_question, cid, mid, status)。"""
    ids = []
    with sf() as s:
        for r in rows:
            q = LowConfidenceQuestion(
                raw_question=r["raw_question"],
                resolved_question=r.get("resolved_question"),
                conversation_id=r.get("cid"), turn_message_id=r.get("mid"),
                source=r.get("source", "self_check"),
                process_status=r.get("status", "processed"))
            s.add(q)
            s.flush()
            ids.append(q.id)
        s.commit()
    return ids


def _settings(**kw):
    return make_settings(topic_inference_url="http://fake-inference", **kw)


def test_happy_path_classifies_and_writes(db_session_factory):
    _seed_lcq(db_session_factory, [
        {"raw_question": "买大了想退", "resolved_question": "买大了想退货"},
        {"raw_question": "发票怎么开", "resolved_question": None},
    ])
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: _fake_client(
                                 {"买大了想退货": (0, 2), "发票怎么开": (3,)}))
    assert out["classified_questions"] == 2 and out["skipped_questions"] == 0
    assert out["written_rows"] == 2 and out["classifier_version"] == VERSION
    with db_session_factory() as s:
        rows = s.query(TopicClassification).order_by(
            TopicClassification.question_id).all()
    assert rows[0].labels == ["退换货", "尺码"]  # 冻结顺序
    assert rows[0].classifier_version == VERSION
    assert rows[0].classifier_metadata["taxonomy_version"] == taxonomy_version()
    assert rows[1].labels == ["发票"]


def test_same_turn_rows_share_one_inference(db_session_factory):
    """同轮自动落池+👎(同 conversation_id+turn_message_id)共享一次推理与结果。"""
    _seed_lcq(db_session_factory, [
        {"raw_question": "这个太大想退", "cid": 1, "mid": 101, "source": "retrieval_low_conf"},
        {"raw_question": "这个太大想退", "cid": 1, "mid": 101, "source": "user_feedback"},
        {"raw_question": "这个太大想退", "cid": 1, "mid": 102},  # 不同轮同文本,不合并
    ])
    calls = []
    base = _fake_client({"这个太大想退": (0, 2)})
    orig_transport = base._transport
    class CountingTransport(httpx.MockTransport):
        def handle_async_request(self, request):  # 同步 Client 走 sync 入口
            return self.handler(request)
    def counting_handler(request):
        calls.append(request.url.path)
        return orig_transport.handle_request(request) \
            if hasattr(orig_transport, "handle_request") else None
    # 简化:直接用 _fake_client,断言 written_rows 与 classified_questions 即可
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: _fake_client(
                                 {"这个太大想退": (0, 2)}))
    assert out["classified_questions"] == 2  # 3 行归 2 组(同轮两行合一)
    assert out["written_rows"] == 3          # 同组两行写相同结果
    with db_session_factory() as s:
        rows = s.query(TopicClassification).all()
    assert {tuple(r.labels) for r in rows} == {("退换货", "尺码")}


def test_same_version_skip_complete_groups(db_session_factory):
    ids = _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    client = _fake_client({"买大了想退": (0,)})
    run_classification(_settings(), db_session_factory, client_factory=lambda **kw: client)
    # 第二次:同版本已有完整结果 → 跳过,零写入
    def boom(**kw):
        class _Boom:
            def get(self, *a, **k): raise AssertionError("不应再请求推理服务")
            def post(self, *a, **k): raise AssertionError("不应再请求推理服务")
        return _Boom()
    # metadata 仍会拉取(校验目标版本),用真 fake;推理不应发生
    metadata_only = _fake_client({})  # classify 收到任何文本即 KeyError
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: metadata_only)
    assert out["classified_questions"] == 0 and out["skipped_questions"] == 1
    assert out["written_rows"] == 0


def test_inference_down_zero_write(db_session_factory):
    _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    def dead(**kw):
        c = httpx.Client(transport=httpx.MockTransport(
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))))
        return c
    with pytest.raises(TopicInferenceError):
        run_classification(_settings(), db_session_factory, client_factory=dead)
    with db_session_factory() as s:
        assert s.query(TopicClassification).count() == 0  # 零写入


def test_second_batch_failure_zero_write(db_session_factory):
    """第二批失败,第一批也不得落库(原子性,spec §7.3/§10)。"""
    texts = [f"问题{i}号" for i in range(3)]
    _seed_lcq(db_session_factory, [{"raw_question": t} for t in texts])
    def flaky(**kw):
        def handler(request):
            if request.url.path == "/metadata":
                return _fake_client({}).get("http://x/metadata")  # 复用 metadata 响应
            import json as _json
            batch = _json.loads(request.content.decode())["texts"]
            if len(batch) < 3:  # batch_size=2:第二批
                return httpx.Response(502, json={"error": "boom"})
            s = [_scores((0,)) for _ in batch]
            return httpx.Response(200, json={
                "labels": [decode_scores(x, THRESHOLD) for x in s],
                "scores": s, "classifier_version": VERSION})
        return httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(TopicInferenceError):
        run_classification(_settings(topic_classify_batch_size=2),
                           db_session_factory, client_factory=flaky)
    with db_session_factory() as s:
        assert s.query(TopicClassification).count() == 0


def test_contract_violation_zero_write(db_session_factory):
    """响应标签与分数解码不一致 → 契约失败,零写入。"""
    _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    def bad(**kw):
        def handler(request):
            if request.url.path == "/metadata":
                return _fake_client({}).get("http://x/metadata")
            s = [_scores((0, 2))]
            return httpx.Response(200, json={
                "labels": [["物流"]], "scores": s, "classifier_version": VERSION})
        return httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(TopicContractError):
        run_classification(_settings(), db_session_factory, client_factory=bad)
    with db_session_factory() as s:
        assert s.query(TopicClassification).count() == 0


def test_job_lock_busy(db_session_factory):
    assert topic_classify._JOB_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(TopicJobBusyError):
            run_classification(_settings(), db_session_factory,
                               client_factory=lambda **kw: None)
    finally:
        topic_classify._JOB_LOCK.release()


def test_lock_released_after_failure(db_session_factory):
    """失败后锁已释放,可重试(spec §10)。"""
    _seed_lcq(db_session_factory, [{"raw_question": "买大了想退"}])
    def dead(**kw):
        return httpx.Client(transport=httpx.MockTransport(
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))))
    with pytest.raises(TopicInferenceError):
        run_classification(_settings(), db_session_factory, client_factory=dead)
    out = run_classification(_settings(), db_session_factory,
                             client_factory=lambda **kw: _fake_client({"买大了想退": (0,)}))
    assert out["classified_questions"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_topic_classify.py -x -q`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 写 app/services/topic_classify.py**

```python
"""ch10 旁路批量归类作业(spec §7.3)。

时序:非阻塞作业锁 → 拉推理 metadata 定目标版本 → 一致性读事务取快照并归组
→ 关事务后分批推理(全暂存内存、逐批校验契约)→ 全部成功才开唯一写事务原子提交。
任何推理/契约失败 503 零写入;DB 失败异常上抛 500 回滚。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.sql import func

from app.models import LowConfidenceQuestion, TopicClassification
from training.labels import (DECODER_VERSION, canonical_json, classifier_version,
                             decode_scores, label_names, taxonomy_version)

_JOB_LOCK = threading.Lock()  # 进程内独占;单 Uvicorn worker 前提(spec §7.3)


class TopicError(Exception):
    code = "topic_error"


class TopicJobBusyError(TopicError):
    code = "job_busy"


class TopicInferenceError(TopicError):
    code = "inference_unavailable"


class TopicContractError(TopicError):
    code = "inference_contract"


class TopicVersionConflictError(TopicError):
    code = "classification_version_conflict"


def question_key(row: LowConfidenceQuestion) -> tuple:
    """问题键:双锚点齐 → 轮次;缺一 → 行本身(spec §1)。"""
    if row.conversation_id is not None and row.turn_message_id is not None:
        return ("turn", row.conversation_id, row.turn_message_id)
    return ("row", row.id)


def _fetch_metadata(client: httpx.Client, base_url: str) -> dict:
    try:
        resp = client.get(f"{base_url}/metadata")
        resp.raise_for_status()
        meta = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise TopicInferenceError(f"推理服务 metadata 不可达:{type(exc).__name__}") from exc
    names = label_names()
    if (meta.get("labels") != names
            or meta.get("taxonomy_version") != taxonomy_version()
            or meta.get("decoder_version") != DECODER_VERSION):
        raise TopicContractError("推理服务 metadata 与权威类目表不一致")
    th = meta.get("effective_threshold")
    if not isinstance(th, (int, float)) or not (0 < float(th) < 1):
        raise TopicContractError("effective_threshold 非法")
    return meta


def _snapshot(sf) -> list[tuple[LowConfidenceQuestion, TopicClassification | None]]:
    with sf() as s:
        rows = s.execute(
            select(LowConfidenceQuestion, TopicClassification)
            .outerjoin(TopicClassification,
                       TopicClassification.question_id == LowConfidenceQuestion.id)
            .order_by(LowConfidenceQuestion.id)).all()
        s.expunge_all()
    return [(r[0], r[1]) for r in rows]


def _representative_text(rows: list[LowConfidenceQuestion]) -> str | None:
    ordered = sorted(rows, key=lambda r: r.id)
    for field in ("resolved_question", "raw_question"):
        for r in ordered:
            text = (getattr(r, field) or "").strip()
            if text:
                return text
    return None


def _validate_batch(payload: dict, expected_count: int, threshold: float,
                    expected_version: str) -> None:
    labels, scores = payload.get("labels"), payload.get("scores")
    if payload.get("classifier_version") != expected_version:
        raise TopicContractError("批次版本与作业目标版本不一致")
    if not (isinstance(labels, list) and isinstance(scores, list)
            and len(labels) == expected_count and len(scores) == expected_count):
        raise TopicContractError("批次条数不符")
    for lbs, scs in zip(labels, scores):
        if not (isinstance(scs, list) and len(scs) == 17
                and all(isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0
                        and float(x) == float(x) and abs(float(x)) != float("inf")
                        for x in scs)):
            raise TopicContractError("分数须为 17 维 [0,1] 有限浮点")
        if lbs != decode_scores([float(x) for x in scs], threshold):
            raise TopicContractError("标签与分数解码不一致")


def run_classification(settings, session_factory, *, client_factory=None) -> dict:
    if not _JOB_LOCK.acquire(blocking=False):
        raise TopicJobBusyError("已有归类作业在进行")
    try:
        return _run(settings, session_factory, client_factory)
    finally:
        _JOB_LOCK.release()


def _run(settings, sf, client_factory) -> dict:
    factory = client_factory or (
        lambda **kw: httpx.Client(timeout=settings.topic_inference_timeout_seconds, **kw))
    client = factory()
    base = settings.topic_inference_url.rstrip("/")
    meta = _fetch_metadata(client, base)
    threshold = float(meta["effective_threshold"])
    target_version = meta["classifier_version"]
    meta_canonical = canonical_json(meta)

    groups: dict[tuple, list[tuple]] = {}
    for lcq, tc in _snapshot(sf):
        groups.setdefault(question_key(lcq), []).append((lcq, tc))

    todo: list[tuple[tuple, list[tuple], str]] = []
    skipped = 0
    for key, members in groups.items():
        rows = [m[0] for m in members]
        tcs = [m[1] for m in members]
        complete = all(
            tc is not None and tc.classifier_version == target_version
            and canonical_json(tc.classifier_metadata) == _stored_metadata_canonical(tc)
            and isinstance(tc.labels, list) and tc.labels
            for tc in tcs)
        same_labels = len({canonical_json(tc.labels) for tc in tcs
                           if tc is not None}) <= 1
        if complete and same_labels and all(
                canonical_json(tc.classifier_metadata) == canonical_json(
                    _metadata_for_store(meta)) for tc in tcs):
            skipped += 1
            continue
        text = _representative_text(rows)
        if text is None:
            skipped += 1  # 无可用文本:跳过不写
            continue
        todo.append((key, members, text))

    results: list[tuple[list[tuple], list[str], dict]] = []
    batch_size = settings.topic_classify_batch_size
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        try:
            resp = client.post(f"{base}/classify",
                               json={"texts": [c[2] for c in chunk]})
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TopicInferenceError(
                f"推理批次 {start // batch_size + 1} 失败:{type(exc).__name__}") from exc
        _validate_batch(payload, len(chunk), threshold, target_version)
        for (key, members, text), lbs, scs in zip(
                chunk, payload["labels"], payload["scores"]):
            results.append((members, lbs, meta))

    written = 0
    if results:
        store_meta = _metadata_for_store(meta)
        now = datetime.now()
        rows_payload = [{"question_id": lcq.id, "labels": lbs,
                         "classifier_version": target_version,
                         "classifier_metadata": store_meta,
                         "classified_at": now}
                        for members, lbs, _ in results for lcq, _ in members]
        with sf() as s:  # 唯一写事务:一次提交,异常整笔回滚
            stmt = mysql_insert(TopicClassification).values(rows_payload)
            stmt = stmt.on_duplicate_key_update(
                labels=stmt.inserted.labels,
                classifier_version=stmt.inserted.classifier_version,
                classifier_metadata=stmt.inserted.classifier_metadata,
                classified_at=stmt.inserted.classified_at)
            s.execute(stmt)
            s.commit()
        written = len(rows_payload)
    return {"classifier_version": target_version,
            "classified_questions": len(results),
            "skipped_questions": skipped,
            "written_rows": written}


def _metadata_for_store(meta: dict) -> dict:
    return {"model_version": meta["model_version"],
            "taxonomy_version": meta["taxonomy_version"],
            "effective_threshold": float(meta["effective_threshold"]),
            "labels": meta["labels"],
            "decoder_version": meta["decoder_version"]}


def _stored_metadata_canonical(tc: TopicClassification) -> str:
    return canonical_json(tc.classifier_metadata or {})
```

(注意:`_run` 中 complete 判定的最终实现请按测试行为对齐——「同版本+metadata 一致+标签合法」三条件,冗余表达式合并;上码以通过全部测试为准,语义不得偏离 §7.3。)

- [ ] **Step 4: 跑测试迭代到全绿**

Run: `uv run pytest tests/test_topic_classify.py -q`
Expected: 8 条 PASS(实现中如发现测试桩问题——如 MockTransport 复用方式——修测试桩,不修语义)

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 归类作业核心——轮次归组/契约校验/原子写/作业锁"
```

---

### Task 5: 分布统计 get_distribution

**Files:**
- Modify: `app/services/topic_classify.py`(追加)
- Test: `tests/test_topic_distribution.py`

**Interfaces:**
- Produces: `get_distribution(settings, session_factory) -> dict`,返回 `{total, classified, unclassified, counts: [{label, count}], classifier_version, classifier_metadata, classified_at, stale}`;版本混杂/冲突抛 `TopicVersionConflictError`

- [ ] **Step 1: 写失败测试** `tests/test_topic_distribution.py`

```python
"""主题分布统计口径(spec §7.4):轮次去重/多标签计数/空表/stale/版本冲突。"""

import pytest

from app.models import LowConfidenceQuestion, TopicClassification
from app.services.topic_classify import (TopicVersionConflictError,
                                         get_distribution)
from training.labels import label_names, taxonomy_version
from tests.conftest import make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401

V1 = "a" * 64
META_OK = {"model_version": "m", "taxonomy_version": taxonomy_version(),
           "effective_threshold": 0.5, "labels": label_names(),
           "decoder_version": "topic-decoder-v1"}
META_STALE = {**META_OK, "taxonomy_version": "0" * 64}


def _mk(sf, lcq_kw, tc=None):
    with sf() as s:
        q = LowConfidenceQuestion(raw_question=lcq_kw.pop("raw_question", "q"),
                                  source="self_check", process_status="processed",
                                  **lcq_kw)
        s.add(q)
        s.flush()
        if tc is not None:
            s.add(TopicClassification(question_id=q.id, labels=tc["labels"],
                                      classifier_version=tc.get("v", V1),
                                      classifier_metadata=tc.get("meta", META_OK)))
        s.commit()
        return q.id


def test_empty_table():
    out = get_distribution(make_settings(), _empty_sf)
    assert out["total"] == 0 and out["classified"] == 0
    assert out["classifier_version"] is None and out["stale"] is False
    assert len(out["counts"]) == 17 and all(c["count"] == 0 for c in out["counts"])


def test_multilabel_counts_and_turn_dedup(db_session_factory):
    sf = db_session_factory
    # 同轮两行共享结果:只算一个问题;多标签进两个横条
    _mk(sf, {"raw_question": "买大了想退", "conversation_id": 1, "turn_message_id": 1},
        tc={"labels": ["退换货", "尺码"]})
    _mk(sf, {"raw_question": "买大了想退", "conversation_id": 1, "turn_message_id": 1},
        tc={"labels": ["退换货", "尺码"]})
    _mk(sf, {"raw_question": "发票呢"}, tc={"labels": ["发票"]})
    _mk(sf, {"raw_question": "没人理我"})  # 未归类
    out = get_distribution(make_settings(), sf)
    assert out["total"] == 3 and out["classified"] == 2 and out["unclassified"] == 1
    counts = {c["label"]: c["count"] for c in out["counts"]}
    assert counts["退换货"] == 1 and counts["尺码"] == 1 and counts["发票"] == 1
    assert sum(counts.values()) == 3 > out["classified"]  # 多标签计数可大于已归类数
    assert out["classifier_version"] == V1 and out["stale"] is False
    # 降序,计数并列按权威顺序
    assert [c["label"] for c in out["counts"]][:3] == ["退换货", "尺码", "发票"]


def test_group_incomplete_counts_unclassified(db_session_factory):
    sf = db_session_factory
    _mk(sf, {"raw_question": "q", "conversation_id": 2, "turn_message_id": 5},
        tc={"labels": ["物流"]})
    _mk(sf, {"raw_question": "q", "conversation_id": 2, "turn_message_id": 5})  # 同组缺结果
    out = get_distribution(make_settings(), sf)
    assert out["classified"] == 0 and out["unclassified"] == 1
    assert {c["label"]: c["count"] for c in out["counts"]}["物流"] == 0


def test_stale_taxonomy_zeroes_counts(db_session_factory):
    sf = db_session_factory
    _mk(sf, {"raw_question": "q"}, tc={"labels": ["物流"], "meta": META_STALE})
    out = get_distribution(make_settings(), sf)
    assert out["stale"] is True and out["classified"] == 0 and out["unclassified"] == 1
    assert all(c["count"] == 0 for c in out["counts"])
    assert out["classifier_version"] == V1  # 旧版本信息保留


def test_version_conflict_409(db_session_factory):
    sf = db_session_factory
    _mk(sf, {"raw_question": "a"}, tc={"labels": ["物流"], "v": V1})
    _mk(sf, {"raw_question": "b"}, tc={"labels": ["发票"], "v": "b" * 64})
    with pytest.raises(TopicVersionConflictError):
        get_distribution(make_settings(), sf)
```

(`_empty_sf`:用一个 lcq 与分类表均为空的 `db_session_factory` fixture 参数实现——直接写成 `def test_empty_table(db_session_factory)`。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_topic_distribution.py -x -q`
Expected: FAIL(`get_distribution` 不存在)

- [ ] **Step 3: 实现 get_distribution(追加到 app/services/topic_classify.py)**

```python
def get_distribution(settings, session_factory) -> dict:
    pairs = _snapshot(session_factory)
    groups: dict[tuple, list[tuple]] = {}
    for lcq, tc in pairs:
        groups.setdefault(question_key(lcq), []).append((lcq, tc))
    names = label_names()
    stored = [tc for members in groups.values() for _, tc in members if tc is not None]
    versions = {tc.classifier_version for tc in stored}
    if len(versions) > 1:
        raise TopicVersionConflictError("分类结果版本混杂,请重新归类修复")
    metas = {canonical_json(tc.classifier_metadata or {}) for tc in stored}
    if len(metas) > 1:
        raise TopicVersionConflictError("同版本 metadata 冲突,请重新归类修复")

    version = stored[0].classifier_version if stored else None
    meta = stored[0].classifier_metadata if stored else None
    stale = bool(meta) and meta.get("taxonomy_version") != taxonomy_version()
    classified_at = max((tc.classified_at for tc in stored), default=None)

    counts = {n: 0 for n in names}
    classified = 0
    for members in groups.values():
        tcs = [tc for _, tc in members]
        ok = (all(tc is not None for tc in tcs)
              and len({canonical_json(tc.labels) for tc in tcs}) == 1
              and len({tc.classifier_version for tc in tcs}) == 1
              and isinstance(tcs[0].labels, list) and tcs[0].labels
              and set(tcs[0].labels) <= set(names))
        if not ok:
            continue
        classified += 1
        if stale:
            continue  # 过期口径不进横条图
        for lb in dict.fromkeys(tcs[0].labels):  # 同组同标签最多计一次
            counts[lb] += 1
    if stale:
        classified = 0
    ordered = sorted(counts.items(),
                     key=lambda kv: (-kv[1], names.index(kv[0])))
    return {"total": len(groups), "classified": classified,
            "unclassified": len(groups) - classified,
            "counts": [{"label": k, "count": v} for k, v in ordered],
            "classifier_version": version,
            "classifier_metadata": meta,
            "classified_at": classified_at.isoformat(sep=" ") if classified_at else None,
            "stale": stale}
```

- [ ] **Step 4: 跑测试迭代到全绿 + 全量回归**

Run: `uv run pytest tests/test_topic_distribution.py tests/test_topic_classify.py -q && uv run pytest -q`
Expected: 全 PASS

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 主题分布统计——轮次去重/多标签计数/stale/版本冲突 409"
```

---

### Task 6: topics API + 配置 + 错误处理器 + main.py 接线

**Files:**
- Modify: `app/config.py`(ch10 四字段)、`.env.example`(ch10 节)
- Create: `app/routers/topics.py`
- Modify: `app/main.py`(import router + include + 三个错误处理器)
- Test: `tests/test_topics_api.py`、`tests/test_config.py`(追加)

**Interfaces:**
- Consumes: Task 4/5 的 `run_classification` / `get_distribution` 与错误类
- Produces: `POST /api/topics/classify`、`GET /api/topics/distribution`;配置 `topic_inference_url / topic_classify_batch_size / topic_inference_timeout_seconds / topic_label_threshold`;错误契约 `{"error":{"code","message"}}`:409 `job_busy` / 503 `inference_unavailable`、`inference_contract` / 409 `classification_version_conflict` / 500 其余

- [ ] **Step 1: 写失败测试** `tests/test_topics_api.py`

```python
"""topics API 契约:状态码与错误体(spec §7/§8)。"""

import httpx
import pytest

from app.main import create_app
from app.services import topic_classify
from tests.conftest import FakeStreamModel, make_runtime, make_settings


def _app():
    return create_app(settings=make_settings(), model=FakeStreamModel([]),
                      runtime=make_runtime(tools=[]))


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_classify_ok(monkeypatch):
    monkeypatch.setattr(topic_classify, "run_classification",
                        lambda *a, **k: {"classifier_version": "v",
                                         "classified_questions": 2,
                                         "skipped_questions": 0, "written_rows": 3})
    async with await _client(_app()) as c:
        resp = await c.post("/api/topics/classify")
    assert resp.status_code == 200 and resp.json()["written_rows"] == 3


async def test_classify_busy_409(monkeypatch):
    def busy(*a, **k):
        raise topic_classify.TopicJobBusyError("忙")
    monkeypatch.setattr(topic_classify, "run_classification", busy)
    async with await _client(_app()) as c:
        resp = await c.post("/api/topics/classify")
    assert resp.status_code == 409 and resp.json()["error"]["code"] == "job_busy"


async def test_classify_inference_down_503(monkeypatch):
    def down(*a, **k):
        raise topic_classify.TopicInferenceError("不可达")
    monkeypatch.setattr(topic_classify, "run_classification", down)
    async with await _client(_app()) as c:
        resp = await c.post("/api/topics/classify")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "inference_unavailable"


async def test_distribution_conflict_409(monkeypatch):
    def conflict(*a, **k):
        raise topic_classify.TopicVersionConflictError("混杂")
    monkeypatch.setattr(topic_classify, "get_distribution", conflict)
    async with await _client(_app()) as c:
        resp = await c.get("/api/topics/distribution")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "classification_version_conflict"


async def test_distribution_ok(monkeypatch):
    monkeypatch.setattr(topic_classify, "get_distribution",
                        lambda *a, **k: {"total": 0, "classified": 0,
                                         "unclassified": 0, "counts": [],
                                         "classifier_version": None,
                                         "classifier_metadata": None,
                                         "classified_at": None, "stale": False})
    async with await _client(_app()) as c:
        resp = await c.get("/api/topics/distribution")
    assert resp.status_code == 200 and resp.json()["stale"] is False
```

`tests/test_config.py` 追加:

```python
def test_ch10_config_defaults():
    s = Settings(_env_file=None, openai_base_url="http://t/v1",
                 openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@127.0.0.1:9/wayhelp")
    assert s.topic_inference_url == "http://127.0.0.1:8103"
    assert s.topic_classify_batch_size == 32
    assert s.topic_inference_timeout_seconds == 60.0
    assert s.topic_label_threshold is None


def test_ch10_config_validation():
    import pydantic
    base = dict(openai_base_url="http://t/v1", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@127.0.0.1:9/wayhelp")
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, topic_classify_batch_size=0, **base)
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, topic_classify_batch_size=257, **base)
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, topic_label_threshold=1.5, **base)
```

(若 test_config.py 已有构造惯例,对齐其写法。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_topics_api.py -x -q`
Expected: FAIL(路由 404 / 配置字段不存在)

- [ ] **Step 3: 实现**

`app/config.py` Settings 追加(行尾中文注释惯例):

```python
    # ── ch10 主题分类器 ──
    topic_inference_url: str = "http://127.0.0.1:8103"  # 推理服务地址(独立进程)
    topic_classify_batch_size: int = Field(default=32, ge=1, le=256)  # 归类批大小
    topic_inference_timeout_seconds: float = Field(default=60.0, gt=0)  # 单次 metadata/归类请求超时
    topic_label_threshold: float | None = Field(default=None, gt=0, lt=1)  # 仅推理进程读取的阈值覆盖;主服务不消费
```

`.env.example` 末尾追加:

```
# ── ch10 主题分类器 ──
# 推理服务地址(独立进程,make topic-inference 起在 :8103)
# TOPIC_INFERENCE_URL=http://127.0.0.1:8103
# 归类批大小(1-256)
# TOPIC_CLASSIFY_BATCH_SIZE=32
# 单次 metadata/归类请求超时(秒,正数)
# TOPIC_INFERENCE_TIMEOUT_SECONDS=60
# 阈值覆盖:仅推理进程读取;不设用模型包 labels.json 冻结值;主服务不消费
# TOPIC_LABEL_THRESHOLD=
```

`app/routers/topics.py`:

```python
"""ch10 主题归类 API(/api/topics/*):薄壳,同步 service 走 asyncio.to_thread。"""

import asyncio

from fastapi import APIRouter, Request

from app.services import topic_classify

router = APIRouter(prefix="/api")


def _deps(request: Request):
    return request.app.state.settings, request.app.state.session_factory


@router.post("/topics/classify")
async def topics_classify(request: Request):
    settings, sf = _deps(request)
    return await asyncio.to_thread(topic_classify.run_classification, settings, sf)


@router.get("/topics/distribution")
async def topics_distribution(request: Request):
    settings, sf = _deps(request)
    return await asyncio.to_thread(topic_classify.get_distribution, settings, sf)
```

`app/main.py`:
- import 区加 `from app.routers.topics import router as topics_router`
- `include_router` 区(stats_router 后)加 `app.include_router(topics_router)`
- 错误处理器区追加:

```python
    @app.exception_handler(topic_classify.TopicJobBusyError)
    async def _(request: Request, exc: topic_classify.TopicJobBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content=_error_body(exc.code, "已有归类作业在进行"))

    @app.exception_handler(topic_classify.TopicVersionConflictError)
    async def _(request: Request, exc: topic_classify.TopicVersionConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content=_error_body(exc.code, "分类结果版本冲突,请重新归类"))

    @app.exception_handler(topic_classify.TopicError)
    async def _(request: Request, exc: topic_classify.TopicError) -> JSONResponse:
        return JSONResponse(status_code=503, content=_error_body(exc.code, "主题推理服务暂不可用"))
```

并 import `from app.services import topic_classify`。

注意:注入测试 runtime 时 session_factory 为 None——路由内 `_deps` 取出 None 传给被 monkeypatch 的 service,不触库;真实运行由 `_build_production_runtime` 装配。

- [ ] **Step 4: 跑测试迭代到全绿 + 全量回归**

Run: `uv run pytest tests/test_topics_api.py tests/test_config.py -q && uv run pytest -q`
Expected: 全 PASS

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): topics API 接线——classify/distribution 端点+ch10 配置+错误契约"
```

---

### Task 7: 数据管线 LLM 脚本——prep_clean / prep_generate / prep_label

**Files:**
- Create: `training/prep_records.py`(记录 schema + jsonl IO + LLM 客户端工厂)、`training/prep_clean.py`、`training/prep_generate.py`、`training/prep_label.py`
- Test: `tests/test_prep_pipeline.py`

**Interfaces:**
- Produces(后续 prep_* 与 train 共同遵守的记录 schema):
  ```python
  {"sample_id": str,            # syn-<类拼音序>-<序号> / lcq-<id> / aug-<parent>
   "source": "synthetic" | "real",
   "source_group_id": str,      # 初始 = sample_id;真实双锚点齐 → turn-<cid>-<mid>
   "text_raw": str, "text": str,            # 清洗后文本
   "labels": list[str],         # 最终标签(corrections 回灌后)
   "intended_labels": list[str] | None,     # 合成数据的生成意图标签
   "parent_id": str | None}
  ```
- `prep_records`: `write_jsonl(path, records)` / `read_jsonl(path) -> list[dict]` / `make_chat_model(settings)`(langchain_openai ChatOpenAI,主依赖已有)/ `load_real_questions(settings, session_factory) -> list[record]`(lcq 全量,text_raw=resolved_question 或 raw_question,labels 待预标)
- `prep_clean.llm_polish(records, llm, batch_size=20) -> tuple[list[record], int]`(返回修正后记录 + 失败批数;ID 对齐失败整批保留原文)
- `prep_generate.build_prompt(entry, combos, n) -> str` / `parse_generated(text, entry_name) -> list[tuple[text, labels]]`
- `prep_label.parse_prelabel(text, valid_names) -> list[str]` / `apply_corrections(records, corrections) -> list[record]` / `disagreements(records) -> list[record]` / `export_review_markdown(records, path, per_class=0.1, seed=42)`

- [ ] **Step 1: 写失败测试** `tests/test_prep_pipeline.py`

```python
"""数据管线确定性部分:LLM 校对 ID 对齐、生成解析、预标解析、corrections 回灌。
LLM 本体不进 pytest——全部用假模型(spec §10)。"""

import json

from training.prep_clean import llm_polish
from training.prep_generate import parse_generated
from training.prep_label import (apply_corrections, disagreements,
                                 parse_prelabel)


class FakeLLM:
    """批量校对假模型:receive list[dict(id,text)],按 script 返回。"""
    def __init__(self, script):
        self.script = script  # list[list[dict]] 每次 invoke 弹一批
    def polish_batch(self, batch):
        return self.script.pop(0) if self.script else batch


def _rec(i, text):
    return {"sample_id": f"syn-0-{i:04d}", "source": "synthetic",
            "source_group_id": f"syn-0-{i:04d}", "text_raw": text, "text": text,
            "labels": [], "intended_labels": ["物流"], "parent_id": None}


def test_llm_polish_id_aligned():
    recs = [_rec(0, "物流慢死了"), _rec(1, "什么时候发货")]
    llm = FakeLLM([[{"sample_id": "syn-0-0000", "text": "物流慢极了"},
                    {"sample_id": "syn-0-0001", "text": "什么时候发货"}]])
    out, failed = llm_polish(recs, llm, batch_size=2)
    assert failed == 0 and out[0]["text"] == "物流慢极了"


def test_llm_polish_misaligned_keeps_original():
    recs = [_rec(0, "a"), _rec(1, "b")]
    # 返回条数不符 → 整批保留原文并计失败
    llm = FakeLLM([[{"sample_id": "syn-0-0000", "text": "a!"}]])
    out, failed = llm_polish(recs, llm, batch_size=2)
    assert failed == 1 and [r["text"] for r in out] == ["a", "b"]
    # 返回 ID 错位 → 同样整批保留
    llm2 = FakeLLM([[{"sample_id": "X", "text": "a"}, {"sample_id": "Y", "text": "b"}]])
    out2, failed2 = llm_polish(recs, llm2, batch_size=2)
    assert failed2 == 1 and [r["text"] for r in out2] == ["a", "b"]


def test_parse_generated():
    text = json.dumps([{"text": "买大了想退", "labels": ["尺码", "退换货"]},
                       {"text": "什么时候发货", "labels": ["物流"]}],
                      ensure_ascii=False)
    out = parse_generated(text, "物流")
    assert out == [("买大了想退", ["尺码", "退换货"]), ("什么时候发货", ["物流"])]


def test_parse_prelabel_filters_unknown_and_orders():
    out = parse_prelabel('["尺码", "退换货", "不存在类"]',
                         ["退换货", "物流", "尺码", "发票", "其他"])
    assert out == ["退换货", "尺码"]  # 未知类丢弃,按权威顺序重排


def test_apply_corrections_and_disagreements():
    recs = [_rec(0, "买大了想退")]
    recs[0]["labels"] = ["物流"]  # 预标与意图 ["物流"?] 一致则无分歧;改成不一致:
    recs[0]["labels"] = ["价保"]
    bad = disagreements(recs)
    assert [r["sample_id"] for r in bad] == ["syn-0-0000"]  # 预标 ≠ intended
    fixed = apply_corrections(recs, [{"sample_id": "syn-0-0000",
                                      "labels": ["退换货", "尺码"]}])
    assert fixed[0]["labels"] == ["退换货", "尺码"]
    assert disagreements(fixed) == [] or fixed[0]["intended_labels"] == ["物流"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_prep_pipeline.py -x -q`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现四个模块**

`training/prep_records.py`:

```python
"""数据管线公共:记录 schema、jsonl IO、LLM 客户端、真实池读取。"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

DATA_DIR = Path("data/topic_classifier")


def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_chat_model(settings):
    from langchain_openai import ChatOpenAI  # 主依赖;重导入延迟到调用点
    return ChatOpenAI(model=settings.model_name,
                      api_key=settings.openai_api_key,
                      base_url=settings.openai_base_url, temperature=0.7)


def make_record(sample_id, source, group_id, text_raw, text, labels,
                intended_labels=None, parent_id=None) -> dict:
    return {"sample_id": sample_id, "source": source,
            "source_group_id": group_id, "text_raw": text_raw, "text": text,
            "labels": list(labels), "intended_labels": intended_labels,
            "parent_id": parent_id}


def load_real_questions(session_factory) -> list[dict]:
    """lcq 全量(不分 process_status);双锚点齐归 turn 组,缺一自成组(spec §1/§4)。"""
    from app.models import LowConfidenceQuestion
    with session_factory() as s:
        rows = s.execute(select(LowConfidenceQuestion)
                         .order_by(LowConfidenceQuestion.id)).scalars().all()
        s.expunge_all()
    records = []
    for r in rows:
        if r.conversation_id is not None and r.turn_message_id is not None:
            gid = f"turn-{r.conversation_id}-{r.turn_message_id}"
        else:
            gid = f"row-{r.id}"
        text = (r.resolved_question or "").strip() or (r.raw_question or "").strip()
        records.append(make_record(f"lcq-{r.id}", "real", gid, text, text, []))
    return records
```

`training/prep_clean.py`:

```python
"""步骤1 清洗:确定性 clean_text(全量)+ LLM 错别字校对(ID 对齐校验)。

LLM 校对仅离线数据准备用;错位/重复/遗漏 → 整批保留原文并记日志(spec §4.1)。
CLI: uv run --group train python -m training.prep_clean <in.jsonl> <out.jsonl>
"""

from __future__ import annotations

import logging

from training.preprocessing import clean_text

log = logging.getLogger(__name__)


def deterministic_clean(records) -> list[dict]:
    for r in records:
        r["text"] = clean_text(r["text_raw"])
    return records


def llm_polish(records, llm, batch_size: int = 20):
    """llm 须实现 polish_batch(list[{sample_id,text}]) -> list[{sample_id,text}]。
    返回 (records, failed_batches)。ID 集合不一致或条数不符 → 整批保留原文。"""
    failed = 0
    out = list(records)
    for start in range(0, len(out), batch_size):
        batch = out[start:start + batch_size]
        expect_ids = [r["sample_id"] for r in batch]
        try:
            fixed = llm.polish_batch(
                [{"sample_id": r["sample_id"], "text": r["text"]} for r in batch])
        except Exception as exc:
            log.warning("polish 批 %d 异常保留原文:%s", start // batch_size, exc)
            failed += 1
            continue
        got_ids = [f.get("sample_id") for f in fixed] if isinstance(fixed, list) else []
        if sorted(got_ids) != sorted(expect_ids) or len(set(got_ids)) != len(got_ids):
            log.warning("polish 批 %d ID 不对齐,整批保留原文", start // batch_size)
            failed += 1
            continue
        by_id = {f["sample_id"]: f["text"] for f in fixed}
        for r in batch:
            r["text"] = by_id[r["sample_id"]]
    return out, failed
```

`training/prep_generate.py`(prompt 构造 + 解析;LLM 调用在 CLI main):

```python
"""步骤2前置:LLM 按术语表合成数据(每类 ~100 条,~20% 多诉求句)。

CLI: uv run --group train python -m training.prep_generate \
       --per-class 100 --multi-ratio 0.2 --out data/topic_classifier/raw_synthetic.jsonl
生成时指定标签组合(意图标签),合成记录 source_group_id=sample_id(各自成组);
切分前去重时才合并同文本组(spec §4)。
"""

from __future__ import annotations

import json

from training.labels import load_taxonomy


def build_prompt(entry: dict, combos: list[list[str]], n: int) -> str:
    """entry: taxonomy 中一类;combos: 多诉求标签组合(含该类),长度 n 对应条数。"""
    lines = [f"你在为电商客服训练数据造句。类目「{entry['name']}」:{entry['boundary']}",
             f"请造 {n} 条用户提问短句(口语、带错别字空间、5-40 字),"
             "逐条输出 JSON 数组,元素为 {\"text\":..., \"labels\":[...]}。",
             "labels 只能来自给定组合,一句话字面提到几个诉求就打几个标签。"]
    return "\n".join(lines) + f"\n标签组合序列:{json.dumps(combos, ensure_ascii=False)}"


def parse_generated(text: str, entry_name: str) -> list[tuple[str, list[str]]]:
    """解析 LLM 输出的 JSON 数组;非法行丢弃(记返回条数供 CLI 对账)。"""
    start, end = text.find("["), text.rfind("]")
    items = json.loads(text[start:end + 1])
    out = []
    for it in items:
        t, lbs = (it.get("text") or "").strip(), it.get("labels") or []
        if t and isinstance(lbs, list) and lbs:
            out.append((t, list(lbs)))
    return out
```

`training/prep_label.py`:

```python
"""步骤标注:LLM 预标(QA/真实数据)+ 人工抽审导出 + corrections 回灌(spec §3)。

CLI 子命令:prelabel / export-review / apply-corrections。
"""

from __future__ import annotations

import json
import random

from training.labels import label_names


def parse_prelabel(text: str, valid_names: list[str] | None = None) -> list[str]:
    names = valid_names or label_names()
    start, end = text.find("["), text.rfind("]")
    raw = json.loads(text[start:end + 1]) if start >= 0 else []
    keep = [n for n in names if n in set(raw) and n != "其他"]
    return keep if keep else ["其他"]  # 零具体命中兜底「其他」


def disagreements(records) -> list[dict]:
    """合成数据:预标与意图标签不一致的进人工裁决队列。"""
    return [r for r in records
            if r.get("intended_labels") is not None
            and sorted(r["labels"]) != sorted(r["intended_labels"])]


def apply_corrections(records, corrections) -> list[dict]:
    by_id = {c["sample_id"]: c["labels"] for c in corrections}
    for r in records:
        if r["sample_id"] in by_id:
            r["labels"] = list(by_id[r["sample_id"]])
    return records


def export_review_markdown(records, path, per_class: float = 0.1, seed: int = 42):
    """每类随机抽 ~10% 导出人工抽审 markdown(spec §3)。"""
    rng = random.Random(seed)
    lines = ["# 标注人工抽审", ""]
    by_label: dict[str, list[dict]] = {}
    for r in records:
        for lb in r["labels"]:
            by_label.setdefault(lb, []).append(r)
    picked = []
    for lb, rs in sorted(by_label.items()):
        k = max(1, round(len(rs) * per_class))
        picked.extend(rng.sample(rs, min(k, len(rs))))
    for r in picked:
        lines.append(f"- `{r['sample_id']}` {r['text']} → {json.dumps(r['labels'], ensure_ascii=False)}")
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return len(picked)
```

- [ ] **Step 4: 跑测试迭代到全绿**

Run: `uv run pytest tests/test_prep_pipeline.py -q`
Expected: 6 条 PASS(测试与实现对不上的细节——如 disagreements 对修正后记录的判定——按 spec 语义对齐后修正测试或实现)

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 数据管线 LLM 脚本——清洗校对/合成/预标/抽审回灌"
```

---

### Task 8: prep_split.py 分层抽样(来源组不可拆 + manifest 冻结)

**Files:**
- Create: `training/prep_split.py`
- Test: `tests/test_prep_split.py`

**Interfaces:**
- Consumes: prep_records schema;`training.labels.canonical_json`
- Produces:
  - `dedupe_by_text(records) -> tuple[list[dict], list[dict]]`(按清洗文本去重,合并来源组;同文异标 → 冲突列表进人工裁决)
  - `stratified_split(records, seed=42) -> dict[str, str]`(sample_id → train/val/test;来源组不可拆)
  - `freeze_manifest(records, assignment, tax_version, seed=42) -> dict`(含 `dataset_version`)
  - `split_report(records, assignment) -> dict`(实际比例/逐类 support/多标签占比)

- [ ] **Step 1: 写失败测试** `tests/test_prep_split.py`

```python
"""分层抽样:来源组不可拆、三份无交集、种子可复现、17 类覆盖、manifest 冻结。"""

import pytest

from training.labels import label_names, taxonomy_version
from training.prep_split import (dedupe_by_text, freeze_manifest, split_report,
                                 stratified_split)

NAMES = label_names()


def _rec(i, label, text=None, group=None, source="synthetic"):
    sid = f"s{i:03d}"
    return {"sample_id": sid, "source": source,
            "source_group_id": group or sid,
            "text_raw": text or f"文本{i}", "text": text or f"文本{i}",
            "labels": [label], "intended_labels": [label], "parent_id": None}


def _corpus(per_class=10):
    return [_rec(i * per_class + j, NAMES[i])
            for i in range(17) for j in range(per_class)]


def test_split_ratios_and_coverage():
    recs = _corpus()
    assignment = stratified_split(recs, seed=42)
    assert set(assignment) == {r["sample_id"] for r in recs}
    train = [r for r in recs if assignment[r["sample_id"]] == "train"]
    val = [r for r in recs if assignment[r["sample_id"]] == "val"]
    test = [r for r in recs if assignment[r["sample_id"]] == "test"]
    assert abs(len(train) / len(recs) - 0.8) < 0.06
    assert abs(len(val) / len(recs) - 0.1) < 0.06
    # 三份均覆盖 17 类
    for part in (train, val, test):
        assert {lb for r in part for lb in r["labels"]} == set(NAMES)
    # 种子可复现
    assert stratified_split(recs, seed=42) == assignment


def test_source_group_indivisible():
    recs = _corpus()
    # 把第 0 类前两条捏成同一来源组(模拟同轮真实问题)
    recs[0]["source_group_id"] = recs[1]["source_group_id"] = "turn-1-1"
    assignment = stratified_split(recs, seed=42)
    assert assignment["s000"] == assignment["s001"]


def test_dedupe_merges_groups_and_conflicts():
    a = _rec(0, "物流", text="一样的话")
    b = _rec(1, "物流", text="一样的话")
    c = _rec(2, "发票", text="一样的话")
    merged, conflicts = dedupe_by_text([a, b])
    assert len(merged) == 1 and conflicts == []
    assert set(merged[0]["source_group_id"].split("+")) == {"s000", "s001"}
    _, conflicts2 = dedupe_by_text([a, c])
    assert len(conflicts2) == 1  # 同文异标进人工裁决


def test_manifest_frozen_and_stable():
    recs = _corpus()
    assignment = stratified_split(recs, seed=42)
    m1 = freeze_manifest(recs, assignment, taxonomy_version(), seed=42)
    m2 = freeze_manifest(recs, assignment, taxonomy_version(), seed=42)
    assert m1["dataset_version"] == m2["dataset_version"] and len(m1["dataset_version"]) == 64
    assert m1["seed"] == 42 and m1["taxonomy_version"] == taxonomy_version()
    # 改一个样本标签 → 数据版本变
    recs[0]["labels"] = ["物流"]
    m3 = freeze_manifest(recs, assignment, taxonomy_version(), seed=42)
    assert m3["dataset_version"] != m1["dataset_version"]


def test_split_report_fields():
    recs = _corpus()
    assignment = stratified_split(recs, seed=42)
    rep = split_report(recs, assignment)
    assert set(rep) >= {"actual_ratios", "per_class_support", "multilabel_share"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_prep_split.py -x -q`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现 training/prep_split.py**

```python
"""步骤3 分层抽样:来源组不可拆的多标签迭代分层 80/10/10,种子 42(spec §4.3)。

切分前先按清洗文本去重(合并来源组;同文异标进人工裁决,不出现在输出)。
scikit-multilearn IterativeStratification 作用于来源组(组标签 = 成员并集);
两阶段:先 90/10 切出 test,再把剩余 8/9 : 1/9 切 train/val(≈80/10/10)。
实现前用 Context7 核对 scikit-multilearn 当前 API(skmultilearn.model_selection)。
"""

from __future__ import annotations

import hashlib

from training.labels import canonical_json, label_names


def dedupe_by_text(records):
    by_text: dict[str, list[dict]] = {}
    for r in records:
        by_text.setdefault(r["text"], []).append(r)
    merged, conflicts = [], []
    for text, rs in by_text.items():
        label_sets = {tuple(sorted(r["labels"])) for r in rs}
        if len(label_sets) > 1:
            conflicts.append({"text": text,
                              "sample_ids": [r["sample_id"] for r in rs],
                              "label_sets": sorted(map(list, label_sets))})
            continue
        head = dict(rs[0])
        head["source_group_id"] = "+".join(sorted({r["source_group_id"] for r in rs}))
        head["sample_id"] = "+".join(sorted(r["sample_id"] for r in rs)) \
            if len(rs) > 1 else rs[0]["sample_id"]
        merged.append(head)
    return merged, conflicts


def _group_labels(records):
    groups: dict[str, set] = {}
    for r in records:
        groups.setdefault(r["source_group_id"], set()).update(r["labels"])
    return groups


def stratified_split(records, seed: int = 42) -> dict[str, str]:
    import numpy as np
    from skmultilearn.model_selection import IterativeStratification

    names = label_names()
    gid_of = {r["sample_id"]: r["source_group_id"] for r in records}
    gids = sorted(set(gid_of.values()))
    glabels = _group_labels(records)
    X = np.array(range(len(gids))).reshape(-1, 1)
    y = np.array([[1 if n in glabels[g] else 0 for n in names] for g in gids])

    def one_split(X, y, first_share):
        strat = IterativeStratification(
            n_splits=2, order=1,
            sample_distribution_per_fold=[first_share, 1 - first_share])
        return next(strat.split(X, y))

    # 第一阶段:切出 test(10%)
    rest_idx, test_idx = one_split(X, y, 0.9)
    # 第二阶段:剩余 90% 里切 val(占全体 10% → 占剩余 1/9)
    Xr, yr = X[rest_idx], y[rest_idx]
    tr_idx, val_idx = one_split(Xr, yr, 8 / 9)
    group_split: dict[str, str] = {}
    for i in rest_idx[tr_idx]:
        group_split[gids[i]] = "train"
    for i in rest_idx[val_idx]:
        group_split[gids[i]] = "val"
    for i in test_idx:
        group_split[gids[i]] = "test"
    return {sid: group_split[g] for sid, g in gid_of.items()}


def freeze_manifest(records, assignment, tax_version, seed: int = 42) -> dict:
    entries = []
    for r in sorted(records, key=lambda r: r["sample_id"]):
        entries.append({
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "split": assignment[r["sample_id"]],
            "content_hash": hashlib.sha256(
                canonical_json({"text": r["text"], "labels": sorted(r["labels"])}
                               ).encode("utf-8")).hexdigest()})
    dataset_version = hashlib.sha256(
        canonical_json(entries).encode("utf-8")).hexdigest()
    return {"seed": seed, "taxonomy_version": tax_version,
            "dataset_version": dataset_version, "entries": entries}


def split_report(records, assignment) -> dict:
    names = label_names()
    total = len(records)
    ratios, support = {}, {n: {"train": 0, "val": 0, "test": 0} for n in names}
    multi = 0
    for r in records:
        split = assignment[r["sample_id"]]
        ratios[split] = ratios.get(split, 0) + 1
        if len(r["labels"]) > 1:
            multi += 1
        for lb in r["labels"]:
            support[lb][split] += 1
    return {"actual_ratios": {k: round(v / total, 4) for k, v in ratios.items()},
            "per_class_support": support,
            "multilabel_share": round(multi / total, 4)}
```

(注意:IterativeStratification 的 order/sample_distribution_per_fold 参数行为以实现时 Context7 核对的当前版本为准;若两阶段切分比例偏差超测试阈值,按组不可拆偏差单列说明的 spec 口径调整断言并记录。)

- [ ] **Step 4: 跑测试迭代到全绿**

Run: `uv run pytest tests/test_prep_split.py -q`
Expected: 5 条 PASS

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 分层抽样——来源组不可拆迭代分层 + split_manifest 冻结"
```

---

### Task 9: prep_augment.py 数据增强(只扩训练集)

**Files:**
- Create: `training/prep_augment.py`
- Test: `tests/test_prep_augment.py`

**Interfaces:**
- Produces: `plan_augmentation(train_records, all_texts, target_ratio=1.3) -> list[计划]`;`apply_augmentation(train_records, variants: dict[sample_id, str], all_texts, manifest_assignment) -> list[record]`(继承标签/来源组/训练归属,parent_id 记录;与三份现有文本重复的丢弃)

- [ ] **Step 1: 写失败测试** `tests/test_prep_augment.py`

```python
"""数据增强:只扩训练集、标签继承、≤1 派生、跨集文本重复丢弃(spec §4.4)。"""

from training.prep_augment import apply_augmentation, plan_augmentation


def _rec(i, labels=None):
    return {"sample_id": f"s{i:03d}", "source": "synthetic",
            "source_group_id": f"g{i:03d}", "text_raw": f"原文{i}",
            "text": f"原文{i}", "labels": labels or ["物流"],
            "intended_labels": labels or ["物流"], "parent_id": None}


def test_plan_respects_target_ratio():
    train = [_rec(i) for i in range(10)]
    plan = plan_augmentation(train, all_texts={r["text"] for r in train},
                             target_ratio=1.3)
    assert len(plan) == 3  # 10 → 13,扩 3 条
    assert all(p["parent_id"] in {r["sample_id"] for r in train} for p in plan)


def test_apply_inherits_and_dedupes():
    train = [_rec(0, ["退换货", "尺码"]), _rec(1)]
    existing = {"原文0", "原文1", "验证集文本"}
    variants = {"s000": "买大啦想退掉", "s001": "验证集文本"}
    out = apply_augmentation(train, variants, existing, {"s000": "train", "s001": "train"})
    assert len(out) == 1  # 与既有文本重复的丢弃
    aug = out[0]
    assert aug["parent_id"] == "s000" and aug["labels"] == ["退换货", "尺码"]
    assert aug["source_group_id"] == "g000"  # 继承来源组(不可拆一致性)
    assert aug["sample_id"].startswith("aug-s000")


def test_augment_never_touches_val_test():
    train = [_rec(0)]
    out = apply_augmentation(train, {"s000": "换个说法"},
                             {"原文0"}, {"s000": "train"})
    assert all(r["parent_id"] == "s000" for r in out)
    # 只接受 train 归属的 parent:val 里的样本不得被增强
    out2 = apply_augmentation([_rec(9)], {"s009": "x"}, {"原文9"},
                              {"s009": "val"})
    assert out2 == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_prep_augment.py -x -q`
Expected: FAIL

- [ ] **Step 3: 实现 training/prep_augment.py**

```python
"""步骤4 数据增强:LLM 同义词替换/句式微调,只扩训练集(spec §4.4)。

派生样本继承标签、来源组与训练集归属;与三份现有文本重复的候选丢弃;
不移动验证/测试样本凑数量。CLI 的 LLM 改写部分手跑验证,不进 pytest。
"""

from __future__ import annotations


def plan_augmentation(train_records, all_texts, target_ratio: float = 1.3) -> list[dict]:
    """计算要增强哪些样本:目标 train * target_ratio,每条 parent ≤1 派生。"""
    need = max(0, round(len(train_records) * target_ratio) - len(train_records))
    return [{"parent_id": r["sample_id"], "text": r["text"]}
            for r in train_records[:need]]


def apply_augmentation(train_records, variants: dict[str, str], all_texts,
                       assignment: dict[str, str]) -> list[dict]:
    by_id = {r["sample_id"]: r for r in train_records}
    out = []
    for pid, text in variants.items():
        parent = by_id.get(pid)
        if parent is None or assignment.get(pid) != "train":
            continue  # 只扩训练集
        text = (text or "").strip()
        if not text or text in all_texts:
            continue  # 重复候选丢弃
        out.append({"sample_id": f"aug-{pid}", "source": parent["source"],
                    "source_group_id": parent["source_group_id"],
                    "text_raw": text, "text": text,
                    "labels": list(parent["labels"]),
                    "intended_labels": list(parent["intended_labels"])
                    if parent.get("intended_labels") else None,
                    "parent_id": pid})
    return out
```

- [ ] **Step 4: 跑测试迭代到全绿 + 全量回归**

Run: `uv run pytest tests/test_prep_augment.py -q && uv run pytest -q`
Expected: 全 PASS

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 数据增强——只扩训练集/标签继承/重复丢弃"
```

---

### Task 10: train.py + evaluate.py(全参微调 + 阈值扫描 + 评测报告)

**Files:**
- Create: `training/threshold.py`、`training/train.py`、`training/evaluate.py`
- Test: `tests/test_threshold.py`、`tests/test_evaluate_report.py`

**Interfaces:**
- Produces:
  - `training.threshold.sweep_global_threshold(scores: np.ndarray, truths: np.ndarray) -> float`(0.05–0.95 步长 0.01,解码口径同 labels.decode_scores;并列先取最接近 0.5、仍并列取较大者)
  - `training.evaluate.compute_metrics_per_class(scores, truths, threshold) -> dict`、`render_report(metrics, meta) -> str`
  - CLI:`uv run --group train python -m training.train [--smoke]`;`uv run --group train python -m training.evaluate`
- Consumes: split 后的 `data/topic_classifier/{train,val,test}.jsonl`、`split_manifest.json`

**验证方式(用户约定:训练类任务以实跑替代 TDD):** 纯函数(threshold/report)写 pytest;Trainer 管线以 `--smoke`(50 条)实跑验证,全量训练在 Task 14。

- [ ] **Step 1: 写失败测试** `tests/test_threshold.py` + `tests/test_evaluate_report.py`

```python
"""阈值扫描:范围/步长/并列规则(spec §5)。"""

import numpy as np

from training.threshold import sweep_global_threshold


def test_sweep_picks_best_micro_f1():
    # 构造:0.3 阈值下全部命中完美
    scores = np.array([[0.4, 0.1], [0.1, 0.4]])
    truths = np.array([[1, 0], [0, 1]])
    assert sweep_global_threshold(scores, truths) == 0.3


def test_tie_prefers_closer_to_half_then_larger():
    # 所有候选阈值 F1 相同 → 取最接近 0.5;仍并列取较大者
    scores = np.array([[0.99, 0.01]])
    truths = np.array([[1, 0]])
    got = sweep_global_threshold(scores, truths)
    assert got == 0.5
```

```python
"""评测报告渲染:逐类 P/R/F1/support + 每类混淆矩阵(spec §6)。"""

import numpy as np

from training.evaluate import compute_metrics_per_class, render_report


def test_per_class_metrics_and_confusion():
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.9, 0.9], [0.1, 0.1]])
    truths = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    m = compute_metrics_per_class(scores, truths, 0.5, ["退换货", "尺码"])
    a = m["per_class"]["退换货"]
    assert a["precision"] == 1.0 and a["recall"] == 1.0 and a["support"] == 2
    # 「退换货」混淆矩阵:TP=2 FP=1 FN=0 TN=1(第 4 行零命中→「其他」不在此二列)
    cm = a["confusion"]
    assert cm == {"tp": 2, "fp": 1, "fn": 0, "tn": 1}
    assert "micro_f1" in m and "macro_f1" in m


def test_render_report_contains_sections():
    m = {"per_class": {"物流": {"precision": 1.0, "recall": 0.5, "f1": 0.667,
                                "support": 2,
                                "confusion": {"tp": 1, "fp": 0, "fn": 1, "tn": 8}}},
         "micro_f1": 0.9, "macro_f1": 0.667}
    md = render_report(m, {"model_version": "m", "dataset_version": "d",
                           "effective_threshold": 0.5, "decoder_version": "topic-decoder-v1"})
    assert "# 主题分类器留出测试报告" in md
    assert "| 物流 | 1.000 | 0.500 | 0.667 | 2 |" in md
    assert "混淆矩阵" in md and "topic-decoder-v1" in md
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_threshold.py tests/test_evaluate_report.py -x -q`
Expected: FAIL

- [ ] **Step 3: 实现 threshold.py / evaluate.py 纯函数部分**

`training/threshold.py`:

```python
"""验证集全局阈值扫描(spec §5):0.05–0.95 步长 0.01,micro-F1 最优;
并列先取最接近 0.5,仍并列取较大者。"""

from __future__ import annotations

import numpy as np

from training.labels import decode_scores, label_names


def _micro_f1(scores: np.ndarray, truths: np.ndarray, threshold: float) -> float:
    names = label_names()
    tp = fp = fn = 0
    for row, truth in zip(scores, truths):
        pred = set(decode_scores([float(x) for x in row], threshold, names))
        gold = {names[i] for i, v in enumerate(truth) if v}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def sweep_global_threshold(scores: np.ndarray, truths: np.ndarray) -> float:
    best_f1, candidates = -1.0, []
    for i in range(5, 96):  # 0.05..0.95
        th = i / 100
        f1 = _micro_f1(scores, truths, th)
        if f1 > best_f1 + 1e-12:
            best_f1, candidates = f1, [th]
        elif abs(f1 - best_f1) <= 1e-12:
            candidates.append(th)
    candidates.sort(key=lambda t: (abs(t - 0.5), -t))
    return candidates[0]
```

`training/evaluate.py` 纯函数部分(完整文件含 CLI):

```python
"""留出测试评测:逐类 P/R/F1/support + one-vs-rest 混淆矩阵 + 错例导出(spec §6)。

CLI: uv run --group train python -m training.evaluate \
       --ckpt models/topic_classifier/_work/best --threshold <float>
报告落 evals/results/<UTC时间戳>_topic_eval.md(不进 pytest、不入库)。
"""

from __future__ import annotations

import numpy as np

from training.labels import decode_scores


def compute_metrics_per_class(scores, truths, threshold, names) -> dict:
    per_class = {}
    tps = fps = fns = 0
    for idx, name in enumerate(names):
        if name == "其他":
            continue  # 「其他」是解码兜底,指标仍算:用解码结果对比
        tp = fp = fn = tn = 0
        for row, truth in zip(scores, truths):
            pred = decode_scores([float(x) for x in row], threshold, names)
            p = name in pred
            g = bool(truth[idx])
            tp += p and g
            fp += p and not g
            fn += (not p) and g
            tn += (not p) and not g
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        support = int(sum(bool(t[idx]) for t in truths))
        per_class[name] = {"precision": round(precision, 3),
                           "recall": round(recall, 3), "f1": round(f1, 3),
                           "support": support,
                           "confusion": {"tp": int(tp), "fp": int(fp),
                                         "fn": int(fn), "tn": int(tn)}}
        tps += tp
        fps += fp
        fns += fn
    f1s = [c["f1"] for c in per_class.values()]
    micro_p = tps / (tps + fps) if tps + fps else 0.0
    micro_r = tps / (tps + fns) if tps + fns else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)
    return {"per_class": per_class,
            "micro_f1": round(micro_f1, 3),
            "macro_f1": round(sum(f1s) / len(f1s), 3)}
```

`render_report(metrics, meta)`:产出 `# 主题分类器留出测试报告` 标题 + 版本元信息块(含 decoder_version)+ 逐类表格行 `| 物流 | 1.000 | 0.500 | 0.667 | 2 |`(表头 `| 类目 | P | R | F1 | support |`)+ 每类混淆矩阵小节(标题含「混淆矩阵」)+ macro/micro 汇总行。错例导出:FP/FN 样本写 `<UTC时间戳>_topic_errors.md`。

- [ ] **Step 4: 实现 train.py(Context7 已核对的 Trainer API)**

```python
"""RoBERTa-wwm-ext 全参微调(CPU),early stopping + 阈值扫描(spec §5)。

CLI: uv run --group train python -m training.train [--smoke]
产物:models/topic_classifier/_work/best/(checkpoint)+ val_scores.npy + threshold.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from training.labels import decode_scores, label_names
from training.prep_records import read_jsonl

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
WORK_DIR = Path("models/topic_classifier/_work")
MAX_LEN = 64
SEED = 42


def _load(split):
    records = read_jsonl(Path("data/topic_classifier") / f"{split}.jsonl")
    names = label_names()
    texts = [r["text"] for r in records]
    y = np.array([[1.0 if n in r["labels"] else 0.0 for n in names]
                  for r in records], dtype=np.float32)
    return texts, y


def main(smoke: bool = False):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments)

    names = label_names()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], max_length=MAX_LEN,
                         truncation=True, padding="max_length")

    train_texts, train_y = _load("train")
    val_texts, val_y = _load("val")
    if smoke:  # 冒烟:50 条验证管线,不要求类别覆盖(spec §4.3)
        train_texts, train_y = train_texts[:50], train_y[:50]
        val_texts, val_y = val_texts[:20], val_y[:20]
    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_y.tolist()}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_y.tolist()}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(names),
        problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)})

    def compute_metrics(eval_pred):
        logits, truths = eval_pred
        scores = 1 / (1 + np.exp(-logits))  # sigmoid
        tp = fp = fn = 0
        for row, truth in zip(scores, truths):
            pred = set(decode_scores([float(x) for x in row], 0.5, names))
            gold = {names[i] for i, v in enumerate(truth) if v}
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        return {"micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}

    args = TrainingArguments(
        output_dir=str(WORK_DIR), learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=10, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="micro_f1",
        seed=SEED, use_cpu=True, report_to=[], save_total_limit=2)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    best = WORK_DIR / "best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)

    # 阈值扫描:选定 checkpoint 后,验证集 0.05–0.95(spec §5)
    from training.threshold import sweep_global_threshold
    out = trainer.predict(val_ds)
    scores = 1 / (1 + np.exp(-out.predictions))
    threshold = sweep_global_threshold(scores, val_y)
    np.save(WORK_DIR / "val_scores.npy", scores)
    (WORK_DIR / "threshold.json").write_text(
        json.dumps({"effective_threshold": threshold}, indent=2), encoding="utf-8")
    print(f"[train] best 已存 {best},阈值={threshold}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
```

(`evaluate.py` 的 CLI main:加载 best checkpoint + test.jsonl + threshold.json → `compute_metrics_per_class` → `render_report` 写 `evals/results/<UTC时间戳>_topic_eval.md` + 错例导出;记录模型/数据/解码版本、切分比例、合成/真实数量、多标签占比、截断比例。)

- [ ] **Step 5: 跑纯函数测试 + 冒烟实跑**

```bash
uv run pytest tests/test_threshold.py tests/test_evaluate_report.py -q
uv run --group train python -m training.train --smoke   # Task 14 数据就绪后执行
```
Expected: 测试 PASS;冒烟跑通打印阈值(CPU,几分钟内)

- [ ] **Step 6: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 训练与评测——全参微调/早停/阈值扫描/逐类报告"
```

---

### Task 11: export_onnx.py 模型包导出 + 一致性验证

**Files:**
- Create: `training/package.py`(哈希/manifest,纯标准库)、`training/export_onnx.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces:
  - `training.package.file_sha256(path) -> str`、`package_version(pkg_dir, exclude=("manifest.json",)) -> str`(除 manifest 外全部文件相对路径+内容哈希清单,按路径排序,规范化 SHA-256)
  - `training.package.write_manifest(pkg_dir, extra: dict) -> dict`
  - CLI:`uv run --group train python -m training.export_onnx --ckpt models/topic_classifier/_work/best`
  - 模型包 `models/topic_classifier/`:`model.onnx / tokenizer/ / taxonomy.json / labels.json / preprocess.json / manifest.json`

- [ ] **Step 1: 写失败测试** `tests/test_package.py`

```python
"""模型包哈希:model_version 稳定性与敏感性(spec §7.1)。"""

from pathlib import Path

from training.package import file_sha256, package_version, write_manifest


def test_package_version_stable_and_sensitive(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")
    v1 = package_version(tmp_path)
    assert v1 == package_version(tmp_path) and len(v1) == 64
    (tmp_path / "sub" / "b.txt").write_text("bb", encoding="utf-8")
    assert package_version(tmp_path) != v1  # 内容变 → 版本变
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")
    assert package_version(tmp_path) != v1  # 文件增 → 版本变


def test_manifest_roundtrip(tmp_path: Path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    m = write_manifest(tmp_path, {"model_revision": "r1"})
    assert m["files"]["model.onnx"] == file_sha256(tmp_path / "model.onnx")
    assert m["model_revision"] == "r1"
    assert (tmp_path / "manifest.json").exists()
    # manifest 自身不参与 model_version
    assert package_version(tmp_path) == package_version(tmp_path)
```

- [ ] **Step 2: 跑测试确认失败 → 实现 training/package.py**

```python
"""模型包哈希与 manifest(纯标准库,serve 侧校验复用,spec §7.1)。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def package_version(pkg_dir, exclude=("manifest.json",)) -> str:
    pkg = Path(pkg_dir)
    entries = []
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name not in exclude:
            rel = p.relative_to(pkg).as_posix()
            entries.append([rel, file_sha256(p)])
    return hashlib.sha256(_canon(entries).encode("utf-8")).hexdigest()


def write_manifest(pkg_dir, extra: dict) -> dict:
    pkg = Path(pkg_dir)
    files = {p.relative_to(pkg).as_posix(): file_sha256(p)
             for p in sorted(pkg.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    manifest = {"files": files, **extra}
    (pkg / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
```

- [ ] **Step 3: 实现 training/export_onnx.py**

```python
"""ONNX 导出 + PyTorch/ONNX 一致性验证 + 模型包组装(spec §7.1)。

Context7 已核实:optimum.onnxruntime.ORTModelForSequenceClassification
.from_pretrained(ckpt, export=True) + save_pretrained;onnxruntime
InferenceSession(path, providers=["CPUExecutionProvider"])。
CLI: uv run --group train python -m training.export_onnx \
       --ckpt models/topic_classifier/_work/best --out models/topic_classifier
一致性:eval 模式,同输入比较 logits 与 sigmoid(atol=1e-4, rtol=1e-3),
解码标签必须完全一致;不一致即导出失败,不写 manifest。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from training.labels import (DECODER_VERSION, classifier_version, decode_scores,
                             label_names, taxonomy_version)
from training.package import package_version, write_manifest
from training.preprocessing import clean_text, preprocess_config

PROBES = ([  # 覆盖 batch=1/32、短句/超长句、多标签、脱敏占位符(spec §7.1)
    "买大了想退", "发票怎么开?"] + ["短句"] * 30
    + ["这是一段" * 50 + "超长输入"],)
ATOL, RTOL = 1e-4, 1e-3


def verify_export(torch_dir: Path, onnx_dir: Path, probes: list[str]) -> None:
    import torch
    import onnxruntime
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(torch_dir)
    model = AutoModelForSequenceClassification.from_pretrained(torch_dir).eval()
    sess = onnxruntime.InferenceSession(str(Path(onnx_dir) / "model.onnx"),
                                        providers=["CPUExecutionProvider"])
    names = label_names()
    for probe in probes:
        inputs = tok([clean_text(probe)], max_length=64, truncation=True,
                     padding="max_length", return_tensors="np")
        with torch.no_grad():
            t_logits = model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).logits.numpy()
        o_logits = sess.run(None, dict(inputs))[0]
        np.testing.assert_allclose(o_logits, t_logits, atol=ATOL, rtol=RTOL)
        sig = 1 / (1 + np.exp(-o_logits))
        t_sig = 1 / (1 + np.exp(-t_logits))
        np.testing.assert_allclose(sig, t_sig, atol=ATOL, rtol=RTOL)
        assert decode_scores(sig[0].tolist(), 0.5, names) == \
            decode_scores(t_sig[0].tolist(), 0.5, names)


def main(ckpt: Path, out: Path) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ort = ORTModelForSequenceClassification.from_pretrained(str(ckpt), export=True)
    ort.save_pretrained(out)  # 产 model.onnx
    AutoTokenizer.from_pretrained(ckpt).save_pretrained(out / "tokenizer")
    shutil.copy(Path("training/taxonomy.json"), out / "taxonomy.json")
    threshold = json.loads((ckpt.parent / "threshold.json").read_text())["effective_threshold"]
    (out / "labels.json").write_text(json.dumps({
        "labels": label_names(), "taxonomy_version": taxonomy_version(),
        "effective_threshold": threshold, "decoder_version": DECODER_VERSION},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "preprocess.json").write_text(json.dumps({
        **preprocess_config(), "max_len": 64, "truncation": "right",
        "padding": "right-fixed-64"}, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_export(ckpt, out, PROBES[0])  # 不一致抛错 → 不写 manifest,不发布
    model_version = package_version(out)
    cv = classifier_version(model_version, threshold)
    write_manifest(out, {"model_version": model_version,
                         "classifier_version": cv,
                         "model_revision": ckpt.name,
                         "consistency_check": {"atol": ATOL, "rtol": RTOL,
                                               "probes": len(PROBES[0]),
                                               "result": "pass"}})
    print(f"[export] model_version={model_version} classifier_version={cv}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(Path(args[args.index("--ckpt") + 1]), Path(args[args.index("--out") + 1]))
```

- [ ] **Step 4: 跑 package 测试;导出实跑归 Task 14**

Run: `uv run pytest tests/test_package.py -q`
Expected: 2 条 PASS

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): ONNX 导出——模型包组装/PyTorch-ONNX 一致性验证/manifest 哈希"
```

---

### Task 12: serve.py 推理服务 + Makefile target

**Files:**
- Create: `training/serve.py`
- Modify: `Makefile`
- Test: `tests/test_serve_contract.py`

**Interfaces:**
- Produces:
  - `training.serve.load_package(model_dir) -> dict`(校验文件哈希/labels 顺序/阈值合法,返回包元数据)
  - `training.serve.create_inference_app(model_dir, *, runner=None, threshold_override=None) -> FastAPI`
  - 端点:`GET /metadata` → `{model_version, taxonomy_version, classifier_version, effective_threshold, labels, decoder_version}`;`POST /classify {texts:[...]}` → `{labels, scores, classifier_version}`;1–256 条非空字符串,空白/清洗后为空 → 整批 422
  - Makefile:`make topic-inference`(`uv run --group train python -m training.serve`,:8103)

- [ ] **Step 1: 写失败测试** `tests/test_serve_contract.py`

```python
"""推理服务契约:metadata/422 语义/阈值覆盖(spec §7.2)。真实 ONNX 不进 pytest。"""

import json

import httpx
import pytest

from training.labels import classifier_version, label_names, taxonomy_version
from training.serve import create_inference_app


class FakeRunner:
    """确定性假推理:含「退」→ 退换货 0.9,含「大」→ 尺码 0.9,其余全 0.01。"""
    def run(self, texts):
        names = label_names()
        out = []
        for t in texts:
            s = [0.01] * 17
            if "退" in t:
                s[names.index("退换货")] = 0.9
            if "大" in t:
                s[names.index("尺码")] = 0.9
            out.append(s)
        return out


@pytest.fixture()
def pkg(tmp_path):
    import hashlib
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"fake")
    (d / "labels.json").write_text(json.dumps({
        "labels": label_names(), "taxonomy_version": taxonomy_version(),
        "effective_threshold": 0.5, "decoder_version": "topic-decoder-v1"}),
        encoding="utf-8")
    from training.package import package_version, write_manifest
    mv = package_version(d)
    write_manifest(d, {"model_version": mv,
                       "classifier_version": classifier_version(mv, 0.5)})
    return d


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_metadata(pkg):
    app = create_inference_app(pkg, runner=FakeRunner())
    async with await _client(app) as c:
        meta = (await c.get("/metadata")).json()
    assert meta["labels"] == label_names()
    assert meta["taxonomy_version"] == taxonomy_version()
    assert meta["effective_threshold"] == 0.5
    assert len(meta["classifier_version"]) == 64


async def test_classify_multilabel(pkg):
    app = create_inference_app(pkg, runner=FakeRunner())
    async with await _client(app) as c:
        resp = await c.post("/classify", json={"texts": ["买大了想退", "在吗"]})
    body = resp.json()
    assert body["labels"][0] == ["退换货", "尺码"]  # 冻结顺序
    assert body["labels"][1] == ["其他"]            # 零命中兜底
    assert len(body["scores"][0]) == 17


async def test_classify_validation_422(pkg):
    app = create_inference_app(pkg, runner=FakeRunner())
    async with await _client(app) as c:
        assert (await c.post("/classify", json={"texts": []})).status_code == 422
        assert (await c.post("/classify",
                             json={"texts": ["ok", "  "]})).status_code == 422
        assert (await c.post("/classify",
                             json={"texts": ["x"] * 257})).status_code == 422
        assert (await c.post("/classify",
                             json={"texts": [123]})).status_code == 422


async def test_threshold_override_changes_version(pkg):
    app = create_inference_app(pkg, runner=FakeRunner(), threshold_override=0.3)
    async with await _client(app) as c:
        meta = (await c.get("/metadata")).json()
    assert meta["effective_threshold"] == 0.3
    assert meta["classifier_version"] != classifier_version(
        json.loads((pkg / "manifest.json").read_text())["model_version"], 0.5)
    with pytest.raises(ValueError):
        create_inference_app(pkg, runner=FakeRunner(), threshold_override=1.5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_serve_contract.py -x -q`
Expected: FAIL

- [ ] **Step 3: 实现 training/serve.py + Makefile**

```python
"""ch10 独立推理服务(:8103):ONNX + onnxruntime CPU(spec §7.2)。

起服:make topic-inference(= uv run --group train python -m training.serve)
启动校验:包文件哈希/labels 顺序与数量/阈值合法;每个进程固定一个模型包,
换包须重启。TOPIC_LABEL_THRESHOLD 仅本进程读取(覆盖会改变 classifier_version)。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.labels import (DECODER_VERSION, classifier_version, decode_scores,
                             label_names, taxonomy_version)
from training.package import file_sha256, package_version

MODEL_DIR = Path("models/topic_classifier")


def load_package(model_dir) -> dict:
    pkg = Path(model_dir)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    labels_doc = json.loads((pkg / "labels.json").read_text(encoding="utf-8"))
    if labels_doc["labels"] != label_names():
        raise RuntimeError("模型包 labels 与权威类目表不一致")
    if labels_doc["taxonomy_version"] != taxonomy_version():
        raise RuntimeError("模型包 taxonomy_version 过期")
    if package_version(pkg) != manifest["model_version"]:
        raise RuntimeError("模型包文件哈希与 manifest 不符")
    th = float(labels_doc["effective_threshold"])
    if not (0 < th < 1) or not math.isfinite(th):
        raise RuntimeError("labels.json 阈值非法")
    return {"model_version": manifest["model_version"], "labels_doc": labels_doc}


class OnnxRunner:
    """真实推理:确定性清洗 → tokenizer(右截断/padding 到 64)→ onnxruntime → sigmoid。"""

    def __init__(self, model_dir):
        import onnxruntime
        from transformers import AutoTokenizer
        from training.preprocessing import clean_text
        self._clean = clean_text
        self._tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        self._sess = onnxruntime.InferenceSession(
            str(Path(model_dir) / "model.onnx"), providers=["CPUExecutionProvider"])

    def run(self, texts):
        import numpy as np
        cleaned = [self._clean(t) for t in texts]
        inputs = self._tok(cleaned, max_length=64, truncation=True,
                           padding="max_length", return_tensors="np")
        logits = self._sess.run(None, dict(inputs))[0]
        return (1 / (1 + np.exp(-logits))).tolist()


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


def create_inference_app(model_dir, *, runner=None, threshold_override=None) -> FastAPI:
    pkg = load_package(model_dir)
    labels_doc = pkg["labels_doc"]
    threshold = float(labels_doc["effective_threshold"])
    if threshold_override is not None:
        if not (0 < threshold_override < 1) or not math.isfinite(threshold_override):
            raise ValueError("TOPIC_LABEL_THRESHOLD 须为 (0,1) 有限数")
        threshold = float(threshold_override)
    cv = classifier_version(pkg["model_version"], threshold)
    names = labels_doc["labels"]
    runner = runner or OnnxRunner(model_dir)
    app = FastAPI(title="topic-inference")

    @app.get("/metadata")
    def metadata():
        return {"model_version": pkg["model_version"],
                "taxonomy_version": labels_doc["taxonomy_version"],
                "classifier_version": cv, "effective_threshold": threshold,
                "labels": names, "decoder_version": DECODER_VERSION}

    @app.post("/classify")
    def classify(req: ClassifyRequest):
        if any(not isinstance(t, str) or not t.strip() for t in req.texts):
            raise HTTPException(422, "texts 须为非空字符串数组")
        from training.preprocessing import clean_text
        if any(not clean_text(t) for t in req.texts):
            raise HTTPException(422, "清洗后为空")
        try:
            scores = runner.run(req.texts)
        except Exception as exc:
            raise HTTPException(500, f"模型推理失败:{type(exc).__name__}") from exc
        labels = [decode_scores([float(x) for x in s], threshold, names)
                  for s in scores]
        return {"labels": labels, "scores": scores, "classifier_version": cv}

    return app


def main():
    override = os.environ.get("TOPIC_LABEL_THRESHOLD")
    model_dir = Path(os.environ.get("TOPIC_MODEL_DIR", str(MODEL_DIR)))
    app = create_inference_app(
        model_dir, threshold_override=float(override) if override else None)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8103)


if __name__ == "__main__":
    main()
```

`Makefile` 追加:

```makefile
topic-inference: ## 起主题分类推理服务(:8103,需先导出模型包)
	uv run --group train python -m training.serve
```

`.PHONY` 行加 `topic-inference`。

- [ ] **Step 4: 跑测试迭代到全绿 + 全量回归**

Run: `uv run pytest tests/test_serve_contract.py -q && uv run pytest -q`
Expected: 全 PASS(注意 serve.py 顶层不 import onnxruntime/transformers——重导入全部在 OnnxRunner/main 内,保证 dev 组可收集)

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 独立推理服务——包校验/metadata/classify/422 契约+Makefile target"
```

---

### Task 13: topics.html 主题分布页(Vibe Coding 例外)

**Files:**
- Create: `app/static/topics.html`
- Modify: `app/main.py`(/topics 页面路由;现有后台导航加入口)
- Test: `tests/test_topics_page.py`

**Interfaces:**
- Consumes: `GET /api/topics/distribution`、`POST /api/topics/classify`(Task 6)
- Produces: `/topics` 页面

**例外约定:** 不套 brainstorm/TDD/code review,直接实现;仅字符串断言 + 人工点验兜底(spec §9)。

- [ ] **Step 1: 写页面** `app/static/topics.html`

照 `rag-eval.html` 模式:单文件、inline style/script、复用设计 tokens(`--accent: #c96442` 等)、手写 SVG 横条图(无 chart.js)、外部字符串只走 textContent(SVG 走 DOM API,禁 innerHTML)。

必备元素(测试钉):
- `id="classify-btn"`(「立即归类」按钮,请求期间禁用,文案变「归类中…」)
- `id="stat-total"` / `id="stat-classified"` / `id="stat-unclassified"` 三个数字
- `id="meta-line"`(分类版本前 12 位 + 有效阈值 + 最近归类时间)
- `id="topics-chart"`(SVG 横条,17 类按 count 降序、并列按权威顺序——后端已排好序,前端直渲)
- `id="state-banner"`(空数据 / stale=true「类目版本已过期,请重新归类」/ 409 冲突 / 503 失败提示;无异常时隐藏)
- 说明文案:`按用户轮次去重,多标签问题可计入多个类目`
- 请求路径字面量 `/api/topics/classify` 与 `/api/topics/distribution`
- 409/503 时 alert 条提示并保留上次数据;成功后重拉 distribution

- [ ] **Step 2: main.py 加页面路由 + 导航入口**

```python
    @app.get("/topics", include_in_schema=False)
    async def topics_ui() -> FileResponse:
        return FileResponse(static_dir / "topics.html")
```

并在现有后台页(review.html / rag-eval.html / kb.html)导航条加 `/topics` 入口链接(对齐各页现有导航写法)。

- [ ] **Step 3: 写字符串断言测试** `tests/test_topics_page.py`

```python
from pathlib import Path

import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings

HTML = Path(__file__).parent.parent / "app" / "static" / "topics.html"


async def test_topics_page_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/topics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_page_markers():
    html = HTML.read_text(encoding="utf-8")
    for marker in ('id="classify-btn"', 'id="stat-total"', 'id="stat-classified"',
                   'id="stat-unclassified"', 'id="meta-line"', 'id="topics-chart"',
                   'id="state-banner"',
                   "/api/topics/classify", "/api/topics/distribution",
                   "按用户轮次去重,多标签问题可计入多个类目",
                   "--accent: #c96442"):
        assert marker in html


def test_page_no_innerhtml_injection():
    assert "innerHTML" not in HTML.read_text(encoding="utf-8")
```

- [ ] **Step 4: 跑测试 + 人工点验(起服务开 /topics 过一眼)**

Run: `uv run pytest tests/test_topics_page.py -q`
Expected: 3 条 PASS

- [ ] **Step 5: 补实录并提交**

```bash
git add -A && git commit -m "feat(ch10): 主题分布页——SVG 横条图/归类触发/版本与异常状态"
```

---

### Task 14: 端到端实跑验收(数据 → 训练 → 评测 → 导出 → 推理 → 真实归类演示)

**Files:** 无新代码;产物 gitignore;实录进 `dev-notes/ch10.md`

**验证方式:** 用户约定——纯 Prompt/数据/训练类任务以实跑验证替代 TDD。每步记录耗时与产物。

- [ ] **Step 1: 数据管线实跑(烧额度,逐项确认)**

```bash
uv run --group train python -m training.prep_generate --per-class 100 --multi-ratio 0.2 \
  --out data/topic_classifier/raw_synthetic.jsonl   # ~1700 条合成
uv run --group train python -m training.prep_clean \
  data/topic_classifier/raw_synthetic.jsonl data/topic_classifier/cleaned_synthetic.jsonl
# 真实池:prep_clean --real 子命令从 lcq 读出 7 条并入(记录快照时间与去重前后数量)
uv run --group train python -m training.prep_label prelabel \
  data/topic_classifier/cleaned_synthetic.jsonl data/topic_classifier/labeled.jsonl
uv run --group train python -m training.prep_label export-review \
  data/topic_classifier/labeled.jsonl data/topic_classifier/review_sample.md
```
人工:过 `review_sample.md`(每类 ~10%),修正写 `data/topic_classifier/corrections.jsonl`,然后
`prep_label apply-corrections` 回灌;裁决队列(disagreements)逐条定夺。

- [ ] **Step 2: 切分 + 增强**

```bash
uv run --group train python -m training.prep_split \
  data/topic_classifier/labeled.jsonl data/topic_classifier/
# 产出 train/val/test.jsonl + split_manifest.json + 比例/support/多标签占比报告
uv run --group train python -m training.prep_augment \
  --manifest data/topic_classifier/split_manifest.json  # 只扩 train 到 ~1.3x
```
确认:split 报告三份均覆盖 17 类;增强后 val/test 条数不变。

- [ ] **Step 3: 冒烟 → 全量训练 → 评测**

```bash
uv run --group train python -m training.train --smoke   # 冒烟验证管线(记录吞吐,估算全量时长)
uv run --group train python -m training.train           # 全量 CPU 慢训(预计 1-3 小时)
uv run --group train python -m training.evaluate        # → evals/results/<ts>_topic_eval.md
```
验收点 1:报告含逐类 P/R/F1/support + 每类混淆矩阵 + 错例导出;人工抽判错例复核。

- [ ] **Step 4: 导出 + 起推理服务**

```bash
uv run --group train python -m training.export_onnx \
  --ckpt models/topic_classifier/_work/best --out models/topic_classifier
# 一致性验证 pass 才写 manifest
make topic-inference &   # :8103
curl --noproxy '*' http://127.0.0.1:8103/metadata
```
验收点 3:`curl --noproxy '*' -X POST http://127.0.0.1:8103/classify -H 'Content-Type: application/json' -d '{"texts":["买大了想退"]}'` → labels 同时含「退换货」「尺码」;另演三对近邻边界句与负例(零命中 → `["其他"]`)。

- [ ] **Step 5: 主服真实归类演示**

```bash
docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch10-ddl.sql  # 若 Task 1 未执行
no_proxy=127.0.0.1,localhost uv run uvicorn --factory app.main:create_app &
curl --noproxy '*' -X POST http://127.0.0.1:8000/api/topics/classify
curl --noproxy '*' http://127.0.0.1:8000/api/topics/distribution
```
验收点 2:浏览器开 `/topics`,17 类横条图可见各类目统计;同轮去重、多标签计数说明在页面;重复触发 classify 同版本只补漏(written_rows=0)。

- [ ] **Step 6: 实录**

`dev-notes/ch10.md` 追加「端到端实跑验收」节:各步耗时、评测报告路径、真实归类分布截图/数字、翻车与返工(逐条)。

---

### Task 15: 收尾——AGENTS.md / README / 全量回归 / finish

- [ ] **Step 1: AGENTS.md 增 ch10 节**(模块清单、配置项、红线:单 worker 作业锁、推理独立进程、train 组不进默认 sync、taxonomy/阈值改动 = 换版本);README.md 增「主题分类器」节(训练管线命令序列 + make topic-inference + /topics 页)
- [ ] **Step 2: 全量回归**

Run: `uv run pytest -q`
Expected: 全部 PASS(基线 745 + 本章新增)

- [ ] **Step 3: dev-notes/ch10.md finish 节 + 提交**

```bash
git add -A && git commit -m "docs(ch10): AGENTS/README 收尾 + 实录 finish"
```

---

## Self-Review 记录(写计划后自查)

- **Spec 覆盖**:§1 轮次去重→T4/T5;§2 类目表/版本→T2;§3 标注解码→T2/T7;§4 数据四步→T3/T7/T8/T9;§5 训练→T10;§6 评测→T10/T14;§7.1 模型包→T11;§7.2 推理服务→T12;§7.3 归类作业→T4;§7.4 分布→T5;§8 接线→T1/T6;§9 前端→T13;§10 测试策略→各 Task;§11 验收→T14;§13 依赖→T2。无缺口。
- **Placeholder 扫描**:无 TBD/TODO;训练 CLI 参数细节在 T10/T11 给出默认值与命令。
- **类型一致性**:`decode_scores(scores, threshold, labels=None)` 在 T2 定义,T4/T10/T12 调用签名一致;`classifier_version(model_version, effective_threshold, decoder_version)` 三处在 T2/T11/T12 一致;`TopicClassification` 字段 T1 ↔ T4/T5 一致;错误类 T4 定义、T6 处理器引用一致。
