# /rag-eval 评估页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RAG 评估从终端命令落成后台一页 `/rag-eval`:四策略检索对照、证据/答案覆盖、编造个案台账(可处置)、页面上可重跑评估作业。

**Architecture:** 评估管道(`evals/run_retrieval_compare.py`)扩为四策略生成段 + 裁判覆盖分,跑完产出规范化 `evals/results/rag_eval.json`(原子发布);服务侧新增最小作业层(`app/jobs/runner.py`,asyncio 子进程跑同一命令)与两个路由(`/api/jobs`、`/api/rag-eval`),页面为纯静态 HTML + 手绘 SVG 图表,零新依赖。

**Tech Stack:** Python 3.13 / FastAPI / SQLAlchemy 2.0(MySQL)/ pytest + pytest-asyncio / uv / 纯静态 HTML+JS。

**Spec:** `docs/superpowers/specs/2026-09-16-rag-eval-page-design.md`

## Global Constraints

- 四策略固定为 `("dense", "bm25", "hybrid", "hybrid_rerank")`;可作答桶 `("A_policy", "B_model", "C_colloquial", "E_multi")`,D_absent 只进拒答统计。
- 证据覆盖度 = CompleteHit@10;答案覆盖度 = 裁判 coverage 分(拒答记 0),分母 = 可作答桶全部 test 题;忠实率 = faithful/judged,任一有 judge_error → 该策略/桶比率为 `null`。
- 只有 hybrid_rerank 的 fabricated 写 `faith_cases` 台账;citations 新形状 `{"run_id", "evidence", "cited_refs"}`;「已解决」= 确认幻觉。
- `rag_eval.json` 固定四个顶层键 `meta/retrieval/generation/gates`,原子发布(`os.replace`),是本轮最后发布的文件;`gates.passed=false` 也照常发布。
- 不新增任何 Python/前端依赖;不引入图表库;`sql/ch04-ddl.sql` 与 `db/init/04-ddl.sql` 保持字节一致(`tests/test_ddl_sync.py` 强制)。
- 测试命令:`uv run pytest tests/<file> -v`;DB 测试需 `docker compose up -d` 在线(dbfixtures 不在线会 `pytest.exit`,会中断整个测试会话,注意单独跑)。
- 提交风格沿用仓库:中文 conventional commit,如 `feat(rag-eval): ...`。

---

### Task 1: 裁判 coverage 维度(JUDGE_PROMPT + parse_judge_output)

**Files:**
- Modify: `app/prompts/faithfulness.py`(全文替换)
- Modify: `evals/run_retrieval_compare.py:129-142`(`parse_judge_output`)+ `_judge`(:234-246)与其调用处(:269)
- Test: `tests/test_eval_metrics.py:137-145`(更新 `test_parse_judge_output` 并新增覆盖分用例)

**Interfaces:**
- Consumes: 现有 `JUDGE_PROMPT` 的 `{query}/{answer}/{evidence}` 占位。
- Produces: `parse_judge_output(text) -> {"verdict", "unsupported_claims", "cited_refs", "coverage"(float)}`;`_judge(model, query, answer, evidence, expect_points)`;`JUDGE_PROMPT` 新增 `{expect_points}` 占位。

- [ ] **Step 1: 更新测试(先红)**

`tests/test_eval_metrics.py` 中替换 `test_parse_judge_output` 并新增:

```python
def test_parse_judge_output():
    out = parse_judge_output('{"verdict": "fabricated", "unsupported_claims": '
                             '[{"claim": "c", "reason": "r"}], "cited_refs": [1, 3], '
                             '"coverage": 0.5}')
    assert out["verdict"] == "fabricated" and out["cited_refs"] == [1, 3]
    assert out["coverage"] == 0.5
    import pytest
    with pytest.raises(ValueError):
        parse_judge_output("not json")
    with pytest.raises(ValueError):
        parse_judge_output('{"verdict": "maybe", "unsupported_claims": [], '
                           '"cited_refs": [], "coverage": 0.5}')


def test_parse_judge_output_coverage_validation():
    """coverage 缺失/字符串/bool/NaN/越界 → ValueError(计入重试与 judge_error)。"""
    import pytest
    base = '"verdict": "faithful", "unsupported_claims": [], "cited_refs": []'
    for payload in (
        "{" + base + "}",                                  # 缺失
        "{" + base + ', "coverage": "0.5"}',               # 字符串
        "{" + base + ', "coverage": true}',                # bool
        "{" + base + ', "coverage": NaN}',                 # 非有限(json.loads 会放行 NaN)
        "{" + base + ', "coverage": 1.5}',                 # 越界
    ):
        with pytest.raises(ValueError):
            parse_judge_output(payload)
    out = parse_judge_output("{" + base + ', "coverage": 1}')   # int 合法
    assert out["coverage"] == 1.0 and isinstance(out["coverage"], float)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_eval_metrics.py::test_parse_judge_output tests/test_eval_metrics.py::test_parse_judge_output_coverage_validation -v`
Expected: FAIL(valid payload 缺 coverage 报 ValueError / coverage 键不存在)

- [ ] **Step 3: 实现**

`app/prompts/faithfulness.py` 全文替换:

```python
JUDGE_PROMPT = """你是回答忠实度与覆盖度裁判。给定用户问题、客服回答、当轮提供给模型的证据列表(编号即回答中的引用角标)、以及该题的标准要点,判断回答是否忠实于证据,并给答案覆盖分。

判定规则:
- faithful:回答中的每个事实性陈述都能在证据中找到依据;引用角标与证据编号对应正确。
- fabricated:回答含证据中没有的事实性陈述(数字、时限、政策、承诺等),或引用了不存在的证据编号。

覆盖分 coverage 规则(逐要点二值):
- 答案明确覆盖一个标准要点记 1,否则记 0;coverage = 覆盖要点数 / 要点总数,取值 0.0 到 1.0。
- 同义表达算覆盖;回答中额外的正确信息不加分。

只输出一行 JSON:
{"verdict": "faithful" 或 "fabricated", "unsupported_claims": [{"claim": "编造的那句话", "reason": "为什么证据不支持"}], "cited_refs": [回答实际引用的编号,整数列表], "coverage": 0.0 到 1.0 的数值}

用户问题:{query}
客服回答:{answer}
证据列表:{evidence}
标准要点:{expect_points}"""
```

`evals/run_retrieval_compare.py` 顶部 import 加 `import math`;`parse_judge_output` 替换为:

```python
def parse_judge_output(text: str) -> dict:
    m = _JUDGE_JSON_RE.search(text or "")
    if not m:
        raise ValueError("judge output unparseable")
    data = json.loads(m.group(0))
    if data.get("verdict") not in ("faithful", "fabricated"):
        raise ValueError("judge verdict invalid")
    claims = data.get("unsupported_claims")
    refs = data.get("cited_refs")
    if not isinstance(claims, list) or not isinstance(refs, list):
        raise ValueError("judge fields invalid")
    coverage = data.get("coverage")
    if (isinstance(coverage, bool) or not isinstance(coverage, (int, float))
            or not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0):
        raise ValueError("judge coverage invalid")
    return {"verdict": data["verdict"],
            "unsupported_claims": claims,
            "cited_refs": [int(x) for x in refs],
            "coverage": float(coverage)}
```

`_judge` 加 `expect_points` 参(签名变 `(model, query, answer, evidence, expect_points)`),prompt 拼接待 Task 1 内同步改:

```python
def _judge(model, query: str, answer: str, evidence: list[dict],
           expect_points) -> tuple[dict | None, str | None]:
    """忠实度+覆盖度裁判;失败(模型异常/解析非法)重试一次,再失败返回 (None, 错误描述)。"""
    prompt = (JUDGE_PROMPT.replace("{query}", query).replace("{answer}", answer)
              .replace("{evidence}", json.dumps(evidence, ensure_ascii=False))
              .replace("{expect_points}", "、".join(expect_points)))
    err = None
    for _ in range(2):
        try:
            resp = model.invoke([("user", prompt)])
            return (parse_judge_output(resp.content
                                       if isinstance(resp.content, str) else ""), None)
        except Exception as exc:  # 模型异常与 ValueError 同样计入重试
            err = f"{type(exc).__name__}: {exc}"
    return None, err
```

`_generation_segment` 内唯一调用处(:269)改为 `verdict, err = _judge(model, c["query"], answer, evidence, c["expect_points"])`,并在它成功分支补 `row["coverage"] = verdict["coverage"]`(该行 Task 4 还会重构,此处只为保绿)。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS(含原有全部用例)

- [ ] **Step 5: Commit**

```bash
git add app/prompts/faithfulness.py evals/run_retrieval_compare.py tests/test_eval_metrics.py
git commit -m "feat(rag-eval): 裁判增 coverage 维度(逐要点二值,严校验非 bool/有限/[0,1])"
```

---

### Task 2: 检索段 by_bucket 三指标聚合

**Files:**
- Modify: `evals/run_retrieval_compare.py:197-202`(`_test_metrics` 的 by_bucket 块)
- Test: `tests/test_eval_metrics.py`(末尾新增用例)

**Interfaces:**
- Consumes: per-case 已算的 `recall5/recall10/ch5/ch10/mrr10/passed`(:183-190)。
- Produces: `by_bucket[bucket] = {"section_recall_at_10", "mrr_at_10", "recall5", "evidence_coverage", "refused"}`(Task 5 的 `build_rag_eval` 依赖 `mrr_at_10/recall5/evidence_coverage` 三个键名)。

- [ ] **Step 1: 写失败测试**

```python
def test_test_metrics_by_bucket_three_metrics():
    """by_bucket 聚合 MRR/Recall@5/证据覆盖度(CH@10);D 桶不进 by_bucket。"""
    from evals.run_retrieval_compare import _test_metrics
    G2 = (("规格",), ("保修",))
    cases = [
        {"bucket": "A_policy", "should_refuse": False, "gt_groups": G2,
         "retrieval": {"dense": {"top1": 0.9, "paths": ["售后 > 保修说明", "产品规格 > 规格"]}}},
        {"bucket": "A_policy", "should_refuse": False, "gt_groups": G2,
         "retrieval": {"dense": {"top1": 0.8, "paths": ["无关"]}}},
        {"bucket": "C_colloquial", "should_refuse": False, "gt_groups": G2,
         "retrieval": {"dense": {"top1": 0.7, "paths": ["产品规格 > 规格"]}}},
        {"bucket": "D_absent", "should_refuse": True, "gt_groups": (),
         "retrieval": {"dense": {"top1": None, "paths": []}}},
    ]
    m = _test_metrics(cases, "dense", 0.5)
    a = m["by_bucket"]["A_policy"]
    assert a["mrr_at_10"] == 0.5            # (1.0 + 0.0) / 2
    assert a["recall5"] == 0.5
    assert a["evidence_coverage"] == 0.5    # CH@10:(1 + 0) / 2
    assert a["section_recall_at_10"] == 0.5 and a["refused"] == 0
    c = m["by_bucket"]["C_colloquial"]
    assert c["recall5"] == 0.5 and c["evidence_coverage"] == 0.0 and c["mrr_at_10"] == 1.0
    assert "D_absent" not in m["by_bucket"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_eval_metrics.py::test_test_metrics_by_bucket_three_metrics -v`
Expected: FAIL(KeyError: 'mrr_at_10')

- [ ] **Step 3: 实现**

`_test_metrics` 的 by_bucket 块替换为:

```python
    by_bucket = {}
    for b in BUCKETS:
        bp = [p for p in pos if p["bucket"] == b]
        if bp:
            by_bucket[b] = {
                "section_recall_at_10": avg([p["recall10"] for p in bp]),
                "mrr_at_10": avg([p["mrr10"] for p in bp]),
                "recall5": avg([p["recall5"] for p in bp]),
                "evidence_coverage": avg([p["ch10"] for p in bp]),
                "refused": sum(1 for p in bp if not p["passed"]),
            }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/run_retrieval_compare.py tests/test_eval_metrics.py
git commit -m "feat(rag-eval): 检索段 by_bucket 补 MRR@10/Recall@5/证据覆盖度(CH@10)聚合"
```

---

### Task 3: faith_cases citations 新形状(run_id + evidence + cited_refs)

**Files:**
- Modify: `evals/run_retrieval_compare.py:145-165`(`upsert_faith_case`)、`_generation_segment`(:249-288,加 `run_id` 参并透传)、`main`(:359 起,生成 run_id 并传入;`out` 加 `"run_id"`)
- Modify: `sql/ch04-ddl.sql:44` 与 `db/init/04-ddl.sql` 同一行(citations 的 COMMENT,两文件保持字节一致)
- Test: `tests/test_faith_cases.py`(全文更新)

**Interfaces:**
- Produces: `upsert_faith_case(s, *, run_id, eval_id, bucket, query, answer, reason, evidence, cited_refs, judge_model) -> None`;落库 `citations = {"run_id": str, "evidence": list, "cited_refs": list[int]}`。Task 8 的 `normalize_citations` 依赖此形状。

- [ ] **Step 1: 更新测试(先红)**

`tests/test_faith_cases.py` 全文:

```python
"""faith_cases upsert 复发语义(ch04 T12):一题一行;已解决复发退回未解决。
citations 新形状(rag-eval):{run_id, evidence, cited_refs}。"""

from datetime import datetime

from app.models import FaithCase
from evals.run_retrieval_compare import upsert_faith_case
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def test_upsert_insert_then_recurrence(db_session_factory):
    with db_session_factory() as s:
        upsert_faith_case(s, run_id="r1", eval_id="A43", bucket="A_policy", query="q",
                          answer="a1", reason="r1", evidence=[{"ref_no": 1}],
                          cited_refs=[1], judge_model="m")
        s.commit()
    with db_session_factory() as s:  # 人工标已解决
        row = s.query(FaithCase).one()
        row.status = "已解决"; row.resolution = "补了文档"
        row.resolved_at = datetime(2026, 9, 15, 12, 0, 0)
        s.commit()
    with db_session_factory() as s:  # 复发:同一 eval_id 再判编造
        upsert_faith_case(s, run_id="r2", eval_id="A43", bucket="A_policy", query="q",
                          answer="a2", reason="r2", evidence=[{"ref_no": 2}],
                          cited_refs=[2], judge_model="m")
        s.commit()
    with db_session_factory() as s:
        row = s.query(FaithCase).one()
        assert row.seen_count == 2 and row.answer == "a2"
        assert row.status == "未解决" and row.resolution is None   # 复发退回,处置清空
        assert row.resolved_at == datetime(2026, 9, 15, 12, 0, 0)  # DDL 语义:复发后仍保留
        assert row.citations == {"run_id": "r2", "evidence": [{"ref_no": 2}],
                                 "cited_refs": [2]}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_faith_cases.py -v`(需 docker compose 在线)
Expected: FAIL(TypeError: unexpected keyword 'run_id')

- [ ] **Step 3: 实现**

`upsert_faith_case` 替换为:

```python
def upsert_faith_case(s, *, run_id, eval_id, bucket, query, answer, reason,
                      evidence, cited_refs, judge_model) -> None:
    """一题一行;重判更新快照 + seen_count+1;已解决复发退回未解决并清空 resolution。
    citations = {run_id, evidence, cited_refs}:哪一轮 / Top-K 证据全集 / 答案引用的 ref_no。"""
    from app.models import FaithCase
    citations = {"run_id": run_id, "evidence": evidence, "cited_refs": cited_refs}
    row = s.query(FaithCase).filter_by(eval_id=eval_id).first()
    now = datetime.now()
    if row is None:
        s.add(FaithCase(eval_id=eval_id, bucket=bucket, query=query,
                        strategy="hybrid_rerank", answer=answer, reason=reason,
                        citations=citations, judge_model=judge_model,
                        first_seen_at=now, last_seen_at=now))
        return
    row.answer = answer
    row.reason = reason
    row.citations = citations
    row.judge_model = judge_model
    row.seen_count += 1
    row.last_seen_at = now
    if row.status == "已解决":
        row.status = "未解决"
        row.resolution = None   # 处置清空;resolved_at 保留,用来标「复发过」
```

`_generation_segment` 签名加 `run_id`(:249 行 def 行末加参,docstring 不动),其 fabricated 分支(:279-286)改为:

```python
                if verdict["verdict"] == "fabricated":
                    with main_sf() as s:
                        upsert_faith_case(s, run_id=run_id, eval_id=c["id"],
                                          bucket=c["bucket"], query=c["query"],
                                          answer=answer,
                                          reason=json.dumps(verdict["unsupported_claims"],
                                                            ensure_ascii=False),
                                          evidence=evidence,
                                          cited_refs=verdict["cited_refs"],
                                          judge_model=settings.model_name)
                        s.commit()
```

`main()`:`max_d_pass = _parse_max_d_pass(argv)` 之后加

```python
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # 含微秒,连跑不重
```

`_generation_segment(...)` 调用处(:448-449)加传 `run_id`;`out` 字典(:472-490)加一行 `"run_id": run_id,`。

DDL 注释:两份文件(`sql/ch04-ddl.sql`、`db/init/04-ddl.sql`)中 citations 行的 COMMENT 均改为:

```
COMMENT '本轮 Top-K 证据快照:{run_id,evidence:[{ref_no,chunk_id,section_path,question,answer,category}],cited_refs:[答案实际引用的 ref_no]};2026-09-16 前为裸 evidence 列表,更老为 NULL'
```

(仅注释,无结构变更;已有库可另跑 `ALTER TABLE faith_cases MODIFY citations JSON NULL COMMENT '...'`,非必须。)

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_faith_cases.py tests/test_ddl_sync.py tests/test_eval_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/run_retrieval_compare.py sql/ch04-ddl.sql db/init/04-ddl.sql tests/test_faith_cases.py
git commit -m "feat(rag-eval): faith_cases.citations 改 {run_id,evidence,cited_refs} 形状,DDL 注释同步"
```

---

### Task 4: 生成段四策略函数化 + 聚合口径 + 不变量

**Files:**
- Create: `evals/rag_eval_report.py`(聚合与不变量;构建/发布在 Task 5 加入)
- Modify: `evals/run_retrieval_compare.py`:`_simulate_second_turn`(:216-231)加 `effective_strategy` 参;`_generation_segment` 删除,替换为 `_run_generation` + `_public_row`;`main` 生成段调用处(:448-449)临时只接 hybrid_rerank 保绿(Task 5 接四策略)
- Test: `tests/test_rag_eval_report.py`(新建)、`tests/test_eval_metrics.py`(可在此加 `_run_generation` 用例)

**Interfaces:**
- Produces(后续任务依赖的确切名字):
  - `evals.rag_eval_report.ANSWERABLE_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")`
  - `evals.rag_eval_report.STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")`
  - `aggregate_generation(rows: list[dict]) -> dict`,返回键:`answerable_total, answered, answerable_refused, judged, faithful, fabricated, judge_errors, answer_coverage, faithful_rate, d_total, d_refused, d_refuse_rate, degraded, by_bucket{bucket: {total, answered, refused, judged, faithful, fabricated, judge_errors, coverage, faithful_rate}}`
  - `check_generation_invariants(agg: dict) -> None`(违反 raise RuntimeError)
  - `_run_generation(test_cases, strat, threshold, settings, model) -> list[dict]`,行键:`id, bucket, should_refuse, low_confidence, refused, answer, degraded, evidence, judge, unsupported_claims?, cited_refs?, coverage?, judge_error?`
  - `_public_row(row) -> dict`(去掉 `evidence`)
  - `_simulate_second_turn(model, query, evidence, effective_strategy)`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_rag_eval_report.py`:

```python
"""生成段聚合口径(rag-eval 规格:生成指标口径一节)。"""

import pytest

from evals.rag_eval_report import (
    aggregate_generation, check_generation_invariants,
)


def _row(cid, bucket, refused=False, judge=None, coverage=None,
         should_refuse=False, degraded=False):
    r = {"id": cid, "bucket": bucket, "should_refuse": should_refuse,
         "low_confidence": False, "refused": refused, "answer": "a",
         "degraded": degraded, "evidence": [], "judge": judge}
    if coverage is not None:
        r["coverage"] = coverage
    return r


def test_aggregate_counts_and_ratios():
    rows = [
        _row("A2", "A_policy", judge="faithful", coverage=1.0),
        _row("A4", "A_policy", judge="fabricated", coverage=0.5),
        _row("C2", "C_colloquial", refused=True),              # 拒答 coverage 记 0
        _row("D2", "D_absent", refused=True, should_refuse=True),
        _row("D4", "D_absent", judge=None, should_refuse=True),  # D 未拒答
    ]
    agg = aggregate_generation(rows)
    assert agg["answerable_total"] == 3 and agg["answered"] == 2
    assert agg["answerable_refused"] == 1
    assert agg["judged"] == 2 and agg["faithful"] == 1 and agg["fabricated"] == 1
    assert agg["judge_errors"] == 0
    assert agg["answer_coverage"] == pytest.approx(1.5 / 3)   # 拒答记 0,分母含拒答
    assert agg["faithful_rate"] == pytest.approx(0.5)         # 分母只含裁判成功的
    assert agg["d_total"] == 2 and agg["d_refused"] == 1
    assert agg["d_refuse_rate"] == pytest.approx(0.5)
    assert agg["by_bucket"]["C_colloquial"]["coverage"] == 0.0
    assert agg["by_bucket"]["A_policy"]["faithful_rate"] == pytest.approx(0.5)
    check_generation_invariants(agg)                          # 不抛即过


def test_aggregate_judge_error_nulls():
    """任一裁判失败:策略与受影响桶的比率为 null;未受影响桶照常。"""
    rows = [
        _row("A2", "A_policy", judge="judge_error"),
        _row("C2", "C_colloquial", judge="faithful", coverage=1.0),
    ]
    agg = aggregate_generation(rows)
    assert agg["answer_coverage"] is None and agg["faithful_rate"] is None
    assert agg["by_bucket"]["A_policy"]["coverage"] is None
    assert agg["by_bucket"]["A_policy"]["faithful_rate"] is None
    assert agg["by_bucket"]["C_colloquial"]["faithful_rate"] == 1.0
    assert agg["judge_errors"] == 1 and agg["judged"] == 1
    check_generation_invariants(agg)   # judged + judge_errors == answered 仍成立


def test_check_generation_invariants_raises():
    with pytest.raises(RuntimeError):
        check_generation_invariants({"answered": 2, "answerable_refused": 1,
                                     "answerable_total": 4,   # 2+1 != 4
                                     "judged": 2, "judge_errors": 0,
                                     "faithful": 1, "fabricated": 1,
                                     "d_refused": 0, "d_total": 0})
```

`tests/test_eval_metrics.py` 末尾加:

```python
def test_simulate_second_turn_uses_real_effective_strategy():
    """rerank 降级时工具出参用实际 effective_strategy,不再硬编码 hybrid_rerank。"""
    from evals.run_retrieval_compare import _simulate_second_turn

    class _Cap:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            class R:
                content = "答"
            return R()

    m = _Cap()
    _simulate_second_turn(m, "q", [], "hybrid")
    import json as _json
    payload = _json.loads(m.messages[-1].content)
    assert payload["effective_strategy"] == "hybrid"


def test_run_generation_counts():
    """_run_generation:低置信硬拒答 / 正常裁判 / D 桶不送裁判 / degraded 标记。"""
    from evals.run_retrieval_compare import _run_generation

    class _Fake:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            class R:
                content = "答"
            if isinstance(messages, list) and messages and isinstance(
                    messages[0], tuple):                       # 裁判调用
                R.content = ('{"verdict": "faithful", "unsupported_claims": [], '
                             '"cited_refs": [1], "coverage": 1.0}')
            return R()

    class _Settings:
        rerank_top_n = 10
        max_tool_result_chars = 4000

    cases = [
        {"id": "A2", "bucket": "A_policy", "should_refuse": False,
         "query": "q1", "expect_points": ("p",),
         "retrieval": {"dense": {"top1": 0.9, "hits": [], "effective_strategy": "dense"}}},
        {"id": "A4", "bucket": "A_policy", "should_refuse": False,
         "query": "q2", "expect_points": ("p",),
         "retrieval": {"dense": {"top1": 0.1, "hits": [], "effective_strategy": "dense"}}},
        {"id": "D2", "bucket": "D_absent", "should_refuse": True,
         "query": "q3", "expect_points": (),
         "retrieval": {"dense": {"top1": None, "hits": [], "effective_strategy": "dense"}}},
    ]
    rows = _run_generation(cases, "dense", 0.5, _Settings(), _Fake())
    a2, a4, d2 = rows
    assert a2["judge"] == "faithful" and a2["coverage"] == 1.0 and not a2["refused"]
    assert a4["refused"] and a4["low_confidence"] and a4["judge"] is None
    assert d2["refused"] and d2["judge"] is None               # D 桶不送裁判
    assert all(not r["degraded"] for r in rows)
```

注:`_run_generation` 的 `hits=[]` 时 `assemble_evidence` 返回空列表,`_simulate_second_turn` 走 fake 模型;裁判调用经 `[("user", prompt)]`(list of tuple)与生成调用(list of Message)区分,如上 fake。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rag_eval_report.py tests/test_eval_metrics.py::test_simulate_second_turn_uses_real_effective_strategy -v`
Expected: FAIL(ModuleNotFoundError / TypeError)

- [ ] **Step 3: 实现**

新建 `evals/rag_eval_report.py`:

```python
"""rag_eval.json 规范化产物:生成段聚合口径、计数不变量、构建、契约校验、原子发布。

规格:docs/superpowers/specs/2026-09-16-rag-eval-page-design.md(生成指标口径一节)。
"""

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
ANSWERABLE_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
GENERATION_COUNTERS = ("answerable_total", "answered", "answerable_refused", "judged",
                       "faithful", "fabricated", "judge_errors", "d_total", "d_refused",
                       "degraded")
REPORT_FILENAME = "rag_eval.json"


def _bucket_agg(bp: list[dict]) -> dict:
    judged = [r for r in bp if r["judge"] in ("faithful", "fabricated")]
    has_error = any(r["judge"] == "judge_error" for r in bp)
    faithful = sum(1 for r in judged if r["judge"] == "faithful")
    return {
        "total": len(bp),
        "answered": sum(1 for r in bp if not r["refused"]),
        "refused": sum(1 for r in bp if r["refused"]),
        "judged": len(judged),
        "faithful": faithful,
        "fabricated": len(judged) - faithful,
        "judge_errors": sum(1 for r in bp if r["judge"] == "judge_error"),
        # 拒答 coverage 记 0,分母 = 桶全部 test 题;裁判失败 → null(不静默缩分母)
        "coverage": None if has_error else sum(r.get("coverage", 0.0) for r in judged) / len(bp),
        "faithful_rate": None if (has_error or not judged) else faithful / len(judged),
    }


def aggregate_generation(rows: list[dict]) -> dict:
    """单策略生成段聚合(test 分片)。rows 见 _run_generation 行结构。"""
    answerable = [r for r in rows if not r["should_refuse"]]
    d_rows = [r for r in rows if r["should_refuse"]]
    judged = [r for r in answerable if r["judge"] in ("faithful", "fabricated")]
    errors = [r for r in answerable if r["judge"] == "judge_error"]
    faithful = sum(1 for r in judged if r["judge"] == "faithful")
    has_error = bool(errors)
    return {
        "answerable_total": len(answerable),
        "answered": sum(1 for r in answerable if not r["refused"]),
        "answerable_refused": sum(1 for r in answerable if r["refused"]),
        "judged": len(judged),
        "faithful": faithful,
        "fabricated": len(judged) - faithful,
        "judge_errors": len(errors),
        "answer_coverage": (None if has_error or not answerable
                            else sum(r.get("coverage", 0.0) for r in judged) / len(answerable)),
        "faithful_rate": None if (has_error or not judged) else faithful / len(judged),
        "d_total": len(d_rows),
        "d_refused": sum(1 for r in d_rows if r["refused"]),
        "d_refuse_rate": (sum(1 for r in d_rows if r["refused"]) / len(d_rows)
                          if d_rows else None),
        "degraded": sum(1 for r in rows if r["degraded"]),
        "by_bucket": {b: _bucket_agg([r for r in answerable if r["bucket"] == b])
                      for b in ANSWERABLE_BUCKETS
                      if any(r["bucket"] == b for r in answerable)},
    }


def check_generation_invariants(agg: dict) -> None:
    """发布前校验计数不变量;违反即 RuntimeError(本轮不发布)。"""
    ok = (agg["answered"] + agg["answerable_refused"] == agg["answerable_total"]
          and agg["judged"] + agg["judge_errors"] == agg["answered"]
          and agg["faithful"] + agg["fabricated"] == agg["judged"]
          and agg["d_refused"] <= agg["d_total"])
    if not ok:
        raise RuntimeError(f"生成段计数不变量违反: {agg}")
```

`evals/run_retrieval_compare.py`:

`_simulate_second_turn` 替换为:

```python
def _simulate_second_turn(model, query: str, evidence: list[dict],
                          effective_strategy: str) -> str:
    """模拟第二轮消息:System(SERVICE) + Human(query) + AIMessage(query_faq tool_call)
    + ToolMessage(query_faq 出参 JSON),同步 invoke 生成答案。
    effective_strategy 用该次检索的真实值(rerank 降级时为 hybrid),不再硬编码。"""
    payload = {"effective_strategy": effective_strategy, "low_confidence": False,
               "note": None, "evidence": evidence}
    call_id = "call_compare"
    messages = [
        SystemMessage(content=SERVICE_SYSTEM_PROMPT),
        HumanMessage(content=query),
        AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": query},
                                           "id": call_id, "type": "tool_call"}]),
        ToolMessage(content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=call_id, name="query_faq"),
    ]
    resp = model.invoke(messages)
    return resp.content if isinstance(resp.content, str) else ""
```

`_generation_segment` 整段(:249-288)删除,替换为:

```python
def _run_generation(test_cases: list[dict], strat: str, threshold: float | None,
                    settings, model) -> list[dict]:
    """单策略生成段(test 分片):低置信不调模型直接拒答话术;非拒答的 A/B/C/E case 交裁判。
    threshold=None 为 ungated 兜底:只有零命中才判低置信。
    行结构:{id,bucket,should_refuse,low_confidence,refused,answer,degraded,evidence,
    judge,unsupported_claims?,cited_refs?,coverage?,judge_error?}"""
    rows = []
    for c in test_cases:
        r = c["retrieval"][strat]
        low = r["top1"] is None or (threshold is not None and r["top1"] < threshold)
        if low:
            answer, evidence = REFUSAL_ANSWER, []
        else:
            evidence = [e.to_dict() for e in assemble_evidence(
                r["hits"], max_items=settings.rerank_top_n,
                budget_chars=settings.max_tool_result_chars, overhead_chars=200)]
            answer = _simulate_second_turn(model, c["query"], evidence,
                                           r["effective_strategy"])
        refused = answer.strip() == REFUSAL_ANSWER
        row = {"id": c["id"], "bucket": c["bucket"], "should_refuse": c["should_refuse"],
               "low_confidence": low, "refused": refused, "answer": answer,
               "degraded": r["effective_strategy"] != strat, "evidence": evidence,
               "judge": None}
        if not refused and not c["should_refuse"]:
            verdict, err = _judge(model, c["query"], answer, evidence, c["expect_points"])
            if verdict is None:
                row["judge"] = "judge_error"
                row["judge_error"] = err
            else:
                row["judge"] = verdict["verdict"]
                row["unsupported_claims"] = verdict["unsupported_claims"]
                row["cited_refs"] = verdict["cited_refs"]
                row["coverage"] = verdict["coverage"]
        rows.append(row)
    return rows


def _public_row(row: dict) -> dict:
    """写进 _compare.json 的行:去掉 evidence(体积大,台账与 rag_eval.json 另有快照)。"""
    return {k: v for k, v in row.items() if k != "evidence"}
```

`main()` 生成段调用处(:448-449)临时替换为(Task 5 再接四策略与台账事务):

```python
        gen_rows = _run_generation(test_cases, "hybrid_rerank",
                                   thresholds["hybrid_rerank"]["threshold"],
                                   settings, model)
        judge_errors = sum(1 for r in gen_rows if r["judge"] == "judge_error")
        for r in gen_rows:  # Task 5 改为单事务批量;此处保持逐 case 语义不变
            if r["judge"] == "fabricated":
                with main_sf() as s:
                    upsert_faith_case(s, run_id=run_id, eval_id=r["id"],
                                      bucket=r["bucket"],
                                      query=by_id[r["id"]]["query"], answer=r["answer"],
                                      reason=json.dumps(r["unsupported_claims"],
                                                        ensure_ascii=False),
                                      evidence=r["evidence"], cited_refs=r["cited_refs"],
                                      judge_model=settings.model_name)
                    s.commit()
```

`by_id` 在 cases 加载后(:397 之后)建一次:`by_id = {c["id"]: c for c in cases}`。

`out["generation"]["rows"]`(:487)改为 `[_public_row(r) for r in gen_rows]`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rag_eval_report.py tests/test_eval_metrics.py tests/test_faith_cases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/rag_eval_report.py evals/run_retrieval_compare.py tests/test_rag_eval_report.py tests/test_eval_metrics.py
git commit -m "feat(rag-eval): 生成段函数化(_run_generation 四策略可用)+聚合口径与计数不变量"
```

---

### Task 5: rag_eval.json 构建/校验/原子发布 + main 四策略接线 + 单事务台账

**Files:**
- Modify: `evals/rag_eval_report.py`(加 `new_run_id/build_rag_eval/validate_rag_eval/publish_rag_eval`)
- Modify: `evals/run_retrieval_compare.py`:`main`(:445-506)生成段之后全部重写;`_render_md`(:291-340)生成段小节更新
- Test: `tests/test_rag_eval_report.py`(追加)

**Interfaces:**
- Consumes: Task 4 的 `aggregate_generation/check_generation_invariants/_run_generation/_public_row`;Task 2 的 by_bucket 三键;Task 3 的 `upsert_faith_case` 与 run_id。
- Produces:
  - `new_run_id(now: datetime) -> str`(`"%Y%m%dT%H%M%S%fZ"`)
  - `build_rag_eval(*, run_id, ts, elapsed_s, settings, cases, corpus_chunks, thresholds, test_metrics, gen_agg, faithfulness_cases, gates) -> dict`
  - `validate_rag_eval(data: dict) -> None`(不合法 raise ValueError)
  - `publish_rag_eval(data: dict, results_dir) -> Path`(先校验,临时文件 + `os.replace`,失败清理并保留旧文件)
  - 固定报告路径:`evals/results/rag_eval.json`(Task 7/8/9 依赖)

- [ ] **Step 1: 写失败测试(追加到 tests/test_rag_eval_report.py)**

```python
import json
from pathlib import Path

from evals.rag_eval_report import (
    build_rag_eval, new_run_id, publish_rag_eval, validate_rag_eval,
)
from evals.run_retrieval_compare import STRATEGIES


class _S:  # 最小 settings
    embedding_model = "bge-m3"
    rerank_model = "bge-reranker-v2-m3"
    model_name = "m"


def _cases():
    return [{"id": "A1", "bucket": "A_policy", "split": "calibration"},
            {"id": "A2", "bucket": "A_policy", "split": "test"},
            {"id": "D2", "bucket": "D_absent", "split": "test"}]


def _metrics():
    tm = {}
    for s in STRATEGIES:
        tm[s] = {"threshold": 0.5, "section_recall_at_5": 0.1, "section_recall_at_10": 0.2,
                 "complete_hit_at_5": 0.3, "complete_hit_at_10": 0.4, "mrr_at_10": 0.5,
                 "d_refuse_correct_rate": 1.0, "over_refusal_rate": 0.0,
                 "by_bucket": {"A_policy": {"section_recall_at_10": 0.2, "mrr_at_10": 0.5,
                                            "recall5": 0.1, "evidence_coverage": 0.4,
                                            "refused": 0}}}
    return tm


def _gen_agg():
    base = aggregate_generation([
        _row("A2", "A_policy", judge="faithful", coverage=1.0),
        _row("D2", "D_absent", refused=True, should_refuse=True),
    ])
    return {s: dict(base) for s in STRATEGIES}


def _build():
    return build_rag_eval(
        run_id="r1", ts="2026-09-16T00:00:00+00:00", elapsed_s=1.5, settings=_S(),
        cases=_cases(), corpus_chunks=50,
        thresholds={s: {"threshold": 0.5} for s in STRATEGIES},
        test_metrics=_metrics(), gen_agg=_gen_agg(),
        faithfulness_cases=[], gates={"judge_error_zero": True,
                                      "bm25_b_model_section_recall_at_10": 1.0,
                                      "bm25_b_model_gate": True, "passed": True})


def test_build_rag_eval_shape():
    rep = _build()
    assert set(rep) == {"meta", "retrieval", "generation", "gates"}
    m = rep["meta"]
    assert m["total_cases"] == 3 and m["calibration_cases"] == 1 and m["test_cases"] == 2
    assert m["answerable_test_cases"] == 1 and m["d_test_cases"] == 1
    assert m["corpus_chunks"] == 50 and m["embedding_model"] == "bge-m3"
    dense = rep["retrieval"]["dense"]
    assert dense["overall"] == {"mrr": 0.5, "recall5": 0.1,
                                "evidence_coverage": 0.4, "sr10": 0.2}
    assert dense["by_bucket"]["A_policy"] == {"mrr": 0.5, "recall5": 0.1,
                                              "evidence_coverage": 0.4}
    assert rep["generation"]["online_strategy"] == "hybrid_rerank"
    assert rep["generation"]["dense"]["answer_coverage"] == 1.0
    validate_rag_eval(rep)  # 不抛即过


def test_validate_rejects_bad():
    import pytest
    with pytest.raises(ValueError):
        validate_rag_eval({"meta": {}})
    bad = _build()
    bad["generation"]["dense"]["answered"] = 99   # 破坏不变量
    with pytest.raises(ValueError):
        validate_rag_eval(bad)


def test_publish_atomic(tmp_path):
    old = tmp_path / "rag_eval.json"
    old.write_text('{"old": true}', encoding="utf-8")
    path = publish_rag_eval(_build(), tmp_path)
    assert path == old
    assert json.loads(old.read_text(encoding="utf-8"))["meta"]["run_id"] == "r1"
    assert not list(tmp_path.glob(".rag_eval-*"))

    # 校验失败:旧文件字节不变,不留临时文件
    import pytest
    bad = _build()
    bad["generation"]["dense"]["fabricated"] = 99
    with pytest.raises(ValueError):
        publish_rag_eval(bad, tmp_path)
    assert json.loads(old.read_text(encoding="utf-8"))["meta"]["run_id"] == "r1"
    assert not list(tmp_path.glob(".rag_eval-*"))


def test_new_run_id_unique():
    from datetime import datetime, timezone
    a = new_run_id(datetime.now(timezone.utc))
    b = new_run_id(datetime.now(timezone.utc))
    assert a != b and len(a) == len("20260916T024957123456Z")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rag_eval_report.py -v`
Expected: FAIL(ImportError: cannot import name 'build_rag_eval')

- [ ] **Step 3: 实现**

`evals/rag_eval_report.py` 顶部 import 加 `import json, os, tempfile`;追加:

```python
def new_run_id(now) -> str:
    """含微秒的 UTC 紧凑时间戳;连续运行不重复。"""
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def build_rag_eval(*, run_id, ts, elapsed_s, settings, cases, corpus_chunks,
                   thresholds, test_metrics, gen_agg, faithfulness_cases, gates) -> dict:
    test = [c for c in cases if c["split"] == "test"]
    retrieval = {}
    for s in STRATEGIES:
        m = test_metrics[s]
        retrieval[s] = {
            "threshold": thresholds[s]["threshold"],
            "ungated": bool(thresholds[s].get("ungated")),
            "overall": {"mrr": m["mrr_at_10"], "recall5": m["section_recall_at_5"],
                        "evidence_coverage": m["complete_hit_at_10"],
                        "sr10": m["section_recall_at_10"]},
            "by_bucket": {b: {"mrr": bb["mrr_at_10"], "recall5": bb["recall5"],
                              "evidence_coverage": bb["evidence_coverage"]}
                          for b, bb in m["by_bucket"].items()},
        }
    return {
        "meta": {"run_id": run_id, "ts": ts, "elapsed_s": round(elapsed_s, 1),
                 "total_cases": len(cases),
                 "calibration_cases": len(cases) - len(test),
                 "test_cases": len(test),
                 "answerable_test_cases": sum(1 for c in test
                                              if c["bucket"] in ANSWERABLE_BUCKETS),
                 "d_test_cases": sum(1 for c in test if c["bucket"] == "D_absent"),
                 "corpus_chunks": corpus_chunks,
                 "embedding_model": settings.embedding_model,
                 "rerank_model": settings.rerank_model,
                 "eval_model": settings.model_name,
                 "judge_model": settings.model_name},
        "retrieval": retrieval,
        "generation": {"online_strategy": "hybrid_rerank",
                       **{s: gen_agg[s] for s in STRATEGIES},
                       "faithfulness_cases": faithfulness_cases},
        "gates": gates,
    }


def validate_rag_eval(data) -> None:
    """发布前契约校验;不合法即 ValueError。"""
    if not isinstance(data, dict) or any(
            k not in data for k in ("meta", "retrieval", "generation", "gates")):
        raise ValueError("rag_eval 缺顶层键 meta/retrieval/generation/gates")
    meta = data["meta"]
    for k in ("run_id", "ts", "total_cases", "calibration_cases", "test_cases",
              "answerable_test_cases", "d_test_cases", "corpus_chunks",
              "embedding_model", "rerank_model", "eval_model", "judge_model"):
        if k not in meta:
            raise ValueError(f"rag_eval meta 缺 {k}")
    gen = data["generation"]
    if gen.get("online_strategy") != "hybrid_rerank":
        raise ValueError("online_strategy 非 hybrid_rerank")
    if not isinstance(gen.get("faithfulness_cases"), list):
        raise ValueError("faithfulness_cases 非列表")
    for s in STRATEGIES:
        r = data["retrieval"].get(s)
        if not isinstance(r, dict) or "overall" not in r or "by_bucket" not in r:
            raise ValueError(f"rag_eval retrieval 缺 {s}")
        g = gen.get(s)
        if not isinstance(g, dict) or any(k not in g for k in GENERATION_COUNTERS):
            raise ValueError(f"rag_eval generation 缺 {s} 计数")
        try:
            check_generation_invariants(g)
        except RuntimeError as exc:          # 不变量违反统一视为校验失败
            raise ValueError(str(exc)) from exc
```

再追加:

```python
def publish_rag_eval(data, results_dir) -> Path:
    """原子发布:先校验;临时文件写全后 os.replace;失败清理临时文件并保留旧报告。"""
    validate_rag_eval(data)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / REPORT_FILENAME
    fd, tmp = tempfile.mkstemp(dir=results_dir, prefix=".rag_eval-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
```

`evals/run_retrieval_compare.py` 顶部 import 加:

```python
from evals.rag_eval_report import (
    STRATEGIES as _REPORT_STRATEGIES,  # 与本模块 STRATEGIES 同值,断言钉住
    aggregate_generation, build_rag_eval, check_generation_invariants,
    new_run_id, publish_rag_eval, validate_rag_eval,
)
assert tuple(STRATEGIES) == tuple(_REPORT_STRATEGIES)
```

`main()` 从 `test_cases = ...`(:445)往下,生成段那一段替换为:

```python
        gen_rows = {s: _run_generation(test_cases, s, thresholds[s]["threshold"],
                                       settings, model)
                    for s in STRATEGIES}
        gen_agg = {s: aggregate_generation(gen_rows[s]) for s in STRATEGIES}
        for s in STRATEGIES:
            check_generation_invariants(gen_agg[s])
        judge_errors = sum(gen_agg[s]["judge_errors"] for s in STRATEGIES)  # 全策略合计
```

计时:`main` 的 import 区顶部加 `import time`;`main` 函数体第一行(`max_d_pass = _parse_max_d_pass(argv)` 之前)加 `_t0 = time.monotonic()`,后面 `build_rag_eval` 用 `elapsed_s=time.monotonic() - _t0`。

bm25 硬门槛与 gates 段(:450-459)保留,`judge_errors` 变量已是新合计。

`out` 的 generation 段(:482-488)替换为:

```python
            "generation": {
                # 既有字段保持 hybrid_rerank 口径,兼容旧读者;.md 同
                "strategy": "hybrid_rerank",
                "refused": sum(1 for r in gen_rows["hybrid_rerank"] if r["refused"]),
                "faithful": gen_agg["hybrid_rerank"]["faithful"],
                "fabricated": gen_agg["hybrid_rerank"]["fabricated"],
                "judge_errors": gen_agg["hybrid_rerank"]["judge_errors"],
                "rows": [_public_row(r) for r in gen_rows["hybrid_rerank"]],
                "per_strategy": {s: {**gen_agg[s],
                                     "rows": [_public_row(r) for r in gen_rows[s]]}
                                 for s in STRATEGIES},
            },
```

`out` 顶层已有 `"run_id": run_id`(Task 3 加入)。

报告文件写出段(:491-497)之后、return 之前,插入 rag_eval 规范化发布段:

```python
        # --- rag_eval.json:规范化报告(校验 → 台账单事务 → 原子发布,最后落盘)---
        faith_rows = [
            {"case_id": r["id"], "bucket": r["bucket"],
             "query": by_id[r["id"]]["query"], "answer": r["answer"],
             "strategy": "hybrid_rerank",
             "unsupported_claims": r["unsupported_claims"],
             "cited_refs": r["cited_refs"], "evidence": r["evidence"]}
            for r in gen_rows["hybrid_rerank"] if r["judge"] == "fabricated"
        ]
        report = build_rag_eval(
            run_id=run_id, ts=out["ts"], elapsed_s=time.monotonic() - _t0,
            settings=settings, cases=cases, corpus_chunks=len(chunk_meta),
            thresholds=thresholds, test_metrics=metrics, gen_agg=gen_agg,
            faithfulness_cases=faith_rows, gates=gates)
        validate_rag_eval(report)          # 校验通过才动台账
        if faith_rows:                     # 单事务批量 upsert;失败 → 不发布本轮
            with main_sf() as s:
                for fr in faith_rows:
                    upsert_faith_case(s, run_id=run_id, eval_id=fr["case_id"],
                                      bucket=fr["bucket"], query=fr["query"],
                                      answer=fr["answer"],
                                      reason=json.dumps(fr["unsupported_claims"],
                                                        ensure_ascii=False),
                                      evidence=fr["evidence"],
                                      cited_refs=fr["cited_refs"],
                                      judge_model=settings.model_name)
                s.commit()
        rag_eval_path = publish_rag_eval(report, RESULTS_DIR)
        print(f"[compare] rag_eval 报告: {rag_eval_path}")
```

同时删除 Task 4 过渡期在生成段之后加的逐 case upsert 循环(被上面的单事务替代)。

`main` 里 Task 3 手写的 `run_id = datetime.now(...)` 一行改为 `run_id = new_run_id(datetime.now(timezone.utc))`。

`_render_md`:在 `flagged` 段之后、硬门槛段之前插入四策略小节:

```python
    lines += ["", "## 生成段四策略覆盖/忠实(test 分片;null = 有裁判失败,不可用)", "",
              "| strategy | answer_coverage | faithful_rate | answered | judged | "
              "judge_error | D拒答率 |", "|---|---|---|---|---|---|---|"]
    for strat in STRATEGIES:
        a = out["generation"]["per_strategy"][strat]

        def _fmt(v):
            return "不可用" if v is None else f"{v:.3f}"

        lines.append(f"| {strat} | {_fmt(a['answer_coverage'])} | "
                     f"{_fmt(a['faithful_rate'])} | "
                     f"{a['answered']}/{a['answerable_total']} | {a['judged']} | "
                     f"{a['judge_errors']} | {_fmt(a['d_refuse_rate'])} |")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rag_eval_report.py tests/test_eval_metrics.py tests/test_faith_cases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/rag_eval_report.py evals/run_retrieval_compare.py tests/test_rag_eval_report.py
git commit -m "feat(rag-eval): 四策略生成段接线 + rag_eval.json 原子发布 + 台账单事务批量 upsert"
```

---

### Task 6: 作业层 JobRunner

**Files:**
- Create: `app/jobs/runner.py`
- Test: `tests/test_job_runner.py`(新建)

**Interfaces:**
- Produces:
  - `JobRunner(log_dir: Path, report_loader: callable | None = None, cwd: Path | None = None)`;`report_loader()` 返回已校验报告 dict 或 None(缺/坏)。
  - `.register(name: str, cmd: list[str]) -> None`;`.names() -> tuple[str, ...]`
  - `await .run(name) -> bool`(运行中 → False;未知名 → KeyError)
  - `.status(name) -> dict`:`{name, status(idle|running|ok|failed), started_at, finished_at, exit_code, quality_passed, report_run_id, log_tail}`(未知名 → KeyError)
  - `await .close() -> None`(运行中的子进程 terminate → 5s 超时 → kill)
  - 成败判定:子进程结束后读报告,`meta.run_id` 与启动前不同 → `ok`(quality_passed = gates.passed);否则 `failed`。无 report_loader 时退化为 exit_code==0 → ok。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_job_runner.py`:

```python
"""JobRunner:状态机 / 日志捕获 / 单并发 / 报告驱动的成败判定 / close 回收。"""

import asyncio
import json
import sys

import pytest

from app.jobs.runner import JobRunner


def _report(run_id, passed=True):
    return {"meta": {"run_id": run_id}, "gates": {"passed": passed}}


async def test_run_ok_and_log(tmp_path):
    r = JobRunner(tmp_path)                 # 无 loader:exit 0 → ok
    r.register("t", [sys.executable, "-c", "print('hello-eval')"])
    assert await r.run("t") is True
    for _ in range(100):
        await asyncio.sleep(0.02)
        if r.status("t")["status"] != "running":
            break
    st = r.status("t")
    assert st["status"] == "ok" and st["exit_code"] == 0
    assert "hello-eval" in st["log_tail"]
    assert st["started_at"] and st["finished_at"]
    with pytest.raises(KeyError):
        r.status("nope")


async def test_conflict_while_running(tmp_path):
    r = JobRunner(tmp_path)
    r.register("t", [sys.executable, "-c", "import time; time.sleep(1.0)"])
    assert await r.run("t") is True
    assert await r.run("t") is False          # 运行中重复触发
    for _ in range(200):
        await asyncio.sleep(0.02)
        if r.status("t")["status"] != "running":
            break
    assert r.status("t")["status"] == "ok"    # 无 loader:exit 0 → ok


async def test_report_drives_verdict(tmp_path):
    """exit 1 但产出新 run_id 报告 → ok + quality_passed=false;无新报告 → failed。"""
    rp = tmp_path / "rag_eval.json"
    rp.write_text(json.dumps(_report("old")), encoding="utf-8")
    loader = lambda: json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    r = JobRunner(tmp_path, report_loader=loader)
    write_and_fail = (
        "import json,sys,pathlib;"
        f"pathlib.Path({str(rp)!r}).write_text(json.dumps({_report('new', False)!r}));"
        "sys.exit(1)")
    r.register("t", [sys.executable, "-c", write_and_fail])
    await r.run("t")
    for _ in range(200):
        await asyncio.sleep(0.02)
        if r.status("t")["status"] != "running":
            break
    st = r.status("t")
    assert st["status"] == "ok" and st["quality_passed"] is False
    assert st["exit_code"] == 1 and st["report_run_id"] == "new"

    r2 = JobRunner(tmp_path, report_loader=loader)  # 不更新报告 → run_id 不变 → failed
    r2.register("t", [sys.executable, "-c", "import sys; sys.exit(1)"])
    await r2.run("t")
    for _ in range(200):
        await asyncio.sleep(0.02)
        if r2.status("t")["status"] != "running":
            break
    st2 = r2.status("t")
    assert st2["status"] == "failed" and st2["quality_passed"] is None


async def test_close_terminates(tmp_path):
    r = JobRunner(tmp_path)
    r.register("t", [sys.executable, "-c", "import time; time.sleep(60)"])
    await r.run("t")
    await asyncio.sleep(0.2)
    await r.close()
    assert r.status("t")["status"] == "failed"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_job_runner.py -v`
Expected: FAIL(ModuleNotFoundError: app.jobs.runner)

- [ ] **Step 3: 实现**

新建 `app/jobs/runner.py`:

```python
"""应用内后台作业:白名单命令 + asyncio 子进程 + 日志文件 + 报告驱动的成败判定。

单并发(同一把锁完成「检查+登记」);内存态重启即丢,日志文件持久。
成败不看 exit_code:子进程结束后读固定报告,meta.run_id 更新即 ok
(脚本因质量门槛退出 1 也属正常完成),quality_passed 取 gates.passed。
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOG_TAIL_BYTES = 8192
_CLOSE_TIMEOUT_S = 5


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobInfo:
    name: str
    status: str = "idle"            # idle|running|ok|failed
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    quality_passed: bool | None = None
    report_run_id: str | None = None
    error: str | None = None
    proc: object = field(default=None, repr=False)  # asyncio.subprocess.Process


class JobRunner:
    def __init__(self, log_dir, report_loader=None, cwd=None):
        self._log_dir = Path(log_dir)
        self._report_loader = report_loader  # callable() -> dict|None(已校验报告)
        self._cwd = str(cwd) if cwd else None
        self._cmds: dict[str, list[str]] = {}
        self._jobs: dict[str, JobInfo] = {}
        self._lock = asyncio.Lock()

    def register(self, name: str, cmd: list[str]) -> None:
        self._cmds[name] = list(cmd)

    def names(self) -> tuple[str, ...]:
        return tuple(self._cmds)

    def _current_run_id(self):
        if self._report_loader is None:
            return None
        try:
            report = self._report_loader() or {}
        except Exception:
            return None
        return (report.get("meta") or {}).get("run_id")

    async def run(self, name: str) -> bool:
        if name not in self._cmds:
            raise KeyError(name)
        async with self._lock:  # 检查+登记原子化:两个并发 POST 只放一个
            job = self._jobs.get(name)
            if job is not None and job.status == "running":
                return False
            job = JobInfo(name=name, status="running", started_at=_utcnow())
            self._jobs[name] = job
        asyncio.create_task(self._exec(job))
        return True

    async def _exec(self, job: JobInfo) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{job.name}.log"
        prev_run_id = self._current_run_id()
        try:
            with open(log_path, "wb") as log:
                job.proc = await asyncio.create_subprocess_exec(
                    *self._cmds[job.name], stdout=log,
                    stderr=asyncio.subprocess.STDOUT, cwd=self._cwd)
                job.exit_code = await job.proc.wait()
        except Exception as exc:  # 启动失败(uv 不存在等)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            with open(log_path, "ab") as log:
                log.write(("\n[runner] 启动失败: " + job.error + "\n").encode())
        else:
            report = None
            if self._report_loader is not None:
                try:
                    report = self._report_loader()
                except Exception:
                    report = None
            run_id = (report or {}).get("meta", {}).get("run_id")
            if self._report_loader is None:
                job.status = "ok" if job.exit_code == 0 else "failed"
            elif run_id and run_id != prev_run_id:
                job.status = "ok"
                gates = report.get("gates") or {}
                job.quality_passed = bool(gates.get("passed"))
                job.report_run_id = run_id
            else:
                job.status = "failed"
        finally:
            job.finished_at = _utcnow()
            job.proc = None

    def status(self, name: str) -> dict:
        if name not in self._cmds:
            raise KeyError(name)
        job = self._jobs.get(name) or JobInfo(name=name)
        tail = ""
        log_path = self._log_dir / f"{name}.log"
        if log_path.exists():
            tail = log_path.read_bytes()[-LOG_TAIL_BYTES:].decode("utf-8", errors="replace")
        return {"name": name, "status": job.status, "started_at": job.started_at,
                "finished_at": job.finished_at, "exit_code": job.exit_code,
                "quality_passed": job.quality_passed, "report_run_id": job.report_run_id,
                "error": job.error, "log_tail": tail}

    async def close(self) -> None:
        for job in self._jobs.values():
            proc = job.proc
            if job.status == "running" and proc is not None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_CLOSE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                job.status = "failed"
                job.error = "server shutdown"
                job.finished_at = _utcnow()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_job_runner.py -v`
Expected: PASS(4 个用例)

- [ ] **Step 5: Commit**

```bash
git add app/jobs/runner.py tests/test_job_runner.py
git commit -m "feat(rag-eval): JobRunner(asyncio 子进程+日志尾+报告驱动成败判定+close 回收)"
```

---

### Task 7: /api/jobs 路由 + create_app 装配

**Files:**
- Create: `app/routers/jobs.py`
- Modify: `app/main.py`(注册路由;`app.state.job_runner`、`app.state.rag_eval_report_path`;lifespan 关闭时 `await job_runner.close()`)
- Test: `tests/test_jobs_api.py`(新建)

**Interfaces:**
- Consumes: Task 6 `JobRunner`;Task 8 的 `app.services.rag_eval.load_report`(装配报告 loader 用,执行顺序保证 Task 8 先落地)。
- Produces: `POST /api/jobs/{name}/run` → 200 `{started: true}` / 409 `job_running` / 404 `job_unknown`;`GET /api/jobs/{name}` → runner.status 字典 / 404。`app.state.job_runner` 与 `app.state.rag_eval_report_path`(Task 9、页面依赖)。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_jobs_api.py`:

```python
"""/api/jobs 路由:白名单作业名、409 并发、报告驱动 ok/failed。"""

import asyncio
import json
import sys

import httpx
import pytest

from app.jobs.runner import JobRunner
from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings


def _report(run_id, passed=True):
    return {"meta": {"run_id": run_id}, "gates": {"passed": passed}}


def _make_app(tmp_path, cmd, prev_report=None):
    """装测试 runner:loader 读 tmp 目录报告文件;cmd 模拟评估(自己写新报告)。"""
    rp = tmp_path / "rag_eval.json"
    if prev_report is not None:
        rp.write_text(json.dumps(prev_report), encoding="utf-8")
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))

    def _loader():
        if not rp.exists():
            return None
        return json.loads(rp.read_text(encoding="utf-8"))

    runner = JobRunner(tmp_path, report_loader=_loader)
    runner.register("eval-rag", cmd)
    app.state.job_runner = runner            # 替换默认 runner,绝不真跑评估
    app.state.rag_eval_report_path = rp
    return app


async def _wait_terminal(app, name, timeout=5.0):
    runner = app.state.job_runner
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        if runner.status(name)["status"] != "running":
            break
    return runner.status(name)


def _write_report_cmd(tmp_path, report, exit_code=0):
    """模拟评估脚本的命令:写新报告 + 打日志 + 按 exit_code 退出。"""
    rp = tmp_path / "rag_eval.json"
    return [sys.executable, "-c",
            "import json,sys,pathlib;"
            f"pathlib.Path({str(rp)!r}).write_text(json.dumps({report!r}));"
            f"print('done');sys.exit({exit_code})"]


async def test_run_and_status(tmp_path):
    app = _make_app(tmp_path, _write_report_cmd(tmp_path, _report("r1")))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/api/jobs/eval-rag")
        assert resp.status_code == 200 and resp.json()["status"] == "idle"
        assert resp.json()["exit_code"] is None
        resp = await client.post("/api/jobs/eval-rag/run")
        assert resp.status_code == 200 and resp.json() == {"started": True}
        st = await _wait_terminal(app, "eval-rag")
        assert st["status"] == "ok" and "done" in st["log_tail"]
        resp = await client.get("/api/jobs/eval-rag")
        body = resp.json()
        assert body["status"] == "ok" and body["exit_code"] == 0
        assert body["quality_passed"] is True and body["report_run_id"] == "r1"
        assert "done" in body["log_tail"]


async def test_unknown_job_404(tmp_path):
    app = _make_app(tmp_path, [sys.executable, "-c", "pass"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.get("/api/jobs/nope")).status_code == 404
        assert (await client.post("/api/jobs/nope/run")).status_code == 404


async def test_conflict_409(tmp_path):
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    runner = JobRunner(tmp_path)
    runner.register("eval-rag",
                    [sys.executable, "-c", "import time; time.sleep(1.0)"])
    app.state.job_runner = runner
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.post("/api/jobs/eval-rag/run")).status_code == 200
        resp = await client.post("/api/jobs/eval-rag/run")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "job_running"
    await _wait_terminal(app, "eval-rag")


async def test_quality_failed_but_ok(tmp_path):
    """新报告 + gates.passed=false + exit 1:status=ok,quality_passed=false(报告驱动)。"""
    cmd = _write_report_cmd(tmp_path, _report("r-new", passed=False), exit_code=1)
    app = _make_app(tmp_path, cmd, prev_report=_report("r-old"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/api/jobs/eval-rag/run")
        st = await _wait_terminal(app, "eval-rag")
    assert st["status"] == "ok" and st["quality_passed"] is False
    assert st["exit_code"] == 1 and st["report_run_id"] == "r-new"


async def test_no_new_report_is_failed(tmp_path):
    """脚本没产出新报告(exit 1)→ failed,quality_passed=null,旧报告不动。"""
    app = _make_app(tmp_path, [sys.executable, "-c", "import sys; sys.exit(1)"],
                    prev_report=_report("r-old"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/api/jobs/eval-rag/run")
        st = await _wait_terminal(app, "eval-rag")
    assert st["status"] == "failed" and st["quality_passed"] is None
    assert st["report_run_id"] is None
```

注:`_make_app` 里 `create_app` 会构造默认 `JobRunner`(纯内存,无副作用),测试随即替换;报告由模拟命令自己写入,与真实评估脚本行为同构,无竞态。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_jobs_api.py -v`
Expected: FAIL(404 — 路由不存在)

- [ ] **Step 3: 实现**

新建 `app/routers/jobs.py`:

```python
"""后台作业 API(/api/jobs/*):白名单作业名;运行中重复触发 409。"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/jobs")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code,
                                                               "message": message}})


@router.post("/{name}/run")
async def job_run(name: str, request: Request):
    runner = request.app.state.job_runner
    if name not in runner.names():
        return _err(404, "job_unknown", "未知作业")
    if not await runner.run(name):
        return _err(409, "job_running", "作业正在运行")
    return {"started": True}


@router.get("/{name}")
async def job_status(name: str, request: Request):
    runner = request.app.state.job_runner
    if name not in runner.names():
        return _err(404, "job_unknown", "未知作业")
    return runner.status(name)
```

`app/main.py`:

import 区加:

```python
from app.jobs.runner import JobRunner
from app.routers.jobs import router as jobs_router
from app.services import rag_eval as rag_eval_service
```

`create_app` 内,`app.include_router(kb_router)` 之后加:

```python
    app.include_router(jobs_router)

    root_dir = Path(__file__).resolve().parent.parent
    app.state.rag_eval_report_path = root_dir / "evals" / "results" / "rag_eval.json"

    def _report_loader():
        try:
            return rag_eval_service.load_report(app.state.rag_eval_report_path)
        except Exception:
            return None

    job_runner = JobRunner(log_dir=root_dir / "data" / "jobs",
                           report_loader=_report_loader, cwd=root_dir)
    job_runner.register("eval-rag",
                        ["uv", "run", "python", "evals/run_retrieval_compare.py"])
    app.state.job_runner = job_runner
```

lifespan 改为:

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.job_runner.close()   # 先收评估子进程
        if owns_runtime and runtime.retriever is not None:
            runtime.retriever.close()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_jobs_api.py -v`(Task 8 未落地时 `app.services.rag_eval` 不存在,先建同名空模块+`load_report` 桩,或直接按序先做 Task 8 再回测)
Expected: PASS(4 个用例)

- [ ] **Step 5: Commit**

```bash
git add app/routers/jobs.py app/main.py tests/test_jobs_api.py
git commit -m "feat(rag-eval): /api/jobs 路由 + create_app 装配 JobRunner(eval-rag 白名单,lifespan 回收)"
```

---

### Task 8: rag_eval 数据服务(报告加载/台账查询/两口径统计/处置)

**Files:**
- Create: `app/services/rag_eval.py`
- Test: `tests/test_rag_eval_service.py`(新建;DB 用例走 dbfixtures)

**Interfaces:**
- Consumes: Task 5 的 rag_eval.json 契约;Task 3 的 citations 新形状;`app.models.FaithCase`。
- Produces:
  - `ReportCorruptError(Exception)`
  - `load_report(path) -> dict | None`(缺 → None;坏 → ReportCorruptError)
  - `validate_report(data) -> None`
  - `normalize_citations(raw) -> {"run_id", "evidence", "cited_refs"}`(兼容裸 list / None)
  - `list_faith_cases(sf, status, page, size) -> {"total", "page", "size", "items"}`;item 键:`id, case_id, bucket, query, strategy, answer, unsupported_claims, citations, judge_model, status, seen_count, first_seen_at, last_seen_at, resolution, resolved_at`
  - `compute_stats(report: dict | None, sf) -> {"judge_rate", "confirmed_rate", "current_run_missing", "ledger": {"total", "未解决", "已解决", "无需解决"}}`
  - `update_faith_case(sf, row_id, status, resolution) -> dict | None`(None = 不存在)

- [ ] **Step 1: 写失败测试**

新建 `tests/test_rag_eval_service.py`:

```python
"""rag_eval 数据服务:报告加载/契约校验、citations 兼容、台账分页与统计、处置。"""

import json
from datetime import datetime

import pytest

from app.models import FaithCase
from app.services import rag_eval as svc
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


def _report(run_id="r1", fabricated=1, judged=4, judge_errors=0, case_ids=("A2",)):
    return {
        "meta": {"run_id": run_id, "ts": "t", "total_cases": 300,
                 "calibration_cases": 150, "test_cases": 150,
                 "answerable_test_cases": 120, "d_test_cases": 30,
                 "corpus_chunks": 50, "embedding_model": "e", "rerank_model": "r",
                 "eval_model": "m", "judge_model": "m"},
        "retrieval": {s: {"threshold": 0.5, "ungated": False,
                          "overall": {"mrr": 0.1, "recall5": 0.1,
                                      "evidence_coverage": 0.1, "sr10": 0.1},
                          "by_bucket": {}}
                      for s in ("dense", "bm25", "hybrid", "hybrid_rerank")},
        "generation": {"online_strategy": "hybrid_rerank",
                       "hybrid_rerank": {"fabricated": fabricated, "judged": judged,
                                          "judge_errors": judge_errors},
                       "faithfulness_cases": [{"case_id": c} for c in case_ids]},
        "gates": {"passed": True},
    }


def _seed(s, case_id, run_id="r1", status="未解决", last=None):
    row = FaithCase(eval_id=case_id, bucket="A_policy", query="q", answer="a",
                    reason='[{"claim": "c", "reason": "r"}]',
                    citations={"run_id": run_id, "evidence": [{"ref_no": 1}],
                               "cited_refs": [1]},
                    judge_model="m", status=status)
    if last:
        row.last_seen_at = last
    s.add(row)
    return row


def test_normalize_citations_legacy():
    assert svc.normalize_citations(None) == {"run_id": None, "evidence": [],
                                             "cited_refs": []}
    out = svc.normalize_citations([{"ref_no": 1}])
    assert out == {"run_id": None, "evidence": [{"ref_no": 1}], "cited_refs": []}
    out = svc.normalize_citations({"run_id": "r1", "evidence": [], "cited_refs": [2]})
    assert out["run_id"] == "r1" and out["cited_refs"] == [2]


def test_load_report_missing_corrupt_ok(tmp_path):
    assert svc.load_report(tmp_path / "none.json") is None
    bad = tmp_path / "rag_eval.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(svc.ReportCorruptError):
        svc.load_report(bad)
    bad.write_text('{"meta": {}}', encoding="utf-8")       # 契约不合法
    with pytest.raises(svc.ReportCorruptError):
        svc.load_report(bad)
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(_report()), encoding="utf-8")
    assert svc.load_report(good)["meta"]["run_id"] == "r1"


def test_list_pagination_and_order(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        same = datetime(2026, 9, 16, 10, 0, 0)
        for cid in ("A1", "A2", "A3"):
            _seed(s, cid, last=same)                       # 同刻 → id DESC 定序
        _seed(s, "B1", status="已解决")
        s.commit()
    out = svc.list_faith_cases(sf, "未解决", page=1, size=2)
    assert out["total"] == 3 and len(out["items"]) == 2
    assert [i["case_id"] for i in out["items"]] == ["A3", "A2"]
    page2 = svc.list_faith_cases(sf, "未解决", page=2, size=2)
    assert [i["case_id"] for i in page2["items"]] == ["A1"]
    item = out["items"][0]
    assert isinstance(item["id"], int) and item["case_id"] == "A3"
    assert item["unsupported_claims"] == [{"claim": "c", "reason": "r"}]
    assert item["citations"]["run_id"] == "r1"
    assert svc.list_faith_cases(sf, "已解决", 1, 10)["total"] == 1


def test_compute_stats(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2", status="已解决")                    # 本轮判出且已确认
        _seed(s, "C2", run_id="r-old")                     # 旧轮快照,不算本轮
        s.commit()
    rep = _report(run_id="r1", fabricated=2, judged=4, case_ids=("A2", "B2"))
    stats = svc.compute_stats(rep, sf)
    assert stats["judge_rate"] == 0.5                      # fabricated / judged
    assert stats["current_run_missing"] == 1               # B2 无本轮台账快照
    assert stats["confirmed_rate"] is None                 # 缺失 → 不输出残缺口径
    assert stats["ledger"]["total"] == 2 and stats["ledger"]["已解决"] == 1

    with sf() as s:                                        # 补齐 B2 本轮快照(未解决)
        _seed(s, "B2", run_id="r1")
        s.commit()
    stats = svc.compute_stats(rep, sf)
    assert stats["current_run_missing"] == 0
    assert stats["confirmed_rate"] == 0.5                  # 2 题中 1 题已解决

    rep_err = _report(run_id="r1", judge_errors=1)
    assert svc.compute_stats(rep_err, sf)["judge_rate"] is None

    stats = svc.compute_stats(None, sf)                    # 无报告:两口径 null,台账照给
    assert stats["judge_rate"] is None and stats["confirmed_rate"] is None
    assert stats["current_run_missing"] == 0 and stats["ledger"]["total"] == 3


def test_update_faith_case(db_session_factory):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2")
        s.commit()
        rid = s.query(FaithCase).one().id
    item = svc.update_faith_case(sf, rid, "已解决", "  已补售后文档  ")
    assert item["status"] == "已解决" and item["resolution"] == "已补售后文档"
    assert item["resolved_at"] is not None
    with sf() as s:
        row = s.query(FaithCase).one()
        assert row.resolved_at.tzinfo is None              # 朴素 UTC(与既有列一致)
    assert svc.update_faith_case(sf, 999999, "已解决", "x") is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rag_eval_service.py -v`(需 docker compose 在线)
Expected: FAIL(ModuleNotFoundError: app.services.rag_eval)

- [ ] **Step 3: 实现**

新建 `app/services/rag_eval.py`:

```python
"""/rag-eval 页面数据服务:固定报告加载/契约校验、faith_cases 台账查询与处置、两口径统计。

report 与 stats 共用 load_report(契约校验同一处);报告损坏时 faith-cases 降级为
「无当前报告」口径,台账处置不被报告文件阻断。
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func

from app.models import FaithCase

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
STATUSES = ("未解决", "已解决", "无需解决")


class ReportCorruptError(Exception):
    """rag_eval.json 存在但解析/契约不合法。"""


def validate_report(data) -> None:
    if not isinstance(data, dict) or any(
            k not in data for k in ("meta", "retrieval", "generation", "gates")):
        raise ReportCorruptError("缺顶层键 meta/retrieval/generation/gates")
    meta = data["meta"]
    if not isinstance(meta, dict) or any(
            k not in meta for k in ("run_id", "ts", "total_cases", "test_cases",
                                    "embedding_model", "judge_model")):
        raise ReportCorruptError("meta 字段不完整")
    gen = data["generation"]
    if (not isinstance(gen, dict) or "online_strategy" not in gen
            or not isinstance(gen.get("faithfulness_cases"), list)):
        raise ReportCorruptError("generation 字段不完整")
    ret = data["retrieval"]
    if not isinstance(ret, dict) or any(s not in ret for s in STRATEGIES):
        raise ReportCorruptError("retrieval 缺策略")


def load_report(path) -> dict | None:
    """不存在 → None;损坏 → ReportCorruptError;合法 → dict。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportCorruptError(f"报告解析失败: {type(exc).__name__}") from exc
    validate_report(data)
    return data


def normalize_citations(raw) -> dict:
    """历史兼容:裸列表 → {run_id: None, evidence: raw, cited_refs: []};None → 空。"""
    if isinstance(raw, dict):
        return {"run_id": raw.get("run_id"),
                "evidence": raw.get("evidence") or [],
                "cited_refs": raw.get("cited_refs") or []}
    if isinstance(raw, list):
        return {"run_id": None, "evidence": raw, "cited_refs": []}
    return {"run_id": None, "evidence": [], "cited_refs": []}


def _parse_reason(raw):
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


def _item(row: FaithCase) -> dict:
    return {"id": row.id, "case_id": row.eval_id, "bucket": row.bucket,
            "query": row.query, "strategy": row.strategy, "answer": row.answer,
            "unsupported_claims": _parse_reason(row.reason),
            "citations": normalize_citations(row.citations),
            "judge_model": row.judge_model, "status": row.status,
            "seen_count": row.seen_count,
            "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
            "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
            "resolution": row.resolution,
            "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None}


def list_faith_cases(sf, status: str, page: int, size: int) -> dict:
    with sf() as s:
        q = s.query(FaithCase).filter_by(status=status)
        total = q.count()
        rows = (q.order_by(FaithCase.last_seen_at.desc(), FaithCase.id.desc())
                 .offset((page - 1) * size).limit(size).all())
        return {"total": total, "page": page, "size": size,
                "items": [_item(r) for r in rows]}


def ledger_counts(sf) -> dict:
    with sf() as s:
        rows = s.query(FaithCase.status, func.count()).group_by(FaithCase.status).all()
    counts = {st: 0 for st in STATUSES}
    for status, n in rows:
        counts[status] = counts.get(status, 0) + n
    return {"total": sum(counts.values()), **counts}


def compute_stats(report: dict | None, sf) -> dict:
    """judge_rate = 上线 fabricated/judged(有 judge_error 或 judged=0 → null);
    confirmed_rate 分子只数本轮报告判出且台账快照 run_id 一致的题;
    有缺快照 → current_run_missing>0 且 confirmed_rate=null。"""
    stats = {"judge_rate": None, "confirmed_rate": None,
             "current_run_missing": 0, "ledger": ledger_counts(sf)}
    if report is None:
        return stats
    gen = report["generation"]
    online = gen[gen["online_strategy"]]
    if online.get("judge_errors") == 0 and online.get("judged", 0) > 0:
        stats["judge_rate"] = online["fabricated"] / online["judged"]
    case_ids = [c["case_id"] for c in gen["faithfulness_cases"]]
    if not case_ids:
        return stats
    run_id = report["meta"]["run_id"]
    with sf() as s:
        rows = s.query(FaithCase).filter(FaithCase.eval_id.in_(case_ids)).all()
    matched = {r.eval_id: r for r in rows
               if normalize_citations(r.citations)["run_id"] == run_id}
    stats["current_run_missing"] = sum(1 for cid in case_ids if cid not in matched)
    if stats["current_run_missing"] == 0:
        confirmed = sum(1 for r in matched.values() if r.status == "已解决")
        stats["confirmed_rate"] = confirmed / len(case_ids)
    return stats


def update_faith_case(sf, row_id: int, status: str, resolution: str) -> dict | None:
    """处置:status ∈ {已解决, 无需解决};resolution 已在外层去白校验;写朴素 UTC resolved_at。"""
    with sf() as s:
        row = s.get(FaithCase, row_id)
        if row is None:
            return None
        row.status = status
        row.resolution = resolution
        row.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        s.commit()
        return _item(row)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rag_eval_service.py -v`
Expected: PASS(5 个用例)

- [ ] **Step 5: Commit**

```bash
git add app/services/rag_eval.py tests/test_rag_eval_service.py
git commit -m "feat(rag-eval): 评估页数据服务(报告契约校验/台账分页/两口径统计/处置)"
```

---

### Task 9: /api/rag-eval 路由

**Files:**
- Create: `app/routers/rag_eval.py`
- Modify: `app/schemas.py`(加 `FaithCasePatchRequest`)、`app/main.py`(注册路由 + `ReportCorruptError` 502 处理器)
- Test: `tests/test_rag_eval_api.py`(新建)

**Interfaces:**
- Consumes: Task 8 全部服务函数;`app.state.session_factory`、`app.state.rag_eval_report_path`(Task 7)。
- Produces:
  - `GET /api/rag-eval/report` → 报告 dict / `{"empty": true}` / 502 `rag_eval_report_corrupt`
  - `GET /api/rag-eval/faith-cases?status=&page=&size=` → `{stats, total, page, size, items}`;size 范围 1–50,越界 422
  - `PATCH /api/rag-eval/faith-cases/{row_id}` → 更新后 item / 404 `faith_case_not_found` / 422
  - 页面(Task 10)只依赖这三个端点的上述形状。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_rag_eval_api.py`:

```python
"""/api/rag-eval 路由:report 空态/损坏、台账分页页签与两口径、PATCH 校验。"""

import json

import httpx

from app.main import AppRuntime, create_app
from app.models import FaithCase
from tests.conftest import FakeStreamModel, UserBoundMemoryStore, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401
from tests.test_rag_eval_service import _report, _seed
from app.tools.business import MOCK_TOOLS, RetrievalTrace, TurnToolset


def _app(db_session_factory, tmp_path, report=None, corrupt=False):
    runtime = AppRuntime(
        store=UserBoundMemoryStore(1000, 100, 8000),
        toolset_factory=lambda sid: TurnToolset(list(MOCK_TOOLS), RetrievalTrace()),
        session_factory=db_session_factory)
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=runtime)
    path = tmp_path / "rag_eval.json"
    if corrupt:
        path.write_text("{oops", encoding="utf-8")
    elif report is not None:
        path.write_text(json.dumps(report), encoding="utf-8")
    app.state.rag_eval_report_path = path   # 测试用临时报告,不碰仓库真实产物
    return app


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://t")


async def test_report_empty_and_ok(db_session_factory, tmp_path):
    app = _app(db_session_factory, tmp_path)
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/report")
        assert resp.status_code == 200 and resp.json() == {"empty": True}
    app = _app(db_session_factory, tmp_path, report=_report())
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/report")
        assert resp.status_code == 200 and resp.json()["meta"]["run_id"] == "r1"


async def test_report_corrupt_502(db_session_factory, tmp_path):
    app = _app(db_session_factory, tmp_path, corrupt=True)
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/report")
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "rag_eval_report_corrupt"
        # 报告损坏不阻断台账:按「无当前报告」口径返回
        resp = await client.get("/api/rag-eval/faith-cases")
        assert resp.status_code == 200
        assert resp.json()["stats"]["judge_rate"] is None


async def test_faith_cases_list_and_stats(db_session_factory, tmp_path):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2", status="已解决")
        _seed(s, "B2", run_id="r1")
        s.commit()
    app = _app(sf, tmp_path,
               report=_report(run_id="r1", fabricated=2, judged=4,
                              case_ids=("A2", "B2")))
    async with await _client(app) as client:
        resp = await client.get("/api/rag-eval/faith-cases")
        body = resp.json()
        assert resp.status_code == 200 and body["total"] == 1   # 缺省未解决页签
        assert body["items"][0]["case_id"] == "B2"
        st = body["stats"]
        assert st["judge_rate"] == 0.5 and st["confirmed_rate"] == 0.5
        assert st["current_run_missing"] == 0
        assert st["ledger"]["total"] == 2 and st["ledger"]["已解决"] == 1
        resp = await client.get("/api/rag-eval/faith-cases",
                                params={"status": "已解决", "page": 1, "size": 1})
        assert resp.json()["total"] == 1
        assert (await client.get("/api/rag-eval/faith-cases",
                                 params={"status": "无效"})).status_code == 422
        assert (await client.get("/api/rag-eval/faith-cases",
                                 params={"size": 51})).status_code == 422


async def test_patch(db_session_factory, tmp_path):
    sf = db_session_factory
    with sf() as s:
        _seed(s, "A2")
        s.commit()
        rid = s.query(FaithCase).one().id
    app = _app(sf, tmp_path)
    async with await _client(app) as client:
        resp = await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                  json={"status": "已解决", "resolution": " 已补文档 "})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "已解决" and body["resolution"] == "已补文档"
        assert body["resolved_at"] is not None
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "无需解决",
                                         "resolution": "误报"})).status_code == 200
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "未解决",
                                         "resolution": "x"})).status_code == 422
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "已解决",
                                         "resolution": "   "})).status_code == 422
        assert (await client.patch(f"/api/rag-eval/faith-cases/{rid}",
                                   json={"status": "已解决",
                                         "resolution": "x" * 301})).status_code == 422
        assert (await client.patch("/api/rag-eval/faith-cases/999999",
                                   json={"status": "已解决",
                                         "resolution": "x"})).status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rag_eval_api.py -v`(需 docker compose 在线)
Expected: FAIL(404 — 路由不存在)

- [ ] **Step 3: 实现**

`app/schemas.py` 追加(沿用现有 BaseModel 风格):

```python
class FaithCasePatchRequest(BaseModel):
    status: Literal["已解决", "无需解决"]
    resolution: str
```

(文件顶部 import 区确认有 `from typing import Literal`;无则加。)

新建 `app/routers/rag_eval.py`:

```python
"""RAG 评估页 API(/api/rag-eval/*):报告只读 + faith_cases 台账分页/处置。"""

import asyncio

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.schemas import FaithCasePatchRequest
from app.services import rag_eval as svc

router = APIRouter(prefix="/api/rag-eval")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code,
                                                               "message": message}})


def _sf(request: Request):
    sf = request.app.state.session_factory
    if sf is None:
        return None
    return sf


@router.get("/report")
async def rag_eval_report(request: Request):
    path = request.app.state.rag_eval_report_path
    data = await asyncio.to_thread(svc.load_report, path)   # 坏 → ReportCorruptError → 502
    if data is None:
        return {"empty": True}
    return data


@router.get("/faith-cases")
async def faith_cases(request: Request,
                      status: str = Query("未解决"),
                      page: int = Query(1, ge=1),
                      size: int = Query(10, ge=1, le=50)):
    if status not in svc.STATUSES:
        return _err(422, "invalid_request", "status 须为 未解决/已解决/无需解决")
    sf = _sf(request)
    if sf is None:
        return _err(503, "rag_eval_unavailable", "数据库依赖未装配")
    path = request.app.state.rag_eval_report_path

    def _work():
        try:
            report = svc.load_report(path)
        except svc.ReportCorruptError:
            report = None   # 报告损坏不阻断台账:按「无当前报告」口径
        data = svc.list_faith_cases(sf, status, page, size)
        data["stats"] = svc.compute_stats(report, sf)
        return data

    return await asyncio.to_thread(_work)


@router.patch("/faith-cases/{row_id}")
async def faith_case_patch(row_id: int, body: FaithCasePatchRequest, request: Request):
    sf = _sf(request)
    if sf is None:
        return _err(503, "rag_eval_unavailable", "数据库依赖未装配")
    resolution = body.resolution.strip()
    if not 1 <= len(resolution) <= 300:
        return _err(422, "invalid_request", "处置说明须为 1-300 字符")
    item = await asyncio.to_thread(svc.update_faith_case, sf, row_id,
                                   body.status, resolution)
    if item is None:
        return _err(404, "faith_case_not_found", "个案不存在")
    return item
```

`app/main.py`:import 区加 `from app.routers.rag_eval import router as rag_eval_router` 与 `from app.services.rag_eval import ReportCorruptError`;`app.include_router(jobs_router)` 后加 `app.include_router(rag_eval_router)`;异常处理器区加:

```python
    @app.exception_handler(ReportCorruptError)
    async def _(request: Request, exc: ReportCorruptError) -> JSONResponse:
        return JSONResponse(status_code=502,
                            content=_error_body("rag_eval_report_corrupt",
                                                "评估报告文件损坏或契约不合法"))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rag_eval_api.py -v`
Expected: PASS(4 个用例)

- [ ] **Step 5: Commit**

```bash
git add app/routers/rag_eval.py app/schemas.py app/main.py tests/test_rag_eval_api.py
git commit -m "feat(rag-eval): /api/rag-eval 路由(report/faith-cases 分页与两口径/PATCH 处置)"
```

---

### Task 10: /rag-eval 页面(HTML + 图表 + 台账 + 重跑)

**Files:**
- Create: `app/static/rag-eval.html`
- Modify: `app/main.py`(加 `/rag-eval` 路由)、`app/static/chat.html`(页脚加导航)、`app/static/kb.html`(back-link 旁加导航)
- Test: `tests/test_rag_eval_page.py`(新建)、`tests/test_chat_page.py` 与 `tests/test_kb_page.py`(各加一条导航断言)

**Interfaces:**
- Consumes: `GET /api/rag-eval/report`(`meta/retrieval/generation/gates` 或 `{"empty": true}`)、`GET /api/rag-eval/faith-cases`、`PATCH /api/rag-eval/faith-cases/{id}`、`POST /api/jobs/eval-rag/run`、`GET /api/jobs/eval-rag`(Task 7/9 的形状)。
- Produces: 页面锚点 id(快照测试依赖):`report-header, gates-banner, kpis, kpi-mrr, kpi-colloquial, kpi-coverage, kpi-refuse, retrieval, metric-tabs, retrieval-chart, retrieval-caption, generation, coverage-chart, faith-chart, refuse-donut, summary, summary-table, fabricated-details, ledger, ledger-stats, ledger-tabs, ledger-list, ledger-pager, dispose-modal, dispose-text, rerun, run-btn, run-status, log-window`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_rag_eval_page.py`:

```python
from pathlib import Path

import httpx

from app.main import create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings

HTML = Path(__file__).parent.parent / "app" / "static" / "rag-eval.html"


async def test_rag_eval_page_served():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/rag-eval")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="kpis"' in resp.text


def test_page_sections_and_hooks():
    html = HTML.read_text(encoding="utf-8")
    for marker in ('id="report-header"', 'id="gates-banner"', 'id="kpis"',
                   'id="retrieval"', 'id="generation"', 'id="summary"',
                   'id="ledger"', 'id="rerun"', 'id="log-window"',
                   'id="metric-tabs"', 'id="summary-table"', 'id="fabricated-details"',
                   'id="ledger-tabs"', 'id="ledger-pager"', 'id="dispose-modal"',
                   'id="run-btn"'):
        assert marker in html
    for path in ("/api/rag-eval/report", "/api/rag-eval/faith-cases",
                 "/api/jobs/eval-rag/run", "/api/jobs/eval-rag"):
        assert path in html
    assert "--accent: #c96442" in html      # 复用设计 tokens


def test_page_no_innerhtml_injection():
    """外部字符串只走 textContent/createTextNode;SVG 走 DOM API(规格:防存储型注入)。"""
    html = HTML.read_text(encoding="utf-8")
    assert "innerHTML" not in html
    assert "createElementNS" in html        # SVG 经 DOM API 创建
    assert "不可用" in html                  # null 展示分支
```

`tests/test_chat_page.py`、`tests/test_kb_page.py` 各加一条:

```python
def test_nav_has_rag_eval():
    html = (Path(__file__).parent.parent / "app" / "static" /
            ("chat.html" if "chat" in __file__ else "kb.html")).read_text(encoding="utf-8")
    assert "/rag-eval" in html
```

(两个文件分别内联写,不要用 `__file__` 判断 —— 上面是示意,落地时各写各的:`chat.html` / `kb.html`。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rag_eval_page.py -v`
Expected: FAIL(404 / FileNotFoundError)

- [ ] **Step 3: 实现页面**

新建 `app/static/rag-eval.html`。`<style>` 的 `:root` 设计 tokens 块从 `app/static/kb.html` 的 `<style>` 开头原样复制(含 `--accent: #c96442` 等,保持三页同源),页面自有样式如下。完整文件:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 评估 - wayhelp</title>
<style>
/* :root 设计 tokens 与 chat.html / kb.html 同源,此处原样复制(略,见 kb.html) */
body { margin: 0; background: var(--bg); color: var(--text);
       font-family: var(--font-sans, "Anthropic Sans", sans-serif); }
.wrap { max-width: 960px; margin: 0 auto; padding: 24px 20px 80px; }
.topnav { display: flex; gap: 16px; margin-bottom: 8px; }
.topnav a { color: var(--accent); text-decoration: none; font-size: 13px; }
h1 { font-size: 22px; margin: 8px 0 4px; }
h2 { font-size: 16px; margin: 28px 0 12px; border-bottom: 1px solid var(--border, #e0dccd);
     padding-bottom: 6px; }
.muted { color: var(--muted, #8a8a7a); font-size: 12px; }
.banner { border-radius: 8px; padding: 10px 14px; margin: 12px 0; font-size: 13px; }
.banner.ok { background: #e7f2ea; color: #2e6b3f; }
.banner.warn { background: #faf0e6; color: #a05a2c; }
.banner.err { background: #f8e8e4; color: #a13c2a; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.card { background: #fff; border: 1px solid var(--border, #e0dccd); border-radius: 10px;
        padding: 14px; }
.card .k { font-size: 12px; color: var(--muted, #8a8a7a); }
.card .v { font-size: 22px; margin-top: 6px; font-variant-numeric: tabular-nums; }
.card .s { font-size: 11px; color: var(--muted, #8a8a7a); margin-top: 4px; }
.tabs { display: flex; gap: 8px; margin-bottom: 10px; }
.tabs button { border: 1px solid var(--border, #e0dccd); background: #fff;
               border-radius: 6px; padding: 4px 12px; cursor: pointer; font-size: 13px; }
.tabs button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.caption { font-size: 12px; color: var(--muted, #8a8a7a); margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
th, td { border: 1px solid var(--border, #e0dccd); padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
tr.best td { background: #f3ece4; font-weight: 600; }
.case { background: #fff; border: 1px solid var(--border, #e0dccd); border-radius: 8px;
        padding: 12px; margin-bottom: 10px; font-size: 13px; }
.case .head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.chip { font-size: 11px; background: #f0ede2; border-radius: 4px; padding: 1px 6px; }
.ev { margin-top: 8px; border-top: 1px dashed var(--border, #e0dccd); padding-top: 8px; }
.ev .item { padding: 6px 8px; border-radius: 6px; margin-bottom: 6px; background: #faf9f4; }
.ev .item.cited { background: #fdf1ea; border-left: 3px solid var(--accent); }
button.act { border: 1px solid var(--accent); color: var(--accent); background: #fff;
             border-radius: 6px; padding: 3px 10px; cursor: pointer; font-size: 12px; }
.pager { display: flex; gap: 8px; align-items: center; margin-top: 10px; font-size: 13px; }
#log-window { background: #23211d; color: #d8d4c8; font-family: var(--font-mono, monospace);
              font-size: 12px; border-radius: 8px; padding: 12px; height: 220px;
              overflow-y: auto; white-space: pre-wrap; margin-top: 10px; }
#dispose-modal { position: fixed; inset: 0; background: rgba(0,0,0,.35);
                 display: none; align-items: center; justify-content: center; }
#dispose-modal .box { background: #fff; border-radius: 10px; padding: 20px;
                      width: 420px; }
#dispose-modal textarea { width: 100%; height: 80px; margin: 8px 0; }
.big-btn { background: var(--accent); color: #fff; border: 0; border-radius: 8px;
           padding: 8px 18px; font-size: 14px; cursor: pointer; }
.big-btn:disabled { opacity: .5; cursor: default; }
.grid2 { display: grid; grid-template-columns: 1fr 260px; gap: 20px; align-items: start; }
</style>
</head>
<body>
<div class="wrap">
  <div class="topnav"><a href="/">← 返回对话</a><a href="/kb">知识库管理</a></div>
  <h1>RAG 评估</h1>

  <section id="report-header">
    <div id="report-meta" class="muted">加载中…</div>
    <div id="gates-banner"></div>
    <div id="run-banner"></div>
  </section>

  <h2>四项 KPI</h2>
  <section id="kpis" class="cards">
    <div class="card" id="kpi-mrr"><div class="k">最佳整体 MRR</div><div class="v">-</div><div class="s"></div></div>
    <div class="card" id="kpi-colloquial"><div class="k">口语桶 MRR 提升</div><div class="v">-</div><div class="s">hybrid_rerank 对 dense / C_colloquial</div></div>
    <div class="card" id="kpi-coverage"><div class="k">答案覆盖度</div><div class="v">-</div><div class="s">上线管线 hybrid_rerank</div></div>
    <div class="card" id="kpi-refuse"><div class="k">库外拒答率</div><div class="v">-</div><div class="s">上线管线 D_absent 桶</div></div>
  </section>

  <h2>检索质量(四策略 × 四桶)</h2>
  <section id="retrieval">
    <div class="tabs" id="metric-tabs">
      <button data-metric="mrr" class="on">MRR</button>
      <button data-metric="recall5">Recall@5</button>
      <button data-metric="evidence_coverage">证据覆盖度</button>
    </div>
    <div id="retrieval-chart"></div>
    <div class="caption" id="retrieval-caption"></div>
  </section>

  <h2>生成质量</h2>
  <section id="generation" class="grid2">
    <div>
      <div class="muted">四策略答案覆盖度</div>
      <div id="coverage-chart"></div>
      <div class="caption" id="coverage-caption"></div>
      <div class="muted" style="margin-top:14px">上线管线四桶忠实率(已回答样本)</div>
      <div id="faith-chart"></div>
      <div class="caption" id="faith-note"></div>
    </div>
    <div>
      <div class="muted">库外拒答率(上线管线)</div>
      <div id="refuse-donut"></div>
    </div>
  </section>

  <h2>完整数据</h2>
  <section id="summary">
    <table id="summary-table"></table>
    <details id="fabricated-details" style="margin-top:12px">
      <summary>编造个案(本轮,上线管线)</summary>
      <div id="fabricated-list" style="margin-top:10px"></div>
    </details>
  </section>

  <h2>编造个案台账</h2>
  <section id="ledger">
    <div id="ledger-stats" class="muted"></div>
    <div class="tabs" id="ledger-tabs" style="margin-top:8px">
      <button data-status="未解决" class="on">未解决</button>
      <button data-status="已解决">已解决</button>
      <button data-status="无需解决">无需解决</button>
    </div>
    <div id="ledger-list"></div>
    <div class="pager" id="ledger-pager">
      <button class="act" id="pager-prev">上一页</button>
      <span id="pager-info"></span>
      <button class="act" id="pager-next">下一页</button>
    </div>
  </section>

  <h2>就地重跑</h2>
  <section id="rerun">
    <button class="big-btn" id="run-btn">重跑 RAG 评估</button>
    <span id="run-status" class="muted" style="margin-left:10px"></span>
    <div id="log-window"></div>
  </section>
</div>

<div id="dispose-modal">
  <div class="box">
    <div id="dispose-title" style="font-size:14px;margin-bottom:6px"></div>
    <textarea id="dispose-text" placeholder="处置说明(必填,1-300 字)"></textarea>
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button class="act" id="dispose-cancel">取消</button>
      <button class="act" id="dispose-ok">确认</button>
    </div>
  </div>
</div>

<script>
var STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"];
var BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"];
var BUCKET_CN = {A_policy: "政策", B_model: "型号", C_colloquial: "口语", E_multi: "多意图"};
var COLORS = ["#8a8a7a", "#4a6fa5", "#c0a03c", "#c96442"];
var SVGNS = "http://www.w3.org/2000/svg";
var report = null;            // 当前报告(或 null)
var ledgerState = {status: "未解决", page: 1, size: 10, total: 0};
var pollTimer = null;

function el(tag, attrs, text) {
  var n = document.createElement(tag);
  if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function sel(id) { return document.getElementById(id); }
function fmt(v, d) { return (v === null || v === undefined) ? "不可用" : Number(v).toFixed(d === undefined ? 3 : d); }
function pct(v) { return (v === null || v === undefined) ? "不可用" : (v * 100).toFixed(1) + "%"; }

function api(path, opts) {
  return fetch(path, opts).then(function (r) {
    if (!r.ok) { var e = new Error("HTTP " + r.status); e.status = r.status; throw e; }
    return r.json();
  });
}

/* ---------- SVG 图(全部 DOM API,无 innerHTML) ---------- */
function svgEl(tag, attrs, text) {
  var n = document.createElementNS(SVGNS, tag);
  if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

/* 分组柱状图:groups=桶, series=策略, values[series][group]=number|null */
function groupedBars(mount, groups, series, values) {
  mount.replaceChildren();
  var W = 760, H = 250, padL = 40, padB = 44, padT = 14;
  var innerW = W - padL, innerH = H - padB - padT;
  var svg = svgEl("svg", {width: W, height: H, role: "img"});
  svg.appendChild(svgEl("line", {x1: padL, y1: H - padB, x2: W, y2: H - padB, stroke: "#c9c4b4"}));
  var gw = innerW / groups.length;
  var bw = Math.min(16, (gw - 30) / series.length);
  groups.forEach(function (g, gi) {
    var gx = padL + gi * gw;
    series.forEach(function (s, si) {
      var v = values[s] ? values[s][g] : null;
      if (v === null || v === undefined) return;      // null 不画 0
      var h = Math.max(1, v * innerH);
      var x = gx + 15 + si * (bw + 3);
      var rect = svgEl("rect", {x: x, y: H - padB - h, width: bw, height: h,
                                fill: COLORS[si % COLORS.length]});
      rect.appendChild(svgEl("title", null, s + " / " + g + ": " + fmt(v)));
      svg.appendChild(rect);
      svg.appendChild(svgEl("text", {x: x + bw / 2, y: H - padB - h - 3,
                                     "text-anchor": "middle", "font-size": 8,
                                     fill: "#6b675c"}, fmt(v, 2)));
    });
    svg.appendChild(svgEl("text", {x: gx + gw / 2, y: H - padB + 14,
                                   "text-anchor": "middle", "font-size": 11,
                                   fill: "#6b675c"}, BUCKET_CN[g] || g));
  });
  series.forEach(function (s, si) {                   // 图例
    var lx = padL + si * 130;
    svg.appendChild(svgEl("rect", {x: lx, y: H - 16, width: 10, height: 10,
                                   fill: COLORS[si % COLORS.length]}));
    svg.appendChild(svgEl("text", {x: lx + 14, y: H - 7, "font-size": 10,
                                   fill: "#6b675c"}, s));
  });
  mount.appendChild(svg);
}

/* 单序列也走分组图:valuesToNested 把 (labels, values) 折成 {label: {label: v}} */
function valuesToNested(labels, values) {
  var out = {}; labels.forEach(function (l, i) { out[l] = {}; out[l][l] = values[i]; });
  return out;
}

/* 圆环:value 0..1 或 null */
function donut(mount, value) {
  mount.replaceChildren();
  var svg = svgEl("svg", {width: 160, height: 160, role: "img"});
  var r = 60, cx = 80, cy = 80, c = 2 * Math.PI * r;
  svg.appendChild(svgEl("circle", {cx: cx, cy: cy, r: r, fill: "none",
                                   stroke: "#e8e4d6", "stroke-width": 16}));
  if (value !== null && value !== undefined) {
    svg.appendChild(svgEl("circle", {cx: cx, cy: cy, r: r, fill: "none",
                                     stroke: "#c96442", "stroke-width": 16,
                                     "stroke-dasharray": (value * c) + " " + c,
                                     transform: "rotate(-90 80 80)"}));
  }
  svg.appendChild(svgEl("text", {x: cx, y: cy + 6, "text-anchor": "middle",
                                 "font-size": 18, fill: "#3d3a33"}, pct(value)));
  mount.appendChild(svg);
}

/* ---------- 报告渲染 ---------- */
function renderReport(rep) {
  report = rep;
  var m = rep.meta;
  sel("report-meta").replaceChildren(
    document.createTextNode(
      "评估集 " + m.total_cases + " 题(校准 " + m.calibration_cases +
      " / 测试 " + m.test_cases + ")| 知识库 " + m.corpus_chunks + " 块 | 嵌入 " +
      m.embedding_model + " | 重排 " + m.rerank_model + " | 裁判 " + m.judge_model +
      " | 上次跑 " + m.ts + "(run " + m.run_id + ")"));

  var g = rep.gates || {};
  var gb = sel("gates-banner");
  gb.replaceChildren();
  var gcls = g.passed ? "banner ok" : "banner warn";
  var gtxt = g.passed ? "质量门槛:通过"
    : "质量门槛:未通过(judge_error_zero=" + String(g.judge_error_zero) +
      ", bm25 B 桶 SR@10=" + fmt(g.bm25_b_model_section_recall_at_10) + ")";
  gb.appendChild(el("div", {class: gcls}, gtxt));

  /* KPI */
  var best = null;
  STRATEGIES.forEach(function (s) {
    var v = rep.retrieval[s].overall.mrr;
    if (v !== null && (best === null || v > best.v)) best = {s: s, v: v};
  });
  setKpi("kpi-mrr", best ? fmt(best.v) : "不可用", best ? best.s : "");
  var colo = rep.retrieval.hybrid_rerank.by_bucket.C_colloquial;
  var coloD = rep.retrieval.dense.by_bucket.C_colloquial;
  var lift = (colo && coloD) ? colo.mrr - coloD.mrr : null;
  setKpi("kpi-colloquial", lift === null ? "不可用" : (lift >= 0 ? "+" : "") + fmt(lift),
         "hybrid_rerank " + fmt(colo && colo.mrr) + " vs dense " + fmt(coloD && coloD.mrr));
  var online = rep.generation[rep.generation.online_strategy];
  setKpi("kpi-coverage", pct(online.answer_coverage),
         "已回答 " + online.answered + "/" + online.answerable_total);
  setKpi("kpi-refuse", pct(online.d_refuse_rate),
         online.d_refused + "/" + online.d_total + " 题");

  renderRetrieval(currentMetric());
  renderGeneration(rep, online);
  renderSummary(rep);
}

function setKpi(id, v, s) {
  sel(id).querySelector(".v").textContent = v;
  sel(id).querySelector(".s").textContent = s || "";
}

function currentMetric() {
  var b = sel("metric-tabs").querySelector("button.on");
  return b ? b.getAttribute("data-metric") : "mrr";
}

function renderRetrieval(metric) {
  var values = {};
  STRATEGIES.forEach(function (s) {
    values[s] = {};
    BUCKETS.forEach(function (b) {
      var bb = report.retrieval[s].by_bucket[b];
      values[s][b] = bb ? bb[metric] : null;
    });
  });
  groupedBars(sel("retrieval-chart"), BUCKETS, STRATEGIES, values);
  sel("retrieval-caption").textContent = caption(metric);
}

function caption(metric) {
  var names = {mrr: "MRR", recall5: "Recall@5", evidence_coverage: "证据覆盖度"};
  var best = null, denseV = null;
  STRATEGIES.forEach(function (s) {
    var vals = BUCKETS.map(function (b) {
      var bb = report.retrieval[s].by_bucket[b];
      return bb ? bb[metric] : null;
    }).filter(function (v) { return v !== null; });
    if (!vals.length) return;
    var avg = vals.reduce(function (a, x) { return a + x; }, 0) / vals.length;
    if (s === "dense") denseV = avg;
    if (best === null || avg > best.v) best = {s: s, v: avg};
  });
  if (!best) return "数据不可用。";
  var tail = "";
  if (denseV !== null && best.s !== "dense") {
    tail = ",较基线 dense " + (best.v - denseV >= 0 ? "+" : "") + fmt(best.v - denseV);
  }
  return names[metric] + " 四桶均值最佳为 " + best.s + "(" + fmt(best.v) + ")" + tail + "。";
}

function renderGeneration(rep, online) {
  groupedBars(sel("coverage-chart"), STRATEGIES, STRATEGIES,
              valuesToNested(STRATEGIES, STRATEGIES.map(function (s) {
                return rep.generation[s].answer_coverage;
              })));
  sel("coverage-caption").textContent =
    "拒答记 0 分;裁判失败的策略为「不可用」。";
  groupedBars(sel("faith-chart"), BUCKETS, BUCKETS,
              valuesToNested(BUCKETS, BUCKETS.map(function (b) {
                var bb = online.by_bucket[b];
                return bb ? bb.faithful_rate : null;
              })));
  sel("faith-note").textContent =
    "忠实率 = faithful/judged(仅已回答且裁判成功样本);本轮已回答 " +
    online.answered + "/" + online.answerable_total + ",裁判失败 " +
    online.judge_errors + " 题。";
  donut(sel("refuse-donut"), online.d_refuse_rate);
}

function renderSummary(rep) {
  var t = sel("summary-table");
  t.replaceChildren();
  var head = el("tr");
  ["策略", "MRR 总体", "MRR 政策", "MRR 型号", "MRR 口语", "MRR 多意图",
   "证据覆盖度", "答案覆盖度"].forEach(function (h) { head.appendChild(el("th", null, h)); });
  t.appendChild(head);
  var best = null;
  STRATEGIES.forEach(function (s) {
    var v = rep.retrieval[s].overall.mrr;
    if (v !== null && (best === null || v > best)) best = v;
  });
  STRATEGIES.forEach(function (s) {
    var r = rep.retrieval[s], genv = rep.generation[s];
    var tr = el("tr");
    if (r.overall.mrr === best) tr.className = "best";   // 总体 MRR 最佳行带底色
    tr.appendChild(el("td", null, s + (r.ungated ? "(仅观测)" : "")));
    tr.appendChild(el("td", null, fmt(r.overall.mrr)));
    BUCKETS.forEach(function (b) {
      var bb = r.by_bucket[b];
      tr.appendChild(el("td", null, bb ? fmt(bb.mrr) : "不可用"));
    });
    tr.appendChild(el("td", null, fmt(r.overall.evidence_coverage)));
    tr.appendChild(el("td", null, fmt(genv.answer_coverage)));
    t.appendChild(tr);
  });

  var list = sel("fabricated-list");
  list.replaceChildren();
  var cases = rep.generation.faithfulness_cases || [];
  sel("fabricated-details").querySelector("summary").textContent =
    "编造个案(本轮,上线管线,共 " + cases.length + " 条)";
  cases.forEach(function (c) {
    var card = el("div", {class: "case"});
    card.appendChild(el("div", {class: "head"}, c.case_id + "(" + c.bucket + ")"));
    card.appendChild(el("div", null, "问题:" + c.query));
    card.appendChild(el("div", null, "答案:" + c.answer));
    var claims = (c.unsupported_claims || []).map(function (x) {
      return (x.claim || "") + " — " + (x.reason || "");
    }).join(";") || "(无理由)";
    card.appendChild(el("div", {class: "muted"}, "裁判理由:" + claims));
    list.appendChild(card);
  });
}

/* ---------- 台账 ---------- */
function loadLedger() {
  var q = "?status=" + encodeURIComponent(ledgerState.status) +
          "&page=" + ledgerState.page + "&size=" + ledgerState.size;
  api("/api/rag-eval/faith-cases" + q).then(function (data) {
    ledgerState.total = data.total;
    renderLedgerStats(data.stats);
    renderLedgerList(data.items);
    var pages = Math.max(1, Math.ceil(data.total / ledgerState.size));
    sel("pager-info").textContent = "第 " + ledgerState.page + " / " + pages + " 页(共 " + data.total + " 条)";
    sel("pager-prev").disabled = ledgerState.page <= 1;
    sel("pager-next").disabled = ledgerState.page >= pages;
  }).catch(function () { sel("ledger-list").textContent = "台账加载失败"; });
}

function renderLedgerStats(st) {
  var box = sel("ledger-stats");
  box.replaceChildren();
  var line1 = "本轮裁判判出率 " + pct(st.judge_rate) +
              " | 本轮人工确认幻觉率 " + pct(st.confirmed_rate) +
              (st.current_run_missing > 0 ? "(缺 " + st.current_run_missing +  " 条本轮快照,口径不可用)" : "");
  var l = st.ledger;
  var line2 = "台账累计 " + l.total + " 条:未解决 " + l["未解决"] +
              " / 已解决 " + l["已解决"] + " / 无需解决 " + l["无需解决"];
  box.appendChild(el("div", null, line1));
  box.appendChild(el("div", null, line2));
}

function renderLedgerList(items) {
  var box = sel("ledger-list");
  box.replaceChildren();
  if (!items.length) { box.appendChild(el("div", {class: "muted"}, "(空)")); return; }
  items.forEach(function (it) {
    var card = el("div", {class: "case"});
    var head = el("div", {class: "head"});
    head.appendChild(el("span", {class: "chip"}, it.case_id));
    head.appendChild(el("span", {class: "chip"}, it.bucket));
    head.appendChild(el("span", {class: "muted"},
                        "出现 " + it.seen_count + " 次 | 最近 " + (it.last_seen_at || "-")));
    if (ledgerState.status === "未解决") {
      ["已解决", "无需解决"].forEach(function (st) {
        var btn = el("button", {class: "act"}, st);
        btn.addEventListener("click", function () { openDispose(it, st); });
        head.appendChild(btn);
      });
    }
    card.appendChild(head);
    card.appendChild(el("div", null, "问题:" + it.query));
    card.appendChild(el("div", null, "答案:" + it.answer));
    var claims = (it.unsupported_claims || []);
    card.appendChild(el("div", {class: "muted"}, "裁判理由:" +
      (Array.isArray(claims) ? claims.map(function (x) {
        return (x.claim || "") + " — " + (x.reason || ""); }).join(";")
        : String(claims))));
    if (it.resolution) card.appendChild(el("div", {class: "muted"}, "处置:" + it.resolution));
    var det = el("details", {class: "ev"});
    det.appendChild(el("summary", null, "本轮 Top-K 证据(" +
      it.citations.evidence.length + " 条,高亮为答案引用)"));
    var cited = {};
    (it.citations.cited_refs || []).forEach(function (n) { cited[n] = true; });
    it.citations.evidence.forEach(function (ev) {
      var item = el("div", {class: "item" + (cited[ev.ref_no] ? " cited" : "")});
      item.appendChild(el("div", null, "[" + ev.ref_no + "] " +
        (ev.section_path || "") + (cited[ev.ref_no] ? "(答案引用)" : "")));
      item.appendChild(el("div", {class: "muted"},
                          (ev.question || "") + " / " + (ev.answer || "")));
      det.appendChild(item);
    });
    card.appendChild(det);
    box.appendChild(card);
  });
}

/* 处置弹窗 */
var disposeTarget = null;
function openDispose(item, st) {
  disposeTarget = {id: item.id, status: st};
  sel("dispose-title").textContent = "将 " + item.case_id + " 标记为「" + st + "」";
  sel("dispose-text").value = "";
  sel("dispose-modal").style.display = "flex";
}
function closeDispose() { sel("dispose-modal").style.display = "none"; disposeTarget = null; }

/* ---------- 重跑 ---------- */
function setRunning(running) {
  sel("run-btn").disabled = running;
  sel("run-status").textContent = running ? "评估运行中…(约几十分钟)" : "";
}
function showRunResult(st) {
  var box = sel("run-banner");
  box.replaceChildren();
  if (st.status === "ok" && st.quality_passed === false) {
    box.appendChild(el("div", {class: "banner warn"}, "评估完成,质量门槛未通过;已展示本轮报告。"));
  } else if (st.status === "ok") {
    box.appendChild(el("div", {class: "banner ok"}, "评估完成,质量门槛通过。"));
  } else {
    box.appendChild(el("div", {class: "banner err"},
                        "评估运行失败(exit " + st.exit_code + "),页面保留上一份报告;详见日志。"));
  }
}
function pollJob() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(function () {
    api("/api/jobs/eval-rag").then(function (st) {
      var lw = sel("log-window");
      lw.textContent = st.log_tail || "";
      lw.scrollTop = lw.scrollHeight;
      if (st.status === "ok" || st.status === "failed") {
        clearInterval(pollTimer); pollTimer = null;
        setRunning(false);
        showRunResult(st);
        loadAll();                       // 跑完自动重新取数
      }
    }).catch(function () { /* 下一轮再试 */ });
  }, 2000);
}

/* ---------- 入口 ---------- */
function loadAll() {
  api("/api/rag-eval/report").then(function (rep) {
    if (rep.empty) {
      sel("report-meta").textContent = "还没有评估报告 — 点下方「重跑 RAG 评估」跑第一轮。";
      return;
    }
    renderReport(rep);
  }).catch(function (e) {
    sel("report-meta").textContent =
      e.status === 502 ? "评估报告文件损坏,请重跑。" : "报告加载失败。";
  });
  loadLedger();
  api("/api/jobs/eval-rag").then(function (st) {   // 页面打开时若正在跑,接着轮询
    sel("log-window").textContent = st.log_tail || "";
    if (st.status === "running") { setRunning(true); pollJob(); }
  }).catch(function () {});
}

sel("metric-tabs").addEventListener("click", function (e) {
  if (e.target.tagName !== "BUTTON") return;
  sel("metric-tabs").querySelectorAll("button").forEach(function (b) { b.className = ""; });
  e.target.className = "on";
  if (report) renderRetrieval(e.target.getAttribute("data-metric"));
});
sel("ledger-tabs").addEventListener("click", function (e) {
  if (e.target.tagName !== "BUTTON") return;
  sel("ledger-tabs").querySelectorAll("button").forEach(function (b) { b.className = ""; });
  e.target.className = "on";
  ledgerState.status = e.target.getAttribute("data-status");
  ledgerState.page = 1;
  loadLedger();
});
sel("pager-prev").addEventListener("click", function () {
  if (ledgerState.page > 1) { ledgerState.page--; loadLedger(); }
});
sel("pager-next").addEventListener("click", function () {
  ledgerState.page++; loadLedger();
});
sel("dispose-cancel").addEventListener("click", closeDispose);
sel("dispose-ok").addEventListener("click", function () {
  var text = sel("dispose-text").value.trim();
  if (!text || !disposeTarget) return;
  api("/api/rag-eval/faith-cases/" + disposeTarget.id, {
    method: "PATCH",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({status: disposeTarget.status, resolution: text}),
  }).then(function () { closeDispose(); loadLedger(); })
    .catch(function (e) { sel("dispose-title").textContent = "提交失败:HTTP " + (e.status || "?"); });
});
sel("run-btn").addEventListener("click", function () {
  api("/api/jobs/eval-rag/run", {method: "POST"}).then(function () {
    setRunning(true); pollJob();
  }).catch(function (e) {
    if (e.status === 409) { setRunning(true); pollJob(); }
  });
});

loadAll();
</script>
</body>
</html>
```

`app/main.py` 在 `/kb` 路由后加:

```python
    @app.get("/rag-eval", include_in_schema=False)
    async def rag_eval_ui() -> FileResponse:
        return FileResponse(static_dir / "rag-eval.html")
```

`app/static/chat.html` 页脚 `/kb` 链接旁加 `<a href="/rag-eval">RAG 评估</a>`(保持该行其余不变);`app/static/kb.html` 的 `<a class="back-link" href="/">← 返回对话</a>` 后加 `<a class="back-link" href="/rag-eval" style="margin-left:12px">RAG 评估</a>`。

注意:复制 `:root` tokens 后删掉上面 `<style>` 第一行的「略,见 kb.html」注释 —— tokens 必须真实落在文件里(快照测试断言 `--accent: #c96442`)。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rag_eval_page.py tests/test_chat_page.py tests/test_kb_page.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/static/rag-eval.html app/main.py app/static/chat.html app/static/kb.html tests/test_rag_eval_page.py tests/test_chat_page.py tests/test_kb_page.py
git commit -m "feat(rag-eval): /rag-eval 页面(报告头/KPI/检索+生成图/汇总表/台账/重跑日志窗)"
```

---

### Task 11: Makefile + README + .gitignore + 全量回归

**Files:**
- Create: `Makefile`
- Modify: `README.md`(用法补 eval-rag + /rag-eval 页 + 单 worker 契约)、`.gitignore`
- Test: 全量

- [ ] **Step 1: Makefile**

仓库根新建 `Makefile`:

```make
.PHONY: eval-rag
eval-rag:
	uv run python evals/run_retrieval_compare.py
```

- [ ] **Step 2: README 与 .gitignore**

`README.md` 用法一节(现有 `uv run python evals/run_retrieval_compare.py` 附近)补:

```markdown
- `make eval-rag`:等价于上面的四策略评估命令;跑完产出 `evals/results/rag_eval.json`,后台 `/rag-eval` 页(报告 / 编造个案台账 / 就地重跑)读这份产物。
- 部署契约:本应用按**单 Uvicorn worker** 运行(内存作业注册表与 Milvus Lite 独占均不支持多 worker);不要配置 `--workers >1`。
```

`.gitignore` 追加:

```
# 滚动「最新一轮」评估报告(时间戳归档版照旧入库)
evals/results/rag_eval.json
```

- [ ] **Step 3: 全量回归**

Run: `uv run pytest tests/ -v`(需 docker compose 在线)
Expected: 全 PASS;特别确认 `test_ddl_sync`、`test_chat_page`、`test_kb_page`、`test_lifespan`、`test_main_boot` 未被打破。

- [ ] **Step 4: 冒烟(可选,需真实 API key)**

`make eval-rag` 真跑一轮 → 起服务 → 打开 `/rag-eval` 人工过一遍七个块;台账处置一条;页面点重跑看日志窗。

- [ ] **Step 5: Commit**

```bash
git add Makefile README.md .gitignore
git commit -m "feat(rag-eval): Makefile eval-rag 目标 + README(页面入口与单 worker 契约)+ 忽略滚动报告"
```

---

## 执行顺序与依赖

严格按:**1 → 2 → 3 → 4 → 5(管道线)→ 8(rag_eval 服务)→ 6(JobRunner)→ 7(jobs 路由装配)→ 9(rag-eval 路由)→ 10(页面)→ 11(收尾)**。
Task 7 装配 `report_loader` 依赖 Task 8 的 `app.services.rag_eval.load_report`;Task 9 依赖 Task 7 的 `app.state` 装配;Task 10 页面依赖 Task 7/9 的 API 形状。
