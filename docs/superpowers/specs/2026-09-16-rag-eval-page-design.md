# /rag-eval 评估页设计(2026-09-16,用户已批准)

## 背景与目标

RAG 评估目前只在终端跑:`uv run python evals/run_retrieval_compare.py`,读 stdout 或生成的 `{ts}_compare.md`。本任务把报告落成后台一页 `/rag-eval`,并支持页面上就地重跑。

页面块与数据来源(用户规格,逐字确认):

| 页面块 | 内容 | 数据来源 |
|---|---|---|
| 报告头 | 评估集题数、知识库块数、嵌入与重排模型、裁判模型、上次跑的时间 | `rag_eval.json` 的 `meta` |
| 四项 KPI | 最佳整体 MRR、口语桶 MRR 提升、答案覆盖度、库外拒答率 | 同一份产物,不另算 |
| 检索质量 | 四策略 × 四个可作答桶分组柱状图,MRR / Recall@5 / 证据覆盖度三个指标切着看,每张图配一句读图 | `retrieval` |
| 生成质量 | 四策略答案覆盖度柱状图、上线管线四桶忠实度、库外拒答率圆环 | `generation` |
| 完整数据 | 分桶 MRR 与两项覆盖度汇总表,总体 MRR 最佳行带底色;编造个案默认收起,展开逐条看问题、生成答案、裁判理由 | `generation.faithfulness_cases` |
| 编造个案台账 | 顶上两个口径幻觉率 + 台账累计一行;跨轮累计个案:状态页签 + 分页,每条带本轮 Top-K 证据全集(标出答案引用哪几条)与「已解决 / 无需解决」按钮,处置必填一句说明 | `faith_cases` 表(`/api/rag-eval/faith-cases`) |
| 就地重跑 | 「重跑 RAG 评估」按钮 + 日志窗口,跑完自动重新取数 | `/api/jobs`(作业名 `eval-rag`) |

## 已确认口径(用户逐项拍板)

1. **答案覆盖度**:生成段从只跑 hybrid_rerank 扩到**四策略**,裁判在忠实性之外按 `expect_points` 打答案覆盖分(0–1)。代价:每轮评估 LLM 调用约 ×4,用户已知情。
2. **证据覆盖度** = CompleteHit@10(期望证据组被 Top-10 100% 覆盖才计 1,按桶平均)。
3. **人工确认幻觉率**:「已解决」= 确认幻觉;「无需解决」= 误报/可接受。复用现有 status 枚举,不改表结构。
4. **产物形态**:管道产出规范化 `evals/results/rag_eval.json`,页面 API 只读文件 + 查库,不做即时派生。

KPI 口径(设计提议,随设计一并批准):
- 最佳整体 MRR = max(四策略 overall.mrr)
- 口语桶 MRR 提升 = hybrid_rerank − dense 的 C_colloquial 桶 MRR
- 答案覆盖度 = 上线管线(hybrid_rerank)answer_coverage
- 库外拒答率 = 上线管线 D_absent 桶拒答率

## 一、管道改动(`evals/run_retrieval_compare.py`)

### 检索段
`_test_metrics` 的 per-case 值(recall5/recall10/ch5/ch10/mrr10)已在算,`by_bucket` 聚合扩展为每桶:
`{mrr, recall5, evidence_coverage(=ch10 均值), sr10, refused}`(仅 A/B/C/E 可作答桶进 by_bucket;D 桶只进拒答正确率,与现状一致)。

### 生成段(四策略)
- 循环 `STRATEGIES`,各用自己的冻结阈值判低置信(ungated 臂:零命中才低置信,与现状兜底一致)。
- `_simulate_second_turn` 工具出参 `effective_strategy` 按当前策略填(现硬编码 hybrid_rerank)。
- D 桶 case 照常走低置信→`REFUSAL_ANSWER` 路径;非拒答且非 should_refuse 的 case 交裁判。
- **只有 hybrid_rerank 的 fabricated 写 `faith_cases` 台账**(该表 `strategy` 字段默认 'hybrid_rerank' 本就是这个语义);其余策略的 fabricated 只进 JSON 统计。
- 现有硬门槛(`judge_error == 0`、bm25 B 桶不设闸 SR@10 ≥ 0.5)语义不变;judge_error 统计扩为全策略合计。

### 裁判(`app/prompts/faithfulness.py`)
- `JUDGE_PROMPT` 增加入参 `expect_points`,输出 JSON 增加 `coverage` 字段(0–1 浮点,答案对标准要点的覆盖度)。
- `parse_judge_output` 校验 coverage ∈ [0,1];非法同现有口径计入重试/判 judge_error。
- verdict 结构:`{verdict, unsupported_claims, cited_refs, coverage}`。

### `faith_cases.citations` 载荷改形
现状存的是裸 evidence 列表;改为 `{"evidence": [...], "cited_refs": [...]}`,让「答案引用了哪几条」跨轮留存。JSON 列无 DDL 变更;需同步改 `upsert_faith_case` 调用处、`tests/test_faith_cases.py` 及后续 API 读取方。

### 新产物 `evals/results/rag_eval.json`
固定文件名 = 最新一轮,与时间戳 `_compare.json/.md` 并存(后两者照旧)。形状:

```json
{
  "meta": {
    "eval_id": "20260916T024957Z", "ts": "...", "elapsed_s": 0,
    "cases": 150, "corpus_chunks": 0,
    "embedding_model": "...", "rerank_model": "...",
    "eval_model": "...", "judge_model": "..."
  },
  "retrieval": {
    "dense": {"threshold": 0.6491, "ungated": false,
      "overall": {"mrr": 0, "recall5": 0, "evidence_coverage": 0, "sr10": 0},
      "by_bucket": {"A_policy": {"mrr": 0, "recall5": 0, "evidence_coverage": 0}, "...": {}}},
    "bm25": {}, "hybrid": {}, "hybrid_rerank": {}
  },
  "generation": {
    "online_strategy": "hybrid_rerank",
    "dense": {"answer_coverage": 0, "faithful": 0, "fabricated": 0, "refused": 0,
              "judge_errors": 0,
              "by_bucket": {"A_policy": {"faithful_rate": 0, "coverage": 0}, "...": {}},
              "d_refuse_rate": 0},
    "bm25": {}, "hybrid": {}, "hybrid_rerank": {},
    "faithfulness_cases": [
      {"id": "...", "bucket": "...", "query": "...", "answer": "...",
       "strategy": "hybrid_rerank", "unsupported_claims": ["..."],
       "cited_refs": [1],
       "evidence": [{"ref_no": 1, "chunk_id": 0, "section_path": "...",
                     "question": "...", "answer": "...", "category": "..."}]}
    ]
  }
}
```

`faithfulness_cases` 只收录上线管线被判 fabricated 的 case(与台账同源);`evidence` 为该轮 Top-K 全集(`Evidence.to_dict()` 原样,`ref_no/chunk_id/section_path/question/answer/category`),`cited_refs` 标的是 `ref_no`。

## 二、作业层(新建)

- `app/jobs/runner.py`:内存作业注册表。`run("eval-rag")` 起 asyncio 子进程 `uv run python evals/run_retrieval_compare.py`(cwd=仓库根,继承环境变量拿 API key),stdout/stderr 落 `data/jobs/eval-rag.log`(每轮截断)。单并发:运行中重复触发返回 None/抛冲突,由路由转 409。
- `app/routers/jobs.py`:
  - `POST /api/jobs/eval-rag/run` → `{started: true}`;运行中 → 409。
  - `GET /api/jobs/eval-rag` → `{status: idle|running|ok|failed, started_at, finished_at, exit_code, log_tail}`(log_tail = 日志尾部约 8KB)。
- 选子进程而非 in-process:规避 Milvus Lite 独占(评估脚本本就用自己的临时库文件,docstring 中「服务已停」为过时防御性备注)、不阻塞事件循环、不污染服务进程的 Milvus 单例。内存态重启即丢,可接受;日志文件持久。

## 三、评估 API(新建 `app/routers/rag_eval.py`)

- `GET /api/rag-eval/report`:读 `evals/results/rag_eval.json` 原样返回;文件不存在 → `200 {"empty": true}`(页面显示「还没跑过」+ 引导重跑)。
- `GET /api/rag-eval/faith-cases?status=&page=&size=`:
  - `status ∈ {未解决,已解决,无需解决}`,缺省未解决;`page` 从 1 起,`size` 默认 10、上限 50。
  - 返回 `{stats, total, page, size, items}`:
    - `stats.judge_rate`:本轮裁判判出率 = rag_eval.json 中 fabricated 题号数 / 本轮生成段判过(非 judge_error)题数。
    - `stats.confirmed_rate`:本轮人工确认幻觉率 = 本轮 fabricated 题号 ∩ faith_cases 中 status=已解决 的数量 / 本轮 fabricated 题号数。本轮题号取自 rag_eval.json;无 rag_eval.json 时两口径为 null。
    - `stats.ledger`:{total, 未解决, 已解决, 无需解决} 全量计数(台账累计那一行)。
  - items 每行含 citations(新形状,带 evidence + cited_refs)、seen_count、first/last_seen_at、status、resolution。
- `PATCH /api/rag-eval/faith-cases/{id}`:body `{status: "已解决"|"无需解决", resolution: str(必填非空)}`,写 `resolved_at`;非法 status / 空 resolution → 422;id 不存在 → 404。
- DB 访问沿用 `make_session_factory` + `asyncio.to_thread`(kb.py 同款模式)。

## 四、页面(`app/static/rag-eval.html`)

- `app/main.py` 加 `/rag-eval` → FileResponse;`chat.html` 页脚与 `kb.html` 返回链接旁各加一处导航。
- 沿用两页同源设计 tokens(Claude/Anthropic 风格 `:root`),vanilla JS + `fetch()` 封装(kb.html 同款),**手绘 SVG** 分组柱状图与圆环,零新依赖、无 CDN。
- 块序:报告头 → 四项 KPI 卡 → 检索质量(三指标切换按钮,4策略×4桶分组柱图,每图下方一句读图,JS 按规则从数据生成,如「hybrid_rerank 在 C_colloquial 桶 MRR 领先 dense 0.12」) → 生成质量(四策略覆盖度柱图、上线管线四桶忠实度柱图、D 桶拒答率圆环) → 完整数据(分桶汇总表,overall.mrr 最佳行带底色;编造个案默认收起,展开显示问题/答案/unsupported_claims) → 台账(两口径 + 累计行、状态页签、分页、每条可展开 evidence 列表并高亮 cited_refs、处置按钮弹出说明输入) → 重跑按钮 + 日志窗口。
- 重跑交互:POST run → 轮询 GET /api/jobs/eval-rag(2s 间隔),日志窗滚动显示 log_tail;终态(ok/failed)后重新拉 report + faith-cases。运行中按钮置灰。

## 五、Makefile

仓库根新增 Makefile,`eval-rag` 目标 = `uv run python evals/run_retrieval_compare.py`,终端与页面同源。README 用法一节补一行。

## 六、测试

- 单测:
  - `_test_metrics` by_bucket 三指标聚合(构造 per-case 数据直算)。
  - `parse_judge_output` 的 coverage 校验(合法/缺失/越界/非数)。
  - rag_eval.json 形状:管道归一化函数喂 `_compare` 中间结构,断言五个顶层键与分桶字段齐全。
  - `upsert_faith_case` 新 citations 形状(改现有 test_faith_cases.py)。
- API 测试(FastAPI TestClient + 测试库,沿用 conftest/dbfixtures 模式):
  - report 空态 / 有数据(临时目录造 rag_eval.json)。
  - faith-cases 分页、status 页签、stats 两口径分子分母。
  - PATCH:正常两状态、空 resolution 422、非法 status 422、不存在 404。
- jobs 测试:以快命令(如 `python -c`)注册临时作业名跑通 runner 状态机(idle→running→ok/failed、日志捕获、409 并发);不真跑评估。
- 页面快照测试:tests/test_rag_eval_page.py,test_kb_page.py 同款(关键块 id / 文案存在性)。

## 七、错误处理

- 评估跑挂:作业 status=failed + exit_code + 日志尾部;report 保持上一份,页面不清空。
- rag_eval.json 与台账不一致(本轮题号缺台账行):统计按交集算,页面正常渲染。
- rag_eval.json 损坏(JSON 解析失败):report 返回 502 + 错误摘要,页面显示错误条。
- 子进程启动失败(uv 不存在等):立即 failed,log_tail 带异常信息。

## 八、非目标(YAGNI)

- 不做多轮历史报告对比/列表(只看最新一轮)。
- 不做作业排队与多作业并发(单作业单并发)。
- 不做台账批量处置、导出。
- 不引入图表库/前端框架。
- 不改动 `_compare.json/.md` 既有产物与门槛逻辑。
