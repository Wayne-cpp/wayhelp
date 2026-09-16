# /rag-eval 评估页设计(2026-09-16,用户已批准)

## 背景与目标

RAG 评估目前只在终端跑:`uv run python evals/run_retrieval_compare.py`,读 stdout 或生成的 `{ts}_compare.md`。本任务把报告落成后台一页 `/rag-eval`,并支持页面上就地重跑。

页面块与数据来源(用户规格,逐字确认):

| 页面块 | 内容 | 数据来源 |
|---|---|---|
| 报告头 | 评估集总题数 / 校准题数 / 测试题数、知识库块数、嵌入与重排模型、裁判模型、上次跑的时间 | `rag_eval.json` 的 `meta` |
| 四项 KPI | 最佳整体 MRR、口语桶 MRR 提升、答案覆盖度、库外拒答率 | 同一份产物,不另算 |
| 检索质量 | 四策略 × 四个可作答桶分组柱状图,MRR / Recall@5 / 证据覆盖度三个指标切着看,每张图配一句读图 | `retrieval` |
| 生成质量 | 四策略答案覆盖度柱状图、上线管线四桶已回答样本忠实率与回答率、库外拒答率圆环 | `generation` |
| 完整数据 | 分桶 MRR 与两项覆盖度汇总表,总体 MRR 最佳行带底色;编造个案默认收起,展开逐条看问题、生成答案、裁判理由 | `generation.faithfulness_cases` |
| 编造个案台账 | 顶上两个口径幻觉率 + 台账累计一行;跨轮累计个案:状态页签 + 分页,每条带本轮 Top-K 证据全集(标出答案引用哪几条)与「已解决 / 无需解决」按钮,处置必填一句说明 | `faith_cases` 表(`/api/rag-eval/faith-cases`) |
| 就地重跑 | 「重跑 RAG 评估」按钮 + 日志窗口,跑完自动重新取数 | `/api/jobs`(作业名 `eval-rag`) |

## 已确认口径(用户逐项拍板)

1. **答案覆盖度**:生成段从只跑 hybrid_rerank 扩到**四策略**,裁判在忠实性之外按 `expect_points` 打答案覆盖分(0–1)。代价:每轮评估 LLM 调用约 ×4,用户已知情。
2. **证据覆盖度** = CompleteHit@10(期望证据组被 Top-10 100% 覆盖才计 1,按桶平均)。
3. **人工确认幻觉率**:「已解决」= 确认幻觉;「无需解决」= 误报/可接受。复用现有 status 枚举,不改表结构。
4. **产物形态**:管道产出规范化 `evals/results/rag_eval.json`,页面 API 只读文件 + 查库,不做即时派生。

### 生成指标口径

所有生成指标只用 test 分片。每个策略的 `answerable_total` 是 A/B/C/E 四个可作答桶的 test 题数,D 桶单独统计拒答。

- `answer_coverage` = 可作答题的 coverage 总分 / `answerable_total`;低置信硬拒答或模型自评拒答的 coverage 记 0,避免高拒答策略因跳过难题得到虚高分。
- `by_bucket.<bucket>.coverage` 同上,分母是该可作答桶的全部 test 题数。
- `faithful_rate` = `faithful / judged`,只表示已经生成且裁判成功的答案中有多少忠实;拒答不进该分母。`judged=0` 时为 null。页面同时展示 `answered / answerable_total`,不得脱离回答率单独解释忠实率。
- `by_bucket.<bucket>.faithful_rate` 使用该桶相同口径。
- `d_refuse_rate` = D 桶中输出固定拒答话术的题数 / `d_total`;无论拒答来自低置信硬闸还是模型自评都计入。
- 任一裁判失败时,受影响策略的 `answer_coverage` 与 `faithful_rate` 为 `null`;受影响桶的对应聚合也为 `null`,同时保留 `judge_errors`、`judged` 等计数。不得用剩余成功样本静默缩小分母。硬门槛仍因此不通过。
- 每个策略必须输出 `answerable_total`、`answered`、`answerable_refused`、`judged`、`faithful`、`fabricated`、`judge_errors`、`d_total`、`d_refused`,使所有比率的分子分母可复核。
- 计数必须满足:`answered + answerable_refused == answerable_total`、`judged + judge_errors == answered`、`faithful + fabricated == judged`、`d_refused <= d_total`;规范化产物发布前校验这些不变量。

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
- `_simulate_second_turn` 工具出参 `effective_strategy` 使用该次检索结果的实际 `effective_strategy`(现硬编码 hybrid_rerank);例如请求 hybrid_rerank 但 rerank 降级时传 hybrid。每个策略的 `degraded` = test 分片中 `requested_strategy != effective_strategy` 的题数。
- D 桶 case 照常走低置信→`REFUSAL_ANSWER` 路径;非拒答且非 should_refuse 的 case 交裁判。
- **只有 hybrid_rerank 的 fabricated 写 `faith_cases` 台账**(该表 `strategy` 字段默认 'hybrid_rerank' 本就是这个语义);其余策略的 fabricated 只进 JSON 统计。
- 现有硬门槛(`judge_error == 0`、bm25 B 桶不设闸 SR@10 ≥ 0.5)语义不变;judge_error 统计扩为全策略合计。
- 生成段先在内存中完成四策略结果与规范化报告,期间不逐 case 提交台账。报告校验通过后,在一个 DB 事务中批量 upsert 本轮 hybrid_rerank fabricated case;DB 提交成功后才发布固定报告。若事务失败,不发布本轮报告。

### 裁判(`app/prompts/faithfulness.py`)
- `JUDGE_PROMPT` 增加入参 `expect_points`,输出 JSON 增加 `coverage` 字段(0–1 浮点,答案对标准要点的覆盖度)。
- coverage 逐要点二值计分:答案明确覆盖一个 `expect_points` 要点记 1,否则记 0,最终为覆盖要点数 / 要点总数;同义表达算覆盖,回答中额外的正确信息不加分。D 桶没有标准要点且不送裁判。
- `parse_judge_output` 要求 coverage 是非 bool 的有限数值且 ∈ [0,1];缺失、字符串、NaN/Infinity 或越界同现有口径计入重试/判 judge_error。
- verdict 结构:`{verdict, unsupported_claims, cited_refs, coverage}`。

### `faith_cases.citations` 载荷改形
现状存的是裸 evidence 列表;新写入形状改为 `{"run_id": "...", "evidence": [...], "cited_refs": [...]}`,让「哪一轮、手里有哪些证据、答案引用了哪几条」跨轮留存。`run_id` 对应报告 `meta.run_id`。JSON 列无 DDL 变更;需同步改 `upsert_faith_case` 调用处、`tests/test_faith_cases.py`、两份 DDL 的列注释及后续 API 读取方。

读取方兼容历史数据:若 `citations` 是裸列表,规范化为 `{"run_id": null, "evidence": citations, "cited_refs": []}`;若为 `null`,规范化为空 evidence。API 始终只返回新对象形状,页面不处理多版本。

### 新产物 `evals/results/rag_eval.json`
固定文件名 = 最新一轮,与时间戳 `_compare.json/.md` 并存(后两者照旧)。`run_id` 表示本轮运行,题目标识统一叫 `case_id`;不再用 `eval_id` 同时表示两者。规范化产物固定四个顶层键:`meta`、`retrieval`、`generation`、`gates`。形状:

```json
{
  "meta": {
    "run_id": "20260916T024957123456Z", "ts": "...", "elapsed_s": 0,
    "total_cases": 300, "calibration_cases": 150, "test_cases": 150,
    "answerable_test_cases": 120, "d_test_cases": 30, "corpus_chunks": 0,
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
    "dense": {
      "answerable_total": 120, "answered": 118, "answerable_refused": 2,
      "judged": 118, "faithful": 112, "fabricated": 6, "judge_errors": 0,
      "answer_coverage": 0.82, "faithful_rate": 0.949,
      "d_total": 30, "d_refused": 29, "d_refuse_rate": 0.967,
      "degraded": 0,
      "by_bucket": {"A_policy": {
        "total": 30, "answered": 30, "refused": 0, "judged": 30,
        "faithful": 29, "fabricated": 1, "judge_errors": 0,
        "coverage": 0.91, "faithful_rate": 0.967
      }, "...": {}}
    },
    "bm25": {}, "hybrid": {}, "hybrid_rerank": {},
    "faithfulness_cases": [
      {"case_id": "A12", "bucket": "A_policy", "query": "...", "answer": "...",
       "strategy": "hybrid_rerank",
       "unsupported_claims": [{"claim": "...", "reason": "..."}],
       "cited_refs": [1],
       "evidence": [{"ref_no": 1, "chunk_id": 0, "section_path": "...",
                     "question": "...", "answer": "...", "category": "..."}]}
    ]
  },
  "gates": {
    "judge_error_zero": true,
    "bm25_b_model_section_recall_at_10": 1.0,
    "bm25_b_model_gate": true,
    "passed": true
  }
}
```

`faithfulness_cases` 只收录上线管线被判 fabricated 的 case(与台账同源);`evidence` 为该轮 Top-K 全集(`Evidence.to_dict()` 原样,`ref_no/chunk_id/section_path/question/answer/category`),`cited_refs` 标的是 `ref_no`。

`run_id` 在脚本启动时生成一次,格式为含微秒的 UTC 紧凑时间戳,保证连续运行也不重复;`meta.ts` 是本轮完成时的 ISO 8601 UTC 时间。所有逐 case 记录、台账 citations 和作业结果复用同一个 `run_id`。

`rag_eval.json` 是发布产物:先在同目录写临时文件,完成 JSON 序列化与形状校验后用 `os.replace` 原子替换,且它是本轮最后发布的文件。成功或失败都清理临时文件;任何更早步骤失败都保留上一份。时间戳 `_compare.json/.md` 继续按既有格式生成;`gates.passed=false` 仍发布完整新报告并保留脚本既有非零退出语义。

## 二、作业层(新建)

- `app/jobs/runner.py`:应用级内存作业注册表,用同一把 `asyncio.Lock` 原子完成「检查是否运行 + 登记任务」,避免两个并发 POST 同时起进程。`run("eval-rag")` 起 asyncio 子进程 `uv run python evals/run_retrieval_compare.py`(cwd=仓库根,继承环境变量拿 API key),stdout/stderr 落 `data/jobs/eval-rag.log`(每轮截断)。命令表可在测试构造 runner 时注入,生产只注册白名单作业名,路由不接受任意命令。
- `create_app` 创建一个 `JobRunner` 并挂到 `app.state.job_runner`,同时把固定报告路径挂到 `app.state.rag_eval_report_path`;路由只从 app.state 取这两个依赖。测试在发请求前替换为临时 runner / 临时路径,不读取或改写仓库真实报告。应用 lifespan 关闭时调用 runner 的异步 `close()`:若子进程仍运行,先 terminate 并等待短超时,再 kill,防止服务退出后遗留无人管理的评估进程。
- `app/routers/jobs.py`:
  - `POST /api/jobs/eval-rag/run` → `{started: true}`;运行中 → 409。
  - `GET /api/jobs/eval-rag` → `{status: idle|running|ok|failed, started_at, finished_at, exit_code, quality_passed, report_run_id, log_tail}`(log_tail = 日志尾部约 8KB)。尚无已完成作业或作业仍运行时,尚未确定的时间、exit_code、quality_passed、report_run_id 均为 null。
- runner 启动前记住旧 `meta.run_id`;子进程结束后读取并校验固定报告。若报告存在新的 `run_id`,本轮就是 `ok`,`quality_passed` 取 `gates.passed`,即使脚本因质量门槛未过退出 1;若没有新的有效报告,本轮才是 `failed`,`quality_passed=null`。`exit_code` 始终保留供诊断。
- 选子进程而非 in-process:规避 Milvus Lite 独占(评估脚本本就用自己的临时库文件,docstring 中「服务已停」为过时防御性备注)、不阻塞事件循环、不污染服务进程的 Milvus 单例。内存态重启即丢,可接受;日志文件持久。该互斥只保证单进程;项目继续遵守现有单 Uvicorn worker 启动契约,README 明确不得为该应用配置多 worker。

## 三、评估 API(新建 `app/routers/rag_eval.py`)

- `GET /api/rag-eval/report`:读 `evals/results/rag_eval.json` 原样返回;文件不存在 → `200 {"empty": true}`(页面显示「还没跑过」+ 引导重跑)。
- `GET /api/rag-eval/faith-cases?status=&page=&size=`:
  - `status ∈ {未解决,已解决,无需解决}`,缺省未解决;`page` 从 1 起,`size` 默认 10、范围 1–50。分页固定按 `last_seen_at DESC, id DESC`,避免同一时间的记录跨页漂移。
  - 返回 `{stats, total, page, size, items}`:
    - `stats.judge_rate`:上线管线本轮裁判判幻觉率 = `generation.hybrid_rerank.fabricated / generation.hybrid_rerank.judged`;该策略有 judge_error 或 `judged=0` 时为 null。
    - `stats.confirmed_rate`:本轮人工确认幻觉率 = 本轮 `faithfulness_cases` 中、台账 `status=已解决` 且 `citations.run_id == meta.run_id` 的题数 / 本轮 `faithfulness_cases` 题数。分母为 0 时为 null。
    - `stats.current_run_missing`:本轮 `faithfulness_cases` 中找不到同题且同 `run_id` 台账快照的数量;大于 0 时 `confirmed_rate` 为 null,不输出残缺口径的比率。
    - `stats.ledger`:{total, 未解决, 已解决, 无需解决} 全量计数(台账累计那一行)。
    - 无 rag_eval.json 时 `judge_rate`、`confirmed_rate` 为 null,`current_run_missing` 为 0,累计台账仍正常返回。
  - items 每行用 `id` 表示数据库数值主键,用 `case_id` 表示题号(映射现有列 `eval_id`),另含规范化后的 citations、seen_count、first/last_seen_at、status、resolution。
- `PATCH /api/rag-eval/faith-cases/{id}` 中 `{id}` 明确是数据库数值主键。body 为 `{status: "已解决"|"无需解决", resolution: str}`,后端对 resolution 去首尾空白后要求 1–300 字符并保存去白值,同时写 UTC `resolved_at`;非法 status / 空白 resolution / 超长 → 422,id 不存在 → 404。成功返回更新后的 item。
- DB 访问沿用 `make_session_factory` + `asyncio.to_thread`(kb.py 同款模式)。
- report 与 stats 共用同一个报告加载/契约校验函数。固定报告损坏时,report 端点返回 502;faith-cases 仍返回列表与累计台账,但按“无当前报告”返回本轮 stats,使台账处置不被报告文件阻断。

## 四、页面(`app/static/rag-eval.html`)

- `app/main.py` 加 `/rag-eval` → FileResponse;`chat.html` 页脚与 `kb.html` 返回链接旁各加一处导航。
- 沿用两页同源设计 tokens(Claude/Anthropic 风格 `:root`),vanilla JS + `fetch()` 封装(kb.html 同款),**手绘 SVG** 分组柱状图与圆环,零新依赖、无 CDN。
- 块序:报告头 → 质量门槛状态 → 四项 KPI 卡 → 检索质量(三指标切换按钮,4策略×4桶分组柱图,每图下方一句读图,JS 按规则从数据生成,如「hybrid_rerank 在 C_colloquial 桶 MRR 领先 dense 0.12」) → 生成质量(四策略覆盖度柱图、上线管线四桶已回答样本忠实率并就近显示 answered/answerable_total、D 桶拒答率圆环) → 完整数据(分桶汇总表,overall.mrr 最佳行带底色;编造个案默认收起,展开显示问题/答案/unsupported_claims) → 台账(两口径 + 累计行、状态页签、分页、每条可展开 evidence 列表并高亮 cited_refs、处置按钮弹出说明输入) → 重跑按钮 + 日志窗口。指标为 null 时显示「不可用」,图表留空并给出 judge_error 提示,不得绘制为 0。
- 重跑交互:POST run → 轮询 GET /api/jobs/eval-rag(2s 间隔),日志窗滚动显示 log_tail;终态后重新拉 report + faith-cases。`ok + quality_passed=false` 显示「评估完成,质量门槛未通过」并照常展示新报告;`failed` 显示运行失败且保留页面上的上一份报告。运行中按钮置灰。
- 问题、答案、裁判理由、证据、日志、处置说明等外部字符串只通过 `textContent` / `createTextNode` 写入 DOM;SVG 也通过 DOM API 创建。页面不得把 API 字符串拼入 `innerHTML`、属性 HTML 或事件处理器,防止报告或数据库内容形成存储型脚本注入。

## 五、Makefile

仓库根新增 Makefile,`eval-rag` 目标 = `uv run python evals/run_retrieval_compare.py`,终端与页面同源。README 用法一节补一行。

## 六、测试

- 单测:
  - `_test_metrics` by_bucket 三指标聚合(构造 per-case 数据直算)。
  - `parse_judge_output` 的 coverage 校验(合法/缺失/越界/非数)。
  - 生成聚合口径:硬拒答/模型拒答 coverage 均按 0;`faithful_rate` 分母只取成功裁判;judge_error 使策略及对应桶比率为 null;上述计数不变量全部成立。
  - 四策略生成都使用各自阈值;hybrid_rerank 降级时模拟工具载荷使用实际 `effective_strategy=hybrid`。
  - rag_eval.json 形状:管道归一化函数喂 `_compare` 中间结构,断言四个顶层键、meta 题数、gates 与分桶字段齐全。
  - 固定报告原子发布:成功替换旧报告;临时写入/校验失败时旧报告字节不变且不遗留被 API 读取的半文件。
  - `upsert_faith_case` 在单事务中批量写入新 citations 形状;中途异常整批回滚。旧裸列表/null citations 的规范化读取有测试;两份 DDL 注释继续由既有同步测试约束。
- API 测试(FastAPI TestClient + 测试库,沿用 conftest/dbfixtures 模式):
  - report 空态 / 有数据(临时目录造 rag_eval.json)。
  - faith-cases 三个 status 页签、稳定排序与分页边界;items 同时返回数值 `id` 和 `case_id`。
  - stats 两口径的分子分母、零分母、judge_error、run_id 匹配与 `current_run_missing`。不匹配时 confirmed_rate 必须为 null。
  - PATCH:正常两状态及返回体、resolution 去白、空白/超长 422、非法 status 422、不存在 404。
- jobs 测试:以注入的快命令注册临时作业名,覆盖 idle→running→ok/failed、日志捕获和并发 POST 只有一个成功;新有效报告 + exit 1 映射为 `ok/quality_passed=false`,exit 1 但无新报告映射为 failed;不真跑评估。
- 页面快照测试:`tests/test_rag_eval_page.py` 断言关键块、五个请求钩子、null 展示分支和无动态 `innerHTML` 写入;`test_chat_page.py` / `test_kb_page.py` 断言新增导航。人工用包含 `<img onerror=...>` 的合成报告确认页面只显示文本。

## 七、错误处理

- 质量门槛未通过但新报告已原子发布:作业 status=ok、quality_passed=false、保留 exit_code;页面展示本轮报告与未通过项。
- 评估执行失败或未产出新的有效报告:作业 status=failed + exit_code + 日志尾部;report 保持上一份,页面不清空。
- DB 事务成功但进程在固定报告替换前异常时,台账可能领先于仍在展示的上一份报告。API 只用 citations.run_id 与报告 run_id 一致的行计算本轮人工确认率;缺失时返回 `current_run_missing` 且 confirmed_rate=null,累计台账与列表仍正常渲染。
- rag_eval.json 损坏(JSON 解析失败或顶层契约不合法):report 返回 502 + 不含文件内容/路径的错误摘要,页面显示错误条;runner 不把它视为本轮成功产物。
- 子进程启动失败(uv 不存在等):立即 failed,log_tail 带异常信息。

## 八、非目标(YAGNI)

- 不做多轮历史报告对比/列表(只看最新一轮)。
- 不做作业排队与多作业并发(单作业单并发)。
- 不做台账批量处置、导出。
- 不引入图表库/前端框架。
- 不删除或重命名 `_compare.json/.md` 的既有字段,不改变两项门槛的判定公式;`judge_error == 0` 的输入按本设计扩为四策略合计。
