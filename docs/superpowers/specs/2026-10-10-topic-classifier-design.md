# ch10 · 多标签主题分类器 · 设计 Spec

日期:2026-10-10
状态:已评审修订(用户已确认轮次去重、零命中归“其他”、演示验收不设 F1 门槛)

## 1. 背景与目标

飞轮(ch09)持续把低置信度问题攒进 `low_confidence_questions`(lcq)池,但缺少主题分布,看不出哪类知识最缺。本章微调一个**多标签主题分类器**,旁路批量把池里问题归入 17 类权威主题,结果落 `topic_classifications` 表,飞轮后台新增主题分布页,帮助决定先补哪块知识。

**实时对话主链路不调用该分类器。** 分类范围包含 lcq 全部 `process_status`,与飞轮处理、审核状态独立。归类由人工触发,不做自动攒批 worker。

“问题量”按**用户轮次**去重:同时具备 `conversation_id` 和 `turn_message_id` 的 lcq,以二者组成的问题键归组;缺任一锚点的历史记录以自身 `lcq.id` 单独成组。同一轮自动落池后再点 👎 只算一个问题;不同轮次即使文本相同仍各算一个。存储仍按 lcq 行关联,同组各行共享一次推理的分类结果。

## 2. 权威类目表(冻结)

17 类的顺序冻结如下,也是训练输出、分数向量和接口标签排序的共同顺序:

`退换货、物流、尺码、发票、质量问题、运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修、账号、会员积分、评价、其他`。

每类配一句边界说明,近邻类目靠边界划开(用户点名三对):

- 修归保修维修、退归退换货。
- 运费管钱、物流管货。
- 价保是补差价、优惠活动是券和满减。

类目表落 `training/taxonomy.json`(`[{name, boundary}]`),为唯一权威来源。测试钉死类数=17、全部类名及顺序、边界非空。`taxonomy_version` 为该 JSON 规范化后的 SHA-256:对象键排序、数组顺序保留、UTF-8、无多余空白。类名、顺序或边界改动均换版本,需同步数据标注与模型产物。

## 3. 多标签标注与解码口径

一句话**字面提到几个诉求就打几个标签**,一个不多一个不少。仅标用户实际表达的诉求,不从背景描述推导未提出的操作。标签获取走折中路线:

- 合成数据:生成时指定标签组合,LLM 再照类目边界全量预标做 QA;预标与指定标签不一致的进入人工裁决队列。
- 真实数据:LLM 照类目表预标,同一轮只生成一条标注样本。
- 人工抽审:每类随机抽取约 10% 样本导出 Markdown 复核,修正以稳定 `sample_id` 写入 `corrections.jsonl`;裁决完成后再冻结数据版本与切分。

“其他”用于没有明确具体类目诉求的问题,**与其余 16 类互斥**,人工标注与合成数据均遵守此规则。负例需包含库外问题、无明确诉求的表达及容易误触发具体标签的背景描述。

统一解码规则:对 17 维 logits 取 sigmoid,其余 16 类中分数 `>= effective_threshold` 的类目全部命中;有具体类目命中时剔除“其他”,一个都未命中时输出 `[其他]`。“其他”的分数保留在分数向量中,不参与兜底判定。标签按 §2 顺序返回,不重复、不为空。

解码实现集中为轻量纯函数,冻结 `decoder_version=topic-decoder-v1`;训练验证、阈值搜索、评测和服务推理共用,不得各自实现不同口径。

## 4. 数据处理四步

数据记录至少包含 `sample_id`、来源(`synthetic` / `real`)、`source_group_id`、原始/清洗文本和标签;增强样本另记 `parent_id`。同一合成种子及其近似改写、同一真实轮次的记录属于同一个来源组,不得跨切分;不能把整个生成批次或同一类目笼统视作一组。真实池的轮次归组与代表文本选择沿用 §1 / §7.3。

1. **清洗**(`prep_clean.py`):规则脱敏(手机号→`[PHONE]`、邮箱→`[EMAIL]`、订单号→`[ORDER_ID]`)+ 全半角/空白/繁简规范化。确定性清洗由训练和推理共用,冻结规则版本与处理顺序。LLM 批量校对错别字仅用于离线数据准备,校验输入输出的样本 ID 一一对应及条数一致;错位、重复或遗漏时整批保留原文并记日志,改变诉求的改写进入人工裁决。在线推理不调用 LLM 校对。
2. **归并术语与去重**:所有生成/预标 prompt 注入对应类目的边界说明。切分前按清洗后的文本去重,冲突标签先人工裁决;相同文本连接的来源组合并,保留来源追溯,避免跨组泄漏。
3. **分层抽样**(`prep_split.py`):以来源组为不可拆单元做多标签迭代分层,目标训练/验证/测试=80/10/10,随机种子冻结为 42。全量数据的三份均需覆盖 17 类;输出实际比例、逐类 support 和多标签占比,组不可拆造成的比例偏差单列说明。约 50 条冒烟数据仅验证流程,不要求完整类别覆盖。
4. **数据增强**(`prep_augment.py`):**只扩训练集**,LLM 同义词替换 + 句式微调,每条原始训练样本派生 ≤1 条,训练集目标约 1.3 倍。派生样本继承标签、来源组和训练集归属,复核诉求不变;与三份现有文本重复的候选丢弃,不移动验证/测试样本凑数量。

基础数据约 1700 条合成,其中约 20% 为多诉求句,各类覆盖目标约 100 条,按多标签实际 support 记录而不要求恰好相等。加入真实池快照(评审时为 7 条 lcq,按轮次去重后再并入);记录快照时间和去重前后数量,不把“7 条”作为脚本常量。

冻结 `split_manifest.json`:记录样本/来源组的切分归属、内容与标签哈希、种子及 `taxonomy_version`,以规范化内容哈希生成 `dataset_version`。人工修正产生新数据版本,已有来源组的切分归属保持不变;新增样本若关联已有来源组则沿用该组归属。若修正导致跨集文本重复或来源组合并冲突,阻止新版本冻结,先人工裁决或显式重建 manifest;重建时既有测试来源组仍留在测试集。测试集只用于最终报告,不得据此选择模型或调整阈值。

中间产物落 `data/topic_classifier/`(JSONL/JSON,gitignore)。单独准备的人工边界/多标签/负例演示集及其近似改写不得进入训练或验证集;若真实池演示问题已参与训练,展示时明确注明。

## 5. 训练

- 模型:`hfl/chinese-roberta-wwm-ext`(HF 拉取,必要时 `HF_ENDPOINT=https://hf-mirror.com`),`BertForSequenceClassification` + `problem_type="multi_label_classification"`(sigmoid + BCEWithLogits)。记录实际拉取的模型 revision。
- 全参微调,**不用 LoRA**。
- 超参(CPU 友好):max_len 64、batch 16、lr 2e-5、weight_decay 0.01、epochs ≤10、early stopping patience 2(monitor 验证集 micro-F1)。训练随机种子为 42。
- 训练、评测和推理统一 tokenizer、确定性清洗规则及参数:长度 64 包含特殊 token,右侧截断、右侧 padding 到固定长度 64;记录长文本截断比例。
- 流程:先约 50 条小样本冒烟跑通管线,再全量训练。环境为**本机 CPU 慢训**,用户已接受;原估计 1–3 小时,以冒烟吞吐估算实际时长并记录。
- checkpoint 选择和 early stopping 统一使用阈值 0.5 及 §3 解码,避免训练中同时漂移阈值。选定最佳 checkpoint 后,在验证集以 0.01 步长扫描全局阈值 0.05–0.95,按相同解码后的 micro-F1 选最优;并列时先取最接近 0.5 的阈值,仍并列取较大者。冻结进 `labels.json`,随后才评测留出测试集。

## 6. 评测

留出测试集输出逐类精确率/召回率/F1/support、macro/micro 汇总及**每类 one-vs-rest 混淆矩阵**(TP/FP/FN/TN)。报告记录模型、类目、数据、解码版本、有效阈值、实际切分比例、合成/真实数量、多标签占比和截断比例;FP/FN 错样本导出 Markdown 供人工抽判复核。

报告落 `evals/results/<UTC时间戳>_topic_eval.md`(不进 pytest、不入库,照 evals 惯例)。人工演示集单列结果,覆盖三对近邻边界、多标签、负例与“其他”兜底,不混入留出测试指标。

**教学演示验收不设自动 F1 门槛**,如实报告各类表现及错例;不能用训练集分数替代留出评测,也不能因为未设门槛省略评测和人工复核。

## 7. 部署与归类链路

### 7.1 完整推理产物与版本

`export_onnx.py` 导出完整的 `models/topic_classifier/` 包,`models/` 加入 gitignore:

| 产物 | 内容与约束 |
| --- | --- |
| `model.onnx` | CPU FP32 模型;动态 batch、固定序列长度 64、17 维 logits,服务侧做 sigmoid |
| `tokenizer/` | 用 `save_pretrained` 保存 tokenizer 的完整本地文件;服务仅从包内加载 |
| `taxonomy.json` | 本次模型对应的权威类目表冻结副本 |
| `labels.json` | 类目顺序、`taxonomy_version`、验证集选出的阈值、`decoder_version` |
| `preprocess.json` | 确定性清洗版本/参数、tokenizer 设置、max_len、截断与 padding 方向 |
| `manifest.json` | 各文件哈希、模型 revision、训练参数/种子、数据版本及导出一致性验证记录 |

`model_version` 为除 manifest 自身外全部包文件的相对路径与内容哈希清单的规范化 SHA-256,清单按相对路径排序;包括 tokenizer、类目、阈值和预处理配置。`classifier_version` 为 `{model_version, effective_threshold, decoder_version}` 的规范化 SHA-256,统一使用 64 位十六进制字符串。改模型、类目、预处理、解码规则或有效阈值均产生新分类版本。

导出时将模型设为 eval 模式,对同一输入、同一清洗与 tokenizer 输出比较 PyTorch 和 ONNX:覆盖 batch=1/32、短句/超长句、多标签和脱敏占位符输入;logits 及 sigmoid 分数按 `atol=1e-4, rtol=1e-3` 比较,最终解码标签必须完全一致。不一致则导出失败,不得发布该包。

服务启动校验文件哈希、配置版本、类目顺序/数量及模型输入输出维度;不兼容则启动失败。每个进程固定加载一个模型包和有效阈值,更换需重启,不在请求中热切换。

### 7.2 独立推理服务

`training/serve.py`,FastAPI 独立进程,端口 8103,onnxruntime CPU 推理;Makefile 加 `topic-inference` target,显式用 `uv run --group train` 启动。

- `GET /metadata` → `{model_version, taxonomy_version, classifier_version, effective_threshold, labels, decoder_version}`;`labels` 是 §2 的完整有序类名数组。
- `POST /classify {texts:[...]}` → `{labels:[[...]], scores:[[...]], classifier_version}`;返回结果与输入顺序一一对应,每条分数为 §2 顺序的 17 个有限浮点数且在 `[0,1]` 内,标签遵守 §3。
- 每次接受 1–256 条非空字符串;类型不对、空白文本或确定性清洗后为空时,整批返回 422,不返回部分成功。模型错误整批失败。

`TOPIC_LABEL_THRESHOLD` **仅由推理进程读取**,未设时使用 `labels.json` 冻结值,覆盖值必须为有限数且 `0 < 值 < 1`。覆盖会改变 `effective_threshold` 和 `classifier_version`,启动日志及 metadata 显式展示。主服务读取 metadata 校验、存档,不自行再覆盖阈值。

### 7.3 主服务批量归类与事务

主服务 `POST /api/topics/classify` 经 `asyncio.to_thread` 执行同步归类作业。采用进程内独占、非阻塞作业锁,覆盖读快照、全部推理和最终写入;并发触发返回 409 `job_busy`。锁由工作线程在 `finally` 释放,客户端断连不提前释放仍在执行的作业锁。沿用项目单 Uvicorn worker 约束。

1. 读取推理 metadata,校验版本、阈值及标签契约,且 `taxonomy_version` / 类目顺序必须与主服务的权威类目表一致;将本次 `classifier_version` 固定为目标版本。
2. 在一致性读事务内读取 lcq LEFT JOIN 分类结果的快照(全部处理状态),按 §1 的问题键归组;读完释放数据库事务后再发网络请求。同组所有行均有合法、一致且属于目标版本的结果,并且存储 metadata 与本次 metadata 一致时跳过;有缺失、旧版本、标签或 metadata 不一致时整组重归类。**同版本只补漏,换版本重算快照内全部存量**,不混合版本。
3. 同组按 lcq.id 升序取第一个非空 `resolved_question`,全部为空时取第一个非空 `raw_question`,每组仅推理一次。按 `TOPIC_CLASSIFY_BATCH_SIZE` 分批请求,全部结果暂存内存。每批校验条数、17 维有限分数、合法有序标签、标签与分数按 §3 解码一致,且版本与作业开始时固定的目标版本完全相同。
4. **全部批次推理和校验成功后,才开启唯一的写事务**。按 `question_id` 唯一键插入/更新待归类组的全部 lcq 行,同组写入相同标签、分类版本、metadata 和时间;完整的同版本组保持原结果。一次提交,任何数据库错误整笔回滚,禁止逐批提交。快照之后新增的 lcq 留待下次触发。

成功响应:`{classifier_version, classified_questions, skipped_questions, written_rows}`。前两个计数字段按去重问题组计,`written_rows` 为实际插入/更新的 lcq 关联行数;没有待归类组时为零写入成功。

推理不可达、超时、错误响应、metadata 不兼容或任一批次版本变化 → 503,**本次作业零写入**;第二批失败也不得保留第一批结果。数据库失败 → 500 并回滚本次全部写入。失败后保留上次成功结果供查询,重试重新取 metadata 与快照。内存锁不取代数据库唯一键。

### 7.4 分布统计

`GET /api/topics/distribution` 使用数据库一致性快照,按 §1 的问题键统计,返回:

`{total, classified, unclassified, counts:[{label,count}], classifier_version, classifier_metadata, classified_at, stale}`。

- `total = classified + unclassified`。同组只有全部 lcq 行都有相同合法标签、同一有效存储版本时才计已归类,否则整组计未归类且不进横条图。同组同一标签最多计一次;多标签问题可出现在多个横条中,因此**各类计数之和可以大于已归类问题数**。
- 统计版本取数据库中上次成功持久化的唯一分类版本,metadata 随结果读取;`classified_at` 为该版本结果的最近归类时间。GET 不调用推理服务,推理离线仍能查询上次结果;换模型后须手动归类才切换统计版本。
- 分类表为空时,17 类计数/已归类数均为 0,版本、metadata、时间为 null,`stale=false`;lcq 中的问题组仍计入总数/未归类数。
- 已存结果的 `taxonomy_version` 与当前权威类目表不同,返回 `stale=true`,17 类计数/已归类数为 0、全部问题组计未归类,保留旧版本信息提示重新归类,不把旧边界的统计展示成当前口径。
- 若发现跨行存储版本混杂或同一版本 metadata 冲突,返回 409 `classification_version_conflict`,提示重新归类修复,不得把多版本结果加总。

## 8. 主服务接线(DDL、测试库与薄壳)

- 以现有 `sql/ch10-ddl.sql` 草案为基础补齐分类版本字段,再与 `db/init/08-ddl.sql` 保持逐字节一致;`tests/test_ddl_sync.py` 的 `DDL_PAIRS` 加 `("08","ch10")`。
- `topic_classifications` 保留 `id / question_id / labels JSON / classified_at`,新增非空 `classifier_version CHAR(64)`、`classifier_metadata JSON`。metadata 至少含 `model_version / taxonomy_version / effective_threshold / labels(完整顺序) / decoder_version`,与写入版本相符。保留 `uk_question_id` 唯一键及指向 lcq 的外键;不给 lcq 增列,不改已有删除语义。
- `app/db.py` 加 `check_ch10_tables`(校验表及必需列,缺失则 RuntimeError 提示执行 DDL),在 `main.py:_build_production_runtime` 与既有 ch04/ch07/ch08/ch09 检查并列调用;注入测试 runtime 时不访问生产数据库。
- `app/models.py` 加包含上述字段的 `TopicClassification` ORM。
- `tests/dbfixtures.py` 的 `DDL_PATHS` 加 `08-ddl.sql`,`TABLES` 加 `topic_classifications` 且放在 lcq 之前,纳入建表/drop/truncate 全流程。
- `app/routers/topics.py`:注册两个 API,同步 service 走 `to_thread`,错误状态与 §7 一致。`app/services/topic_classify.py` 封装问题归组、metadata 校验、推理客户端、事务写入与分布统计;锁在实际工作线程内管理。
- 配置集中在 `app/config.py` / `.env.example` ch10 节,采用两行式注释:`TOPIC_INFERENCE_URL`(默认 `http://127.0.0.1:8103`)、`TOPIC_CLASSIFY_BATCH_SIZE`(默认 32,范围 1–256)、`TOPIC_INFERENCE_TIMEOUT_SECONDS`(默认 60,正数,单次 metadata/归类请求超时)。另明确 `TOPIC_LABEL_THRESHOLD` 为推理进程专用可选覆盖,主服务不消费。

## 9. 前端(Vibe Coding 例外)

`app/static/topics.html` 单文件,照 `rag-eval.html` 模式:手写 SVG 横条图,17 类按问题量降序,并列按权威类目顺序;顶部“立即归类”按钮及总数/已归类/未归类三个数字。`main.py` 加 `/topics` 页面路由,现有后台导航增加入口。

页面注明“按用户轮次去重,多标签问题可计入多个类目”,展示分类版本、有效阈值和最近归类时间。归类请求期间禁用按钮;成功后重拉统计,409 忙/503 失败时明确提示并保留上次数据。类目版本过期、版本冲突和空数据各有明确状态,提示手动重新归类。推理服务离线仍可打开历史分布。

前端遵循已确认的 Vibe Coding 例外,采用 `test_chat_page` 式字符串断言与人工点验。

## 10. 测试策略

普通 `uv run pytest` 使用默认 dev 依赖,不安装 `train` 组,不加载真实模型、不访问推理服务或 LLM。纯逻辑用固定样本验证,推理响应用替身,数据库事务走既有测试库。

- 接线:DDL 配对、ORM/启动列校验、测试库建表与清表、两个 API/页面路由及配置范围、页面字符串断言。
- 数据与解码:taxonomy 类数/类名/顺序/边界钉;清洗确定性;具体标签阈值等号、零命中“其他”及互斥;按来源组切分无交叉、清洗文本无跨集重复、固定种子可复现、全量数据三份类别覆盖/实际比例报告、人工修正不漂移既有归属、增强只进训练且保留 parent/标签。
- 幂等与统计:同版本跳过完整组/补漏、版本变化重算、同轮自动落池+👎 共用一次推理/统计、不同轮同文本不合并、缺锚点逐行计数、同组缺结果/标签不一致计未归类、总数恒等式与多标签计数、空表/类目过期/存储版本冲突、推理离线仍可 GET。
- 原子性与并发:第二批失败、条数/维度/标签/分数非法、批次版本变化均零写入;写事务异常全回滚;并发作业 409,成功/失败后释放锁可重试;作业期间新增 lcq 延后处理。

清洗/标注/生成的 LLM 效果、训练质量、真实模型评测和 ONNX 导出以实跑验证替代 TDD:小样本冒烟 → 全量训练 → 留出报告及人工错例复核 → 导出一致性 → 真实问题归类演示。离线 LLM ID 对齐失败等确定性错误处理仍用替身测试,耗时/计费任务不进 pytest。

## 11. 验收映射(用户定)

1. 留出测试集各类 P/R/F1/support、macro/micro 和混淆矩阵报告能跑出,包含版本、切分与错例 → `evaluate.py` 产出 `evals/results/<ts>_topic_eval.md`;按演示定位报告真实表现,不设 F1 达标门槛。
2. 拿真实问题手动归类,后台主题分布页可见 17 类统计 → 调 `/api/topics/classify` 后打开 `/topics`;同轮自动落池+👎 只计一次,多标签跨类计数有说明,重复归类同版本不重复写。
3. “买大了想退”同时命中尺码、退换货 → `curl --noproxy '*'` 请求 `http://127.0.0.1:8103/classify`,按权威顺序返回 `[退换货, 尺码]`;另演示三对近邻边界和负例,无具体类命中时仅返回 `[其他]`。
4. 推理故障及第二批失败零写入、模型/阈值换版本后的存量重算由替身集成测试钉住;真实导出包通过 PyTorch/ONNX 分数容差及标签一致性验证。

## 12. 技术栈与约束(定死,走不通停下问)

- 哈工大讯飞 RoBERTa-wwm-ext,全参 CPU 微调。
- 推理服务化:ONNX / transformers 服务(本章取 ONNX + onnxruntime 独立进程)。
- **不做**:LoRA/QLoRA、embedding 微调、意图识别微调。
- 涉及 FastAPI/SQLAlchemy/transformers/onnxruntime/uv 等库和工具用法,先查 Context7 官方文档再实现。
- Conventional Commits;改 prompt/检索/知识库后跑全量 pytest 再交付。
- 过程留痕 `dev-notes/ch10.md`,每阶段补记用户关键原话、关键产出、纠偏及返工。

## 13. 目录与依赖

```text
training/                  # 新顶层包,包初始化不导入重型训练栈
  taxonomy.json            # 唯一权威类目表
  labels.py                # 轻量类目/版本校验与统一解码,只依赖标准库
  preprocessing.py         # 训练/推理共用的确定性清洗
  prep_clean.py / prep_generate.py / prep_label.py / prep_split.py / prep_augment.py
  train.py / evaluate.py / export_onnx.py / serve.py
data/topic_classifier/     # 数据、corrections、split_manifest 等产物(gitignore)
models/topic_classifier/   # 完整模型包(gitignore)
```

新增 uv dependency-group `topic-data` 放可单测的数据处理依赖(numpy、scikit-learn、scikit-multilearn、繁简转换用 OpenCC 实现),`dev` 和 `train` 通过 `include-group` 共用。`train` 另含 torch、transformers、datasets、onnx、onnxruntime、optimum[onnxruntime],由锁文件固定实际兼容版本。

普通 `uv sync` / `uv run pytest` 可导入切分、清洗和解码模块,不安装 torch/transformers/onnxruntime。主服务的类目校验/解码只依赖标准库,不通过包初始化间接引入训练或数据科学库,主服务生产依赖不变。训练、评测、导出和推理服务显式用 `uv run --group train` 手动运行;耗时/计费脚本不进 pytest。
