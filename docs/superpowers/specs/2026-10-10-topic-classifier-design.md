# ch10 · 多标签主题分类器 · 设计 Spec

日期:2026-10-10
状态:已评审定稿(brainstorm 三轮分段确认通过)

## 1. 背景与目标

飞轮(ch09)持续把低置信度问题攒进 `low_confidence_questions`(lcq)池,但池子只有「待审/已审」维度,看不出问题按主题的分布——不知道哪类知识最缺。本章微调一个**多标签主题分类器**,旁路批量把池里问题归入 17 类权威主题,结果落 `topic_classifications` 表,飞轮后台新增主题分布页,一眼看出哪类堆得多、先补哪块知识。

**实时对话主链路不调用该分类器。**

## 2. 权威类目表(冻结)

17 类,领头四类:退换货、物流、尺码、发票;其余:质量问题、运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修、账号、会员积分、评价、其他。

每类配一句边界说明,近邻类目靠边界划开(用户点名三对):
- 修归保修维修、退归退换货
- 运费管钱、物流管货
- 价保是补差价、优惠活动是券和满减

类目表落 `training/taxonomy.json`(`[{name, boundary}]`),为唯一权威来源,测试钉死类数=17 与全部类名;改动 = 换版本。

## 3. 多标签标注口径

一句话**字面提到几个诉求就打几个标签**,一个不多一个不少。标签获取走折中路线:

- 合成数据:生成时即指定标签组合(意图标签),LLM 再照术语表全量预标做 QA——预标与意图不一致的进人工裁决队列
- 真实数据:LLM 照术语表预标
- 人工抽审:每类随机 ~10% 导出 markdown 人工过一遍,修正写 `corrections.jsonl` 回灌重切分

## 4. 数据处理四步

1. **清洗**(`prep_clean.py`):规则脱敏(手机号→`[PHONE]`、邮箱→`[EMAIL]`、订单号→`[ORDER_ID]`)+ 全半角/空白/繁简规范化;LLM 批量校对错别字(输入输出条数对齐校验,对不上的原样保留并记日志)
2. **归并术语表**:见 §2,所有生成/预标 prompt 都注入对应类目的边界说明防串类
3. **分层抽样**(`prep_split.py`):多标签迭代分层(iterative stratification)80/10/10,各类别按比例进训练/验证/测试三份;切分种子冻结
4. **数据增强**(`prep_augment.py`):**只扩训练集**,LLM 同义词替换 + 句式微调,每条原始训练样本派生 ≤1 条、标签继承,训练集扩到 ~1.3 倍;验证/测试集绝不动

数据规模:每类 ~100 条合成,共 ~1700 条 + 真实池 7 条清洗并入。中间产物落 `data/topic_classifier/`(jsonl,gitignore)。

## 5. 训练

- 模型:`hfl/chinese-roberta-wwm-ext`(HF 拉取,必要时 `HF_ENDPOINT=https://hf-mirror.com`),`BertForSequenceClassification` + `problem_type="multi_label_classification"`(sigmoid + BCEWithLogits)
- 全参微调,**不用 LoRA**
- 超参(CPU 友好):max_len 64、batch 16、lr 2e-5、weight_decay 0.01、epochs ≤10、early stopping patience 2(monitor 验证集 micro-F1)
- 流程:先 ~50 条小样本冒烟跑通管线,再全量训练
- 阈值:默认 0.5,训练后在验证集上扫全局最优阈值,冻结进 `labels.json`
- 环境:**本机 CPU 慢训**(无 GPU,用户已确认接受,估计 1-3 小时)

## 6. 评测

留出测试集逐类精确率/召回率/F1 + macro/micro 汇总 + **每类 one-vs-rest 混淆矩阵**,报告落 `evals/results/<UTC时间戳>_topic_eval.md`(不进 pytest、不入库,照 evals 惯例);FP/FN 错样本导出 markdown 供人工抽判复核。

## 7. 部署与归类链路

- **导出**:`export_onnx.py` → `models/topic_classifier/model.onnx` + `labels.json`(类目顺序+阈值),models/ gitignore
- **推理服务**:`training/serve.py`,FastAPI 独立进程(端口 8103,Makefile 加 `topic-inference` target),`POST /classify {texts:[...]}` → `{labels:[[...]], scores:[[...]]}`;onnxruntime CPU 推理
- **归类服务**(主服务内,薄):`POST /api/topics/classify` → `asyncio.to_thread` → lcq LEFT JOIN topic_classifications 捞未归类行(全量,不分 process_status;分类文本取 `resolved_question`,为空回退 `raw_question`)→ 分批 POST 推理服务 → 事务写 `topic_classifications`;`uk_question_id` 唯一键幂等,重复触发只补漏;推理服务不可达 → 503 且零写入
- **分布统计**:`GET /api/topics/distribution` → 每类计数 + 总数/已归类/未归类数
- 触发方式:**手动 API 触发**(不做自动攒批 worker)

## 8. 主服务接线(DDL 三件套 + 薄壳)

- `db/init/08-ddl.sql` 与 `sql/ch10-ddl.sql` 逐字节一致;`tests/test_ddl_sync.py` 的 `DDL_PAIRS` 加 `("08","ch10")`
- `app/db.py` 加 `check_ch10_tables`(SHOW TABLES/COLUMNS,缺则 RuntimeError 提示执行 DDL),`main.py` lifespan 调用
- `app/models.py` 加 `TopicClassification` ORM(id / question_id FK / labels JSON / classified_at)
- `app/routers/topics.py`:`POST /api/topics/classify`、`GET /api/topics/distribution`,薄壳走 `to_thread`
- `app/services/topic_classify.py`:捞未归类 / 调推理 / 写表
- 配置(`.env.example` 增 ch10 节,两行式注释):`TOPIC_INFERENCE_URL`(默认 `http://127.0.0.1:8103`)、`TOPIC_CLASSIFY_BATCH_SIZE`(默认 32)、`TOPIC_LABEL_THRESHOLD` 说明(阈值以 labels.json 冻结值为准,此处仅覆盖用)

## 9. 前端(Vibe Coding 例外)

`app/static/topics.html` 单文件,照 `rag-eval.html` 模式:手写 SVG 横条图,17 类按问题量降序;顶部「立即归类」按钮 + 总数/已归类/未归类三个数字;`main.py` 加 `/topics` 页面路由。不套 brainstorm/TDD/code review,用户描述效果直接改;仅 `test_chat_page` 式字符串断言兜底。

## 10. 测试策略

- pytest(照常):DDL 配对、启动校验、taxonomy 冻结钉(类数/类名/边界非空)、分层抽样比例与三份无交集、增强只进训练集、归类服务幂等/只捞未归类/推理故障零写入、topics API 契约、页面字符串断言
- 非单测部分(清洗/生成/标注/训练/评测/导出)按约定以实跑验证替代 TDD:小样本冒烟 → 全量训练 → 评估集报告 → 真实问题归类演示

## 11. 验收映射(用户定)

1. 测试集各类目 F1 和混淆矩阵报告能跑出来 → `evaluate.py` 产出 `evals/results/<ts>_topic_eval.md`
2. 拿真实问题跑归类,后台主题分布页能看到各类目统计 → `/api/topics/classify` 后打开 `/topics`
3. 「买大了想退」类多诉求句同时命中多个类目 → `curl :8103/classify` 演示命中 `[尺码, 退换货]`

## 12. 技术栈与约束(定死,走不通停下问)

- 哈工大讯飞 RoBERTa-wwm-ext,全参微调
- 推理服务化:ONNX / transformers 服务(本章取 ONNX + onnxruntime)
- **不做**:LoRA/QLoRA、embedding 微调、意图识别微调
- 涉及 FastAPI/SQLAlchemy/transformers/onnxruntime 等库用法,一律先查 Context7 官方文档再动手
- Conventional Commits;改 prompt/检索/知识库后跑全量 pytest 再交付
- 过程留痕 `dev-notes/ch10.md`,每阶段补记(用户关键原话 / 关键产出 / 纠偏 / 翻车)

## 13. 目录与依赖

```
training/                  # 新顶层包
  taxonomy.json            # 冻结类目表(纳版本)
  prep_clean.py / prep_generate.py / prep_label.py / prep_split.py / prep_augment.py
  train.py / evaluate.py / export_onnx.py / serve.py
data/topic_classifier/     # 数据产物(gitignore)
models/topic_classifier/   # 模型产物(gitignore)
```

依赖:新增 uv dependency-group `train`(torch、transformers、datasets、scikit-learn、scikit-multilearn、onnx、onnxruntime、optimum[onnxruntime]),主服务依赖不变。训练/评测/导出脚本手跑不进 pytest(烧额度/耗时长,照 probe 惯例)。
