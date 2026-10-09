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
