-- =============================================================
-- ch03 · RAG 基础 · 建表 DDL(2026-09-11 按 spec §4.5 修订)
-- 本章新建三张表:knowledge_chunks(知识库原文权威源)
--   qa_extraction_staging(挖 QA 暂存)/ qa_mining_progress(抽取进度)
-- 向量落 Milvus Lite 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态
-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量
-- 修订:文档块按 source_doc + chunk_index 唯一键复用 ID(uk_doc_chunk);
--   挖掘 QA 两字段为 NULL(MySQL 唯一索引允许多行 NULL);source_doc 大小写敏感(utf8mb4_bin)
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual / qa_mined,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,仅同文档相邻,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,仅同文档相邻,元数据',
  source_doc       VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT '文档来源标识:仓库内相对/仓库外绝对 POSIX 路径,大小写敏感;挖掘 QA 为 NULL',
  chunk_index      INT UNSIGNED    NULL                    COMMENT '同文档从 1 开始的连续序号;挖掘 QA 为 NULL',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_doc_chunk (source_doc, chunk_index),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话标识,形如 conv:<id>,由程序填充,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可人工清空';

CREATE TABLE qa_mining_progress (
  conversation_id BIGINT UNSIGNED NOT NULL COMMENT '成功抽取的会话 ID,与 conversations.id 对应',
  batch_no        VARCHAR(64)     NOT NULL COMMENT '成功抽取所属批次',
  qa_count        INT UNSIGNED    NOT NULL COMMENT '去重前抽出的 QA 数量,允许为 0',
  extracted_at    DATETIME        NOT NULL COMMENT '程序填入的 UTC 抽取成功时间',
  PRIMARY KEY (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话挖知识的独立抽取进度:一行=该会话抽取结果已成功提交,重跑跳过(含零 QA)';
