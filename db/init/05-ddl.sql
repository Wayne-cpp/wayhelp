-- =============================================================
-- ch07 · 会话上下文管理 · 建表 DDL(单文件,两步合一)
-- 原计划的两个文件合并为这一份:给 ch02 的 conversations 表加三列
-- (投影摘要、摘要覆盖锚点、层 1 起点锚点),并新建一张分段摘要表,
-- 一次 apply 全部到位。
--
-- 层的边界靠消息 id 表达,不搬数据:
--   id ≤ summary_upto_msg_id          已进摘要
--   summary_upto < id ≤ layer1_from   层 2,渲染成半压形态
--   id > layer1_from                  层 1,原样
--
-- 摘要一段一行、只追加:压完的段落不再回炉重压,一个事实只经历一次有损压缩。
-- 反过来滚动重写的话,第五版摘要就是最早那批内容被压了五次的结果,
-- 订单号哪一次被判成不重要给丢了,事后谁也查不出来。
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary             TEXT            NULL COMMENT '最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后' AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到哪条消息,滑窗从其后接原文' AFTER summary;

ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL
    COMMENT '层1(原文)起点;此 id 之后原样,之前渲染成半压形态'
    AFTER summary_upto_msg_id;

CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT '第几段,从 1 开始',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的消息区间,闭区间',
  upto_msg_id     BIGINT UNSIGNED NOT NULL,
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id),
  CONSTRAINT fk_summary_conv FOREIGN KEY (conversation_id)
    REFERENCES conversations (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要,一段一行只追加';
