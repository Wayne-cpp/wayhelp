from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.mysql import BIGINT, INTEGER, TINYINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conv_status"), default="进行中"
    )
    # ch07 会话上下文:投影摘要 + 双锚点(层边界靠消息 id 表达,不搬数据)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="msg_role"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    seq: Mapped[int] = mapped_column(Integer)
    from_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    upto_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
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
        BIGINT(unsigned=True), ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type")
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"), default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (UniqueConstraint("source_doc", "chunk_index", name="uk_doc_chunk"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id"), nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id"), nullable=True)
    source_doc: Mapped[str | None] = mapped_column(
        String(255, collation="utf8mb4_bin"), nullable=True)
    chunk_index: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vec_status"), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_status"), default="extracted")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class QaMiningProgress(Base):
    __tablename__ = "qa_mining_progress"

    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    qa_count: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    extracted_at: Mapped[datetime] = mapped_column(DateTime)


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=True)
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback", name="lcq_source"))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    resolved_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_message_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("review_queue.id", ondelete="SET NULL"),
        nullable=True)
    process_status: Mapped[str] = mapped_column(
        Enum("pending", "processed", "failed", name="lcq_process_status"),
        default="pending")
    attempt_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_status"), default="未解决")
    seen_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp", name="tool_source"))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"))
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(TINYINT(unsigned=True), default=0)
    duration_ms: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolWriteIdempotency(Base):
    __tablename__ = "tool_write_idempotency"

    idempotency_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    arguments_sha256: Mapped[str] = mapped_column(String(64))
    ticket_no: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), default=1)
    review_status: Mapped[str] = mapped_column(
        Enum("待审", "写入中", "通过", "驳回", name="review_status"), default="待审")
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    knowledge_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    last_write_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())


class ChatFeedback(Base):
    __tablename__ = "chat_feedback"
    __table_args__ = (UniqueConstraint("conversation_id", "assistant_message_id",
                                       name="uk_cf_conv_msg"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"))
    assistant_message_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("messages.id"))
    turn_message_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("messages.id"))
    sentiment: Mapped[str] = mapped_column(Enum("up", "down", name="fb_sentiment"))
    low_confidence_question_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("low_confidence_questions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    triggered_by: Mapped[str] = mapped_column(
        Enum("定时", "手动", name="eval_trigger"), default="定时")
    dataset_size: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    corpus_mode: Mapped[str] = mapped_column(String(32), default="knowledge_docs_baseline")
    corpus_version: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
