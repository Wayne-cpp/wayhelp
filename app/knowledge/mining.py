"""对话挖知识(spec §7):拉会话 → 分批 LLM 抽取 → 整体去重 → 入库向量化。

幂等设计:qa_mining_progress 记成功抽取(含零 QA);staging 记候选与去重结果;
kept 入库按三格精确匹配重放;向量化复用 ingest.vectorize_pending。
"""

import logging
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.knowledge.ingest import vectorize_pending
from app.knowledge.milvus_store import MilvusKnowledgeStore
from app.models import (
    Conversation, KnowledgeChunk, Message, QaExtractionStaging, QaMiningProgress,
)

logger = logging.getLogger(__name__)


class MiningError(Exception):
    pass


class MinedQA(BaseModel):
    question: str
    answer: str


class MinedConversation(BaseModel):
    conversation_id: int
    items: list[MinedQA]


class MiningBatchResult(BaseModel):
    conversations: list[MinedConversation]


_PUNCT_RE = re.compile(r"[\s，。、！？；：,.!?;:'\"“”‘’（）()【】\[\]<>《》—…·~\-]+")

MINING_SYSTEM = """你是客服知识挖掘器。输入是一批客服对话,每段以「会话 <id>」开头。
从每段对话抽取可复用的客服知识问答对(政策、规则、流程、费用类),
不要订单号、用户个人信息等一次性内容;没有可抽取内容的会话返回空 items。
不得合并不同会话的事实;问答必须有对话原文依据,不得编造。
每个输入会话在输出中恰好出现一次,不得遗漏、重复或编造会话 ID。"""


def normalize_question(q: str) -> str:
    return _PUNCT_RE.sub("", q).lower()


def _dialogue_text(s, conversation_id: int) -> str:
    rows = (s.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at, Message.id).all())
    lines = []
    for m in rows:
        if m.role == "user" and m.content and m.content.strip():
            lines.append(f"用户: {m.content.strip()}")
        elif m.role == "assistant" and m.content and m.content.strip():
            lines.append(f"客服: {m.content.strip()}")
    return "\n".join(lines)


def _extract_batch(settings: Settings, session_factory: sessionmaker, model,
                   conv_ids: list[int], batch_no: str) -> None:
    """一批会话:LLM 抽取 → 结构校验 → staging+进度同事务提交。失败抛 MiningError。"""
    with session_factory() as s:
        parts = [f"会话 {cid}\n{_dialogue_text(s, cid)}" for cid in conv_ids]
    human = "\n\n".join(parts)
    messages = [SystemMessage(content=MINING_SYSTEM), HumanMessage(content=human)]
    if count_tokens_approximately(messages) > settings.max_input_tokens:
        if len(conv_ids) == 1:
            raise MiningError(f"单个会话超输入预算: conv {conv_ids[0]}")
        mid = len(conv_ids) // 2  # 超预算缩批:两半各自递归
        _extract_batch(settings, session_factory, model, conv_ids[:mid], batch_no + "a")
        _extract_batch(settings, session_factory, model, conv_ids[mid:], batch_no + "b")
        return
    structured = model.with_structured_output(
        MiningBatchResult, method=settings.structured_output_method, include_raw=True)
    try:
        result = structured.invoke(messages)
    except Exception as exc:
        raise MiningError(f"LLM 抽取失败: {type(exc).__name__}") from exc
    if result.get("parsing_error") is not None or result.get("parsed") is None:
        raise MiningError("structured output 解析失败")
    parsed: MiningBatchResult = result["parsed"]
    got_ids = [c.conversation_id for c in parsed.conversations]
    if sorted(got_ids) != sorted(conv_ids) or len(set(got_ids)) != len(got_ids):
        raise MiningError(f"输出会话集合不符: 期望 {sorted(conv_ids)},实际 {sorted(got_ids)}")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_factory() as s:
        try:
            for conv in parsed.conversations:
                for item in conv.items:
                    q, a = item.question.strip(), item.answer.strip()
                    if not q or not a or not normalize_question(q):
                        raise MiningError(
                            f"空白 question/answer: conv {conv.conversation_id}")
                    s.add(QaExtractionStaging(
                        batch_no=batch_no, source_ref=f"conv:{conv.conversation_id}",
                        question=q, answer=a, status="extracted"))
                s.add(QaMiningProgress(
                    conversation_id=conv.conversation_id, batch_no=batch_no,
                    qa_count=len(conv.items), extracted_at=now))
            s.commit()
        except MiningError:
            s.rollback()
            raise
        except Exception as exc:
            s.rollback()
            raise MiningError(f"批次提交失败: {type(exc).__name__}") from exc


def _dedup_staging(session_factory: sessionmaker) -> None:
    """整体去重:已有知识优先,候选内部留最早 ID(spec §7 步骤 3)。"""
    with session_factory() as s:
        known: set[str] = set()
        for (questions,) in s.query(KnowledgeChunk.questions).all():
            known.update(normalize_question(line) for line in questions.splitlines())
        rows = (s.query(QaExtractionStaging)
                .filter(QaExtractionStaging.status.in_(["extracted", "kept"]))
                .order_by(QaExtractionStaging.id).all())
        changed = False
        for row in rows:
            key = normalize_question(row.question)
            if key in known:
                if row.status != "discarded":
                    row.status = "discarded"
                    changed = True
            else:
                known.add(key)
                if row.status != "kept":
                    row.status = "kept"
                    changed = True
        if changed:
            s.commit()


def _load_kept(session_factory: sessionmaker) -> None:
    """kept → knowledge_chunks(三格精确匹配重放防重);新行 pending。"""
    with session_factory() as s:
        kept = (s.query(QaExtractionStaging).filter_by(status="kept")
                .order_by(QaExtractionStaging.id).all())
        existing = {
            (r.category, r.questions, r.answer)
            for r in s.query(KnowledgeChunk)
            .filter_by(content_type="qa_mined", source_doc=None).all()
        }
        added = False
        for row in kept:
            triple = ("对话挖掘", row.question, row.answer)
            if triple in existing:
                continue
            s.add(KnowledgeChunk(
                category="对话挖掘", questions=row.question, answer=row.answer,
                section_path=None, content_type="qa_mined", is_key_clause=False,
                source_doc=None, chunk_index=None, vectorize_status="pending"))
            existing.add(triple)
            added = True
        if added:
            s.commit()


def run_mining(settings: Settings, session_factory: sessionmaker, model, embed,
               store: MilvusKnowledgeStore) -> int:
    # embedding key 前置校验在 CLI(app/jobs/mine_qa.py)做,与 ingest 模式一致,
    # 库函数不拦(测试以 FakeEmbeddings 注入,无真实 key)。
    failed = False
    store.ensure_collection()
    try:
        # 恢复:历史 staging 的去重/kept 入库/pending 向量化先补齐
        _dedup_staging(session_factory)
        _load_kept(session_factory)
        vectorize_pending(settings, session_factory, embed, store)
    except Exception as exc:
        # 阶段可重入(spec §7):恢复失败不阻断抽取,finalize 阶段会重试,退出码记失败
        failed = True
        logger.warning("mining recover failed: %s: %s", type(exc).__name__, exc)
    with session_factory() as s:
        done_ids = {r[0] for r in s.query(QaMiningProgress.conversation_id).all()}
        conv_ids = [r[0] for r in s.query(Conversation.id).order_by(Conversation.id).all()
                    if r[0] not in done_ids]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seq = 0
    for i in range(0, len(conv_ids), settings.mining_batch_size):
        batch = conv_ids[i:i + settings.mining_batch_size]
        seq += 1
        batch_no = f"{stamp}-{seq:03d}"
        try:
            _extract_batch(settings, session_factory, model, batch, batch_no)
            logger.info("mining batch %s ok: %d conversations", batch_no, len(batch))
        except MiningError as exc:
            failed = True
            logger.warning("mining batch %s failed (convs=%s): %s", batch_no, batch, exc)
    try:
        _dedup_staging(session_factory)
        _load_kept(session_factory)
        vectorize_pending(settings, session_factory, embed, store)
    except Exception as exc:
        failed = True
        logger.warning("mining finalize failed: %s: %s", type(exc).__name__, exc)
    return 1 if failed else 0
