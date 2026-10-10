"""数据管线公共:记录 schema、jsonl IO、LLM 客户端、真实池读取。"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

DATA_DIR = Path("data/topic_classifier")


def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_chat_model(settings):
    from langchain_openai import ChatOpenAI  # 主依赖;重导入延迟到调用点
    return ChatOpenAI(model=settings.model_name,
                      api_key=settings.openai_api_key,
                      base_url=settings.openai_base_url, temperature=0.7)


def make_record(sample_id, source, group_id, text_raw, text, labels,
                intended_labels=None, parent_id=None) -> dict:
    return {"sample_id": sample_id, "source": source,
            "source_group_id": group_id, "text_raw": text_raw, "text": text,
            "labels": list(labels), "intended_labels": intended_labels,
            "parent_id": parent_id}


def load_real_questions(session_factory) -> list[dict]:
    """lcq 全量(不分 process_status);双锚点齐归 turn 组,缺一自成组(spec §1/§4)。"""
    from app.models import LowConfidenceQuestion
    with session_factory() as s:
        rows = s.execute(select(LowConfidenceQuestion)
                         .order_by(LowConfidenceQuestion.id)).scalars().all()
        s.expunge_all()
    records = []
    for r in rows:
        if r.conversation_id is not None and r.turn_message_id is not None:
            gid = f"turn-{r.conversation_id}-{r.turn_message_id}"
        else:
            gid = f"row-{r.id}"
        text = (r.resolved_question or "").strip() or (r.raw_question or "").strip()
        records.append(make_record(f"lcq-{r.id}", "real", gid, text, text, []))
    return records
