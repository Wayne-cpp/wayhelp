import re
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Intent(str, Enum):
    RETURN_REFUND = "退货退款"
    EXCHANGE = "换货"
    REFUND_ONLY = "仅退款"
    REPAIR = "维修"
    URGE_SHIPPING = "催发货"
    COMPLAINT = "投诉"
    OTHER = "其他"


class Expectation(str, Enum):
    FULL_REFUND = "全额退款"
    PARTIAL_REFUND = "部分退款"
    EXCHANGE = "换货"
    RESHIP = "补发"
    REPAIR = "维修"
    COMPENSATION = "赔偿"
    OTHER = "其他"


def _require_non_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("must contain non-whitespace text")
    return v


_SESSION_ID_RE = re.compile(r"^([1-9]\d{0,18}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")


def _validate_uuid(v: str) -> str:
    try:
        uuid.UUID(v)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("user_id must be a canonical UUID string")
    return v


def _validate_session_id(v: str | None) -> str | None:
    if v is None:
        return v
    if not _SESSION_ID_RE.match(v):
        raise ValueError("session_id must be a decimal id or canonical UUID string")
    return v


class ChatStreamRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str

    @field_validator("user_id")
    @classmethod
    def _user_id_must_be_uuid(cls, v: str) -> str:
        return _validate_uuid(v)

    @field_validator("session_id")
    @classmethod
    def _session_id_form(cls, v: str | None) -> str | None:
        return _validate_session_id(v)

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        return _require_non_blank(v)


class ChatResumeRequest(BaseModel):
    """ch06(spec §9.1):按 interrupt ID 恢复挂起轮并绑定所选订单。"""
    user_id: str
    session_id: str
    interrupt_id: str = Field(min_length=1, max_length=128)
    order_id: str = Field(min_length=1, max_length=64)

    @field_validator("user_id")
    @classmethod
    def _user_id_must_be_uuid(cls, v: str) -> str:
        return _validate_uuid(v)

    @field_validator("session_id")
    @classmethod
    def _session_id_form(cls, v: str | None) -> str | None:
        return _validate_session_id(v)


_SOURCE_MSG_ID_RE = re.compile(r"^[1-9]\d{0,18}$")

# ch06 spec §9.2:退款原因固定六类目
RefundReason = Literal["质量问题", "商品破损", "发错货", "未收到货",
                       "拍错或多拍", "七天无理由"]


class ChatActionRequest(BaseModel):
    user_id: str
    session_id: str
    source_message_id: str
    action: Literal["create_ticket", "create_refund"]
    ticket_type: Literal["售后", "投诉", "咨询"] | None = None
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    refund_reason: RefundReason | None = None

    @field_validator("user_id")
    @classmethod
    def _user_id(cls, v):  # 与 ChatStreamRequest 同一校验
        return _validate_uuid(v)

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, v):
        return _validate_session_id(v)

    @field_validator("source_message_id")
    @classmethod
    def _source_msg_id(cls, v: str) -> str:
        if not _SOURCE_MSG_ID_RE.match(v):
            raise ValueError("source_message_id must be a decimal id string")
        return v

    @model_validator(mode="after")
    def _action_contract(self):
        if self.action == "create_ticket":
            if self.ticket_type is None:
                raise ValueError("create_ticket requires ticket_type")
            if self.order_id is not None or self.refund_reason is not None:
                raise ValueError("create_ticket does not take refund fields")
        else:
            if self.order_id is None or self.refund_reason is None:
                raise ValueError("create_refund requires order_id and refund_reason")
            if self.ticket_type is not None and self.ticket_type != "售后":
                raise ValueError("create_refund ticket_type must be 售后 or omitted")
        return self


class ExtractRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        return _require_non_blank(v)


class KbManualRequest(BaseModel):
    """/kb 手工录入:服务端拼 frontmatter 后走 chunk_document;vectorize 仅 ingest 用。"""
    doc_type: Literal["faq", "policy", "manual"]
    title: str
    markdown: str
    vectorize: bool = True

    @field_validator("title", "markdown")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        return _require_non_blank(v)


class KbSearchRequest(BaseModel):
    """strategy/scope 缺省 None:由服务端装配的 retriever 按配置自选。"""
    query: str
    top_k: int = Field(default=5, gt=0, le=50)
    min_score: float = Field(default=0.623, ge=-1, le=1)
    strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] | None = None
    scope: Literal["faq", "policy", "product_spec", "after_sales_manual",
                   "qa_mined", "manual"] | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        return _require_non_blank(v)


class FaithCasePatchRequest(BaseModel):
    """台账处置:resolution 由路由去首尾空白后校验 1-300 字符。"""
    status: Literal["已解决", "无需解决"]
    resolution: str


class AfterSaleExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str | None = Field(default=None, description="订单号;用户没提则为 null;保留前导零")
    intent: Intent = Field(description="诉求类型")
    expectation: Expectation | None = Field(default=None, description="期望方案;未表达则为 null")
    summary: str = Field(description="一句话问题摘要")
