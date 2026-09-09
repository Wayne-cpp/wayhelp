import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ChatStreamRequest(BaseModel):
    session_id: uuid.UUID | None = None
    message: str

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        return _require_non_blank(v)


class ExtractRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        return _require_non_blank(v)


class AfterSaleExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str | None = Field(default=None, description="订单号;用户没提则为 null;保留前导零")
    intent: Intent = Field(description="诉求类型")
    expectation: Expectation | None = Field(default=None, description="期望方案;未表达则为 null")
    summary: str = Field(description="一句话问题摘要")
