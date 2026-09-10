import uuid

import pytest
from pydantic import ValidationError

from app.schemas import (
    AfterSaleExtraction,
    ChatStreamRequest,
    Expectation,
    ExtractRequest,
    Intent,
)
from tests.conftest import TEST_USER_ID


def test_chat_request_ok():
    r = ChatStreamRequest(user_id=TEST_USER_ID, message="你好")
    assert r.session_id is None
    sid = uuid.uuid4()
    r2 = ChatStreamRequest(user_id=TEST_USER_ID, session_id=str(sid), message="在吗")
    assert r2.session_id == str(sid)  # session_id 不再 coerced 为 UUID,保持字符串形态


def test_chat_request_blank_message_422():
    with pytest.raises(ValidationError):
        ChatStreamRequest(user_id=TEST_USER_ID, message="   ")


def test_chat_request_bad_session_id_422():
    with pytest.raises(ValidationError):
        ChatStreamRequest(user_id=TEST_USER_ID, session_id="not-a-uuid", message="hi")


def test_extract_request_blank_422():
    with pytest.raises(ValidationError):
        ExtractRequest(text="\n\t ")


def test_extraction_schema_roundtrip():
    e = AfterSaleExtraction(
        order_id="009123",
        intent=Intent.REFUND_ONLY,
        expectation=Expectation.PARTIAL_REFUND,
        summary="少发一件,要求退差价",
    )
    d = e.model_dump(mode="json")
    assert d == {
        "order_id": "009123",
        "intent": "仅退款",
        "expectation": "部分退款",
        "summary": "少发一件,要求退差价",
    }


def test_extraction_nullable_fields():
    e = AfterSaleExtraction(order_id=None, intent=Intent.OTHER, expectation=None, summary="咨询会员")
    assert e.model_dump(mode="json")["order_id"] is None


def test_extraction_forbids_extra():
    with pytest.raises(ValidationError):
        AfterSaleExtraction(
            order_id=None,
            intent=Intent.OTHER,
            expectation=None,
            summary="x",
            bogus=1,
        )
