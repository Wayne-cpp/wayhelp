import httpx
import logging

from app.chains.extract_chain import run_extraction
from app.errors import MessageTooLongError, UpstreamError
from app.main import create_app
from app.schemas import AfterSaleExtraction, Intent
from tests.conftest import make_runtime, make_settings
import pytest

GOOD = AfterSaleExtraction(order_id="001", intent=Intent.REFUND_ONLY, expectation=None, summary="未收到货要求退款")


class FakeStructuredModel:
    def __init__(self, output=None, error=None):
        self._output = output
        self._error = error
        self.calls = []

    def with_structured_output(self, schema, *, method, include_raw):
        self.calls.append({"schema": schema, "method": method, "include_raw": include_raw})
        output, error = self._output, self._error

        class _Runnable:
            async def ainvoke(self, messages):
                if error is not None:
                    raise error
                return output

        return _Runnable()


def make_extract_app(model, **over):
    return create_app(settings=make_settings(**over), model=model,
                      runtime=make_runtime(tools=[]))


async def post_extract(app, text):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.post("/v1/extract", json={"text": text})


async def test_extract_success_json():
    model = FakeStructuredModel(output={"raw": None, "parsed": GOOD, "parsing_error": None})
    app = make_extract_app(model)
    resp = await post_extract(app, "订单001没收到货,我要退款")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"order_id", "intent", "expectation", "summary"}
    assert body["order_id"] == "001"
    assert body["intent"] == "仅退款"
    assert model.calls[0]["method"] == "json_schema"
    assert model.calls[0]["include_raw"] is True


async def test_extract_method_from_settings():
    model = FakeStructuredModel(output={"raw": None, "parsed": GOOD, "parsing_error": None})
    app = make_extract_app(model, structured_output_method="json_mode")
    resp = await post_extract(app, "订单001没收到货")
    assert resp.status_code == 200
    assert model.calls[0]["method"] == "json_mode"


async def test_extract_parsing_error_502():
    model = FakeStructuredModel(output={"raw": None, "parsed": None, "parsing_error": ValueError("x")})
    app = make_extract_app(model)
    resp = await post_extract(app, "随便一段描述")
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_error"


async def test_extract_upstream_exception_502():
    model = FakeStructuredModel(error=RuntimeError("boom"))
    app = make_extract_app(model)
    resp = await post_extract(app, "随便一段描述")
    assert resp.status_code == 502
    assert "boom" not in resp.text


async def test_extract_overlong_text_422():
    model = FakeStructuredModel(output={"raw": None, "parsed": GOOD, "parsing_error": None})
    app = make_extract_app(model, max_message_chars=10)
    resp = await post_extract(app, "这" * 20)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "message_too_long"


async def test_extract_token_budget_precheck():
    model = FakeStructuredModel(output={"raw": None, "parsed": GOOD, "parsing_error": None})
    with pytest.raises(MessageTooLongError):
        await run_extraction(model, "json_schema", "一" * 500, max_input_tokens=50)


async def test_extract_parsing_error_log_sanitized(caplog):
    model = FakeStructuredModel(output={"raw": None, "parsed": None, "parsing_error": ValueError("secret-x")})
    app = make_extract_app(model)
    with caplog.at_level(logging.WARNING, logger="app.chains.extract_chain"):
        resp = await post_extract(app, "随便一段描述")
    assert resp.status_code == 502
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "parsing failed" in joined
    assert "ValueError" in joined
    assert "secret-x" not in joined


async def test_extract_upstream_exception_log_sanitized(caplog):
    model = FakeStructuredModel(error=RuntimeError("boom"))
    app = make_extract_app(model)
    with caplog.at_level(logging.WARNING, logger="app.chains.extract_chain"):
        resp = await post_extract(app, "随便一段描述")
    assert resp.status_code == 502
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "upstream error" in joined
    assert "RuntimeError" in joined
    assert "boom" not in joined
