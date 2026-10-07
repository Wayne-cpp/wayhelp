import pytest
from pydantic import BaseModel, Field

from app.tools.catalog import (
    ToolCatalog, ToolSpec, TurnContext, canonical_args_sha, classify_mcp_tool,
    idempotency_key_for, mcp_args_model,
)


class _Args(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)


def _spec(name="t1", **over):
    base = dict(name=name, description="d", args_schema=_Args.model_json_schema(),
                args_model=_Args, source="builtin", permission="read")
    base.update(over)
    return ToolSpec(**base)


def test_catalog_register_and_duplicate_reject():
    c = ToolCatalog()
    c.register(_spec("a"))
    assert c.get("a").name == "a"
    assert [s.name for s in c.specs()] == ["a"]
    with pytest.raises(ValueError):
        c.register(_spec("a"))
    assert c.get("ghost") is None


def test_turncontext_defaults():
    ctx = TurnContext(user_id="u", conversation_id=None, resolved_query="q",
                      retriever=None, settings=None)
    assert ctx.order_snapshots == [] and ctx.session_factory is None


def test_classify_mcp_tool_two_shapes_only():
    assert classify_mcp_tool("logistics", {"type": "object", "properties": {},
                                           "required": []}) == ("read", "none")
    order_schema = {"type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}
    assert classify_mcp_tool("after_sales", order_schema) == ("read", "order")
    # 未知 Server / 额外参数 / 写意味参数 / order_id 非必填 → 拒
    assert classify_mcp_tool("warehouse", order_schema) is None
    assert classify_mcp_tool("logistics", {"type": "object", "properties": {
        "order_id": {"type": "string"}, "note": {"type": "string"}},
        "required": ["order_id"]}) is None
    assert classify_mcp_tool("logistics", {"type": "object", "properties": {
        "order_id": {"type": "string"}}, "required": []}) is None


def test_mcp_args_model_shapes():
    m = mcp_args_model("order")
    assert m(order_id="X").order_id == "X"
    with pytest.raises(Exception):
        m()
    empty = mcp_args_model("none")
    assert empty().model_dump() == {}


def test_canonical_args_sha_stable_and_key():
    a = canonical_args_sha({"b": 1, "a": "汉"})
    b = canonical_args_sha({"a": "汉", "b": 1})
    assert a == b and len(a) == 64
    assert idempotency_key_for(7, "call-1") == idempotency_key_for(7, "call-1")
    assert idempotency_key_for(7, "call-1") != idempotency_key_for(8, "call-1")
