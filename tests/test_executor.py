"""ch08 执行引擎(spec §2/§3):六步管线 / 超时重试 / 写确认路径 / 批次契约 / 审计。"""
import asyncio
import json
import time

import pytest
from pydantic import BaseModel, Field

from app.models import Ticket, ToolAuditLog, ToolWriteIdempotency, Conversation
from app.tools.catalog import (
    ToolSpec, TurnContext, canonical_args_sha, idempotency_key_for,
)
from app.tools.executor import (
    PendingWrite, ToolExecutor, ToolFace, batch_violation,
)
from tests.conftest import TEST_USER_ID, make_settings
from tests.dbfixtures import db_engine, db_session_factory  # noqa: F401


class _X(BaseModel):
    x: str = Field(min_length=1, max_length=64)


class _OrderId(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)


def _spec(name, fn, permission="read", ownership="none", source="builtin",
          mcp_server=None, formatter=None, args_model=_X):
    return ToolSpec(name=name, description="d", args_schema=args_model.model_json_schema(),
                    args_model=args_model, source=source, permission=permission,
                    ownership=ownership, mcp_server=mcp_server, fn=fn,
                    formatter=formatter)


def _ctx(sf, **over):
    base = dict(user_id=TEST_USER_ID, conversation_id=None, resolved_query="q",
                retriever=None, settings=make_settings(), session_factory=sf)
    base.update(over)
    return TurnContext(**base)


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def _audits(sf, tool_call_id="c1"):
    with sf() as s:
        return s.query(ToolAuditLog).filter_by(tool_call_id=tool_call_id).all()


# ── 面与契约 ──

def test_face_rejects_duplicate():
    f1 = _spec("a", lambda a, c: "1")
    with pytest.raises(ValueError):
        ToolFace([f1, _spec("a", lambda a, c: "2")])


def test_batch_violation_rules():
    w = _spec("create_ticket", lambda a, c, m: "", permission="write")
    r = _spec("query_order", lambda a, c: "{}")
    face = ToolFace([w, r])
    ok = [_call("query_order", {"x": "1"}, "c1"), _call("create_ticket", {"x": "1"}, "c2")]
    assert batch_violation(ok, face) is None
    two_writes = ok + [_call("create_ticket", {"x": "1"}, "c3")]
    assert batch_violation(two_writes, face) is not None
    write_not_last = [_call("create_ticket", {"x": "1"}, "c1"),
                      _call("query_order", {"x": "1"}, "c2")]
    assert batch_violation(write_not_last, face) is not None


# ── 六步管线 ──

async def test_unknown_tool(db_session_factory):
    ex = ToolExecutor(ToolFace([]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("ghost", {"x": "1"}, "c9"), _ctx(db_session_factory))
    assert out.message.status == "error" and out.record.error_code == "unknown_tool"
    assert _audits(db_session_factory, "c9")[0].status == "失败"


async def test_invalid_args_carries_detail(db_session_factory):
    ex = ToolExecutor(ToolFace([_spec("t", lambda a, c: "ok")]), make_settings(),
                      session_factory=db_session_factory)
    out = await ex.execute(_call("t", {"x": ""}, "c7"), _ctx(db_session_factory))
    assert out.message.status == "error" and out.record.error_code == "invalid_args"
    assert "参数" in out.message.content and "x" in out.message.content  # 校验说明回灌
    assert out.record.retry_count == 0
    assert _audits(db_session_factory, "c7")[0].status == "校验拦下"


async def test_ownership_guard_short_circuits(db_session_factory):
    called = {"n": 0}

    def spy(args, ctx):
        called["n"] += 1
        return "{}"

    spec = _spec("query_order", spy, ownership="order", args_model=_OrderId)
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("query_order", {"order_id": "9999-1001"}),
                           _ctx(db_session_factory))
    assert called["n"] == 0                                   # 归属把门未过,未分发
    assert json.loads(out.message.content) == {"error": "订单不存在或不属于当前用户"}
    assert out.message.status == "success"                    # 业务空结果(spec §3 终态映射)
    assert _audits(db_session_factory)[0].status == "成功"


async def test_read_success_audit_fields(db_session_factory):
    spec = _spec("query_product", lambda a, c: json.dumps({"ok": 1}, ensure_ascii=False))
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("query_product", {"x": "1"}), _ctx(db_session_factory))
    assert out.message.status == "success"
    row = _audits(db_session_factory)[0]
    assert (row.status, row.tool_source, row.mcp_server, row.retry_count) == \
           ("成功", "builtin", None, 0)
    assert row.duration_ms is not None and row.arguments == {"x": "1"}


async def test_timeout_retry_exhausted_audits_timeout(db_session_factory):
    def slow(args, ctx):
        time.sleep(1)
        return "done"

    spec = _spec("slow_tool", slow)
    ex = ToolExecutor(ToolFace([spec]),
                      make_settings(tool_max_retries=1,
                                    tool_timeout_overrides={"slow_tool": 0.05}),
                      session_factory=db_session_factory)
    t0 = time.monotonic()
    out = await ex.execute(_call("slow_tool", {"x": "1"}), _ctx(db_session_factory))
    assert time.monotonic() - t0 < 3
    assert out.record.error_code == "tool_unavailable" and out.record.retry_count == 1
    row = _audits(db_session_factory)[0]
    assert row.status == "超时" and row.retry_count == 1      # 验收 6:重试次数/超时/耗时齐


async def test_retry_eventually_succeeds(db_session_factory):
    calls = {"n": 0}

    def flaky(args, ctx):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("jitter")
        return "ok-after-retry"

    spec = _spec("flaky", flaky)
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("flaky", {"x": "1"}), _ctx(db_session_factory))
    assert out.message.content == "ok-after-retry" and out.record.retry_count == 2
    assert _audits(db_session_factory)[0].status == "成功"


async def test_business_error_not_retried_and_sanitized(db_session_factory):
    def boom(args, ctx):
        raise ValueError("business broke")

    spec = _spec("biz", boom)
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("biz", {"x": "1"}), _ctx(db_session_factory))
    assert out.record.retry_count == 0 and "business broke" not in out.message.content
    assert _audits(db_session_factory)[0].status == "失败"


async def test_formatter_picks_fields(db_session_factory):
    spec = _spec("mcpish", None, source="mcp", mcp_server="logistics",
                 formatter=lambda raw: json.dumps({"company": "顺丰速运"},
                                                  ensure_ascii=False))

    class FakeMcp:
        async def call(self, server, name, args):
            return json.dumps({"company": "顺丰速运", "internal_code": "SF01"})

    ex = ToolExecutor(ToolFace([spec]), make_settings(),
                      session_factory=db_session_factory, mcp=FakeMcp())
    out = await ex.execute(_call("mcpish", {"x": "1"}), _ctx(db_session_factory))
    assert "internal_code" not in out.message.content and "顺丰速运" in out.message.content
    row = _audits(db_session_factory)[0]
    assert (row.tool_source, row.mcp_server) == ("mcp", "logistics")


async def test_write_call_rejected_on_plain_execute(db_session_factory):
    called = {"n": 0}

    def wfn(args, ctx, meta):
        called["n"] += 1
        return "{}"

    spec = _spec("create_ticket", wfn, permission="write")
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute(_call("create_ticket", {"x": "1"}), _ctx(db_session_factory))
    assert called["n"] == 0 and out.record.error_code == "permission_denied"
    assert _audits(db_session_factory)[0].status == "权限拒绝"


async def test_mark_invalid_batch_audits_each(db_session_factory):
    ex = ToolExecutor(ToolFace([]), make_settings(), session_factory=db_session_factory)
    msgs = await ex.mark_invalid_batch(
        [_call("a", {"x": "1"}, "c1"), _call("b", {"x": "1"}, "c2")],
        _ctx(db_session_factory), "写操作必须是最后一步调用")
    assert [m.tool_call_id for m in msgs] == ["c1", "c2"]
    assert all(m.status == "error" for m in msgs)
    with db_session_factory() as s:
        rows = s.query(ToolAuditLog).filter(ToolAuditLog.tool_call_id.in_(["c1", "c2"])).all()
    assert {r.status for r in rows} == {"校验拦下"} and len(rows) == 2


# ── execute_confirmed 写确认路径 ──

def _seed_conv(sf, cid=1):
    with sf() as s:
        s.add(Conversation(id=cid, user_id=TEST_USER_ID))
        s.commit()


def _ticket_spec():
    from app.tools.builtin.ticket import create_ticket_fn
    return {s.name: s for s in __import__(
        "app.tools.builtin", fromlist=["scan_builtin_specs"]).scan_builtin_specs()
    }["create_ticket"]


def _pending(args=None):
    args = args or {"description": "商品质量问题", "ticket_type": "售后"}
    return PendingWrite(tool_call_id="tc1", name="create_ticket", args=args,
                        args_sha256=canonical_args_sha(args))


async def test_confirmed_write_success(db_session_factory):
    _seed_conv(db_session_factory)
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    out = await ex.execute_confirmed(_pending(), ctx)
    payload = json.loads(out.message.content)
    assert out.message.status == "success" and payload["ticket_no"].startswith("T")
    with db_session_factory() as s:
        assert s.get(Ticket, payload["ticket_no"]) is not None
        key = idempotency_key_for(1, "tc1")
        assert s.get(ToolWriteIdempotency, key).ticket_no == payload["ticket_no"]
    row = _audits(db_session_factory, "tc1")[0]
    assert row.status == "成功" and row.retry_count == 0


async def test_confirmed_write_idempotent_replay(db_session_factory):
    _seed_conv(db_session_factory)
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    first = json.loads((await ex.execute_confirmed(_pending(), ctx)).message.content)
    again = json.loads((await ex.execute_confirmed(_pending(), ctx)).message.content)
    assert again["ticket_no"] == first["ticket_no"]  # 重放返回原工单,不重复建
    with db_session_factory() as s:
        assert s.query(Ticket).count() == 1


async def test_confirmed_write_args_conflict_rejected(db_session_factory):
    _seed_conv(db_session_factory)
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    await ex.execute_confirmed(_pending(), ctx)
    bad = _pending(args={"description": "被篡改", "ticket_type": "投诉"})
    out = await ex.execute_confirmed(bad, ctx)
    assert out.record.error_code == "idempotency_conflict"
    rows = _audits(db_session_factory, "tc1")
    assert rows[-1].status == "失败" and "被篡改" not in (rows[-1].error_message or "")


async def test_confirmed_write_requires_conversation(db_session_factory):
    spec = _ticket_spec()
    ex = ToolExecutor(ToolFace([spec]), make_settings(), session_factory=db_session_factory)
    out = await ex.execute_confirmed(_pending(), _ctx(db_session_factory))  # cid=None
    assert out.record.error_code == "invalid_args"
    assert _audits(db_session_factory, "tc1")[0].status == "校验拦下"


async def test_confirmed_write_timeout_no_retry_background_lands(db_session_factory):
    _seed_conv(db_session_factory)
    import time as _time
    from app.tools.builtin import ticket as ticket_mod
    orig_fn = ticket_mod.create_ticket_fn

    def slow_fn(args, ctx, meta):
        _time.sleep(0.3)
        return orig_fn(args, ctx, meta)

    spec = ToolSpec(**{**_ticket_spec().__dict__, "fn": slow_fn})
    ex = ToolExecutor(ToolFace([spec]),
                      make_settings(tool_timeout_overrides={"create_ticket": 0.05}),
                      session_factory=db_session_factory)
    ctx = _ctx(db_session_factory, conversation_id=1)
    out = await ex.execute_confirmed(_pending(), ctx)
    assert out.record.error_code == "write_timeout" and out.record.retry_count == 0
    assert "请勿重复提交" in out.message.content
    rows = _audits(db_session_factory, "tc1")
    assert len(rows) == 1 and rows[0].status == "超时" and rows[0].retry_count == 0
    await asyncio.sleep(0.6)  # 后台 shielded 写最终落地;审计仍只有一条
    with db_session_factory() as s:
        key = idempotency_key_for(1, "tc1")
        assert s.get(ToolWriteIdempotency, key) is not None
    assert len(_audits(db_session_factory, "tc1")) == 1


async def test_deny_write_audits_denied(db_session_factory):
    ex = ToolExecutor(ToolFace([]), make_settings(), session_factory=db_session_factory)
    out = await ex.deny_write(_pending(), _ctx(db_session_factory), "用户取消")
    assert out.message.status == "error" and "取消" in out.message.content
    row = _audits(db_session_factory, "tc1")[0]
    assert row.status == "权限拒绝" and row.arguments["description"] == "商品质量问题"
