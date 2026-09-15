import asyncio
import json
import time

import pytest
from langchain_core.tools import tool
from pydantic import ValidationError

from app.knowledge.retriever import RetryableKnowledgeError
from app.tools.executor import ToolExecutor, ToolRegistry


@tool
def slow_tool(x: str) -> str:
    """慢工具"""
    time.sleep(10)
    return "done"


# StructuredTool 为 pydantic 模型,不允许挂任意属性(计划原文写法实测报
# "has no field 'calls'"),计数器改用模块级 dict,语义不变
FLAKY_CALLS = {"n": 0}


@tool
def flaky_tool(x: str) -> str:
    """前两次超时第三次成功"""
    FLAKY_CALLS["n"] += 1
    if FLAKY_CALLS["n"] < 3:
        time.sleep(10)
    return "ok-after-retry"


@tool
def biz_error_tool(x: str) -> str:
    """业务错误"""
    raise ValueError("business broke")


@tool
def write_tool(x: str) -> str:
    """写工具"""
    return "written"


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


async def test_registry_rejects_duplicate():
    with pytest.raises(ValueError):
        ToolRegistry([slow_tool, slow_tool])


async def test_timeout_then_retry_exhausted():
    ex = ToolExecutor(ToolRegistry([slow_tool]), timeout_seconds=0.1, max_retries=1,
                      max_result_chars=4000)
    t0 = time.monotonic()
    outcome = await ex.execute(_call("slow_tool", {"x": "1"}))
    assert outcome.message.status == "error"
    assert outcome.record.retry_count == 1
    assert outcome.record.error_type == "TimeoutError"
    assert time.monotonic() - t0 < 3  # wait_for 生效,不等满 10s


async def test_retry_eventually_succeeds():
    FLAKY_CALLS["n"] = 0
    ex = ToolExecutor(ToolRegistry([flaky_tool]), timeout_seconds=0.1, max_retries=2,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("flaky_tool", {"x": "1"}))
    assert outcome.message.status == "success"
    assert outcome.message.content == "ok-after-retry"
    assert outcome.record.retry_count == 2


async def test_business_error_not_retried():
    ex = ToolExecutor(ToolRegistry([biz_error_tool]), timeout_seconds=1, max_retries=2,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("biz_error_tool", {"x": "1"}))
    assert outcome.message.status == "error"
    assert outcome.record.retry_count == 0
    assert "business broke" not in outcome.message.content  # 脱敏
    assert outcome.record.error_type == "ValueError"


async def test_unknown_tool():
    ex = ToolExecutor(ToolRegistry([]), timeout_seconds=1, max_retries=0,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("ghost_tool", {"x": "1"}, cid="c9"))
    assert outcome.message.status == "error"
    assert outcome.message.tool_call_id == "c9"
    assert outcome.record.error_type == "UnknownTool"


async def test_invalid_args_becomes_error_tool_message():
    @tool
    def strict_tool(n: int) -> str:
        """严格参数"""
        return str(n)

    ex = ToolExecutor(ToolRegistry([strict_tool]), timeout_seconds=1, max_retries=2,
                      max_result_chars=4000)
    outcome = await ex.execute(_call("strict_tool", {"n": "不是数字"}, cid="c7"))
    assert outcome.message.status == "error"
    assert outcome.message.tool_call_id == "c7"
    assert outcome.record.error_type == "ValidationError"
    assert outcome.record.retry_count == 0


async def test_result_truncated_but_valid_envelope_content():
    @tool
    def big_tool(x: str) -> str:
        """大结果"""
        return "汉" * 5000

    ex = ToolExecutor(ToolRegistry([big_tool]), timeout_seconds=5, max_retries=0,
                      max_result_chars=300)
    outcome = await ex.execute(_call("big_tool", {"x": "1"}))
    assert outcome.message.status == "success"
    assert len(outcome.message.content) < 5000  # 回灌模型的内容本身已截断
    from app.tool_envelope import wrap
    payload = json.loads(wrap(outcome.message.content, True, None, 300))
    assert "truncated" not in payload  # truncate_content 已一次到位,wrap 不再二次截断
    assert payload["content"] == outcome.message.content  # 落库与模型所见逐字一致


async def test_write_tool_cancel_waits_for_thread_to_land():
    events = []  # 记录先后:线程落地 vs 取消抛给调用方

    @tool
    def slow_write(x: str) -> str:
        """慢写工具"""
        time.sleep(0.25)
        events.append("thread-done")
        return "written"

    ex = ToolExecutor(ToolRegistry([slow_write]), timeout_seconds=5, max_retries=0,
                      max_result_chars=4000, write_tools={"slow_write"})
    task = asyncio.ensure_future(ex.execute(_call("slow_write", {"x": "1"})))
    await asyncio.sleep(0.05)  # 已进入写路径
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task  # 取消仍传播(不转成 error outcome)
    events.append("cancel-raised")
    assert task.cancelled()
    assert events == ["thread-done", "cancel-raised"]  # 取消放行前线程已落地


async def test_write_tool_no_wait_for_no_retry(monkeypatch):
    called = {"wait_for": 0}
    real_wait_for = asyncio.wait_for

    async def spy(coro, timeout):
        called["wait_for"] += 1
        return await real_wait_for(coro, timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy)
    ex = ToolExecutor(ToolRegistry([write_tool]), timeout_seconds=0.001, max_retries=3,
                      max_result_chars=4000, write_tools={"write_tool"})
    outcome = await ex.execute(_call("write_tool", {"x": "1"}))
    assert outcome.message.status == "success"
    assert called["wait_for"] == 0  # 写工具不经过 wait_for


async def test_query_faq_policy_no_retry():
    calls = {"n": 0}

    @tool
    def query_faq(keyword: str) -> str:
        """查知识库。"""
        calls["n"] += 1
        raise RetryableKnowledgeError("boom")

    reg = ToolRegistry([query_faq])
    ex = ToolExecutor(reg, timeout_seconds=5, max_retries=2, max_result_chars=4000,
                      tool_policies={"query_faq": (20.0, 0)})
    # 注:补 "type": "tool_call" 信封字段(T6 既载坑:缺它 langchain 把信封当 args,工具体不执行)
    outcome = await ex.execute({"name": "query_faq", "args": {"keyword": "x"}, "id": "1",
                                "type": "tool_call"})
    assert outcome.record.error_code == "tool_unavailable"
    assert calls["n"] == 1            # 零整链重试
    assert outcome.record.retry_count == 0


async def test_other_tools_keep_default_policy():
    calls = {"n": 0}

    @tool
    def query_order(order_id: str) -> str:
        """查订单。"""
        calls["n"] += 1
        raise TimeoutError()

    reg = ToolRegistry([query_order])
    ex = ToolExecutor(reg, timeout_seconds=5, max_retries=2, max_result_chars=4000,
                      tool_policies={"query_faq": (20.0, 0)})
    outcome = await ex.execute({"name": "query_order", "args": {"order_id": "1"}, "id": "1",
                                "type": "tool_call"})
    assert outcome.record.error_code == "tool_unavailable"
    assert calls["n"] == 3            # 默认 2 次重试不变
