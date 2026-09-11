import pytest

from app.main import create_app  # T6 才改 main;本任务直接用 ChatService 装配
from app.services.chat_service import ChatService, DeltaEvent
from app.sessions import InMemorySessionStore, StoredMessage
from app.tool_envelope import wrap
from app.tools.business import MOCK_TOOLS
from app.tools.executor import ToolExecutor, ToolRegistry
from tests.conftest import TEST_USER_ID, FakeStreamModel, make_settings

TOOL_CHUNKS = [
    {"name": "query_order", "args": "{\"order_id\": \"10", "id": "call_1", "index": 0},
    {"name": None, "args": "01\"}", "id": None, "index": 0},
]

# 第二次调用只产出 tool_calls 时的兜底话术(须与 chat_service.FALLBACK_ANSWER 一致)
FALLBACK = "抱歉,暂时没有查到相关信息。您可以换个说法问我,或回复「转人工」,让人工客服帮您处理。"


def make_service(script, tools=None):
    settings = make_settings()
    store = InMemorySessionStore(10, 100, 8000)
    model = FakeStreamModel(script)
    factory_calls = []

    def toolset_factory(sid: str):
        factory_calls.append(sid)
        return tools if tools is not None else MOCK_TOOLS

    service = ChatService(store, model, settings, "system", toolset_factory)
    return service, model, store, factory_calls


async def collect(service, turn):
    return [e async for e in service.stream(turn)]


async def test_no_tool_single_call():
    service, model, store, factory_calls = make_service(["你", "好"], tools=[])
    turn = await service.prepare(TEST_USER_ID, None, "在吗")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    assert types == ["SessionEvent", "DeltaEvent", "DeltaEvent", "DoneEvent"]
    assert model.received_tools == [None]  # 工具集为空时不 bind
    assert factory_calls == [turn.session_id]


async def test_tool_call_full_sequence():
    script = [
        ("tool", TOOL_CHUNKS),
        ("then", ["物流", "在", "路上"]),
    ]
    service, model, store, factory_calls = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "订单 1001 呢")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    assert types == ["SessionEvent", "ToolStartEvent", "ToolEndEvent",
                     "DeltaEvent", "DeltaEvent", "DeltaEvent", "DoneEvent"]
    start = events[1]
    assert start.name == "query_order" and start.tool_call_id == "call_1"
    assert start.args == {"order_id": "1001"}
    end = events[2]
    assert end.ok is True and end.summary
    # 第二次调用绑定工具承接结构化通道,但不再执行任何 tool_calls
    # (types 已断言无第二次 ToolStart/ToolEnd 帧)
    assert model.received_tools[0] == ["query_order", "query_product", "query_logistics"]
    assert model.received_tools[1] == ["query_order", "query_product", "query_logistics"]
    # 落库 4 条:user / assistant(tool_calls) / tool / assistant
    snap = await store.snapshot(turn.session_id)
    assert [m.role for m in snap] == ["user", "assistant", "tool", "assistant"]
    assert snap[1].tool_calls[0]["name"] == "query_order"
    from app.tool_envelope import unwrap
    body, ok2 = unwrap(snap[2].content)
    assert ok2 is True and "order_id" in body
    assert snap[3].content == "物流在路上"


async def test_invalid_tool_call_frame():
    script = [("tool", [{"name": "query_order", "args": "{损坏", "id": "c1", "index": 0}])]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert type(events[-1]).__name__ == "ErrorEvent"
    assert events[-1].code == "invalid_tool_call"
    assert await store.snapshot(turn.session_id) == []


async def test_too_many_tool_calls_rejected():
    chunks = [
        {"name": "query_order", "args": "{\"order_id\": \"1\"}", "id": f"c{i}", "index": i}
        for i in range(6)
    ]
    service, model, store, _ = make_service([("tool", chunks)])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert events[-1].code == "invalid_tool_call"


async def test_duplicate_create_ticket_rejected():
    chunks = [
        {"name": "create_ticket", "args": "{}", "id": "c0", "index": 0},
        {"name": "create_ticket", "args": "{}", "id": "c1", "index": 1},
    ]
    service, model, store, _ = make_service([("tool", chunks)])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert events[-1].code == "invalid_tool_call"


async def test_first_call_length_finish_no_tools_executed():
    script = [("tool", TOOL_CHUNKS), ("finish", "length")]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    assert events[-1].code == "output_too_long"
    assert await store.snapshot(turn.session_id) == []


async def test_tool_error_still_answers():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def broken(x: str) -> str:
        """总是失败"""
        raise ValueError("db down")

    script = [
        ("tool", [{"name": "broken", "args": "{\"x\": \"1\"}", "id": "c1", "index": 0}]),
        ("then", ["查询失败,转人工"]),
    ]
    service, model, store, _ = make_service(script, tools=[broken])
    turn = await service.prepare(TEST_USER_ID, None, "hi")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    assert types == ["SessionEvent", "ToolStartEvent", "ToolEndEvent",
                     "DeltaEvent", "DoneEvent"]
    assert events[2].ok is False
    snap = await store.snapshot(turn.session_id)
    from app.tool_envelope import unwrap
    body, ok = unwrap(snap[2].content)
    assert ok is False and "db down" not in body  # 脱敏


async def test_second_call_context_contains_tool_results():
    script = [("tool", TOOL_CHUNKS), ("then", ["答"])]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "查一下")
    await collect(service, turn)
    second_messages = model.received[1]
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in second_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "call_1"
    ai_with_calls = [m for m in second_messages
                     if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", None)]
    assert ai_with_calls and ai_with_calls[0].tool_calls[0]["id"] == "call_1"


async def test_second_call_tool_calls_dropped_with_fallback():
    """第二次调用(已绑工具)模型仍想换关键词重试:tool_calls 一律不执行,
    无第二次工具帧,以兜底话术作答并原样落库。"""
    retry_chunks = [{"name": "query_faq",
                     "args": "{\"keyword\": \"运费\"}", "id": "call_2", "index": 0}]
    script = [
        ("tool", TOOL_CHUNKS),
        ("then", [("tool", retry_chunks), ("finish", "tool_calls")]),
    ]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "邮费是多少")
    events = await collect(service, turn)
    types = [type(e).__name__ for e in events]
    # 只有第一次调用的 tool 帧;第二次的 tool_calls 被丢弃,不执行不推帧
    assert types == ["SessionEvent", "ToolStartEvent", "ToolEndEvent",
                     "DeltaEvent", "DoneEvent"]
    deltas = [e.content for e in events if isinstance(e, DeltaEvent)]
    assert deltas == [FALLBACK]
    # 第二次调用绑定了工具(承接结构化通道),但未执行其中任何调用
    assert model.received_tools[1] == ["query_order", "query_product", "query_logistics"]
    # 落库 4 条;最终 assistant 行剥离 tool_calls,内容为兜底话术
    snap = await store.snapshot(turn.session_id)
    assert [m.role for m in snap] == ["user", "assistant", "tool", "assistant"]
    assert snap[1].tool_calls[0]["name"] == "query_order"
    assert not snap[3].tool_calls
    assert snap[3].content == FALLBACK


async def test_second_call_text_answer_not_replaced_by_fallback():
    """第二次调用正常文本作答:文本照常流出,不受兜底话术影响。"""
    script = [("tool", TOOL_CHUNKS), ("then", ["运费以订单页结算为准"])]
    service, model, store, _ = make_service(script)
    turn = await service.prepare(TEST_USER_ID, None, "邮费是多少")
    events = await collect(service, turn)
    deltas = [e.content for e in events if isinstance(e, DeltaEvent)]
    assert deltas == ["运费以订单页结算为准"]
    assert type(events[-1]).__name__ == "DoneEvent"
    snap = await store.snapshot(turn.session_id)
    assert snap[3].content == "运费以订单页结算为准"


async def test_lock_recreated_lazily_for_existing_session():
    """模拟重启:新 ChatService(锁表为空)对已有 session 仍能串行。"""
    service1, model, store, _ = make_service(["一"])
    turn = await service1.prepare(TEST_USER_ID, None, "hi")
    await collect(service1, turn)
    service2, model2, _, _ = make_service(["二"])
    service2._store = store  # 同一存储,新锁表
    turn2 = await service2.prepare(TEST_USER_ID, turn.session_id, "again")
    events = await collect(service2, turn2)
    assert type(events[-1]).__name__ == "DoneEvent"
    assert model2.received[0]  # 历史被读取
