"""ch09 turn_committed:正常完成发帧;挂起轮不发。"""
import asyncio

from app.services.chat_service import ChatService, DoneEvent, TurnCommittedEvent


class _FakeSnapshot:
    def __init__(self, values, tasks=()):
        self.values = values
        self.tasks = tasks


class _FakeGraphOK:
    async def astream(self, *a, **k):
        return
        yield  # pragma: no cover(空流)

    async def aget_state(self, config):
        return _FakeSnapshot({"turn_message_id": 11,
                              "final_assistant_message_id": 12})


class _FakeGraphPending:
    async def astream(self, *a, **k):
        return
        yield  # pragma: no cover

    async def aget_state(self, config):
        intr = type("I", (), {"id": "i1", "value": {"type": "order_selector",
                                                    "orders": []}})()
        task = type("T", (), {"interrupts": [intr]})()
        return _FakeSnapshot({}, [task])


def _service(graph):
    from app.config import Settings
    s = Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                 database_url="mysql+pymysql://u:p@h/d")
    svc = ChatService(store=None, model=None, settings=s, system_prompt="")
    svc.set_graph(graph)
    return svc


class _Turn:
    session_id = "9"
    user_id = "u"
    user_text = "q"
    lock_key = "9"
    resume_command = None
    user_db_id = None   # stream 走 new_turn_state(user_db_id=...);假轮无落库行
    released = True   # release_turn 走 locks;绕过:测 service 直调 stream 前替换 release


def _collect(graph):
    svc = _service(graph)
    svc.release_turn = lambda turn: None   # 测试旁路锁注册表
    turn = _Turn()
    return asyncio.run(_collect_events(svc, turn))


async def _collect_events(svc, turn):
    return [e async for e in svc.stream(turn)]


def test_committed_frame_emitted_before_done():
    events = _collect(_FakeGraphOK())
    kinds = [type(e).__name__ for e in events]
    assert "TurnCommittedEvent" in kinds
    assert kinds.index("TurnCommittedEvent") < kinds.index("DoneEvent")
    ev = next(e for e in events if isinstance(e, TurnCommittedEvent))
    assert ev.assistant_message_id == "12" and ev.turn_message_id == "11"


def test_no_frame_when_pending_interrupt():
    events = _collect(_FakeGraphPending())
    assert not any(isinstance(e, TurnCommittedEvent) for e in events)
