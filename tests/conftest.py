import asyncio

import pytest

from app.config import Settings
from app.main import AppRuntime
from app.sessions import InMemorySessionStore
from app.tools.business import MOCK_TOOLS, RetrievalTrace, TurnToolset

TEST_USER_ID = "11111111-1111-1111-1111-111111111111"


class UserBoundMemoryStore(InMemorySessionStore):
    """测试用内存存储。app 层的 InMemorySessionStore 按 T3 计划不核对 user_id,
    而 user_mismatch 404 契约测试需要可校验归属 —— 在测试侧补上绑定。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owners: dict[str, str] = {}

    async def create(self, user_id: str) -> str:
        sid = await super().create(user_id)
        self._owners[sid] = user_id
        return sid

    async def exists(self, session_id: str, user_id: str) -> bool:
        return session_id in self._sessions and self._owners.get(session_id) == user_id


def make_runtime(tools=None, store=None):
    return AppRuntime(
        store=store or UserBoundMemoryStore(1000, 100, 8000),
        toolset_factory=lambda sid: TurnToolset(
            list(MOCK_TOOLS if tools is None else tools), RetrievalTrace()),
    )


def make_settings(**overrides) -> Settings:
    base = dict(
        openai_base_url="http://test/v1",
        openai_api_key="test-key",
        model_name="test-model",
        database_url="mysql+pymysql://u:p@127.0.0.1:9/wayhelp",  # 占位,测试注入 runtime 不连接
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeChunk:
    def __init__(self, content: str = "", finish_reason: str | None = None,
                 tool_call_chunks: list[dict] | None = None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.response_metadata = {} if finish_reason is None else {"finish_reason": finish_reason}


class FakeStreamModel:
    """script 元素:str(delta 文本)| Exception | ("finish", reason) | ("tool", [tool_call_chunks...])
    多段脚本用 ("then", next_script) 分隔第二次调用(两次均可绑定工具,由调用方决定)。"""

    def __init__(self, script):
        self._scripts = [list(script)]
        self.received: list = []
        self.received_tools: list[list | None] = []  # 每次调用绑定的工具名列表或 None

    def bind_tools(self, tools):
        self._bound = [t.name for t in tools]
        return self

    async def astream(self, messages):
        self.received.append(messages)
        self.received_tools.append(getattr(self, "_bound", None))
        self._bound = None
        script = self._scripts.pop(0) if self._scripts else ["(空)"]
        for item in script:
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple) and item[0] == "finish":
                yield FakeChunk("", item[1])
            elif isinstance(item, tuple) and item[0] == "tool":
                yield FakeChunk("", tool_call_chunks=item[1])
            elif isinstance(item, tuple) and item[0] == "then":
                self._scripts.insert(0, list(item[1]))
            else:
                yield FakeChunk(item)
            await asyncio.sleep(0)
