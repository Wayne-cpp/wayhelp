import asyncio

import pytest

from app.config import Settings

TEST_USER_ID = "11111111-1111-1111-1111-111111111111"


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
    多段脚本用 ("then", next_script) 分隔第二次调用(bind_tools 后第一次、plain 第二次)。"""

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
