import asyncio

import pytest

from app.config import Settings


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
    def __init__(self, content: str = "", finish_reason: str | None = None):
        self.content = content
        self.response_metadata = {} if finish_reason is None else {"finish_reason": finish_reason}


class FakeStreamModel:
    """script 元素:str(delta 文本)| Exception(迭代到即抛出)| ("finish", reason)"""

    def __init__(self, script):
        self._script = list(script)
        self.received: list = []

    async def astream(self, messages):
        self.received.append(messages)
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple) and item[0] == "finish":
                yield FakeChunk("", item[1])
            else:
                yield FakeChunk(item)
            await asyncio.sleep(0)
