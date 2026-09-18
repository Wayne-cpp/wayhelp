import asyncio

import pytest

from app.config import Settings
from app.knowledge.state import KnowledgeStateHolder
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
        knowledge_state=KnowledgeStateHolder(),  # T11:/kb/api/state 的 knowledge_state 字段
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
                 tool_call_chunks: list[dict] | None = None, usage_metadata: dict | None = None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.usage_metadata = usage_metadata
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

    # 裁决适配(Task 13):图的 classify_intent 节点走 ainvoke;返回罐头业务意图
    # (订单/无需知识 → business 直进 main_agent),不消耗 astream 脚本、不记
    # received——让 test_chat_api 既有 SSE 协议用例的脚本全数留给 main_agent。
    async def ainvoke(self, messages, config=None, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content='{"intent":"订单","needs_knowledge":false}')

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
            elif isinstance(item, tuple) and item[0] == "usage":
                yield FakeChunk("", usage_metadata=item[1])
            elif isinstance(item, tuple) and item[0] == "then":
                self._scripts.insert(0, list(item[1]))
            else:
                yield FakeChunk(item)
            await asyncio.sleep(0)


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.chains.tool_chat_chain import finalize_tool_calls, merge_tool_call_chunks


class ScriptedChatModel(BaseChatModel):
    """脚本化假模型(真 Runnable → LangChain 回调链完整,图 messages 流可用)。
    scripts: 每段是一次调用的脚本,元素:
      str → 文本 delta;("tool", [tool_call_chunks]) → 工具调用;
      ("finish", reason) → finish_reason;("usage", dict) → usage_metadata;
      Exception → 抛错。bind_tools 记录工具名并返回自身。"""

    scripts: list
    bound: list | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound = [t.name for t in tools]
        return self

    def _next_script(self) -> list:
        return list(self.scripts.pop(0)) if self.scripts else ["(空)"]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text, calls, usage = "", [], None
        for item in self._next_script():
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                text += item
            elif item[0] == "tool":
                acc: dict[int, dict] = {}
                merge_tool_call_chunks(acc, item[1])
                parsed, _ = finalize_tool_calls(acc)
                calls.extend(parsed)
            elif item[0] == "usage":
                usage = item[1]
        msg = AIMessage(content=text, tool_calls=calls)
        if usage:
            msg.usage_metadata = usage
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # 回调(on_llm_new_token)由 BaseChatModel.astream 包装层负责,这里只 yield
        for item in self._next_script():
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                chunk = AIMessageChunk(content=item)
            elif item[0] == "tool":
                chunk = AIMessageChunk(content="", tool_call_chunks=item[1])
            elif item[0] == "finish":
                chunk = AIMessageChunk(content="",
                                       response_metadata={"finish_reason": item[1]})
            elif item[0] == "usage":
                chunk = AIMessageChunk(content="", usage_metadata=item[1])
            else:
                continue
            yield ChatGenerationChunk(message=chunk)
