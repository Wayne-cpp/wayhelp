"""main 装配契约(T11):title、knowledge_state 暴露、注入态不触发生产校验。

生产路径的启动 DDL 校验(check_ch04_tables 缺表 RuntimeError)由 T1 的
test_db_ch04.py + README 演示覆盖;本文件只断言注入 runtime 时的装配形态。
"""

from app.knowledge.state import KnowledgeState, KnowledgeStateHolder
from app.main import AppRuntime, create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings


def test_create_app_title_is_ch04():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    assert app.title == "wayhelp-ch04"


def test_create_app_exposes_knowledge_state():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    holder = app.state.knowledge_state
    assert isinstance(holder, KnowledgeStateHolder)
    assert holder.get() == KnowledgeState.READY.value  # 初值 ready


def test_create_app_fills_default_holder_when_runtime_lacks_one():
    """直接构造的 AppRuntime(测试/嵌入)缺 knowledge_state:装配补默认 holder。"""
    rt = make_runtime(tools=[])
    bare = AppRuntime(store=rt.store, toolset_factory=rt.toolset_factory)
    assert bare.knowledge_state is None
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=bare)
    assert isinstance(app.state.knowledge_state, KnowledgeStateHolder)
    assert app.state.knowledge_state.get() == KnowledgeState.READY.value
