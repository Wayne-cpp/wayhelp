"""main 装配契约(T11):title、knowledge_state 暴露、注入态不触发生产校验。

生产路径的启动 DDL 校验(check_ch04_tables 缺表 RuntimeError)由 T1 的
test_db_ch04.py + README 演示覆盖;本文件只断言注入 runtime 时的装配形态。
ch07 Task 11 追加:log/app.log FileHandler 幂等 + 启动预算自检日志。
"""

import logging

from app.knowledge.state import KnowledgeState, KnowledgeStateHolder
from app.main import AppRuntime, create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings


def test_create_app_title_is_ch04():
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=make_runtime(tools=[]))
    assert app.title == "wayhelp"


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


def test_file_handler_idempotent_and_budget_log(caplog):
    """多次 create_app:log/app.log 的 FileHandler 只挂一次;启动打全分量预算日志。"""
    settings = make_settings()
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        create_app(settings=settings, model=FakeStreamModel([]),
                   runtime=make_runtime(tools=[]))
        create_app(settings=settings, model=FakeStreamModel([]),
                   runtime=make_runtime(tools=[]))
    root = logging.getLogger()
    fhs = [h for h in root.handlers
           if isinstance(h, logging.FileHandler) and h.baseFilename.endswith("app.log")]
    assert len(fhs) == 1
    assert "context budget" in caplog.text


def test_budget_selfcheck_critical_on_tiny_window(caplog):
    """窗口装不下一轮 steady 开销:启动打 critical(不 raise,交运维决策)。"""
    tiny = make_settings(model_context_window=100, safety_margin_tokens=0)
    with caplog.at_level(logging.CRITICAL):
        create_app(settings=tiny, model=FakeStreamModel([]),
                   runtime=make_runtime(tools=[]))
    assert "上下文预算不足" in caplog.text
