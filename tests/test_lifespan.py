from fastapi.testclient import TestClient

from app.main import AppRuntime, create_app
from tests.conftest import FakeStreamModel, make_runtime, make_settings


class _RecordingRetriever:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _runtime_with_retriever(retriever):
    rt = make_runtime(tools=[])
    return AppRuntime(store=rt.store, toolset_factory=rt.toolset_factory,
                      retriever=retriever)


def test_owned_runtime_retriever_closed_on_shutdown(monkeypatch):
    """自建 runtime:应用关闭时必须释放 retriever(spec §8)。"""
    retriever = _RecordingRetriever()
    monkeypatch.setattr("app.main._build_production_runtime",
                        lambda settings: _runtime_with_retriever(retriever))
    app = create_app(settings=make_settings(), model=FakeStreamModel([]))
    with TestClient(app):
        pass
    assert retriever.closed is True


def test_injected_runtime_retriever_not_closed():
    """注入 runtime(测试/嵌入场景):所有权在调用方,应用不得关闭。"""
    retriever = _RecordingRetriever()
    app = create_app(settings=make_settings(), model=FakeStreamModel([]),
                     runtime=_runtime_with_retriever(retriever))
    with TestClient(app):
        pass
    assert retriever.closed is False
