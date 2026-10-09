"""ch09 边界 span:disabled 时三个边界零行为变化且不抛。"""
from app.services.langfuse_tracing import observation
from tests.test_langfuse_tracing import _settings  # 复用构造器


def test_observation_disabled_yields_none(monkeypatch):
    # 防御开发 shell 里手工 export 过密钥:disabled 判定必须与外部环境解耦
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with observation("retriever", "knowledge.search", {"query": "x"}) as obs:
        assert obs is None


def test_observation_enabled_mock(monkeypatch):
    import app.services.langfuse_tracing as lt
    calls = []

    class FakeObs:
        def update(self, **kw):
            calls.append(kw)

    class FakeClient:
        def start_as_current_observation(self, as_type, name, input=None):
            calls.append({"as_type": as_type, "name": name, "input": input})
            class Cm:
                def __enter__(self): return FakeObs()
                def __exit__(self, *a): return False
            return Cm()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-t")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-t")
    monkeypatch.setattr(lt, "_get_client", lambda: FakeClient())
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-t",
                  langfuse_secret_key="sk-lf-t")
    with lt.observation("tool", "tool.execute", {"tool": "query_order"}, settings=s) as obs:
        obs.update(output={"status": "成功"})
    assert calls[0]["as_type"] == "tool"
    assert calls[1]["output"]["status"] == "成功"
