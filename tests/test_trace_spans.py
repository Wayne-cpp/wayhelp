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
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
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
    # enabled 之外还需 ambient span(聊天轮由 handler attach 提供)
    span_ctx = otel_trace.SpanContext(
        trace_id=0x1234567890abcdef1234567890abcdef, span_id=0x1234567890abcdef,
        is_remote=False, trace_flags=otel_trace.TraceFlags(1))
    ctx = otel_trace.set_span_in_context(otel_trace.NonRecordingSpan(span_ctx))
    token = otel_context.attach(ctx)
    try:
        with lt.observation("tool", "tool.execute", {"tool": "query_order"},
                            settings=s) as obs:
            obs.update(output={"status": "成功"})
    finally:
        otel_context.detach(token)
    assert calls[0]["as_type"] == "tool"
    assert calls[1]["output"]["status"] == "成功"
def test_observation_without_active_span_skips_client(monkeypatch):
    """验收回归:无 ambient span(评估子进程/脚本直跑)→ 连 client 都不碰,
    不产孤儿 trace(eval 两晚刷出 599 条 knowledge.search 孤儿掩埋聊天 trace)。"""
    import app.services.langfuse_tracing as lt
    calls = []

    class FakeClient:
        def start_as_current_observation(self, **kw):
            calls.append("created")   # 被调用即证明尝试建 span
            class Cm:
                def __enter__(self): return object()
                def __exit__(self, *a): return False
            return Cm()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-t")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-t")
    monkeypatch.setattr(lt, "_get_client", lambda: FakeClient())
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-t",
                  langfuse_secret_key="sk-lf-t")
    with lt.observation("retriever", "knowledge.search", {"query": "x"},
                        settings=s) as obs:
        assert obs is None
    assert calls == []   # 关键断言:无 ambient 时不得尝试建 span


def test_observation_with_active_span_creates_child(monkeypatch):
    """聊天轮(handler 已 attach 节点 span)→ 正常建子 span。"""
    import app.services.langfuse_tracing as lt
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
    calls = []

    class FakeObs:
        def update(self, **kw):
            calls.append(kw)

    class FakeClient:
        def start_as_current_observation(self, as_type, name, input=None):
            calls.append({"as_type": as_type, "name": name})
            class Cm:
                def __enter__(self): return FakeObs()
                def __exit__(self, *a): return False
            return Cm()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-t")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-t")
    monkeypatch.setattr(lt, "_get_client", lambda: FakeClient())
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-t",
                  langfuse_secret_key="sk-lf-t")
    span_ctx = otel_trace.SpanContext(
        trace_id=0x1234567890abcdef1234567890abcdef, span_id=0x1234567890abcdef,
        is_remote=False, trace_flags=otel_trace.TraceFlags(1))
    ctx = otel_trace.set_span_in_context(otel_trace.NonRecordingSpan(span_ctx))
    token = otel_context.attach(ctx)
    try:
        with lt.observation("tool", "tool.execute", {"tool": "query_order"},
                            settings=s) as obs:
            assert obs is not None
            obs.update(output={"status": "成功"})
    finally:
        otel_context.detach(token)
    assert calls[0]["name"] == "tool.execute"
    assert calls[1]["output"]["status"] == "成功"
