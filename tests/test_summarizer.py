import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from app.services.summarizer import SummaryRunner
from app.sessions import InMemorySessionStore, StoredMessage


class _FakeModel:
    def __init__(self, text="用户问过订单A1001的退款进度,诉求是尽快退款。"):
        self._text = text
        self.calls = 0

    async def ainvoke(self, msgs, **kw):
        self.calls += 1
        prompt = msgs[0].content
        assert "待压缩对话" in prompt and "订单A1001" in prompt
        return AIMessage(content=self._text)


def _settings():
    from app.config import Settings
    return Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                    database_url="mysql+pymysql://u:p@h/d")


async def _seed(store):
    sid = await store.create("u1")
    r = await store.commit_turn(sid, [StoredMessage("user", "订单A1001怎么了"),
                                      StoredMessage("assistant", "在处理,请稍等")])
    upto = int(r.message_ids[-1])
    await store.move_layer1_from(sid, "u1", upto)   # 全部落入层2
    return sid, upto


@pytest.mark.asyncio
async def test_summary_done_and_projection(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid, upto = await _seed(store)
    runner = SummaryRunner(store, _FakeModel(), _settings())
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        runner.maybe_trigger(sid, "u1")
        await asyncio.sleep(0.05)
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary_upto == upto and "A1001" in (meta.summary or "")
    assert "summary start" in caplog.text and "summary done" in caplog.text
    assert "第1段" in caplog.text
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_skip_empty_span_and_inflight(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid = await store.create("u1")                    # 无消息 → 区间为空
    model = _FakeModel()
    runner = SummaryRunner(store, model, _settings())
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        runner.maybe_trigger(sid, "u1")
        await asyncio.sleep(0.05)
    assert model.calls == 0 and "summary skip" in caplog.text
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_cas_failure_keeps_anchor(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid, upto = await _seed(store)
    await store.append_summary(sid, 0, upto, "别人先压完了", 10_000)  # 抢占锚点
    runner = SummaryRunner(store, _FakeModel(), _settings())
    with caplog.at_level(logging.INFO, logger="wayhelp.graph"):
        runner.maybe_trigger(sid, "u1")
        await asyncio.sleep(0.05)
    meta = await store.get_context_meta(sid, "u1")
    assert "别人先压完了" in (meta.summary or "")                      # 未被覆盖
    assert "summary skip" in caplog.text
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_truncates_overflow(caplog):
    store = InMemorySessionStore(10, 100, 8000)
    sid, upto = await _seed(store)
    runner = SummaryRunner(store, _FakeModel("长" * 500), _settings())
    runner.maybe_trigger(sid, "u1")
    await asyncio.sleep(0.05)
    meta = await store.get_context_meta(sid, "u1")
    assert meta.summary is not None and len(meta.summary) <= 200       # summary_max_chars
    await runner.aclose()
