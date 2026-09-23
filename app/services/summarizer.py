"""ch07 后台摘要任务(spec §9):触发只挪锚点不搬数据;CAS 追加,单调前移。"""

import asyncio
import contextlib
import logging
import time

from langchain_core.messages import HumanMessage

from app.prompts.summary import SUMMARY_PROMPT

logger = logging.getLogger("wayhelp.graph")


class SummaryRunner:
    def __init__(self, store, model, settings):
        self._store = store
        self._model = model
        self._settings = settings
        self._inflight: dict[str, asyncio.Task] = {}

    def maybe_trigger(self, session_id: str, user_id: str) -> None:
        task = self._inflight.get(session_id)
        if task is not None and not task.done():
            logger.info("summary skip session=%s reason=in-flight", session_id)
            return
        self._inflight[session_id] = asyncio.ensure_future(
            self._run_guarded(session_id, user_id))

    async def _run_guarded(self, session_id: str, user_id: str) -> None:
        try:
            await self._run(session_id, user_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("summary failed session=%s", session_id)

    async def _run(self, session_id: str, user_id: str) -> None:
        started = time.monotonic()
        meta = await self._store.get_context_meta(session_id, user_id)
        from_id = meta.summary_upto or 0
        upto = meta.layer1_from
        if upto is None or upto <= from_id:
            logger.info("summary skip session=%s reason=empty-span", session_id)
            return
        rows = [(i, r, c) for i, r, c in
                await self._store.fetch_span_texts(session_id, from_id, upto) if c]
        if not rows:
            logger.info("summary skip session=%s reason=empty-span", session_id)
            return
        logger.info("summary start session=%s 区间 (%d, %d]", session_id, from_id, upto)
        transcript = "\n".join(
            f"{'用户' if r == 'user' else '助手'}:{c}" for _, r, c in rows)
        prompt = (SUMMARY_PROMPT
                  .replace("{background}", meta.summary or "(无)")
                  .replace("{transcript}", transcript)
                  .replace("{max_chars}", str(self._settings.summary_max_chars)))
        resp = await self._model.ainvoke([HumanMessage(content=prompt)])
        content = (resp.content if isinstance(resp.content, str) else "").strip()
        if len(content) > self._settings.summary_max_chars:
            content = content[: self._settings.summary_max_chars]
            logger.info("summary truncated session=%s", session_id)
        result = await self._store.append_summary(
            session_id, from_id, upto, content, self._settings.summary_projection_tokens)
        if not result.applied:
            logger.info("summary skip session=%s reason=%s", session_id, result.reason)
            return
        logger.info("summary done session=%s 第%d段 覆盖 (%d, %d] 耗时 %dms",
                    session_id, result.seq, from_id, upto,
                    int((time.monotonic() - started) * 1000))

    async def aclose(self) -> None:
        for task in self._inflight.values():
            task.cancel()
        for task in self._inflight.values():
            # CancelledError 在 3.8+ 是 BaseException,suppress(Exception) 拦不住;
            # 不补这行的话 lifespan 关停取消 in-flight 摘要时会穿透炸掉收尾段
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._inflight.clear()
