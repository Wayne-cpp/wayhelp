"""知识库状态机(spec §6):ready | rebuilding | rebuild_required,进程内唯一持有者。"""

import threading
from enum import Enum


class KnowledgeState(str, Enum):
    READY = "ready"
    REBUILDING = "rebuilding"
    REBUILD_REQUIRED = "rebuild_required"


class KnowledgeStateHolder:
    def __init__(self, initial: KnowledgeState = KnowledgeState.READY):
        self._lock = threading.Lock()
        self._state = initial

    def get(self) -> str:
        with self._lock:
            return self._state.value

    def set(self, state: KnowledgeState) -> None:
        with self._lock:
            self._state = state
