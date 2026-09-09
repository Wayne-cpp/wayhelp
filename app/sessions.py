import uuid
from dataclasses import dataclass
from typing import Protocol

from app.errors import SessionCapacityReachedError


@dataclass(frozen=True)
class StoredMessage:
    role: str  # "user" | "assistant"
    content: str


class SessionStore(Protocol):
    def create(self) -> str: ...
    def exists(self, session_id: str) -> bool: ...
    def snapshot(self, session_id: str) -> list[StoredMessage]: ...
    def commit_turn(self, session_id: str, user_text: str, assistant_text: str) -> None: ...


class InMemorySessionStore:
    def __init__(self, max_sessions: int, max_messages_per_session: int, max_message_chars: int):
        self._max_sessions = max_sessions
        self._max_messages = max_messages_per_session
        self._max_chars = max_message_chars
        self._sessions: dict[str, list[StoredMessage]] = {}

    def create(self) -> str:
        if len(self._sessions) >= self._max_sessions:
            raise SessionCapacityReachedError("session capacity reached")
        sid = str(uuid.uuid4())
        self._sessions[sid] = []
        return sid

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def snapshot(self, session_id: str) -> list[StoredMessage]:
        return list(self._sessions[session_id])

    def commit_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        if len(user_text) > self._max_chars or len(assistant_text) > self._max_chars:
            raise ValueError("message text exceeds MAX_MESSAGE_CHARS")
        msgs = self._sessions[session_id]
        msgs.append(StoredMessage("user", user_text))
        msgs.append(StoredMessage("assistant", assistant_text))
        while len(msgs) > self._max_messages:
            del msgs[:2]
