"""
In-memory store for testing and development.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import Store


class MemoryStore(Store):
    """Dict-backed in-memory store. No persistence across process restarts."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def save_session(self, session) -> None:

        session_id = session.id
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "id": session.id,
                "user_id": session.user_id,
                "events": [],
                "state": {},
            }
        self._sessions[session_id]["state"] = deepcopy(session.state)

    async def load_session(self, session_id: str):
        from spindle.session import Session

        data = self._sessions.get(session_id)
        if data is None:
            return None

        session = Session(
            id=data["id"],
            user_id=data["user_id"],
            state=deepcopy(data["state"]),
        )
        session._events = deepcopy(data["events"])
        return session

    async def append_events(self, session_id: str, events: list) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        self._sessions[session_id]["events"].extend(deepcopy(events))

    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        self._sessions[session_id]["state"] = deepcopy(state)
