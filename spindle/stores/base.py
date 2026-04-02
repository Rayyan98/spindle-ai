"""
Abstract base class for persistence stores.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spindle.event import Event
    from spindle.session import Session


class Store(abc.ABC):
    """Abstract persistence store for sessions and events."""

    @abc.abstractmethod
    async def save_session(self, session: Session) -> None:
        """Persist session metadata (id, user_id, state). Creates or updates."""

    @abc.abstractmethod
    async def load_session(self, session_id: str) -> Session | None:
        """Load a session with all its events. Returns None if not found."""

    @abc.abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to an existing session."""

    @abc.abstractmethod
    async def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist session state."""
