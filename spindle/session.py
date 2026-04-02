"""
Session — an ordered sequence of events with key-value state.

The session is an in-memory buffer. Events accumulate in memory and are only
persisted when you explicitly call flush(). This gives you full control over
when and what gets written to the database.
"""

from __future__ import annotations

from typing import Any

from .event import Event
from .stores.base import Store
from .types import EventRole, EventType


class Session:
    """Conversation session with in-memory event buffer and explicit flush."""

    def __init__(
        self,
        *,
        id: str,
        user_id: str,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.user_id = user_id
        self.state: dict[str, Any] = state or {}
        self._events: list[Event] = []
        self._pending: list[Event] = []

    # -- Properties ---------------------------------------------------------

    @property
    def events(self) -> list[Event]:
        """All events — flushed and pending."""
        return self._events

    @property
    def pending_events(self) -> list[Event]:
        """Events not yet flushed to persistence."""
        return self._pending

    # -- Add Events ---------------------------------------------------------

    def add_user_message(self, text: str, *, metadata: dict[str, Any] | None = None) -> Event:
        """Add a user message to the session."""
        event = Event.user_message(text, metadata=metadata)
        return self._append(event)

    def add_agent_message(
        self,
        text: str,
        *,
        author: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Add an agent message to the session."""
        event = Event.agent_message(text, author=author, metadata=metadata)
        return self._append(event)

    def add_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        author: str = "agent",
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Add a tool call event to the session."""
        event = Event.tool_call(name, args, author=author, call_id=call_id, metadata=metadata)
        return self._append(event)

    def add_tool_result(
        self,
        name: str,
        result: Any,
        *,
        call_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Add a tool result event to the session."""
        event = Event.tool_result(name, result, call_id=call_id, metadata=metadata)
        return self._append(event)

    # -- History ------------------------------------------------------------

    def history(
        self,
        *,
        roles: list[EventRole] | None = None,
        types: list[EventType] | None = None,
    ) -> list[Event]:
        """Return events, optionally filtered by role and/or type."""
        result = self._events
        if roles is not None:
            result = [e for e in result if e.role in roles]
        if types is not None:
            result = [e for e in result if e.type in types]
        return result

    # -- Flush --------------------------------------------------------------

    async def flush(self, store: Store) -> None:
        """Persist pending events and state to the store.

        After flush, pending_events is cleared but events remain in memory.
        """
        if store is None:
            raise TypeError("store cannot be None")

        await store.save_session(self)

        if self._pending:
            await store.append_events(self.id, self._pending)
            self._pending = []

        await store.save_state(self.id, self.state)

    # -- Internal -----------------------------------------------------------

    def _append(self, event: Event) -> Event:
        self._events.append(event)
        self._pending.append(event)
        return event
