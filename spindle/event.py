"""
Event model with factory methods for building conversation events.

Events are the atomic unit of conversation in Spindle. Every interaction —
user message, agent response, tool call, tool result — is an Event.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .types import EventRole, EventType


def _new_id() -> str:
    return str(uuid4())


def _now() -> float:
    return datetime.now().timestamp()


class Event(BaseModel):
    """A single event in a conversation."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    timestamp: float = Field(default_factory=_now)
    role: EventRole
    type: EventType
    author: str
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result_data: Any | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] | None = None

    # -- Factory Methods ----------------------------------------------------

    @staticmethod
    def user_message(text: str, *, metadata: dict[str, Any] | None = None) -> "Event":
        """Create a user message event."""
        return Event(
            role=EventRole.USER,
            type=EventType.MESSAGE,
            author="user",
            content=text,
            metadata=metadata,
        )

    @staticmethod
    def agent_message(
        text: str,
        *,
        author: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> "Event":
        """Create an agent message event."""
        return Event(
            role=EventRole.AGENT,
            type=EventType.MESSAGE,
            author=author,
            content=text,
            metadata=metadata,
        )

    @staticmethod
    def tool_call(
        name: str,
        args: dict[str, Any],
        *,
        author: str = "agent",
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Event":
        """Create a tool call event."""
        return Event(
            role=EventRole.AGENT,
            type=EventType.TOOL_CALL,
            author=author,
            tool_name=name,
            tool_args=args,
            tool_call_id=call_id or _new_id(),
            metadata=metadata,
        )

    @staticmethod
    def tool_result(
        name: str,
        result: Any,
        *,
        call_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> "Event":
        """Create a tool result event."""
        return Event(
            role=EventRole.TOOL,
            type=EventType.TOOL_RESULT,
            author="tool",
            tool_name=name,
            tool_result_data=result,
            tool_call_id=call_id,
            metadata=metadata,
        )
