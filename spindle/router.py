"""
Application-level router for handling messages before the LLM.

Routes run in registration order. Each route inspects the message and
either handles it (returns True) or passes through (returns False).
Routes can also enrich the session with synthetic events (tool calls,
results) without handling the message — the LLM then sees the enriched
context.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .session import Session


@dataclass
class RouteContext:
    """Context passed to route handlers."""

    session: Session
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Router:
    """Pre-LLM message router."""

    def __init__(self) -> None:
        self._routes: list[Callable] = []

    def route(self, fn: Callable) -> Callable:
        """Register a route handler.

        Handlers receive a RouteContext and return True (handled, skip LLM)
        or False (not handled, continue to LLM).

        Usage:
            @router.route
            async def greet(ctx: RouteContext) -> bool:
                if ctx.message.lower() in ("hello", "hi"):
                    ctx.session.add_agent_message("Hi!")
                    return True
                return False
        """
        self._routes.append(fn)
        return fn

    async def try_route(
        self,
        session: Session,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Try to route a message through registered handlers.

        Returns True if a handler handled the message (skip LLM).
        Returns False if no handler handled it (continue to LLM).
        """
        ctx = RouteContext(session=session, message=message, metadata=metadata or {})

        for handler in self._routes:
            if inspect.iscoroutinefunction(handler):
                result = await handler(ctx)
            else:
                result = handler(ctx)

            if result is True:
                return True

        return False
