"""
Agent — composition of LLM, tools, router, and instruction.

The agent defines behavior. The runner executes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .llm.base import LLM
    from .router import Router
    from .session import Session
    from .tool import Tool


class Agent:
    """An agent composed of an LLM, tools, router, and instruction."""

    def __init__(
        self,
        *,
        name: str,
        llm: LLM,
        instruction: str | Callable[[Session], str] | None = None,
        tools: list[Tool] | None = None,
        router: Router | None = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.instruction = instruction
        self.tools = tools or []
        self.router = router
        self._sub_agents: list[Agent] = []

    def add_sub_agent(self, agent: Agent) -> None:
        """Register a sub-agent."""
        self._sub_agents.append(agent)

    def resolve_instruction(self, session: Session) -> str | None:
        """Resolve instruction — static string or dynamic callable."""
        if self.instruction is None:
            return None
        if callable(self.instruction):
            return self.instruction(session)
        return self.instruction
