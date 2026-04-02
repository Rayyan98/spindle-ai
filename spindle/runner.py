"""
Runner — the cold observable agent loop.

The runner is a lazy async generator. Each iteration executes exactly one step.
Between iterations, you can inspect results, inject messages, flush the session,
or stop. Nothing happens until you pull the next value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from .agent import Agent
from .session import Session
from .tool import Tool
from .types import GenerateConfig, StepType

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """A single step yielded by the runner."""

    type: StepType
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    tool_call_id: str | None = None
    usage: Any = None
    thinking: str | None = None


class Runner:
    """Pull-based agent runner. Each call to run() returns an async generator."""

    def __init__(self, *, agent: Agent) -> None:
        self.agent = agent

    async def run(
        self,
        session: Session,
        message: str,
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        config: GenerateConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Step, None]:
        """Run the agent loop as a lazy async generator.

        Args:
            session: The conversation session.
            message: The user's message.
            system_prompt: Override the agent's instruction for this run.
            tools: Override the agent's tools for this run.
            config: Generation config override for this run.
            metadata: Metadata passed to the router.
        """
        # Add user message to session
        session.add_user_message(message)

        # Try router first
        if self.agent.router:
            handled = await self.agent.router.try_route(session, message, metadata=metadata)
            if handled:
                yield Step(type=StepType.ROUTE_HANDLED)
                return

        # Resolve instruction and tools
        effective_prompt = system_prompt or self.agent.resolve_instruction(session)
        effective_tools = tools if tools is not None else self.agent.tools
        tool_map = {t.name: t for t in effective_tools}

        # Agent loop — LLM call → tool execution → repeat
        while True:
            response = await self.agent.llm.generate(
                session.events,
                system_prompt=effective_prompt,
                tools=effective_tools if effective_tools else None,
                config=config,
            )

            # Handle text response
            if response.content:
                session.add_agent_message(response.content, author=self.agent.name)
                yield Step(
                    type=StepType.LLM_RESPONSE,
                    content=response.content,
                    usage=response.usage,
                    thinking=response.thinking,
                )

            # No tool calls — turn complete
            if not response.tool_calls:
                break

            # Execute tool calls
            for tc in response.tool_calls:
                call_event = session.add_tool_call(
                    tc.name, tc.args, author=self.agent.name, call_id=tc.id
                )
                yield Step(
                    type=StepType.TOOL_CALL,
                    tool_name=tc.name,
                    tool_args=tc.args,
                    tool_call_id=tc.id,
                )

                # Execute the tool
                result = await _execute_tool(tool_map, tc.name, tc.args)

                session.add_tool_result(tc.name, result, call_id=call_event.tool_call_id)
                yield Step(
                    type=StepType.TOOL_RESULT,
                    tool_name=tc.name,
                    tool_result=result,
                    tool_call_id=call_event.tool_call_id,
                )


async def _execute_tool(tool_map: dict[str, Tool], name: str, args: dict[str, Any]) -> Any:
    """Execute a tool by name, returning the result or error."""
    t = tool_map.get(name)
    if t is None:
        return {"error": f"Tool '{name}' not found"}

    try:
        return await t.execute(args)
    except Exception as e:
        logger.error(f"Tool '{name}' raised: {e}")
        return {"error": str(e)}
