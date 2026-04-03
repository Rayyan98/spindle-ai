"""
Runner — the cold observable agent loop.

The runner is a lazy async generator. Each iteration executes exactly one step.
Between iterations, you can inspect results, inject messages, flush the session,
or stop. Nothing happens until you pull the next value.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from .agent import Agent
from .session import Session
from .tool import Tool
from .types import CodeExecution, GenerateConfig, StepType, ToolCallData

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """A single step yielded by the runner."""

    type: StepType
    content: str | None = None
    tool_calls: list[ToolCallData] | None = None
    code_executions: list[CodeExecution] | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    tool_call_id: str | None = None
    usage: Any = None
    thinking: str | None = None
    partial: bool = False


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
        stream: bool = False,
    ) -> AsyncGenerator[Step, None]:
        """Run the agent loop as a lazy async generator.

        Args:
            session: The conversation session.
            message: The user's message.
            system_prompt: Override the agent's instruction for this run.
            tools: Override the agent's tools for this run.
            config: Generation config override for this run.
            metadata: Metadata passed to the router.
            stream: If True, yield LLM_CHUNK steps with partial text deltas.
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
            if stream:
                response = None
                async for step in self._stream_llm(
                    session, effective_prompt, effective_tools, config
                ):
                    yield step
                    if not step.partial and step.type == StepType.LLM_RESPONSE:
                        response = step
            else:
                llm_response = await self.agent.llm.generate(
                    session.events,
                    system_prompt=effective_prompt,
                    tools=effective_tools if effective_tools else None,
                    config=config,
                )

                # Add agent message to session if there's text content
                if llm_response.content:
                    meta = {}
                    if llm_response.thinking:
                        meta["thinking"] = llm_response.thinking
                    session.add_agent_message(
                        llm_response.content,
                        author=self.agent.name,
                        metadata=meta or None,
                    )

                # Yield LLM_RESPONSE — the semantic unit from the LLM
                response = Step(
                    type=StepType.LLM_RESPONSE,
                    content=llm_response.content,
                    tool_calls=llm_response.tool_calls,
                    code_executions=llm_response.code_executions,
                    usage=llm_response.usage,
                    thinking=llm_response.thinking,
                )
                yield response

            # No tool calls — turn complete
            if not response or not response.tool_calls:
                break

            # Execute tool calls in parallel
            call_events = []
            for tc in response.tool_calls:
                call_event = session.add_tool_call(
                    tc.name, tc.args, author=self.agent.name, call_id=tc.id
                )
                call_events.append((tc, call_event))

            results = await asyncio.gather(
                *[_execute_tool(tool_map, tc.name, tc.args) for tc, _ in call_events],
                return_exceptions=True,
            )

            for (tc, call_event), result in zip(call_events, results):
                if isinstance(result, BaseException):
                    result = {"error": str(result)}

                session.add_tool_result(tc.name, result, call_id=call_event.tool_call_id)
                yield Step(
                    type=StepType.TOOL_RESULT,
                    tool_name=tc.name,
                    tool_result=result,
                    tool_call_id=call_event.tool_call_id,
                )

    async def _stream_llm(
        self,
        session: Session,
        system_prompt: str | None,
        tools: list[Tool],
        config: GenerateConfig | None,
    ) -> AsyncGenerator[Step, None]:
        """Stream LLM response, yielding chunks then a final LLM_RESPONSE."""
        accumulated_text = ""
        accumulated_thinking = ""
        accumulated_tool_calls: list[ToolCallData] | None = None
        last_usage = None

        async for chunk in self.agent.llm.stream(
            session.events,
            system_prompt=system_prompt,
            tools=tools if tools else None,
            config=config,
        ):
            if chunk.content_delta:
                accumulated_text += chunk.content_delta
                yield Step(
                    type=StepType.LLM_CHUNK,
                    content=chunk.content_delta,
                    partial=True,
                )

            if chunk.thinking_delta:
                accumulated_thinking += chunk.thinking_delta

            if chunk.tool_calls:
                accumulated_tool_calls = chunk.tool_calls

            if chunk.usage:
                last_usage = chunk.usage

        # Add agent message to session
        final_text = accumulated_text or None
        if final_text:
            meta = {}
            if accumulated_thinking:
                meta["thinking"] = accumulated_thinking
            session.add_agent_message(
                final_text,
                author=self.agent.name,
                metadata=meta or None,
            )

        # Yield final LLM_RESPONSE step
        yield Step(
            type=StepType.LLM_RESPONSE,
            content=final_text,
            tool_calls=accumulated_tool_calls,
            usage=last_usage,
            thinking=accumulated_thinking or None,
            partial=False,
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
