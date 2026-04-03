"""
Abstract LLM interface.

Providers implement this interface to integrate with different LLM backends.
The interface operates on Spindle Events — each provider handles conversion
to/from its native message format internally.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, AsyncGenerator

from ..types import GenerateConfig, LLMChunk, LLMResponse

if TYPE_CHECKING:
    from ..event import Event
    from ..tool import Tool


class LLM(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    async def generate(
        self,
        history: list[Event],
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse:
        """Generate a response given conversation history.

        Args:
            history: Ordered list of conversation events.
            system_prompt: System instruction for this generation call.
            tools: Tools available for the LLM to call.
            config: Generation configuration (temperature, max_tokens, etc.).

        Returns:
            LLMResponse with either text content, tool calls, or both.
        """

    async def stream(
        self,
        history: list[Event],
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        config: GenerateConfig | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Stream response chunks. Override for true streaming.

        Default implementation falls back to generate() and yields one chunk.
        """
        response = await self.generate(
            history, system_prompt=system_prompt, tools=tools, config=config
        )
        yield LLMChunk(
            content_delta=response.content,
            tool_calls=response.tool_calls,
            thinking_delta=response.thinking,
            usage=response.usage,
            finished=True,
        )
