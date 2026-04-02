"""
Abstract LLM interface.

Providers implement this interface to integrate with different LLM backends.
The interface operates on Spindle Events — each provider handles conversion
to/from its native message format internally.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from ..types import GenerateConfig, LLMResponse

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
