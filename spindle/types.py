"""
Core type definitions for Spindle.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventRole(StrEnum):
    """Who produced the event."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"


class EventType(StrEnum):
    """What kind of event this is."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class StepType(StrEnum):
    """What kind of step the runner yielded."""

    ROUTE_HANDLED = "route_handled"
    LLM_RESPONSE = "llm_response"
    LLM_CHUNK = "llm_chunk"
    TOOL_RESULT = "tool_result"


class ContentType(StrEnum):
    """Type of content part in a multimodal event."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"


class ContentPart(BaseModel):
    """A single part of multimodal content."""

    type: ContentType
    text: str | None = None
    mime_type: str | None = None
    data: bytes | None = None
    uri: str | None = None
    metadata: dict[str, Any] | None = None


class ThinkingConfig(BaseModel):
    """Configuration for LLM thinking/reasoning."""

    enabled: bool = False
    budget: int | None = None


class GenerateConfig(BaseModel):
    """Configuration for a single LLM generation call."""

    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    thinking: ThinkingConfig | None = None
    response_schema: dict[str, Any] | None = None
    provider_config: dict[str, Any] | None = None


class UsageMetadata(BaseModel):
    """Token usage from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int | None = None


class ToolCallData(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from an LLM generation call."""

    content: str | None = None
    tool_calls: list[ToolCallData] | None = None
    usage: UsageMetadata | None = None
    thinking: str | None = None
    model: str | None = None


class LLMChunk(BaseModel):
    """A single chunk from an LLM streaming response."""

    content_delta: str | None = None
    tool_calls: list[ToolCallData] | None = None
    thinking_delta: str | None = None
    usage: UsageMetadata | None = None
    finished: bool = False
