"""
Gemini LLM provider via google-genai SDK.

Supports Vertex AI and API key authentication.
Converts between Spindle Events and Gemini message format.
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from ..event import Event
from ..tool import Tool
from ..types import (
    EventRole,
    EventType,
    GenerateConfig,
    LLMResponse,
    ToolCallData,
    UsageMetadata,
)
from .base import LLM

logger = logging.getLogger(__name__)


class GeminiLLM(LLM):
    """Gemini provider using google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        vertexai: bool = True,
        project: str | None = None,
        location: str = "us-central1",
        api_key: str | None = None,
    ) -> None:
        self.model = model

        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client(
                vertexai=vertexai,
                project=project,
                location=location,
            )

    async def generate(
        self,
        history: list[Event],
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse:
        contents = _events_to_contents(history)
        gen_config = _build_generate_config(system_prompt, tools, config)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=gen_config,
        )

        return _parse_response(response)


# -- Conversion: Spindle Events -> Gemini Contents --------------------------


def _events_to_contents(events: list[Event]) -> list[types.Content]:
    """Convert Spindle events to Gemini content messages."""
    contents: list[types.Content] = []

    for event in events:
        if event.type == EventType.MESSAGE:
            role = "user" if event.role == EventRole.USER else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=event.content or "")],
                )
            )

        elif event.type == EventType.TOOL_CALL:
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=event.tool_name,
                                args=event.tool_args or {},
                            ),
                        )
                    ],
                )
            )

        elif event.type == EventType.TOOL_RESULT:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=event.tool_name,
                                response=_ensure_dict(event.tool_result_data),
                            ),
                        )
                    ],
                )
            )

    return contents


def _ensure_dict(data: Any) -> dict:
    """Wrap non-dict data in a dict for Gemini's FunctionResponse."""
    if isinstance(data, dict):
        return data
    return {"result": data}


# -- Conversion: Tool -> Gemini FunctionDeclaration -------------------------


def _tools_to_gemini(tools: list[Tool]) -> list[types.Tool]:
    """Convert Spindle tools to Gemini tool declarations."""
    declarations = []
    for t in tools:
        schema = t.parameters_schema
        properties = {}
        for name, prop in schema.get("properties", {}).items():
            properties[name] = types.Schema(
                type=_json_type_to_gemini(prop.get("type", "string")),
            )

        declarations.append(
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=schema.get("required", []),
                )
                if properties
                else None,
            )
        )

    return [types.Tool(function_declarations=declarations)] if declarations else []


_TYPE_MAP = {
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
    "array": types.Type.ARRAY,
    "object": types.Type.OBJECT,
}


def _json_type_to_gemini(json_type: str) -> types.Type:
    return _TYPE_MAP.get(json_type, types.Type.STRING)


# -- Build GenerateContentConfig -------------------------------------------


def _build_generate_config(
    system_prompt: str | None,
    tools: list[Tool] | None,
    config: GenerateConfig | None,
) -> types.GenerateContentConfig:
    """Build Gemini's GenerateContentConfig from Spindle config."""
    kwargs: dict[str, Any] = {}

    if system_prompt:
        kwargs["system_instruction"] = system_prompt

    if tools:
        kwargs["tools"] = _tools_to_gemini(tools)

    if config:
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.max_tokens is not None:
            kwargs["max_output_tokens"] = config.max_tokens
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences
        if config.thinking and config.thinking.enabled:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=config.thinking.budget or 4096,
            )

    return types.GenerateContentConfig(**kwargs)


# -- Parse Gemini Response --------------------------------------------------


def _parse_response(response: types.GenerateContentResponse) -> LLMResponse:
    """Convert Gemini response to Spindle LLMResponse."""
    content_text: str | None = None
    tool_calls: list[ToolCallData] | None = None
    thinking_text: str | None = None

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_parts = []
            thinking_parts = []
            tc_list = []

            for part in candidate.content.parts:
                if part.function_call:
                    tc_list.append(
                        ToolCallData(
                            id=f"call_{part.function_call.name}_{id(part)}",
                            name=part.function_call.name,
                            args=dict(part.function_call.args) if part.function_call.args else {},
                        )
                    )
                elif part.thought:
                    thinking_parts.append(part.text or "")
                elif part.text:
                    text_parts.append(part.text)

            if text_parts:
                content_text = "".join(text_parts)
            if tc_list:
                tool_calls = tc_list
            if thinking_parts:
                thinking_text = "".join(thinking_parts)

    usage: UsageMetadata | None = None
    if response.usage_metadata:
        usage = UsageMetadata(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
            total_tokens=response.usage_metadata.total_token_count or 0,
        )

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        usage=usage,
        thinking=thinking_text,
        model=response.model_version if hasattr(response, "model_version") else None,
    )
