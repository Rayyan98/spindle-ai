"""Tests for spindle LLM abstraction and MockLLM."""

from spindle.event import Event
from spindle.llm.base import LLM
from spindle.tool import tool
from spindle.types import GenerateConfig, LLMChunk, LLMResponse, ToolCallData


# -- MockLLM for testing without real API calls -----------------------------


class MockLLM(LLM):
    """LLM that returns pre-configured responses in sequence."""

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        chunks: list[list[LLMChunk]] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._chunks = chunks
        self._call_count = 0
        self.call_history: list[dict] = []

    async def generate(
        self,
        history: list[Event],
        *,
        system_prompt: str | None = None,
        tools: list | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse:
        self.call_history.append(
            {
                "history": history,
                "system_prompt": system_prompt,
                "tools": tools,
                "config": config,
            }
        )
        if self._call_count >= len(self._responses):
            return LLMResponse(content="(no more responses)")
        response = self._responses[self._call_count]
        self._call_count += 1
        return response

    async def stream(self, history, *, system_prompt=None, tools=None, config=None):
        self.call_history.append(
            {
                "history": history,
                "system_prompt": system_prompt,
                "tools": tools,
                "config": config,
            }
        )
        if self._chunks and self._call_count < len(self._chunks):
            for chunk in self._chunks[self._call_count]:
                yield chunk
            self._call_count += 1
        else:
            async for chunk in super().stream(
                history, system_prompt=system_prompt, tools=tools, config=config
            ):
                yield chunk


# -- LLM Interface ---------------------------------------------------------


class TestMockLLM:
    async def test_returns_text_response(self):
        llm = MockLLM([LLMResponse(content="Hello!")])
        response = await llm.generate([Event.user_message("hi")])
        assert response.content == "Hello!"

    async def test_returns_tool_calls(self):
        llm = MockLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="search", args={"query": "test"}),
                    ]
                ),
            ]
        )
        response = await llm.generate([Event.user_message("search for test")])
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"

    async def test_returns_responses_in_sequence(self):
        llm = MockLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="search", args={"q": "test"}),
                    ]
                ),
                LLMResponse(content="Found 3 results."),
            ]
        )

        r1 = await llm.generate([Event.user_message("search")])
        assert r1.tool_calls is not None

        r2 = await llm.generate([Event.user_message("search")])
        assert r2.content == "Found 3 results."

    async def test_records_call_history(self):
        llm = MockLLM([LLMResponse(content="ok")])
        await llm.generate(
            [Event.user_message("hi")],
            system_prompt="Be helpful",
        )
        assert len(llm.call_history) == 1
        assert llm.call_history[0]["system_prompt"] == "Be helpful"

    async def test_passes_tools_to_generate(self):
        @tool
        async def search(query: str) -> list:
            """Search."""
            return []

        llm = MockLLM([LLMResponse(content="ok")])
        await llm.generate(
            [Event.user_message("hi")],
            tools=[search],
        )
        assert llm.call_history[0]["tools"] == [search]

    async def test_passes_config_to_generate(self):
        llm = MockLLM([LLMResponse(content="ok")])
        config = GenerateConfig(temperature=0.5, max_tokens=100)
        await llm.generate([Event.user_message("hi")], config=config)
        assert llm.call_history[0]["config"] == config

    async def test_structured_output_config(self):
        llm = MockLLM([LLMResponse(content='{"name": "test"}')])
        config = GenerateConfig(
            response_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        )
        await llm.generate([Event.user_message("give me json")], config=config)
        assert llm.call_history[0]["config"].response_schema is not None


# -- Default Stream Fallback ------------------------------------------------


class TestDefaultStream:
    async def test_fallback_yields_single_chunk(self):
        llm = MockLLM([LLMResponse(content="Hello!", thinking="I should greet")])
        chunks = [chunk async for chunk in llm.stream([Event.user_message("hi")])]

        assert len(chunks) == 1
        assert chunks[0].content_delta == "Hello!"
        assert chunks[0].thinking_delta == "I should greet"
        assert chunks[0].finished is True
