"""
End-to-end tests with real Gemini models via Vertex AI.

These tests make actual API calls and cost money. Run with:
    pytest -m e2e -v
"""

import json

import pytest

from spindle import Agent, GenerateConfig, Runner, Session, StepType, ThinkingConfig, tool
from spindle.llm.gemini import GeminiLLM

pytestmark = pytest.mark.e2e


# -- Fixtures ---------------------------------------------------------------


MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


@tool
async def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    weather_data = {
        "riyadh": {"temp": 42, "condition": "sunny"},
        "london": {"temp": 15, "condition": "cloudy"},
        "tokyo": {"temp": 28, "condition": "humid"},
    }
    return weather_data.get(city.lower(), {"temp": 20, "condition": "unknown"})


@tool
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# -- Basic Text Response ----------------------------------------------------


class TestBasicResponse:
    @pytest.mark.parametrize("model", MODELS)
    async def test_responds_to_greeting(self, model: str):
        llm = GeminiLLM(model=model)
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Respond to greetings briefly in one sentence.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "Hello!")]
        assert len(steps) >= 1
        response_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]
        assert len(response_steps) >= 1
        assert len(response_steps[-1].content) > 0

    async def test_follows_system_prompt(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Always respond with exactly the word 'PONG' and nothing else.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "ping")]
        response_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]
        assert "PONG" in response_steps[-1].content.upper()


# -- Tool Calling -----------------------------------------------------------


class TestToolCalling:
    @pytest.mark.parametrize("model", MODELS)
    async def test_calls_weather_tool(self, model: str):
        llm = GeminiLLM(model=model)
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Use the get_weather tool to answer weather questions. Be brief.",
            tools=[get_weather],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "What's the weather in Riyadh?")]
        step_types = [s.type for s in steps]

        # Tool calls are on the LLM_RESPONSE step
        llm_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]
        assert any(s.tool_calls for s in llm_steps)

        assert StepType.TOOL_RESULT in step_types
        assert StepType.LLM_RESPONSE in step_types

        # The final response should mention the temperature or weather
        final_response = [s for s in steps if s.type == StepType.LLM_RESPONSE][-1]
        assert any(word in final_response.content.lower() for word in ["42", "sunny", "riyadh"])

    async def test_calls_add_tool(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Use the add_numbers tool to do math. Be brief.",
            tools=[add_numbers],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "What is 17 + 25?")]
        result_steps = [s for s in steps if s.type == StepType.TOOL_RESULT]
        assert any(s.tool_result == 42 for s in result_steps)


# -- Session Integrity ------------------------------------------------------


class TestSessionIntegrity:
    async def test_multi_turn_conversation(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Be brief. Remember what the user says.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        # Turn 1
        [step async for step in runner.run(session, "My name is Ahmed")]

        # Turn 2
        steps = [step async for step in runner.run(session, "What is my name?")]
        final = [s for s in steps if s.type == StepType.LLM_RESPONSE][-1]
        assert "ahmed" in final.content.lower()

    async def test_session_events_after_tool_call(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Use tools when appropriate. Be brief.",
            tools=[get_weather],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "Weather in London?")]

        # Session should have: user_msg, tool_call, tool_result, agent_msg (minimum)
        assert len(session.events) >= 4
        assert session.events[0].content == "Weather in London?"


# -- Per-message Controls ---------------------------------------------------


class TestPerMessageControls:
    async def test_override_system_prompt(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="You are a pirate. Respond like a pirate.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        # Override the pirate instruction
        steps = [
            step
            async for step in runner.run(
                session,
                "Say hello",
                system_prompt="Always respond with exactly 'HELLO WORLD' and nothing else.",
            )
        ]
        final = [s for s in steps if s.type == StepType.LLM_RESPONSE][-1]
        assert "HELLO" in final.content.upper()

    async def test_override_tools(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Use tools when asked. Be brief.",
            tools=[get_weather],  # default tools
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        # Override with different tools
        steps = [
            step
            async for step in runner.run(
                session,
                "What is 5 + 3?",
                tools=[add_numbers],
            )
        ]
        result_steps = [s for s in steps if s.type == StepType.TOOL_RESULT]
        assert any(s.tool_result == 8 for s in result_steps)


# -- Thinking ---------------------------------------------------------------


class TestThinking:
    async def test_thinking_with_gemini_2_5(self):
        llm = GeminiLLM(model="gemini-2.5-flash")
        agent = Agent(name="test", llm=llm)
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        config = GenerateConfig(thinking=ThinkingConfig(enabled=True, budget=2048))
        steps = [
            step
            async for step in runner.run(
                session,
                "What is the capital of France?",
                config=config,
            )
        ]
        response = [s for s in steps if s.type == StepType.LLM_RESPONSE][-1]
        assert response.content is not None
        assert "paris" in response.content.lower()


# -- Structured Output -----------------------------------------------------


class TestStructuredOutput:
    async def test_json_schema_response(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Extract the person's name and age from the text.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        config = GenerateConfig(
            response_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }
        )
        steps = [
            step
            async for step in runner.run(
                session,
                "Ahmed is 28 years old.",
                config=config,
            )
        ]
        response = [s for s in steps if s.type == StepType.LLM_RESPONSE][-1]
        data = json.loads(response.content)
        assert data["name"].lower() == "ahmed"
        assert data["age"] == 28


# -- Streaming -------------------------------------------------------------


class TestStreaming:
    async def test_streaming_returns_chunks_and_response(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Say hello briefly.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "Hi!", stream=True)]

        chunk_steps = [s for s in steps if s.type == StepType.LLM_CHUNK]
        response_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]

        assert len(chunk_steps) >= 1
        assert len(response_steps) == 1
        assert all(s.partial for s in chunk_steps)
        assert not response_steps[0].partial

    async def test_streaming_with_tool_call(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Use tools when asked. Be brief.",
            tools=[add_numbers],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "What is 3 + 4?", stream=True)]
        step_types = [s.type for s in steps]

        assert StepType.LLM_RESPONSE in step_types
        assert StepType.TOOL_RESULT in step_types


# -- Code Execution --------------------------------------------------------


class TestCodeExecution:
    async def test_code_execution_returns_result(self):
        llm = GeminiLLM(model="gemini-2.0-flash")
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Use code execution to compute the answer. Be brief.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        config = GenerateConfig(code_execution=True)
        steps = [
            step
            async for step in runner.run(
                session,
                "What is the 10th fibonacci number? Use code to compute it.",
                config=config,
            )
        ]

        response = [s for s in steps if s.type == StepType.LLM_RESPONSE][-1]
        assert response.content is not None
        # The model should have executed code
        all_code_execs = [
            s.code_executions
            for s in steps
            if s.type == StepType.LLM_RESPONSE and s.code_executions
        ]
        assert len(all_code_execs) >= 1
