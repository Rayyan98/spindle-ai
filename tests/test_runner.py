"""Tests for spindle runner — the cold observable agent loop."""

from spindle.agent import Agent
from spindle.router import RouteContext, Router
from spindle.runner import Runner
from spindle.session import Session
from spindle.tool import tool
from spindle.types import (
    EventType,
    GenerateConfig,
    LLMChunk,
    LLMResponse,
    StepType,
    ToolCallData,
)
from tests.test_llm import MockLLM


# -- Fixtures ---------------------------------------------------------------


@tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
async def search(query: str) -> list:
    """Search for items."""
    return [{"name": "MacBook", "price": 999}, {"name": "ThinkPad", "price": 799}]


@tool
async def failing_tool(x: int) -> dict:
    """A tool that always fails."""
    raise ValueError("something broke")


def _make_agent(responses, tools=None, router=None, instruction=None, chunks=None):
    llm = MockLLM(responses, chunks=chunks)
    return Agent(
        name="test_agent",
        llm=llm,
        tools=tools or [],
        router=router,
        instruction=instruction or "You are a test assistant.",
    )


# -- Basic Text Response ----------------------------------------------------


class TestTextResponse:
    async def test_yields_llm_response_step(self):
        agent = _make_agent([LLMResponse(content="Hello!")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "hi")]
        assert len(steps) == 1
        assert steps[0].type == StepType.LLM_RESPONSE
        assert steps[0].content == "Hello!"

    async def test_adds_user_message_to_session(self):
        agent = _make_agent([LLMResponse(content="Hello!")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "hi")]
        assert session.events[0].content == "hi"
        assert session.events[0].role.value == "user"

    async def test_adds_agent_response_to_session(self):
        agent = _make_agent([LLMResponse(content="Hello!")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "hi")]
        assert session.events[1].content == "Hello!"
        assert session.events[1].role.value == "agent"


# -- Semantic Grouping ------------------------------------------------------


class TestSemanticGrouping:
    async def test_llm_response_carries_tool_calls(self):
        """LLM_RESPONSE step includes tool_calls when LLM wants to call tools."""
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 2, "b": 3}),
                    ]
                ),
                LLMResponse(content="The answer is 5."),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add 2 and 3")]
        llm_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]

        # First LLM response has tool calls
        assert llm_steps[0].tool_calls is not None
        assert llm_steps[0].tool_calls[0].name == "add"

        # Second LLM response has text
        assert llm_steps[1].content == "The answer is 5."
        assert llm_steps[1].tool_calls is None

    async def test_text_and_tool_calls_in_one_step(self):
        """When LLM returns both text and tool calls, they're in one step."""
        agent = _make_agent(
            [
                LLMResponse(
                    content="Let me search for that.",
                    tool_calls=[
                        ToolCallData(id="c1", name="search", args={"query": "laptop"}),
                    ],
                ),
                LLMResponse(content="Found 2 results."),
            ],
            tools=[search],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "find laptops")]
        first = steps[0]

        assert first.type == StepType.LLM_RESPONSE
        assert first.content == "Let me search for that."
        assert first.tool_calls is not None
        assert first.tool_calls[0].name == "search"

    async def test_no_separate_tool_call_steps(self):
        """StepType for tool_call should not appear as a yielded step."""
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 1, "b": 2}),
                    ]
                ),
                LLMResponse(content="3"),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add")]
        step_types = [s.type for s in steps]
        assert StepType.LLM_RESPONSE in step_types
        assert StepType.TOOL_RESULT in step_types
        assert all(t in (StepType.LLM_RESPONSE, StepType.TOOL_RESULT) for t in step_types)


# -- Tool Calling -----------------------------------------------------------


class TestToolCalling:
    async def test_yields_tool_result_step(self):
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 2, "b": 3}),
                    ]
                ),
                LLMResponse(content="The answer is 5."),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add 2 and 3")]
        step_types = [s.type for s in steps]
        assert StepType.TOOL_RESULT in step_types

    async def test_tool_result_contains_actual_result(self):
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 2, "b": 3}),
                    ]
                ),
                LLMResponse(content="5"),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add 2+3")]
        result_steps = [s for s in steps if s.type == StepType.TOOL_RESULT]
        assert result_steps[0].tool_result == 5

    async def test_full_tool_loop_session_events(self):
        """User msg -> tool call -> tool result -> agent response."""
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="search", args={"query": "laptop"}),
                    ]
                ),
                LLMResponse(content="Found 2 laptops."),
            ],
            tools=[search],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "search for laptops")]

        assert len(session.events) == 4
        assert session.events[0].type == EventType.MESSAGE  # user
        assert session.events[1].type == EventType.TOOL_CALL
        assert session.events[2].type == EventType.TOOL_RESULT
        assert session.events[3].type == EventType.MESSAGE  # agent

    async def test_multiple_tool_calls_in_one_response(self):
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 1, "b": 2}),
                        ToolCallData(id="c2", name="add", args={"a": 3, "b": 4}),
                    ]
                ),
                LLMResponse(content="Results: 3 and 7."),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add both")]

        # LLM_RESPONSE carries both tool calls
        llm_step = [s for s in steps if s.type == StepType.LLM_RESPONSE][0]
        assert len(llm_step.tool_calls) == 2

        # Two TOOL_RESULT steps
        result_steps = [s for s in steps if s.type == StepType.TOOL_RESULT]
        assert len(result_steps) == 2

    async def test_tool_error_returns_error_result(self):
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="failing_tool", args={"x": 1}),
                    ]
                ),
                LLMResponse(content="Sorry, that failed."),
            ],
            tools=[failing_tool],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "do thing")]
        result_steps = [s for s in steps if s.type == StepType.TOOL_RESULT]
        assert "error" in str(result_steps[0].tool_result).lower()


# -- Parallel Tool Execution -----------------------------------------------


class TestParallelToolExecution:
    async def test_parallel_results_all_returned(self):
        """All tool call results should be returned even when run in parallel."""
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 1, "b": 2}),
                        ToolCallData(id="c2", name="add", args={"a": 10, "b": 20}),
                        ToolCallData(id="c3", name="add", args={"a": 100, "b": 200}),
                    ]
                ),
                LLMResponse(content="Done."),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add all")]
        results = [s.tool_result for s in steps if s.type == StepType.TOOL_RESULT]
        assert sorted(results) == [3, 30, 300]

    async def test_parallel_one_failure_others_succeed(self):
        """If one tool fails, others still return results."""

        @tool
        async def maybe_fail(x: int) -> int:
            """Maybe fail."""
            if x == 2:
                raise ValueError("boom")
            return x * 10

        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="maybe_fail", args={"x": 1}),
                        ToolCallData(id="c2", name="maybe_fail", args={"x": 2}),
                        ToolCallData(id="c3", name="maybe_fail", args={"x": 3}),
                    ]
                ),
                LLMResponse(content="Done."),
            ],
            tools=[maybe_fail],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "run all")]
        results = [s for s in steps if s.type == StepType.TOOL_RESULT]

        assert results[0].tool_result == 10
        assert "error" in str(results[1].tool_result).lower()
        assert results[2].tool_result == 30


# -- Router Integration ----------------------------------------------------


class TestRouterIntegration:
    async def test_router_handles_message(self):
        router = Router()

        @router.route
        async def greet(ctx: RouteContext) -> bool:
            if ctx.message == "hello":
                ctx.session.add_agent_message("Hi!")
                return True
            return False

        agent = _make_agent([LLMResponse(content="should not reach")], router=router)
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "hello")]

        assert len(steps) == 1
        assert steps[0].type == StepType.ROUTE_HANDLED
        assert session.events[-1].content == "Hi!"

    async def test_router_enriches_then_passes_to_llm(self):
        router = Router()

        @router.route
        async def enrich(ctx: RouteContext) -> bool:
            if "receipt" in ctx.message:
                call = ctx.session.add_tool_call("ocr", {"text": ctx.message})
                ctx.session.add_tool_result("ocr", {"type": "invoice"}, call_id=call.tool_call_id)
            return False

        agent = _make_agent(
            [LLMResponse(content="I see an invoice.")],
            router=router,
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "here is a receipt")]
        llm_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]
        assert len(llm_steps) == 1
        assert llm_steps[0].content == "I see an invoice."

        # Session has: user_msg, tool_call, tool_result, agent_msg
        assert len(session.events) == 4

    async def test_no_router_goes_straight_to_llm(self):
        agent = _make_agent([LLMResponse(content="Hi!")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "hello")]
        assert steps[0].type == StepType.LLM_RESPONSE


# -- Pull-based Control ----------------------------------------------------


class TestPullBasedControl:
    async def test_stop_early_via_break(self):
        """Breaking from the generator stops yielding further steps."""
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="add", args={"a": 1, "b": 2}),
                        ToolCallData(id="c2", name="add", args={"a": 3, "b": 4}),
                        ToolCallData(id="c3", name="add", args={"a": 5, "b": 6}),
                    ]
                ),
                LLMResponse(content="Done"),
            ],
            tools=[add],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        collected = []
        async for step in runner.run(session, "do 3 things"):
            collected.append(step)
            if len(collected) >= 2:
                break

        # First step is LLM_RESPONSE, then we only got 1 TOOL_RESULT before breaking
        assert collected[0].type == StepType.LLM_RESPONSE
        assert collected[1].type == StepType.TOOL_RESULT
        assert len(collected) == 2

    async def test_mid_run_injection(self):
        """Inject a user message between tool result and next LLM call."""
        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="search", args={"query": "all"}),
                    ]
                ),
                LLMResponse(content="Filtered results."),
            ],
            tools=[search],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        async for step in runner.run(session, "find suppliers"):
            if step.type == StepType.TOOL_RESULT:
                session.add_user_message("only show Riyadh suppliers")

        # The injected message should be in the session
        user_messages = [e for e in session.events if e.role.value == "user"]
        assert len(user_messages) == 2
        assert user_messages[1].content == "only show Riyadh suppliers"


# -- Thinking ---------------------------------------------------------------


class TestThinking:
    async def test_thinking_flows_to_step(self):
        agent = _make_agent([LLMResponse(content="Paris", thinking="France's capital is Paris")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "capital of France?")]
        assert steps[0].thinking == "France's capital is Paris"

    async def test_thinking_stored_in_event_metadata(self):
        agent = _make_agent([LLMResponse(content="42", thinking="The answer to everything")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "meaning of life?")]

        agent_event = session.events[1]
        assert agent_event.metadata is not None
        assert agent_event.metadata["thinking"] == "The answer to everything"


# -- Streaming --------------------------------------------------------------


class TestStreaming:
    async def test_stream_yields_chunks_then_response(self):
        agent = _make_agent(
            [LLMResponse(content="Hello world")],
            chunks=[
                [
                    LLMChunk(content_delta="Hello ", finished=False),
                    LLMChunk(content_delta="world", finished=False),
                    LLMChunk(finished=True),
                ]
            ],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "hi", stream=True)]

        chunk_steps = [s for s in steps if s.type == StepType.LLM_CHUNK]
        assert len(chunk_steps) == 2
        assert chunk_steps[0].content == "Hello "
        assert chunk_steps[1].content == "world"
        assert all(s.partial for s in chunk_steps)

        response_steps = [s for s in steps if s.type == StepType.LLM_RESPONSE]
        assert len(response_steps) == 1
        assert response_steps[0].content == "Hello world"
        assert not response_steps[0].partial

    async def test_stream_false_is_default(self):
        agent = _make_agent([LLMResponse(content="Hi!")])
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "hello")]
        assert all(s.type != StepType.LLM_CHUNK for s in steps)

    async def test_stream_with_tool_calls(self):
        agent = _make_agent(
            [LLMResponse(content="ok")],
            tools=[add],
            chunks=[
                [
                    LLMChunk(content_delta="Let me ", finished=False),
                    LLMChunk(content_delta="add.", finished=False),
                    LLMChunk(
                        tool_calls=[ToolCallData(id="c1", name="add", args={"a": 1, "b": 2})],
                        finished=True,
                    ),
                ],
                [
                    LLMChunk(content_delta="The answer is 3.", finished=False),
                    LLMChunk(finished=True),
                ],
            ],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        steps = [step async for step in runner.run(session, "add 1+2", stream=True)]

        step_types = [s.type for s in steps]
        assert StepType.LLM_CHUNK in step_types
        assert StepType.LLM_RESPONSE in step_types
        assert StepType.TOOL_RESULT in step_types


# -- Per-message Overrides --------------------------------------------------


class TestPerMessageOverrides:
    async def test_override_system_prompt(self):
        llm = MockLLM([LLMResponse(content="ok")])
        agent = Agent(
            name="test",
            llm=llm,
            instruction="default instruction",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "hi", system_prompt="custom instruction")]

        assert llm.call_history[0]["system_prompt"] == "custom instruction"

    async def test_override_tools(self):
        llm = MockLLM([LLMResponse(content="ok")])
        agent = Agent(
            name="test",
            llm=llm,
            tools=[search],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "hi", tools=[add])]

        assert llm.call_history[0]["tools"] == [add]

    async def test_override_config(self):
        llm = MockLLM([LLMResponse(content="ok")])
        agent = Agent(name="test", llm=llm)
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        config = GenerateConfig(temperature=0.0)
        [step async for step in runner.run(session, "hi", config=config)]

        assert llm.call_history[0]["config"] == config

    async def test_default_uses_agent_instruction(self):
        llm = MockLLM([LLMResponse(content="ok")])
        agent = Agent(
            name="test",
            llm=llm,
            instruction="Be helpful and concise.",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        [step async for step in runner.run(session, "hi")]

        assert llm.call_history[0]["system_prompt"] == "Be helpful and concise."

    async def test_dynamic_instruction(self):
        llm = MockLLM([LLMResponse(content="ok")])
        agent = Agent(
            name="test",
            llm=llm,
            instruction=lambda session: f"User is {session.state.get('name', 'unknown')}",
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1", state={"name": "Ahmed"})

        [step async for step in runner.run(session, "hi")]

        assert llm.call_history[0]["system_prompt"] == "User is Ahmed"
