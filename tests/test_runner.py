"""Tests for spindle runner — the cold observable agent loop."""

from spindle.agent import Agent
from spindle.router import RouteContext, Router
from spindle.runner import Runner
from spindle.session import Session
from spindle.tool import tool
from spindle.types import (
    EventType,
    GenerateConfig,
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


def _make_agent(responses: list[LLMResponse], tools=None, router=None, instruction=None):
    llm = MockLLM(responses)
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


# -- Tool Calling -----------------------------------------------------------


class TestToolCalling:
    async def test_yields_tool_call_step(self):
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
        assert StepType.TOOL_CALL in step_types

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
        tool_call_steps = [s for s in steps if s.type == StepType.TOOL_CALL]
        tool_result_steps = [s for s in steps if s.type == StepType.TOOL_RESULT]
        assert len(tool_call_steps) == 2
        assert len(tool_result_steps) == 2

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
        """Breaking from the generator stops the agent — no more work done."""
        call_count = 0

        @tool
        async def slow_tool(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        agent = _make_agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCallData(id="c1", name="slow_tool", args={"x": 1}),
                        ToolCallData(id="c2", name="slow_tool", args={"x": 2}),
                        ToolCallData(id="c3", name="slow_tool", args={"x": 3}),
                    ]
                ),
                LLMResponse(content="Done"),
            ],
            tools=[slow_tool],
        )
        runner = Runner(agent=agent)
        session = Session(id="s1", user_id="u1")

        step_count = 0
        async for step in runner.run(session, "do 3 things"):
            step_count += 1
            if step_count >= 2:
                break

        # Should have stopped before processing all tool calls
        assert call_count < 3

    async def test_mid_run_injection(self):
        """Inject a user message between tool call and next LLM call."""
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
