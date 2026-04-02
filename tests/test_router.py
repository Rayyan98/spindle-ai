"""Tests for spindle router."""

from spindle.router import RouteContext, Router
from spindle.session import Session
from spindle.types import EventRole


# -- Route Registration ----------------------------------------------------


class TestRouteRegistration:
    def test_register_route(self):
        router = Router()

        @router.route
        async def greet(ctx: RouteContext) -> bool:
            return False

        assert len(router._routes) == 1

    def test_register_multiple_routes(self):
        router = Router()

        @router.route
        async def greet(ctx: RouteContext) -> bool:
            return False

        @router.route
        async def upload(ctx: RouteContext) -> bool:
            return False

        assert len(router._routes) == 2


# -- Route Execution -------------------------------------------------------


class TestRouteExecution:
    async def test_returns_false_when_no_routes_match(self):
        router = Router()

        @router.route
        async def greet(ctx: RouteContext) -> bool:
            if ctx.message == "hello":
                return True
            return False

        session = Session(id="s1", user_id="u1")
        handled = await router.try_route(session, "what is 2+2?")
        assert handled is False

    async def test_returns_true_when_route_handles(self):
        router = Router()

        @router.route
        async def greet(ctx: RouteContext) -> bool:
            if ctx.message.lower() in ("hello", "hi"):
                ctx.session.add_agent_message("Hi there!")
                return True
            return False

        session = Session(id="s1", user_id="u1")
        handled = await router.try_route(session, "hello")
        assert handled is True

    async def test_handler_adds_events_to_session(self):
        router = Router()

        @router.route
        async def greet(ctx: RouteContext) -> bool:
            if ctx.message.lower() == "hello":
                ctx.session.add_agent_message("Hi!")
                return True
            return False

        session = Session(id="s1", user_id="u1")
        await router.try_route(session, "hello")

        assert len(session.events) == 1
        assert session.events[0].content == "Hi!"
        assert session.events[0].role == EventRole.AGENT

    async def test_routes_run_in_order(self):
        router = Router()
        call_order = []

        @router.route
        async def first(ctx: RouteContext) -> bool:
            call_order.append("first")
            return False

        @router.route
        async def second(ctx: RouteContext) -> bool:
            call_order.append("second")
            return True

        session = Session(id="s1", user_id="u1")
        await router.try_route(session, "test")
        assert call_order == ["first", "second"]

    async def test_first_true_stops_execution(self):
        router = Router()
        call_order = []

        @router.route
        async def first(ctx: RouteContext) -> bool:
            call_order.append("first")
            return True

        @router.route
        async def second(ctx: RouteContext) -> bool:
            call_order.append("second")
            return True

        session = Session(id="s1", user_id="u1")
        await router.try_route(session, "test")
        assert call_order == ["first"]

    async def test_enrichment_without_handling(self):
        """Route adds context (tool call) but returns False to continue to LLM."""
        router = Router()

        @router.route
        async def enrich(ctx: RouteContext) -> bool:
            if "receipt" in ctx.message:
                call = ctx.session.add_tool_call("ocr", {"text": ctx.message})
                ctx.session.add_tool_result("ocr", {"type": "invoice"}, call_id=call.tool_call_id)
            return False  # always pass to LLM

        session = Session(id="s1", user_id="u1")
        handled = await router.try_route(session, "here is a receipt")

        assert handled is False
        assert len(session.events) == 2  # tool_call + tool_result


# -- RouteContext -----------------------------------------------------------


class TestRouteContext:
    def test_has_session(self):
        session = Session(id="s1", user_id="u1")
        ctx = RouteContext(session=session, message="hello")
        assert ctx.session is session

    def test_has_message(self):
        session = Session(id="s1", user_id="u1")
        ctx = RouteContext(session=session, message="hello world")
        assert ctx.message == "hello world"

    def test_has_metadata(self):
        session = Session(id="s1", user_id="u1")
        ctx = RouteContext(
            session=session,
            message="hello",
            metadata={"attachment_type": "receipt"},
        )
        assert ctx.metadata["attachment_type"] == "receipt"

    def test_metadata_defaults_to_empty(self):
        session = Session(id="s1", user_id="u1")
        ctx = RouteContext(session=session, message="hello")
        assert ctx.metadata == {}


# -- Sync Routes ------------------------------------------------------------


class TestSyncRoutes:
    async def test_sync_route_function(self):
        router = Router()

        @router.route
        def greet(ctx: RouteContext) -> bool:
            if ctx.message == "hello":
                ctx.session.add_agent_message("Hi!")
                return True
            return False

        session = Session(id="s1", user_id="u1")
        handled = await router.try_route(session, "hello")
        assert handled is True
        assert session.events[0].content == "Hi!"
