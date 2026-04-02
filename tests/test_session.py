"""Tests for spindle session model."""

import pytest

from spindle.event import Event
from spindle.session import Session
from spindle.stores.memory import MemoryStore
from spindle.types import EventRole, EventType


# -- Construction -----------------------------------------------------------


class TestConstruction:
    def test_creates_with_id_and_user_id(self):
        session = Session(id="s1", user_id="u1")
        assert session.id == "s1"
        assert session.user_id == "u1"

    def test_starts_with_empty_events(self):
        session = Session(id="s1", user_id="u1")
        assert session.events == []

    def test_starts_with_empty_state(self):
        session = Session(id="s1", user_id="u1")
        assert session.state == {}

    def test_starts_with_empty_pending(self):
        session = Session(id="s1", user_id="u1")
        assert session.pending_events == []

    def test_initial_state(self):
        session = Session(id="s1", user_id="u1", state={"timezone": "Asia/Riyadh"})
        assert session.state["timezone"] == "Asia/Riyadh"


# -- Adding Events ----------------------------------------------------------


class TestAddUserMessage:
    def test_adds_to_events(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        assert len(session.events) == 1
        assert session.events[0].content == "hello"

    def test_adds_to_pending(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        assert len(session.pending_events) == 1

    def test_returns_created_event(self):
        session = Session(id="s1", user_id="u1")
        event = session.add_user_message("hello")
        assert isinstance(event, Event)
        assert event.role == EventRole.USER
        assert event.content == "hello"


class TestAddAgentMessage:
    def test_adds_to_events(self):
        session = Session(id="s1", user_id="u1")
        session.add_agent_message("Hi there!")
        assert len(session.events) == 1
        assert session.events[0].content == "Hi there!"

    def test_custom_author(self):
        session = Session(id="s1", user_id="u1")
        session.add_agent_message("Processing...", author="receipt_agent")
        assert session.events[0].author == "receipt_agent"


class TestAddToolCall:
    def test_adds_to_events(self):
        session = Session(id="s1", user_id="u1")
        session.add_tool_call("search", {"query": "laptop"})
        assert len(session.events) == 1
        assert session.events[0].tool_name == "search"
        assert session.events[0].tool_args == {"query": "laptop"}

    def test_returns_event_with_call_id(self):
        session = Session(id="s1", user_id="u1")
        event = session.add_tool_call("search", {"query": "laptop"})
        assert event.tool_call_id is not None


class TestAddToolResult:
    def test_adds_to_events(self):
        session = Session(id="s1", user_id="u1")
        session.add_tool_result("search", [{"name": "MacBook"}], call_id="c1")
        assert len(session.events) == 1
        assert session.events[0].tool_result_data == [{"name": "MacBook"}]

    def test_links_to_call_id(self):
        session = Session(id="s1", user_id="u1")
        call = session.add_tool_call("search", {"query": "laptop"})
        result = session.add_tool_result("search", [{"name": "MacBook"}], call_id=call.tool_call_id)
        assert result.tool_call_id == call.tool_call_id


# -- Event Ordering ---------------------------------------------------------


class TestEventOrdering:
    def test_events_maintain_insertion_order(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        session.add_agent_message("hi")
        session.add_user_message("how are you")

        assert session.events[0].content == "hello"
        assert session.events[1].content == "hi"
        assert session.events[2].content == "how are you"


# -- History Filtering ------------------------------------------------------


class TestHistory:
    def test_returns_all_events_by_default(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        session.add_agent_message("hi")
        assert len(session.history()) == 2

    def test_filter_by_role(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        session.add_agent_message("hi")
        session.add_user_message("bye")

        user_events = session.history(roles=[EventRole.USER])
        assert len(user_events) == 2
        assert all(e.role == EventRole.USER for e in user_events)

    def test_filter_by_type(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("search for laptops")
        call = session.add_tool_call("search", {"query": "laptops"})
        session.add_tool_result("search", [], call_id=call.tool_call_id)

        tool_events = session.history(types=[EventType.TOOL_CALL, EventType.TOOL_RESULT])
        assert len(tool_events) == 2

    def test_filter_by_role_and_type(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        session.add_agent_message("hi")
        call = session.add_tool_call("search", {"query": "test"})
        session.add_tool_result("search", [], call_id=call.tool_call_id)

        agent_messages = session.history(roles=[EventRole.AGENT], types=[EventType.MESSAGE])
        assert len(agent_messages) == 1
        assert agent_messages[0].content == "hi"


# -- State ------------------------------------------------------------------


class TestState:
    def test_set_and_get_state(self):
        session = Session(id="s1", user_id="u1")
        session.state["timezone"] = "Asia/Riyadh"
        assert session.state["timezone"] == "Asia/Riyadh"

    def test_update_state(self):
        session = Session(id="s1", user_id="u1", state={"a": 1})
        session.state["b"] = 2
        assert session.state == {"a": 1, "b": 2}

    def test_delete_state_key(self):
        session = Session(id="s1", user_id="u1", state={"a": 1, "b": 2})
        del session.state["a"]
        assert "a" not in session.state


# -- Flush ------------------------------------------------------------------


class TestFlush:
    async def test_flush_clears_pending(self):
        store = MemoryStore()
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        session.add_agent_message("hi")
        assert len(session.pending_events) == 2

        await session.flush(store)
        assert len(session.pending_events) == 0

    async def test_flush_preserves_events(self):
        store = MemoryStore()
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        await session.flush(store)

        assert len(session.events) == 1
        assert session.events[0].content == "hello"

    async def test_flushed_events_persisted_to_store(self):
        store = MemoryStore()
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        session.add_agent_message("hi")
        await session.flush(store)

        loaded = await store.load_session("s1")
        assert loaded is not None
        assert len(loaded.events) == 2

    async def test_incremental_flush(self):
        store = MemoryStore()
        session = Session(id="s1", user_id="u1")

        session.add_user_message("hello")
        await session.flush(store)

        session.add_agent_message("hi")
        await session.flush(store)

        loaded = await store.load_session("s1")
        assert len(loaded.events) == 2

    async def test_flush_persists_state(self):
        store = MemoryStore()
        session = Session(id="s1", user_id="u1")
        session.state["timezone"] = "Asia/Riyadh"
        await session.flush(store)

        loaded = await store.load_session("s1")
        assert loaded.state["timezone"] == "Asia/Riyadh"

    async def test_flush_with_no_pending_is_noop(self):
        store = MemoryStore()
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        await session.flush(store)
        await session.flush(store)  # should not error

        loaded = await store.load_session("s1")
        assert len(loaded.events) == 1

    async def test_flush_without_store_raises(self):
        session = Session(id="s1", user_id="u1")
        session.add_user_message("hello")
        with pytest.raises(TypeError):
            await session.flush(None)
