"""Tests for spindle event model and factory methods."""

import pytest

from spindle.event import Event
from spindle.types import EventRole, EventType


# -- Factory Methods --------------------------------------------------------


class TestUserMessage:
    def test_creates_event_with_correct_role(self):
        event = Event.user_message("hello")
        assert event.role == EventRole.USER

    def test_creates_event_with_correct_type(self):
        event = Event.user_message("hello")
        assert event.type == EventType.MESSAGE

    def test_stores_text_content(self):
        event = Event.user_message("hello world")
        assert event.content == "hello world"

    def test_author_defaults_to_user(self):
        event = Event.user_message("hello")
        assert event.author == "user"

    def test_generates_unique_id(self):
        e1 = Event.user_message("hello")
        e2 = Event.user_message("hello")
        assert e1.id != e2.id

    def test_generates_timestamp(self):
        event = Event.user_message("hello")
        assert event.timestamp > 0


class TestAgentMessage:
    def test_creates_event_with_agent_role(self):
        event = Event.agent_message("Hi there!")
        assert event.role == EventRole.AGENT

    def test_creates_event_with_message_type(self):
        event = Event.agent_message("Hi there!")
        assert event.type == EventType.MESSAGE

    def test_stores_text_content(self):
        event = Event.agent_message("Hi there!")
        assert event.content == "Hi there!"

    def test_author_defaults_to_agent(self):
        event = Event.agent_message("Hi there!")
        assert event.author == "agent"

    def test_custom_author(self):
        event = Event.agent_message("Hi!", author="receipt_processor")
        assert event.author == "receipt_processor"


class TestToolCall:
    def test_creates_event_with_agent_role(self):
        event = Event.tool_call("search", {"query": "laptop"})
        assert event.role == EventRole.AGENT

    def test_creates_event_with_tool_call_type(self):
        event = Event.tool_call("search", {"query": "laptop"})
        assert event.type == EventType.TOOL_CALL

    def test_stores_tool_name(self):
        event = Event.tool_call("search", {"query": "laptop"})
        assert event.tool_name == "search"

    def test_stores_tool_args(self):
        event = Event.tool_call("search", {"query": "laptop", "limit": 5})
        assert event.tool_args == {"query": "laptop", "limit": 5}

    def test_generates_tool_call_id(self):
        event = Event.tool_call("search", {"query": "laptop"})
        assert event.tool_call_id is not None
        assert len(event.tool_call_id) > 0

    def test_custom_author(self):
        event = Event.tool_call("search", {"query": "laptop"}, author="assistant")
        assert event.author == "assistant"

    def test_content_is_none(self):
        event = Event.tool_call("search", {"query": "laptop"})
        assert event.content is None


class TestToolResult:
    def test_creates_event_with_tool_role(self):
        event = Event.tool_result("search", [{"name": "MacBook"}], call_id="c1")
        assert event.role == EventRole.TOOL

    def test_creates_event_with_tool_result_type(self):
        event = Event.tool_result("search", [{"name": "MacBook"}], call_id="c1")
        assert event.type == EventType.TOOL_RESULT

    def test_stores_tool_name(self):
        event = Event.tool_result("search", [{"name": "MacBook"}], call_id="c1")
        assert event.tool_name == "search"

    def test_stores_result_data(self):
        data = [{"name": "MacBook", "price": 999}]
        event = Event.tool_result("search", data, call_id="c1")
        assert event.tool_result_data == data

    def test_stores_call_id(self):
        event = Event.tool_result("search", {"found": True}, call_id="call-123")
        assert event.tool_call_id == "call-123"

    def test_content_is_none(self):
        event = Event.tool_result("search", {"found": True}, call_id="c1")
        assert event.content is None


# -- Immutability -----------------------------------------------------------


class TestImmutability:
    def test_event_is_frozen(self):
        event = Event.user_message("hello")
        with pytest.raises(Exception):
            event.content = "modified"


# -- Serialization ----------------------------------------------------------


class TestSerialization:
    def test_roundtrip_user_message(self):
        original = Event.user_message("hello")
        data = original.model_dump()
        restored = Event.model_validate(data)
        assert restored == original

    def test_roundtrip_tool_call(self):
        original = Event.tool_call("search", {"query": "test"})
        data = original.model_dump()
        restored = Event.model_validate(data)
        assert restored == original

    def test_roundtrip_tool_result(self):
        original = Event.tool_result("search", [1, 2, 3], call_id="c1")
        data = original.model_dump()
        restored = Event.model_validate(data)
        assert restored == original

    def test_json_roundtrip(self):
        original = Event.user_message("hello")
        json_str = original.model_dump_json()
        restored = Event.model_validate_json(json_str)
        assert restored == original
