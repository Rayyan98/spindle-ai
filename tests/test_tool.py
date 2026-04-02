"""Tests for spindle tool decorator and Tool class."""

from spindle.tool import Tool, tool


# -- Decorator Basics -------------------------------------------------------


class TestToolDecorator:
    def test_preserves_function_name(self):
        @tool
        async def search(query: str) -> list:
            """Search for items."""
            return []

        assert search.name == "search"

    def test_preserves_docstring_as_description(self):
        @tool
        async def search(query: str) -> list:
            """Search for items in the catalog."""
            return []

        assert search.description == "Search for items in the catalog."

    def test_function_remains_callable(self):
        @tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # The tool wraps the function but should still be invocable
        assert isinstance(add, Tool)

    def test_no_docstring_uses_empty_description(self):
        @tool
        async def mystery(x: int) -> int:
            return x

        assert mystery.description == ""


# -- Parameter Extraction --------------------------------------------------


class TestParameterExtraction:
    def test_extracts_parameter_names(self):
        @tool
        async def search(query: str, limit: int = 10) -> list:
            """Search."""
            return []

        schema = search.parameters_schema
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]

    def test_extracts_parameter_types(self):
        @tool
        async def search(query: str, limit: int = 10) -> list:
            """Search."""
            return []

        schema = search.parameters_schema
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"

    def test_required_parameters(self):
        @tool
        async def search(query: str, limit: int = 10) -> list:
            """Search."""
            return []

        schema = search.parameters_schema
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_boolean_parameter(self):
        @tool
        async def toggle(enabled: bool) -> dict:
            """Toggle."""
            return {}

        schema = toggle.parameters_schema
        assert schema["properties"]["enabled"]["type"] == "boolean"

    def test_float_parameter(self):
        @tool
        async def scale(factor: float) -> dict:
            """Scale."""
            return {}

        schema = scale.parameters_schema
        assert schema["properties"]["factor"]["type"] == "number"

    def test_no_parameters(self):
        @tool
        async def ping() -> str:
            """Ping."""
            return "pong"

        schema = ping.parameters_schema
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_list_parameter(self):
        @tool
        async def batch(ids: list) -> dict:
            """Batch process."""
            return {}

        schema = batch.parameters_schema
        assert schema["properties"]["ids"]["type"] == "array"

    def test_dict_parameter(self):
        @tool
        async def configure(options: dict) -> dict:
            """Configure."""
            return {}

        schema = configure.parameters_schema
        assert schema["properties"]["options"]["type"] == "object"


# -- Execution --------------------------------------------------------------


class TestExecution:
    async def test_call_tool_directly(self):
        @tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add.execute({"a": 3, "b": 4})
        assert result == 7

    async def test_call_tool_with_defaults(self):
        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        result = await greet.execute({"name": "World"})
        assert result == "Hello, World!"

    async def test_call_tool_with_no_args(self):
        @tool
        async def ping() -> str:
            """Ping."""
            return "pong"

        result = await ping.execute({})
        assert result == "pong"

    async def test_sync_tool_function(self):
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        result = await multiply.execute({"a": 3, "b": 4})
        assert result == 12


# -- JSON Schema Generation ------------------------------------------------


class TestJsonSchema:
    def test_schema_has_type_object(self):
        @tool
        async def search(query: str) -> list:
            """Search."""
            return []

        schema = search.parameters_schema
        assert schema["type"] == "object"

    def test_schema_structure(self):
        @tool
        async def search(query: str, limit: int = 10) -> list:
            """Search items."""
            return []

        schema = search.parameters_schema
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
