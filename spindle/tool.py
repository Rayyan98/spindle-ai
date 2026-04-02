"""
Tool decorator and Tool class for defining callable tools.

Tools are plain functions decorated with @tool. The decorator extracts
name, description, and parameter schema from the function signature and
docstring. Tools can be called by the LLM, by your router, by application
code, or by tests — they are just functions with metadata.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints


_PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    """A callable function with metadata for LLM function calling."""

    def __init__(self, fn: Callable, name: str, description: str) -> None:
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self.name = name
        self.description = description
        self._parameters_schema: dict[str, Any] | None = None

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's parameters."""
        if self._parameters_schema is None:
            self._parameters_schema = _build_parameters_schema(self._fn)
        return self._parameters_schema

    async def execute(self, args: dict[str, Any]) -> Any:
        """Execute the tool with the given arguments."""
        sig = inspect.signature(self._fn)
        bound = sig.bind(**args)
        bound.apply_defaults()

        if self._is_async:
            return await self._fn(**bound.arguments)
        return self._fn(**bound.arguments)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def tool(fn: Callable) -> Tool:
    """Decorator that wraps a function as a Tool.

    Usage:
        @tool
        async def search(query: str, limit: int = 10) -> list:
            \"\"\"Search for items.\"\"\"
            return await catalog.search(query, limit=limit)
    """
    name = fn.__name__
    description = (fn.__doc__ or "").strip()
    return Tool(fn=fn, name=name, description=description)


def _build_parameters_schema(fn: Callable) -> dict[str, Any]:
    """Extract JSON Schema from function signature and type hints."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "return"):
            continue

        type_hint = hints.get(param_name, Any)
        json_type = _PYTHON_TYPE_TO_JSON.get(type_hint, "string")

        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
