"""A tiny tool decorator for the agent.

Tools are plain Python functions. Decorate them with :func:`tool` and the
decorator will:

* take the tool name from the function name,
* read the description and per-argument docs from the docstring,
* build the JSON schema the OpenAI API expects from the type hints,
* expose ``.invoke(arguments)`` so the agent can call it with a dict.

This replaces the LangChain ``@tool`` decorator with a small, dependency-free
equivalent (only the standard library is used).
"""

from __future__ import annotations

import inspect
import re
import types
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

_PRIMITIVE_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_SECTION_HEADERS = {
    "Args": "args",
    "Arguments": "args",
    "Parameters": "args",
    "Returns": "other",
    "Raises": "other",
    "Yields": "other",
    "Examples": "other",
    "Example": "other",
    "Notes": "other",
}


@dataclass
class Tool:
    """A callable tool plus the metadata needed to advertise it to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def invoke(self, arguments: dict[str, Any] | None = None) -> str:
        """Run the tool with a dict of arguments and always return a string."""
        return str(self.func(**dict(arguments or {})))

    def to_openai(self) -> dict[str, Any]:
        """The schema shape required by ``tools=[...]`` in the OpenAI API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


def tool(func: Callable[..., Any]) -> Tool:
    """Turn a plain function into a :class:`Tool`."""
    if isinstance(func, Tool):
        return func

    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    description, parameter_docs = _parse_docstring(inspect.getdoc(func) or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        annotation = type_hints.get(name, parameter.annotation)
        schema = _json_schema(annotation)

        if name in parameter_docs:
            schema["description"] = parameter_docs[name]

        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = parameter.default

        properties[name] = schema

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return Tool(
        name=func.__name__,
        description=description,
        parameters=parameters,
        func=func,
    )


def _json_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        options = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(options) == 1:
            return _json_schema(options[0])
        return {"anyOf": [_json_schema(option) for option in options]}

    if annotation in _PRIMITIVE_TYPES:
        return {"type": _PRIMITIVE_TYPES[annotation]}

    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        args = get_args(annotation)
        if args:
            schema["items"] = _json_schema(args[0])
        return schema

    if origin is dict:
        return {"type": "object"}

    return {"type": "string"}


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a description and argument docs."""
    description_lines: list[str] = []
    parameter_docs: dict[str, str] = {}
    section: str | None = None
    current_parameter: str | None = None

    for raw_line in docstring.splitlines():
        line = raw_line.rstrip()
        header = line.strip().rstrip(":")
        if header in _SECTION_HEADERS:
            section = _SECTION_HEADERS[header]
            current_parameter = None
            continue

        if section is None:
            description_lines.append(line)
        elif section == "args":
            match = re.match(r"^\s*([A-Za-z_]\w*)\s*:\s*(.*)$", line)
            if match:
                current_parameter = match.group(1)
                parameter_docs[current_parameter] = match.group(2).strip()
            elif current_parameter and line.strip():
                parameter_docs[current_parameter] += " " + line.strip()

    return "\n".join(description_lines).strip(), parameter_docs
