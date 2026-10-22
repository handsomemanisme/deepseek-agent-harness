"""Registration, validation, and execution of agent tools.

The registry deliberately separates validation from execution.  The agent's
interactive layer can therefore reject a malformed call before asking a user
to approve it, and validation can never trigger a tool's side effects.
"""

from __future__ import annotations

import copy
from typing import Any

import jsonschema

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    """An ordered collection of tools available to an agent session."""

    def __init__(self, tools: list[ToolSpec]):
        # Keep the original order for the model-facing schemas.  A separate
        # mapping makes exact, case-sensitive lookup inexpensive.
        self._ordered_tools = list(tools)
        self._tools = {tool.name: tool for tool in self._ordered_tools}

    def schemas(self) -> list[dict[str, Any]]:
        """Return only the model-visible parts of every registered tool."""

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": copy.deepcopy(tool.parameters),
            }
            for tool in self._ordered_tools
        ]

    def get(self, name: str) -> ToolSpec | None:
        """Return one registered tool without exposing the registry mapping."""

        return self._tools.get(name)

    def validate(self, name: str, args: dict) -> ToolResult | None:
        """Validate a prospective call without executing its function.

        ``None`` means the call is ready to run.  Every failure is converted
        to a ``ToolResult`` so malformed model output never escapes as an
        exception from the registry.
        """

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                status="unknown_tool",
                content=f"Unknown tool: {name!r}.",
            )

        try:
            jsonschema.validate(instance=args, schema=tool.parameters)
        except Exception as exc:
            return ToolResult(
                status="invalid_args",
                content=f"Invalid arguments for tool {name!r}: {exc}",
            )

        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        """Validate and execute one tool call, converting all failures."""

        validation_error = self.validate(name, args)
        if validation_error is not None:
            return validation_error

        tool = self._tools[name]
        try:
            value = tool.fn(**args)
        except Exception as exc:
            return ToolResult(
                status="tool_error",
                content=f"Tool {name!r} failed: {exc}",
            )

        return ToolResult(status="ok", content=str(value))
