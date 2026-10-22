"""Cardinal's restricted terminal tool.

The course-provided :mod:`cs329z_hw1.sandbox` owns command validation,
workspace preparation, process isolation, output capture, and timeouts.  This
module only adapts its structured result to the agent tool protocol.
"""

from __future__ import annotations

import json

from cs329z_hw1 import sandbox
from cs329z_hw1.cardinal import cardinal_spec
from cs329z_hw1.types import AgentConfig, ToolSpec


class TerminalCommandError(RuntimeError):
    """A command was observed by the sandbox but did not succeed."""


def _format_result(result: sandbox.TerminalResult) -> str:
    """Serialize every terminal outcome field for the model."""

    return json.dumps(
        {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "rejected": result.rejected,
            "reason": result.reason,
        },
        ensure_ascii=False,
    )


class TerminalTools:
    """Run commands with one session's workspace and timeout settings."""

    def __init__(self, config: AgentConfig):
        self.workspace = config.workspace
        self.timeout = config.terminal_timeout

    def run_terminal(self, cmd: str) -> str:
        """Run one command and turn unsuccessful outcomes into tool errors."""

        result = sandbox.run_terminal(
            cmd,
            workspace=self.workspace,
            timeout=self.timeout,
        )
        content = _format_result(result)

        if result.rejected or result.timed_out or result.exit_code != 0:
            # ToolRegistry converts exceptions into a ToolResult with status
            # ``tool_error`` while preserving this diagnostic content.
            raise TerminalCommandError(content)

        return content


def build_terminal_tools(config: AgentConfig) -> list[ToolSpec]:
    """Build the terminal spec without starting a command or workspace."""

    implementation = TerminalTools(config)
    return [
        cardinal_spec(
            "run_terminal",
            (
                "Run one shell command in the restricted workspace and return "
                "stdout, stderr, exit_code, timeout, rejection, and reason "
                "fields. Use allowed command-line programs for workspace "
                "inspection, calculations, and file-oriented tasks. Prefer "
                "focused commands such as grep, head, tail, or wc instead of "
                "printing an entire large file. This tool can change files and "
                "may require user approval."
            ),
            implementation.run_terminal,
        )
    ]
