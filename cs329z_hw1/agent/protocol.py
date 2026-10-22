"""Text protocol used for model-generated tool calls.

The course model interface returns only text, so tool calls are represented as
JSON inside explicit boundary markers.  Parsing uses ``JSONDecoder.raw_decode``
instead of searching for the first closing marker: an argument string may
itself quote a complete formatted tool call without confusing the parser.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cs329z_hw1.types import ParsedResponse, ToolCall


CALL_START = "<tool_call>"
CALL_END = "</tool_call>"

# Some DeepSeek text-only endpoints serialize an inferred tool call with their
# internal DSML prefix even when the prompt asks for our portable JSON format.
# The response still arrives through ``message.content`` (not the API's native
# ``tool_calls`` field), so it has to be normalized at this text-protocol
# boundary.  Keep the match deliberately narrow: arbitrary XML-like text must
# continue to be treated as an ordinary final answer.
DEEPSEEK_CALLS_START = "<｜｜DSML｜｜ calls>"
DEEPSEEK_INVOKE_RE = re.compile(
    r'<｜｜DSML｜｜ invoke name="([A-Za-z_][A-Za-z0-9_.:-]*)">'
)
DEEPSEEK_INVOKE_END = "</｜｜DSML｜｜ invoke>"
DEEPSEEK_CALLS_END = "</｜｜DSML｜｜ calls>"
DEEPSEEK_PARAMETER_RE = re.compile(
    r'<｜｜DSML｜｜ parameter name="([A-Za-z_][A-Za-z0-9_.:-]*)" '
    r'string="(true|false)">(.*?)</｜｜DSML｜｜ parameter>',
    re.DOTALL,
)


def render_tools(schemas: list[dict]) -> str:
    """Render tool schemas and the exact call protocol for the model."""

    rendered_schemas = json.dumps(schemas, ensure_ascii=False, indent=2)
    return (
        "You may call one of the tools listed below when a tool is needed.\n"
        "Each tool entry gives its exact name, purpose, and JSON Schema for "
        "its arguments.\n\n"
        f"TOOLS\n{rendered_schemas}\n\n"
        "TOOL-CALL FORMAT\n"
        "To call a tool, output exactly one block in this form:\n"
        f"{CALL_START}\n"
        '{"name":"exact_tool_name","args":{"parameter_name":"value"}}\n'
        f"{CALL_END}\n\n"
        "Use the exact tool name and put every argument in the args object. "
        "The args object must satisfy that tool's parameters schema. Preserve "
        "JSON types: strings need quotes, numbers and booleans do not. Call at "
        "most one tool in a response. Do not wrap the block in a Markdown code "
        "fence. You may put a short explanation before the block. If no tool is "
        "needed, answer normally and do not emit a tool_call block."
    )


def format_tool_call(text: str, call: ToolCall) -> str:
    """Format assistant text and one tool call without modifying ``call``."""

    payload = {"name": call.name, "args": call.args}
    call_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    block = f"{CALL_START}\n{call_json}\n{CALL_END}"
    leading_text = text.strip()
    return f"{leading_text}\n\n{block}" if leading_text else block


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _call_from_payload(payload: Any) -> ToolCall | None:
    """Convert a decoded JSON value to a call only when its shape is exact."""

    if not isinstance(payload, dict):
        return None
    if set(payload) != {"name", "args"}:
        return None

    name = payload.get("name")
    args = payload.get("args")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, args=args)


def _parse_deepseek_response(reply: str) -> ParsedResponse | None:
    """Parse the strict DSML variant emitted by DeepSeek text responses.

    A successful DeepSeek response currently looks like::

        <｜｜DSML｜｜ calls>
        <｜｜DSML｜｜ invoke name="echo">
        {"text":"hello"}
        </tool_call>

    Return ``None`` when the marker is absent so the normal Cardinal parser
    remains authoritative for every other response.
    """

    start = reply.find(DEEPSEEK_CALLS_START)
    invokes = list(DEEPSEEK_INVOKE_RE.finditer(reply))
    if start < 0 and not invokes:
        return None
    if start < 0 or len(invokes) != 1 or reply.count(DEEPSEEK_CALLS_START) != 1:
        return ParsedResponse(
            text=reply.strip(),
            error="A DeepSeek response must contain exactly one complete tool call.",
        )

    invoke = invokes[0]
    if invoke.start() < start + len(DEEPSEEK_CALLS_START):
        return ParsedResponse(
            text=reply.strip(),
            error="The DeepSeek tool call has malformed boundary markers.",
        )

    between = reply[start + len(DEEPSEEK_CALLS_START) : invoke.start()]
    if between.strip():
        return ParsedResponse(
            text=reply.strip(),
            error="Unexpected text appeared inside the DeepSeek tool-call header.",
        )

    try:
        body_start = _skip_whitespace(reply, invoke.end())
        parameter = DEEPSEEK_PARAMETER_RE.match(reply, body_start)
        if parameter is not None:
            args: dict[str, Any] = {}
            cursor = body_start
            while True:
                parameter = DEEPSEEK_PARAMETER_RE.match(reply, cursor)
                if parameter is None:
                    break
                name, is_string, raw_value = parameter.groups()
                if name in args:
                    return ParsedResponse(
                        text=reply.strip(),
                        error=f"DeepSeek repeated tool-call parameter {name!r}.",
                    )
                args[name] = (
                    raw_value
                    if is_string == "true"
                    else json.loads(raw_value.strip())
                )
                cursor = _skip_whitespace(reply, parameter.end())

            if not reply.startswith(DEEPSEEK_INVOKE_END, cursor):
                return ParsedResponse(
                    text=reply.strip(),
                    error="The DeepSeek tool invocation is missing its closing marker.",
                )
            cursor = _skip_whitespace(reply, cursor + len(DEEPSEEK_INVOKE_END))
            if not reply.startswith(DEEPSEEK_CALLS_END, cursor):
                return ParsedResponse(
                    text=reply.strip(),
                    error="The DeepSeek tool-call group is missing its closing marker.",
                )
            after_call = cursor + len(DEEPSEEK_CALLS_END)
        else:
            # Some DeepSeek endpoints use the DSML prefix but put a single
            # JSON args object in the body and close it with our requested
            # marker.  This is the variant seen in the protocol smoke test.
            args, json_end = json.JSONDecoder().raw_decode(reply, body_start)
            marker_start = _skip_whitespace(reply, json_end)
            if not reply.startswith(CALL_END, marker_start):
                return ParsedResponse(
                    text=reply.strip(),
                    error="The DeepSeek tool call is incomplete or missing its closing marker.",
                )
            if not isinstance(args, dict):
                return ParsedResponse(
                    text=reply.strip(),
                    error="DeepSeek tool-call arguments must be a JSON object.",
                )
            after_call = marker_start + len(CALL_END)

        if not isinstance(args, dict):
            return ParsedResponse(
                text=reply.strip(),
                error="DeepSeek tool-call arguments must be a JSON object.",
            )

        suffix = reply[after_call:]
        if (
            CALL_START in suffix
            or DEEPSEEK_CALLS_START in suffix
            or DEEPSEEK_INVOKE_RE.search(suffix)
        ):
            return ParsedResponse(
                text=reply.strip(),
                error="A response may contain at most one tool call.",
            )

        remaining_text = (reply[:start] + suffix).strip()
        return ParsedResponse(
            text=remaining_text,
            tool_call=ToolCall(name=invoke.group(1), args=args),
        )
    except Exception as exc:
        return ParsedResponse(
            text=reply.strip(),
            error=f"Could not parse the DeepSeek tool call: {exc}",
        )


def parse_response(text: str) -> ParsedResponse:
    """Parse at most one marked tool call from an arbitrary model reply.

    This function is a trust boundary: every malformed input is represented by
    ``ParsedResponse.error`` and no parser exception is allowed to escape.
    """

    reply = text if isinstance(text, str) else str(text)
    start = reply.find(CALL_START)
    if start < 0:
        deepseek = _parse_deepseek_response(reply)
        return deepseek if deepseek is not None else ParsedResponse(text=reply.strip())

    try:
        json_start = _skip_whitespace(reply, start + len(CALL_START))
        payload, json_end = json.JSONDecoder().raw_decode(reply, json_start)
        marker_start = _skip_whitespace(reply, json_end)

        if not reply.startswith(CALL_END, marker_start):
            return ParsedResponse(
                text=reply.strip(),
                error="The tool call is incomplete or is missing its closing marker.",
            )

        after_call = marker_start + len(CALL_END)
        suffix = reply[after_call:]
        if CALL_START in suffix:
            return ParsedResponse(
                text=reply.strip(),
                error="A response may contain at most one tool call.",
            )

        call = _call_from_payload(payload)
        if call is None:
            return ParsedResponse(
                text=reply.strip(),
                error=(
                    "A tool call must be a JSON object with exactly a non-empty "
                    "string 'name' and an object 'args'."
                ),
            )

        remaining_text = (reply[:start] + suffix).strip()
        return ParsedResponse(text=remaining_text, tool_call=call)
    except Exception as exc:
        return ParsedResponse(
            text=reply.strip(),
            error=f"Could not parse the tool call: {exc}",
        )
