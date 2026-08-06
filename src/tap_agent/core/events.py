"""Events do agent loop emit ra.

CLI subscribe những event này để render. Provider-neutral.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from .types import AssistantMessage, ToolResult


@dataclass(frozen=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(frozen=True)
class AgentEndEvent:
    reason: Literal["completed", "max_turns", "error"] = "completed"
    type: Literal["agent_end"] = "agent_end"


@dataclass(frozen=True)
class MessageStartEvent:
    role: Literal["user", "assistant", "tool"]
    type: Literal["message_start"] = "message_start"


@dataclass(frozen=True)
class MessageEndEvent:
    message: AssistantMessage
    type: Literal["message_end"] = "message_end"


@dataclass(frozen=True)
class ToolExecutionStartEvent:
    tool_name: str
    tool_call_id: str
    arguments: dict
    type: Literal["tool_start"] = "tool_start"


@dataclass(frozen=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    result: ToolResult
    type: Literal["tool_end"] = "tool_end"


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    recoverable: bool = False
    type: Literal["error"] = "error"


Event = Union[
    AgentStartEvent,
    AgentEndEvent,
    MessageStartEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    ErrorEvent,
]
