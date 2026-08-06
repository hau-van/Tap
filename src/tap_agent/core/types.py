"""Provider-neutral types cho tap_agent core.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, TypeAlias, Union


# ─── JSON types ──────────────────────────────────────────────

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = Union[
    JSONPrimitive,
    list["JSONValue"],
    dict[str, "JSONValue"],
]
JSONObject: TypeAlias = dict[str, JSONValue]


# ─── ID helpers ──────────────────────────────────────────────

def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"

def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


# ─── Tool primitives ────────────────────────────────────────

@dataclass(frozen=True)
class ToolCall:
    """LLM's request to run a tool. Provider-neutral."""
    name: str
    arguments: JSONObject
    id: str = field(default_factory=_call_id)


@dataclass(frozen=True)
class ToolResult:
    """Output of a tool execution."""
    ok: bool
    content: str


ToolExecutor = Callable[[JSONObject], Awaitable[ToolResult]]


@dataclass(frozen=True)
class AgentTool:
    """Descriptor của 1 tool mà agent có thể gọi."""
    name: str
    description: str
    input_schema: JSONObject   # JSON schema
    executor: ToolExecutor


# ─── Messages ────────────────────────────────────────────────

@dataclass(frozen=True)
class UserMessage:
    content: str
    id: str = field(default_factory=_msg_id)
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantMessage:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error"] | None = None
    metadata: JSONObject = field(default_factory=dict)
    id: str = field(default_factory=_msg_id)
    role: Literal["assistant"] = "assistant"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class ToolResultMessage:
    tool_call_id: str
    content: str
    ok: bool = True
    id: str = field(default_factory=_msg_id)
    role: Literal["tool"] = "tool"


AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage]
