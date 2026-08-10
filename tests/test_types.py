"""Unit tests cho tap_agent.core_types.

Kiểm tra 2 nhóm:
1. Object creation cho từng model
2. JSON serialization/deserialization round-trip
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tap_agent.core_types import (
    AgentEvent,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


# ══════════════════════════════════════════════════════════════
# 1. Object creation
# ══════════════════════════════════════════════════════════════

class TestToolCallCreation:
    def test_basic_fields(self):
        tc = ToolCall(id="call_1", name="read", arguments={"path": "a.py"})
        assert tc.id == "call_1"
        assert tc.name == "read"
        assert tc.arguments == {"path": "a.py"}

    def test_empty_arguments(self):
        tc = ToolCall(id="call_1", name="ping", arguments={})
        assert tc.arguments == {}

    def test_nested_arguments(self):
        tc = ToolCall(
            id="call_1",
            name="query",
            arguments={"filters": {"lang": "py", "min_size": 100}},
        )
        assert tc.arguments["filters"]["lang"] == "py"

    def test_frozen(self):
        tc = ToolCall(id="call_1", name="read", arguments={})
        with pytest.raises(ValidationError):
            tc.name = "changed"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ToolCall(id="call_1", name="read")   # missing arguments


class TestToolResultCreation:
    def test_success_case(self):
        tr = ToolResult(tool_call_id="call_1", success=True, output="hello")
        assert tr.success is True
        assert tr.output == "hello"
        assert tr.error is None

    def test_error_case(self):
        tr = ToolResult(
            tool_call_id="call_1",
            success=False,
            output="",
            error="File not found",
        )
        assert tr.success is False
        assert tr.error == "File not found"

    def test_error_defaults_to_none(self):
        tr = ToolResult(tool_call_id="call_1", success=True, output="x")
        assert tr.error is None


class TestToolDefinitionCreation:
    def test_basic_fields(self):
        td = ToolDefinition(
            name="read",
            description="Read a file",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        assert td.name == "read"
        assert td.description == "Read a file"
        assert "properties" in td.parameters_schema


class TestMessageCreation:
    def test_user_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="read", arguments={"path": "a.py"})
        msg = Message(role="assistant", content="Let me check", tool_calls=[tc])
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read"

    def test_tool_message(self):
        msg = Message(role="tool", content="file contents...")
        assert msg.role == "tool"
        assert msg.tool_calls is None

    def test_timestamp_defaults_to_utc_now(self):
        before = datetime.now(timezone.utc)
        msg = Message(role="user", content="hi")
        after = datetime.now(timezone.utc)
        assert msg.timestamp.tzinfo is not None
        assert before <= msg.timestamp <= after

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="system", content="hi")   # noqa — Literal chỉ có 3 value


class TestAgentEventCreation:
    def test_event_with_message_payload(self):
        msg = Message(role="user", content="hi")
        event = AgentEvent(type="message", payload=msg)
        assert event.type == "message"
        assert isinstance(event.payload, Message)

    def test_event_with_tool_call_payload(self):
        tc = ToolCall(id="call_1", name="read", arguments={})
        event = AgentEvent(type="tool_call", payload=tc)
        assert event.type == "tool_call"
        assert isinstance(event.payload, ToolCall)

    def test_event_with_tool_result_payload(self):
        tr = ToolResult(tool_call_id="call_1", success=True, output="ok")
        event = AgentEvent(type="tool_result", payload=tr)
        assert event.type == "tool_result"
        assert isinstance(event.payload, ToolResult)

    def test_event_error_string_payload(self):
        event = AgentEvent(type="error", payload="Something broke")
        assert event.type == "error"
        assert event.payload == "Something broke"

    def test_event_done(self):
        event = AgentEvent(type="done", payload="")
        assert event.type == "done"

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            AgentEvent(type="unknown", payload="x")   # noqa


# ══════════════════════════════════════════════════════════════
# 2. JSON serialization / deserialization
# ══════════════════════════════════════════════════════════════

class TestToolCallSerialization:
    def test_roundtrip(self):
        original = ToolCall(
            id="call_1",
            name="read",
            arguments={"path": "a.py", "line": 10},
        )
        json_str = original.model_dump_json()
        restored = ToolCall.model_validate_json(json_str)
        assert restored == original


class TestToolResultSerialization:
    def test_success_roundtrip(self):
        original = ToolResult(tool_call_id="call_1", success=True, output="ok")
        json_str = original.model_dump_json()
        restored = ToolResult.model_validate_json(json_str)
        assert restored == original

    def test_error_roundtrip(self):
        original = ToolResult(
            tool_call_id="call_1",
            success=False,
            output="",
            error="not found",
        )
        json_str = original.model_dump_json()
        restored = ToolResult.model_validate_json(json_str)
        assert restored == original


class TestToolDefinitionSerialization:
    def test_roundtrip(self):
        original = ToolDefinition(
            name="read",
            description="Read a file",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        json_str = original.model_dump_json()
        restored = ToolDefinition.model_validate_json(json_str)
        assert restored == original


class TestMessageSerialization:
    def test_simple_message_roundtrip(self):
        original = Message(role="user", content="hello")
        json_str = original.model_dump_json()
        restored = Message.model_validate_json(json_str)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp
        assert restored.tool_calls is None

    def test_message_with_tool_calls_roundtrip(self):
        original = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
                ToolCall(id="call_2", name="bash", arguments={"command": "ls"}),
            ],
        )
        json_str = original.model_dump_json()
        restored = Message.model_validate_json(json_str)
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].name == "read"
        assert restored.tool_calls[1].name == "bash"

    def test_timestamp_preserved_across_roundtrip(self):
        original = Message(role="user", content="hi")
        json_str = original.model_dump_json()
        restored = Message.model_validate_json(json_str)
        assert restored.timestamp == original.timestamp
        assert restored.timestamp.tzinfo is not None


class TestAgentEventSerialization:
    def test_message_payload_roundtrip(self):
        msg = Message(role="user", content="hi")
        original = AgentEvent(type="message", payload=msg)
        json_str = original.model_dump_json()
        restored = AgentEvent.model_validate_json(json_str)
        assert restored.type == "message"
        assert isinstance(restored.payload, Message)
        assert restored.payload.content == "hi"

    def test_tool_call_payload_roundtrip(self):
        tc = ToolCall(id="call_1", name="read", arguments={"path": "a.py"})
        original = AgentEvent(type="tool_call", payload=tc)
        json_str = original.model_dump_json()
        restored = AgentEvent.model_validate_json(json_str)
        assert isinstance(restored.payload, ToolCall)
        assert restored.payload.id == "call_1"

    def test_tool_result_payload_roundtrip(self):
        tr = ToolResult(tool_call_id="call_1", success=True, output="ok")
        original = AgentEvent(type="tool_result", payload=tr)
        json_str = original.model_dump_json()
        restored = AgentEvent.model_validate_json(json_str)
        assert isinstance(restored.payload, ToolResult)
        assert restored.payload.success is True

    def test_error_string_payload_roundtrip(self):
        original = AgentEvent(type="error", payload="Bad thing")
        json_str = original.model_dump_json()
        restored = AgentEvent.model_validate_json(json_str)
        assert restored.type == "error"
        assert restored.payload == "Bad thing"

    def test_done_event_roundtrip(self):
        original = AgentEvent(type="done", payload="")
        json_str = original.model_dump_json()
        restored = AgentEvent.model_validate_json(json_str)
        assert restored.type == "done"


class TestDictRoundtrip:
    """model_dump() → dict → model_validate() cũng phải hoạt động."""

    def test_message_dict_roundtrip(self):
        original = Message(role="user", content="hi")
        data = original.model_dump()
        restored = Message.model_validate(data)
        assert restored == original

    def test_tool_call_dict_roundtrip(self):
        original = ToolCall(id="call_1", name="read", arguments={"x": 1})
        data = original.model_dump()
        restored = ToolCall.model_validate(data)
        assert restored == original
