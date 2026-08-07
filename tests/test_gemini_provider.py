"""Unit tests cho tap_agent.gemini_provider.

Kiểm tra 2 nhóm:
1. Helper conversion cho Gemini provider
2. GeminiProvider response mapping
"""
from __future__ import annotations

import asyncio
import json

import pytest
from google.genai import types

from tap_agent.core_types import Message, ToolDefinition
from tap_agent.gemini_provider import (
	GeminiProvider,
	_messages_to_contents,
	_to_gemini_schema,
)


# ══════════════════════════════════════════════════════════════
# 1. Gemini helper conversions
# ══════════════════════════════════════════════════════════════


class _FakeModels:
	def __init__(self, response: types.GenerateContentResponse) -> None:
		self.response = response
		self.calls: list[dict[str, object]] = []

	async def generate_content(self, **kwargs):
		self.calls.append(kwargs)
		return self.response


class _FakeAio:
	def __init__(self, models: _FakeModels) -> None:
		self.models = models


class _FakeClient:
	def __init__(self, response: types.GenerateContentResponse) -> None:
		self.models = _FakeModels(response)
		self.aio = _FakeAio(self.models)


class _FakeResponse:
	def __init__(self) -> None:
		self.parts = [
			types.Part(text="Hello "),
			types.Part(functionCall=types.FunctionCall(id="call_1", name="read", args={"path": "a.txt"})),
		]
		self.text = None


class TestGeminiSchemaConversion:
	def test_normalizes_types(self):
		tool = ToolDefinition(
			name="read",
			description="Read a file",
			parameters_schema={
				"type": "object",
				"properties": {
					"path": {"type": "string"},
					"options": {
						"type": "object",
						"properties": {
							"lines": {"type": "array", "items": {"type": "integer"}},
							"recursive": {"type": "boolean"},
						},
						"required": ["lines"],
					},
				},
				"required": ["path"],
			},
		)

		schema = _to_gemini_schema(tool)

		assert schema["type"] == "OBJECT"
		assert schema["properties"]["path"]["type"] == "STRING"
		assert schema["properties"]["options"]["type"] == "OBJECT"
		assert schema["properties"]["options"]["properties"]["lines"]["type"] == "ARRAY"
		assert schema["properties"]["options"]["properties"]["lines"]["items"]["type"] == "INTEGER"
		assert schema["properties"]["options"]["properties"]["recursive"]["type"] == "BOOLEAN"


class TestGeminiMessageConversion:
	def test_tool_json_converts_to_function_response_dict(self):
		message = Message(
			role="tool",
			content=json.dumps({"success": True, "output": "hello", "error": None}),
		)

		contents = _messages_to_contents([message])

		assert contents[0].role == "user"
		assert contents[0].parts[0].function_response.response == {
			"success": True,
			"output": "hello",
			"error": None,
		}

	def test_invalid_tool_json_rejected(self):
		message = Message(role="tool", content="not-json")

		with pytest.raises(json.JSONDecodeError):
			_messages_to_contents([message])


class TestGeminiProvider:
	def test_generate_reply_maps_response_parts_to_message(self):
		response = _FakeResponse()
		client = _FakeClient(response)
		provider = GeminiProvider(model="gemini-test", client=client)

		result = asyncio.run(
			provider.generate_reply(
				messages=[Message(role="user", content="hi")],
				tools=[ToolDefinition(name="read", description="Read", parameters_schema={"type": "object"})],
			)
		)

		assert result.role == "assistant"
		assert result.content == "Hello "
		assert result.tool_calls is not None
		assert result.tool_calls[0].id == "call_1"
		assert result.tool_calls[0].name == "read"
		assert result.tool_calls[0].arguments == {"path": "a.txt"}
		assert client.models.calls[0]["model"] == "gemini-test"
		assert len(client.models.calls[0]["contents"]) == 1
		assert len(client.models.calls[0]["config"].tools) == 1
