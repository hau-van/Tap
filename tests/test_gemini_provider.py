"""Unit tests cho tap_agent.gemini_provider.

Kiểm tra 2 nhóm:
1. Helper conversion cho Gemini provider
2. GeminiProvider response mapping
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors, types

from tap_agent.core_types import Message, ToolDefinition
from tap_agent.gemini_provider import (
	GeminiProvider,
	GeminiProviderError,
	RateLimiter,
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

	@pytest.mark.asyncio
	async def test_rate_limiter_wait(self):
		limiter = RateLimiter(requests_per_minute=600)  # 10 req/sec => 0.1s interval

		start_time = time.monotonic()
		await limiter.wait()  # first call should not wait
		await limiter.wait()  # second call should wait ~0.1s
		end_time = time.monotonic()

		elapsed = end_time - start_time
		assert elapsed >= 0.09, f"RateLimiter didn't wait long enough: {elapsed}"

	@pytest.mark.asyncio
	async def test_backoff_retry_success(self):
		mock_client = MagicMock()
		mock_client.aio.models.generate_content = AsyncMock()

		error_429 = errors.ClientError(
			429,
			{"error": {"code": 429, "message": "Too Many Requests", "status": "RESOURCE_EXHAUSTED"}},
			None,
		)

		success_response = types.GenerateContentResponse(
			candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="Success!")]))]
		)

		mock_client.aio.models.generate_content.side_effect = [
			error_429,
			error_429,
			success_response,
		]

		provider = GeminiProvider(
			model="test-model",
			client=mock_client,
			requests_per_minute=0,  # disable rate limiter for faster tests
			max_retries=3,
		)

		start_time = time.monotonic()
		response = await provider.generate_reply([Message(role="user", content="Hi")], [])
		end_time = time.monotonic()

		assert response.content == "Success!"
		assert mock_client.aio.models.generate_content.call_count == 3
		assert end_time - start_time >= 2.0  # ~1s + ~2s

	@pytest.mark.asyncio
	async def test_backoff_retry_exceeds_max_retries(self):
		mock_client = MagicMock()
		mock_client.aio.models.generate_content = AsyncMock()

		error_429 = errors.ClientError(
			429,
			{"error": {"code": 429, "message": "Too Many Requests", "status": "RESOURCE_EXHAUSTED"}},
			None,
		)

		mock_client.aio.models.generate_content.side_effect = error_429

		provider = GeminiProvider(
			model="test-model",
			client=mock_client,
			requests_per_minute=0,
			max_retries=2,
		)

		start_time = time.monotonic()
		with pytest.raises(GeminiProviderError) as exc_info:
			await provider.generate_reply([Message(role="user", content="Hi")], [])
		end_time = time.monotonic()

		assert mock_client.aio.models.generate_content.call_count == 3  # 1 initial + 2 retries
		assert "Gemini request failed" in str(exc_info.value)
		assert end_time - start_time >= 2.0

	@pytest.mark.asyncio
	async def test_fails_fast_on_non_429_error(self):
		mock_client = MagicMock()
		mock_client.aio.models.generate_content = AsyncMock()

		error_400 = errors.ClientError(
			400,
			{"error": {"code": 400, "message": "Bad Request", "status": "INVALID_ARGUMENT"}},
			None,
		)

		mock_client.aio.models.generate_content.side_effect = error_400

		provider = GeminiProvider(
			model="test-model",
			client=mock_client,
			requests_per_minute=0,
			max_retries=5,
		)

		with pytest.raises(GeminiProviderError) as exc_info:
			await provider.generate_reply([Message(role="user", content="Hi")], [])

		assert mock_client.aio.models.generate_content.call_count == 1
		assert "Gemini request failed" in str(exc_info.value)
