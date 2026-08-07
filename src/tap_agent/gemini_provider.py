from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from google.genai import Client, errors, types

from tap_agent.core_types import Message, ToolCall, ToolDefinition
from tap_agent.provider import ModelProvider


class GeminiProviderError(RuntimeError):
	"""Raised when Gemini request or response handling fails."""


_TYPE_MAP = {
	"string": "STRING",
	"object": "OBJECT",
	"array": "ARRAY",
	"boolean": "BOOLEAN",
	"integer": "INTEGER",
	"number": "NUMBER",
	"null": "NULL",
}


def _normalize_schema_type(value: Any) -> Any:
	if isinstance(value, str):
		return _TYPE_MAP.get(value.lower(), value.upper())
	if isinstance(value, list):
		return [_normalize_schema_type(item) for item in value]
	return value


def _normalize_schema(schema: Any) -> Any:
	if isinstance(schema, Mapping):
		normalized: dict[str, Any] = {}
		for key, value in schema.items():
			if key == "type":
				normalized[key] = _normalize_schema_type(value)
			elif key == "properties" and isinstance(value, Mapping):
				normalized[key] = {
					property_name: _normalize_schema(property_schema)
					for property_name, property_schema in value.items()
				}
			elif key in {"items", "additionalProperties", "anyOf", "oneOf", "allOf", "defs"}:
				normalized[key] = _normalize_schema(value)
			else:
				normalized[key] = _normalize_schema(value)
		return normalized
	if isinstance(schema, list):
		return [_normalize_schema(item) for item in schema]
	return schema


def _to_gemini_schema(tool: ToolDefinition) -> dict[str, Any]:
	"""Convert standard JSON Schema into the Gemini/OpenAPI-compatible shape.

	Gemini accepts a slightly different schema vocabulary than plain JSON
	Schema, most notably upper-cased type names in the function declaration
	payload.
	"""

	return _normalize_schema(tool.parameters_schema)


def _tool_to_gemini_function(tool: ToolDefinition) -> types.FunctionDeclaration:
	return types.FunctionDeclaration(
		name=tool.name,
		description=tool.description,
		parametersJsonSchema=_to_gemini_schema(tool),
	)


def _tools_to_gemini_tools(tools: list[ToolDefinition]) -> list[types.Tool]:
	if not tools:
		return []
	return [types.Tool(functionDeclarations=[_tool_to_gemini_function(tool) for tool in tools])]


def _tool_message_to_function_response(message: Message) -> types.Part:
	"""Tool messages must carry JSON text so Gemini can parse a function response.

	The message content is expected to be a JSON string with the ToolResult-like
	payload. Plain text is rejected so downstream phases can rely on a stable
	tool-response shape.
	"""

	payload = json.loads(message.content)
	return types.Part(functionResponse=types.FunctionResponse(response=payload))


def _messages_to_contents(messages: list[Message]) -> list[types.Content]:
	contents: list[types.Content] = []
	for message in messages:
		if message.role == "tool":
			parts = [_tool_message_to_function_response(message)]
			contents.append(types.Content(role="user", parts=parts))
			continue

		parts: list[types.Part] = []
		if message.content:
			parts.append(types.Part(text=message.content))

		if message.tool_calls:
			for tool_call in message.tool_calls:
				function_call_kwargs: dict[str, Any] = {
					"name": tool_call.name,
					"args": tool_call.arguments,
				}
				if tool_call.id:
					function_call_kwargs["id"] = tool_call.id
				parts.append(types.Part(functionCall=types.FunctionCall(**function_call_kwargs)))

		role = "user" if message.role == "user" else "model"
		contents.append(types.Content(role=role, parts=parts))

	return contents


def _response_to_message(response: types.GenerateContentResponse) -> Message:
	content_text: list[str] = []
	tool_calls: list[ToolCall] = []

	parts = list(response.parts or [])
	if not parts and response.text:
		content_text.append(response.text)

	for part in parts:
		if part.text:
			content_text.append(part.text)

		if part.function_call:
			function_call = part.function_call
			tool_call_kwargs: dict[str, Any] = {
				"name": function_call.name or "",
				"arguments": function_call.args or {},
			}
			if function_call.id:
				tool_call_kwargs["id"] = function_call.id
			tool_calls.append(ToolCall(**tool_call_kwargs))

	return Message(
		role="assistant",
		content="".join(content_text),
		tool_calls=tool_calls or None,
	)


class GeminiProvider(ModelProvider):
	def __init__(
		self,
		*,
		model: str,
		api_key: str | None = None,
		client: Any | None = None,
	) -> None:
		self._model = model
		if client is not None:
			self._client = client
		elif api_key is not None:
			self._client = Client(api_key=api_key)
		else:
			self._client = Client()

	async def generate_reply(
		self,
		messages: list[Message],
		tools: list[ToolDefinition],
	) -> Message:
		gemini_tools = _tools_to_gemini_tools(tools)
		config = types.GenerateContentConfig(tools=gemini_tools or None)

		try:
			response = await self._client.aio.models.generate_content(
				model=self._model,
				contents=_messages_to_contents(messages),
				config=config,
			)
		except (
			errors.APIError,
			errors.ClientError,
			errors.ServerError,
			errors.UnknownApiResponseError,
			TimeoutError,
		) as exc:
			raise GeminiProviderError(f"Gemini request failed: {exc}") from exc

		return _response_to_message(response)
