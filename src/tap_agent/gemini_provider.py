from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

from google.genai import Client, errors, types

from tap_agent.core_types import Message, ToolCall, ToolDefinition
from tap_agent.provider import ModelProvider


class GeminiProviderError(RuntimeError):
	"""Raised when Gemini request or response handling fails."""


def _to_gemini_schema(tool: ToolDefinition) -> dict[str, Any]:
	def normalize(schema: dict[str, Any]) -> dict[str, Any]:
		res = schema.copy()
		if "type" in res and isinstance(res["type"], str):
			res["type"] = res["type"].upper()
		if "properties" in res:
			res["properties"] = {k: normalize(v) for k, v in res["properties"].items()}
		if "items" in res:
			res["items"] = normalize(res["items"])
		return res
	return normalize(tool.parameters_schema)


def _tool_to_gemini_function(tool: ToolDefinition) -> types.FunctionDeclaration:
	return types.FunctionDeclaration(
		name=tool.name,
		description=tool.description,
		parameters_json_schema=_to_gemini_schema(tool),
	)


def _tools_to_gemini_tools(tools: list[ToolDefinition]) -> list[types.Tool]:
	if not tools:
		return []
	return [types.Tool(function_declarations=[_tool_to_gemini_function(tool) for tool in tools])]


def _tool_message_to_function_response(message: Message) -> types.Part:
	payload = json.loads(message.content)
	return types.Part(
		function_response=types.FunctionResponse(
			name=payload.get("name", ""),
			id=payload.get("id"),
			response=payload,
		)
	)


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
				part_kwargs: dict[str, Any] = {
            		"function_call": types.FunctionCall(**function_call_kwargs)
        		}
				if tool_call.thought_signature:                          
					part_kwargs["thought_signature"] = tool_call.thought_signature
				parts.append(types.Part(**part_kwargs))
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
			if part.thought_signature:                                  
				tool_call_kwargs["thought_signature"] = part.thought_signature
			tool_calls.append(ToolCall(**tool_call_kwargs))

	return Message(
		role="assistant",
		content="".join(content_text),
		tool_calls=tool_calls or None,
	)


class RateLimiter:
	def __init__(self, requests_per_minute: float) -> None:
		self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
		self._lock = asyncio.Lock()
		self._last_call: float = 0.0

	async def wait(self) -> None:
		if self._min_interval <= 0:
			return
		async with self._lock:
			now = time.monotonic()
			elapsed = now - self._last_call
			if elapsed < self._min_interval:
				await asyncio.sleep(self._min_interval - elapsed)
			self._last_call = time.monotonic()


class GeminiProvider(ModelProvider):
	def __init__(
		self,
		*,
		model: str,
		api_key: str | None = None,
		client: Any | None = None,
		requests_per_minute: float = 10.0,
		max_retries: int = 5,
		system_instruction: str | None = None,
	) -> None:
		self._model = model
		self._max_retries = max_retries
		self._system_instruction = system_instruction
		self._rate_limiter = RateLimiter(requests_per_minute)
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
		config_kwargs: dict[str, Any] = {}
		if gemini_tools:
			config_kwargs["tools"] = gemini_tools
		if self._system_instruction:
			config_kwargs["system_instruction"] = self._system_instruction
			
		config = types.GenerateContentConfig(**config_kwargs)

		for attempt in range(1, self._max_retries + 2):
			await self._rate_limiter.wait()
			try:
				response = await self._client.aio.models.generate_content(
					model=self._model,
					contents=_messages_to_contents(messages),
					config=config,
				)
				return _response_to_message(response)
			except (
				errors.APIError,
				errors.ClientError,
				errors.ServerError,
				errors.UnknownApiResponseError,
				TimeoutError,
			) as exc:
				is_rate_limit = isinstance(exc, errors.APIError) and exc.code == 429
				if is_rate_limit and attempt <= self._max_retries:
					# Exponential backoff: 1s, 2s, 4s, 8s, 16s + jitter
					delay = (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
					await asyncio.sleep(delay)
					continue
				
				raise GeminiProviderError(f"Gemini request failed: {exc}") from exc
		
		# Unreachable, added for type checker completeness
		raise GeminiProviderError("Max retries exceeded")
