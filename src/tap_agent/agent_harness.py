from __future__ import annotations

from collections.abc import Awaitable, Callable

from tap_agent.agent_loop import run_agent_loop
from tap_agent.core_types import AgentEvent, Message, ToolCall, ToolDefinition, ToolResult
from tap_agent.events import EventStream
from tap_agent.provider import ModelProvider

# Contract with Phase 5: a ToolExecutor takes one ToolCall and MUST return a
# ToolResult — it must never raise (see tools.execute_tool()'s docstring).
# AgentHarness still guards against a misbehaving executor defensively (see
# `run()` below), but that guard is a last resort, not a substitute for
# Phase 5 honoring its own contract.
ToolExecutor = Callable[[ToolCall], Awaitable[ToolResult]]


class AgentLoopError(RuntimeError):
	"""Raised when run_agent_loop() terminates with an AgentEvent.error
	(e.g. "max iterations reached") instead of a final message."""


class AgentHarness:
	

	def __init__(
		self,
		*,
		provider: ModelProvider,
		tools: list[ToolDefinition],
		tool_executor: ToolExecutor,
		events: EventStream | None = None,
		max_iterations: int = 10,
	) -> None:
		self._provider = provider
		self._tools = tools
		self._tool_executor = tool_executor
		self._events = events
		self._max_iterations = max_iterations

	async def run(self, messages: list[Message]) -> Message:
		
		gen = run_agent_loop(
			messages, self._tools, self._provider, max_iterations=self._max_iterations
		)
		sent: ToolResult | None = None
		final_message: Message | None = None

		while True:
			try:
				event = await gen.asend(sent)
			except StopAsyncIteration:
				break

			if self._events is not None:
				await self._events.publish(event)
			sent = None

			if event.type == "tool_call":
				tool_call = event.payload
				assert isinstance(tool_call, ToolCall)
				result = await self._execute_tool_safely(tool_call)
				if self._events is not None:
					await self._events.publish(AgentEvent.tool_result(result))
				sent = result
			elif event.type == "message":
				assert isinstance(event.payload, Message)
				final_message = event.payload
			elif event.type == "error":
				raise AgentLoopError(event.payload)

		if final_message is None:
			raise AgentLoopError("agent loop ended without a final message")
		return final_message

	async def _execute_tool_safely(self, tool_call: ToolCall) -> ToolResult:
		"""Call the injected tool_executor, with a defensive fallback.

		Phase 5's `execute_tool()` must never raise — but a single bad
		tool call must never be able to crash the whole session even if
		that contract is ever violated by a future change.
		"""
		try:
			return await self._tool_executor(tool_call)
		except Exception as exc:  # pragma: no cover - defensive last resort
			return ToolResult.fail(tool_call.id, f"tool executor crashed: {exc}")