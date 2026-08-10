from __future__ import annotations

from tap_agent.agent_harness import AgentHarness, AgentLoopError
from tap_agent.core_types import Message

DEFAULT_MAX_HISTORY_MESSAGES = 40


class CodingSession:

	def __init__(
		self,
		harness: AgentHarness,
		*,
		max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
	) -> None:
		self._harness = harness
		self._max_history_messages = max_history_messages
		self.messages: list[Message] = []

	async def prompt(self, user_input: str) -> Message:
		self.messages.append(Message(role="user", content=user_input))

		try:
			reply = await self._harness.run(list(self.messages))
		except AgentLoopError as exc:
			reply = Message(role="assistant", content=f"[agent error] {exc}")

		self.messages.append(reply)
		self._trim_history()
		return reply

	def _trim_history(self) -> None:
		if len(self.messages) > self._max_history_messages:
			self.messages = self.messages[-self._max_history_messages :]