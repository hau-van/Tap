from __future__ import annotations

from typing import Protocol

from tap_agent.core_types import Message, ToolDefinition


class ModelProvider(Protocol):
    """Stable interface for all model providers used by later phases."""
    
    async def generate_reply(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Message:
        ...


class MockProvider:
    """Deterministic provider used for tests and offline agent-loop runs."""

    def __init__(self, script: list[Message]) -> None:
        self._script = list(script)
        self._index = 0

    async def generate_reply(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Message:
        del messages, tools

        if self._index >= len(self._script):
            raise RuntimeError("MockProvider script is exhausted")

        reply = self._script[self._index]
        self._index += 1
        return reply
    
