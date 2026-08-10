"""Unit tests cho tap_agent.provider.

Kiểm tra 1 nhóm:
1. MockProvider trả reply theo script tuần tự
"""
from __future__ import annotations

import asyncio

import pytest

from tap_agent.core_types import Message
from tap_agent.provider import MockProvider


# ══════════════════════════════════════════════════════════════
# 1. MockProvider behavior
# ══════════════════════════════════════════════════════════════


class TestMockProvider:
	def test_returns_scripted_messages_in_order(self):
		script = [
			Message(role="assistant", content="first"),
			Message(role="assistant", content="second"),
		]
		provider = MockProvider(script)

		first = asyncio.run(provider.generate_reply(messages=[], tools=[]))
		second = asyncio.run(provider.generate_reply(messages=[], tools=[]))

		assert first.content == "first"
		assert second.content == "second"

	def test_raises_when_script_is_exhausted(self):
		provider = MockProvider([Message(role="assistant", content="only")])

		asyncio.run(provider.generate_reply(messages=[], tools=[]))

		with pytest.raises(RuntimeError, match="MockProvider script is exhausted"):
			asyncio.run(provider.generate_reply(messages=[], tools=[]))