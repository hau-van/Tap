"""Simple async event stream cho AgentEvent.

Multi-subscriber pub/sub dùng asyncio.Queue. Mỗi subscriber có queue
riêng, publish() fan-out ra tất cả. Subscriber tự kết thúc khi nhận
được event có type="done".

Chỉ chứa cơ chế pub/sub. Không có event processing, filtering, replay.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from tap_agent.core_types import AgentEvent


class EventStream:
    """Async pub/sub cho AgentEvent.

    Usage:
        stream = EventStream()

        # Publisher (agent loop):
        await stream.publish(AgentEvent(type="message", payload=msg))

        # Subscriber (CLI, logger, ...):
        async for event in stream.subscribe():
            handle(event)
            # tự break khi nhận event type="done"
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[AgentEvent]] = []

    async def publish(self, event: AgentEvent) -> None:
        """Gửi event tới tất cả subscribers hiện tại.

        Nếu chưa có subscriber nào, event bị drop (không buffer).
        """
        for queue in self._subscribers:
            await queue.put(event)

    async def subscribe(self) -> AsyncIterator[AgentEvent]:
        """Yield từng event cho đến khi nhận được event type="done".

        Queue riêng cho subscriber này, tự động cleanup khi generator đóng
        (break, exception, hoặc consumer hủy).
        """
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.type == "done":
                    break
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    @property
    def subscriber_count(self) -> int:
        """Số subscriber đang active. Hữu ích cho testing/debugging."""
        return len(self._subscribers)
    