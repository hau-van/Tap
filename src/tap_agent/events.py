
from __future__ import annotations
import asyncio
from typing import AsyncIterator
from tap_agent.core_types import AgentEvent

class EventStream:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[AgentEvent]] = []

    async def publish(self, event: AgentEvent) -> None:
       for queue in list(self._subscribers):
            queue.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[AgentEvent]:
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
        return len(self._subscribers)
    
