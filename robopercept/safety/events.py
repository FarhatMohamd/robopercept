"""Tiny publish/subscribe bus used to fan out events to sinks (log, MQTT, web)."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable

from ..structures import Event

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, history: int = 200) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self.history: deque[Event] = deque(maxlen=history)

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        self._subscribers.append(callback)

    def publish(self, event: Event) -> None:
        self.history.append(event)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:  # a faulty sink must never stop perception
                log.exception("Event subscriber failed")
