from __future__ import annotations

from threading import Condition
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestValueBuffer(Generic[T]):
    """Thread-safe one-slot buffer that replaces unread stale work.

    A producer never blocks behind a slow consumer. Publishing while an item is
    waiting replaces that item and increments ``replaced``. The consumer therefore
    receives the newest available value rather than processing an unbounded backlog.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._item: T | None = None
        self._closed = False
        self._published = 0
        self._replaced = 0

    def publish(self, item: T) -> bool:
        with self._condition:
            if self._closed:
                return False
            if self._item is not None:
                self._replaced += 1
            self._item = item
            self._published += 1
            self._condition.notify()
            return True

    def take(self, timeout_s: float | None = None) -> T | None:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._item is not None or self._closed,
                timeout=timeout_s,
            )
            if not ready or self._item is None:
                return None
            item = self._item
            self._item = None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "published": self._published,
                "replaced": self._replaced,
            }
