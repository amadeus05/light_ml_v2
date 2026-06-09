from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class IndicatorCache:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        if key not in self._values:
            self._values[key] = factory()
        return self._values[key]  # type: ignore[return-value]
