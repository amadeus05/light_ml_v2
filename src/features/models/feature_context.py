from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.features.models.indicator_cache import IndicatorCache
from src.timeframes import duration_to_bars


@dataclass(slots=True)
class FeatureContext:
    frame: pd.DataFrame
    symbol: str
    timeframe: str = "1h"
    htf_timeframe: str = "4h"
    source_timeframe: str | None = None
    base_feature_map: dict[str, pd.DataFrame] | None = None
    htf_feature_map: dict[str, pd.DataFrame] | None = None
    indicator_cache: IndicatorCache = field(default_factory=IndicatorCache)
    shared_cache: dict[str, Any] = field(default_factory=dict)

    def bars(self, duration: str, *, minimum: int = 1) -> int:
        return duration_to_bars(
            duration,
            self.source_timeframe or self.timeframe,
            minimum=minimum,
        )
