from __future__ import annotations

import math

import pandas as pd


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    try:
        delta = pd.to_timedelta(timeframe)
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe or duration: {timeframe}") from exc
    if delta <= pd.Timedelta(0):
        raise ValueError(f"Timeframe must be positive: {timeframe}")
    return delta


def duration_to_bars(
    duration: str | pd.Timedelta,
    timeframe: str,
    *,
    minimum: int = 1,
) -> int:
    duration_delta = timeframe_delta(str(duration))
    bar_delta = timeframe_delta(timeframe)
    return max(minimum, int(math.ceil(duration_delta / bar_delta)))


def timeframe_ratio(slower_timeframe: str, faster_timeframe: str) -> float:
    return float(timeframe_delta(slower_timeframe) / timeframe_delta(faster_timeframe))


def timeframe_hours(timeframe: str) -> float:
    return float(timeframe_delta(timeframe) / pd.Timedelta(hours=1))
