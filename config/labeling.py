from .timeframes import get_timeframe_profile

from src.timeframes import duration_to_bars


HORIZON = 12
HORIZON_DURATION = "12h"
LABELING_CONTRACT_VERSION = "triple_barrier_horizon_close_v1"
VERTICAL_BARRIER_EXIT = "horizon_close"
TP_PCT = 0.03
SL_PCT = 0.015

ENABLE_ADAPTIVE_HORIZON = True
ADAPTIVE_HORIZON_MIN = 8
ADAPTIVE_HORIZON_MAX = 20
ADAPTIVE_HORIZON_MIN_DURATION = "8h"
ADAPTIVE_HORIZON_MAX_DURATION = "20h"
ADAPTIVE_HORIZON_VOL_LOW = 0.005
ADAPTIVE_HORIZON_VOL_HIGH = 0.025

USE_DYNAMIC_BARRIERS = True
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_RVOL_MULTIPLIER = 0.75
BARRIER_TP_TO_SL_RATIO = 2.0
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06


def effective_max_label_horizon(timeframe_profile: str | None = None) -> int:
    profile = get_timeframe_profile(timeframe_profile or "1h_v1")
    timeframe = profile["timeframe"]
    base_horizon = duration_to_bars(HORIZON_DURATION, timeframe)
    if not ENABLE_ADAPTIVE_HORIZON:
        return base_horizon

    min_horizon = duration_to_bars(ADAPTIVE_HORIZON_MIN_DURATION, timeframe)
    max_horizon = max(
        min_horizon,
        duration_to_bars(ADAPTIVE_HORIZON_MAX_DURATION, timeframe),
    )
    return max(base_horizon, max_horizon)
