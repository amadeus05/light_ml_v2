HORIZON = 12
TP_PCT = 0.03
SL_PCT = 0.015

ENABLE_ADAPTIVE_HORIZON = True
ADAPTIVE_HORIZON_MIN = 8
ADAPTIVE_HORIZON_MAX = 20
ADAPTIVE_HORIZON_VOL_LOW = 0.005
ADAPTIVE_HORIZON_VOL_HIGH = 0.025

USE_DYNAMIC_BARRIERS = True
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_RVOL_MULTIPLIER = 0.75
BARRIER_TP_TO_SL_RATIO = 2.0
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06


def effective_max_label_horizon() -> int:
    base_horizon = max(1, int(HORIZON))
    if not ENABLE_ADAPTIVE_HORIZON:
        return base_horizon

    min_horizon = max(1, int(ADAPTIVE_HORIZON_MIN))
    max_horizon = max(min_horizon, int(ADAPTIVE_HORIZON_MAX))
    return max(base_horizon, max_horizon)
