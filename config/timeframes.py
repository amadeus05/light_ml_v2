from .data import HTF_TIMEFRAME, TIMEFRAME


TIMEFRAME_PROFILES = {
    "1h_v1": {
        "timeframe": TIMEFRAME,
        "htf_timeframe": HTF_TIMEFRAME,
    },
    "15m_v1": {
        "timeframe": "15m",
        "htf_timeframe": "1h",
    },
}

ACTIVE_TIMEFRAME_PROFILE = "1h_v1"


def parse_timeframe_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    raw_values = value.split(",") if isinstance(value, str) else value
    timeframes = list(
        dict.fromkeys(
            str(item).strip().lower()
            for item in raw_values
            if str(item).strip()
        )
    )
    if not timeframes:
        raise ValueError("At least one higher timeframe must be configured.")
    return timeframes


def get_timeframe_profile(name: str = ACTIVE_TIMEFRAME_PROFILE) -> dict:
    if name not in TIMEFRAME_PROFILES:
        known = ", ".join(sorted(TIMEFRAME_PROFILES))
        raise ValueError(f"Unknown timeframe profile '{name}'. Known profiles: {known}")

    profile = dict(TIMEFRAME_PROFILES[name])
    htf_timeframes = parse_timeframe_list(profile["htf_timeframe"])
    profile["htf_timeframes"] = htf_timeframes
    profile["htf_timeframe"] = htf_timeframes[0]
    profile["name"] = name
    return profile
