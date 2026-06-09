TIMEFRAME_PROFILES = {
    "1h_v1": {
        "timeframe": "1h",
        "htf_timeframe": "4h",
    },
    "15m_v1": {
        "timeframe": "15m",
        "htf_timeframe": "1h",
    },
}

ACTIVE_TIMEFRAME_PROFILE = "1h_v1"


def get_timeframe_profile(name: str = ACTIVE_TIMEFRAME_PROFILE) -> dict:
    if name not in TIMEFRAME_PROFILES:
        known = ", ".join(sorted(TIMEFRAME_PROFILES))
        raise ValueError(f"Unknown timeframe profile '{name}'. Known profiles: {known}")

    profile = dict(TIMEFRAME_PROFILES[name])
    profile["name"] = name
    return profile
