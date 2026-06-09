ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99
USE_SYMBOL_FEATURE = False

DUAL_V1_FEATURES = [
    "adx_4h",
    "atr_ratio_1h",
    "cross_sectional_rank_ema_fast_slow_1h",
    "distance_to_resistance_1h",
    "distance_to_support_1h",
    "ema_fast_slow",
    "ema_slope_4h",
    "flat_efficiency_1h_24",
    "funding_rate_8h",
    "market_breadth_ema_fast_slow_1h",
    "market_breadth_pos_return_4h_3",
    "mcc_sign_agreement_btc_24h",
    "premium_index_change_24h",
    "price_position_1h",
    "price_position_4h",
    "range_center_distance_atr_1h_48",
    "realized_vol_4h_returns_20",
    "return_4h_14",
    "rsi_1h",
    "volatility_regime_change_1h",
    "zscore_vs_vwap_4h",
]

# Independent copies are intentional: long and short feature selection can
# evolve separately without changing the existing dual model.
LONG_V1_FEATURES = [*DUAL_V1_FEATURES]
SHORT_V1_FEATURES = [*DUAL_V1_FEATURES]
MODEL_UNION_V1_FEATURES = sorted(
    set(DUAL_V1_FEATURES) | set(LONG_V1_FEATURES) | set(SHORT_V1_FEATURES)
)

FEATURE_PROFILES = {
    "all": "__all__",
    "empty": [],
    "dual_v1": DUAL_V1_FEATURES,
    "long_v1": LONG_V1_FEATURES,
    "short_v1": SHORT_V1_FEATURES,
    "model_union_v1": MODEL_UNION_V1_FEATURES,
    "base_only": [
        "realized_vol_1h",
        "ema_fast_slow",
        "return_1h_24",
        "linear_regression_slope_atr_1h_24",
        "trend_efficiency_24h",
        "atr_ratio_1h",
        "range_compression_1h",
        "price_position_1h",
        "relative_strength_vs_btc_24h",
        "residual_return_24h",
        "return_4h_3",
        "return_4h_14",
        "ema_slope_4h",
        "realized_vol_4h_returns_20",
        "market_breadth_pos_return_4h_3",
        "market_dispersion_return_4h_3",
        "zscore_vs_vwap_4h",
        "vol_ratio",
        "price_position_4h",
        "adx_4h",
    ],
}

MODEL_PROFILES = {
    "dual_v1": {
        "mode": "dual",
        "feature_profile": "dual_v1",
        "target_column": "Target",
        "artifact_name": "lightgbm_target",
    },
    "long_v1": {
        "mode": "long",
        "feature_profile": "long_v1",
        "target_column": "TargetLong",
        "positive_label": 1,
        "artifact_name": "lightgbm_long",
    },
    "short_v1": {
        "mode": "short",
        "feature_profile": "short_v1",
        "target_column": "TargetShort",
        "positive_label": 1,
        "artifact_name": "lightgbm_short",
    },
}

ACTIVE_MODEL_PROFILE = "dual_v1"


def get_model_profile(name: str = ACTIVE_MODEL_PROFILE) -> dict:
    if name not in MODEL_PROFILES:
        known = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown model profile '{name}'. Known profiles: {known}")

    profile = dict(MODEL_PROFILES[name])
    feature_profile = profile.get("feature_profile")
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(
            f"Model profile '{name}' references unknown feature profile "
            f"'{feature_profile}'."
        )
    if not profile.get("target_column"):
        raise ValueError(f"Model profile '{name}' must define target_column.")
    if not profile.get("artifact_name"):
        raise ValueError(f"Model profile '{name}' must define artifact_name.")
    return profile


FEATURE_BUILD_REQUEST = {
    "profile": "model_union_v1",
    "include_features": [],
    "exclude_features": [],
    "exclude_blocks": [],
}

# Features required to produce labels or other pipeline outputs, but excluded
# from the model input unless they are also present in the selected profile.
FEATURE_PIPELINE_REQUIRED_FEATURES = [
    "realized_vol_1h",
]

MCC_SIGN_BTC_WINDOW = 24
MCC_SIGN_BTC_MIN_PERIODS = None
RSI_LENGTH = 14
