ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99
USE_SYMBOL_FEATURE = False

PRODUCTION_V1_FEATURES = [
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

FEATURE_PROFILES = {
    "all": "__all__",
    "empty": [],
    "production_v1": PRODUCTION_V1_FEATURES,
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

FEATURE_BUILD_REQUEST = {
    "profile": "production_v1",
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
