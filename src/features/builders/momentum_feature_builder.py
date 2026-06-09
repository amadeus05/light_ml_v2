from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, compute_linear_regression_slope, compute_rsi, compute_trend_efficiency, safe_ratio
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class MomentumFeatureBuilder(FeatureBuilderContract):
    block_name = "momentum"
    FEATURE_SPECS = {
        "return_1h_6": feature_spec(
            "return_1h_6",
            block_name,
            "log(close / close.shift(6))",
            description="6-bar log return on the main timeframe.",
            inputs=("close",),
        ),
        "return_1h_12": feature_spec(
            "return_1h_12",
            block_name,
            "log(close / close.shift(12))",
            description="12-bar log return on the main timeframe.",
            inputs=("close",),
        ),
        "return_1h_24": feature_spec(
            "return_1h_24",
            block_name,
            "log(close / close.shift(24))",
            description="24-bar log return on the main timeframe.",
            inputs=("close",),
        ),
        "ema_fast_slow": feature_spec(
            "ema_fast_slow",
            block_name,
            "(ema(close, {EMA_FAST_WINDOW}) - ema(close, {EMA_SLOW_WINDOW})) / ema(close, {EMA_SLOW_WINDOW})",
            description="Relative spread between fast and slow EMA.",
            params=(
                feature_param("EMA_FAST_WINDOW", 12),
                feature_param("EMA_SLOW_WINDOW", 48),
            ),
            inputs=("close",),
        ),
        "rsi_1h": feature_spec(
            "rsi_1h",
            block_name,
            "RSI(close, {RSI_LENGTH}) Wilder on main timeframe",
            description="Relative Strength Index (Wilder) on main timeframe close.",
            params=(feature_param("RSI_LENGTH", 14),),
            inputs=("close",),
        ),
        "linear_regression_slope_atr_1h_12": feature_spec(
            "linear_regression_slope_atr_1h_12",
            block_name,
            "linear_regression_slope(close, 12) / ATR(high, low, close, 14)",
            description="12-bar linear regression slope normalized by ATR.",
            inputs=("close", "high", "low"),
        ),
        "linear_regression_slope_atr_1h_24": feature_spec(
            "linear_regression_slope_atr_1h_24",
            block_name,
            "linear_regression_slope(close, 24) / ATR(high, low, close, 14)",
            description="24-bar linear regression slope normalized by ATR.",
            inputs=("close", "high", "low"),
        ),
        "trend_persistence_score_12": feature_spec(
            "trend_persistence_score_12",
            block_name,
            "rolling_mean(sign(diff(close)), 12)",
            description="Average sign of close-to-close moves over 12 bars.",
            inputs=("close",),
        ),
        "trend_persistence_score_24": feature_spec(
            "trend_persistence_score_24",
            block_name,
            "rolling_mean(sign(diff(close)), 24)",
            description="Average sign of close-to-close moves over 24 bars.",
            inputs=("close",),
        ),
        "trend_efficiency_24h": feature_spec(
            "trend_efficiency_24h",
            block_name,
            "abs(close - close.shift(24)) / rolling_sum(abs(diff(close)), 24)",
            description="Trend efficiency ratio over the last 24 bars.",
            inputs=("close",),
        ),
        "slope_acceleration_1h_12_24": feature_spec(
            "slope_acceleration_1h_12_24",
            block_name,
            "(linear_regression_slope(close, 12) - linear_regression_slope(close, 24)) / ATR(high, low, close, 14)",
            description="Difference between short and long slope, normalized by ATR.",
            inputs=("close", "high", "low"),
        ),
        "ema_slope_acceleration_1h": feature_spec(
            "ema_slope_acceleration_1h",
            block_name,
            "ema_fast_slow - ema_fast_slow.shift(3)",
            description="Three-bar acceleration of the EMA spread signal.",
            dependencies=("ema_fast_slow",),
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        close = frame["close"]
        ema_fast_window = max(2, int(getattr(cfg, "EMA_FAST_WINDOW", 12)))
        ema_slow_window = max(ema_fast_window + 1, int(getattr(cfg, "EMA_SLOW_WINDOW", 48)))

        if {"return_1h_6", "return_1h_12", "return_1h_24"}.intersection(active):
            for period in (6, 12, 24):
                feature_name = f"return_1h_{period}"
                if feature_name in active:
                    output[feature_name] = np.log(close / close.shift(period))

        if "rsi_1h" in active:
            rsi_len = max(2, int(getattr(cfg, "RSI_LENGTH", 14)))
            output["rsi_1h"] = context.indicator_cache.get_or_create(
                f"rsi_close_{rsi_len}",
                lambda: compute_rsi(close, rsi_len),
            )

        ema_fast_slow = None
        if {"ema_fast_slow", "ema_slope_acceleration_1h"}.intersection(active):
            ema_fast = context.indicator_cache.get_or_create(
                "ema_fast",
                lambda: close.ewm(span=ema_fast_window, adjust=False).mean(),
            )
            ema_slow = context.indicator_cache.get_or_create(
                "ema_slow",
                lambda: close.ewm(span=ema_slow_window, adjust=False).mean(),
            )
            ema_fast_slow = context.indicator_cache.get_or_create(
                "ema_fast_slow",
                lambda: safe_ratio(ema_fast - ema_slow, ema_slow),
            )
            if "ema_fast_slow" in active:
                output["ema_fast_slow"] = ema_fast_slow

        slope_request = {
            "linear_regression_slope_atr_1h_12",
            "linear_regression_slope_atr_1h_24",
            "slope_acceleration_1h_12_24",
        }
        if slope_request.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(frame["high"], frame["low"], close, length=14),
            )
            slope_12 = None
            slope_24 = None
            if {"linear_regression_slope_atr_1h_12", "slope_acceleration_1h_12_24"}.intersection(active):
                slope_12 = context.indicator_cache.get_or_create(
                    "lr_close_12",
                    lambda: compute_linear_regression_slope(close, 12),
                )
            if {"linear_regression_slope_atr_1h_24", "slope_acceleration_1h_12_24"}.intersection(active):
                slope_24 = context.indicator_cache.get_or_create(
                    "lr_close_24",
                    lambda: compute_linear_regression_slope(close, 24),
                )
            if "linear_regression_slope_atr_1h_12" in active and slope_12 is not None:
                output["linear_regression_slope_atr_1h_12"] = safe_ratio(slope_12, atr_14)
            if "linear_regression_slope_atr_1h_24" in active and slope_24 is not None:
                output["linear_regression_slope_atr_1h_24"] = safe_ratio(slope_24, atr_14)
            if "slope_acceleration_1h_12_24" in active and slope_12 is not None and slope_24 is not None:
                output["slope_acceleration_1h_12_24"] = safe_ratio(slope_12 - slope_24, atr_14)

        if {"trend_persistence_score_12", "trend_persistence_score_24"}.intersection(active):
            signed_step = context.indicator_cache.get_or_create(
                "signed_step",
                lambda: pd.Series(np.sign(close.diff()), index=close.index),
            )
            if "trend_persistence_score_12" in active:
                output["trend_persistence_score_12"] = signed_step.rolling(12).mean()
            if "trend_persistence_score_24" in active:
                output["trend_persistence_score_24"] = signed_step.rolling(24).mean()

        if "trend_efficiency_24h" in active:
            output["trend_efficiency_24h"] = context.indicator_cache.get_or_create(
                "trend_efficiency_24h",
                lambda: compute_trend_efficiency(close, 24),
            )

        if "ema_slope_acceleration_1h" in active:
            if ema_fast_slow is None:
                raise ValueError("Feature 'ema_slope_acceleration_1h' requires 'ema_fast_slow'.")
            output["ema_slope_acceleration_1h"] = ema_fast_slow - ema_fast_slow.shift(3)

        return output
