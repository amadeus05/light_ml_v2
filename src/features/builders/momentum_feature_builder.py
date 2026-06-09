from __future__ import annotations

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
        return_windows = {
            "return_1h_6": context.bars("6h"),
            "return_1h_12": context.bars("12h"),
            "return_1h_24": context.bars("24h"),
        }
        ema_fast_window = context.bars("12h", minimum=2)
        ema_slow_window = context.bars("48h", minimum=ema_fast_window + 1)
        rsi_window = context.bars("14h", minimum=2)
        atr_window = context.bars("14h", minimum=2)
        slope_short_window = context.bars("12h", minimum=2)
        slope_long_window = context.bars("24h", minimum=slope_short_window + 1)
        acceleration_window = context.bars("3h")

        if {"return_1h_6", "return_1h_12", "return_1h_24"}.intersection(active):
            for feature_name, period in return_windows.items():
                if feature_name in active:
                    output[feature_name] = np.log(close / close.shift(period))

        if "rsi_1h" in active:
            output["rsi_1h"] = context.indicator_cache.get_or_create(
                f"rsi_close_{rsi_window}",
                lambda: compute_rsi(close, rsi_window),
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
                f"atr_{atr_window}",
                lambda: compute_atr(frame["high"], frame["low"], close, length=atr_window),
            )
            slope_12 = None
            slope_24 = None
            if {"linear_regression_slope_atr_1h_12", "slope_acceleration_1h_12_24"}.intersection(active):
                slope_12 = context.indicator_cache.get_or_create(
                    f"lr_close_{slope_short_window}",
                    lambda: compute_linear_regression_slope(close, slope_short_window),
                )
            if {"linear_regression_slope_atr_1h_24", "slope_acceleration_1h_12_24"}.intersection(active):
                slope_24 = context.indicator_cache.get_or_create(
                    f"lr_close_{slope_long_window}",
                    lambda: compute_linear_regression_slope(close, slope_long_window),
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
                output["trend_persistence_score_12"] = signed_step.rolling(slope_short_window).mean()
            if "trend_persistence_score_24" in active:
                output["trend_persistence_score_24"] = signed_step.rolling(slope_long_window).mean()

        if "trend_efficiency_24h" in active:
            output["trend_efficiency_24h"] = context.indicator_cache.get_or_create(
                "trend_efficiency_24h",
                lambda: compute_trend_efficiency(close, slope_long_window),
            )

        if "ema_slope_acceleration_1h" in active:
            if ema_fast_slow is None:
                raise ValueError("Feature 'ema_slope_acceleration_1h' requires 'ema_fast_slow'.")
            output["ema_slope_acceleration_1h"] = (
                ema_fast_slow - ema_fast_slow.shift(acceleration_window)
            )

        return output
