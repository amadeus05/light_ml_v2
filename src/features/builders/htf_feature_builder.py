from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_adx, compute_atr, compute_linear_regression_slope, compute_rolling_vwap, safe_ratio
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class HtfFeatureBuilder(FeatureBuilderContract):
    block_name = "htf"
    FEATURE_SPECS = {
        "return_4h_1": feature_spec(
            "return_4h_1",
            block_name,
            "shift(log(close / close.shift(1)), 1)",
            description="Lagged 1-bar HTF log return.",
            inputs=("close",),
            shift=1,
        ),
        "return_4h_3": feature_spec(
            "return_4h_3",
            block_name,
            "shift(log(close / close.shift(3)), 1)",
            description="Lagged 3-bar HTF log return.",
            inputs=("close",),
            shift=1,
        ),
        "return_4h_7": feature_spec(
            "return_4h_7",
            block_name,
            "shift(log(close / close.shift(7)), 1)",
            description="Lagged 7-bar HTF log return.",
            inputs=("close",),
            shift=1,
        ),
        "return_4h_14": feature_spec(
            "return_4h_14",
            block_name,
            "shift(log(close / close.shift(14)), 1)",
            description="Lagged 14-bar HTF log return.",
            inputs=("close",),
            shift=1,
        ),
        "realized_vol_4h_returns_20": feature_spec(
            "realized_vol_4h_returns_20",
            block_name,
            "shift(rolling_std(log(close / close.shift(1)), {REALIZED_VOL_WINDOW_4H}), 1)",
            description="Lagged realized volatility of 4H returns.",
            params=(feature_param("REALIZED_VOL_WINDOW_4H", 20),),
            inputs=("close",),
            shift=1,
        ),
        "adx_4h": feature_spec(
            "adx_4h",
            block_name,
            "shift(ADX(high, low, close, 14), 1)",
            description="Lagged ADX on the higher timeframe.",
            inputs=("high", "low", "close"),
            shift=1,
        ),
        "price_position_4h": feature_spec(
            "price_position_4h",
            block_name,
            "shift((close - rolling_min(low, {RANGE_WINDOW_4H})) / (rolling_max(high, {RANGE_WINDOW_4H}) - rolling_min(low, {RANGE_WINDOW_4H})), 1)",
            description="Lagged normalized position inside the recent HTF range.",
            params=(feature_param("RANGE_WINDOW_4H", 14),),
            inputs=("close", "high", "low"),
            shift=1,
        ),
        "distance_to_rolling_high_4h": feature_spec(
            "distance_to_rolling_high_4h",
            block_name,
            "shift((close - rolling_max(high, {RANGE_WINDOW_4H})) / ATR(high, low, close, 14), 1)",
            description="Lagged distance to the recent HTF high in ATR units.",
            params=(feature_param("RANGE_WINDOW_4H", 14),),
            inputs=("close", "high", "low"),
            shift=1,
        ),
        "distance_to_rolling_low_4h": feature_spec(
            "distance_to_rolling_low_4h",
            block_name,
            "shift((close - rolling_min(low, {RANGE_WINDOW_4H})) / ATR(high, low, close, 14), 1)",
            description="Lagged distance to the recent HTF low in ATR units.",
            params=(feature_param("RANGE_WINDOW_4H", 14),),
            inputs=("close", "high", "low"),
            shift=1,
        ),
        "breakout_quality_4h": feature_spec(
            "breakout_quality_4h",
            block_name,
            "shift(signed_breakout_distance(close, rolling_high(high, {RANGE_WINDOW_4H}).shift(1), rolling_low(low, {RANGE_WINDOW_4H}).shift(1)) / ATR(high, low, close, 14), 1)",
            description="Lagged signed Donchian breakout strength normalized by ATR.",
            params=(feature_param("RANGE_WINDOW_4H", 14),),
            inputs=("close", "high", "low"),
            shift=1,
        ),
        "donchian_width_atr_4h": feature_spec(
            "donchian_width_atr_4h",
            block_name,
            "shift((rolling_max(high, {RANGE_WINDOW_4H}) - rolling_min(low, {RANGE_WINDOW_4H})) / ATR(high, low, close, 14), 1)",
            description="Lagged Donchian channel width in ATR units.",
            params=(feature_param("RANGE_WINDOW_4H", 14),),
            inputs=("close", "high", "low"),
            shift=1,
        ),
        "donchian_width_change_4h": feature_spec(
            "donchian_width_change_4h",
            block_name,
            "shift(donchian_width_atr_4h_raw - donchian_width_atr_4h_raw.shift(3), 1)",
            description="Lagged three-bar change in Donchian width.",
            params=(feature_param("RANGE_WINDOW_4H", 14),),
            inputs=("close", "high", "low"),
            dependencies=("donchian_width_atr_4h",),
            shift=1,
        ),
        "ema_slope_4h": feature_spec(
            "ema_slope_4h",
            block_name,
            "shift(linear_regression_slope(ema(close, {EMA_SLOPE_BASE_WINDOW_4H}), {EMA_SLOPE_WINDOW_4H}) / ATR(high, low, close, 14), 1)",
            description="Lagged slope of the HTF EMA baseline normalized by ATR.",
            params=(
                feature_param("EMA_SLOPE_BASE_WINDOW_4H", 21),
                feature_param("EMA_SLOPE_WINDOW_4H", 6),
            ),
            inputs=("close", "high", "low"),
            shift=1,
        ),
        "zscore_vs_vwap_4h": feature_spec(
            "zscore_vs_vwap_4h",
            block_name,
            "shift((close - rolling_vwap(close, high, low, volume, {VWAP_WINDOW_4H}) - rolling_mean(close - rolling_vwap(close, high, low, volume, {VWAP_WINDOW_4H}), {VWAP_WINDOW_4H})) / rolling_std(close - rolling_vwap(close, high, low, volume, {VWAP_WINDOW_4H}), {VWAP_WINDOW_4H}), 1)",
            description="Lagged VWAP distance z-score on the higher timeframe.",
            params=(feature_param("VWAP_WINDOW_4H", 20),),
            inputs=("close", "high", "low", "volume"),
            shift=1,
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        realized_vol_window = max(2, int(getattr(cfg, "REALIZED_VOL_WINDOW_4H", 20)))
        range_window = max(2, int(getattr(cfg, "RANGE_WINDOW_4H", 14)))
        vwap_window = max(2, int(getattr(cfg, "VWAP_WINDOW_4H", 20)))
        ema_slope_base_window = max(2, int(getattr(cfg, "EMA_SLOPE_BASE_WINDOW_4H", 21)))
        ema_slope_window = max(2, int(getattr(cfg, "EMA_SLOPE_WINDOW_4H", 6)))

        if {"return_4h_1", "return_4h_3", "return_4h_7", "return_4h_14"}.intersection(active):
            for period in (1, 3, 7, 14):
                feature_name = f"return_4h_{period}"
                if feature_name in active:
                    output[feature_name] = np.log(close / close.shift(period))

        if "realized_vol_4h_returns_20" in active:
            log_return_4h_1 = np.log(close / close.shift(1))
            output["realized_vol_4h_returns_20"] = log_return_4h_1.rolling(realized_vol_window).std()

        if "adx_4h" in active:
            output["adx_4h"] = compute_adx(high, low, close, length=14)

        structure_request = {
            "price_position_4h",
            "distance_to_rolling_high_4h",
            "distance_to_rolling_low_4h",
            "breakout_quality_4h",
            "donchian_width_atr_4h",
            "donchian_width_change_4h",
            "ema_slope_4h",
        }
        if structure_request.intersection(active):
            rolling_low = low.rolling(range_window).min()
            rolling_high = high.rolling(range_window).max()
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )
            if "price_position_4h" in active:
                output["price_position_4h"] = safe_ratio(close - rolling_low, rolling_high - rolling_low)
            if "distance_to_rolling_high_4h" in active:
                output["distance_to_rolling_high_4h"] = safe_ratio(close - rolling_high, atr_14)
            if "distance_to_rolling_low_4h" in active:
                output["distance_to_rolling_low_4h"] = safe_ratio(close - rolling_low, atr_14)
            if "breakout_quality_4h" in active:
                prev_rolling_low = rolling_low.shift(1)
                prev_rolling_high = rolling_high.shift(1)
                breakout_quality = pd.Series(0.0, index=close.index)
                upside_mask = close > prev_rolling_high
                downside_mask = close < prev_rolling_low
                breakout_quality.loc[upside_mask] = safe_ratio(
                    close.loc[upside_mask] - prev_rolling_high.loc[upside_mask],
                    atr_14.loc[upside_mask],
                )
                breakout_quality.loc[downside_mask] = -safe_ratio(
                    prev_rolling_low.loc[downside_mask] - close.loc[downside_mask],
                    atr_14.loc[downside_mask],
                )
                output["breakout_quality_4h"] = breakout_quality
            if {"donchian_width_atr_4h", "donchian_width_change_4h"}.intersection(active):
                donchian_width_atr = safe_ratio(rolling_high - rolling_low, atr_14)
                if "donchian_width_atr_4h" in active:
                    output["donchian_width_atr_4h"] = donchian_width_atr
                if "donchian_width_change_4h" in active:
                    output["donchian_width_change_4h"] = donchian_width_atr - donchian_width_atr.shift(3)
            if "ema_slope_4h" in active:
                ema_base = context.indicator_cache.get_or_create(
                    "ema_base_4h",
                    lambda: close.ewm(span=ema_slope_base_window, adjust=False).mean(),
                )
                output["ema_slope_4h"] = safe_ratio(
                    compute_linear_regression_slope(ema_base, ema_slope_window),
                    atr_14,
                )

        if "zscore_vs_vwap_4h" in active:
            rolling_vwap = compute_rolling_vwap(close, high, low, frame["volume"], vwap_window)
            vwap_distance = close - rolling_vwap
            vwap_distance_mean = vwap_distance.rolling(vwap_window).mean()
            vwap_distance_std = vwap_distance.rolling(vwap_window).std().replace(0, np.nan)
            output["zscore_vs_vwap_4h"] = (vwap_distance - vwap_distance_mean) / vwap_distance_std

        feature_columns = [column for column in output.columns if column != "timestamp"]
        output[feature_columns] = output[feature_columns].shift(1)
        return output
