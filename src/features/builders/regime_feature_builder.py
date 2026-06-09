from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, safe_ratio
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class RegimeFeatureBuilder(FeatureBuilderContract):
    block_name = "regime"
    FEATURE_SPECS = {
        "realized_vol_1h": feature_spec("realized_vol_1h", block_name, "rolling_std(log(close / close.shift(1)), {REALIZED_VOL_WINDOW_1H})", description="Realized volatility of 1H log returns.", params=(feature_param("REALIZED_VOL_WINDOW_1H", 24),), inputs=("close",)),
        "atr_ratio_1h": feature_spec("atr_ratio_1h", block_name, "ATR(high, low, close, 14) / ATR(high, low, close, 100)", description="Short ATR relative to long ATR.", inputs=("high", "low", "close")),
        "volatility_regime_change_1h": feature_spec("volatility_regime_change_1h", block_name, "ATR(high, low, close, 6) / ATR(high, low, close, 48)", description="Short-vs-long ATR ratio used as a regime shift proxy.", inputs=("high", "low", "close")),
        "range_compression_1h": feature_spec("range_compression_1h", block_name, "(rolling_max(high, {RANGE_COMPRESSION_SHORT_WINDOW_1H}) - rolling_min(low, {RANGE_COMPRESSION_SHORT_WINDOW_1H})) / (rolling_max(high, {RANGE_COMPRESSION_LONG_WINDOW_1H}) - rolling_min(low, {RANGE_COMPRESSION_LONG_WINDOW_1H}))", description="Short range divided by long range.", params=(feature_param("RANGE_COMPRESSION_SHORT_WINDOW_1H", 12), feature_param("RANGE_COMPRESSION_LONG_WINDOW_1H", 48)), inputs=("high", "low")),
        "volatility_acceleration_1h": feature_spec("volatility_acceleration_1h", block_name, "volatility_regime_change_1h - volatility_regime_change_1h.shift(3)", description="Three-bar acceleration of the regime change signal.", dependencies=("volatility_regime_change_1h",)),
        "volume_24h": feature_spec("volume_24h", block_name, "rolling_sum(volume, {VOLUME_24H_WINDOW_1H})", description="Rolling traded volume over the last 24 bars.", params=(feature_param("VOLUME_24H_WINDOW_1H", 24),), inputs=("volume",)),
        "volume_ratio_1h": feature_spec("volume_ratio_1h", block_name, "volume / rolling_mean(volume, {VOLUME_RATIO_WINDOW_1H})", description="Current volume relative to its rolling mean.", params=(feature_param("VOLUME_RATIO_WINDOW_1H", 24),), inputs=("volume",)),
        "volume_zscore_1h": feature_spec("volume_zscore_1h", block_name, "(volume - rolling_mean(volume, {VOLUME_ZSCORE_WINDOW_1H})) / rolling_std(volume, {VOLUME_ZSCORE_WINDOW_1H})", description="Volume z-score on a long rolling window.", params=(feature_param("VOLUME_ZSCORE_WINDOW_1H", 24 * 7),), inputs=("volume",)),
        "dollar_volume_zscore_1h": feature_spec("dollar_volume_zscore_1h", block_name, "(log1p(max(close * volume, 0)) - rolling_mean(log1p(max(close * volume, 0)), {VOLUME_ZSCORE_WINDOW_1H})) / rolling_std(log1p(max(close * volume, 0)), {VOLUME_ZSCORE_WINDOW_1H})", description="Dollar-volume z-score with log scaling.", params=(feature_param("VOLUME_ZSCORE_WINDOW_1H", 24 * 7),), inputs=("close", "volume")),
        "vol_of_vol_1h": feature_spec("vol_of_vol_1h", block_name, "rolling_std(realized_vol_1h, 12)", description="Volatility of realized volatility.", dependencies=("realized_vol_1h",)),
        "realized_vol_vs_ema": feature_spec("realized_vol_vs_ema", block_name, "(realized_vol_1h - ema(realized_vol_1h, 48)) / ema(realized_vol_1h, 48)", description="Deviation of realized volatility from its EMA baseline.", dependencies=("realized_vol_1h",)),
        "volatility_regime_stability": feature_spec("volatility_regime_stability", block_name, "clip(run_length(vol_regime_label(realized_vol_1h, rolling_mean(realized_vol_1h, 96), low=0.8, high=1.2)), 0, 48) / 48", description="Normalized age of the current volatility regime.", dependencies=("realized_vol_1h",)),
        "high_vol_stress_indicator": feature_spec("high_vol_stress_indicator", block_name, "1{ATR(high, low, close, 14) > 1.1 * ATR(high, low, close, 48) and realized_vol_1h > rolling_quantile(realized_vol_1h, 96, 0.75)}", description="Binary stress flag for expanding ranges during high volatility.", inputs=("high", "low", "close"), dependencies=("realized_vol_1h",)),
        "vol_regime_classification": feature_spec("vol_regime_classification", block_name, "1{realized_vol_1h > rolling_quantile(realized_vol_1h, 96, 0.33)} + 1{realized_vol_1h > rolling_quantile(realized_vol_1h, 96, 0.67)}", description="Discrete volatility regime bucket: 0 low, 1 normal, 2 high.", dependencies=("realized_vol_1h",)),
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
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        realized_vol_window = context.bars("24h", minimum=2)
        range_short_window = context.bars("12h", minimum=2)
        range_long_window = context.bars("48h", minimum=range_short_window + 1)
        volume_ratio_window = context.bars("24h", minimum=2)
        volume_zscore_window = context.bars("7d", minimum=24)
        volume_24h_window = context.bars("24h", minimum=2)
        atr_6_window = context.bars("6h", minimum=2)
        atr_14_window = context.bars("14h", minimum=2)
        atr_48_window = context.bars("48h", minimum=atr_14_window + 1)
        atr_100_window = context.bars("100h", minimum=atr_48_window + 1)
        acceleration_window = context.bars("3h")
        regime_window = context.bars("96h", minimum=10)
        regime_age_window = context.bars("48h", minimum=2)
        vol_of_vol_window = context.bars("12h", minimum=2)

        if "realized_vol_1h" in active:
            log_return_1h_1 = np.log(close / close.shift(1))
            hourly_scale = np.sqrt(context.bars("1h"))
            output["realized_vol_1h"] = (
                log_return_1h_1.rolling(realized_vol_window).std() * hourly_scale
            )

        if {"atr_ratio_1h", "volatility_regime_change_1h", "volatility_acceleration_1h"}.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                f"atr_{atr_14_window}",
                lambda: compute_atr(high, low, close, length=atr_14_window),
            )
            if "atr_ratio_1h" in active:
                atr_100 = context.indicator_cache.get_or_create(
                    f"atr_{atr_100_window}",
                    lambda: compute_atr(high, low, close, length=atr_100_window),
                )
                output["atr_ratio_1h"] = safe_ratio(atr_14, atr_100)

            if {"volatility_regime_change_1h", "volatility_acceleration_1h"}.intersection(active):
                atr_6 = context.indicator_cache.get_or_create(
                    f"atr_{atr_6_window}",
                    lambda: compute_atr(high, low, close, length=atr_6_window),
                )
                atr_48 = context.indicator_cache.get_or_create(
                    f"atr_{atr_48_window}",
                    lambda: compute_atr(high, low, close, length=atr_48_window),
                )
                regime_change = context.indicator_cache.get_or_create(
                    "volatility_regime_change_1h",
                    lambda: safe_ratio(atr_6, atr_48),
                )
                if "volatility_regime_change_1h" in active:
                    output["volatility_regime_change_1h"] = regime_change
                if "volatility_acceleration_1h" in active:
                    output["volatility_acceleration_1h"] = (
                        regime_change - regime_change.shift(acceleration_window)
                    )

        if "range_compression_1h" in active:
            range_short = high.rolling(range_short_window).max() - low.rolling(range_short_window).min()
            range_long = high.rolling(range_long_window).max() - low.rolling(range_long_window).min()
            output["range_compression_1h"] = safe_ratio(range_short, range_long)

        if "volume_24h" in active:
            output["volume_24h"] = volume.rolling(volume_24h_window).sum()

        volume_request = {"volume_ratio_1h", "volume_zscore_1h", "dollar_volume_zscore_1h"}
        if volume_request.intersection(active):
            if {"volume_ratio_1h", "dollar_volume_zscore_1h"}.intersection(active):
                volume_mean = volume.rolling(volume_ratio_window).mean()
            if "volume_ratio_1h" in active:
                output["volume_ratio_1h"] = safe_ratio(volume, volume_mean)

            if {"volume_zscore_1h", "dollar_volume_zscore_1h"}.intersection(active):
                volume_roll_mean = volume.rolling(volume_zscore_window).mean()
                volume_roll_std = volume.rolling(volume_zscore_window).std().replace(0, np.nan)
                dollar_volume = np.log1p((close * volume).clip(lower=0))
                dollar_roll_mean = dollar_volume.rolling(volume_zscore_window).mean()
                dollar_roll_std = dollar_volume.rolling(volume_zscore_window).std().replace(0, np.nan)
            if "volume_zscore_1h" in active:
                output["volume_zscore_1h"] = (volume - volume_roll_mean) / volume_roll_std
            if "dollar_volume_zscore_1h" in active:
                output["dollar_volume_zscore_1h"] = (dollar_volume - dollar_roll_mean) / dollar_roll_std

        regime_request = {
            "vol_of_vol_1h",
            "realized_vol_vs_ema",
            "volatility_regime_stability",
            "high_vol_stress_indicator",
            "vol_regime_classification",
        }
        if regime_request.intersection(active):
            rvol = output.get("realized_vol_1h")
            if rvol is None and "realized_vol_1h" in active:
                log_return_1h_1 = np.log(close / close.shift(1))
                rvol = (
                    log_return_1h_1.rolling(realized_vol_window).std()
                    * np.sqrt(context.bars("1h"))
                )

            if rvol is not None:
                if "vol_of_vol_1h" in active:
                    output["vol_of_vol_1h"] = rvol.rolling(vol_of_vol_window).std()

                if "realized_vol_vs_ema" in active:
                    rvol_ema = rvol.ewm(span=regime_age_window, adjust=False).mean()
                    output["realized_vol_vs_ema"] = safe_ratio(rvol - rvol_ema, rvol_ema)

                if "volatility_regime_stability" in active:
                    rvol_long_mean = rvol.rolling(regime_window).mean()
                    regime_high = rvol > (rvol_long_mean * 1.2)
                    regime_low = rvol < (rvol_long_mean * 0.8)
                    regime_label = pd.Series(1, index=frame.index, dtype=int)
                    regime_label.loc[regime_low] = 0
                    regime_label.loc[regime_high] = 2
                    regime_groups = (regime_label != regime_label.shift(1)).cumsum()
                    regime_age = regime_label.groupby(regime_groups).cumcount() + 1
                    output["volatility_regime_stability"] = (
                        regime_age.clip(0, regime_age_window) / float(regime_age_window)
                    )

                if "high_vol_stress_indicator" in active:
                    atr_14 = context.indicator_cache.get_or_create(
                        f"atr_{atr_14_window}",
                        lambda: compute_atr(high, low, close, length=atr_14_window),
                    )
                    atr_48 = context.indicator_cache.get_or_create(
                        f"atr_{atr_48_window}",
                        lambda: compute_atr(high, low, close, length=atr_48_window),
                    )
                    atr_expanding = atr_14 > atr_48 * 1.1
                    vol_high = rvol > rvol.rolling(regime_window).quantile(0.75)
                    output["high_vol_stress_indicator"] = (atr_expanding & vol_high).astype(float)

                if "vol_regime_classification" in active:
                    rvol_33 = rvol.rolling(regime_window).quantile(0.33)
                    rvol_67 = rvol.rolling(regime_window).quantile(0.67)
                    output["vol_regime_classification"] = (
                        (rvol > rvol_33).astype(int) + (rvol > rvol_67).astype(int)
                    )

        return output
