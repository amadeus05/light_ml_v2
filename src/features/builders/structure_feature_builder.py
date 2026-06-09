from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import compute_atr, compute_rolling_vwap, compute_trend_efficiency, safe_ratio
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_spec


class StructureFeatureBuilder(FeatureBuilderContract):
    block_name = "structure"
    FEATURE_SPECS = {
        "price_position_1h": feature_spec("price_position_1h", block_name, "(close - rolling_min(low, 24)) / (rolling_max(high, 24) - rolling_min(low, 24))", description="Normalized position inside the recent 24-bar range.", inputs=("close", "high", "low")),
        "distance_to_support_1h": feature_spec("distance_to_support_1h", block_name, "(close - rolling_min(low, 24)) / ATR(high, low, close, 14)", description="Distance to the recent support level in ATR units.", inputs=("close", "high", "low")),
        "distance_to_resistance_1h": feature_spec("distance_to_resistance_1h", block_name, "(rolling_max(high, 24) - close) / ATR(high, low, close, 14)", description="Distance to the recent resistance level in ATR units.", inputs=("close", "high", "low")),
        "distance_to_session_high_1h": feature_spec("distance_to_session_high_1h", block_name, "(session_cummax(high) - close) / ATR(high, low, close, 14)", description="Distance from close to the running daily session high.", inputs=("timestamp", "close", "high", "low")),
        "distance_to_session_low_1h": feature_spec("distance_to_session_low_1h", block_name, "(close - session_cummin(low)) / ATR(high, low, close, 14)", description="Distance from close to the running daily session low.", inputs=("timestamp", "close", "high", "low")),
        "range_position_1h_48": feature_spec("range_position_1h_48", block_name, "(close - rolling_min(low, 48)) / (rolling_max(high, 48) - rolling_min(low, 48))", description="Normalized position inside the 48-bar flat range.", inputs=("close", "high", "low")),
        "range_width_atr_1h_48": feature_spec("range_width_atr_1h_48", block_name, "(rolling_max(high, 48) - rolling_min(low, 48)) / ATR(high, low, close, 14)", description="48-bar range width in ATR units.", inputs=("close", "high", "low")),
        "range_center_distance_atr_1h_48": feature_spec("range_center_distance_atr_1h_48", block_name, "(close - ((rolling_max(high, 48) + rolling_min(low, 48)) / 2)) / ATR(high, low, close, 14)", description="Distance from close to the center of the 48-bar range.", inputs=("close", "high", "low")),
        "flat_efficiency_1h_24": feature_spec("flat_efficiency_1h_24", block_name, "1 - clip(trend_efficiency(close, 24), 0, 1)", description="Inverse of the 24-bar trend efficiency.", inputs=("close",)),
        "mean_reversion_pressure_1h": feature_spec("mean_reversion_pressure_1h", block_name, "-clip(((range_position_1h_48 - 0.5) * 2), -1, 1)", description="Signed pull back toward the center of the 48-bar range.", dependencies=("range_position_1h_48",)),
        "zscore_vs_vwap_1h": feature_spec("zscore_vs_vwap_1h", block_name, "(close - rolling_vwap(close, high, low, volume, 24)) / rolling_std(close - rolling_vwap(close, high, low, volume, 24), 24)", description="Main timeframe VWAP distance z-score.", inputs=("close", "high", "low", "volume")),
        "bollinger_percent_b_1h_20": feature_spec("bollinger_percent_b_1h_20", block_name, "(close - (rolling_mean(close, 20) - 2 * rolling_std(close, 20))) / ((rolling_mean(close, 20) + 2 * rolling_std(close, 20)) - (rolling_mean(close, 20) - 2 * rolling_std(close, 20)))", description="Bollinger percent-b on a 20-bar window.", inputs=("close",)),
        "bollinger_bandwidth_atr_1h_20": feature_spec("bollinger_bandwidth_atr_1h_20", block_name, "((rolling_mean(close, 20) + 2 * rolling_std(close, 20)) - (rolling_mean(close, 20) - 2 * rolling_std(close, 20))) / ATR(high, low, close, 14)", description="Bollinger band width in ATR units.", inputs=("close", "high", "low")),
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
        volume = frame["volume"]

        if {"price_position_1h", "distance_to_support_1h", "distance_to_resistance_1h"}.intersection(active):
            rolling_low = low.rolling(24).min()
            rolling_high = high.rolling(24).max()
            if "price_position_1h" in active:
                output["price_position_1h"] = safe_ratio(close - rolling_low, rolling_high - rolling_low)

            if {"distance_to_support_1h", "distance_to_resistance_1h"}.intersection(active):
                atr_14 = context.indicator_cache.get_or_create(
                    "atr_14",
                    lambda: compute_atr(high, low, close, length=14),
                )
                if "distance_to_support_1h" in active:
                    output["distance_to_support_1h"] = safe_ratio(close - rolling_low, atr_14)
                if "distance_to_resistance_1h" in active:
                    output["distance_to_resistance_1h"] = safe_ratio(rolling_high - close, atr_14)

        if {"distance_to_session_high_1h", "distance_to_session_low_1h"}.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )
            session_key = frame["timestamp"].dt.floor("D")
            session_high = high.groupby(session_key).cummax()
            session_low = low.groupby(session_key).cummin()
            if "distance_to_session_high_1h" in active:
                output["distance_to_session_high_1h"] = safe_ratio(session_high - close, atr_14)
            if "distance_to_session_low_1h" in active:
                output["distance_to_session_low_1h"] = safe_ratio(close - session_low, atr_14)

        flat_request = {
            "range_position_1h_48",
            "range_width_atr_1h_48",
            "range_center_distance_atr_1h_48",
            "flat_efficiency_1h_24",
            "mean_reversion_pressure_1h",
            "zscore_vs_vwap_1h",
            "bollinger_percent_b_1h_20",
            "bollinger_bandwidth_atr_1h_20",
        }
        if flat_request.intersection(active):
            atr_14 = context.indicator_cache.get_or_create(
                "atr_14",
                lambda: compute_atr(high, low, close, length=14),
            )

            if {
                "range_position_1h_48",
                "range_width_atr_1h_48",
                "range_center_distance_atr_1h_48",
                "mean_reversion_pressure_1h",
            }.intersection(active):
                range_low_48 = low.rolling(48).min()
                range_high_48 = high.rolling(48).max()
                range_width_48 = range_high_48 - range_low_48
                range_position_48 = safe_ratio(close - range_low_48, range_width_48)

                if "range_position_1h_48" in active:
                    output["range_position_1h_48"] = range_position_48
                if "range_width_atr_1h_48" in active:
                    output["range_width_atr_1h_48"] = safe_ratio(range_width_48, atr_14)
                if "range_center_distance_atr_1h_48" in active:
                    range_center_48 = (range_high_48 + range_low_48) / 2.0
                    output["range_center_distance_atr_1h_48"] = safe_ratio(close - range_center_48, atr_14)
                if "mean_reversion_pressure_1h" in active:
                    distance_from_center = (range_position_48 - 0.5) * 2.0
                    output["mean_reversion_pressure_1h"] = -distance_from_center.clip(-1.0, 1.0)

            if "flat_efficiency_1h_24" in active:
                trend_efficiency = compute_trend_efficiency(close, 24)
                output["flat_efficiency_1h_24"] = 1.0 - trend_efficiency.clip(0.0, 1.0)

            if "zscore_vs_vwap_1h" in active:
                rolling_vwap = compute_rolling_vwap(close, high, low, volume, 24)
                vwap_distance = close - rolling_vwap
                vwap_distance_std = vwap_distance.rolling(24).std().replace(0, np.nan)
                output["zscore_vs_vwap_1h"] = safe_ratio(vwap_distance, vwap_distance_std)

            if {"bollinger_percent_b_1h_20", "bollinger_bandwidth_atr_1h_20"}.intersection(active):
                rolling_mean_20 = close.rolling(20).mean()
                rolling_std_20 = close.rolling(20).std()
                upper_band = rolling_mean_20 + 2.0 * rolling_std_20
                lower_band = rolling_mean_20 - 2.0 * rolling_std_20
                band_width = upper_band - lower_band
                if "bollinger_percent_b_1h_20" in active:
                    output["bollinger_percent_b_1h_20"] = safe_ratio(close - lower_band, band_width)
                if "bollinger_bandwidth_atr_1h_20" in active:
                    output["bollinger_bandwidth_atr_1h_20"] = safe_ratio(band_width, atr_14)

        return output
