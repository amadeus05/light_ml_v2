from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import safe_ratio
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class InteractionFeatureBuilder(FeatureBuilderContract):
    block_name = "interactions"
    FEATURE_SPECS = {
        "vol_ratio": feature_spec("vol_ratio", block_name, "realized_vol_1h / (realized_vol_4h_returns_20 / sqrt(4))", description="Main timeframe volatility relative to HTF volatility expressed per hour.", dependencies=("realized_vol_1h", "realized_vol_4h_returns_20")),
        "delta_market_breadth_ema_fast_slow_1h": feature_spec("delta_market_breadth_ema_fast_slow_1h", block_name, "diff(market_breadth_ema_fast_slow_1h, 1)", description="One-bar change in market breadth.", dependencies=("market_breadth_ema_fast_slow_1h",)),
        "market_breadth_ema_fast_slow_1h_zscore": feature_spec("market_breadth_ema_fast_slow_1h_zscore", block_name, "(market_breadth_ema_fast_slow_1h - rolling_mean(market_breadth_ema_fast_slow_1h, {MARKET_ZSCORE_WINDOW})) / rolling_std(market_breadth_ema_fast_slow_1h, {MARKET_ZSCORE_WINDOW})", description="Z-score of market breadth over the configured window.", params=(feature_param("MARKET_ZSCORE_WINDOW", 96),), dependencies=("market_breadth_ema_fast_slow_1h",)),
        "ema_fast_slow_x_market_breadth_ema_fast_slow_1h": feature_spec("ema_fast_slow_x_market_breadth_ema_fast_slow_1h", block_name, "ema_fast_slow * market_breadth_ema_fast_slow_1h", description="Signal strength weighted by market breadth.", dependencies=("ema_fast_slow", "market_breadth_ema_fast_slow_1h")),
        "market_directional_pressure_1h": feature_spec("market_directional_pressure_1h", block_name, "(market_breadth_ema_fast_slow_1h - 0.5) * 2", description="Breadth remapped from [0,1] to [-1,1].", dependencies=("market_breadth_ema_fast_slow_1h",)),
        "signal_market_agreement_1h": feature_spec("signal_market_agreement_1h", block_name, "ema_fast_slow * market_directional_pressure_1h", description="Agreement between local signal and market direction.", dependencies=("ema_fast_slow", "market_directional_pressure_1h")),
        "counter_market_penalty_1h": feature_spec("counter_market_penalty_1h", block_name, "abs(ema_fast_slow) * clip(-sign(ema_fast_slow) * market_directional_pressure_1h, lower=0)", description="Penalty for trading against the market pressure.", dependencies=("ema_fast_slow", "market_directional_pressure_1h")),
        "trend_efficiency_24h_x_volatility_regime_change_1h": feature_spec("trend_efficiency_24h_x_volatility_regime_change_1h", block_name, "trend_efficiency_24h * volatility_regime_change_1h", description="Interaction between trend quality and volatility regime shift.", dependencies=("trend_efficiency_24h", "volatility_regime_change_1h")),
        "trend_alignment_1h_4h": feature_spec("trend_alignment_1h_4h", block_name, "ema_fast_slow * ema_slope_4h", description="Agreement between main timeframe and HTF trend signals.", dependencies=("ema_fast_slow", "ema_slope_4h")),
        "breakout_quality_4h_x_volume_ratio_1h": feature_spec("breakout_quality_4h_x_volume_ratio_1h", block_name, "breakout_quality_4h * volume_ratio_1h", description="HTF breakout quality confirmed by elevated main-timeframe volume.", dependencies=("breakout_quality_4h", "volume_ratio_1h")),
        "ema_fast_slow_x_vol_of_vol": feature_spec("ema_fast_slow_x_vol_of_vol", block_name, "ema_fast_slow * vol_of_vol_1h", description="Momentum signal scaled by volatility-of-volatility.", dependencies=("ema_fast_slow", "vol_of_vol_1h")),
        "trend_efficiency_x_vol_stability": feature_spec("trend_efficiency_x_vol_stability", block_name, "trend_efficiency_24h * volatility_regime_stability", description="Trend efficiency weighted by regime persistence.", dependencies=("trend_efficiency_24h", "volatility_regime_stability")),
        "market_pressure_x_vol_regime": feature_spec("market_pressure_x_vol_regime", block_name, "market_directional_pressure_1h * (vol_regime_classification - 1)", description="Market pressure scaled by normalized volatility regime.", dependencies=("market_directional_pressure_1h", "vol_regime_classification")),
        "signal_x_high_vol_stress": feature_spec("signal_x_high_vol_stress", block_name, "ema_fast_slow * high_vol_stress_indicator", description="Signal strength during high-volatility stress states.", dependencies=("ema_fast_slow", "high_vol_stress_indicator")),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        if "vol_ratio" in active:
            if "realized_vol_1h" not in frame.columns or "realized_vol_4h_returns_20" not in frame.columns:
                raise ValueError("Feature 'vol_ratio' requires 'realized_vol_1h' and 'realized_vol_4h_returns_20'.")
            realized_vol_4h_per_hour = frame["realized_vol_4h_returns_20"] / np.sqrt(4.0)
            output["vol_ratio"] = safe_ratio(frame["realized_vol_1h"], realized_vol_4h_per_hour)

        breadth_request = {
            "delta_market_breadth_ema_fast_slow_1h",
            "market_breadth_ema_fast_slow_1h_zscore",
            "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
            "market_directional_pressure_1h",
            "signal_market_agreement_1h",
            "counter_market_penalty_1h",
            "market_pressure_x_vol_regime",
        }
        if breadth_request.intersection(active):
            if "market_breadth_ema_fast_slow_1h" not in frame.columns:
                raise ValueError("Market breadth interaction features require 'market_breadth_ema_fast_slow_1h'.")
            breadth = pd.to_numeric(frame["market_breadth_ema_fast_slow_1h"], errors="coerce")
            market_pressure = (breadth - 0.5) * 2.0
            if "delta_market_breadth_ema_fast_slow_1h" in active:
                output["delta_market_breadth_ema_fast_slow_1h"] = breadth.diff(1)
            if "market_breadth_ema_fast_slow_1h_zscore" in active:
                zscore_window = max(10, int(getattr(cfg, "MARKET_ZSCORE_WINDOW", 96)))
                breadth_mean = breadth.rolling(zscore_window).mean()
                breadth_std = breadth.rolling(zscore_window).std().replace(0, np.nan)
                output["market_breadth_ema_fast_slow_1h_zscore"] = (breadth - breadth_mean) / breadth_std
            if "market_directional_pressure_1h" in active:
                output["market_directional_pressure_1h"] = market_pressure
            if "ema_fast_slow_x_market_breadth_ema_fast_slow_1h" in active:
                if "ema_fast_slow" not in frame.columns:
                    raise ValueError(
                        "Feature 'ema_fast_slow_x_market_breadth_ema_fast_slow_1h' requires 'ema_fast_slow'."
                    )
                output["ema_fast_slow_x_market_breadth_ema_fast_slow_1h"] = frame["ema_fast_slow"] * breadth
            if {"signal_market_agreement_1h", "counter_market_penalty_1h"}.intersection(active):
                if "ema_fast_slow" not in frame.columns:
                    raise ValueError("Breadth agreement features require 'ema_fast_slow'.")
                signal_strength = pd.to_numeric(frame["ema_fast_slow"], errors="coerce")
                signal_direction = np.sign(signal_strength)
                if "signal_market_agreement_1h" in active:
                    output["signal_market_agreement_1h"] = signal_strength * market_pressure
                if "counter_market_penalty_1h" in active:
                    disagreement = (-signal_direction * market_pressure).clip(lower=0)
                    output["counter_market_penalty_1h"] = signal_strength.abs() * disagreement
            if "market_pressure_x_vol_regime" in active:
                if "vol_regime_classification" not in frame.columns:
                    raise ValueError("Feature 'market_pressure_x_vol_regime' requires 'vol_regime_classification'.")
                regime_normalized = (pd.to_numeric(frame["vol_regime_classification"], errors="coerce") - 1.0) / 1.0
                output["market_pressure_x_vol_regime"] = market_pressure * regime_normalized

        if "trend_efficiency_24h_x_volatility_regime_change_1h" in active:
            required = {"trend_efficiency_24h", "volatility_regime_change_1h"}
            if not required.issubset(frame.columns):
                missing = ", ".join(sorted(required - set(frame.columns)))
                raise ValueError("Feature 'trend_efficiency_24h_x_volatility_regime_change_1h' requires: " + missing)
            output["trend_efficiency_24h_x_volatility_regime_change_1h"] = (
                frame["trend_efficiency_24h"] * frame["volatility_regime_change_1h"]
            )

        if "trend_alignment_1h_4h" in active:
            required = {"ema_fast_slow", "ema_slope_4h"}
            if not required.issubset(frame.columns):
                missing = ", ".join(sorted(required - set(frame.columns)))
                raise ValueError("Feature 'trend_alignment_1h_4h' requires: " + missing)
            output["trend_alignment_1h_4h"] = frame["ema_fast_slow"] * frame["ema_slope_4h"]

        if "breakout_quality_4h_x_volume_ratio_1h" in active:
            required = {"breakout_quality_4h", "volume_ratio_1h"}
            if not required.issubset(frame.columns):
                missing = ", ".join(sorted(required - set(frame.columns)))
                raise ValueError("Feature 'breakout_quality_4h_x_volume_ratio_1h' requires: " + missing)
            output["breakout_quality_4h_x_volume_ratio_1h"] = frame["breakout_quality_4h"] * frame["volume_ratio_1h"]

        if "ema_fast_slow_x_vol_of_vol" in active:
            if "ema_fast_slow" in frame.columns and "vol_of_vol_1h" in frame.columns:
                output["ema_fast_slow_x_vol_of_vol"] = frame["ema_fast_slow"] * frame["vol_of_vol_1h"]

        if "trend_efficiency_x_vol_stability" in active:
            if "trend_efficiency_24h" in frame.columns and "volatility_regime_stability" in frame.columns:
                output["trend_efficiency_x_vol_stability"] = (
                    frame["trend_efficiency_24h"] * frame["volatility_regime_stability"]
                )

        if "signal_x_high_vol_stress" in active:
            if "ema_fast_slow" in frame.columns and "high_vol_stress_indicator" in frame.columns:
                output["signal_x_high_vol_stress"] = frame["ema_fast_slow"] * frame["high_vol_stress_indicator"]

        return output
