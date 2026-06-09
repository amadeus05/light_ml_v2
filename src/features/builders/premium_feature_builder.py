from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class PremiumFeatureBuilder(FeatureBuilderContract):
    block_name = "premium_index"
    FEATURE_SPECS = {
        "premium_index_1h": feature_spec("premium_index_1h", block_name, "premium_index_close", description="Raw premium index aligned to the main timeframe.", inputs=("premium_index_close",)),
        "premium_index_zscore_7d": feature_spec("premium_index_zscore_7d", block_name, "(premium_index_close - rolling_mean(premium_index_close, {PREMIUM_INDEX_ZSCORE_WINDOW_1H})) / rolling_std(premium_index_close, {PREMIUM_INDEX_ZSCORE_WINDOW_1H})", description="Premium index z-score on a rolling 7-day window.", params=(feature_param("PREMIUM_INDEX_ZSCORE_WINDOW_1H", 24 * 7),), inputs=("premium_index_close",)),
        "premium_index_change_24h": feature_spec("premium_index_change_24h", block_name, "premium_index_close - premium_index_close.shift({PREMIUM_INDEX_CHANGE_LOOKBACK_1H})", description="Premium index change over the configured lookback.", params=(feature_param("PREMIUM_INDEX_CHANGE_LOOKBACK_1H", 24),), inputs=("premium_index_close",)),
        "crowded_longs_score_1h": feature_spec("crowded_longs_score_1h", block_name, "clip(premium_index_zscore_7d, lower=0) * clip(funding_rate_zscore_7d, lower=0)", description="Joint long crowding score from premium and funding extremes.", inputs=("premium_index_close", "funding_rate"), dependencies=("premium_index_zscore_7d", "funding_rate_zscore_7d")),
        "crowded_shorts_score_1h": feature_spec("crowded_shorts_score_1h", block_name, "clip(-premium_index_zscore_7d, lower=0) * clip(-funding_rate_zscore_7d, lower=0)", description="Joint short crowding score from premium and funding extremes.", inputs=("premium_index_close", "funding_rate"), dependencies=("premium_index_zscore_7d", "funding_rate_zscore_7d")),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        if "premium_index_close" not in frame.columns:
            raise ValueError("Premium features require raw column 'premium_index_close'.")

        premium_index = pd.to_numeric(frame["premium_index_close"], errors="coerce")
        zscore_window = context.bars("7d", minimum=24)
        change_lookback = context.bars("24h")

        premium_zscore = None
        if {"premium_index_zscore_7d", "crowded_longs_score_1h", "crowded_shorts_score_1h"}.intersection(active):
            rolling_mean = premium_index.rolling(zscore_window).mean()
            rolling_std = premium_index.rolling(zscore_window).std().replace(0, np.nan)
            premium_zscore = (premium_index - rolling_mean) / rolling_std

        if "premium_index_1h" in active:
            output["premium_index_1h"] = premium_index
        if "premium_index_zscore_7d" in active and premium_zscore is not None:
            output["premium_index_zscore_7d"] = premium_zscore
        if "premium_index_change_24h" in active:
            output["premium_index_change_24h"] = premium_index - premium_index.shift(change_lookback)

        crowding_request = {"crowded_longs_score_1h", "crowded_shorts_score_1h"}.intersection(active)
        if crowding_request:
            if "funding_rate" not in frame.columns:
                raise ValueError("Crowding premium features require raw column 'funding_rate'.")
            funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
            funding_mean = funding_rate.rolling(zscore_window).mean()
            funding_std = funding_rate.rolling(zscore_window).std().replace(0, np.nan)
            funding_zscore = (funding_rate - funding_mean) / funding_std

            if "crowded_longs_score_1h" in active and premium_zscore is not None:
                output["crowded_longs_score_1h"] = premium_zscore.clip(lower=0) * funding_zscore.clip(lower=0)
            if "crowded_shorts_score_1h" in active and premium_zscore is not None:
                output["crowded_shorts_score_1h"] = (-premium_zscore).clip(lower=0) * (-funding_zscore).clip(lower=0)

        return output
