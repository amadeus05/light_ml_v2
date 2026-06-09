from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class FundingFeatureBuilder(FeatureBuilderContract):
    block_name = "funding"
    FEATURE_SPECS = {
        "funding_rate_8h": feature_spec("funding_rate_8h", block_name, "funding_rate", description="Raw funding rate aligned to the main timeframe.", inputs=("funding_rate",)),
        "funding_rate_zscore_7d": feature_spec("funding_rate_zscore_7d", block_name, "(funding_rate - rolling_mean(funding_rate, {FUNDING_ZSCORE_WINDOW_1H})) / rolling_std(funding_rate, {FUNDING_ZSCORE_WINDOW_1H})", description="Funding rate z-score on a rolling 7-day window.", params=(feature_param("FUNDING_ZSCORE_WINDOW_1H", 24 * 7),), inputs=("funding_rate",)),
        "funding_rate_change_24h": feature_spec("funding_rate_change_24h", block_name, "funding_rate - funding_rate.shift({FUNDING_CHANGE_LOOKBACK_1H})", description="Funding rate change over the configured lookback.", params=(feature_param("FUNDING_CHANGE_LOOKBACK_1H", 24),), inputs=("funding_rate",)),
        "longs_overheated_1h": feature_spec("longs_overheated_1h", block_name, "clip(funding_rate_zscore_7d, lower=0)", description="Positive funding z-score only.", dependencies=("funding_rate_zscore_7d",)),
        "shorts_overheated_1h": feature_spec("shorts_overheated_1h", block_name, "clip(-funding_rate_zscore_7d, lower=0)", description="Negative funding z-score only, sign-flipped.", dependencies=("funding_rate_zscore_7d",)),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        if "funding_rate" not in frame.columns:
            raise ValueError("Funding features require raw column 'funding_rate'.")

        funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
        zscore_window = context.bars("7d", minimum=24)
        change_lookback = context.bars("24h")

        funding_zscore = None
        if {"funding_rate_zscore_7d", "longs_overheated_1h", "shorts_overheated_1h"}.intersection(active):
            rolling_mean = funding_rate.rolling(zscore_window).mean()
            rolling_std = funding_rate.rolling(zscore_window).std().replace(0, np.nan)
            funding_zscore = (funding_rate - rolling_mean) / rolling_std

        if "funding_rate_8h" in active:
            output["funding_rate_8h"] = funding_rate
        if "funding_rate_zscore_7d" in active and funding_zscore is not None:
            output["funding_rate_zscore_7d"] = funding_zscore
        if "funding_rate_change_24h" in active:
            output["funding_rate_change_24h"] = funding_rate - funding_rate.shift(change_lookback)
        if "longs_overheated_1h" in active and funding_zscore is not None:
            output["longs_overheated_1h"] = funding_zscore.clip(lower=0)
        if "shorts_overheated_1h" in active and funding_zscore is not None:
            output["shorts_overheated_1h"] = (-funding_zscore).clip(lower=0)

        return output
