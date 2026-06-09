from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_param, feature_spec


class OpenInterestFeatureBuilder(FeatureBuilderContract):
    block_name = "open_interest"
    FEATURE_SPECS = {
        "open_interest_zscore_7d": feature_spec("open_interest_zscore_7d", block_name, "(open_interest - rolling_mean(open_interest, {OPEN_INTEREST_ZSCORE_WINDOW_1H})) / rolling_std(open_interest, {OPEN_INTEREST_ZSCORE_WINDOW_1H})", description="Open interest z-score on a rolling 7-day window.", params=(feature_param("OPEN_INTEREST_ZSCORE_WINDOW_1H", 24 * 7),), inputs=("open_interest",)),
        "open_interest_change_pct_8h": feature_spec("open_interest_change_pct_8h", block_name, "pct_change(open_interest, {OPEN_INTEREST_CHANGE_LOOKBACK_8H})", description="Open interest percentage change over the 8-hour lookback.", params=(feature_param("OPEN_INTEREST_CHANGE_LOOKBACK_8H", 8),), inputs=("open_interest",)),
        "open_interest_change_pct_24h": feature_spec("open_interest_change_pct_24h", block_name, "pct_change(open_interest, {OPEN_INTEREST_CHANGE_LOOKBACK_24H})", description="Open interest percentage change over the 24-hour lookback.", params=(feature_param("OPEN_INTEREST_CHANGE_LOOKBACK_24H", 24),), inputs=("open_interest",)),
        "open_interest_funding_crowding_1h": feature_spec("open_interest_funding_crowding_1h", block_name, "open_interest_change_pct_24h * funding_rate_zscore_7d", description="Crowding proxy from rising open interest and extreme funding.", inputs=("open_interest", "funding_rate"), dependencies=("open_interest_change_pct_24h", "funding_rate_zscore_7d")),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        frame = context.frame
        output = frame[["timestamp"]].copy()
        if not active:
            return output

        if "open_interest" not in frame.columns:
            raise ValueError("Open interest features require raw column 'open_interest'.")

        open_interest = pd.to_numeric(frame["open_interest"], errors="coerce")
        zscore_window = context.bars("7d", minimum=24)
        lookback_8h = context.bars("8h")
        lookback_24h = context.bars("24h")

        rolling_mean = open_interest.rolling(zscore_window).mean()
        rolling_std = open_interest.rolling(zscore_window).std().replace(0, np.nan)
        open_interest_zscore = (open_interest - rolling_mean) / rolling_std

        oi_change_pct_8h = open_interest.pct_change(lookback_8h)
        oi_change_pct_24h = open_interest.pct_change(lookback_24h)

        if "open_interest_zscore_7d" in active:
            output["open_interest_zscore_7d"] = open_interest_zscore
        if "open_interest_change_pct_8h" in active:
            output["open_interest_change_pct_8h"] = oi_change_pct_8h
        if "open_interest_change_pct_24h" in active:
            output["open_interest_change_pct_24h"] = oi_change_pct_24h
        if "open_interest_funding_crowding_1h" in active:
            if "funding_rate" not in frame.columns:
                raise ValueError("Feature 'open_interest_funding_crowding_1h' requires raw column 'funding_rate'.")
            funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
            funding_mean = funding_rate.rolling(zscore_window).mean()
            funding_std = funding_rate.rolling(zscore_window).std().replace(0, np.nan)
            funding_zscore = (funding_rate - funding_mean) / funding_std
            output["open_interest_funding_crowding_1h"] = oi_change_pct_24h * funding_zscore

        return output
