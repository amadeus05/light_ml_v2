from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_spec


class BaseMarketContextFeatureBuilder(FeatureBuilderContract):
    block_name = "market_context"
    FEATURE_SPECS = {
        "market_breadth_ema_fast_slow_1h": feature_spec(
            "market_breadth_ema_fast_slow_1h",
            block_name,
            "mean(1{ema_fast_slow > 0} across symbols at timestamp)",
            description="Share of symbols with positive EMA spread on the main timeframe.",
            dependencies=("ema_fast_slow",),
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.base_feature_map is None:
            raise ValueError("Market context block requires base_feature_map.")
        if "ema_fast_slow" not in context.frame.columns:
            raise ValueError("Market breadth requires 'ema_fast_slow'.")

        cache_key = "market_context:base:ema_fast_slow"
        context_df = context.shared_cache.get(cache_key)
        if context_df is None:
            frames = []
            for source_df in context.base_feature_map.values():
                if "ema_fast_slow" not in source_df.columns:
                    continue
                frame = source_df[["timestamp", "ema_fast_slow"]].dropna(subset=["ema_fast_slow"]).copy()
                if not frame.empty:
                    frames.append(frame)
            if not frames:
                context_df = pd.DataFrame(columns=["timestamp", "market_breadth_ema_fast_slow_1h"])
            else:
                context_source = pd.concat(frames, ignore_index=True)
                grouped = context_source.groupby("timestamp")["ema_fast_slow"]
                context_df = pd.DataFrame({"timestamp": grouped.size().index})
                context_df["market_breadth_ema_fast_slow_1h"] = grouped.apply(lambda series: float((series > 0).mean())).values
            context.shared_cache[cache_key] = context_df

        return output.merge(context_df, on="timestamp", how="left")
