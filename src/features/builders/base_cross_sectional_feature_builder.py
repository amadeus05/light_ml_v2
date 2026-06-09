from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_spec


def _normalize_rank(series: pd.Series) -> pd.Series:
    if len(series) == 1:
        return pd.Series(0.5, index=series.index)
    ranked = series.rank(method="average")
    return (ranked - 1) / (len(series) - 1)


class BaseCrossSectionalFeatureBuilder(FeatureBuilderContract):
    block_name = "cross_sectional"
    FEATURE_SPECS = {
        "cross_sectional_rank_ema_fast_slow_1h": feature_spec(
            "cross_sectional_rank_ema_fast_slow_1h",
            block_name,
            "normalized_rank(ema_fast_slow across symbols at timestamp)",
            description="Cross-sectional rank of the EMA spread signal on each main-timeframe bar.",
            dependencies=("ema_fast_slow",),
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.base_feature_map is None:
            raise ValueError("Cross-sectional feature block requires base_feature_map.")
        if "ema_fast_slow" not in context.frame.columns:
            raise ValueError("Cross-sectional 1H rank requires 'ema_fast_slow'.")

        cache_key = "rank_map:base:ema_fast_slow"
        rank_map = context.shared_cache.get(cache_key)
        if rank_map is None:
            rank_frames = []
            for symbol, source_df in context.base_feature_map.items():
                if "ema_fast_slow" not in source_df.columns:
                    continue
                frame = source_df[["timestamp", "ema_fast_slow"]].dropna(subset=["ema_fast_slow"]).copy()
                if frame.empty:
                    continue
                frame["symbol"] = symbol
                rank_frames.append(frame)

            rank_map = {}
            if rank_frames:
                rank_df = pd.concat(rank_frames, ignore_index=True)
                rank_df["cross_sectional_rank_ema_fast_slow_1h"] = rank_df.groupby("timestamp")["ema_fast_slow"].transform(_normalize_rank)
                for symbol in context.base_feature_map:
                    rank_map[symbol] = rank_df.loc[
                        rank_df["symbol"] == symbol,
                        ["timestamp", "cross_sectional_rank_ema_fast_slow_1h"],
                    ].copy()
            context.shared_cache[cache_key] = rank_map

        symbol_rank = rank_map.get(context.symbol)
        if symbol_rank is None or symbol_rank.empty:
            output["cross_sectional_rank_ema_fast_slow_1h"] = pd.NA
            return output
        return output.merge(symbol_rank, on="timestamp", how="left")
