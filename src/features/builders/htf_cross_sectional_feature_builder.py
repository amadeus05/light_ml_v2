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


class HtfCrossSectionalFeatureBuilder(FeatureBuilderContract):
    block_name = "cross_sectional"
    FEATURE_SPECS = {
        "cross_sectional_rank_4h": feature_spec(
            "cross_sectional_rank_4h",
            block_name,
            "normalized_rank(return_4h_3 across symbols at timestamp)",
            description="Cross-sectional rank of 3-bar HTF return on each HTF bar.",
            dependencies=("return_4h_3",),
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.htf_feature_map is None:
            raise ValueError("HTF cross-sectional feature block requires htf_feature_map.")
        if "return_4h_3" not in context.frame.columns:
            raise ValueError("Cross-sectional 4H rank requires 'return_4h_3'.")

        cache_key = "rank_map:htf:return_4h_3"
        rank_map = context.shared_cache.get(cache_key)
        if rank_map is None:
            rank_frames = []
            for symbol, source_df in context.htf_feature_map.items():
                if "return_4h_3" not in source_df.columns:
                    continue
                frame = source_df[["timestamp", "return_4h_3"]].dropna(subset=["return_4h_3"]).copy()
                if frame.empty:
                    continue
                frame["symbol"] = symbol
                rank_frames.append(frame)

            rank_map = {}
            if rank_frames:
                rank_df = pd.concat(rank_frames, ignore_index=True)
                rank_df["cross_sectional_rank_4h"] = rank_df.groupby("timestamp")["return_4h_3"].transform(_normalize_rank)
                for symbol in context.htf_feature_map:
                    rank_map[symbol] = rank_df.loc[
                        rank_df["symbol"] == symbol,
                        ["timestamp", "cross_sectional_rank_4h"],
                    ].copy()
            context.shared_cache[cache_key] = rank_map

        symbol_rank = rank_map.get(context.symbol)
        if symbol_rank is None or symbol_rank.empty:
            output["cross_sectional_rank_4h"] = pd.NA
            return output
        return output.merge(symbol_rank, on="timestamp", how="left")
