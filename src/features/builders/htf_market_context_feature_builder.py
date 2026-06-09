from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_spec


class HtfMarketContextFeatureBuilder(FeatureBuilderContract):
    block_name = "market_context"
    FEATURE_SPECS = {
        "market_breadth_pos_return_4h_3": feature_spec("market_breadth_pos_return_4h_3", block_name, "mean(1{return_4h_3 > 0} across symbols at timestamp)", description="Share of symbols with positive 3-bar HTF return.", dependencies=("return_4h_3",)),
        "market_dispersion_return_4h_3": feature_spec("market_dispersion_return_4h_3", block_name, "std(return_4h_3 across symbols at timestamp)", description="Cross-sectional dispersion of 3-bar HTF returns.", dependencies=("return_4h_3",)),
        "market_breadth_pos_return_4h_14": feature_spec("market_breadth_pos_return_4h_14", block_name, "mean(1{return_4h_14 > 0} across symbols at timestamp)", description="Share of symbols with positive 14-bar HTF return.", dependencies=("return_4h_14",)),
        "market_mean_return_4h_14": feature_spec("market_mean_return_4h_14", block_name, "mean(return_4h_14 across symbols at timestamp)", description="Cross-sectional mean of 14-bar HTF returns.", dependencies=("return_4h_14",)),
        "market_avg_ema_slope_4h": feature_spec("market_avg_ema_slope_4h", block_name, "mean(ema_slope_4h across symbols at timestamp)", description="Average HTF EMA slope across the market universe.", dependencies=("ema_slope_4h",)),
        "market_abs_avg_ema_slope_4h": feature_spec("market_abs_avg_ema_slope_4h", block_name, "mean(abs(ema_slope_4h) across symbols at timestamp)", description="Average absolute HTF EMA slope across the market universe.", dependencies=("ema_slope_4h",)),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.htf_feature_map is None:
            raise ValueError("HTF market context block requires htf_feature_map.")
        required_dependencies = set()
        if {"market_breadth_pos_return_4h_3", "market_dispersion_return_4h_3"}.intersection(active):
            required_dependencies.add("return_4h_3")
        if {"market_breadth_pos_return_4h_14", "market_mean_return_4h_14"}.intersection(active):
            required_dependencies.add("return_4h_14")
        if {"market_avg_ema_slope_4h", "market_abs_avg_ema_slope_4h"}.intersection(active):
            required_dependencies.add("ema_slope_4h")
        missing_dependencies = required_dependencies - set(context.frame.columns)
        if missing_dependencies:
            raise ValueError(
                "HTF market context requires: " + ", ".join(sorted(missing_dependencies))
            )

        cache_key = "market_context:htf:regime_bundle"
        context_df = context.shared_cache.get(cache_key)
        if context_df is None:
            frames = []
            for source_df in context.htf_feature_map.values():
                available_columns = {"timestamp"}
                for column in ("return_4h_3", "return_4h_14", "ema_slope_4h"):
                    if column in source_df.columns:
                        available_columns.add(column)
                if len(available_columns) == 1:
                    continue
                frame = source_df[list(available_columns)].copy()
                frame = frame.replace([np.inf, -np.inf], np.nan)
                if not frame.empty:
                    frames.append(frame)
            if not frames:
                context_df = pd.DataFrame(
                    columns=[
                        "timestamp",
                        "market_breadth_pos_return_4h_3",
                        "market_dispersion_return_4h_3",
                        "market_breadth_pos_return_4h_14",
                        "market_mean_return_4h_14",
                        "market_avg_ema_slope_4h",
                        "market_abs_avg_ema_slope_4h",
                    ]
                )
            else:
                context_source = pd.concat(frames, ignore_index=True)
                context_df = pd.DataFrame(
                    {"timestamp": pd.Index(sorted(context_source["timestamp"].dropna().unique()), name="timestamp")}
                )
                if "return_4h_3" in context_source.columns:
                    grouped_return_4h_3 = context_source.dropna(subset=["return_4h_3"]).groupby("timestamp")["return_4h_3"]
                    if not grouped_return_4h_3.ngroups == 0:
                        breadth_4h_3 = grouped_return_4h_3.apply(lambda series: float((series > 0).mean())).rename("market_breadth_pos_return_4h_3")
                        dispersion_4h_3 = grouped_return_4h_3.apply(
                            lambda series: float(np.nanstd(series.to_numpy(dtype=float), ddof=0))
                        ).rename("market_dispersion_return_4h_3")
                        context_df = context_df.merge(breadth_4h_3.reset_index(), on="timestamp", how="left")
                        context_df = context_df.merge(dispersion_4h_3.reset_index(), on="timestamp", how="left")
                if "return_4h_14" in context_source.columns:
                    grouped_return_4h_14 = context_source.dropna(subset=["return_4h_14"]).groupby("timestamp")["return_4h_14"]
                    if not grouped_return_4h_14.ngroups == 0:
                        breadth_4h_14 = grouped_return_4h_14.apply(lambda series: float((series > 0).mean())).rename("market_breadth_pos_return_4h_14")
                        mean_4h_14 = grouped_return_4h_14.mean().astype(float).rename("market_mean_return_4h_14")
                        context_df = context_df.merge(breadth_4h_14.reset_index(), on="timestamp", how="left")
                        context_df = context_df.merge(mean_4h_14.reset_index(), on="timestamp", how="left")
                if "ema_slope_4h" in context_source.columns:
                    grouped_ema_slope_4h = context_source.dropna(subset=["ema_slope_4h"]).groupby("timestamp")["ema_slope_4h"]
                    if not grouped_ema_slope_4h.ngroups == 0:
                        avg_ema_slope = grouped_ema_slope_4h.mean().astype(float).rename("market_avg_ema_slope_4h")
                        abs_avg_ema_slope = grouped_ema_slope_4h.apply(
                            lambda series: float(np.nanmean(np.abs(series.to_numpy(dtype=float))))
                        ).rename("market_abs_avg_ema_slope_4h")
                        context_df = context_df.merge(avg_ema_slope.reset_index(), on="timestamp", how="left")
                        context_df = context_df.merge(abs_avg_ema_slope.reset_index(), on="timestamp", how="left")
            context.shared_cache[cache_key] = context_df

        return output.merge(context_df, on="timestamp", how="left")
