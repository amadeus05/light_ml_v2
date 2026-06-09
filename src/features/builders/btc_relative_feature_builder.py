from __future__ import annotations

import config as cfg
import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import rolling_matthews_corrcoef_sign_agreement
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import feature_spec


class BtcRelativeFeatureBuilder(FeatureBuilderContract):
    block_name = "btc_relative"
    FEATURE_SPECS = {
        "relative_strength_vs_btc_24h": feature_spec("relative_strength_vs_btc_24h", block_name, "return_1h_24 - btc_return_1h_24", description="Asset 24-bar return relative to BTC over the same timestamps.", inputs=("close",), dependencies=("return_1h_24",)),
        "beta_to_btc_24h": feature_spec("beta_to_btc_24h", block_name, "rolling_cov(log(close / close.shift(1)), log(btc_close / btc_close.shift(1)), 24) / rolling_var(log(btc_close / btc_close.shift(1)), 24)", description="Rolling 24-bar beta of the asset to BTC.", inputs=("close",), dependencies=("return_1h_24",)),
        "residual_return_24h": feature_spec("residual_return_24h", block_name, "return_1h_24 - beta_to_btc_24h * btc_return_1h_24", description="BTC-neutralized 24-bar return.", inputs=("close",), dependencies=("return_1h_24", "beta_to_btc_24h")),
        "mcc_sign_agreement_btc_24h": feature_spec(
            "mcc_sign_agreement_btc_24h",
            block_name,
            "MCC(1{r_asset>0}, 1{r_btc>0}) over MCC_SIGN_BTC_WINDOW 1h log-returns",
            description="Rolling Matthews correlation between up/down of 1h returns (asset vs BTC).",
            inputs=("close",),
            dependencies=("return_1h_24",),
        ),
    }

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        active = self.provides().intersection(requested_features)
        output = context.frame[["timestamp"]].copy()
        if not active:
            return output
        if context.base_feature_map is None:
            raise ValueError("BTC-relative feature block requires base_feature_map.")
        if "return_1h_24" not in context.frame.columns:
            raise ValueError("BTC-relative features require 'return_1h_24'.")

        cache_key = "btc_reference_frame"
        btc_reference = context.shared_cache.get(cache_key)
        if btc_reference is None:
            btc_df = context.base_feature_map.get("BTC/USDT")
            if btc_df is None or btc_df.empty:
                btc_reference = pd.DataFrame(columns=["timestamp", "btc_close", "btc_return_1h_24"])
            else:
                btc_reference = btc_df[["timestamp", "close", "return_1h_24"]].copy().rename(
                    columns={
                        "close": "btc_close",
                        "return_1h_24": "btc_return_1h_24",
                    }
                )
            context.shared_cache[cache_key] = btc_reference

        merged = context.frame[["timestamp", "close", "return_1h_24"]].merge(
            btc_reference,
            on="timestamp",
            how="left",
        )
        if merged["btc_close"].isna().all():
            for feature_name in sorted(active):
                output[feature_name] = np.nan
            return output

        asset_return_1h = np.log(merged["close"] / merged["close"].shift(1))
        btc_return_1h = np.log(merged["btc_close"] / merged["btc_close"].shift(1))
        btc_var_24h = btc_return_1h.rolling(24).var().replace(0, np.nan)
        beta_24h = asset_return_1h.rolling(24).cov(btc_return_1h) / btc_var_24h

        if "relative_strength_vs_btc_24h" in active:
            output["relative_strength_vs_btc_24h"] = merged["return_1h_24"] - merged["btc_return_1h_24"]
        if "beta_to_btc_24h" in active:
            output["beta_to_btc_24h"] = beta_24h
        if "residual_return_24h" in active:
            output["residual_return_24h"] = merged["return_1h_24"] - (beta_24h * merged["btc_return_1h_24"])

        if "mcc_sign_agreement_btc_24h" in active:
            win = max(2, int(getattr(cfg, "MCC_SIGN_BTC_WINDOW", 24)))
            min_override = getattr(cfg, "MCC_SIGN_BTC_MIN_PERIODS", None)
            min_periods = max(3, int(min_override)) if min_override is not None else max(3, win // 2)
            output["mcc_sign_agreement_btc_24h"] = rolling_matthews_corrcoef_sign_agreement(
                asset_return_1h,
                btc_return_1h,
                window=win,
                min_periods=min_periods,
            )

        if context.symbol == "BTC/USDT":
            if "relative_strength_vs_btc_24h" in active:
                output["relative_strength_vs_btc_24h"] = 0.0
            if "beta_to_btc_24h" in active:
                output["beta_to_btc_24h"] = 1.0
            if "residual_return_24h" in active:
                output["residual_return_24h"] = 0.0
            if "mcc_sign_agreement_btc_24h" in active:
                output["mcc_sign_agreement_btc_24h"] = 1.0

        return output
