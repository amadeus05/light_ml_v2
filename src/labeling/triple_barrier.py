from __future__ import annotations

import logging

import config as cfg
import numpy as np
import pandas as pd

from src.features.indicators import compute_atr, safe_ratio
from src.labeling.schema import (
    BARRIER_STOP_PCT_COLUMN,
    BARRIER_TAKE_PCT_COLUMN,
    BINARY_TARGET_VALUES,
    DUAL_TARGET_VALUES,
    EXIT_REASONS,
    LABEL_HORIZON_BARS_COLUMN,
    LABEL_TO_CLASS,
    LABELING_SCHEMA,
    LONG_EXIT_REASON_COLUMN,
    LONG_LABEL_PNL_COLUMN,
    SHORT_EXIT_REASON_COLUMN,
    SHORT_LABEL_PNL_COLUMN,
    TARGET_COLUMN,
    TARGET_LONG_COLUMN,
    TARGET_SHORT_COLUMN,
)
from src.timeframes import duration_to_bars, timeframe_hours

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """Single source of truth for triple-barrier labels and trade exit math."""

    ENTRY_POLICY = "next_bar_open"
    SAME_BAR_POLICY = "sl_first"

    def __init__(self, timeframe: str = "1h") -> None:
        self.timeframe = timeframe

    def metadata(self) -> dict:
        """Describe the exact labeling contract used by ETL, training, and backtests."""
        taker_com = float(getattr(cfg, "TAKER_COM", 0.0004))
        slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
        return {
            "labeling_contract_version": str(
                getattr(cfg, "LABELING_CONTRACT_VERSION", "")
            ),
            "timeframe": self.timeframe,
            "entry_policy": self.ENTRY_POLICY,
            "same_bar_policy": self.SAME_BAR_POLICY,
            "vertical_exit_policy": str(
                getattr(cfg, "VERTICAL_BARRIER_EXIT", "horizon_close")
            ),
            "cost_model": {
                "commission_model": "round_trip_taker",
                "taker_commission_per_side": taker_com,
                "round_trip_commission": taker_com + taker_com,
                "slippage_per_fill": slippage,
                "entry_slippage_applied": True,
                "exit_slippage_applied": True,
            },
            "target_contract": {
                "target_column": LABELING_SCHEMA.target,
                "target_long_column": LABELING_SCHEMA.target_long,
                "target_short_column": LABELING_SCHEMA.target_short,
                "target_output_columns": list(LABELING_SCHEMA.targets),
                "dual_target_values": list(DUAL_TARGET_VALUES),
                "binary_target_values": list(BINARY_TARGET_VALUES),
                "label_to_class": {str(key): value for key, value in LABEL_TO_CLASS.items()},
            },
            "barrier_contract": {
                "barrier_stop_pct_column": LABELING_SCHEMA.barrier_stop_pct,
                "barrier_take_pct_column": LABELING_SCHEMA.barrier_take_pct,
                "label_horizon_bars_column": LABELING_SCHEMA.label_horizon_bars,
                "contract_columns": sorted(LABELING_SCHEMA.barriers),
                "diagnostic_columns": sorted(LABELING_SCHEMA.diagnostics),
                "exit_reasons": list(EXIT_REASONS),
            },
            "barrier_policy": {
                "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
                "fixed_take_pct": float(getattr(cfg, "TP_PCT", 0.0)),
                "fixed_stop_pct": float(getattr(cfg, "SL_PCT", 0.0)),
                "atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
                "rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
                "tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
                "min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
                "max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
            },
            "horizon_policy": {
                "horizon_duration": str(getattr(cfg, "HORIZON_DURATION", "12h")),
                "base_horizon_bars": self.get_base_horizon(),
                "adaptive_horizon_enabled": bool(
                    getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)
                ),
                "adaptive_min_duration": str(
                    getattr(cfg, "ADAPTIVE_HORIZON_MIN_DURATION", "8h")
                ),
                "adaptive_max_duration": str(
                    getattr(cfg, "ADAPTIVE_HORIZON_MAX_DURATION", "20h")
                ),
                "adaptive_vol_low": float(
                    getattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.0)
                ),
                "adaptive_vol_high": float(
                    getattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.0)
                ),
            },
        }

    def validate_contract(self, frame: pd.DataFrame) -> dict:
        """Validate target, barrier, diagnostic, and tail-padding semantics."""
        required_columns = LABELING_SCHEMA.required_columns
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"Labeling frame is missing contract columns: {missing}")

        horizons = self.resolve_effective_horizons(frame)
        max_horizon = int(np.max(horizons)) if len(horizons) else 0

        for column in (LABELING_SCHEMA.barrier_stop_pct, LABELING_SCHEMA.barrier_take_pct):
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{column} must contain finite positive values.")

        target = pd.to_numeric(frame[LABELING_SCHEMA.target], errors="raise").astype(int)
        target_long = pd.to_numeric(frame[LABELING_SCHEMA.target_long], errors="raise").astype(int)
        target_short = pd.to_numeric(frame[LABELING_SCHEMA.target_short], errors="raise").astype(int)
        unknown_target = sorted(set(target.unique()) - set(DUAL_TARGET_VALUES))
        unknown_long = sorted(set(target_long.unique()) - set(BINARY_TARGET_VALUES))
        unknown_short = sorted(set(target_short.unique()) - set(BINARY_TARGET_VALUES))
        if unknown_target:
            raise ValueError(f"{LABELING_SCHEMA.target} contains unexpected values: {unknown_target}")
        if unknown_long:
            raise ValueError(f"{LABELING_SCHEMA.target_long} contains unexpected values: {unknown_long}")
        if unknown_short:
            raise ValueError(f"{LABELING_SCHEMA.target_short} contains unexpected values: {unknown_short}")

        padding_mask = (
            frame[LABELING_SCHEMA.long_label_pnl].isna()
            & frame[LABELING_SCHEMA.short_label_pnl].isna()
            & frame[LABELING_SCHEMA.long_exit_reason].isna()
            & frame[LABELING_SCHEMA.short_exit_reason].isna()
        )
        padded_tail_rows = int(padding_mask.sum())
        if padded_tail_rows:
            first_padding_idx = int(np.flatnonzero(padding_mask.to_numpy())[0])
            if not bool(padding_mask.iloc[first_padding_idx:].all()):
                raise ValueError("NaN label diagnostics must be a contiguous tail only.")
            if padded_tail_rows > max_horizon:
                raise ValueError(
                    f"Padded tail rows ({padded_tail_rows}) exceed max horizon ({max_horizon})."
                )
            padded_targets = frame.loc[
                padding_mask,
                list(LABELING_SCHEMA.targets),
            ].astype(int)
            if not (padded_targets == 0).all().all():
                raise ValueError("Padded tail target columns must all be zero.")

        valid_mask = ~padding_mask
        for column in (LABELING_SCHEMA.long_label_pnl, LABELING_SCHEMA.short_label_pnl):
            values = pd.to_numeric(frame.loc[valid_mask, column], errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all():
                raise ValueError(f"{column} must be finite outside the padded tail.")

        for column in (LABELING_SCHEMA.long_exit_reason, LABELING_SCHEMA.short_exit_reason):
            if frame.loc[valid_mask, column].isna().any():
                raise ValueError(f"{column} must be present outside the padded tail.")
            unknown_reasons = sorted(
                set(frame.loc[valid_mask, column].astype(str).unique()) - set(EXIT_REASONS)
            )
            if unknown_reasons:
                raise ValueError(f"{column} contains unexpected values: {unknown_reasons}")

        long_pnl = pd.to_numeric(frame.loc[valid_mask, LABELING_SCHEMA.long_label_pnl], errors="raise")
        short_pnl = pd.to_numeric(frame.loc[valid_mask, LABELING_SCHEMA.short_label_pnl], errors="raise")
        if not (
            target_long.loc[valid_mask].to_numpy()
            == long_pnl.gt(0).astype(int).to_numpy()
        ).all():
            raise ValueError(
                f"{LABELING_SCHEMA.target_long} is inconsistent with {LABELING_SCHEMA.long_label_pnl}."
            )
        if not (
            target_short.loc[valid_mask].to_numpy()
            == short_pnl.gt(0).astype(int).to_numpy()
        ).all():
            raise ValueError(
                f"{LABELING_SCHEMA.target_short} is inconsistent with {LABELING_SCHEMA.short_label_pnl}."
            )

        expected_target = pd.Series(0, index=long_pnl.index, dtype=int)
        expected_target.loc[(long_pnl > 0) & (short_pnl <= 0)] = 1
        expected_target.loc[(short_pnl > 0) & (long_pnl <= 0)] = -1
        if not (target.loc[valid_mask].to_numpy() == expected_target.to_numpy()).all():
            raise ValueError(f"{LABELING_SCHEMA.target} is inconsistent with long/short label PnL.")

        return {
            "rows": int(len(frame)),
            "validated_rows": int(valid_mask.sum()),
            "padded_tail_rows": padded_tail_rows,
            "max_horizon_bars": max_horizon,
            "contract_version": str(getattr(cfg, "LABELING_CONTRACT_VERSION", "")),
        }

    def get_base_horizon(self) -> int:
        duration = str(
            getattr(cfg, "HORIZON_DURATION", f"{int(getattr(cfg, 'HORIZON', 16))}h")
        )
        return duration_to_bars(duration, self.timeframe)

    def compute_effective_horizons(self, df: pd.DataFrame) -> np.ndarray:
        base_horizon = max(1, self.get_base_horizon())
        if not bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)):
            return np.full(len(df), base_horizon, dtype=np.int32)

        if "realized_vol_1h" not in df.columns:
            logger.warning(
                "Adaptive horizon enabled, but 'realized_vol_1h' is missing. "
                "Falling back to fixed horizon=%s.",
                base_horizon,
            )
            return np.full(len(df), base_horizon, dtype=np.int32)

        min_horizon = duration_to_bars(
            str(getattr(cfg, "ADAPTIVE_HORIZON_MIN_DURATION", "8h")),
            self.timeframe,
        )
        max_horizon = duration_to_bars(
            str(getattr(cfg, "ADAPTIVE_HORIZON_MAX_DURATION", "20h")),
            self.timeframe,
        )
        if min_horizon > max_horizon:
            min_horizon, max_horizon = max_horizon, min_horizon
        min_horizon = max(1, min_horizon)
        max_horizon = max(min_horizon, max_horizon)

        vol_low = float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.005))
        vol_high = float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.025))
        if not np.isfinite(vol_low) or not np.isfinite(vol_high) or vol_high <= vol_low:
            logger.warning(
                "Invalid adaptive horizon volatility bounds (low=%s, high=%s). "
                "Falling back to fixed horizon=%s.",
                vol_low,
                vol_high,
                base_horizon,
            )
            return np.full(len(df), base_horizon, dtype=np.int32)

        vol = pd.Series(df["realized_vol_1h"], copy=False).astype(float).abs()
        normalized = ((vol - vol_low) / (vol_high - vol_low)).clip(
            lower=0.0,
            upper=1.0,
        )
        normalized_values = normalized.to_numpy()
        adaptive_raw = np.rint(max_horizon - normalized_values * (max_horizon - min_horizon))
        adaptive = np.full(len(df), base_horizon, dtype=np.int32)
        valid_mask = np.isfinite(adaptive_raw)
        adaptive[valid_mask] = np.clip(
            adaptive_raw[valid_mask],
            min_horizon,
            max_horizon,
        ).astype(np.int32)
        return adaptive

    def resolve_effective_horizons(self, df: pd.DataFrame) -> np.ndarray:
        if LABEL_HORIZON_BARS_COLUMN not in df.columns:
            return self.compute_effective_horizons(df)

        values = pd.to_numeric(df[LABEL_HORIZON_BARS_COLUMN], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{LABEL_HORIZON_BARS_COLUMN} contains non-finite values.")
        rounded = np.rint(values)
        if not np.array_equal(values, rounded) or np.any(rounded < 1):
            invalid = sorted(set(values[(values != rounded) | (rounded < 1)].tolist()))
            raise ValueError(
                f"{LABEL_HORIZON_BARS_COLUMN} must contain positive integer values; found: {invalid}"
            )
        return rounded.astype(np.int32)

    def compute_dynamic_barrier_stop_pct(
        self,
        close: pd.Series,
        atr_14: pd.Series,
        realized_vol_1h: pd.Series,
        effective_horizons: np.ndarray | None = None,
    ) -> pd.Series:
        atr_pct = safe_ratio(atr_14, close).abs()
        if effective_horizons is None:
            horizon_sqrt = np.sqrt(
                float(self.get_base_horizon()) * timeframe_hours(self.timeframe)
            )
        else:
            horizon_sqrt = np.sqrt(
                np.maximum(effective_horizons.astype(float), 1.0)
                * timeframe_hours(self.timeframe)
            )
        horizon_vol_pct = realized_vol_1h.abs() * horizon_sqrt

        stop_pct = pd.concat(
            [
                atr_pct * float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25)),
                horizon_vol_pct * float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.75)),
            ],
            axis=1,
        ).max(axis=1)

        min_pct = float(getattr(cfg, "BARRIER_MIN_PCT", getattr(cfg, "SL_PCT", 0.015)))
        max_pct = float(getattr(cfg, "BARRIER_MAX_PCT", getattr(cfg, "TP_PCT", 0.03)))
        return stop_pct.clip(lower=min_pct, upper=max_pct)

    @staticmethod
    def compute_dynamic_barrier_take_pct(stop_pct: pd.Series) -> pd.Series:
        return stop_pct * float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0))

    def attach_barrier_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        close = output["close"]
        atr_window = duration_to_bars("14h", self.timeframe, minimum=2)
        atr_14 = compute_atr(output["high"], output["low"], close, length=atr_window)
        effective_horizons = self.compute_effective_horizons(output)
        output[LABEL_HORIZON_BARS_COLUMN] = effective_horizons

        if bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", True)):
            if "realized_vol_1h" not in output.columns:
                raise ValueError("Dynamic barriers require feature 'realized_vol_1h' to be enabled.")
            output[BARRIER_STOP_PCT_COLUMN] = self.compute_dynamic_barrier_stop_pct(
                close,
                atr_14,
                output["realized_vol_1h"],
                effective_horizons=effective_horizons,
            )
            output[BARRIER_TAKE_PCT_COLUMN] = self.compute_dynamic_barrier_take_pct(
                output[BARRIER_STOP_PCT_COLUMN]
            )
        else:
            output[BARRIER_STOP_PCT_COLUMN] = float(getattr(cfg, "SL_PCT", 0.015))
            output[BARRIER_TAKE_PCT_COLUMN] = float(getattr(cfg, "TP_PCT", 0.03))
        return output

    @staticmethod
    def compute_clean_pnl(direction: int, entry_price: float, exit_price: float) -> float:
        if direction == 1:
            raw_pnl = (exit_price - entry_price) / entry_price
        else:
            raw_pnl = (entry_price - exit_price) / entry_price
        taker_com = float(getattr(cfg, "TAKER_COM", 0.0004))
        return raw_pnl - (taker_com + taker_com)

    @staticmethod
    def resolve_trade_exit(
        direction: int,
        entry_price: float,
        next_open: float,
        next_high: float,
        next_low: float,
        stop_pct: float,
        take_pct: float,
    ) -> tuple[float | None, str | None]:
        slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
        if direction == 1:
            stop_price = entry_price * (1 - stop_pct)
            take_price = entry_price * (1 + take_pct)

            if next_low <= stop_price:
                exit_price = (next_open if next_open < stop_price else stop_price) * (
                    1 - slippage
                )
                return exit_price, "SL"
            if next_high >= take_price:
                exit_price = take_price * (1 - slippage)
                return exit_price, "TP"
        else:
            stop_price = entry_price * (1 + stop_pct)
            take_price = entry_price * (1 - take_pct)

            if next_high >= stop_price:
                exit_price = (next_open if next_open > stop_price else stop_price) * (
                    1 + slippage
                )
                return exit_price, "SL"
            if next_low <= take_price:
                exit_price = take_price * (1 + slippage)
                return exit_price, "TP"

        return None, None

    @staticmethod
    def resolve_vertical_barrier_exit(direction: int, close_price: float) -> tuple[float, str]:
        slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
        exit_price = close_price * (1 - slippage) if direction == 1 else close_price * (1 + slippage)
        return exit_price, "TIMEOUT"

    @classmethod
    def simulate_trade_outcome(
        cls,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        stop_pcts: np.ndarray,
        take_pcts: np.ndarray,
        start_idx: int,
        direction: int,
        horizon: int,
    ) -> tuple[float, str | None]:
        slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
        base_open = opens[start_idx + 1]
        entry_price = base_open * (1 + slippage) if direction == 1 else base_open * (1 - slippage)
        stop_pct = stop_pcts[start_idx]
        take_pct = take_pcts[start_idx]

        if np.isnan(stop_pct) or np.isnan(take_pct):
            return 0.0, None

        for j in range(1, horizon + 1):
            candle_idx = start_idx + j
            if candle_idx >= len(opens):
                break

            exit_price, reason = cls.resolve_trade_exit(
                direction,
                entry_price,
                opens[candle_idx],
                highs[candle_idx],
                lows[candle_idx],
                stop_pct,
                take_pct,
            )
            if exit_price is not None:
                return cls.compute_clean_pnl(direction, entry_price, exit_price), reason

        vertical_barrier_idx = min(start_idx + horizon, len(closes) - 1)
        vertical_close = closes[vertical_barrier_idx]
        if not np.isfinite(vertical_close):
            return 0.0, None
        exit_price, reason = cls.resolve_vertical_barrier_exit(direction, vertical_close)
        return cls.compute_clean_pnl(direction, entry_price, exit_price), reason

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        labels = []
        long_labels = []
        short_labels = []
        long_pnls = []
        short_pnls = []
        long_exit_reasons = []
        short_exit_reasons = []
        effective_horizons = self.resolve_effective_horizons(df)
        max_horizon = (
            int(np.max(effective_horizons))
            if len(effective_horizons) > 0
            else self.get_base_horizon()
        )

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        stop_pcts = df[BARRIER_STOP_PCT_COLUMN].values
        take_pcts = df[BARRIER_TAKE_PCT_COLUMN].values

        for i in range(len(df) - max_horizon):
            label = 0
            horizon = (
                int(effective_horizons[i])
                if i < len(effective_horizons)
                else self.get_base_horizon()
            )
            long_pnl, long_reason = self.simulate_trade_outcome(
                opens,
                highs,
                lows,
                closes,
                stop_pcts,
                take_pcts,
                i,
                direction=1,
                horizon=horizon,
            )
            short_pnl, short_reason = self.simulate_trade_outcome(
                opens,
                highs,
                lows,
                closes,
                stop_pcts,
                take_pcts,
                i,
                direction=-1,
                horizon=horizon,
            )
            long_labels.append(int(long_pnl > 0))
            short_labels.append(int(short_pnl > 0))
            long_pnls.append(float(long_pnl))
            short_pnls.append(float(short_pnl))
            long_exit_reasons.append(long_reason)
            short_exit_reasons.append(short_reason)

            if long_pnl > 0 and short_pnl <= 0:
                label = 1
            elif short_pnl > 0 and long_pnl <= 0:
                label = -1

            labels.append(label)

        labels.extend([0] * max_horizon)
        long_labels.extend([0] * max_horizon)
        short_labels.extend([0] * max_horizon)
        long_pnls.extend([np.nan] * max_horizon)
        short_pnls.extend([np.nan] * max_horizon)
        long_exit_reasons.extend([None] * max_horizon)
        short_exit_reasons.extend([None] * max_horizon)

        output = df.copy()
        output[TARGET_COLUMN] = labels
        output[TARGET_LONG_COLUMN] = long_labels
        output[TARGET_SHORT_COLUMN] = short_labels
        output[LONG_LABEL_PNL_COLUMN] = long_pnls
        output[SHORT_LABEL_PNL_COLUMN] = short_pnls
        output[LONG_EXIT_REASON_COLUMN] = long_exit_reasons
        output[SHORT_EXIT_REASON_COLUMN] = short_exit_reasons
        return output
