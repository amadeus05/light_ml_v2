import argparse
import logging
import re
from datetime import datetime, timezone

import config as cfg
import numpy as np
import pandas as pd

from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_service import BinanceService
from src.exchanges.bybit.bybit_service import BybitService
from src.features import MasterFeatureBuilder
from src.labeling import (
    LABELING_SCHEMA,
    TripleBarrierLabeler,
)
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository
from src.timeframes import duration_to_bars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build timeframe-specific feature datasets.")
    parser.add_argument(
        "--timeframe-profile",
        choices=sorted(cfg.TIMEFRAME_PROFILES),
        default=cfg.ACTIVE_TIMEFRAME_PROFILE,
    )
    return parser.parse_args()


def get_base_horizon(timeframe: str = "1h") -> int:
    return TripleBarrierLabeler(timeframe=timeframe).get_base_horizon()


def compute_effective_horizons(df: pd.DataFrame, timeframe: str = "1h") -> np.ndarray:
    return TripleBarrierLabeler(timeframe=timeframe).compute_effective_horizons(df)


def resolve_effective_horizons(df: pd.DataFrame, timeframe: str = "1h") -> np.ndarray:
    return TripleBarrierLabeler(timeframe=timeframe).resolve_effective_horizons(df)


def compute_dynamic_barrier_stop_pct(
    close: pd.Series,
    atr_14: pd.Series,
    realized_vol_1h: pd.Series,
    effective_horizons: np.ndarray | None = None,
    timeframe: str = "1h",
) -> pd.Series:
    return TripleBarrierLabeler(timeframe=timeframe).compute_dynamic_barrier_stop_pct(
        close,
        atr_14,
        realized_vol_1h,
        effective_horizons=effective_horizons,
    )


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series) -> pd.Series:
    return TripleBarrierLabeler.compute_dynamic_barrier_take_pct(stop_pct)


def attach_barrier_columns(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    return TripleBarrierLabeler(timeframe=timeframe).attach_barrier_columns(df)


def compute_clean_pnl(direction: int, entry_price: float, exit_price: float) -> float:
    return TripleBarrierLabeler.compute_clean_pnl(direction, entry_price, exit_price)


def resolve_trade_exit(
    direction: int,
    entry_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_pct: float,
    take_pct: float,
) -> tuple[float | None, str | None]:
    return TripleBarrierLabeler.resolve_trade_exit(
        direction,
        entry_price,
        next_open,
        next_high,
        next_low,
        stop_pct,
        take_pct,
    )


def simulate_trade_outcome(
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
    return TripleBarrierLabeler.simulate_trade_outcome(
        opens,
        highs,
        lows,
        closes,
        stop_pcts,
        take_pcts,
        start_idx,
        direction,
        horizon,
    )


def triple_barrier_labeling(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    return TripleBarrierLabeler(timeframe=timeframe).label(df)


def finalize_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    timeframe: str = "1h",
) -> pd.DataFrame:
    output = df.copy()
    input_rows = len(output)
    effective_horizons = resolve_effective_horizons(output, timeframe)
    max_horizon = int(np.max(effective_horizons)) if len(effective_horizons) > 0 else get_base_horizon(timeframe)
    horizon_dropped_rows = min(max_horizon, len(output)) if max_horizon > 0 else 0
    if max_horizon > 0:
        if len(output) <= max_horizon:
            empty_columns = BASE_OUTPUT_COLUMNS + feature_columns + LABELING_SCHEMA.output_columns + list(LABELING_SCHEMA.targets)
            logger.warning(
                "Feature finalize removed all rows: input_rows=%s max_horizon=%s output_rows=0",
                input_rows,
                max_horizon,
            )
            return output.iloc[0:0][empty_columns].copy()
        output = output.iloc[:-max_horizon].copy()

        output_columns = BASE_OUTPUT_COLUMNS + feature_columns + LABELING_SCHEMA.output_columns + list(LABELING_SCHEMA.targets)
    for column in output_columns:
        if column not in output.columns:
            output[column] = np.nan

    output = output[output_columns].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before_dropna = len(output)
    missing_counts = output.isna().sum()
    rows_with_missing = int(output.isna().any(axis=1).sum())
    if rows_with_missing > 0:
        top_missing_columns = {
            column: int(count)
            for column, count in missing_counts[missing_counts > 0].sort_values(ascending=False).head(10).items()
        }
        logger.warning(
            "Feature finalize dropna removed rows: input_rows=%s horizon_dropped=%s rows_before_dropna=%s "
            "dropna_rows=%s top_missing_columns=%s",
            input_rows,
            horizon_dropped_rows,
            rows_before_dropna,
            rows_with_missing,
            top_missing_columns,
        )
    output.dropna(inplace=True)
    if input_rows > 0 and output.empty:
        logger.warning(
            "Feature finalize produced empty output: input_rows=%s horizon_dropped=%s dropna_rows=%s",
            input_rows,
            horizon_dropped_rows,
            rows_with_missing,
        )
    output.reset_index(drop=True, inplace=True)
    return output


def create_exchange_service() -> ExchangeContract:
    exchange_name = str(getattr(cfg, "ACTIVE_EXCHANGE", "bybit")).strip().lower()
    if exchange_name == "bybit":
        return BybitService()
    if exchange_name == "binance":
        return BinanceService()
    raise ValueError(f"Unsupported ACTIVE_EXCHANGE: {exchange_name}")


def build_labeling_snapshot(timeframe_profile: dict | None = None) -> dict:
    timeframe_profile = timeframe_profile or cfg.get_timeframe_profile()
    timeframe = timeframe_profile["timeframe"]
    labeler = TripleBarrierLabeler(timeframe=timeframe)
    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "timeframe_profile": timeframe_profile["name"],
        "timeframe": timeframe,
        "htf_timeframe": timeframe_profile["htf_timeframe"],
        "htf_timeframes": list(timeframe_profile["htf_timeframes"]),
        "horizon": get_base_horizon(timeframe),
        "horizon_duration": str(getattr(cfg, "HORIZON_DURATION", "12h")),
        "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
        "barrier_atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
        "barrier_rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
        "barrier_tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
        "barrier_min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
        "barrier_max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
        "adaptive_horizon": bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)),
        "adaptive_horizon_min": duration_to_bars(
            str(getattr(cfg, "ADAPTIVE_HORIZON_MIN_DURATION", "8h")),
            timeframe,
        ),
        "adaptive_horizon_max": duration_to_bars(
            str(getattr(cfg, "ADAPTIVE_HORIZON_MAX_DURATION", "20h")),
            timeframe,
        ),
        "adaptive_horizon_vol_low": float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.0)),
        "adaptive_horizon_vol_high": float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.0)),
        "labeling_contract": str(getattr(cfg, "LABELING_CONTRACT_VERSION", "")),
        "vertical_barrier_exit": str(getattr(cfg, "VERTICAL_BARRIER_EXIT", "horizon_close")),
        "labeling_metadata": labeler.metadata(),
    }


def format_yellow_warning(message: str) -> str:
    return f"{ANSI_YELLOW}{message}{ANSI_RESET}"


def parse_iso_datetime_to_utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1000)


def timeframe_to_ms(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe.strip().lower())
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    amount = int(match.group(1))
    unit = match.group(2)
    unit_to_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    return amount * unit_to_ms[unit]


def with_decision_timestamps(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return frame

    output = frame.copy()
    open_time = pd.to_datetime(output["timestamp"], errors="coerce")
    output["open_time"] = open_time
    output["close_time"] = open_time + pd.to_timedelta(timeframe_to_ms(timeframe), unit="ms")
    output["timestamp"] = output["close_time"]
    return output


def align_to_next_candle_open(timestamp_ms: int, timeframe: str) -> int:
    timeframe_ms = timeframe_to_ms(timeframe)
    remainder = timestamp_ms % timeframe_ms
    if remainder == 0:
        return timestamp_ms
    return timestamp_ms + (timeframe_ms - remainder)


def warn_if_history_starts_late(
    repository: HistoricalKlineRepository,
    symbol: str,
    timeframe: str,
    requested_start_date: str,
) -> None:
    requested_start_ts = parse_iso_datetime_to_utc_ms(requested_start_date)
    expected_first_open_ts = align_to_next_candle_open(requested_start_ts, timeframe)
    first_open_time = repository.get_first_open_time(symbol, timeframe)
    if first_open_time is None or first_open_time <= expected_first_open_ts:
        return

    logger.warning(
        format_yellow_warning(
            f"[{symbol}-{timeframe}] incomplete history: requested from "
            f"{datetime.fromtimestamp(requested_start_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}, "
            f"expected first candle at "
            f"{datetime.fromtimestamp(expected_first_open_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}, "
            f"but first available candle starts at "
            f"{datetime.fromtimestamp(first_open_time / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}. "
            "The asset was likely listed after the requested start date."
        )
    )


def build_candle_maps(
    repository: HistoricalKlineRepository,
    symbols_to_load: list,
    timeframe: str,
    htf_timeframe: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_candle_map: dict[str, pd.DataFrame] = {}
    htf_candle_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = with_decision_timestamps(repository.load_candles(symbol, timeframe), timeframe)
        htf_df = with_decision_timestamps(repository.load_candles(symbol, htf_timeframe), htf_timeframe)
        funding_df = repository.load_funding_rates(symbol)
        premium_index_df = with_decision_timestamps(repository.load_premium_index_klines(symbol, timeframe), timeframe)
        open_interest_df = repository.load_open_interest(symbol, timeframe)
        if df.empty or htf_df.empty:
            logger.warning("%s: no data in DB (main=%s, htf=%s)", symbol_name, len(df), len(htf_df))
            continue

        df = attach_funding_context(df, funding_df)
        df = attach_premium_index_context(df, premium_index_df)
        df = attach_open_interest_context(df, open_interest_df)
        logger.info(
            "%s: main=%s, htf=%s rows, funding=%s points, premium=%s points, open_interest=%s points",
            symbol_name,
            len(df),
            len(htf_df),
            len(funding_df),
            len(premium_index_df),
            len(open_interest_df),
        )
        base_candle_map[symbol_name] = df
        htf_candle_map[symbol_name] = htf_df

    return base_candle_map, htf_candle_map


def attach_funding_context(
    base_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if funding_df is None or funding_df.empty:
        output["funding_rate"] = np.nan
        return output

    funding_frame = funding_df[["timestamp", "funding_rate"]].copy().sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        output,
        funding_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def attach_premium_index_context(
    base_df: pd.DataFrame,
    premium_index_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if premium_index_df is None or premium_index_df.empty:
        output["premium_index_close"] = np.nan
        return output

    premium_frame = (
        premium_index_df[["timestamp", "premium_index_close"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        output,
        premium_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def attach_open_interest_context(
    base_df: pd.DataFrame,
    open_interest_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().sort_values("timestamp").reset_index(drop=True)
    if open_interest_df is None or open_interest_df.empty:
        output["open_interest"] = np.nan
        return output

    open_interest_frame = (
        open_interest_df[["timestamp", "open_interest"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        output,
        open_interest_frame,
        on="timestamp",
        direction="backward",
    )
    return merged


def main() -> None:
    args = parse_args()
    timeframe_profile = cfg.get_timeframe_profile(args.timeframe_profile)
    timeframe = timeframe_profile["timeframe"]
    htf_timeframe = timeframe_profile["htf_timeframe"]
    htf_timeframes = timeframe_profile["htf_timeframes"]
    exchange_service = create_exchange_service()
    repository = HistoricalKlineRepository(exchange_code=exchange_service.get_exchange_code())
    repository.init_schema()
    labeling_snapshot = build_labeling_snapshot(timeframe_profile)

    logger.info(
        "Experiment=%s | labeling_profile=%s | training_profile=%s | timeframe_profile=%s (%s -> %s)",
        labeling_snapshot["experiment"],
        labeling_snapshot["labeling_profile"],
        labeling_snapshot["training_profile"],
        labeling_snapshot["timeframe_profile"],
        timeframe,
        ", ".join(htf_timeframes),
    )
    logger.info(
        "ETL labeling config: horizon=%s | dynamic_barriers=%s | stop[min=%.4f max=%.4f] | tp/sl=%.2f",
        labeling_snapshot["horizon"],
        labeling_snapshot["use_dynamic_barriers"],
        labeling_snapshot["barrier_min_pct"],
        labeling_snapshot["barrier_max_pct"],
        labeling_snapshot["barrier_tp_to_sl_ratio"],
    )
    logger.info(
        "Adaptive horizon: enabled=%s | min=%s | max=%s | vol_low=%.4f | vol_high=%.4f",
        labeling_snapshot["adaptive_horizon"],
        labeling_snapshot["adaptive_horizon_min"],
        labeling_snapshot["adaptive_horizon_max"],
        labeling_snapshot["adaptive_horizon_vol_low"],
        labeling_snapshot["adaptive_horizon_vol_high"],
    )

    symbols_to_load = [exchange_service.normalize_symbol(symbol) for symbol in dict.fromkeys(getattr(cfg, "SYMBOLS", []))]

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        start_date = str(getattr(cfg, "START_DATE", "2023-01-01"))
        end_date = getattr(cfg, "END_DATE", None)

        logger.info("Loading %s %s from %s...", symbol_name, timeframe, start_date)
        loaded = repository.sync_candles(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, timeframe, loaded)
        warn_if_history_starts_late(repository, symbol_name, timeframe, start_date)

        for current_htf in htf_timeframes:
            logger.info("Loading %s %s from %s...", symbol_name, current_htf, start_date)
            htf_loaded = repository.sync_candles(
                exchange_service,
                symbol,
                current_htf,
                start_date,
                end_date,
            )
            logger.info("%s %s: %s new candles", symbol_name, current_htf, htf_loaded)
            warn_if_history_starts_late(
                repository,
                symbol_name,
                current_htf,
                start_date,
            )

        logger.info("Loading %s funding from %s...", symbol_name, start_date)
        funding_loaded = repository.sync_funding_rates(exchange_service, symbol, start_date, end_date)
        logger.info("%s funding: %s new points", symbol_name, funding_loaded)

        logger.info("Loading %s premium index %s from %s...", symbol_name, timeframe, start_date)
        premium_loaded = repository.sync_premium_index_klines(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s premium index %s: %s new candles", symbol_name, timeframe, premium_loaded)

        logger.info("Loading %s open interest %s from %s...", symbol_name, timeframe, start_date)
        open_interest_loaded = repository.sync_open_interest(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s open interest %s: %s new points", symbol_name, timeframe, open_interest_loaded)

    base_candle_map, htf_candle_map = build_candle_maps(
        repository,
        symbols_to_load,
        timeframe,
        htf_timeframe,
    )
    feature_builder = MasterFeatureBuilder(
        timeframe=timeframe,
        htf_timeframe=htf_timeframe,
    )
    pipeline_result = feature_builder.build(
        base_candle_map,
        htf_candle_map,
        profile_name=f"model_union_{timeframe_profile['name']}",
    )
    logger.info(
        "Feature build request resolved: profile=%s | blocks=%s | features=%s",
        pipeline_result.profile_name,
        ", ".join(pipeline_result.active_blocks) or "none",
        len(pipeline_result.feature_columns),
    )

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        feature_df = pipeline_result.feature_map.get(symbol_name)
        if feature_df is None or feature_df.empty:
            logger.warning("%s: skipped, missing prepared feature inputs", symbol_name)
            continue

        feature_df = attach_barrier_columns(feature_df, timeframe=timeframe)
        feature_df = triple_barrier_labeling(feature_df, timeframe=timeframe)
        feature_df = finalize_feature_frame(
            feature_df,
            list(pipeline_result.feature_columns),
            timeframe=timeframe,
        )
        repository.save_features(symbol, feature_df, timeframe=timeframe)
        logger.info("%s: saved %s rows with %s requested features", symbol_name, len(feature_df), len(pipeline_result.feature_columns))


if __name__ == "__main__":
    main()
