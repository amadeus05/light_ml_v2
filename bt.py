import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from config import *
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

# Futures settings come from config.py
TAKER_COM = globals().get("TAKER_COM", 0.0004)
SLIPPAGE = globals().get("SLIPPAGE", 0.0003)
LEVERAGE = globals().get("LEVERAGE", 1)
RISK_PER_TRADE = globals().get("RISK_PER_TRADE", 0.01)
BACKTEST_SL_COOLDOWN_BARS = int(globals().get("BACKTEST_SL_COOLDOWN_BARS", 0))
BACKTEST_MAX_SL_PER_DAY = int(globals().get("BACKTEST_MAX_SL_PER_DAY", 0))
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = int(
    globals().get("BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0)
)
BACKTEST_REDUCED_RISK_PER_TRADE = float(globals().get("BACKTEST_REDUCED_RISK_PER_TRADE", RISK_PER_TRADE))
DIRECTIONAL_PROBA_THRESHOLD = globals().get(
    "DIRECTIONAL_PROBA_THRESHOLD",
    globals().get("CONFIDENCE_THRESHOLD", 0.5),
)
BACKTEST_INITIAL_BALANCE = float(globals().get("BACKTEST_INITIAL_BALANCE", 100.0))
USE_DYNAMIC_BARRIERS = bool(globals().get("USE_DYNAMIC_BARRIERS", True))
BACKTEST_CHARTS_DIR = Path(globals().get("BACKTEST_CHARTS_DIR", "backtest_charts"))
BACKTEST_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
EQUITY_CURVE_PATH = BACKTEST_CHARTS_DIR / "equity_curve.png"
DEFAULT_MODEL_NAME = "lightgbm_target"

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


def get_end_date_cutoff():
    if not globals().get("END_DATE"):
        return None
    return pd.to_datetime(globals().get("END_DATE"), errors="coerce")


def apply_end_date_cutoff(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty or timestamp_column not in df.columns:
        return df

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is None or pd.isna(end_cutoff):
        return df

    return df.loc[df[timestamp_column] <= end_cutoff].copy()


def normalize_timestamp_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def parse_period_payload(period_payload: dict | None, period_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not period_payload:
        raise RuntimeError(
            f"Model metadata does not include {period_name}. Re-run train.py with holdout-aware artifacts first."
        )

    start = pd.to_datetime(period_payload.get("start"), errors="coerce")
    end = pd.to_datetime(period_payload.get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise RuntimeError(f"Model metadata has an invalid {period_name}: {period_payload}")
    if start > end:
        raise RuntimeError(f"Model metadata has {period_name} start after end: {period_payload}")
    return start, end


def timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TF_MS[timeframe]


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def format_pnl_pct(pnl_pct: float) -> str:
    color = ANSI_GREEN if pnl_pct >= 0 else ANSI_RED
    return colorize(f"{pnl_pct:+.2f}%", color)


def format_reason(reason: str) -> str:
    if reason == "TP":
        return colorize("✅ TP", ANSI_GREEN)
    if reason == "SL":
        return colorize("❌ SL", ANSI_RED)
    return reason


def build_entry_score(direction_prob: float, signal_gap: float) -> float:
    edge = max(0.0, direction_prob - DIRECTIONAL_PROBA_THRESHOLD)
    return edge * 10 + signal_gap


def resolve_directional_signal(p_long: float, p_short: float) -> tuple[int, float, float]:
    signal_gap = abs(p_long - p_short)

    if (
        p_long >= DIRECTIONAL_PROBA_THRESHOLD
        and (p_long - p_short) >= MIN_SIGNAL_GAP
    ):
        return 1, p_long, signal_gap
    if (
        p_short >= DIRECTIONAL_PROBA_THRESHOLD
        and (p_short - p_long) >= MIN_SIGNAL_GAP
    ):
        return -1, p_short, signal_gap
    return 0, max(p_long, p_short), signal_gap


def normalize_direction_probabilities(
    proba,
    model_mode: str,
    external_predictions: bool = False,
) -> tuple[float, float]:
    if external_predictions or model_mode == "dual":
        return float(proba[0]), float(proba[1])
    if model_mode == "long":
        return 0.0, float(proba[1])
    if model_mode == "short":
        return float(proba[1]), 0.0
    raise ValueError(f"Unsupported model mode for direct probabilities: {model_mode}")


def get_barrier_pcts(feature_row: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None

    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None

    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    if stop_pct <= 0 or take_pct <= 0:
        return None, None
    return stop_pct, take_pct


def compact_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def format_signed_dollars(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def format_percent_value(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def compute_net_pnl_pct(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        raw_pnl = (exit_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - exit_price) / entry_price
    return raw_pnl - (TAKER_COM + TAKER_COM)


def compute_trade_outcome(position: dict, exit_price: float) -> tuple[float, float, float]:
    pnl_clean = compute_net_pnl_pct(position["dir"], position["entry"], exit_price)
    position_notional = float(position["size"])
    commission = position_notional * (TAKER_COM + TAKER_COM)
    trade_profit = position_notional * pnl_clean
    return pnl_clean, trade_profit, commission


def resolve_trade_exit(
    direction: int,
    entry_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_pct: float,
    take_pct: float,
) -> tuple[float | None, str | None]:
    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)
        if next_low <= stop_price:
            exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
            return exit_price, "SL"
        if next_high >= take_price:
            exit_price = take_price * (1 - SLIPPAGE)
            return exit_price, "TP"
    else:
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)
        if next_high >= stop_price:
            exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
            return exit_price, "SL"
        if next_low <= take_price:
            exit_price = take_price * (1 + SLIPPAGE)
            return exit_price, "TP"
    return None, None


def resolve_entry_candidate_exit(candidate: dict, market_batch: dict) -> tuple[float | None, str | None]:
    candidate_ctx = market_batch[candidate["sym"]]
    return resolve_trade_exit(
        candidate["signal"],
        candidate["entry_price"],
        candidate_ctx["next_open"],
        candidate_ctx["next_high"],
        candidate_ctx["next_low"],
        candidate["stop_pct"],
        candidate["take_pct"],
    )


def compute_portfolio_equity(balance: float, positions: dict, mark_prices: dict[str, float]) -> float:
    equity = float(balance)
    for sym, position in positions.items():
        if position is None:
            continue

        mark_price = mark_prices.get(sym)
        if mark_price is None or not np.isfinite(mark_price):
            continue

        pnl_clean = compute_net_pnl_pct(position["dir"], position["entry"], float(mark_price))
        equity += float(position["size"]) * pnl_clean
    return equity


def update_drawdown_stats(equity: float, peak_equity: float, max_drawdown: float) -> tuple[float, float]:
    if equity > peak_equity:
        peak_equity = equity

    if peak_equity > 0:
        current_dd = (peak_equity - equity) / peak_equity * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd

    return peak_equity, max_drawdown


def print_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> None:
    right_align = right_align or set()
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(row_values):
        formatted = []
        for idx, cell in enumerate(row_values):
            text = str(cell)
            if idx in right_align:
                formatted.append(text.rjust(widths[idx]))
            else:
                formatted.append(text.ljust(widths[idx]))
        return "│ " + " │ ".join(formatted) + " │"

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    print(top)
    print(format_row(headers))
    print(mid)
    for row in rows:
        print(format_row(row))
    print(bottom)


def load_raw_candles(symbol: str, timeframe: str, db_path: str | None = None) -> pd.DataFrame:
    repository = HistoricalKlineRepository(db_path=db_path or DB_PATH)
    df = repository.load_candles(symbol, timeframe)

    if df.empty:
        return df

    df["timestamp"] = normalize_timestamp_series(df["timestamp"])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    tf_ms = timeframe_to_ms(timeframe)
    df["close_time"] = df["timestamp"] + pd.to_timedelta(tf_ms, unit="ms")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df = apply_end_date_cutoff(df)
    return df


def execution_timestamp_for_decision_time(decision_ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    return pd.to_datetime(decision_ts) - pd.to_timedelta(timeframe_to_ms(timeframe), unit="ms")


def load_all_raw_data(
    symbols,
    timeframe: str = TIMEFRAME,
    htf_timeframe: str = HTF_TIMEFRAME,
    db_path: str | None = None,
):
    all_data = {}

    print(f"Loading raw data for {len(symbols)} symbols...")
    for sym in symbols:
        try:
            df_main = load_raw_candles(sym, timeframe, db_path=db_path)
            df_htf = load_raw_candles(sym, htf_timeframe, db_path=db_path)

            if df_main.empty:
                print(f"Warning: {sym} has no main TF data ({timeframe})")
                continue

            if df_htf.empty:
                print(f"Warning: {sym} has no HTF data ({htf_timeframe})")
                continue

            all_data[sym] = {"main": df_main, "htf": df_htf}
            print(f"{sym}: main={len(df_main)} candles, htf={len(df_htf)} candles")
        except Exception as e:
            print(f"Warning: failed loading {sym}: {e}")

    return all_data


def load_precomputed_features(
    symbol: str,
    symbol_categories=None,
    required_columns: list | None = None,
    timeframe: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    repository = HistoricalKlineRepository(db_path=db_path or DB_PATH)
    try:
        df = repository.load_features(symbol, timeframe=timeframe)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if required_columns:
        required_order = list(dict.fromkeys(["timestamp"] + required_columns + ["symbol"]))
        missing_columns = [column for column in required_order if column not in df.columns]
        if missing_columns:
            return pd.DataFrame()
        df = df[required_order].copy()

    df = apply_end_date_cutoff(df)
    if symbol_categories is None:
        df["symbol"] = df["symbol"].astype("category")
    else:
        df["symbol"] = pd.Categorical(df["symbol"], categories=symbol_categories)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_timestamp_index(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    return df.set_index("timestamp", drop=False).to_dict("index")


def get_union_main_timestamps(all_data: dict) -> list:
    if not all_data:
        return []

    ts_sets = [set(payload["main"]["timestamp"]) for payload in all_data.values()]
    return sorted(list(set.union(*ts_sets))) if ts_sets else []


def filter_symbols_with_period_overlap(all_data: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp, min_candles: int = 2):
    if not all_data:
        return {}, []

    filtered = {}
    dropped = []
    for symbol, payload in all_data.items():
        period_rows = payload["main"].loc[
            (payload["main"]["timestamp"] >= start_ts) & (payload["main"]["timestamp"] <= end_ts)
        ]
        if len(period_rows) < min_candles:
            dropped.append(symbol)
            continue
        filtered[symbol] = payload

    return filtered, dropped


def build_execution_window(
    decision_start_ts: pd.Timestamp,
    decision_end_ts: pd.Timestamp,
    timeframe: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamp_shift = pd.to_timedelta(timeframe_to_ms(timeframe), unit="ms")
    return pd.to_datetime(decision_start_ts) - timestamp_shift, pd.to_datetime(decision_end_ts) - timestamp_shift


def get_feature_row_precomputed(df: pd.DataFrame, ts: pd.Timestamp, feature_names: list):
    row = df[df["timestamp"] == ts]
    if row.empty:
        return None

    latest_row = row.iloc[[-1]].copy()
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None

    if latest_row[feature_names].isna().any(axis=None):
        return None

    return latest_row


def get_exec_row_by_ts_index(indexed_rows: dict, ts: pd.Timestamp):
    row = indexed_rows.get(ts)
    if row is None:
        return None
    return row


def normalize_features_for_model(latest_row: pd.DataFrame, feature_names: list, symbol_categories=None):
    features = latest_row[feature_names].copy()
    if "symbol" in features.columns:
        if symbol_categories is None:
            features["symbol"] = features["symbol"].astype("category")
        else:
            features["symbol"] = pd.Categorical(features["symbol"], categories=symbol_categories)
    return features


def apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict):
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def prepare_precomputed_feature_store(
    feat_df: pd.DataFrame,
    feature_names: list,
    symbol_categories=None,
    clip_bounds: dict | None = None,
) -> pd.DataFrame:
    if feat_df is None or feat_df.empty:
        return pd.DataFrame(columns=feature_names)

    missing = [feature for feature in feature_names if feature not in feat_df.columns]
    if missing:
        return pd.DataFrame(columns=feature_names)

    timestamps = feat_df["timestamp"].copy()
    prepared = normalize_features_for_model(
        feat_df,
        feature_names,
        symbol_categories=symbol_categories,
    )
    prepared = apply_feature_clip_bounds(prepared, clip_bounds or {})
    prepared.insert(0, "timestamp", timestamps.values)
    prepared = prepared.dropna(subset=feature_names)
    prepared = prepared.drop_duplicates(subset=["timestamp"], keep="last")
    return prepared.set_index("timestamp", drop=True).sort_index()


def get_feature_batch_precomputed(
    feature_store: dict,
    symbols: list,
    ts: pd.Timestamp,
):
    batch_frames = []
    batch_symbols = []

    for symbol in symbols:
        prepared = feature_store.get(symbol)
        if prepared is None or prepared.empty or ts not in prepared.index:
            continue

        batch_frames.append(prepared.loc[[ts]])
        batch_symbols.append(symbol)

    if not batch_frames:
        return [], pd.DataFrame()

    return batch_symbols, pd.concat(batch_frames, axis=0)


def build_prediction_lookup(predictions: pd.DataFrame | None) -> dict:
    if predictions is None or predictions.empty:
        return {}

    required_columns = {"timestamp", "symbol", "p_short", "p_long"}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Walk-forward predictions missing columns: {missing_columns}")

    prepared = predictions.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared = prepared.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return {
        (row.timestamp, row.symbol): (float(row.p_short), float(row.p_long))
        for row in prepared.itertuples(index=False)
    }


def backtest(
    model_name: str = DEFAULT_MODEL_NAME,
    model=None,
    features_meta: dict | None = None,
    predictions: pd.DataFrame | None = None,
    equity_curve_path: Path | None = None,
    result_title: str = "PORTFOLIO BACKTEST RESULTS",
    db_path: str | None = None,
):
    print("Loading model and features...")

    # --- Load LightGBM model / metadata ---
    using_external_predictions = predictions is not None
    prediction_lookup = build_prediction_lookup(predictions)
    if using_external_predictions and not prediction_lookup:
        print("Error: walk-forward predictions are empty.")
        return

    if features_meta is None:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        features_meta_path = MODELS_DIR / f"{model_name}_features.json"
        print(
            "Warning: standalone bt.py replays the production model over its artifact period. "
            "This is not an OOS evaluation; use bt_walk_forward.py for honest walk-forward OOS results."
        )

        if not features_meta_path.exists():
            print(f"Error: features metadata not found at {features_meta_path}. Run train.py first.")
            return

        with open(features_meta_path, "r", encoding="utf-8") as f:
            features_meta = json.load(f)

        if model is None and not using_external_predictions:
            if not model_path.exists():
                print(f"Error: model not found at {model_path}. Run train.py first.")
                return
            model = joblib.load(model_path)
    elif model is None and not using_external_predictions:
        print("Error: model must be provided when features_meta is passed without predictions.")
        return

    model_mode = str(features_meta.get("model_mode", "dual"))
    timeframe = str(features_meta.get("timeframe", TIMEFRAME))
    htf_timeframe = str(features_meta.get("htf_timeframe", HTF_TIMEFRAME))
    if model_mode == "long_short" and not using_external_predictions:
        print("Error: long_short mode requires combined external predictions.")
        return
    model_supports_longs = model_mode in {"dual", "long", "long_short"}
    model_supports_shorts = model_mode in {"dual", "short", "long_short"}
    allow_longs = bool(ALLOW_LONGS and model_supports_longs)
    allow_shorts = bool(ALLOW_SHORTS and model_supports_shorts)
    if not allow_longs and not allow_shorts:
        print(
            "Error: no trade direction is enabled by both the model profile "
            "and ALLOW_LONGS/ALLOW_SHORTS."
        )
        return

    feature_names = features_meta["feature_columns"]
    try:
        test_start_ts, test_end_ts = parse_period_payload(
            features_meta.get("train_period") or features_meta.get("test_period"),
            "train_period",
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return
    trained_symbols = list(features_meta.get("symbols", SYMBOLS))
    backtest_symbols = list(dict.fromkeys(trained_symbols or list(SYMBOLS)))
    use_symbol_feature = "symbol" in feature_names
    unseen_symbols = [symbol for symbol in backtest_symbols if symbol not in trained_symbols]
    if use_symbol_feature:
        symbol_categories = list(dict.fromkeys(trained_symbols + backtest_symbols))
        if unseen_symbols:
            print(
                "Warning: backtest includes symbols absent from training metadata: "
                + ", ".join(unseen_symbols)
            )
    else:
        symbol_categories = None
    feature_clip_meta = features_meta.get("feature_clip", {})
    clip_bounds = feature_clip_meta.get("bounds", {})
    resolved_db_path = db_path or str(globals().get("DB_PATH", DB_PATH))
    if using_external_predictions:
        print(f"Loaded walk-forward OOS predictions: {len(prediction_lookup)} symbol/timestamp rows")
    else:
        print(f"Loaded LightGBM model with {len(feature_names)} features")
    if clip_bounds:
        print(
            "Feature clipping: "
            f"{len(clip_bounds)} columns "
            f"[{feature_clip_meta.get('lower_q', 0.01) * 100:.2f}%, "
            f"{feature_clip_meta.get('upper_q', 0.99) * 100:.2f}%]"
        )
    print(f"Backtest window: {test_start_ts.isoformat()} to {test_end_ts.isoformat()}")
    all_raw = load_all_raw_data(
        backtest_symbols,
        timeframe=timeframe,
        htf_timeframe=htf_timeframe,
        db_path=resolved_db_path,
    )
    if not all_raw:
        print("Error: no raw data for backtest.")
        return

    if using_external_predictions:
        print("Feature mode: external predictions + DB barriers")
        required_feature_columns = ["barrier_stop_pct", "barrier_take_pct"]
    else:
        print("Feature mode: precomputed DB features (ETL)")
        required_feature_columns = feature_names + ["barrier_stop_pct", "barrier_take_pct"]
    feature_timestamp_shift = pd.to_timedelta(timeframe_to_ms(timeframe), unit="ms")
    execution_start_ts, execution_end_ts = build_execution_window(
        test_start_ts,
        test_end_ts,
        timeframe,
    )

    all_features = {}
    all_main_index = {}
    for sym in list(all_raw.keys()):
        feat_df = load_precomputed_features(
            sym,
            symbol_categories=symbol_categories,
            required_columns=required_feature_columns,
            timeframe=timeframe,
            db_path=resolved_db_path,
        )
        if feat_df.empty:
            print(f"Warning: {sym} has no required precomputed feature/barrier rows.")
            continue
        all_features[sym] = feat_df

    all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features}
    if not all_raw:
        print("Error: no symbols with required precomputed feature/barrier rows available.")
        return

    for sym, payload in all_raw.items():
        all_main_index[sym] = build_timestamp_index(payload["main"])

    all_features_prepared = {}
    if not using_external_predictions:
        for sym, feat_df in all_features.items():
            prepared = prepare_precomputed_feature_store(
                feat_df,
                feature_names,
                symbol_categories=symbol_categories,
                clip_bounds=clip_bounds,
            )
            if prepared.empty:
                print(f"Warning: {sym} has no usable precomputed feature rows after preparation.")
                continue
            all_features_prepared[sym] = prepared

        all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features_prepared}
        if not all_raw:
            print("Error: no symbols with prepared precomputed features available.")
            return

    all_raw, dropped_symbols = filter_symbols_with_period_overlap(
        all_raw,
        execution_start_ts,
        execution_end_ts,
        min_candles=2,
    )
    if dropped_symbols:
        print(
            "Warning: dropped symbols without enough overlap inside the backtest window: "
            + ", ".join(dropped_symbols)
        )
    if not all_raw:
        print("Error: no symbols with enough overlap inside the backtest window.")
        return

    market_timestamps = get_union_main_timestamps(all_raw)
    test_timestamps = [
        ts for ts in market_timestamps
        if execution_start_ts <= ts <= execution_end_ts
    ]

    if len(test_timestamps) < 2:
        print("Error: too little common data inside the backtest period.")
        return

    balance = BACKTEST_INITIAL_BALANCE
    initial_balance = balance
    positions = {sym: None for sym in all_raw}
    trades = []
    equity_curve = []
    equity_timestamps = []
    monthly_stats = {}
    peak_equity = balance
    max_drawdown = 0.0
    used_margin = 0.0
    next_trade_number = 1
    stop_cooldown_until_index = {sym: -1 for sym in all_raw}
    current_trade_day = None
    daily_sl_count = 0
    daily_stop_announced = False
    consecutive_loss_count = 0

    print("\n📋 Backtest Configuration:")
    print(f"   Period: {test_timestamps[0].isoformat()} to {test_timestamps[-1].isoformat()}")
    print(f"   Symbols: {', '.join(compact_symbol(sym) for sym in all_raw.keys())}")
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Risk per Trade: {RISK_PER_TRADE * 100:.0f}%")
    print(f"   Leverage: {LEVERAGE:.0f}x")
    print(f"   Main TF: {timeframe} | HTF: {htf_timeframe}")
    if allow_longs and allow_shorts:
        direction_mode = "LONG+SHORT"
    elif allow_longs:
        direction_mode = "LONG ONLY"
    else:
        direction_mode = "SHORT ONLY"
    print(f"   Direction mode: {direction_mode}")
    print(f"   Directional probability threshold: {DIRECTIONAL_PROBA_THRESHOLD:.2f}")
    print(f"   Min signal gap: {MIN_SIGNAL_GAP:.2f}")
    print(f"   Batch entries per bar: {BACKTEST_MAX_NEW_POSITIONS_PER_BAR}")
    print(f"   Max open positions: {BACKTEST_MAX_OPEN_POSITIONS}")
    print(f"   SL cooldown bars: {BACKTEST_SL_COOLDOWN_BARS}")
    print(f"   Max SL per day: {BACKTEST_MAX_SL_PER_DAY}")
    print(
        "   Reduced risk after consecutive losses: "
        f"{BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES} -> {BACKTEST_REDUCED_RISK_PER_TRADE * 100:.2f}%"
    )
    if USE_DYNAMIC_BARRIERS:
        print(
            "   Dynamic barriers: "
            f"ATRx{globals().get('BARRIER_ATR_MULTIPLIER', 1.25):.2f}, "
            f"RVOLx{globals().get('BARRIER_RVOL_MULTIPLIER', 0.75):.2f}, "
            f"TP/SL={globals().get('BARRIER_TP_TO_SL_RATIO', 2.0):.2f}"
        )
    else:
        print(f"   TP: {TP_PCT:.4f} | SL: {SL_PCT:.4f}")
    print(f"\nBacktest on {len(test_timestamps)} candles")
    print("-" * 80)

    num_candles = len(test_timestamps)

    for i in range(num_candles - 1):
        current_ts = test_timestamps[i]
        next_ts = test_timestamps[i + 1]
        trade_day = next_ts.normalize()
        if current_trade_day is None or trade_day != current_trade_day:
            current_trade_day = trade_day
            daily_sl_count = 0
            daily_stop_announced = False

        month_key = next_ts.strftime("%Y-%m")
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {
                "pnl_abs": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "start_balance": balance,
            }

        market_batch = {}
        for sym, payload in all_raw.items():
            curr_exec = get_exec_row_by_ts_index(all_main_index[sym], current_ts)
            next_exec = get_exec_row_by_ts_index(all_main_index[sym], next_ts)
            if curr_exec is None or next_exec is None:
                continue

            market_batch[sym] = {
                "main": payload["main"],
                "htf": payload["htf"],
                "current_close": float(curr_exec["close"]),
                "next_open": float(next_exec["open"]),
                "next_high": float(next_exec["high"]),
                "next_low": float(next_exec["low"]),
                "next_close": float(next_exec["close"]),
            }

        current_mark_prices = {sym: ctx["current_close"] for sym, ctx in market_batch.items()}
        current_equity = compute_portfolio_equity(balance, positions, current_mark_prices)
        equity_curve.append(current_equity)
        equity_timestamps.append(current_ts)
        peak_equity, max_drawdown = update_drawdown_stats(current_equity, peak_equity, max_drawdown)

        entry_balance = balance
        entry_used_margin = used_margin
        entry_consecutive_loss_count = consecutive_loss_count
        entry_daily_sl_count = daily_sl_count
        entry_stop_cooldown_until_index = stop_cooldown_until_index.copy()
        entry_open_symbols = {sym for sym, pos in positions.items() if pos is not None}
        entry_available_balance = max(0.0, entry_balance - entry_used_margin)

        # Phase 1: all exits are evaluated on the same market snapshot.
        for sym, ctx in market_batch.items():
            if positions[sym] is None:
                continue

            pos = positions[sym]
            exit_price, reason = resolve_trade_exit(
                pos["dir"],
                pos["entry"],
                ctx["next_open"],
                ctx["next_high"],
                ctx["next_low"],
                pos["stop_pct"],
                pos["take_pct"],
            )
            if exit_price is None or reason is None:
                continue

            pnl_clean, trade_profit, commission = compute_trade_outcome(pos, exit_price)
            previous_loss_streak = consecutive_loss_count

            used_margin -= pos["margin"]
            if used_margin < 0:
                used_margin = 0.0

            balance += trade_profit

            trades.append(
                {
                    "trade_number": pos["trade_number"],
                    "sym": sym,
                    "direction": "LONG" if pos["dir"] == 1 else "SHORT",
                    "reason": reason,
                    "pnl_pct": pnl_clean,
                    "pnl_abs": trade_profit,
                    "commission": commission,
                    "ts": next_ts,
                }
            )
            monthly_stats[month_key]["pnl_abs"] += trade_profit
            monthly_stats[month_key]["trades"] += 1
            if pnl_clean > 0:
                monthly_stats[month_key]["wins"] += 1
                consecutive_loss_count = 0
            else:
                monthly_stats[month_key]["losses"] += 1
                consecutive_loss_count += 1

            if reason == "SL":
                if BACKTEST_SL_COOLDOWN_BARS > 0:
                    stop_cooldown_until_index[sym] = i + BACKTEST_SL_COOLDOWN_BARS
                daily_sl_count += 1

            positions[sym] = None

            exit_icon = "\u274C" if reason == "SL" else "\u2705" if reason == "TP" else "\u2139\uFE0F"
            print(
                f"[{next_ts}] \u2116 {pos['trade_number']} {exit_icon} {sym}: {format_reason(reason)} | "
                f"PnL: {format_pnl_pct(pnl_clean * 100)} | "
                f"Com: {commission:.2f}$ | "
                f"Bal: {balance:.2f}"
            )
            if (
                BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES > 0
                and BACKTEST_REDUCED_RISK_PER_TRADE < RISK_PER_TRADE
            ):
                if (
                    previous_loss_streak < BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES
                    <= consecutive_loss_count
                ):
                    print(
                        f"[{next_ts}] \u26A0\uFE0F Loss streak {consecutive_loss_count}: "
                        f"risk per trade reduced to {BACKTEST_REDUCED_RISK_PER_TRADE * 100:.2f}%"
                    )
                elif (
                    pnl_clean > 0
                    and previous_loss_streak >= BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES
                ):
                    print(
                        f"[{next_ts}] \u2139\uFE0F Loss streak reset: "
                        f"risk per trade restored to {RISK_PER_TRADE * 100:.2f}%"
                    )
            if (
                reason == "SL"
                and BACKTEST_MAX_SL_PER_DAY > 0
                and daily_sl_count >= BACKTEST_MAX_SL_PER_DAY
                and not daily_stop_announced
            ):
                daily_stop_announced = True
                print(
                    f"[{next_ts}] \u26D4 Daily SL limit reached ({daily_sl_count}), "
                    "new entries are paused until next day"
                )

        # Phase 2: collect all entry candidates first, then rank them.
        effective_risk_per_trade = RISK_PER_TRADE
        if (
            BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES > 0
            and BACKTEST_REDUCED_RISK_PER_TRADE > 0
            and BACKTEST_REDUCED_RISK_PER_TRADE < RISK_PER_TRADE
            and entry_consecutive_loss_count >= BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES
        ):
            effective_risk_per_trade = BACKTEST_REDUCED_RISK_PER_TRADE

        if BACKTEST_MAX_SL_PER_DAY > 0 and entry_daily_sl_count >= BACKTEST_MAX_SL_PER_DAY:
            continue

        snapshot_balance = entry_balance
        entry_candidates = []
        if using_external_predictions:
            candidate_symbols = [
                sym for sym in market_batch
                if (
                    sym not in entry_open_symbols
                    and sym in all_features
                    and i >= entry_stop_cooldown_until_index.get(sym, -1)
                )
            ]
            batch_symbols = []
            batch_proba = []
            prediction_ts = current_ts + feature_timestamp_shift
            for sym in candidate_symbols:
                proba = prediction_lookup.get((prediction_ts, sym))
                if proba is None:
                    continue
                batch_symbols.append(sym)
                batch_proba.append(proba)
            has_signal_batch = bool(batch_symbols)
        else:
            candidate_symbols = [
                sym for sym in market_batch
                if (
                    sym not in entry_open_symbols
                    and sym in all_features_prepared
                    and i >= entry_stop_cooldown_until_index.get(sym, -1)
                )
            ]
            batch_symbols, batch_features = get_feature_batch_precomputed(
                all_features_prepared,
                candidate_symbols,
                current_ts + feature_timestamp_shift,
            )
            has_signal_batch = not batch_features.empty
            if has_signal_batch:
                batch_proba = model.predict_proba(batch_features[feature_names])

        if has_signal_batch:
            for sym, proba in zip(batch_symbols, batch_proba):
                ctx = market_batch[sym]
                required_row_columns = ["barrier_stop_pct", "barrier_take_pct"]
                if not using_external_predictions:
                    required_row_columns = feature_names + required_row_columns
                feature_row = get_feature_row_precomputed(
                    all_features[sym],
                    current_ts + feature_timestamp_shift,
                    required_row_columns,
                )
                stop_pct, take_pct = get_barrier_pcts(feature_row)
                if stop_pct is None or take_pct is None:
                    continue
                p_short, p_long = normalize_direction_probabilities(
                    proba,
                    model_mode=model_mode,
                    external_predictions=using_external_predictions,
                )

                signal, direction_prob, signal_gap = resolve_directional_signal(p_long, p_short)

                if signal == 0:
                    continue

                if signal == 1 and not allow_longs:
                    continue
                if signal == -1 and not allow_shorts:
                    continue

                if signal == 1:
                    direction_str = "LONG"
                    entry_price = ctx["next_open"] * (1 + SLIPPAGE)
                else:
                    direction_str = "SHORT"
                    entry_price = ctx["next_open"] * (1 - SLIPPAGE)

                risk_capital = snapshot_balance * effective_risk_per_trade
                position_notional = min(risk_capital / stop_pct, snapshot_balance * LEVERAGE)
                required_margin = position_notional / LEVERAGE

                if position_notional < 10:
                    continue

                entry_candidates.append(
                    {
                        "sym": sym,
                        "signal": signal,
                        "direction_str": direction_str,
                        "entry_price": entry_price,
                        "position_notional": position_notional,
                        "required_margin": required_margin,
                        "stop_pct": stop_pct,
                        "take_pct": take_pct,
                        "p_long": p_long,
                        "p_short": p_short,
                        "direction_prob": direction_prob,
                        "score": build_entry_score(direction_prob, signal_gap),
                    }
                )

        if not entry_candidates:
            continue

        entry_candidates.sort(
            key=lambda candidate: (
                candidate["score"],
                candidate["direction_prob"],
            ),
            reverse=True,
        )

        opened_this_bar = 0
        open_positions_count = len(entry_open_symbols)
        available_balance_for_entries = entry_available_balance
        for candidate in entry_candidates:
            if opened_this_bar >= BACKTEST_MAX_NEW_POSITIONS_PER_BAR:
                break
            if open_positions_count >= BACKTEST_MAX_OPEN_POSITIONS:
                break

            if available_balance_for_entries <= 0:
                break

            required_margin = min(candidate["required_margin"], available_balance_for_entries)
            position_notional = min(candidate["position_notional"], required_margin * LEVERAGE)

            if position_notional < 10 or required_margin <= 0:
                continue

            available_balance_for_entries -= required_margin
            trade_number = next_trade_number
            next_trade_number += 1
            used_margin += required_margin
            positions[candidate["sym"]] = {
                "trade_number": trade_number,
                "dir": candidate["signal"],
                "entry": candidate["entry_price"],
                "size": position_notional,
                "margin": required_margin,
                "stop_pct": candidate["stop_pct"],
                "take_pct": candidate["take_pct"],
                "ts_open": next_ts,
            }
            opened_this_bar += 1
            open_positions_count += 1

            print(
                f"[{next_ts}] \u2116 {trade_number} \U0001F525 OPEN {candidate['direction_str']}: {candidate['sym']} "
                f"(Long={candidate['p_long']:.2f}, Short={candidate['p_short']:.2f}, "
                f"Score={candidate['score']:.3f}) "
                f"at {candidate['entry_price']:.4f} | "
                f"Size: {position_notional:.2f}$ "
                f"Margin: {required_margin:.2f}$"
            )

            # Keep backtest execution aligned with ETL labeling:
            # a newly opened trade can be stopped/taken on the entry candle.
            exit_price, reason = resolve_entry_candidate_exit(candidate, market_batch)
            if exit_price is None or reason is None:
                continue

            pos = positions[candidate["sym"]]
            if pos is None:
                continue

            pnl_clean, trade_profit, commission = compute_trade_outcome(pos, exit_price)
            previous_loss_streak = consecutive_loss_count

            used_margin -= pos["margin"]
            if used_margin < 0:
                used_margin = 0.0

            balance += trade_profit
            trades.append(
                {
                    "trade_number": pos["trade_number"],
                    "sym": candidate["sym"],
                    "direction": "LONG" if candidate["signal"] == 1 else "SHORT",
                    "reason": reason,
                    "pnl_pct": pnl_clean,
                    "pnl_abs": trade_profit,
                    "commission": commission,
                    "ts": next_ts,
                }
            )
            monthly_stats[month_key]["pnl_abs"] += trade_profit
            monthly_stats[month_key]["trades"] += 1
            if pnl_clean > 0:
                monthly_stats[month_key]["wins"] += 1
                consecutive_loss_count = 0
            else:
                monthly_stats[month_key]["losses"] += 1
                consecutive_loss_count += 1

            if reason == "SL":
                if BACKTEST_SL_COOLDOWN_BARS > 0:
                    stop_cooldown_until_index[candidate["sym"]] = i + BACKTEST_SL_COOLDOWN_BARS
                daily_sl_count += 1

            positions[candidate["sym"]] = None

            exit_icon = "\u274C" if reason == "SL" else "\u2705" if reason == "TP" else "\u2139\uFE0F"
            print(
                f"[{next_ts}] \u2116 {pos['trade_number']} {exit_icon} {candidate['sym']}: {format_reason(reason)} | "
                f"PnL: {format_pnl_pct(pnl_clean * 100)} | "
                f"Com: {commission:.2f}$ | "
                f"Bal: {balance:.2f}"
            )
            if (
                BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES > 0
                and BACKTEST_REDUCED_RISK_PER_TRADE < RISK_PER_TRADE
            ):
                if (
                    previous_loss_streak < BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES
                    <= consecutive_loss_count
                ):
                    print(
                        f"[{next_ts}] \u26A0\uFE0F Loss streak {consecutive_loss_count}: "
                        f"risk per trade reduced to {BACKTEST_REDUCED_RISK_PER_TRADE * 100:.2f}%"
                    )
                elif (
                    pnl_clean > 0
                    and previous_loss_streak >= BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES
                ):
                    print(
                        f"[{next_ts}] \u2139\uFE0F Loss streak reset: "
                        f"risk per trade restored to {RISK_PER_TRADE * 100:.2f}%"
                    )
            if (
                reason == "SL"
                and BACKTEST_MAX_SL_PER_DAY > 0
                and daily_sl_count >= BACKTEST_MAX_SL_PER_DAY
                and not daily_stop_announced
            ):
                daily_stop_announced = True
                print(
                    f"[{next_ts}] \u26D4 Daily SL limit reached ({daily_sl_count}), "
                    "new entries are paused until next day"
                )

    last_timestamp = test_timestamps[-1]
    last_mark_prices = {}
    for sym in all_raw:
        last_exec = get_exec_row_by_ts_index(all_main_index[sym], last_timestamp)
        if last_exec is None:
            continue
        last_mark_prices[sym] = float(last_exec["close"])

    final_month_key = last_timestamp.strftime("%Y-%m")
    if final_month_key not in monthly_stats:
        monthly_stats[final_month_key] = {
            "pnl_abs": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "start_balance": balance,
        }

    for sym, pos in list(positions.items()):
        if pos is None:
            continue

        mark_price = last_mark_prices.get(sym)
        if mark_price is None or not np.isfinite(mark_price):
            continue

        exit_price = mark_price * (1 - SLIPPAGE) if pos["dir"] == 1 else mark_price * (1 + SLIPPAGE)
        pnl_clean, trade_profit, commission = compute_trade_outcome(pos, exit_price)

        used_margin -= pos["margin"]
        if used_margin < 0:
            used_margin = 0.0

        balance += trade_profit
        trades.append(
            {
                "trade_number": pos["trade_number"],
                "sym": sym,
                "direction": "LONG" if pos["dir"] == 1 else "SHORT",
                "reason": "FINAL",
                "pnl_pct": pnl_clean,
                "pnl_abs": trade_profit,
                "commission": commission,
                "ts": last_timestamp,
            }
        )
        monthly_stats[final_month_key]["pnl_abs"] += trade_profit
        monthly_stats[final_month_key]["trades"] += 1
        if pnl_clean > 0:
            monthly_stats[final_month_key]["wins"] += 1
        else:
            monthly_stats[final_month_key]["losses"] += 1

        positions[sym] = None
        print(
            f"[{last_timestamp}] \u2116 {pos['trade_number']} \u23F9 CLOSE {sym}: FINAL | "
            f"PnL: {format_pnl_pct(pnl_clean * 100)} | "
            f"Com: {commission:.2f}$ | "
            f"Bal: {balance:.2f}"
        )

    final_equity = compute_portfolio_equity(balance, positions, last_mark_prices)
    equity_curve.append(final_equity)
    equity_timestamps.append(last_timestamp)
    peak_equity, max_drawdown = update_drawdown_stats(final_equity, peak_equity, max_drawdown)

    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve, index=equity_timestamps)
        daily_equity = equity_series.resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
            cagr = ((daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1) if total_days > 0 else 0
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe = (mean_daily_return / std_daily_return) * np.sqrt(365)

            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                sortino = (mean_daily_return / downside_returns.std()) * np.sqrt(365)
            else:
                sortino = 0.0

            calmar = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0
            calmar = 0.0
            cagr = 0.0

        if trades:
            returns = np.array([t["pnl_abs"] for t in trades])
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            pf = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0
        cagr = 0.0
        pf = 0.0

    total_trades = len(trades)
    total_wins = sum(1 for trade in trades if trade["pnl_abs"] > 0)
    total_losses = total_trades - total_wins
    total_pnl_abs = sum(trade["pnl_abs"] for trade in trades)
    total_fees = sum(trade.get("commission", 0.0) for trade in trades)
    final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return_pct = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
    expectancy = (total_pnl_abs / total_trades) if total_trades > 0 else 0.0

    print("\nSimulation finished.\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print(f"║{result_title[:59].center(59)}║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\n📊 Trades: {total_trades} (W: {total_wins} / L: {total_losses})")
    print("💰 Equity:")
    print(f"   Start: ${initial_balance:.2f}")
    print(f"   End:   ${balance:.2f}")
    print(f"   PnL:   {format_signed_dollars(total_pnl_abs)} ({format_percent_value(total_return_pct)})")
    print(f"   Fees:  ${total_fees:.2f}")
    print("📉 Risk:")
    print(f"   Max DD: {max_drawdown:.2f}%")
    print(f"   Profit Factor: {pf:.2f}")
    print(f"   Expectancy: {format_signed_dollars(expectancy)}")
    print(f"   Sharpe: {sharpe:.2f}")

    monthly_rows = []
    for month_key in sorted(monthly_stats.keys()):
        stats = monthly_stats[month_key]
        monthly_rows.append(
            [
                month_key,
                format_signed_dollars(stats["pnl_abs"]),
                str(stats["trades"]),
                str(stats["wins"]),
                str(stats["losses"]),
            ]
        )

    if monthly_rows:
        print("\n📅 Monthly Performance Extended:")
        print_table(
            ["Month", "PnL", "Total", "Wins", "Losses"],
            monthly_rows,
            right_align={1, 2, 3, 4},
        )

    symbol_stats = {}
    for trade in trades:
        stats = symbol_stats.setdefault(
            trade["sym"],
            {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        )
        stats["trades"] += 1
        stats["pnl_abs"] += trade["pnl_abs"]
        if trade["reason"] == "TP":
            stats["tp"] += 1
        if trade["reason"] == "SL":
            stats["sl"] += 1
        if trade["pnl_abs"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    if symbol_stats:
        coin_rows = []
        for symbol in sorted(symbol_stats.keys()):
            stats = symbol_stats[symbol]
            winrate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0.0
            coin_rows.append(
                [
                    compact_symbol(symbol),
                    str(stats["trades"]),
                    str(stats["tp"]),
                    str(stats["sl"]),
                    f"{winrate:.1f}%",
                    format_signed_dollars(stats["pnl_abs"]),
                ]
            )

        print("\n📊 Summary by Coin:")
        print_table(
            ["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"],
            coin_rows,
            right_align={1, 2, 3, 4, 5},
        )

    direction_stats = {
        "LONG": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        "SHORT": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
    }
    for trade in trades:
        stats = direction_stats[trade["direction"]]
        stats["total"] += 1
        stats["pnl_abs"] += trade["pnl_abs"]
        if trade["pnl_abs"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    direction_rows = []
    for direction in ("LONG", "SHORT"):
        stats = direction_stats[direction]
        direction_rows.append(
            [
                direction,
                str(stats["total"]),
                str(stats["wins"]),
                str(stats["losses"]),
                format_signed_dollars(stats["pnl_abs"]),
            ]
        )

    print("\n📈 Long / Short Summary:")
    print_table(
        ["Direction", "Total", "Wins", "Losses", "PnL"],
        direction_rows,
        right_align={1, 2, 3, 4},
    )

    if len(equity_curve) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_timestamps, equity_curve, label="Portfolio Equity")
        plt.axhline(y=initial_balance, linestyle="--")
        plt.title(f"Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_drawdown:.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend()
        output_chart_path = Path(equity_curve_path or EQUITY_CURVE_PATH)
        output_chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_chart_path, dpi=150)
        plt.show()
        print(f"\nSaved chart: {output_chart_path}")


if __name__ == "__main__":
    backtest()
