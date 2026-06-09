import argparse
import json
import logging
import shutil
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

import config as cfg
from src.features import MasterFeatureBuilder
from src.features.models.feature_spec import serialize_feature_specs
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & label mappings
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Target"
MODEL_TARGET_COLUMNS = {"Target", "TargetLong", "TargetShort"}
TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
RESERVED_COLUMNS = {
    *MODEL_TARGET_COLUMNS,
    TIMESTAMP_COLUMN,
    "barrier_stop_pct",
    "barrier_take_pct",
}
EXCLUDED_RAW_FEATURE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}
LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {0: -1, 1: 1}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI / data loading
# ═══════════════════════════════════════════════════════════════════════════

def get_end_date_cutoff():
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return None
    return pd.to_datetime(end_date, errors="coerce")


def build_experiment_snapshot(model_profile: dict | None = None) -> dict:
    model_profile = model_profile or {
        "name": getattr(cfg, "ACTIVE_MODEL_PROFILE", "dual_v1"),
        **cfg.get_model_profile(),
    }
    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "labeling": {
            "horizon": int(getattr(cfg, "HORIZON", 0)),
            "tp_pct": float(getattr(cfg, "TP_PCT", 0.0)),
            "sl_pct": float(getattr(cfg, "SL_PCT", 0.0)),
            "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
            "barrier_atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
            "barrier_rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
            "barrier_tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
            "barrier_min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
            "barrier_max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
        },
        "training": {
            "model_profile": model_profile["name"],
            "model_mode": model_profile["mode"],
            "target_column": model_profile["target_column"],
            "feature_profile": model_profile["feature_profile"],
            "feature_clip_enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "feature_clip_lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.0)),
            "feature_clip_upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 1.0)),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM classifier on ETL feature tables.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Symbols to load, for example ETH/USDT SOL/USDT.",
    )
    parser.add_argument(
        "--model-profile",
        choices=sorted(cfg.MODEL_PROFILES),
        default=cfg.ACTIVE_MODEL_PROFILE,
        help="Model specification controlling target, features, and default artifact name.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the artifact base filename defined by --model-profile.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of expanding-window folds for Walk-Forward Validation.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["tscv", "monthly"],
        default="tscv",
        help="Walk-forward split mode: tscv uses TimeSeriesSplit, monthly uses rolling calendar OOS tests.",
    )
    parser.add_argument(
        "--monthly-train-months",
        type=int,
        default=6,
        help="Training window in months for --split-mode monthly.",
    )
    parser.add_argument(
        "--monthly-window-mode",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Monthly WFV train mode: expanding uses all history, rolling uses only the latest train window.",
    )
    parser.add_argument(
        "--monthly-test-months",
        type=int,
        default=1,
        help="Test window size in months for --split-mode monthly.",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=cfg.effective_max_label_horizon(),
        help=(
            "Purge gap in timestamps between train and test folds to avoid triple-barrier "
            "target leakage (default = cfg.effective_max_label_horizon())."
        ),
    )
    return parser.parse_args()


def resolve_model_profile(name: str) -> dict:
    return {"name": name, **cfg.get_model_profile(name)}


def format_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return str(timestamp)


def build_timestamp_profile(values):
    timestamps = pd.Series(pd.to_datetime(values, errors="coerce")).dropna()
    if timestamps.empty:
        return {
            "count": 0,
            "first": None,
            "last": None,
        }
    return {
        "count": int(timestamps.nunique()),
        "first": format_timestamp(timestamps.min()),
        "last": format_timestamp(timestamps.max()),
    }


def build_symbol_row_profile(frame, target_column=TARGET_COLUMN):
    if frame.empty or SYMBOL_COLUMN not in frame.columns:
        return {}

    profile = {}
    for symbol, symbol_frame in frame.groupby(SYMBOL_COLUMN, observed=True):
        symbol_key = str(symbol)
        row = {"rows": int(len(symbol_frame))}
        if TIMESTAMP_COLUMN in symbol_frame.columns:
            timestamp_profile = build_timestamp_profile(symbol_frame[TIMESTAMP_COLUMN])
            row.update(
                {
                    "unique_timestamps": timestamp_profile["count"],
                    "first_timestamp": timestamp_profile["first"],
                    "last_timestamp": timestamp_profile["last"],
                }
            )
        if target_column in symbol_frame.columns:
            target_counts = symbol_frame[target_column].value_counts(dropna=False).sort_index()
            row["target_counts"] = {str(key): int(value) for key, value in target_counts.items()}
        profile[symbol_key] = row
    return profile


def load_training_frame(db_path, symbols, model_profile=None):
    """Load the profile target and normalize it to the internal binary class."""
    model_profile = model_profile or {
        "name": getattr(cfg, "ACTIVE_MODEL_PROFILE", "dual_v1"),
        **cfg.get_model_profile(),
    }
    source_target = str(model_profile["target_column"])
    mode = str(model_profile["mode"])

    repository = HistoricalKlineRepository(db_path=db_path)
    dataset = repository.load_feature_dataset(symbols)
    if source_target not in dataset.columns:
        raise RuntimeError(
            f"Dataset is missing target column '{source_target}' for model profile "
            f"'{model_profile['name']}'. Re-run etl.py."
        )
    feature_table_row_counts_by_symbol = build_symbol_row_profile(dataset, source_target)

    dataset = dataset.dropna(subset=[TIMESTAMP_COLUMN, source_target]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    required_non_null_rows = int(len(dataset))

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is not None and not pd.isna(end_cutoff):
        before_rows = len(dataset)
        dataset = dataset.loc[dataset[TIMESTAMP_COLUMN] <= end_cutoff].copy()
        logger.info(
            "Applied END_DATE cutoff at %s: kept %s/%s rows",
            end_cutoff,
            len(dataset),
            before_rows,
        )

    rows_before_filter_by_symbol = build_symbol_row_profile(dataset, source_target)
    all_timestamps = np.sort(dataset[TIMESTAMP_COLUMN].dropna().unique())
    raw_labels = dataset[source_target].astype(int)

    if mode == "dual":
        unknown_labels = sorted(set(raw_labels.unique()) - {-1, 0, 1})
        if unknown_labels:
            raise ValueError(f"Unexpected labels in {source_target}: {unknown_labels}")
        selected_mask = raw_labels != 0
        excluded_rows = int((~selected_mask).sum())
        selected_rows_by_symbol = build_symbol_row_profile(
            dataset.loc[selected_mask],
            source_target,
        )
        dataset = dataset.loc[selected_mask].copy()
        dataset[TARGET_COLUMN] = dataset[source_target].astype(int).map(LABEL_TO_CLASS)
    else:
        unknown_labels = sorted(set(raw_labels.unique()) - {0, 1})
        if unknown_labels:
            raise ValueError(f"Unexpected labels in {source_target}: {unknown_labels}")
        excluded_rows = 0
        selected_rows_by_symbol = build_symbol_row_profile(dataset, source_target)
        dataset[TARGET_COLUMN] = raw_labels

    dataset[SYMBOL_COLUMN] = dataset[SYMBOL_COLUMN].astype("category")
    dataset.attrs["excluded_non_directional_rows"] = excluded_rows
    dataset.attrs["model_profile"] = model_profile["name"]
    dataset.attrs["model_mode"] = mode
    dataset.attrs["source_target_column"] = source_target
    dataset.attrs["all_timestamps"] = all_timestamps
    dataset.attrs["all_timestamps_profile"] = build_timestamp_profile(all_timestamps)
    dataset.attrs["required_non_null_rows"] = required_non_null_rows
    dataset.attrs["rows_before_filter_by_symbol"] = rows_before_filter_by_symbol
    dataset.attrs["directional_rows_by_symbol"] = selected_rows_by_symbol
    dataset.attrs["feature_table_row_counts_by_symbol"] = feature_table_row_counts_by_symbol
    return dataset


# ═══════════════════════════════════════════════════════════════════════════
#  Feature selection & clipping
# ═══════════════════════════════════════════════════════════════════════════

def select_feature_columns(dataset, feature_profile=None):
    if feature_profile is None:
        feature_profile = cfg.get_model_profile()["feature_profile"]
    request = MasterFeatureBuilder().resolve_request(feature_profile)
    profile_name = request.profile
    if dict(getattr(cfg, "FEATURE_PROFILES", {})).get(profile_name) == "__all__":
        raise RuntimeError(
            "Training requires an explicit feature whitelist; profile 'all' is not allowed."
        )

    feature_columns = list(request.active_features)

    use_symbol_feature = bool(getattr(cfg, "USE_SYMBOL_FEATURE", True))
    if use_symbol_feature and SYMBOL_COLUMN not in feature_columns:
        feature_columns.append(SYMBOL_COLUMN)

    invalid_reserved = sorted(set(feature_columns).intersection(RESERVED_COLUMNS | EXCLUDED_RAW_FEATURE_COLUMNS))
    if invalid_reserved:
        raise RuntimeError(
            "Feature profile contains reserved/raw columns: " + ", ".join(invalid_reserved)
        )

    missing_columns = sorted(set(feature_columns) - set(dataset.columns))
    if missing_columns:
        raise RuntimeError(
            f"Feature profile '{profile_name}' is missing columns in the dataset: "
            + ", ".join(missing_columns)
            + ". Re-run etl.py with the same profile."
        )

    non_numeric_columns = [
        column
        for column in feature_columns
        if column != SYMBOL_COLUMN and not pd.api.types.is_numeric_dtype(dataset[column])
    ]
    if non_numeric_columns:
        raise RuntimeError(
            "Feature profile contains non-numeric columns: " + ", ".join(non_numeric_columns)
        )

    if not feature_columns:
        raise RuntimeError("No usable feature columns found in the dataset.")
    logger.info(
        "Using explicit feature profile '%s' with %s model features",
        profile_name,
        len(feature_columns),
    )
    return feature_columns


def get_clippable_feature_columns(dataset, feature_columns):
    clippable_columns = []
    for column in feature_columns:
        if column == SYMBOL_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            clippable_columns.append(column)
    return clippable_columns


def build_feature_clip_bounds(train_df, feature_columns):
    """Compute winsorization bounds from the training set quantiles."""
    if not bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)):
        return {}

    lower_q = float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01))
    upper_q = float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99))
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("FEATURE_CLIP_LOWER_Q and FEATURE_CLIP_UPPER_Q must satisfy 0 <= lower < upper <= 1.")

    clip_bounds = {}
    for column in get_clippable_feature_columns(train_df, feature_columns):
        series = train_df[column].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        lower = series.quantile(lower_q)
        upper = series.quantile(upper_q)
        if pd.isna(lower) or pd.isna(upper):
            continue
        clip_bounds[column] = {"lower": float(lower), "upper": float(upper)}
    return clip_bounds


def apply_feature_clip_bounds(frame, clip_bounds):
    """Apply precomputed winsorization bounds to a dataframe."""
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def resolve_internal_eval_plan(y_train: pd.Series) -> dict:
    """
    Pick a suffix eval slice that remains time-ordered but is less class-skewed.
    """
    n_rows = int(len(y_train))
    if n_rows <= 1:
        base_rate = float(y_train.mean()) if n_rows else 0.5
        return {
            "eval_size": 1,
            "eval_fraction": 1.0,
            "fit_rate": base_rate,
            "eval_rate": base_rate,
            "was_expanded": False,
        }

    min_fraction = float(getattr(cfg, "INTERNAL_EVAL_MIN_FRACTION", 0.15))
    max_fraction = float(getattr(cfg, "INTERNAL_EVAL_MAX_FRACTION", 0.40))
    step_fraction = float(getattr(cfg, "INTERNAL_EVAL_STEP_FRACTION", 0.05))
    max_rate_diff = float(getattr(cfg, "INTERNAL_EVAL_MAX_CLASS_RATE_DIFF", 0.08))

    min_fraction = min(max(min_fraction, 0.05), 0.45)
    max_fraction = min(max(max_fraction, min_fraction), 0.50)
    step_fraction = min(max(step_fraction, 0.01), 0.10)

    candidate_fractions = []
    current_fraction = min_fraction
    while current_fraction <= max_fraction + 1e-9:
        candidate_fractions.append(round(current_fraction, 4))
        current_fraction += step_fraction

    best_plan = None
    for idx, fraction in enumerate(candidate_fractions):
        eval_size = max(1, int(n_rows * fraction))
        if eval_size >= n_rows:
            eval_size = n_rows - 1
        if eval_size <= 0:
            continue

        y_fit = y_train.iloc[:-eval_size]
        y_eval = y_train.iloc[-eval_size:]
        if y_fit.empty:
            continue
        if len(set(y_fit.unique().tolist())) < 2 or len(set(y_eval.unique().tolist())) < 2:
            continue

        fit_rate = float(y_fit.mean())
        eval_rate = float(y_eval.mean())
        plan = {
            "eval_size": int(eval_size),
            "eval_fraction": float(eval_size / n_rows),
            "fit_rate": fit_rate,
            "eval_rate": eval_rate,
            "rate_diff": abs(eval_rate - fit_rate),
            "was_expanded": idx > 0,
        }
        if best_plan is None or plan["rate_diff"] < best_plan["rate_diff"]:
            best_plan = plan
        if plan["rate_diff"] <= max_rate_diff:
            return plan

    if best_plan is not None:
        return best_plan

    eval_size = max(1, int(n_rows * min_fraction))
    if eval_size >= n_rows:
        eval_size = n_rows - 1
    y_fit = y_train.iloc[:-eval_size]
    y_eval = y_train.iloc[-eval_size:]
    return {
        "eval_size": int(eval_size),
        "eval_fraction": float(eval_size / n_rows),
        "fit_rate": float(y_fit.mean()) if not y_fit.empty else float(y_train.mean()),
        "eval_rate": float(y_eval.mean()) if not y_eval.empty else float(y_train.mean()),
        "was_expanded": False,
    }


def resolve_internal_eval_slices(n_rows: int, eval_size: int, requested_purge_gap: int | None = None) -> dict:
    eval_size = int(eval_size)
    if n_rows <= eval_size:
        return {
            "fit_end": 0,
            "eval_start": max(0, n_rows - eval_size),
            "eval_size": max(0, min(eval_size, n_rows)),
            "requested_purge_gap": int(requested_purge_gap or 0),
            "applied_purge_gap": 0,
            "purge_reduced": bool(requested_purge_gap),
        }

    requested_gap = cfg.effective_max_label_horizon() if requested_purge_gap is None else int(requested_purge_gap)
    requested_gap = max(0, requested_gap)
    eval_start = n_rows - eval_size
    max_gap = max(0, eval_start - 1)
    applied_gap = min(requested_gap, max_gap)
    return {
        "fit_end": eval_start - applied_gap,
        "eval_start": eval_start,
        "eval_size": eval_size,
        "requested_purge_gap": requested_gap,
        "applied_purge_gap": applied_gap,
        "purge_reduced": applied_gap < requested_gap,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Model building
# ═══════════════════════════════════════════════════════════════════════════

def compute_sample_weights(
    frame_or_timestamps,
    half_life_days: float | None = None,
    regime_aware: bool = True,
) -> np.ndarray:
    """
    Вычисляет веса сэмплов с экспоненциальным затуханием.

    Параметры:
    - half_life_days: период полураспада в днях (default: из конфига или 90)
    - regime_aware: если True, добавляет буст для самых свежих данных
    """
    # Получаем half-life из конфига или используем дефолт 90 дней (было 365)
    if half_life_days is None:
        half_life_days = float(getattr(cfg, "SAMPLE_WEIGHT_HALF_LIFE_DAYS", 90.0))

    if isinstance(frame_or_timestamps, pd.DataFrame):
        frame = frame_or_timestamps
        timestamps = frame[TIMESTAMP_COLUMN]
    else:
        frame = None
        timestamps = frame_or_timestamps

    ts = pd.to_datetime(timestamps)
    days_ago = (ts.max() - ts).dt.total_seconds() / 86400.0
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago.values)

    # Regime-aware буст: самые свежие 30 дней получают дополнительный вес
    if regime_aware and getattr(cfg, "REGIME_AWARE_WEIGHTING", True):
        recent_days = float(getattr(cfg, "REGIME_RECENT_DAYS_BOOST", 30.0))
        boost_factor = float(getattr(cfg, "REGIME_RECENT_BOOST_FACTOR", 2.0))

        recent_mask = days_ago <= recent_days
        weights = np.where(recent_mask, weights * boost_factor, weights)

        if frame is not None:
            regime_score = np.zeros(len(frame), dtype=float)
            regime_terms = 0

            if "market_breadth_ema_fast_slow_1h" in frame.columns:
                breadth_1h = pd.to_numeric(frame["market_breadth_ema_fast_slow_1h"], errors="coerce").fillna(0.5)
                regime_score += np.abs((breadth_1h.to_numpy() * 2.0) - 1.0).clip(0.0, 1.0)
                regime_terms += 1

            if "market_breadth_pos_return_4h_3" in frame.columns:
                breadth_4h = pd.to_numeric(frame["market_breadth_pos_return_4h_3"], errors="coerce").fillna(0.5)
                regime_score += np.abs((breadth_4h.to_numpy() * 2.0) - 1.0).clip(0.0, 1.0)
                regime_terms += 1

            if "ema_slope_4h" in frame.columns:
                ema_slope_4h = pd.to_numeric(frame["ema_slope_4h"], errors="coerce").fillna(0.0)
                slope_scale = float(getattr(cfg, "REGIME_WEIGHT_SLOPE_SCALE_4H", 0.08))
                if slope_scale > 0:
                    regime_score += np.tanh(np.abs(ema_slope_4h.to_numpy()) / slope_scale)
                    regime_terms += 1

            if regime_terms > 0:
                regime_score /= float(regime_terms)
                regime_strength = np.clip(
                    regime_score * float(getattr(cfg, "REGIME_WEIGHT_STRENGTH", 0.18)),
                    0.0,
                    float(getattr(cfg, "REGIME_WEIGHT_STRENGTH_CAP", 0.25)),
                )
                weights = weights * (1.0 + regime_strength)

    min_weight = float(getattr(cfg, "SAMPLE_WEIGHT_MIN", 0.8))
    max_weight = float(getattr(cfg, "SAMPLE_WEIGHT_MAX", 1.35))
    if max_weight < min_weight:
        max_weight = min_weight

    return np.clip(weights, min_weight, max_weight)


def build_model(seed, n_estimators=800):
    """Instantiate LightGBM binary classifier. class_weight=None → calibrated probs."""
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.005,
        num_leaves=15,
        min_child_samples=150,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        min_split_gain=0.01,
        subsample_freq=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation — accepts raw vectors, not a dataframe
# ═══════════════════════════════════════════════════════════════════════════

def fit_model_with_internal_eval(
    x_train,
    y_train,
    w_train,
    feature_columns,
    seed,
    best_iterations_so_far=None,
):
    model = build_model(seed=seed)
    eval_plan = resolve_internal_eval_plan(y_train.reset_index(drop=True))
    internal_eval_size = int(eval_plan["eval_size"])
    internal_eval_slices = resolve_internal_eval_slices(
        n_rows=len(y_train),
        eval_size=internal_eval_size,
    )
    fit_end = int(internal_eval_slices["fit_end"])
    eval_start = int(internal_eval_slices["eval_start"])
    if internal_eval_slices["purge_reduced"] and internal_eval_slices["requested_purge_gap"] > 0:
        logger.warning(
            "Internal eval purge gap reduced: requested=%s applied=%s rows=%s eval_size=%s",
            internal_eval_slices["requested_purge_gap"],
            internal_eval_slices["applied_purge_gap"],
            len(y_train),
            internal_eval_size,
        )

    x_fit = x_train.iloc[:fit_end]
    y_fit = y_train.iloc[:fit_end]
    w_fit = w_train[:fit_end]
    x_eval = x_train.iloc[eval_start:]
    y_eval = y_train.iloc[eval_start:]

    model.fit(
        x_fit,
        y_fit,
        sample_weight=w_fit,
        eval_set=[(x_eval, y_eval)],
        eval_metric="binary_logloss",
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(getattr(cfg, "EARLY_STOPPING_ROUNDS", 200)),
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = int(model.best_iteration_ or model.n_estimators_)
    unstable_min_best_iter = int(getattr(cfg, "UNSTABLE_FOLD_MIN_BEST_ITER", 25))
    fallback_used = False
    fallback_n_estimators = None

    if best_iter < unstable_min_best_iter:
        prior_median = int(np.median(best_iterations_so_far)) if best_iterations_so_far else int(
            getattr(cfg, "UNSTABLE_FOLD_FALLBACK_DEFAULT_ESTIMATORS", 250)
        )
        fallback_n_estimators = max(
            int(getattr(cfg, "UNSTABLE_FOLD_FALLBACK_MIN_ESTIMATORS", 150)),
            prior_median,
        )
        model = build_model(seed=seed, n_estimators=fallback_n_estimators)
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
        )
        best_iter = int(model.n_estimators_)
        fallback_used = True

    fit_metadata = {
        "internal_eval_size": internal_eval_size,
        "eval_plan": eval_plan,
        "internal_eval_slices": internal_eval_slices,
        "best_iter": best_iter,
        "fallback_used": fallback_used,
        "fallback_n_estimators": fallback_n_estimators,
    }
    return model, fit_metadata


def has_both_classes(values) -> bool:
    return len(set(np.asarray(values).tolist())) > 1


def safe_roc_auc(y_true, p_long) -> float | None:
    if not has_both_classes(y_true):
        return None
    return float(roc_auc_score(y_true, p_long))


def safe_pr_auc(y_true, p_long) -> float | None:
    if not has_both_classes(y_true):
        return None
    return float(average_precision_score(y_true, p_long))


def format_metric_for_log(value, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def evaluate_model(y_true, y_pred, y_proba, split_name, n_rows=None, model_mode="dual"):
    """
    Compute a full metrics dictionary from pre-assembled OOS vectors.

    Parameters
    ----------
    y_true   : array-like of {0, 1}
    y_pred   : array-like of {0, 1}
    y_proba  : ndarray of shape (N, 2) — class probabilities
    split_name : str, used as a label in the metrics dict
    n_rows   : optional int, total rows evaluated (defaults to len(y_true))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    p_positive = y_proba[:, 1]
    if n_rows is None:
        n_rows = len(y_true)

    class_names = ["short", "long"] if model_mode == "dual" else ["no_signal", model_mode]
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # ---- Confidence-threshold breakdown ----
    confidence_thresholds = sorted(
        {
            round(max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5))), 2),
            0.55,
            0.60,
            0.65,
            0.70,
        }
    )
    probability_threshold_metrics = {}
    p_negative = y_proba[:, 0]
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    for threshold in confidence_thresholds:
        threshold = float(threshold)
        signal = np.full(n_rows, -1, dtype=int)
        signal[p_positive >= threshold] = 1
        if model_mode == "dual":
            signal[p_negative >= threshold] = 0
        mask = signal != -1
        selected = int(mask.sum())
        coverage = float(selected / n_rows) if n_rows else 0.0
        positive_signals = int((signal == 1).sum())
        negative_signals = int((signal == 0).sum())
        long_signals = positive_signals if model_mode in {"dual", "long"} else 0
        short_signals = (
            negative_signals if model_mode == "dual"
            else positive_signals if model_mode == "short"
            else 0
        )
        no_trade = int((signal == -1).sum())

        if selected == 0:
            probability_threshold_metrics[f"{threshold:.2f}"] = {
                "rows": 0,
                "coverage": coverage,
                "long_signals": long_signals,
                "short_signals": short_signals,
                "positive_signals": positive_signals,
                "positive_class": "long" if model_mode == "dual" else model_mode,
                "no_trade": no_trade,
                "signal_accuracy": None,
                "signal_balanced_accuracy": None,
                "signal_f1_macro": None,
                "signal_confusion_matrix": None,
                "signal_classification_report": None,
                "long_precision": None,
                "short_precision": None,
                "positive_precision": None,
                "positive_recall_all": 0.0,
                "long_recall_all": 0.0,
                "short_recall_all": 0.0,
            }
            continue

        subset_y_true = y_true_series.loc[mask]
        subset_y_pred = pd.Series(signal[mask], index=subset_y_true.index)
        subset_report = classification_report(
            subset_y_true,
            subset_y_pred,
            labels=[0, 1],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        positive_tp = int(((signal == 1) & (y_true_series.values == 1)).sum())
        negative_tp = int(((signal == 0) & (y_true_series.values == 0)).sum())
        total_positive = int((y_true_series.values == 1).sum())
        total_negative = int((y_true_series.values == 0).sum())
        long_tp = positive_tp if model_mode in {"dual", "long"} else 0
        short_tp = negative_tp if model_mode == "dual" else positive_tp if model_mode == "short" else 0
        positive_recall = positive_tp / total_positive if total_positive > 0 else 0.0
        if model_mode == "dual":
            long_recall = positive_recall
            short_recall = negative_tp / total_negative if total_negative > 0 else 0.0
        elif model_mode == "long":
            long_recall = positive_recall
            short_recall = 0.0
        else:
            long_recall = 0.0
            short_recall = positive_recall

        probability_threshold_metrics[f"{threshold:.2f}"] = {
            "rows": selected,
            "coverage": coverage,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "positive_signals": positive_signals,
            "positive_class": "long" if model_mode == "dual" else model_mode,
            "no_trade": no_trade,
            "signal_accuracy": float(accuracy_score(subset_y_true, subset_y_pred)),
            "signal_balanced_accuracy": float(balanced_accuracy_score(subset_y_true, subset_y_pred)),
            "signal_f1_macro": float(f1_score(subset_y_true, subset_y_pred, average="macro")),
            "signal_confusion_matrix": confusion_matrix(subset_y_true, subset_y_pred, labels=[0, 1]).tolist(),
            "signal_classification_report": subset_report,
            "long_precision": float(long_tp / long_signals) if long_signals > 0 else None,
            "short_precision": float(short_tp / short_signals) if short_signals > 0 else None,
            "positive_precision": float(positive_tp / positive_signals) if positive_signals > 0 else None,
            "positive_recall_all": float(positive_recall),
            "long_recall_all": float(long_recall),
            "short_recall_all": float(short_recall),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": safe_roc_auc(y_true, p_positive),
        "pr_auc": safe_pr_auc(y_true, p_positive),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
        f"{split_name}_rows": n_rows,
        "probability_threshold_metrics": probability_threshold_metrics,
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Walk-Forward Validation  (Expanding Window + Purge Gap)
# ═══════════════════════════════════════════════════════════════════════════

def iter_monthly_timestamp_splits(unique_ts, train_months, test_months, window_mode="expanding"):
    timestamps = pd.Series(pd.to_datetime(unique_ts, errors="coerce")).dropna().sort_values()
    if timestamps.empty:
        return
    if window_mode not in {"expanding", "rolling"}:
        raise ValueError("window_mode must be either 'expanding' or 'rolling'.")

    first_ts = timestamps.iloc[0]
    last_ts = timestamps.iloc[-1]
    train_end = first_ts + pd.DateOffset(months=train_months)
    fold_idx = 1

    while train_end < last_ts:
        test_end = train_end + pd.DateOffset(months=test_months)
        if window_mode == "rolling":
            train_start = train_end - pd.DateOffset(months=train_months)
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
        else:
            train_mask = timestamps < train_end
        test_mask = (timestamps >= train_end) & (timestamps < test_end)
        train_timestamps = timestamps.loc[train_mask].to_numpy()
        test_timestamps = timestamps.loc[test_mask].to_numpy()

        if len(train_timestamps) > 0 and len(test_timestamps) > 0:
            yield fold_idx, train_timestamps, test_timestamps
            fold_idx += 1

        train_end = test_end


def build_timestamp_splits(
    unique_ts,
    n_splits,
    split_mode,
    monthly_train_months,
    monthly_test_months,
    monthly_window_mode="expanding",
):
    if split_mode == "monthly":
        if monthly_train_months <= 0 or monthly_test_months <= 0:
            raise ValueError("monthly train/test windows must be positive month counts.")
        return list(
            iter_monthly_timestamp_splits(
                unique_ts,
                monthly_train_months,
                monthly_test_months,
                window_mode=monthly_window_mode,
            )
        )

    n_timestamps = len(unique_ts)
    if n_timestamps < n_splits + 1:
        raise RuntimeError(
            f"Only {n_timestamps} unique timestamps — need at least {n_splits + 1} for {n_splits}-fold WFV."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [
        (fold_idx, unique_ts[train_ts_idx], unique_ts[test_ts_idx])
        for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(tscv.split(unique_ts), start=1)
    ]


def walk_forward_validation(
    dataset,
    feature_columns,
    seed,
    n_splits=5,
    purge_gap=None,
    split_mode="tscv",
    monthly_train_months=6,
    monthly_test_months=1,
    monthly_window_mode="expanding",
):
    """
    Expanding-window walk-forward cross-validation with embargo / purge gap.

    For each fold produced by TimeSeriesSplit (operating on *unique sorted
    timestamps*) we:
      1. Remove `purge_gap` timestamps from the END of the training window
         to prevent triple-barrier target leakage across the boundary.
      2. Train a fresh LightGBM on the purged training set.
      3. Predict on the test set and accumulate OOS predictions.

    Returns
    -------
    oos_metrics : dict   — honest out-of-sample metrics over all folds
    fold_details : list  — per-fold diagnostic summaries
    median_best_iter : int — median best_iteration across folds (useful for
                             choosing n_estimators for the final production model)
    """
    if purge_gap is None:
        purge_gap = cfg.effective_max_label_horizon()

    logger.info("=" * 72)
    logger.info(
        "Walk-Forward Validation | split_mode=%s | n_splits=%s | monthly_train=%s | "
        "monthly_test=%s | monthly_window=%s | purge_gap=%s timestamps",
        split_mode,
        n_splits,
        monthly_train_months,
        monthly_test_months,
        monthly_window_mode,
        purge_gap,
    )
    logger.info("=" * 72)

    # --- Unique sorted timestamp index for time-aware splitting -----------
    # Use the full bar timeline when available. Event-filtered/directional
    # rows can be sparse, so purging N selected timestamps is not the same as
    # purging N market bars near the fold boundary.
    unique_ts = np.asarray(dataset.attrs.get("all_timestamps", np.sort(dataset[TIMESTAMP_COLUMN].unique())))
    n_timestamps = len(unique_ts)
    if n_timestamps < n_splits + 1:
        raise RuntimeError(
            f"Only {n_timestamps} unique timestamps — need at least {n_splits + 1} for {n_splits}-fold WFV."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    timestamp_splits = build_timestamp_splits(
        unique_ts=unique_ts,
        n_splits=n_splits,
        split_mode=split_mode,
        monthly_train_months=monthly_train_months,
        monthly_test_months=monthly_test_months,
        monthly_window_mode=monthly_window_mode,
    )
    if not timestamp_splits:
        raise RuntimeError("No walk-forward timestamp splits were produced.")

    # Accumulators for the single OOS vector
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_details = []
    best_iterations = []
    fold_importance_frames = []

    for fold_idx, original_train_timestamps, test_timestamps in timestamp_splits:
        # --- Resolve timestamp boundaries --------------------------------
        train_timestamps = original_train_timestamps

        # Purge: remove `purge_gap` latest timestamps from train to create
        # an embargo zone that prevents triple-barrier label contamination.
        if purge_gap > 0 and len(train_timestamps) > purge_gap:
            train_timestamps = train_timestamps[:-purge_gap]
        elif purge_gap > 0:
            logger.warning(
                "Fold %s: purge_gap=%s >= train timestamps (%s), skipping purge",
                fold_idx, purge_gap, len(train_timestamps),
            )

        train_ts_set = set(train_timestamps)
        test_ts_set = set(test_timestamps)

        train_mask = dataset[TIMESTAMP_COLUMN].isin(train_ts_set)
        test_mask = dataset[TIMESTAMP_COLUMN].isin(test_ts_set)

        train_df = dataset.loc[train_mask].copy()
        test_df = dataset.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            logger.warning("Fold %s produced empty train or test — skipping", fold_idx)
            continue

        # Ensure both classes in training set
        train_classes = sorted(train_df[TARGET_COLUMN].unique().tolist())
        if len(train_classes) < 2:
            logger.warning("Fold %s has only class(es) %s in train — skipping", fold_idx, train_classes)
            continue

        # --- Feature clipping (fit on train, apply to train+test) --------
        fold_clip_bounds = build_feature_clip_bounds(train_df, feature_columns)
        train_df = apply_feature_clip_bounds(train_df, fold_clip_bounds)
        test_df = apply_feature_clip_bounds(test_df, fold_clip_bounds)

        x_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        x_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]
        w_train = compute_sample_weights(train_df)

        # --- Train -------------------------------------------------------
        model, fit_metadata = fit_model_with_internal_eval(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            feature_columns=feature_columns,
            seed=seed,
            best_iterations_so_far=best_iterations,
        )

        eval_plan = fit_metadata["eval_plan"]
        internal_eval_size = int(fit_metadata["internal_eval_size"])
        best_iter = int(fit_metadata["best_iter"])
        best_iterations.append(best_iter)
        fold_importance_frames.append(build_fold_importance_frame(model, feature_columns, fold_idx))

        # --- Predict on OOS test fold ------------------------------------
        y_pred_fold = model.predict(x_test)
        y_proba_fold = model.predict_proba(x_test)

        all_y_true.append(y_test.values)
        all_y_pred.append(y_pred_fold)
        all_y_proba.append(y_proba_fold)

        # Per-fold quick summary
        fold_acc = float(accuracy_score(y_test, y_pred_fold))
        fold_auc = safe_roc_auc(y_test, y_proba_fold[:, 1])
        fold_info = {
            "fold": fold_idx,
            "split_mode": split_mode,
            "monthly_window_mode": monthly_window_mode if split_mode == "monthly" else None,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "purged_timestamps": purge_gap,
            "timestamp_boundaries": {
                "train_start": format_timestamp(train_timestamps[0]),
                "train_end_before_purge": format_timestamp(original_train_timestamps[-1]),
                "train_end_after_purge": format_timestamp(train_timestamps[-1]),
                "test_start": format_timestamp(test_timestamps[0]),
                "test_end": format_timestamp(test_timestamps[-1]),
                "train_unique_timestamps_after_purge": int(len(train_timestamps)),
                "test_unique_timestamps": int(len(test_timestamps)),
            },
            "train_period": {
                "start": str(train_df[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(train_df[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "test_period": {
                "start": str(test_df[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(test_df[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "internal_eval": {
                "rows": int(internal_eval_size),
                "fraction": float(eval_plan["eval_fraction"]),
                "fit_positive_rate": float(eval_plan["fit_rate"]),
                "eval_positive_rate": float(eval_plan["eval_rate"]),
                "was_expanded": bool(eval_plan["was_expanded"]),
                "fallback_used": bool(fit_metadata["fallback_used"]),
                "fallback_n_estimators": (
                    int(fit_metadata["fallback_n_estimators"])
                    if fit_metadata["fallback_n_estimators"] is not None
                    else None
                ),
            },
            "best_iteration": best_iter,
            "accuracy": fold_acc,
            "roc_auc": fold_auc,
        }
        fold_details.append(fold_info)

        logger.info(
            "Fold %s/%s | train=%s rows [%s → %s] | test=%s rows [%s → %s] | "
            "eval=%s (fit_pos=%.3f eval_pos=%.3f%s%s) | best_iter=%s | acc=%.4f | auc=%s",
            fold_idx,
            len(timestamp_splits),
            fold_info["train_rows"],
            fold_info["train_period"]["start"],
            fold_info["train_period"]["end"],
            fold_info["test_rows"],
            fold_info["test_period"]["start"],
            fold_info["test_period"]["end"],
            internal_eval_size,
            float(eval_plan["fit_rate"]),
            float(eval_plan["eval_rate"]),
            ", expanded" if eval_plan["was_expanded"] else "",
            f", fallback={fit_metadata['fallback_n_estimators']}" if fit_metadata["fallback_used"] else "",
            best_iter,
            fold_acc,
            format_metric_for_log(fold_auc),
        )

    # --- Aggregate OOS vector ------------------------------------------------
    if not all_y_true:
        raise RuntimeError("All WFV folds were skipped — cannot compute OOS metrics.")

    oos_y_true = np.concatenate(all_y_true)
    oos_y_pred = np.concatenate(all_y_pred)
    oos_y_proba = np.vstack(all_y_proba)

    oos_metrics = evaluate_model(
        y_true=oos_y_true,
        y_pred=oos_y_pred,
        y_proba=oos_y_proba,
        split_name="oos",
        n_rows=len(oos_y_true),
        model_mode=str(dataset.attrs.get("model_mode", "dual")),
    )

    median_best_iter = int(np.median(best_iterations))

    logger.info("-" * 72)
    logger.info(
        "OOS aggregate (%s folds, %s rows) | acc=%.4f | bal_acc=%.4f | "
        "f1_macro=%.4f | roc_auc=%s | pr_auc=%s | mcc=%.4f",
        len(fold_details),
        len(oos_y_true),
        oos_metrics["accuracy"],
        oos_metrics["balanced_accuracy"],
        oos_metrics["f1_macro"],
        format_metric_for_log(oos_metrics["roc_auc"]),
        format_metric_for_log(oos_metrics["pr_auc"]),
        oos_metrics["mcc"],
    )
    logger.info("Median best_iteration across folds: %s", median_best_iter)
    logger.info("=" * 72)

    fold_importance = aggregate_fold_importance(fold_importance_frames, feature_columns)
    if not fold_importance.empty:
        logger.info("Fold feature importance top by mean gain:")
        for rank, row in enumerate(fold_importance.head(10).itertuples(index=False), start=1):
            logger.info(
                "%s. %s | mean_gain=%.6f | std_gain=%.6f | top10_folds=%s | top20_folds=%s",
                rank,
                row.feature,
                float(row.mean_gain_by_fold),
                float(row.std_gain_by_fold),
                int(row.top_10_fold_count),
                int(row.top_20_fold_count),
            )

    return oos_metrics, fold_details, median_best_iter, fold_importance


# ═══════════════════════════════════════════════════════════════════════════
#  Production model (retrain on 100% of data)
# ═══════════════════════════════════════════════════════════════════════════

def train_production_model(dataset, feature_columns, seed, n_estimators):
    """
    Train the final deployment model on the ENTIRE dataset.

    The n_estimators is typically the median best_iteration from WFV,
    so we do NOT use early stopping here — every row is training data,
    and we have no hold-out to compute an eval metric on.
    """
    logger.info(
        "Training production model on 100%% of data (%s rows) with n_estimators=%s",
        len(dataset), n_estimators,
    )
    model = build_model(seed=seed, n_estimators=n_estimators)
    w_prod = compute_sample_weights(dataset)
    model.fit(
        dataset[feature_columns],
        dataset[TARGET_COLUMN],
        sample_weight=w_prod,
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities — importance / persistence
# ═══════════════════════════════════════════════════════════════════════════

def build_fold_importance_frame(model, feature_columns, fold_idx):
    return pd.DataFrame(
        {
            "feature": feature_columns,
            f"fold_{fold_idx}_gain": model.booster_.feature_importance(importance_type="gain"),
            f"fold_{fold_idx}_split": model.booster_.feature_importance(importance_type="split"),
        }
    )


def aggregate_fold_importance(fold_importance_frames, feature_columns):
    importance = pd.DataFrame({"feature": feature_columns})
    if not fold_importance_frames:
        return importance

    for frame in fold_importance_frames:
        importance = importance.merge(frame, on="feature", how="left")

    gain_columns = [column for column in importance.columns if column.endswith("_gain")]
    split_columns = [column for column in importance.columns if column.endswith("_split")]
    importance[gain_columns + split_columns] = importance[gain_columns + split_columns].fillna(0.0)

    gain_ranks = importance[gain_columns].rank(axis=0, ascending=False, method="min")
    importance["mean_gain_by_fold"] = importance[gain_columns].mean(axis=1)
    importance["std_gain_by_fold"] = importance[gain_columns].std(axis=1).fillna(0.0)
    importance["mean_split_by_fold"] = importance[split_columns].mean(axis=1)
    importance["top_10_fold_count"] = (gain_ranks <= 10).sum(axis=1).astype(int)
    importance["top_20_fold_count"] = (gain_ranks <= 20).sum(axis=1).astype(int)
    importance["nonzero_gain_fold_count"] = (importance[gain_columns] > 0).sum(axis=1).astype(int)

    ordered_columns = [
        "feature",
        "mean_gain_by_fold",
        "std_gain_by_fold",
        "mean_split_by_fold",
        "top_10_fold_count",
        "top_20_fold_count",
        "nonzero_gain_fold_count",
        *gain_columns,
        *split_columns,
    ]
    return importance[ordered_columns].sort_values(
        ["mean_gain_by_fold", "top_10_fold_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_fold_importance_summary(fold_importance, limit=20):
    if fold_importance is None or fold_importance.empty:
        return []

    summary_columns = [
        "feature",
        "mean_gain_by_fold",
        "std_gain_by_fold",
        "top_10_fold_count",
        "top_20_fold_count",
        "nonzero_gain_fold_count",
    ]
    return fold_importance.head(limit)[summary_columns].to_dict(orient="records")


def build_period_payload(frame):
    if frame.empty:
        return None
    return {
        "start": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "end": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
    }


def build_fold_boundary_summary(fold_details):
    return [
        {
            "fold": int(fold["fold"]),
            "train_rows": int(fold["train_rows"]),
            "test_rows": int(fold["test_rows"]),
            **fold.get("timestamp_boundaries", {}),
        }
        for fold in fold_details
    ]


def build_dataset_diagnostics(dataset, fold_details):
    all_timestamp_profile = dataset.attrs.get("all_timestamps_profile")
    if not all_timestamp_profile:
        all_timestamp_profile = build_timestamp_profile(dataset.attrs.get("all_timestamps", []))

    excluded_non_directional_rows = int(dataset.attrs.get("excluded_non_directional_rows", 0))

    return {
        "model_profile": dataset.attrs.get("model_profile"),
        "model_mode": dataset.attrs.get("model_mode"),
        "source_target_column": dataset.attrs.get("source_target_column"),
        "all_timestamps_count": int(all_timestamp_profile["count"]),
        "first_all_timestamp": all_timestamp_profile["first"],
        "last_all_timestamp": all_timestamp_profile["last"],
        "feature_rows_after_required_columns": int(dataset.attrs.get("required_non_null_rows", 0)),
        "directional_rows_after_filter": int(len(dataset)),
        "excluded_non_directional_rows": excluded_non_directional_rows,
        "dataset_period_after_filters": build_period_payload(dataset),
        "fold_boundary_timestamps": build_fold_boundary_summary(fold_details),
        "feature_table_row_counts_by_symbol": dataset.attrs.get("feature_table_row_counts_by_symbol", {}),
        "rows_before_filter_by_symbol": dataset.attrs.get("rows_before_filter_by_symbol", {}),
        "directional_rows_by_symbol": dataset.attrs.get("directional_rows_by_symbol", {}),
    }


def optional_float(value):
    return float(value) if value is not None else None


def build_fold_stability_payload(fold_details):
    if not fold_details:
        return {
            "accuracy_std": None,
            "accuracy_range": None,
            "roc_auc_std": None,
            "roc_auc_range": None,
        }

    accuracy_values = np.asarray([float(fold["accuracy"]) for fold in fold_details], dtype=float)
    roc_auc_values = np.asarray(
        [float(fold["roc_auc"]) for fold in fold_details if fold.get("roc_auc") is not None],
        dtype=float,
    )
    return {
        "accuracy_std": float(np.std(accuracy_values)),
        "accuracy_range": float(np.max(accuracy_values) - np.min(accuracy_values)),
        "roc_auc_std": float(np.std(roc_auc_values)) if len(roc_auc_values) > 0 else None,
        "roc_auc_range": (
            float(np.max(roc_auc_values) - np.min(roc_auc_values))
            if len(roc_auc_values) > 0 else None
        ),
    }


def get_train_history_path(model_name):
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    return cfg.MODELS_DIR / f"{model_name}_train_history.json"


def load_train_history(history_path):
    if not history_path.exists():
        return []

    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Train history file is corrupted, resetting history: %s", history_path)
        return []

    if not isinstance(payload, list):
        logger.warning("Train history file has unexpected format, resetting history: %s", history_path)
        return []
    return payload


def extract_configured_signal_metrics(oos_metrics):
    threshold = round(max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5))), 2)
    threshold_key = f"{threshold:.2f}"
    threshold_metrics = (oos_metrics.get("probability_threshold_metrics") or {}).get(threshold_key, {})
    return {
        "configured_threshold": threshold,
        "signal_accuracy": threshold_metrics.get("signal_accuracy"),
        "signal_coverage": threshold_metrics.get("coverage"),
        "signal_rows": threshold_metrics.get("rows"),
    }


def build_train_history_entry(args, metrics, experiment_snapshot):
    oos_metrics = metrics["oos_metrics"]
    fold_stability = build_fold_stability_payload(metrics.get("fold_details", []))
    configured_signal_metrics = extract_configured_signal_metrics(oos_metrics)
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_name": args.model_name,
        "model_profile": args.model_profile,
        "model_mode": metrics.get("model_mode"),
        "target_column": metrics.get("source_target_column"),
        "experiment": experiment_snapshot["experiment"],
        "labeling_profile": experiment_snapshot["labeling_profile"],
        "training_profile": experiment_snapshot["training_profile"],
        "accuracy": float(oos_metrics["accuracy"]),
        "balanced_accuracy": float(oos_metrics["balanced_accuracy"]),
        "f1_macro": float(oos_metrics["f1_macro"]),
        "roc_auc": optional_float(oos_metrics["roc_auc"]),
        "pr_auc": optional_float(oos_metrics["pr_auc"]),
        "mcc": float(oos_metrics["mcc"]),
        "fold_stability_pct": (
            float(fold_stability["accuracy_std"]) * 100 if fold_stability["accuracy_std"] is not None else None
        ),
        "fold_accuracy_range_pct": (
            float(fold_stability["accuracy_range"]) * 100 if fold_stability["accuracy_range"] is not None else None
        ),
        "fold_roc_auc_stability_pct": (
            float(fold_stability["roc_auc_std"]) * 100 if fold_stability["roc_auc_std"] is not None else None
        ),
        "configured_threshold": float(configured_signal_metrics["configured_threshold"]),
        "signal_accuracy": (
            float(configured_signal_metrics["signal_accuracy"])
            if configured_signal_metrics["signal_accuracy"] is not None else None
        ),
        "signal_coverage": (
            float(configured_signal_metrics["signal_coverage"])
            if configured_signal_metrics["signal_coverage"] is not None else None
        ),
        "signal_rows": (
            int(configured_signal_metrics["signal_rows"])
            if configured_signal_metrics["signal_rows"] is not None else None
        ),
        "median_best_iteration": int(metrics["median_best_iteration"]),
        "total_rows": int(metrics["total_rows"]),
        "feature_count": int(metrics["feature_count"]),
    }


def save_train_history(history_path, history_entry, limit=200):
    history = load_train_history(history_path)
    history.append(history_entry)
    history = history[-limit:]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def format_compact_metric_value(value, percent=False, decimals=4):
    if value is None:
        return "-"
    if percent:
        return f"{float(value):.{decimals}f}%"
    return f"{float(value):.{decimals}f}"


def build_current_run_summary_lines(history_entry):
    configured_threshold = float(history_entry.get("configured_threshold", max(0.5, float(getattr(cfg, "CONFIDENCE_THRESHOLD", 0.5)))))
    rows = [
        ("Accuracy", format_compact_metric_value(history_entry["accuracy"] * 100, percent=True, decimals=2)),
        (
            f"Signal acc @{configured_threshold:.2f}",
            format_compact_metric_value(
                (
                    float(history_entry["signal_accuracy"]) * 100
                    if history_entry.get("signal_accuracy") is not None else None
                ),
                percent=True,
                decimals=2,
            ),
        ),
        (
            f"Coverage @{configured_threshold:.2f}",
            format_compact_metric_value(
                (
                    float(history_entry["signal_coverage"]) * 100
                    if history_entry.get("signal_coverage") is not None else None
                ),
                percent=True,
                decimals=2,
            ),
        ),
        ("MCC", format_compact_metric_value(history_entry["mcc"], decimals=3)),
        ("ROC AUC", format_compact_metric_value(history_entry["roc_auc"], decimals=3)),
        ("PR AUC", format_compact_metric_value(history_entry["pr_auc"], decimals=3)),
        ("Fold stability", format_compact_metric_value(history_entry["fold_stability_pct"], percent=True, decimals=2)),
    ]
    metric_width = max(len("Metric"), *(len(name) for name, _ in rows))
    value_width = max(len("Current"), *(len(value) for _, value in rows))
    border = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    lines = [
        border,
        f"| {'Metric'.ljust(metric_width)} | {'Current'.ljust(value_width)} |",
        border,
    ]
    for name, value in rows:
        lines.append(f"| {name.ljust(metric_width)} | {value.ljust(value_width)} |")
    lines.append(border)
    return lines


def build_recent_runs_table_lines(history, limit=10):
    recent_entries = list(reversed(history[-limit:]))
    if not recent_entries:
        return ["No train history yet."]

    columns = [
        ("Run", lambda item: str(item.get("run_timestamp_utc", ""))[5:16]),
        ("Exp", lambda item: str(item.get("experiment", ""))[:18]),
        ("Acc", lambda item: format_compact_metric_value(item.get("accuracy", 0.0) * 100, percent=True, decimals=2)),
        (
            "Sig",
            lambda item: format_compact_metric_value(
                item.get("signal_accuracy") * 100 if item.get("signal_accuracy") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        (
            "Cov",
            lambda item: format_compact_metric_value(
                item.get("signal_coverage") * 100 if item.get("signal_coverage") is not None else None,
                percent=True,
                decimals=2,
            ),
        ),
        ("MCC", lambda item: format_compact_metric_value(item.get("mcc"), decimals=3)),
        ("ROC", lambda item: format_compact_metric_value(item.get("roc_auc"), decimals=3)),
        ("PR", lambda item: format_compact_metric_value(item.get("pr_auc"), decimals=3)),
        ("Stab", lambda item: format_compact_metric_value(item.get("fold_stability_pct"), percent=True, decimals=2)),
        ("Rows", lambda item: str(item.get("total_rows", "-"))),
    ]

    rendered_rows = [[formatter(entry) for _, formatter in columns] for entry in recent_entries]
    widths = [
        max(len(header), *(len(row[idx]) for row in rendered_rows))
        for idx, (header, _) in enumerate(columns)
    ]

    def render_border():
        return "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render_row(values):
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        render_border(),
        render_row([header for header, _ in columns]),
        render_border(),
    ]
    for row in rendered_rows:
        lines.append(render_row(row))
    lines.append(render_border())
    return lines


def log_train_history_summary(history_entry, history, limit=10):
    logger.info("=" * 72)
    logger.info("Current training summary:")
    for line in build_current_run_summary_lines(history_entry):
        logger.info(line)
    logger.info("Recent training runs (latest %s):", min(limit, len(history)))
    for line in build_recent_runs_table_lines(history, limit=limit):
        logger.info(line)


def top_feature_importance(model, feature_columns, limit=25):
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    return importance.head(limit)


def log_feature_importance_ranking(model, feature_columns):
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    logger.info("Feature importance ranking:")
    for rank, row in enumerate(importance.itertuples(index=False), start=1):
        logger.info(
            "%s. %s | gain=%.6f | split=%s",
            rank,
            row.feature,
            float(row.importance_gain),
            int(row.importance_split),
        )


def build_feature_formulas_payload(feature_columns, model_name, symbols, experiment_snapshot):
    builder = MasterFeatureBuilder()
    allowed_untracked = {SYMBOL_COLUMN}
    tracked_feature_columns = [column for column in feature_columns if column not in allowed_untracked]
    feature_specs = builder.collect_feature_specs(set(tracked_feature_columns))
    missing_specs = sorted(set(tracked_feature_columns) - set(feature_specs))
    if missing_specs:
        raise RuntimeError(
            "Missing FeatureSpec metadata for trained features: " + ", ".join(missing_specs)
        )

    return {
        "model_name": model_name,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "feature_columns": list(feature_columns),
        "tracked_feature_columns": tracked_feature_columns,
        "untracked_feature_columns": [column for column in feature_columns if column in allowed_untracked],
        "symbols": list(symbols),
        "experiment": experiment_snapshot,
        "features": serialize_feature_specs(feature_specs, cfg),
    }


def build_model_label_metadata(model_profile: dict) -> dict:
    if model_profile["mode"] == "dual":
        return {
            "label_mapping": {"short": 0, "long": 1},
            "inverse_label_mapping": {str(key): value for key, value in CLASS_TO_LABEL.items()},
        }
    return {
        "label_mapping": {"no_signal": 0, model_profile["mode"]: 1},
        "positive_label": int(model_profile.get("positive_label", 1)),
    }


def save_directional_artifacts(
    model,
    metrics,
    dataset,
    feature_columns,
    clip_bounds,
    fold_importance,
    args,
    experiment_snapshot,
):
    cfg.MODELS_DIR.mkdir(exist_ok=True)

    model_path = cfg.MODELS_DIR / f"{args.model_name}.joblib"
    metrics_path = cfg.MODELS_DIR / f"{args.model_name}_metrics.json"
    features_path = cfg.MODELS_DIR / f"{args.model_name}_features.json"
    importance_path = cfg.MODELS_DIR / f"{args.model_name}_feature_importance.csv"
    fold_importance_path = cfg.MODELS_DIR / f"{args.model_name}_fold_feature_importance.csv"
    feature_formulas_path = cfg.MODELS_DIR / f"{args.model_name}_feature_formulas.json"
    artifact_paths = [model_path, metrics_path, features_path, importance_path, fold_importance_path, feature_formulas_path]

    model_profile = resolve_model_profile(args.model_profile)
    payload = {
        "feature_columns": feature_columns,
        "model_profile": model_profile["name"],
        "model_mode": model_profile["mode"],
        "target_column": model_profile["target_column"],
        "feature_profile": model_profile["feature_profile"],
        **build_model_label_metadata(model_profile),
        "symbols": list(args.symbols),
        "rows": int(len(dataset)),
        "task_type": f"binary_{model_profile['mode']}",
        "train_period": build_period_payload(dataset),
        "wfv_n_splits": args.n_splits,
        "wfv_purge_gap": args.purge_gap,
        "wfv_split_mode": args.split_mode,
        "wfv_monthly_train_months": args.monthly_train_months,
        "wfv_monthly_test_months": args.monthly_test_months,
        "wfv_monthly_window_mode": args.monthly_window_mode,
        "feature_clip": {
            "enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)),
            "upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)),
            "bounds": clip_bounds,
        },
        "feature_formulas_artifact": feature_formulas_path.name,
    }
    feature_formulas_payload = build_feature_formulas_payload(
        feature_columns=feature_columns,
        model_name=args.model_name,
        symbols=args.symbols,
        experiment_snapshot=experiment_snapshot,
    )

    backup_existing_artifacts(artifact_paths, args.model_name)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    feature_formulas_path.write_text(json.dumps(feature_formulas_payload, indent=2), encoding="utf-8")
    top_feature_importance(model, feature_columns).to_csv(importance_path, index=False)
    if fold_importance is not None and not fold_importance.empty:
        fold_importance.to_csv(fold_importance_path, index=False)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved feature metadata to %s", features_path)
    logger.info("Saved feature formulas to %s", feature_formulas_path)
    logger.info("Saved feature importance to %s", importance_path)
    if fold_importance is not None and not fold_importance.empty:
        logger.info("Saved fold feature importance to %s", fold_importance_path)


def backup_existing_artifacts(paths, model_name):
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = cfg.MODELS_DIR / "backups" / f"{model_name}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing_paths:
        shutil.copy2(path, backup_dir / path.name)

    logger.info(
        "Backed up %s existing model artifacts to %s",
        len(existing_paths),
        backup_dir,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    try:
        args = parse_args()
        model_profile = resolve_model_profile(args.model_profile)
        if args.model_name is None:
            args.model_name = model_profile["artifact_name"]
        experiment_snapshot = build_experiment_snapshot(model_profile)
        dataset = load_training_frame(args.db_path, args.symbols, model_profile)
        feature_columns = select_feature_columns(
            dataset,
            model_profile["feature_profile"],
        )

        logger.info("Loaded %s rows with %s features", len(dataset), len(feature_columns))
        logger.info("Using symbols: %s", ", ".join(args.symbols))
        logger.info(
            "Model profile=%s | mode=%s | target=%s | artifact=%s",
            model_profile["name"],
            model_profile["mode"],
            model_profile["target_column"],
            args.model_name,
        )
        logger.info(
            "Experiment=%s | labeling_profile=%s | training_profile=%s",
            experiment_snapshot["experiment"],
            experiment_snapshot["labeling_profile"],
            experiment_snapshot["training_profile"],
        )
        logger.info(
            "Labeling config: horizon=%s | dynamic_barriers=%s | stop[min=%.4f max=%.4f] | tp/sl=%.2f",
            experiment_snapshot["labeling"]["horizon"],
            experiment_snapshot["labeling"]["use_dynamic_barriers"],
            experiment_snapshot["labeling"]["barrier_min_pct"],
            experiment_snapshot["labeling"]["barrier_max_pct"],
            experiment_snapshot["labeling"]["barrier_tp_to_sl_ratio"],
        )
        logger.info(
            "Training config: feature_profile=%s | clip=%s [%.2f%%, %.2f%%]",
            experiment_snapshot["training"]["feature_profile"],
            experiment_snapshot["training"]["feature_clip_enabled"],
            experiment_snapshot["training"]["feature_clip_lower_q"] * 100,
            experiment_snapshot["training"]["feature_clip_upper_q"] * 100,
        )
        if model_profile["mode"] == "dual":
            logger.info(
                "Dual dataset: excluded %s rows with Target=0 before split",
                int(dataset.attrs.get("excluded_non_directional_rows", 0)),
            )
        else:
            logger.info(
                "%s dataset: retained all rows; positive rate=%.4f",
                model_profile["mode"].capitalize(),
                float(dataset[TARGET_COLUMN].mean()),
            )
        timeline_profile = dataset.attrs.get("all_timestamps_profile", {})
        logger.info(
            "Dataset timeline | all_timestamps=%s [%s -> %s] | rows before label filter=%s | directional=%s",
            int(timeline_profile.get("count", 0)),
            timeline_profile.get("first"),
            timeline_profile.get("last"),
            int(dataset.attrs.get("required_non_null_rows", len(dataset))),
            len(dataset),
        )
        feature_rows_by_symbol = dataset.attrs.get("feature_table_row_counts_by_symbol", {})
        if feature_rows_by_symbol:
            logger.info(
                "Feature table rows by symbol: %s",
                ", ".join(
                    f"{symbol}={profile.get('rows', 0)}"
                    for symbol, profile in sorted(feature_rows_by_symbol.items())
                ),
            )

        # ── Step 1: Walk-Forward Validation → honest OOS metrics ──────────
        oos_metrics, fold_details, median_best_iter, fold_importance = walk_forward_validation(
            dataset=dataset,
            feature_columns=feature_columns,
            seed=args.seed,
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
            split_mode=args.split_mode,
            monthly_train_months=args.monthly_train_months,
            monthly_test_months=args.monthly_test_months,
            monthly_window_mode=args.monthly_window_mode,
        )
        dataset_diagnostics = build_dataset_diagnostics(dataset, fold_details)

        # ── Step 2: Train production model on 100% of data ───────────────
        #    Clip bounds are computed on the FULL dataset because there is
        #    no hold-out anymore — this model sees everything we have.
        prod_clip_bounds = build_feature_clip_bounds(dataset, feature_columns)
        clipped_dataset = apply_feature_clip_bounds(dataset, prod_clip_bounds)

        if prod_clip_bounds:
            logger.info(
                "Feature clipping (production): %s numeric columns clipped to [%.2f%%, %.2f%%] quantiles",
                len(prod_clip_bounds),
                float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)) * 100,
                float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)) * 100,
            )

        prod_model = train_production_model(
            dataset=clipped_dataset,
            feature_columns=feature_columns,
            seed=args.seed,
            n_estimators=median_best_iter,
        )

        # ── Step 3: Assemble final metrics payload & persist ─────────────
        metrics = {
            "model_profile": model_profile["name"],
            "model_mode": model_profile["mode"],
            "source_target_column": model_profile["target_column"],
            "feature_profile": model_profile["feature_profile"],
            "oos_metrics": oos_metrics,
            "fold_details": fold_details,
            "fold_stability": build_fold_stability_payload(fold_details),
            "fold_feature_importance_top": build_fold_importance_summary(fold_importance, limit=20),
            "median_best_iteration": median_best_iter,
            "total_rows": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
            "n_splits": args.n_splits,
            "purge_gap": args.purge_gap,
            "split_mode": args.split_mode,
            "monthly_train_months": args.monthly_train_months,
            "monthly_test_months": args.monthly_test_months,
            "monthly_window_mode": args.monthly_window_mode,
            "excluded_non_directional_rows": int(dataset.attrs.get("excluded_non_directional_rows", 0)),
            "all_timestamps_count": dataset_diagnostics["all_timestamps_count"],
            "first_all_timestamp": dataset_diagnostics["first_all_timestamp"],
            "last_all_timestamp": dataset_diagnostics["last_all_timestamp"],
            "directional_rows_after_filter": dataset_diagnostics["directional_rows_after_filter"],
            "fold_boundary_timestamps": dataset_diagnostics["fold_boundary_timestamps"],
            "feature_table_row_counts_by_symbol": dataset_diagnostics["feature_table_row_counts_by_symbol"],
            "dataset_diagnostics": dataset_diagnostics,
            "experiment": experiment_snapshot,
            "dataset_period": build_period_payload(dataset),
        }

        logger.info(
            "OOS metrics | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | "
            "roc_auc=%s | pr_auc=%s | mcc=%.4f",
            oos_metrics["accuracy"],
            oos_metrics["balanced_accuracy"],
            oos_metrics["f1_macro"],
            format_metric_for_log(oos_metrics["roc_auc"]),
            format_metric_for_log(oos_metrics["pr_auc"]),
            oos_metrics["mcc"],
        )

        log_feature_importance_ranking(prod_model, feature_columns)
        save_directional_artifacts(
            prod_model,
            metrics,
            dataset,
            feature_columns,
            prod_clip_bounds,
            fold_importance,
            args,
            experiment_snapshot,
        )
        history_path = get_train_history_path(args.model_name)
        history_entry = build_train_history_entry(args, metrics, experiment_snapshot)
        history = save_train_history(history_path, history_entry)
        logger.info("Saved train history to %s", history_path)
        log_train_history_summary(history_entry, history, limit=10)

    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
