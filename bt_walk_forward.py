import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import bt
import config as cfg
import train
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run portfolio backtest using only walk-forward out-of-sample predictions."
    )
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Symbols to load, for example ETH/USDT SOL/USDT.",
    )
    parser.add_argument(
        "--model-profiles",
        nargs="+",
        choices=sorted(cfg.MODEL_PROFILES),
        default=[cfg.ACTIVE_MODEL_PROFILE],
        help=(
            "One model profile, or both long_v1 short_v1. "
            "Examples: dual_v1 | long_v1 | short_v1 | long_v1 short_v1."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument(
        "--split-mode",
        choices=["tscv", "monthly"],
        default="tscv",
        help="Walk-forward split mode: tscv uses TimeSeriesSplit, monthly uses 1-month rolling OOS tests.",
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
            "Purge gap in timestamps between train and test folds (default: "
            "cfg.effective_max_label_horizon() for triple-barrier y safety)."
        ),
    )
    parser.add_argument(
        "--predictions-name",
        default=None,
        help="Base filename for saved OOS predictions. Defaults to a profile-specific name.",
    )
    return parser.parse_args()


def resolve_model_profiles(profile_names: list[str]) -> list[dict]:
    if len(set(profile_names)) != len(profile_names):
        raise ValueError("--model-profiles must not contain duplicates.")
    names = list(dict.fromkeys(profile_names))
    profiles = [{"name": name, **cfg.get_model_profile(name)} for name in names]
    modes = {profile["mode"] for profile in profiles}

    valid = len(profiles) == 1 or (len(profiles) == 2 and modes == {"long", "short"})
    if not valid:
        raise ValueError(
            "--model-profiles accepts one profile, or exactly long_v1 and short_v1 together."
        )
    profiles.sort(key=lambda profile: {"dual": 0, "long": 1, "short": 2}[profile["mode"]])
    return profiles


def default_predictions_name(model_profiles: list[dict]) -> str:
    if [profile["name"] for profile in model_profiles] == ["dual_v1"]:
        return "walk_forward_oos_predictions"
    suffix = "_".join(profile["name"] for profile in model_profiles)
    return f"walk_forward_oos_predictions_{suffix}"


def apply_end_date_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return frame

    end_cutoff = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_cutoff):
        return frame

    return frame.loc[frame[train.TIMESTAMP_COLUMN] <= end_cutoff].copy()


def load_candidate_and_training_frames(
    db_path: str,
    symbols: list[str],
    model_profiles: list[dict],
):
    repository = HistoricalKlineRepository(db_path=db_path)
    frame = repository.load_feature_dataset(symbols)
    required_targets = [profile["target_column"] for profile in model_profiles]
    missing_targets = sorted(set(required_targets) - set(frame.columns))
    if missing_targets:
        raise RuntimeError(
            f"Dataset is missing target columns {missing_targets}. Re-run etl.py."
        )

    frame = frame.dropna(subset=[train.TIMESTAMP_COLUMN])
    frame = frame.sort_values(train.TIMESTAMP_COLUMN).reset_index(drop=True)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame = apply_end_date_cutoff(frame)

    candidate_frame = frame.copy()
    symbol_categories = list(dict.fromkeys(symbols))
    candidate_frame[train.SYMBOL_COLUMN] = pd.Categorical(
        candidate_frame[train.SYMBOL_COLUMN],
        categories=symbol_categories,
    )

    profile_contexts = {}
    for profile in model_profiles:
        target_column = profile["target_column"]
        training_frame = candidate_frame.dropna(subset=[target_column]).copy()
        labels = training_frame[target_column].astype(int)

        if profile["mode"] == "dual":
            unknown = sorted(set(labels.unique()) - {-1, 0, 1})
            if unknown:
                raise ValueError(f"Unexpected labels in {target_column}: {unknown}")
            training_frame = training_frame.loc[labels != 0].copy()
            training_frame[train.TARGET_COLUMN] = (
                training_frame[target_column].astype(int).map(train.LABEL_TO_CLASS)
            )
        else:
            unknown = sorted(set(labels.unique()) - {0, 1})
            if unknown:
                raise ValueError(f"Unexpected labels in {target_column}: {unknown}")
            training_frame[train.TARGET_COLUMN] = labels

        feature_columns = train.select_feature_columns(
            training_frame,
            profile["feature_profile"],
        )
        profile_contexts[profile["name"]] = {
            "profile": profile,
            "training_frame": training_frame,
            "feature_columns": feature_columns,
        }

    return frame, candidate_frame, profile_contexts


def fit_fold_model(train_df: pd.DataFrame, feature_columns: list[str], seed: int):
    clip_bounds = train.build_feature_clip_bounds(train_df, feature_columns)
    clipped_train = train.apply_feature_clip_bounds(train_df, clip_bounds)

    x_train = clipped_train[feature_columns]
    y_train = clipped_train[train.TARGET_COLUMN]
    w_train = train.compute_sample_weights(clipped_train)

    if len(x_train) >= 20:
        model, fit_metadata = train.fit_model_with_internal_eval(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            feature_columns=feature_columns,
            seed=seed,
        )
        return model, clip_bounds, fit_metadata

    model = train.build_model(seed=seed)
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        categorical_feature=[train.SYMBOL_COLUMN] if train.SYMBOL_COLUMN in feature_columns else "auto",
    )
    fit_metadata = {
        "internal_eval_size": 0,
        "eval_plan": None,
        "internal_eval_slices": None,
        "best_iter": int(getattr(model, "n_estimators_", 0) or getattr(model, "n_estimators", 0)),
        "fallback_used": False,
        "fallback_n_estimators": None,
    }
    return model, clip_bounds, fit_metadata


def build_profile_fold_predictions(
    profile_contexts: dict[str, dict],
    candidate_frame: pd.DataFrame,
    train_timestamps,
    test_timestamps,
    fold_idx: int,
    purge_gap: int,
    seed: int,
) -> tuple[pd.DataFrame | None, dict]:
    if purge_gap > 0 and len(train_timestamps) > purge_gap:
        train_timestamps = train_timestamps[:-purge_gap]

    train_timestamp_set = set(train_timestamps)
    test_timestamp_set = set(test_timestamps)
    combined = None
    model_details = {}

    for profile_offset, (profile_name, context) in enumerate(profile_contexts.items()):
        profile = context["profile"]
        feature_columns = context["feature_columns"]
        train_df = context["training_frame"].loc[
            context["training_frame"][train.TIMESTAMP_COLUMN].isin(train_timestamp_set)
        ].copy()
        test_df = candidate_frame.loc[
            candidate_frame[train.TIMESTAMP_COLUMN].isin(test_timestamp_set)
        ].copy()

        train_classes = sorted(train_df[train.TARGET_COLUMN].dropna().unique().tolist())
        if train_df.empty or test_df.empty or len(train_classes) < 2:
            model_details[profile_name] = {
                "train_rows": int(len(train_df)),
                "prediction_rows": 0,
                "classes": train_classes,
                "skipped": True,
            }
            return None, model_details

        model, clip_bounds, fit_metadata = fit_fold_model(
            train_df,
            feature_columns,
            seed + fold_idx + profile_offset,
        )
        clipped_test = train.apply_feature_clip_bounds(test_df, clip_bounds)
        clipped_test = clipped_test.dropna(subset=feature_columns)
        if clipped_test.empty:
            model_details[profile_name] = {
                "train_rows": int(len(train_df)),
                "prediction_rows": 0,
                "classes": train_classes,
                "skipped": True,
            }
            return None, model_details

        proba = model.predict_proba(clipped_test[feature_columns])
        profile_predictions = pd.DataFrame(
            {
                "timestamp": clipped_test[train.TIMESTAMP_COLUMN].values,
                "symbol": clipped_test[train.SYMBOL_COLUMN].astype(str).values,
            }
        )
        if profile["mode"] == "dual":
            profile_predictions["p_short"] = proba[:, 0]
            profile_predictions["p_long"] = proba[:, 1]
        else:
            profile_predictions[f"p_{profile['mode']}"] = proba[:, 1]

        if combined is None:
            combined = profile_predictions
        else:
            combined = combined.merge(
                profile_predictions,
                on=["timestamp", "symbol"],
                how="inner",
                validate="one_to_one",
            )

        model_details[profile_name] = {
            "mode": profile["mode"],
            "target_column": profile["target_column"],
            "feature_profile": profile["feature_profile"],
            "feature_count": int(len(feature_columns)),
            "train_rows": int(len(train_df)),
            "prediction_rows": int(len(profile_predictions)),
            "classes": train_classes,
            "best_iteration": int(fit_metadata["best_iter"]),
            "internal_eval": {
                "rows": int(fit_metadata["internal_eval_size"]),
                "eval_plan": fit_metadata["eval_plan"],
                "slices": fit_metadata["internal_eval_slices"],
            },
            "fallback_used": bool(fit_metadata["fallback_used"]),
            "fallback_n_estimators": fit_metadata["fallback_n_estimators"],
            "skipped": False,
        }

    if combined is None or combined.empty:
        return None, model_details

    if "p_long" not in combined:
        combined["p_long"] = 0.0
    if "p_short" not in combined:
        combined["p_short"] = 0.0
    combined["fold"] = fold_idx
    return combined[["timestamp", "symbol", "p_short", "p_long", "fold"]], model_details


def build_walk_forward_predictions(
    full_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    profile_contexts: dict[str, dict],
    n_splits: int,
    purge_gap: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    unique_ts = np.sort(full_frame[train.TIMESTAMP_COLUMN].dropna().unique())
    if len(unique_ts) < n_splits + 1:
        raise RuntimeError(
            f"Only {len(unique_ts)} unique timestamps, need at least {n_splits + 1} for {n_splits} folds."
        )

    predictions = []
    fold_details = []
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(splitter.split(unique_ts), start=1):
        train_timestamps = unique_ts[train_ts_idx]
        test_timestamps = unique_ts[test_ts_idx]
        effective_train_timestamps = train_timestamps
        if purge_gap > 0 and len(effective_train_timestamps) > purge_gap:
            effective_train_timestamps = effective_train_timestamps[:-purge_gap]
        fold_predictions, model_details = build_profile_fold_predictions(
            profile_contexts=profile_contexts,
            candidate_frame=candidate_frame,
            train_timestamps=effective_train_timestamps,
            test_timestamps=test_timestamps,
            fold_idx=fold_idx,
            purge_gap=0,
            seed=seed,
        )
        if fold_predictions is None:
            print(f"Fold {fold_idx}: skipped ({model_details})")
            continue
        predictions.append(fold_predictions)

        fold_info = {
            "fold": fold_idx,
            "prediction_rows": int(len(fold_predictions)),
            "purged_timestamps": int(len(train_timestamps) - len(effective_train_timestamps)),
            "train_start": str(pd.to_datetime(effective_train_timestamps).min()),
            "train_end": str(pd.to_datetime(effective_train_timestamps).max()),
            "test_start": str(pd.to_datetime(fold_predictions["timestamp"]).min()),
            "test_end": str(pd.to_datetime(fold_predictions["timestamp"]).max()),
            "models": model_details,
        }
        fold_details.append(fold_info)
        print(
            f"Fold {fold_idx}/{n_splits}: profiles={','.join(profile_contexts)} "
            f"pred={fold_info['prediction_rows']} "
            f"[{fold_info['test_start']} -> {fold_info['test_end']}]"
        )

    if not predictions:
        raise RuntimeError("All walk-forward folds were skipped.")

    return pd.concat(predictions, ignore_index=True), fold_details


def iter_monthly_splits(unique_ts, train_months: int, test_months: int, window_mode: str = "expanding"):
    timestamps = pd.Series(pd.to_datetime(unique_ts)).dropna().sort_values()
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


def build_monthly_walk_forward_predictions(
    full_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    profile_contexts: dict[str, dict],
    train_months: int,
    test_months: int,
    window_mode: str,
    purge_gap: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if train_months <= 0 or test_months <= 0:
        raise ValueError("monthly train/test windows must be positive month counts.")

    unique_ts = np.sort(full_frame[train.TIMESTAMP_COLUMN].dropna().unique())
    predictions = []
    fold_details = []
    splits = list(iter_monthly_splits(unique_ts, train_months, test_months, window_mode=window_mode))

    for fold_idx, train_timestamps, test_timestamps in splits:
        original_train_timestamps = train_timestamps
        effective_train_timestamps = train_timestamps
        if purge_gap > 0 and len(effective_train_timestamps) > purge_gap:
            effective_train_timestamps = effective_train_timestamps[:-purge_gap]
        fold_predictions, model_details = build_profile_fold_predictions(
            profile_contexts=profile_contexts,
            candidate_frame=candidate_frame,
            train_timestamps=effective_train_timestamps,
            test_timestamps=test_timestamps,
            fold_idx=fold_idx,
            purge_gap=0,
            seed=seed,
        )
        if fold_predictions is None:
            print(f"Fold {fold_idx}: skipped ({model_details})")
            continue
        predictions.append(fold_predictions)

        fold_info = {
            "fold": fold_idx,
            "split_mode": "monthly",
            "monthly_window_mode": window_mode,
            "prediction_rows": int(len(fold_predictions)),
            "purged_timestamps": int(len(train_timestamps) - len(effective_train_timestamps)),
            "train_start": str(pd.to_datetime(effective_train_timestamps).min()),
            "train_end": str(pd.to_datetime(effective_train_timestamps).max()),
            "train_end_before_purge": str(pd.to_datetime(original_train_timestamps[-1])),
            "test_start": str(pd.to_datetime(fold_predictions["timestamp"]).min()),
            "test_end": str(pd.to_datetime(fold_predictions["timestamp"]).max()),
            "models": model_details,
        }
        fold_details.append(fold_info)
        print(
            f"Fold {fold_idx}/{len(splits)} monthly: profiles={','.join(profile_contexts)} "
            f"pred={fold_info['prediction_rows']} "
            f"[{fold_info['test_start']} -> {fold_info['test_end']}]"
        )

    if not predictions:
        raise RuntimeError("All monthly walk-forward folds were skipped.")

    return pd.concat(predictions, ignore_index=True), fold_details


def build_features_meta(
    predictions: pd.DataFrame,
    profile_contexts: dict[str, dict],
    symbols: list[str],
    args,
):
    profiles = [context["profile"] for context in profile_contexts.values()]
    feature_columns_by_profile = {
        name: context["feature_columns"]
        for name, context in profile_contexts.items()
    }
    feature_columns = sorted(
        {
            column
            for columns in feature_columns_by_profile.values()
            for column in columns
        }
    )
    model_mode = profiles[0]["mode"] if len(profiles) == 1 else "long_short"
    profile_modes = {profile["mode"] for profile in profiles}
    probability_semantics = {
        "p_long": (
            "P(Target=1)" if model_mode == "dual"
            else "P(TargetLong=1)" if "long" in profile_modes
            else "unavailable; constant 0"
        ),
        "p_short": (
            "P(Target=-1)" if model_mode == "dual"
            else "P(TargetShort=1)" if "short" in profile_modes
            else "unavailable; constant 0"
        ),
    }
    metadata = {
        "feature_columns": feature_columns,
        "feature_columns_by_profile": feature_columns_by_profile,
        "model_profiles": [profile["name"] for profile in profiles],
        "model_mode": model_mode,
        "target_columns": {
            profile["name"]: profile["target_column"]
            for profile in profiles
        },
        "feature_profiles": {
            profile["name"]: profile["feature_profile"]
            for profile in profiles
        },
        "probability_semantics": probability_semantics,
        "symbols": list(symbols),
        "rows": int(len(predictions)),
        "task_type": f"binary_{model_mode}_walk_forward_oos",
        "train_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "wfv_n_splits": int(args.n_splits),
        "wfv_purge_gap": int(args.purge_gap),
        "wfv_split_mode": str(args.split_mode),
        "wfv_monthly_train_months": int(args.monthly_train_months),
        "wfv_monthly_test_months": int(args.monthly_test_months),
        "wfv_monthly_window_mode": str(args.monthly_window_mode),
        "feature_clip": {
            "enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)),
            "upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)),
            "bounds": {},
            "note": "Fold-specific clipping was applied before creating OOS predictions.",
        },
    }
    if len(profiles) == 1:
        profile = profiles[0]
        metadata.update(
            {
                "model_profile": profile["name"],
                "target_column": profile["target_column"],
                "feature_profile": profile["feature_profile"],
            }
        )
        metadata.update(train.build_model_label_metadata(profile))
    return metadata


def save_walk_forward_payload(predictions: pd.DataFrame, fold_details: list[dict], args) -> None:
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    predictions_path = cfg.MODELS_DIR / f"{args.predictions_name}.csv"
    summary_path = cfg.MODELS_DIR / f"{args.predictions_name}_summary.json"

    predictions.to_csv(predictions_path, index=False)
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_profiles": list(args.model_profiles),
        "symbols": list(args.symbols),
        "n_splits": int(args.n_splits),
        "purge_gap": int(args.purge_gap),
        "split_mode": str(args.split_mode),
        "monthly_train_months": int(args.monthly_train_months),
        "monthly_test_months": int(args.monthly_test_months),
        "monthly_window_mode": str(args.monthly_window_mode),
        "prediction_rows": int(len(predictions)),
        "prediction_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "fold_details": fold_details,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved OOS predictions: {predictions_path}")
    print(f"Saved WFV summary: {summary_path}")


def main():
    args = parse_args()
    model_profiles = resolve_model_profiles(args.model_profiles)
    args.model_profiles = [profile["name"] for profile in model_profiles]
    if args.predictions_name is None:
        args.predictions_name = default_predictions_name(model_profiles)

    full_frame, candidate_frame, profile_contexts = load_candidate_and_training_frames(
        args.db_path,
        args.symbols,
        model_profiles,
    )
    profile_summary = ", ".join(
        f"{name}: rows={len(context['training_frame'])}, features={len(context['feature_columns'])}"
        for name, context in profile_contexts.items()
    )
    print(
        f"Loaded candidate rows={len(candidate_frame)} | {profile_summary}"
    )

    if args.split_mode == "monthly":
        predictions, fold_details = build_monthly_walk_forward_predictions(
            full_frame=full_frame,
            candidate_frame=candidate_frame,
            profile_contexts=profile_contexts,
            train_months=args.monthly_train_months,
            test_months=args.monthly_test_months,
            window_mode=args.monthly_window_mode,
            purge_gap=args.purge_gap,
            seed=args.seed,
        )
    else:
        predictions, fold_details = build_walk_forward_predictions(
            full_frame=full_frame,
            candidate_frame=candidate_frame,
            profile_contexts=profile_contexts,
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
            seed=args.seed,
        )
    save_walk_forward_payload(predictions, fold_details, args)

    features_meta = build_features_meta(
        predictions=predictions,
        profile_contexts=profile_contexts,
        symbols=args.symbols,
        args=args,
    )
    profile_suffix = "_".join(args.model_profiles)
    chart_name = (
        "equity_curve_walk_forward.png"
        if args.model_profiles == ["dual_v1"]
        else f"equity_curve_walk_forward_{profile_suffix}.png"
    )
    chart_path = cfg.BACKTEST_CHARTS_DIR / chart_name
    bt.backtest(
        features_meta=features_meta,
        predictions=predictions,
        equity_curve_path=chart_path,
        result_title="WALK-FORWARD OOS BACKTEST",
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
