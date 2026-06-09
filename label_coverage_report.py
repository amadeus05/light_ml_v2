import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import train
from label_coverage_report_template import LABEL_COVERAGE_HTML_TEMPLATE
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


TARGET_VALUES = (-1, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report Target coverage produced by the current ETL labeling settings."
    )
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument("--symbols", nargs="+", default=cfg.SYMBOLS, help="Symbols to inspect.")
    parser.add_argument(
        "--timeframe-profile",
        choices=sorted(cfg.TIMEFRAME_PROFILES),
        default=cfg.ACTIVE_TIMEFRAME_PROFILE,
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=default_sequence_lengths(),
        help="Sequence lengths to estimate eligible directional samples for LSTM-style datasets.",
    )
    parser.add_argument(
        "--min-directional-rows",
        type=int,
        default=200,
        help="Minimum directional rows per symbol to mark as enough for training.",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Also print monthly target coverage.",
    )
    parser.add_argument(
        "--stability-low-ratio",
        type=float,
        default=0.5,
        help="Warn when a month's directional coverage is below this fraction of the monthly mean.",
    )
    parser.add_argument(
        "--html-output",
        default=str(cfg.MODELS_DIR / "label_coverage_report.html"),
        help="Path to write the HTML report.",
    )
    return parser.parse_args()


def default_sequence_lengths() -> list[int]:
    lengths = []
    try:
        from lstm import config_lstm

        lengths.append(int(config_lstm.SEQUENCE_LENGTH))
    except Exception:
        pass
    try:
        from lstm_candles import config_lstm_candles

        lengths.append(int(config_lstm_candles.SEQUENCE_LENGTH))
    except Exception:
        pass
    return sorted(set(length for length in lengths if length > 0)) or [48, 64]


def load_target_frame(
    db_path: str,
    symbols: list[str],
    timeframe_profile: str = cfg.ACTIVE_TIMEFRAME_PROFILE,
) -> pd.DataFrame:
    repository = HistoricalKlineRepository(db_path=db_path)
    timeframe = cfg.get_timeframe_profile(timeframe_profile)["timeframe"]
    frame = repository.load_feature_dataset(symbols, timeframe=timeframe)
    frame = frame.dropna(subset=[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN, train.TARGET_COLUMN]).copy()
    frame[train.TIMESTAMP_COLUMN] = pd.to_datetime(frame[train.TIMESTAMP_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[train.TIMESTAMP_COLUMN])

    end_cutoff = train.get_end_date_cutoff()
    if end_cutoff is not None and not pd.isna(end_cutoff):
        frame = frame.loc[frame[train.TIMESTAMP_COLUMN] <= end_cutoff].copy()

    frame[train.TARGET_COLUMN] = frame[train.TARGET_COLUMN].astype(int)
    unknown = sorted(set(frame[train.TARGET_COLUMN].unique()) - set(TARGET_VALUES))
    if unknown:
        raise ValueError(f"Unexpected Target values: {unknown}")
    return frame.sort_values([train.SYMBOL_COLUMN, train.TIMESTAMP_COLUMN]).reset_index(drop=True)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_number(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):,.4f}"
        return "inf"
    return str(value)


def build_target_summary(frame: pd.DataFrame, group_columns: Iterable[str] = ()) -> pd.DataFrame:
    group_columns = list(group_columns)
    if group_columns:
        grouped = frame.groupby(group_columns, observed=True, dropna=False)
    else:
        grouped = [((), frame)]

    rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        counts = group[train.TARGET_COLUMN].value_counts().to_dict()
        short_rows = int(counts.get(-1, 0))
        neutral_rows = int(counts.get(0, 0))
        long_rows = int(counts.get(1, 0))
        total_rows = int(len(group))
        directional_rows = short_rows + long_rows
        row = {
            "rows": total_rows,
            "short": short_rows,
            "neutral": neutral_rows,
            "long": long_rows,
            "directional": directional_rows,
            "directional_rate": directional_rows / total_rows if total_rows else 0.0,
            "short_rate_total": short_rows / total_rows if total_rows else 0.0,
            "long_rate_total": long_rows / total_rows if total_rows else 0.0,
            "short_share_directional": short_rows / directional_rows if directional_rows else 0.0,
            "long_share_directional": long_rows / directional_rows if directional_rows else 0.0,
        }
        for column, value in zip(group_columns, key, strict=False):
            row[column] = value
        rows.append(row)

    columns = group_columns + [
        "rows",
        "short",
        "neutral",
        "long",
        "directional",
        "directional_rate",
        "short_share_directional",
        "long_share_directional",
    ]
    return pd.DataFrame(rows)[columns]


def estimate_sequence_samples(frame: pd.DataFrame, sequence_lengths: list[int]) -> pd.DataFrame:
    rows = []
    for symbol, symbol_frame in frame.groupby(train.SYMBOL_COLUMN, observed=True):
        symbol_frame = symbol_frame.sort_values(train.TIMESTAMP_COLUMN).reset_index(drop=True)
        directional_mask = symbol_frame[train.TARGET_COLUMN].astype(int) != 0
        directional_positions = np.flatnonzero(directional_mask.to_numpy())
        row = {
            train.SYMBOL_COLUMN: str(symbol),
            "rows": int(len(symbol_frame)),
            "directional": int(directional_mask.sum()),
        }
        for length in sequence_lengths:
            length = int(length)
            eligible = int((directional_positions >= max(0, length - 1)).sum())
            row[f"seq_{length}_eligible"] = eligible
        rows.append(row)
    return pd.DataFrame(rows)


def print_config_snapshot() -> None:
    print("Labeling config:")
    print(f"  HORIZON={int(getattr(cfg, 'HORIZON', 0))}")
    print(f"  ENABLE_ADAPTIVE_HORIZON={bool(getattr(cfg, 'ENABLE_ADAPTIVE_HORIZON', False))}")
    if bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)):
        print(
            "  ADAPTIVE_HORIZON="
            f"{int(getattr(cfg, 'ADAPTIVE_HORIZON_MIN', 0))}"
            f"..{int(getattr(cfg, 'ADAPTIVE_HORIZON_MAX', 0))}"
        )
    print(f"  USE_DYNAMIC_BARRIERS={bool(getattr(cfg, 'USE_DYNAMIC_BARRIERS', False))}")
    if bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)):
        print(
            "  Dynamic stop="
            f"ATRx{float(getattr(cfg, 'BARRIER_ATR_MULTIPLIER', 0.0)):.2f}, "
            f"RVOLx{float(getattr(cfg, 'BARRIER_RVOL_MULTIPLIER', 0.0)):.2f}, "
            f"min={float(getattr(cfg, 'BARRIER_MIN_PCT', 0.0)):.4f}, "
            f"max={float(getattr(cfg, 'BARRIER_MAX_PCT', 0.0)):.4f}"
        )
        print(f"  TP/SL ratio={float(getattr(cfg, 'BARRIER_TP_TO_SL_RATIO', 0.0)):.2f}")
    else:
        print(f"  TP_PCT={float(getattr(cfg, 'TP_PCT', 0.0)):.4f}")
        print(f"  SL_PCT={float(getattr(cfg, 'SL_PCT', 0.0)):.4f}")
    print(f"  TAKER_COM={float(getattr(cfg, 'TAKER_COM', 0.0)):.6f}")
    print(f"  SLIPPAGE={float(getattr(cfg, 'SLIPPAGE', 0.0)):.6f}")
    print()


def print_summary_table(title: str, summary: pd.DataFrame) -> None:
    printable = summary.copy()
    for column in ("directional_rate", "short_share_directional", "long_share_directional"):
        if column in printable.columns:
            printable[column] = printable[column].map(percent)
    print(title)
    print(printable.to_string(index=False))
    print()


def build_monthly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    monthly_frame = frame.copy()
    monthly_frame["month"] = monthly_frame[train.TIMESTAMP_COLUMN].dt.to_period("M").astype(str)
    return build_target_summary(monthly_frame, ["month"]).sort_values("month").reset_index(drop=True)


def build_monthly_stability(monthly: pd.DataFrame, low_ratio: float) -> dict:
    rates = monthly["directional_rate"].astype(float)
    rows = monthly["rows"].astype(int)
    directional = monthly["directional"].astype(int)
    mean_rate = float(rates.mean()) if len(rates) else 0.0
    std_rate = float(rates.std(ddof=0)) if len(rates) else 0.0
    min_rate = float(rates.min()) if len(rates) else 0.0
    max_rate = float(rates.max()) if len(rates) else 0.0
    low_threshold = mean_rate * max(0.0, float(low_ratio))
    low_months = monthly.loc[rates < low_threshold, ["month", "rows", "directional", "directional_rate"]]
    weakest = monthly.nsmallest(min(5, len(monthly)), "directional_rate")[
        ["month", "rows", "directional", "directional_rate"]
    ]
    strongest = monthly.nlargest(min(5, len(monthly)), "directional_rate")[
        ["month", "rows", "directional", "directional_rate"]
    ]
    return {
        "months": int(len(monthly)),
        "total_rows": int(rows.sum()),
        "total_directional": int(directional.sum()),
        "mean_rate": mean_rate,
        "std_rate": std_rate,
        "cv": std_rate / mean_rate if mean_rate > 0 else 0.0,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "max_to_min": max_rate / min_rate if min_rate > 0 else float("inf"),
        "low_threshold": low_threshold,
        "low_months": low_months,
        "weakest": weakest,
        "strongest": strongest,
    }


def print_rate_table(title: str, frame: pd.DataFrame) -> None:
    printable = frame.copy()
    if "directional_rate" in printable.columns:
        printable["directional_rate"] = printable["directional_rate"].map(percent)
    print(title)
    if printable.empty:
        print("none")
    else:
        print(printable.to_string(index=False))
    print()


def print_monthly_stability(monthly: pd.DataFrame, low_ratio: float) -> None:
    stability = build_monthly_stability(monthly, low_ratio)
    print("Monthly coverage stability:")
    print(f"  months={stability['months']}")
    print(f"  total_rows={stability['total_rows']}")
    print(f"  total_directional={stability['total_directional']}")
    print(f"  mean_directional_rate={percent(stability['mean_rate'])}")
    print(f"  std_directional_rate={percent(stability['std_rate'])}")
    print(f"  coefficient_of_variation={stability['cv']:.3f}")
    print(f"  min_directional_rate={percent(stability['min_rate'])}")
    print(f"  max_directional_rate={percent(stability['max_rate'])}")
    max_to_min = stability["max_to_min"]
    max_to_min_text = "inf" if not np.isfinite(max_to_min) else f"{max_to_min:.2f}x"
    print(f"  max_to_min_rate={max_to_min_text}")
    print(f"  low_month_threshold={percent(stability['low_threshold'])}")
    print()

    print_rate_table("Weakest months by directional coverage:", stability["weakest"])
    print_rate_table("Strongest months by directional coverage:", stability["strongest"])
    print_rate_table("Months below stability threshold:", stability["low_months"])


def html_table(
    frame: pd.DataFrame,
    percent_columns: Iterable[str] = (),
    table_id: str | None = None,
    row_year_column: str | None = None,
) -> str:
    percent_columns = set(percent_columns)
    table_attr = f' id="{escape(table_id)}"' if table_id else ""
    header = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            if column in percent_columns:
                text = percent(float(value)) if pd.notna(value) else ""
            else:
                text = format_number(value)
            cells.append(f"<td>{escape(text)}</td>")
        row_attr = ""
        if row_year_column and row_year_column in row:
            row_attr = f' data-year="{escape(str(row[row_year_column])[:4])}"'
        rows.append(f"<tr{row_attr}>" + "".join(cells) + "</tr>")
    return f"<table{table_attr}><thead><tr>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def metric_card(label: str, value: str, hint: str = "", css_class: str = "") -> str:
    return (
        f'<div class="panel card">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value {escape(css_class)}">{escape(value)}</div>'
        f'<div class="hint">{escape(hint)}</div>'
        f"</div>"
    )


def config_item(label: str, value: str) -> str:
    return f"<div><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"


def target_distribution_svg(overall: pd.DataFrame) -> str:
    row = overall.iloc[0]
    values = [
        ("Short", int(row["short"]), "#fb7185"),
        ("Neutral", int(row["neutral"]), "#94a3b8"),
        ("Long", int(row["long"]), "#34d399"),
    ]
    total = max(sum(value for _, value, _ in values), 1)
    width = 680
    height = 300
    bar_area_width = 420
    x0 = 160
    y0 = 48
    gap = 30
    bar_height = 46
    items = [
        '<defs><filter id="barGlow" x="-20%" y="-80%" width="140%" height="260%">'
        '<feGaussianBlur stdDeviation="7" result="blur"/><feMerge><feMergeNode in="blur"/>'
        '<feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
    ]
    for idx, (label, value, color) in enumerate(values):
        y = y0 + idx * (bar_height + gap)
        bar_width = int(bar_area_width * value / total)
        share = value / total
        items.append(f'<text x="34" y="{y + 29}" fill="#cbd5e1" font-weight="700">{escape(label)}</text>')
        items.append(
            f'<rect x="{x0}" y="{y}" width="{bar_area_width}" height="{bar_height}" rx="13" '
            'fill="rgba(148,163,184,0.12)" />'
        )
        items.append(
            f'<rect x="{x0}" y="{y}" width="{bar_width}" height="{bar_height}" rx="13" '
            f'fill="{color}" filter="url(#barGlow)" />'
        )
        items.append(f'<text x="{x0 + bar_area_width + 18}" y="{y + 20}" fill="#f5f7fb" font-weight="800">{value:,}</text>')
        items.append(f'<text x="{x0 + bar_area_width + 18}" y="{y + 39}" fill="#94a3b8">{percent(share)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="300" role="img">{"".join(items)}</svg>'


def monthly_coverage_svg(monthly: pd.DataFrame, low_threshold: float) -> str:
    width = 980
    height = 360
    left = 64
    right = 28
    top = 34
    bottom = 66
    plot_w = width - left - right
    plot_h = height - top - bottom
    if monthly.empty:
        return f'<svg viewBox="0 0 {width} {height}" width="100%" height="360"></svg>'
    rates = monthly["directional_rate"].astype(float).to_numpy()
    max_rate = max(float(np.max(rates)), low_threshold, 0.01)
    axis_max = min(1.0, max_rate * 1.16)
    bar_w = max(5, plot_w / len(monthly) * 0.64)
    step = plot_w / len(monthly)
    threshold_y = top + plot_h - (low_threshold / axis_max) * plot_h
    items = [
        '<defs><linearGradient id="coverageFill" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0%" stop-color="#93c5fd"/><stop offset="100%" stop-color="#3b82f6"/></linearGradient></defs>',
    ]
    for tick in np.linspace(0, axis_max, 5):
        y = top + plot_h - (float(tick) / axis_max) * plot_h
        items.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="rgba(148,163,184,0.12)" />')
        items.append(f'<text x="12" y="{y + 4:.2f}">{percent(float(tick))}</text>')
    items.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="rgba(148,163,184,0.22)" />',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="rgba(148,163,184,0.22)" />',
            f'<line x1="{left}" y1="{threshold_y:.2f}" x2="{left + plot_w}" y2="{threshold_y:.2f}" stroke="#fbbf24" stroke-dasharray="7 7">'
            f'<title>Low-month threshold: {percent(low_threshold)}</title></line>',
            f'<rect x="{left + plot_w - 188}" y="10" width="178" height="24" rx="12" fill="rgba(251,191,36,0.10)" stroke="rgba(251,191,36,0.26)" />',
            f'<line x1="{left + plot_w - 176}" y1="22" x2="{left + plot_w - 144}" y2="22" stroke="#fbbf24" stroke-dasharray="6 5" />',
            f'<text x="{left + plot_w - 134}" y="26" fill="#fbbf24">low threshold {percent(low_threshold)}</text>',
        ]
    )
    for idx, row in monthly.reset_index(drop=True).iterrows():
        rate = float(row["directional_rate"])
        x = left + idx * step + (step - bar_w) / 2
        bar_h = (rate / axis_max) * plot_h if axis_max > 0 else 0
        y = top + plot_h - bar_h
        color = "#fb7185" if rate < low_threshold else "url(#coverageFill)"
        items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="6" fill="{color}">'
            f'<title>{escape(str(row["month"]))}: {percent(rate)} ({int(row["directional"]):,}/{int(row["rows"]):,})</title>'
            '</rect>'
        )
        if idx % max(1, len(monthly) // 12) == 0:
            label = str(row["month"])[2:]
            items.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{height - 28}" text-anchor="middle">{escape(label)}</text>'
            )
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="360" role="img">{"".join(items)}</svg>'


def build_config_items() -> str:
    items = [
        config_item("HORIZON", str(int(getattr(cfg, "HORIZON", 0)))),
        config_item("Adaptive horizon", str(bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)))),
        config_item("Dynamic barriers", str(bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)))),
        config_item("TP/SL ratio", f"{float(getattr(cfg, 'BARRIER_TP_TO_SL_RATIO', 0.0)):.2f}"),
        config_item("Barrier min", f"{float(getattr(cfg, 'BARRIER_MIN_PCT', 0.0)):.4f}"),
        config_item("Barrier max", f"{float(getattr(cfg, 'BARRIER_MAX_PCT', 0.0)):.4f}"),
        config_item("ATR multiplier", f"{float(getattr(cfg, 'BARRIER_ATR_MULTIPLIER', 0.0)):.2f}"),
        config_item("RVOL multiplier", f"{float(getattr(cfg, 'BARRIER_RVOL_MULTIPLIER', 0.0)):.2f}"),
        config_item("Taker commission", f"{float(getattr(cfg, 'TAKER_COM', 0.0)):.6f}"),
        config_item("Slippage", f"{float(getattr(cfg, 'SLIPPAGE', 0.0)):.6f}"),
        config_item("END_DATE", str(getattr(cfg, "END_DATE", None))),
    ]
    return "".join(items)


def build_year_filter_buttons(monthly: pd.DataFrame) -> str:
    years = sorted({str(month)[:4] for month in monthly["month"].astype(str).tolist()})
    buttons = ['<button class="filter-btn" type="button" data-year-filter="all">Все</button>']
    buttons.extend(
        f'<button class="filter-btn" type="button" data-year-filter="{escape(year)}">{escape(year)}</button>'
        for year in years
    )
    return "".join(buttons)


def render_html_report(
    output_path: str,
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    by_symbol: pd.DataFrame,
    sequence_samples: pd.DataFrame,
    monthly: pd.DataFrame,
    low_ratio: float,
) -> Path:
    stability = build_monthly_stability(monthly, low_ratio)
    overall_row = overall.iloc[0]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = "Label Coverage Report"
    subtitle = (
        f"Период: {frame[train.TIMESTAMP_COLUMN].min()} -> {frame[train.TIMESTAMP_COLUMN].max()} | "
        f"Символы: {', '.join(sorted(frame[train.SYMBOL_COLUMN].astype(str).unique()))}"
    )
    metric_cards = "".join(
        [
            metric_card("Всего строк", f"{int(overall_row['rows']):,}", "после ETL и END_DATE", "blue"),
            metric_card("Directional coverage", percent(float(overall_row["directional_rate"])), "Target != 0", "green"),
            metric_card("Long share", percent(float(overall_row["long_share_directional"])), "среди directional", "green"),
            metric_card("Short share", percent(float(overall_row["short_share_directional"])), "среди directional", "red"),
            metric_card("Monthly mean", percent(stability["mean_rate"]), "среднее покрытие", "blue"),
            metric_card("Monthly CV", f"{stability['cv']:.3f}", "ниже = стабильнее", "yellow"),
            metric_card("Max/min", "inf" if not np.isfinite(stability["max_to_min"]) else f"{stability['max_to_min']:.2f}x", "разброс месяцев", "purple"),
            metric_card("Low months", str(len(stability["low_months"])), f"ниже {percent(stability['low_threshold'])}", "red"),
        ]
    )
    stability_summary = (
        "<div class='config'>"
        + config_item("Месяцев", str(stability["months"]))
        + config_item("Всего directional", f"{stability['total_directional']:,}")
        + config_item("Std coverage", percent(stability["std_rate"]))
        + config_item("Min coverage", percent(stability["min_rate"]))
        + config_item("Max coverage", percent(stability["max_rate"]))
        + config_item("Low threshold", percent(stability["low_threshold"]))
        + "</div>"
    )
    percent_columns = ("directional_rate", "short_share_directional", "long_share_directional")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = LABEL_COVERAGE_HTML_TEMPLATE.format(
        title=escape(title),
        subtitle=escape(subtitle),
        generated_at=escape(generated_at),
        metric_cards=metric_cards,
        config_items=build_config_items(),
        target_distribution_chart=target_distribution_svg(overall),
        monthly_coverage_chart=monthly_coverage_svg(monthly, stability["low_threshold"]),
        stability_summary=stability_summary,
        weakest_months_table=html_table(stability["weakest"], percent_columns=("directional_rate",)),
        strongest_months_table=html_table(stability["strongest"], percent_columns=("directional_rate",)),
        by_symbol_table=html_table(by_symbol, percent_columns=percent_columns),
        sequence_samples_table=html_table(sequence_samples),
        monthly_year_filters=build_year_filter_buttons(monthly),
        monthly_table=html_table(
            monthly,
            percent_columns=percent_columns,
            table_id="monthly-coverage-table",
            row_year_column="month",
        ),
        interpretation_note=escape(
            "Смотри не только общий directional coverage, но и месячный CV/max-min. "
            "Если несколько месяцев сильно ниже среднего, labels зависят от режима рынка, "
            "и модель может переобучаться на редкие периоды с высокой торговой активностью."
        ),
    )
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    frame = load_target_frame(args.db_path, args.symbols, args.timeframe_profile)
    if frame.empty:
        raise RuntimeError("No labeled feature rows found. Run etl.py first.")

    print_config_snapshot()
    print(f"Loaded labeled rows: {len(frame)}")
    print(f"Period: {frame[train.TIMESTAMP_COLUMN].min()} -> {frame[train.TIMESTAMP_COLUMN].max()}")
    print(f"Symbols: {', '.join(sorted(frame[train.SYMBOL_COLUMN].astype(str).unique()))}")
    print()

    overall = build_target_summary(frame)
    by_symbol = build_target_summary(frame, [train.SYMBOL_COLUMN]).sort_values(train.SYMBOL_COLUMN)
    print_summary_table("Overall Target coverage:", overall)
    print_summary_table("Target coverage by symbol:", by_symbol)

    sequence_samples = estimate_sequence_samples(frame, args.sequence_lengths)
    for length in args.sequence_lengths:
        column = f"seq_{int(length)}_eligible"
        sequence_samples[f"{column}_ok"] = sequence_samples[column] >= int(args.min_directional_rows)
    print("Estimated directional sequence samples:")
    print(sequence_samples.to_string(index=False))
    print()

    monthly = build_monthly_frame(frame)
    print_monthly_stability(monthly, args.stability_low_ratio)

    if args.monthly:
        print_summary_table("Monthly Target coverage:", monthly)

    html_path = render_html_report(
        output_path=args.html_output,
        frame=frame,
        overall=overall,
        by_symbol=by_symbol,
        sequence_samples=sequence_samples,
        monthly=monthly,
        low_ratio=args.stability_low_ratio,
    )
    print(f"HTML report saved: {html_path}")


if __name__ == "__main__":
    main()
