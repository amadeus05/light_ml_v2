import os
from pathlib import Path


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


# --- BASE ---
DB_PATH = str(_env_str("DB_PATH", "./data/market_data.db"))
SYMBOLS = [
    "BTC/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "SOL/USDT",

    "XRP/USDT",
    "ADA/USDT",

    # "1000PEPE/USDT",
    # # "LTC/USDT",
    # "1000FLOKI/USDT",
    # "SHIB1000/USDT",


    # "WIF/USDT",
    # "1000000MOG/USDT",
    # "1000BONK/USDT",
    # "MEW/USDT",
    # "POPCAT/USDT",

    # "AVAX/USDT",
    # "DOT/USDT",

    # "DOGE/USDT",
]

TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"  # Старший таймфрейм для мульти-TF фичей

# --- DATA LOADING ---
ACTIVE_EXCHANGE = "bybit"  # bybit: bybit, binance
SIMULATION_BASE_EXCHANGE = "bybit"
SIMULATION_DB_PATH = DB_PATH
START_DATE = "2023-01-01"
END_DATE = "2026-03-27 21:00:00"
BYBIT_LIMIT = 1000
BYBIT_RETRY_SLEEP = 0.33
BINANCE_BASE_URL = "https://fapi.binance.com"
BINANCE_LIMIT = 1000
BINANCE_TIMEOUT = 20
BINANCE_MAX_WORKERS = 3
BINANCE_RETRY_COUNT = 5
BINANCE_RETRY_SLEEP = 0.25
BINANCE_REQUEST_WEIGHT_LIMIT_PER_MINUTE = 2400
BACKTEST_INITIAL_BALANCE = 100

ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99
USE_SYMBOL_FEATURE = False
ACTIVE_EXPERIMENT = "hybrid_v1_labels_v2_train"
LABELING_PROFILE = "v1"
TRAINING_PROFILE = "v2"
MANUAL_DISABLED_FEATURE_COLUMNS = [
    "return_1h_24",
    "return_4h_3",
    # Низкий importance (< 1000 gain), создают шум:
    "distance_to_session_low_1h",
    "slope_acceleration_1h_12_24",
    # "price_position_4h",
    # "price_position_1h",
    # "distance_to_support_1h",
    "range_compression_1h",
    "hour_cos_1h",
    # "distance_to_resistance_1h",
    "relative_strength_vs_btc_24h",
    "return_4h_1",
    "distance_to_session_high_1h",
    # "cross_sectional_rank_ema_fast_slow_1h",
    "return_1h_12",
    "return_1h_6",
    # "cross_sectional_rank_4h",
    "is_weekend_1h",
    "crowded_longs_score_1h", #$
    "crowded_shorts_score_1h", #@
    # "premium_index_change_24h",
    "linear_regression_slope_atr_1h_24",
    "distance_to_rolling_low_4h",
    "residual_return_24h",
    # Regime features - оставляем только работающие (см. feature importance)
    # "volatility_regime_change_1h",  # ВКЛЮЧЕН - gain=282, хорошо работает
    # "volatility_acceleration_1h",  # ВКЛЮЧЕН - gain=100
    # "volatility_regime_stability",  # ВКЛЮЧЕН - gain=264, хорошо
    # "vol_of_vol_1h",  # ВКЛЮЧЕН - gain=957, ТОП-6, отлично!
    # "realized_vol_vs_ema",  # ВКЛЮЧЕН - gain=107
    # "trend_persistence_score_24",   # ВКЛЮЧЕН - gain=26, слабый но оставим
    # "trend_efficiency_24h",       # ВКЛЮЧЕН - gain=23, слабый но оставим
    # "trend_persistence_score_12", # ВКЛЮЧЕН - gain=206
    # Volume features - ВКЛЮЧЕНЫ
    # "volume_ratio_1h",              # ВКЛЮЧЕН - gain=294
    # "volume_zscore_1h",             # ВКЛЮЧЕН - gain=206
    # "dollar_volume_zscore_1h",      # ВКЛЮЧЕН - gain=114
    # Отключаем НЕРАБОТАЮЩИЕ признаки (gain=0 или низкий):
    "delta_market_breadth_ema_fast_slow_1h",  # gain=0 - не работает
    "high_vol_stress_indicator",  # gain=0 - не работает
    "breakout_quality_4h",  # gain=0 - не работает (Donchian разрыв?)
    "counter_market_penalty_1h",  # gain=0 - не работает
    "signal_x_high_vol_stress",  # gain=0 - не работает
    "vol_regime_classification",  # gain=85 - слабый, отключаем (категориальный нестабильный)
    "trend_efficiency_x_vol_stability",  # gain=64 - слабый, отключаем
    "signal_market_agreement_1h",  # gain=80 - слабый, отключаем
    "breakout_quality_4h_x_volume_ratio_1h",  # gain=8 - очень слабый, отключаем
    # Отключаем слабые по новому тесту (gain < 100):
    "premium_index_zscore_7d",  # gain=55 - очень слабый
    "volume_zscore_1h",  # gain=62 - очень слабый
    "shorts_overheated_1h",  # gain=121 - слабый
    "trend_persistence_score_24",  # gain=146 - слабый
    "volatility_acceleration_1h",  # gain=167 - слабый
    "dollar_volume_zscore_1h",  # gain=167 - слабый
    "premium_index_1h",  # gain=115 - слабый
    # Отключаем regime интеракции, оставляем только базовые regime признаки
    "realized_vol_vs_ema",  # оставляем vol_of_vol, vol_regime_change, vol_stability
    "ema_fast_slow_x_vol_of_vol",  # интеракция - отключаем
    "trend_efficiency_24h_x_volatility_regime_change_1h",  # интеракция - отключаем
    "market_pressure_x_vol_regime",  # интеракция - отключаем
    "trend_efficiency_24h",  # мало влияет в новом режиме
    "trend_persistence_score_12",  # слабый
    "donchian_width_change_4h",  # можно отключить
    "open_interest_zscore_7d",  # слабый
    # "adx_4h",
    "beta_to_btc_24h",
    "ema_fast_slow_x_market_breadth_ema_fast_slow_1h",
    # "market_breadth_ema_fast_slow_1h",
    "market_breadth_ema_fast_slow_1h_zscore",
    # "market_breadth_pos_return_4h_3",
    "market_directional_pressure_1h",
    "market_dispersion_return_4h_3",

    # "range_position_1h_48",
    "range_width_atr_1h_48",
    # "range_center_distance_atr_1h_48",
    # "flat_efficiency_1h_24",
    # "mean_reversion_pressure_1h",
    # "zscore_vs_vwap_1h",

    # "bollinger_percent_b_1h_20",
    # "bollinger_bandwidth_atr_1h_20",

    "volume_24h",
    "open_interest_change_pct_24h",
    "open_interest_change_pct_8h",
    "hour_sin_1h",
    "volume_ratio_1h",
    "longs_overheated_1h",
    "funding_rate_change_24h",

    "open_interest_funding_crowding_1h",
    "donchian_width_atr_4h",
    "ema_slope_acceleration_1h",
    #=======================
    "vol_ratio",
    "ema_slope_acceleration_1h",
    "funding_rate_zscore_7d",
        #===
    "trend_alignment_1h_4h",
    "return_4h_7",
    "vol_of_vol_1h", #!
    "realized_vol_1h", #! | 04-18 16:09 | hybrid_v1_labels_v | 56.06% | 0.119 | 0.575 | 0.553 | 1.39% | 50488 |
    "bollinger_bandwidth_atr_1h_20",
    "bollinger_percent_b_1h_20",
    "distance_to_rolling_high_4h",
    "volatility_regime_stability",
    "market_abs_avg_ema_slope_4h",
    "market_avg_ema_slope_4h",
    "market_breadth_pos_return_4h_14",
    "market_mean_return_4h_14",
    "cross_sectional_rank_4h",
    # "price_position_1h",
    "zscore_vs_vwap_1h",

    # "cross_sectional_rank_ema_fast_slow_1h",
    # "distance_to_resistance_1h",
    # "flat_efficiency_1h_24",
    "linear_regression_slope_atr_1h_12",

    "range_position_1h_48",
    "mean_reversion_pressure_1h",
]
# --- FEATURE BUILD ---
FEATURE_PROFILES = {
    "all": "__all__",
    "empty": [],
    "base_only": [
        "realized_vol_1h",
        "ema_fast_slow",
        "return_1h_24",
        "linear_regression_slope_atr_1h_24",
        "trend_efficiency_24h",
        "atr_ratio_1h",
        "range_compression_1h",
        "price_position_1h",
        "relative_strength_vs_btc_24h",
        "residual_return_24h",
        "return_4h_3",
        "return_4h_14",
        "ema_slope_4h",
        "realized_vol_4h_returns_20",
        "market_breadth_pos_return_4h_3",
        "market_dispersion_return_4h_3",
        "zscore_vs_vwap_4h",
        "vol_ratio",
        "price_position_4h",
        "adx_4h",
    ],
}
FEATURE_BUILD_REQUEST = {
    "profile": "all",
    "include_features": [],
    "exclude_features": [],
    "exclude_blocks": [],
}

# Rolling MCC (asset vs BTC 1h return sign): window in bars; min_periods None = max(3, window//2)
MCC_SIGN_BTC_WINDOW = 24
MCC_SIGN_BTC_MIN_PERIODS = None

# Main-TF Wilder RSI (momentum feature `rsi_1h`)
RSI_LENGTH = 14

# Hybrid setup:
# - ETL labeling stays on v1.
# - Training-side filtering and feature pruning stay on v2.
# --- ML LABELING (Triple Barrier) ---
HORIZON = 12
TP_PCT = 0.03   # legacy fixed TP, kept for backward compatibility
SL_PCT = 0.015  # legacy fixed SL, kept for backward compatibility

# --- ADAPTIVE HORIZON (multi-symbol) ---
ENABLE_ADAPTIVE_HORIZON = True
ADAPTIVE_HORIZON_MIN = 8
ADAPTIVE_HORIZON_MAX = 20
ADAPTIVE_HORIZON_VOL_LOW = 0.005
ADAPTIVE_HORIZON_VOL_HIGH = 0.025


def effective_max_label_horizon() -> int:
    """
    Upper bound (in main-TF bars) on how far the triple-barrier label can look ahead
    from a given row — used e.g. for WFV purge_gap so train y does not use test-period
    prices. Mirrors the extrema of etl.compute_effective_horizons.
    """
    base_horizon = max(1, int(HORIZON))
    if not bool(ENABLE_ADAPTIVE_HORIZON):
        return base_horizon
    h_min = int(ADAPTIVE_HORIZON_MIN)
    h_max = int(ADAPTIVE_HORIZON_MAX)
    if h_min > h_max:
        h_min, h_max = h_max, h_min
    h_min = max(1, h_min)
    h_max = max(h_min, h_max)
    return max(base_horizon, h_max)


# --- DYNAMIC BARRIERS ---
USE_DYNAMIC_BARRIERS = True
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_RVOL_MULTIPLIER = 0.75
BARRIER_TP_TO_SL_RATIO = 2.0
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06




# TEMPORAL SAMPLE WEIGHTING - 2026-04-05
# Усиленное взвешивание для адаптации к смене режима
# ═══════════════════════════════════════════════════════════════════
SAMPLE_WEIGHT_HALF_LIFE_DAYS = 180.0    # Было 90
REGIME_AWARE_WEIGHTING = True           # Дополнительный буст свежим данным
REGIME_RECENT_DAYS_BOOST = 60.0         # Сколько дней считать "свежими" - было 20
REGIME_RECENT_BOOST_FACTOR = 1.5        # Во сколько раз увеличить вес свежих - было 2.0

# --- RAW REBUILD SAFETY ---
ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY = False

# --- TRADING ---
TAKER_COM = 0.0004
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01
DIRECTIONAL_PROBA_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.55
MIN_SIGNAL_GAP = 0.01
ALLOW_LONGS = True
ALLOW_SHORTS = True
BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 1     # 1 = берем лучший сигнал на баре, >1 = топ-N сигналов
BACKTEST_MAX_OPEN_POSITIONS = 1            # максимум одновременно открытых позиций
BACKTEST_SL_COOLDOWN_BARS = 8
BACKTEST_MAX_SL_PER_DAY = 3
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = 2
BACKTEST_REDUCED_RISK_PER_TRADE = 0.005
# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BACKTEST_CHARTS_DIR = Path("backtest_charts")
BACKTEST_CHARTS_DIR.mkdir(exist_ok=True)

# --- EXECUTION LOG (paper / live, paper.py) ---
# None → та же БД, что и ETL (DB_PATH). Отдельный файл — только если нужно изолировать WAL.
EXECUTION_DB_PATH = _env_str("EXECUTION_DB_PATH")

# --- EXECUTION DB TYPE ---
# "sqlite" - локальная SQLite база (по умолчанию)
# "supabase" - облачная PostgreSQL через Supabase
EXECUTION_DB_TYPE = str(_env_str("EXECUTION_DB_TYPE", "sqlite")).lower()

# --- SUPABASE CONFIG (только если EXECUTION_DB_TYPE = "supabase") ---
# URL проекта Supabase (например: "https://xxxxxx.supabase.co")
SUPABASE_URL = _env_str("SUPABASE_URL", "https://jjuatlyxubeglxkrpaji.supabase.co")
# SUPABASE_KEY должен быть задан в переменных окружения:
# PowerShell: $env:SUPABASE_KEY="your-anon-key-or-service-key"
SUPABASE_KEY = _env_str("SUPABASE_KEY") or _env_str("SUPABASE_SERVICE_KEY")

# Regime-aware sample-weight tuning overrides.
REGIME_WEIGHT_STRENGTH = 0.18
REGIME_WEIGHT_STRENGTH_CAP = 0.25
REGIME_WEIGHT_SLOPE_SCALE_4H = 0.08
SAMPLE_WEIGHT_MIN = 0.8
SAMPLE_WEIGHT_MAX = 1.35
EARLY_STOPPING_ROUNDS = 200
UNSTABLE_FOLD_MIN_BEST_ITER = 25
UNSTABLE_FOLD_FALLBACK_MIN_ESTIMATORS = 150
UNSTABLE_FOLD_FALLBACK_DEFAULT_ESTIMATORS = 250
