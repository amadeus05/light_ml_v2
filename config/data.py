from .env import env_str


DB_PATH = str(env_str("DB_PATH", "./data/market_data.db"))

SYMBOLS = [
    "BTC/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "SOL/USDT",
]

TIMEFRAME = "15m"
# The first timeframe is used for existing HTF features. All listed
# timeframes are downloaded by ETL.
HTF_TIMEFRAME = "1h,4h"

ACTIVE_EXCHANGE = "bybit"

START_DATE = "2023-01-01"
END_DATE = "2026-03-27 21:00:00"

ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY = False
