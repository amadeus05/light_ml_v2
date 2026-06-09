BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_PREMIUM_INDEX_KLINE_URL = (
    "https://api.bybit.com/v5/market/premium-index-price-kline"
)
BYBIT_OPEN_INTEREST_URL = "https://api.bybit.com/v5/market/open-interest"
BYBIT_FUNDING_RATE_URL = "https://api.bybit.com/v5/market/funding/history"
BYBIT_PUBLIC_WS_URL = "wss://stream.bybit.com/v5/public/linear"
BYBIT_CATEGORY = "linear"

BYBIT_LIMIT = 1000
BYBIT_FUNDING_LIMIT = 200
BYBIT_OPEN_INTEREST_LIMIT = 200
BYBIT_FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000

BYBIT_TIMEOUT = 20
BYBIT_RETRY_COUNT = 5
BYBIT_RETRY_SLEEP = 0.33
BYBIT_MAX_WORKERS = 6

BYBIT_WS_PING_SEC = 20.0
BYBIT_WS_RECONNECT_SEC = 3.0
