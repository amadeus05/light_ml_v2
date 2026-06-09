from __future__ import annotations

import config as cfg

from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_mapper import BinanceMapper
from src.exchanges.bybit.bybit_mapper import BybitMapper
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository
from src.types.common import FundingRatePoint, HistoricalKline, OpenInterestPoint, Symbol


class SimulationExchangeService(ExchangeContract):
    def __init__(
        self,
        *,
        db_path: str | None = None,
        source_exchange: str | None = None,
    ) -> None:
        self.source_exchange = self._normalize_source_exchange(
            source_exchange or getattr(cfg, "SIMULATION_BASE_EXCHANGE", "bybit")
        )
        self.db_path = db_path or getattr(cfg, "SIMULATION_DB_PATH", getattr(cfg, "DB_PATH", "market_data.db"))
        self.repository = HistoricalKlineRepository(
            db_path=self.db_path,
            exchange_code=self.source_exchange,
        )
        self.mapper = self._build_mapper(self.source_exchange)

    def get_exchange_code(self) -> str:
        return "simulation"

    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        return self.mapper.normalize_symbol(symbol)

    def get_timeframe_ms(self, timeframe: str) -> int:
        return self.mapper.timeframe_to_ms(timeframe)

    def fetch_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        normalized_symbol = self.normalize_symbol(symbol)
        return self.repository.fetch_candles_range(normalized_symbol, timeframe, start_ts, end_ts)

    def fetch_premium_index_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        normalized_symbol = self.normalize_symbol(symbol)
        return self.repository.fetch_premium_index_klines_range(normalized_symbol, timeframe, start_ts, end_ts)

    def fetch_open_interest(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[OpenInterestPoint]:
        normalized_symbol = self.normalize_symbol(symbol)
        return self.repository.fetch_open_interest_range(normalized_symbol, timeframe, start_ts, end_ts)

    def get_funding_interval_ms(self, symbol: str | Symbol) -> int:
        if self.source_exchange == "binance":
            return int(getattr(cfg, "BINANCE_FUNDING_INTERVAL_MS", 8 * 60 * 60 * 1000))
        return int(getattr(cfg, "BYBIT_FUNDING_INTERVAL_MS", 8 * 60 * 60 * 1000))

    def fetch_funding_rates(
        self,
        symbol: str | Symbol,
        start_ts: int,
        end_ts: int,
    ) -> list[FundingRatePoint]:
        normalized_symbol = self.normalize_symbol(symbol)
        return self.repository.fetch_funding_rates_range(normalized_symbol, start_ts, end_ts)

    @staticmethod
    def _normalize_source_exchange(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"bybit", "binance"}:
            raise ValueError(f"Unsupported simulation source exchange: {value}")
        return normalized

    @staticmethod
    def _build_mapper(source_exchange: str):
        if source_exchange == "binance":
            return BinanceMapper()
        if source_exchange == "bybit":
            return BybitMapper()
        raise ValueError(f"Unsupported simulation source exchange: {source_exchange}")
