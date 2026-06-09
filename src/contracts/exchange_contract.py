from __future__ import annotations

from abc import ABC, abstractmethod

from src.types.common import FundingRatePoint, HistoricalKline, OpenInterestPoint, Symbol


class ExchangeContract(ABC):
    @abstractmethod
    def get_exchange_code(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        raise NotImplementedError

    @abstractmethod
    def get_timeframe_ms(self, timeframe: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        raise NotImplementedError

    @abstractmethod
    def fetch_premium_index_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        raise NotImplementedError

    @abstractmethod
    def fetch_open_interest(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[OpenInterestPoint]:
        raise NotImplementedError

    @abstractmethod
    def get_funding_interval_ms(self, symbol: str | Symbol) -> int:
        raise NotImplementedError

    @abstractmethod
    def fetch_funding_rates(
        self,
        symbol: str | Symbol,
        start_ts: int,
        end_ts: int,
    ) -> list[FundingRatePoint]:
        raise NotImplementedError
