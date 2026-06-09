"""
Абстрактный базовый класс для репозитория сделок.

Позволяет переключаться между SQLite (локально) и Supabase (облако)
через единый интерфейс.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

ExecutionType = Literal["paper", "live"]
TradeStatus = Literal["open", "closed"]


@dataclass(frozen=True)
class OpenTradeRow:
    id: int
    symbol: str
    direction: int
    entry_price: float
    entry_notional: float
    stop_pct: float
    take_pct: float
    last_1m_scan_open_ms: int | None
    entry_ts_ms: int | None = None
    p_long: float | None = None
    p_short: float | None = None


class BaseTradesRepository(ABC):
    """Абстрактный базовый класс для хранения сделок."""

    @abstractmethod
    def init_schema(self) -> None:
        """Инициализирует схему базы данных (таблицы, индексы)."""
        pass

    @abstractmethod
    def get_state(self, key: str) -> str | None:
        """Получает значение состояния по ключу."""
        pass

    @abstractmethod
    def set_state(self, key: str, value: str) -> None:
        """Устанавливает значение состояния по ключу."""
        pass

    @abstractmethod
    def insert_open_trade(
        self,
        *,
        execution_type: ExecutionType,
        exchange_code: str,
        symbol: str,
        direction: int,
        timeframe_signal: str,
        signal_bar_open_ms: int,
        entry_bar_open_ms: int | None,
        entry_ts_ms: int,
        entry_price: float,
        entry_notional: float,
        stop_pct: float,
        take_pct: float,
        p_long: float | None,
        p_short: float | None,
        signal_gap: float | None,
        model_name: str | None,
        last_1m_scan_open_ms: int | None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        """Создает новую открытую сделку. Возвращает ID сделки."""
        pass

    @abstractmethod
    def close_trade(
        self,
        trade_id: int,
        *,
        exit_ts_ms: int,
        exit_price: float,
        exit_reason: str,
        pnl_pct: float,
        pnl_quote: float,
        fees_quote: float,
    ) -> None:
        """Закрывает сделку с расчетом PnL."""
        pass

    @abstractmethod
    def update_last_1m_scan(self, trade_id: int, last_1m_scan_open_ms: int) -> None:
        """Обновляет timestamp последнего 1m скана для открытой сделки."""
        pass

    @abstractmethod
    def list_open_trades(self, execution_type: ExecutionType) -> list[OpenTradeRow]:
        """Возвращает список всех открытых сделок."""
        pass

    @abstractmethod
    def count_open_trades(self, execution_type: ExecutionType) -> int:
        """Возвращает количество открытых сделок."""
        pass

    @abstractmethod
    def count_sl_today_utc(
        self, execution_type: ExecutionType, symbol: str, day_start_ms: int, now_ms: int
    ) -> int:
        """Считает количество SL сегодня по символу."""
        pass

    @abstractmethod
    def last_sl_exit_ms(
        self, execution_type: ExecutionType, symbol: str
    ) -> int | None:
        """Возвращает timestamp последнего SL по символу."""
        pass

    @abstractmethod
    def sum_closed_pnl_quote(self, execution_type: ExecutionType) -> float:
        """Суммарный PnL всех закрытых сделок."""
        pass

    @abstractmethod
    def sum_open_margin_quote(
        self, execution_type: ExecutionType, leverage: float
    ) -> float:
        """Суммарная маржа открытых позиций (notional / leverage)."""
        pass
