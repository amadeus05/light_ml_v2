"""
Журнал сделок (paper / live) в Supabase (PostgreSQL).

Использует Supabase Python клиент для работы с облачной базой.
Таблицы должны быть созданы в Supabase заранее через SQL Editor.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

try:
    from supabase import create_client, Client
except ImportError:
    Client = None  # type: ignore

from src.persistence.repositories.base_trades_repository import (
    BaseTradesRepository,
    ExecutionType,
    OpenTradeRow,
)

logger = logging.getLogger(__name__)


class SupabaseTradesRepository(BaseTradesRepository):
    """Supabase (PostgreSQL) реализация репозитория сделок."""

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        if Client is None:
            raise ImportError(
                "supabase package не установлен. "
                "Установите: pip install supabase"
            )
        self.client: Client = create_client(supabase_url, supabase_key)

    def _check_connection(self) -> None:
        """Проверяет подключение к Supabase выполняя health-check запрос."""
        try:
            # Пробуем получить версию или сделать простой запрос
            response = self.client.table("execution_engine_state").select("k").limit(1).execute()
        except Exception as e:
            error_msg = str(e).lower()
            # Таблица может отсутствовать на этапе первичной настройки.
            # Это не проблема подключения/аутентификации, а проблема схемы.
            if "could not find the table" in error_msg and "schema cache" in error_msg:
                logger.info(
                    "Подключение к Supabase есть, но таблица execution_engine_state пока не создана"
                )
                return
            if "invalid api key" in error_msg or "jwt" in error_msg:
                raise ConnectionError(
                    "❌ Ошибка аутентификации Supabase: неверный API ключ. "
                    "Проверьте переменную окружения SUPABASE_KEY."
                ) from e
            elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                raise ConnectionError(
                    "❌ Ошибка сети: не удалось подключиться к Supabase. "
                    "Проверьте подключение к интернету и URL проекта."
                ) from e
            else:
                raise ConnectionError(
                    f"❌ Ошибка подключения к Supabase: {e}"
                ) from e

    def _check_tables_exist(self) -> None:
        """Проверяет что необходимые таблицы существуют в Supabase."""
        required_tables = ["execution_trades", "execution_engine_state"]
        missing_tables = []

        for table in required_tables:
            try:
                self.client.table(table).select("count").limit(1).execute()
            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "does not exist" in error_msg
                    or "404" in error_msg
                    or ("could not find the table" in error_msg and "schema cache" in error_msg)
                ):
                    missing_tables.append(table)

        if missing_tables:
            raise RuntimeError(
                f"❌ В Supabase отсутствуют таблицы: {', '.join(missing_tables)}.\n"
                f"Создайте таблицы через SQL Editor в Supabase Dashboard:\n"
                f"1. Откройте Dashboard → SQL Editor\n"
                f"2. Выполните SQL скрипт из документации (см. supabase_trades_repository.py)"
            )

    def _check_rpc_functions(self) -> None:
        """Проверяет что RPC функции для агрегаций существуют."""
        required_functions = {
            "count_sl_today": {
                "p_execution_type": "paper",
                "p_symbol": "BTC/USDT",
                "p_day_start_ms": 0,
                "p_now_ms": int(time.time() * 1000),
            },
            "last_sl_exit_ms": {
                "p_execution_type": "paper",
                "p_symbol": "BTC/USDT",
            },
            "sum_closed_pnl": {"p_execution_type": "paper"},
            "sum_open_notional": {"p_execution_type": "paper"},
        }
        missing_functions = []

        for func, params in required_functions.items():
            try:
                self.client.rpc(func, params).execute()
            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "function" in error_msg
                    and "does not exist" in error_msg
                ) or (
                    "could not find the function" in error_msg
                    and "schema cache" in error_msg
                ):
                    missing_functions.append(func)

        if missing_functions:
            raise RuntimeError(
                f"❌ В Supabase отсутствуют RPC функции: {', '.join(missing_functions)}.\n"
                f"Создайте функции через SQL Editor в Supabase Dashboard:\n"
                f"1. Откройте Dashboard → SQL Editor\n"
                f"2. Выполните SQL скрипт с RPC функциями из документации"
            )

    def init_schema(self) -> None:
        """
        Проверяет подключение и существование таблиц/функций в Supabase.

        Выполняет проверку:
        1. Подключения к Supabase (аутентификация, сеть)
        2. Существования таблиц execution_trades и execution_engine_state
        3. Существования RPC функций для агрегаций

        SQL для создания таблиц и функций находится в документации этого класса.
        """
        logger.info("Подключение к Supabase...")

        # Шаг 1: Проверяем подключение
        self._check_connection()
        logger.info("✅ Аутентификация Supabase успешна")

        # Шаг 2: Проверяем что таблицы существуют
        self._check_tables_exist()
        logger.info("✅ Таблицы execution_trades и execution_engine_state найдены")

        # Шаг 3: Проверяем RPC функции
        self._check_rpc_functions()
        logger.info("✅ RPC функции для агрегаций доступны")

        logger.info("🚀 Supabase репозиторий готов к работе")

    def _get_schema_sql(self) -> str:
        """Возвращает SQL для создания таблиц и функций (для удобства копирования)."""
        return """
-- Создание таблиц для paper trading в Supabase

CREATE TABLE IF NOT EXISTS execution_trades (
    id SERIAL PRIMARY KEY,
    execution_type TEXT NOT NULL CHECK (execution_type IN ('paper', 'live')),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    exchange_code TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction INTEGER NOT NULL CHECK (direction IN (1, -1)),
    timeframe_signal TEXT NOT NULL DEFAULT '1h',
    signal_bar_open_ms BIGINT NOT NULL,
    entry_bar_open_ms BIGINT,
    entry_ts_ms BIGINT NOT NULL,
    entry_price REAL NOT NULL,
    entry_notional REAL NOT NULL,
    stop_pct REAL NOT NULL,
    take_pct REAL NOT NULL,
    p_long REAL,
    p_short REAL,
    signal_gap REAL,
    model_name TEXT,
    last_1m_scan_open_ms BIGINT,
    exit_ts_ms BIGINT,
    exit_price REAL,
    exit_reason TEXT,
    pnl_pct REAL,
    pnl_quote REAL,
    fees_quote REAL,
    meta_json TEXT,
    created_at_ms BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exec_trades_type_status_time
    ON execution_trades (execution_type, status, entry_ts_ms);
CREATE INDEX IF NOT EXISTS idx_exec_trades_symbol_entry
    ON execution_trades (symbol, entry_ts_ms);
CREATE INDEX IF NOT EXISTS idx_exec_trades_exit_time
    ON execution_trades (execution_type, exit_ts_ms);

CREATE TABLE IF NOT EXISTS execution_engine_state (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

-- RPC функции для агрегаций

CREATE OR REPLACE FUNCTION count_sl_today(
    p_execution_type TEXT,
    p_symbol TEXT,
    p_day_start_ms BIGINT,
    p_now_ms BIGINT
) RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*) FROM execution_trades
        WHERE execution_type = p_execution_type
          AND status = 'closed'
          AND symbol = p_symbol
          AND exit_reason = 'SL'
          AND exit_ts_ms >= p_day_start_ms
          AND exit_ts_ms <= p_now_ms
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION last_sl_exit_ms(
    p_execution_type TEXT,
    p_symbol TEXT
) RETURNS BIGINT AS $$
BEGIN
    RETURN (
        SELECT MAX(exit_ts_ms) FROM execution_trades
        WHERE execution_type = p_execution_type
          AND status = 'closed'
          AND symbol = p_symbol
          AND exit_reason = 'SL'
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION sum_closed_pnl(
    p_execution_type TEXT
) RETURNS REAL AS $$
BEGIN
    RETURN COALESCE(
        (SELECT SUM(pnl_quote) FROM execution_trades
         WHERE execution_type = p_execution_type AND status = 'closed'),
        0
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION sum_open_notional(
    p_execution_type TEXT
) RETURNS REAL AS $$
BEGIN
    RETURN COALESCE(
        (SELECT SUM(entry_notional) FROM execution_trades
         WHERE execution_type = p_execution_type AND status = 'open'),
        0
    );
END;
$$ LANGUAGE plpgsql;
        """

    def get_state(self, key: str) -> str | None:
        response = self.client.table("execution_engine_state").select("v").eq("k", key).execute()
        data = response.data
        return data[0]["v"] if data else None

    def set_state(self, key: str, value: str) -> None:
        self.client.table("execution_engine_state").upsert({
            "k": key,
            "v": value,
        }).execute()

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
        now = int(time.time() * 1000)
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None

        data = {
            "execution_type": execution_type,
            "status": "open",
            "exchange_code": exchange_code,
            "symbol": symbol,
            "direction": direction,
            "timeframe_signal": timeframe_signal,
            "signal_bar_open_ms": signal_bar_open_ms,
            "entry_bar_open_ms": entry_bar_open_ms,
            "entry_ts_ms": entry_ts_ms,
            "entry_price": entry_price,
            "entry_notional": entry_notional,
            "stop_pct": stop_pct,
            "take_pct": take_pct,
            "p_long": p_long,
            "p_short": p_short,
            "signal_gap": signal_gap,
            "model_name": model_name,
            "last_1m_scan_open_ms": last_1m_scan_open_ms,
            "meta_json": meta_json,
            "created_at_ms": now,
            "updated_at_ms": now,
        }

        response = self.client.table("execution_trades").insert(data).execute()
        return response.data[0]["id"]

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
        now = int(time.time() * 1000)
        data = {
            "status": "closed",
            "exit_ts_ms": exit_ts_ms,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_pct": pnl_pct,
            "pnl_quote": pnl_quote,
            "fees_quote": fees_quote,
            "updated_at_ms": now,
            "last_1m_scan_open_ms": None,
        }
        self.client.table("execution_trades").update(data).eq("id", trade_id).execute()

    def update_last_1m_scan(self, trade_id: int, last_1m_scan_open_ms: int) -> None:
        now = int(time.time() * 1000)
        self.client.table("execution_trades").update({
            "last_1m_scan_open_ms": last_1m_scan_open_ms,
            "updated_at_ms": now,
        }).eq("id", trade_id).eq("status", "open").execute()

    def list_open_trades(self, execution_type: ExecutionType) -> list[OpenTradeRow]:
        response = self.client.table("execution_trades").select(
            "id, symbol, direction, entry_price, entry_notional, stop_pct, take_pct, "
            "last_1m_scan_open_ms, entry_ts_ms, p_long, p_short"
        ).eq("execution_type", execution_type).eq("status", "open").order("id").execute()

        rows = []
        for r in response.data:
            rows.append(
                OpenTradeRow(
                    id=int(r["id"]),
                    symbol=str(r["symbol"]),
                    direction=int(r["direction"]),
                    entry_price=float(r["entry_price"]),
                    entry_notional=float(r["entry_notional"]),
                    stop_pct=float(r["stop_pct"]),
                    take_pct=float(r["take_pct"]),
                    last_1m_scan_open_ms=r.get("last_1m_scan_open_ms"),
                    entry_ts_ms=int(r["entry_ts_ms"]) if r.get("entry_ts_ms") is not None else None,
                    p_long=float(r["p_long"]) if r.get("p_long") is not None else None,
                    p_short=float(r["p_short"]) if r.get("p_short") is not None else None,
                )
            )
        return rows

    def count_open_trades(self, execution_type: ExecutionType) -> int:
        response = self.client.table("execution_trades").select(
            "count", count="exact"
        ).eq("execution_type", execution_type).eq("status", "open").execute()
        return response.count or 0

    def count_sl_today_utc(
        self, execution_type: ExecutionType, symbol: str, day_start_ms: int, now_ms: int
    ) -> int:
        # Используем RPC для сложного запроса с агрегацией
        response = self.client.rpc(
            "count_sl_today",
            {
                "p_execution_type": execution_type,
                "p_symbol": symbol,
                "p_day_start_ms": day_start_ms,
                "p_now_ms": now_ms,
            }
        ).execute()
        return response.data or 0

    def last_sl_exit_ms(
        self, execution_type: ExecutionType, symbol: str
    ) -> int | None:
        response = self.client.rpc(
            "last_sl_exit_ms",
            {
                "p_execution_type": execution_type,
                "p_symbol": symbol,
            }
        ).execute()
        return response.data

    def sum_closed_pnl_quote(self, execution_type: ExecutionType) -> float:
        response = self.client.rpc(
            "sum_closed_pnl",
            {"p_execution_type": execution_type}
        ).execute()
        return float(response.data or 0)

    def sum_open_margin_quote(self, execution_type: ExecutionType, leverage: float) -> float:
        if leverage <= 0:
            return 0.0
        response = self.client.rpc(
            "sum_open_notional",
            {"p_execution_type": execution_type}
        ).execute()
        total_notional = float(response.data or 0)
        return total_notional / leverage

    # SQL скрипт доступен через метод _get_schema_sql() для удобного копирования
