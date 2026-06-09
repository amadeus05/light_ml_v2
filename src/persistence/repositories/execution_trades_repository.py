"""
Журнал сделок (paper / live) в SQLite для последующих метрик.

Таблица нормализована под агрегации: фильтр по execution_type, status,
диапазону entry_ts_ms / exit_ts_ms, symbol.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

from src.persistence.repositories.base_trades_repository import (
    BaseTradesRepository,
    ExecutionType,
    OpenTradeRow,
    TradeStatus,
)

logger = logging.getLogger(__name__)


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class ExecutionTradesRepository(BaseTradesRepository):
    """SQLite реализация репозитория сделок."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def init_schema(self) -> None:
        logger.info(f"Инициализация SQLite схемы: {self.db_path}")
        with _conn(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_type TEXT NOT NULL CHECK (execution_type IN ('paper', 'live')),
                    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
                    exchange_code TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction INTEGER NOT NULL CHECK (direction IN (1, -1)),
                    timeframe_signal TEXT NOT NULL DEFAULT '1h',
                    signal_bar_open_ms INTEGER NOT NULL,
                    entry_bar_open_ms INTEGER,
                    entry_ts_ms INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_notional REAL NOT NULL,
                    stop_pct REAL NOT NULL,
                    take_pct REAL NOT NULL,
                    p_long REAL,
                    p_short REAL,
                    signal_gap REAL,
                    model_name TEXT,
                    last_1m_scan_open_ms INTEGER,
                    exit_ts_ms INTEGER,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_pct REAL,
                    pnl_quote REAL,
                    fees_quote REAL,
                    meta_json TEXT,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
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
                """
            )
            conn.commit()
        logger.info("✅ SQLite схема инициализирована")

    def get_state(self, key: str) -> str | None:
        with _conn(self.db_path) as conn:
            cur = conn.execute("SELECT v FROM execution_engine_state WHERE k = ?", (key,))
            row = cur.fetchone()
            return None if row is None else str(row[0])

    def set_state(self, key: str, value: str) -> None:
        with _conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO execution_engine_state (k, v) VALUES (?, ?)
                ON CONFLICT(k) DO UPDATE SET v = excluded.v
                """,
                (key, value),
            )
            conn.commit()

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
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO execution_trades (
                    execution_type, status, exchange_code, symbol, direction,
                    timeframe_signal, signal_bar_open_ms, entry_bar_open_ms,
                    entry_ts_ms, entry_price, entry_notional,
                    stop_pct, take_pct, p_long, p_short, signal_gap,
                    model_name, last_1m_scan_open_ms,
                    exit_ts_ms, exit_price, exit_reason, pnl_pct, pnl_quote, fees_quote,
                    meta_json, created_at_ms, updated_at_ms
                ) VALUES (
                    ?, 'open', ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?,
                    NULL, NULL, NULL, NULL, NULL, NULL,
                    ?, ?, ?
                )
                """,
                (
                    execution_type,
                    exchange_code,
                    symbol,
                    direction,
                    timeframe_signal,
                    signal_bar_open_ms,
                    entry_bar_open_ms,
                    entry_ts_ms,
                    entry_price,
                    entry_notional,
                    stop_pct,
                    take_pct,
                    p_long,
                    p_short,
                    signal_gap,
                    model_name,
                    last_1m_scan_open_ms,
                    meta_json,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

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
        with _conn(self.db_path) as conn:
            conn.execute(
                """
                UPDATE execution_trades SET
                    status = 'closed',
                    exit_ts_ms = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    pnl_pct = ?,
                    pnl_quote = ?,
                    fees_quote = ?,
                    updated_at_ms = ?,
                    last_1m_scan_open_ms = NULL
                WHERE id = ?
                """,
                (
                    exit_ts_ms,
                    exit_price,
                    exit_reason,
                    pnl_pct,
                    pnl_quote,
                    fees_quote,
                    now,
                    trade_id,
                ),
            )
            conn.commit()

    def update_last_1m_scan(self, trade_id: int, last_1m_scan_open_ms: int) -> None:
        now = int(time.time() * 1000)
        with _conn(self.db_path) as conn:
            conn.execute(
                """
                UPDATE execution_trades SET last_1m_scan_open_ms = ?, updated_at_ms = ?
                WHERE id = ? AND status = 'open'
                """,
                (last_1m_scan_open_ms, now, trade_id),
            )
            conn.commit()

    def list_open_trades(self, execution_type: ExecutionType) -> list[OpenTradeRow]:
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, symbol, direction, entry_price, entry_notional,
                       stop_pct, take_pct, last_1m_scan_open_ms, entry_ts_ms,
                       p_long, p_short
                FROM execution_trades
                WHERE execution_type = ? AND status = 'open'
                ORDER BY id ASC
                """,
                (execution_type,),
            )
            rows = []
            for r in cur.fetchall():
                rows.append(
                    OpenTradeRow(
                        id=int(r["id"]),
                        symbol=str(r["symbol"]),
                        direction=int(r["direction"]),
                        entry_price=float(r["entry_price"]),
                        entry_notional=float(r["entry_notional"]),
                        stop_pct=float(r["stop_pct"]),
                        take_pct=float(r["take_pct"]),
                        last_1m_scan_open_ms=(
                            int(r["last_1m_scan_open_ms"])
                            if r["last_1m_scan_open_ms"] is not None
                            else None
                        ),
                        entry_ts_ms=int(r["entry_ts_ms"]) if r["entry_ts_ms"] is not None else None,
                        p_long=float(r["p_long"]) if r["p_long"] is not None else None,
                        p_short=float(r["p_short"]) if r["p_short"] is not None else None,
                    )
                )
            return rows

    def count_open_trades(self, execution_type: ExecutionType) -> int:
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM execution_trades WHERE execution_type = ? AND status = 'open'",
                (execution_type,),
            )
            return int(cur.fetchone()[0])

    def count_sl_today_utc(
        self, execution_type: ExecutionType, symbol: str, day_start_ms: int, now_ms: int
    ) -> int:
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM execution_trades
                WHERE execution_type = ?
                  AND status = 'closed'
                  AND symbol = ?
                  AND exit_reason = 'SL'
                  AND exit_ts_ms >= ? AND exit_ts_ms <= ?
                """,
                (execution_type, symbol, day_start_ms, now_ms),
            )
            return int(cur.fetchone()[0])

    def last_sl_exit_ms(
        self, execution_type: ExecutionType, symbol: str
    ) -> int | None:
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT MAX(exit_ts_ms) FROM execution_trades
                WHERE execution_type = ?
                  AND status = 'closed'
                  AND symbol = ?
                  AND exit_reason = 'SL'
                """,
                (execution_type, symbol),
            )
            row = cur.fetchone()[0]
            return None if row is None else int(row)

    def sum_closed_pnl_quote(self, execution_type: ExecutionType) -> float:
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT COALESCE(SUM(pnl_quote), 0) FROM execution_trades
                WHERE execution_type = ? AND status = 'closed'
                """,
                (execution_type,),
            )
            return float(cur.fetchone()[0])

    def sum_open_margin_quote(self, execution_type: ExecutionType, leverage: float) -> float:
        if leverage <= 0:
            return 0.0
        with _conn(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT COALESCE(SUM(entry_notional), 0) FROM execution_trades
                WHERE execution_type = ? AND status = 'open'
                """,
                (execution_type,),
            )
            total_notional = float(cur.fetchone()[0])
            return total_notional / leverage
