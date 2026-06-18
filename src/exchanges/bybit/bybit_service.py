from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.exchanges.market_data_client import MarketDataClient
from src.exchanges.bybit.bybit_adapter import BybitAdapter
from src.exchanges.bybit.bybit_mapper import BybitMapper
from src.types.common import FundingRatePoint, HistoricalKline, OpenInterestPoint, Symbol


logger = logging.getLogger(__name__)


class BybitService(MarketDataClient):
    def __init__(
        self,
        adapter: BybitAdapter | None = None,
        mapper: BybitMapper | None = None,
    ) -> None:
        self.adapter = adapter or BybitAdapter()
        self.mapper = mapper or BybitMapper()

    def get_exchange_code(self) -> str:
        return "bybit"

    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        return self.mapper.normalize_symbol(symbol)

    def get_timeframe_ms(self, timeframe: str) -> int:
        return self.mapper.timeframe_to_ms(timeframe)

    def get_funding_interval_ms(self, symbol: str | Symbol) -> int:
        return int(getattr(self.adapter, "funding_interval_ms", 8 * 60 * 60 * 1000))

    def build_request_windows(
        self,
        start_ts: int,
        end_ts: int,
        timeframe_ms: int,
        limit: int | None = None,
    ) -> list[tuple[int, int]]:
        if start_ts > end_ts:
            return []

        effective_limit = self.adapter.limit if limit is None else int(limit)
        max_span_ms = timeframe_ms * max(effective_limit - 1, 1)
        windows = []
        window_start = start_ts

        while window_start <= end_ts:
            window_end = min(window_start + max_span_ms, end_ts)
            windows.append((window_start, window_end))
            window_start = window_end + timeframe_ms

        return windows

    def fetch_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        normalized_symbol = self.normalize_symbol(symbol)
        timeframe_ms = self.get_timeframe_ms(timeframe)
        windows = self.build_request_windows(start_ts, end_ts, timeframe_ms)
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        interval = self.mapper.to_interval(timeframe)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_klines: list[HistoricalKline] = []

        logger.info(
            f"[{normalized_symbol}-{timeframe}] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_kline_window,
                    api_symbol,
                    interval,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                klines = self.mapper.to_klines(payload)
                if klines:
                    all_klines.extend(klines)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = klines[-1].open_time if klines else window_end
                    logger.info(
                        f"[{normalized_symbol}-{timeframe}] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        all_klines.sort(key=lambda candle: candle.open_time)
        return all_klines

    def fetch_premium_index_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        normalized_symbol = self.normalize_symbol(symbol)
        timeframe_ms = self.get_timeframe_ms(timeframe)
        windows = self.build_request_windows(start_ts, end_ts, timeframe_ms)
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        interval = self.mapper.to_interval(timeframe)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_klines: list[HistoricalKline] = []

        logger.info(
            f"[{normalized_symbol}-premium-{timeframe}] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_premium_index_kline_window,
                    api_symbol,
                    interval,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                klines = self.mapper.to_price_klines(payload)
                if klines:
                    all_klines.extend(klines)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = klines[-1].open_time if klines else window_end
                    logger.info(
                        f"[{normalized_symbol}-premium-{timeframe}] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        deduped = {candle.open_time: candle for candle in all_klines}
        return [deduped[key] for key in sorted(deduped)]

    def fetch_open_interest(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[OpenInterestPoint]:
        normalized_symbol = self.normalize_symbol(symbol)
        timeframe_ms = self.get_timeframe_ms(timeframe)
        windows = self.build_request_windows(
            start_ts,
            end_ts,
            timeframe_ms,
            limit=self.adapter.open_interest_limit,
        )
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        interval_time = self.mapper.to_open_interest_interval(timeframe)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_points: list[OpenInterestPoint] = []

        logger.info(
            f"[{normalized_symbol}-{timeframe}-open-interest] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.open_interest_limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_open_interest_window,
                    api_symbol,
                    interval_time,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                points = self.mapper.to_open_interest_points(payload)
                if points:
                    all_points.extend(points)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = points[-1].timestamp if points else window_end
                    logger.info(
                        f"[{normalized_symbol}-{timeframe}-open-interest] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        deduped_points = {point.timestamp: point for point in all_points}
        return [deduped_points[key] for key in sorted(deduped_points)]

    def fetch_funding_rates(
        self,
        symbol: str | Symbol,
        start_ts: int,
        end_ts: int,
    ) -> list[FundingRatePoint]:
        normalized_symbol = self.normalize_symbol(symbol)
        interval_ms = self.get_funding_interval_ms(normalized_symbol)
        windows = self.build_request_windows(
            start_ts,
            end_ts,
            interval_ms,
            limit=self.adapter.funding_limit,
        )
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_points: list[FundingRatePoint] = []

        logger.info(
            f"[{normalized_symbol}-funding] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.funding_limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_funding_rate_window,
                    api_symbol,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                points = self.mapper.to_funding_rates(payload)
                if points:
                    all_points.extend(points)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = points[-1].funding_time if points else window_end
                    logger.info(
                        f"[{normalized_symbol}-funding] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        deduped_points = {point.funding_time: point for point in all_points}
        return [deduped_points[key] for key in sorted(deduped_points)]
