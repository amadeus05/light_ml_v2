from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass

import config as cfg

try:
    import websocket
except ImportError:  # pragma: no cover - handled by caller environment
    websocket = None


logger = logging.getLogger(__name__)


def _default_public_ws_url() -> str:
    category = str(getattr(cfg, "BYBIT_CATEGORY", "linear")).lower()
    if category == "spot":
        return "wss://stream.bybit.com/v5/public/spot"
    if category == "inverse":
        return "wss://stream.bybit.com/v5/public/inverse"
    if category == "option":
        return "wss://stream.bybit.com/v5/public/option"
    return "wss://stream.bybit.com/v5/public/linear"


@dataclass(frozen=True)
class KlineClosedEvent:
    topic: str
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    source_ts_ms: int


class BybitKlineStream:
    def __init__(self, *, url: str | None = None) -> None:
        if websocket is None:
            raise ImportError("Install websocket-client to use BybitKlineStream")

        self.url = url or str(getattr(cfg, "BYBIT_PUBLIC_WS_URL", _default_public_ws_url()))
        self.ping_interval_sec = float(getattr(cfg, "BYBIT_WS_PING_SEC", 20.0))
        self.reconnect_sleep_sec = float(getattr(cfg, "BYBIT_WS_RECONNECT_SEC", 3.0))

        self._queue: queue.Queue[KlineClosedEvent] = queue.Queue()
        self._ws = None
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._lock = threading.Lock()
        self._desired_topics: set[str] = set()
        self._subscribed_topics: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, daemon=True, name="bybit-kline-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._connected_event.clear()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("Failed to close Bybit websocket cleanly")

    def get_event(self, timeout: float | None = None) -> KlineClosedEvent:
        return self._queue.get(timeout=timeout)

    def sync_topics(self, topics: set[str]) -> None:
        with self._lock:
            desired = set(topics)
            current_desired = set(self._desired_topics)
            self._desired_topics = desired
            to_subscribe = desired - self._subscribed_topics
            to_unsubscribe = self._subscribed_topics - desired

        if self._connected_event.is_set():
            if to_subscribe:
                self._send_command("subscribe", sorted(to_subscribe))
            if to_unsubscribe:
                self._send_command("unsubscribe", sorted(to_unsubscribe))
        else:
            logger.info("Queued WS topic sync | desired=%s | previous=%s", len(desired), len(current_desired))

    def wait_until_connected(self, timeout: float = 15.0) -> bool:
        return self._connected_event.wait(timeout=timeout)

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception:
                logger.exception("Bybit websocket loop failed")
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_sleep_sec)

    def _run_once(self) -> None:
        app = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = app
        app.run_forever()

    def _start_ping_loop(self) -> None:
        def ping_loop() -> None:
            while self._connected_event.is_set() and not self._stop_event.is_set():
                time.sleep(self.ping_interval_sec)
                if not self._connected_event.is_set() or self._stop_event.is_set():
                    break
                try:
                    self._send_raw({"op": "ping"})
                except Exception:
                    logger.exception("Failed to send websocket ping")
                    break

        self._ping_thread = threading.Thread(target=ping_loop, daemon=True, name="bybit-kline-ping")
        self._ping_thread.start()

    def _on_open(self, ws) -> None:
        logger.info("Bybit WS connected: %s", self.url)
        self._connected_event.set()
        with self._lock:
            desired = sorted(self._desired_topics)
            self._subscribed_topics.clear()
        if desired:
            self._send_command("subscribe", desired)
        self._start_ping_loop()

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Invalid WS JSON payload: %s", message[:200])
            return

        if payload.get("op") in {"ping", "pong", "subscribe", "unsubscribe"}:
            data = payload.get("data", {})
            success_topics = data.get("successTopics", [])
            if payload.get("op") == "subscribe":
                with self._lock:
                    self._subscribed_topics.update(success_topics or payload.get("args", []))
            elif payload.get("op") == "unsubscribe":
                with self._lock:
                    for topic in success_topics or payload.get("args", []):
                        self._subscribed_topics.discard(topic)
            return

        topic = payload.get("topic")
        if not topic or not topic.startswith("kline."):
            return

        data = payload.get("data", [])
        if not isinstance(data, list):
            return

        for row in data:
            if not row.get("confirm"):
                continue
            parts = topic.split(".")
            if len(parts) < 3:
                continue
            event = KlineClosedEvent(
                topic=topic,
                interval=str(parts[1]),
                symbol=str(parts[2]),
                start_ms=int(row["start"]),
                end_ms=int(row["end"]),
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                source_ts_ms=int(payload.get("ts", row.get("timestamp", row["end"]))),
            )
            self._queue.put(event)

    def _on_error(self, ws, error) -> None:
        if self._stop_event.is_set():
            return
        logger.warning("Bybit WS error: %s", error)

    def _on_close(self, ws, status_code, close_msg) -> None:
        self._connected_event.clear()
        with self._lock:
            self._subscribed_topics.clear()
        if not self._stop_event.is_set():
            logger.warning("Bybit WS closed | code=%s | msg=%s", status_code, close_msg)

    def _send_command(self, op: str, topics: list[str]) -> None:
        if not topics:
            return
        self._send_raw({"op": op, "args": topics})
        with self._lock:
            if op == "subscribe":
                self._subscribed_topics.update(topics)
            elif op == "unsubscribe":
                for topic in topics:
                    self._subscribed_topics.discard(topic)

    def _send_raw(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return
        ws.send(json.dumps(payload))
