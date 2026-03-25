"""통합 WebSocket 스트림 모듈.

Binance combined stream으로 여러 심볼의 5분봉을 1개 연결로 수신한다.
"""

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

import websockets

from scanner.config import SCANNER_SETTINGS

logger = logging.getLogger(__name__)

# Binance Futures combined stream base URL
_WS_BASE = "wss://fstream.binance.com/stream?streams="


class MultiStream:
    """멀티 심볼 통합 WebSocket 스트림.

    Binance combined stream을 사용하여 1개 WebSocket 연결로
    여러 심볼의 kline 데이터를 수신한다.
    """

    def __init__(
        self,
        symbols: list[str],
        interval: str = "5m",
        on_candle_closed: Callable[[str, dict], Awaitable[None]] | None = None,
        on_price_update: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> None:
        """
        Args:
            symbols: 구독할 심볼 리스트.
            interval: 캔들 간격 (기본 5m).
            on_candle_closed: 캔들 확정 시 콜백 (symbol, kline_data).
            on_price_update: 가격 업데이트 콜백 (symbol, kline_data).
        """
        self._symbols = [s.lower() for s in symbols]
        self._interval = interval
        self.on_candle_closed = on_candle_closed
        self.on_price_update = on_price_update

        self._ws = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_msg_time: float = time.time()
        self._reconnect_count = 0

    @property
    def last_msg_time(self) -> float:
        """마지막 메시지 수신 시각."""
        return self._last_msg_time

    @property
    def is_running(self) -> bool:
        return self._running

    def _build_url(self) -> str:
        """combined stream URL을 생성한다."""
        streams = "/".join(
            f"{s}@kline_{self._interval}" for s in self._symbols
        )
        return _WS_BASE + streams

    def update_symbols(self, symbols: list[str]) -> None:
        """구독 심볼 목록을 업데이트한다. 다음 재연결 시 적용."""
        self._symbols = [s.lower() for s in symbols]
        logger.info("Symbol list updated: %d symbols", len(self._symbols))

    async def start(self) -> None:
        """WebSocket 스트림을 시작한다."""
        self._running = True
        self._task = asyncio.create_task(self._ws_loop())
        logger.info(
            "MultiStream started: %d symbols, interval=%s",
            len(self._symbols), self._interval,
        )

    async def stop(self) -> None:
        """WebSocket 스트림을 중지한다."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MultiStream stopped")

    async def _ws_loop(self) -> None:
        """WebSocket 수신 루프 (자동 재연결 포함)."""
        timeout = SCANNER_SETTINGS["ws_timeout_seconds"]

        while self._running:
            url = self._build_url()
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._reconnect_count = 0
                    logger.info(
                        "WebSocket connected (%d streams)",
                        len(self._symbols),
                    )

                    while self._running:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "WebSocket %ds timeout, reconnecting...", timeout,
                            )
                            break

                        self._last_msg_time = time.time()
                        await self._handle_message(raw)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("WebSocket disconnected: %s", e)
            except Exception:
                logger.exception("WebSocket error")

            if self._running:
                self._reconnect_count += 1
                delay = min(2 ** self._reconnect_count, 30)
                logger.info("Reconnecting in %.0fs... (attempt #%d)", delay, self._reconnect_count)
                await asyncio.sleep(delay)

    async def _handle_message(self, raw: str) -> None:
        """수신 메시지를 파싱하고 콜백을 호출한다."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # combined stream 형식: {"stream": "btcusdt@kline_5m", "data": {...}}
        data = msg.get("data")
        if not data:
            return

        event_type = data.get("e")
        if event_type != "kline":
            return

        kline = data.get("k", {})
        symbol = kline.get("s", "").upper()  # 심볼은 대문자로 통일
        is_closed = kline.get("x", False)

        if is_closed and self.on_candle_closed:
            try:
                await self.on_candle_closed(symbol, kline)
            except Exception:
                logger.exception("%s: on_candle_closed callback error", symbol)

        if not is_closed and self.on_price_update:
            try:
                await self.on_price_update(symbol, kline)
            except Exception:
                logger.exception("%s: on_price_update callback error", symbol)
