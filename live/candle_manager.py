"""실시간 캔들 관리 모듈.

WebSocket으로 5분봉 캔들을 수신하고, 지표 계산에 필요한 최근 N봉을 메모리에 유지한다.
"""

import asyncio
import logging
import time

import pandas as pd
from binance import AsyncClient, BinanceSocketManager

from config.settings import SETTINGS

logger = logging.getLogger(__name__)

KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume"]
MAX_CANDLES = 200  # 메모리에 유지할 최대 봉 수


class CandleManager:
    """실시간 캔들 관리자.

    REST API로 초기 히스토리를 로드하고,
    WebSocket으로 실시간 봉을 수신하여 DataFrame을 유지한다.
    """

    def __init__(
        self,
        client: AsyncClient,
        symbol: str | None = None,
        interval: str | None = None,
        on_candle_closed: "asyncio.coroutines | None" = None,
        on_price_update: "asyncio.coroutines | None" = None,
    ) -> None:
        self.client = client
        self.symbol = symbol or SETTINGS["symbol"]
        self.interval = interval or SETTINGS["interval"]
        self.on_candle_closed = on_candle_closed
        self.on_price_update = on_price_update

        self._df: pd.DataFrame | None = None
        self._bm: BinanceSocketManager | None = None
        self._ws_task: asyncio.Task | None = None
        self._last_msg_time: float = time.time()
        self._running = False

    @property
    def last_msg_time(self) -> float:
        """마지막 WebSocket 메시지 수신 시각 (epoch)."""
        return self._last_msg_time

    def get_dataframe(self) -> pd.DataFrame:
        """현재 캔들 DataFrame을 반환한다."""
        if self._df is None:
            raise RuntimeError("캔들 데이터가 로드되지 않았습니다.")
        return self._df.copy()

    async def load_initial_candles(self, count: int = 100) -> None:
        """REST API로 최근 N봉을 로드한다 (지표 warm-up 용)."""
        logger.info("Loading %d initial candles: %s %s", count, self.symbol, self.interval)

        klines = await self.client.futures_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=count,
        )

        rows = []
        for k in klines:
            rows.append({
                "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": pd.Timestamp(k[6], unit="ms", tz="UTC"),
                "quote_volume": float(k[7]),
            })

        df = pd.DataFrame(rows)
        df = df.set_index("open_time")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self._df = df

        logger.info(
            "Initial candles loaded: %d candles, %s ~ %s",
            len(df), df.index[0], df.index[-1],
        )

    async def start_websocket(self) -> None:
        """WebSocket 스트림을 시작한다."""
        self._running = True
        self._bm = BinanceSocketManager(self.client)
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("WebSocket started: %s %s", self.symbol, self.interval)

    async def stop(self) -> None:
        """WebSocket을 중지한다."""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket stopped")

    async def _ws_loop(self) -> None:
        """WebSocket 수신 루프."""
        reconnect_delay = 1

        while self._running:
            try:
                ts = self._bm.kline_futures_socket(
                    symbol=self.symbol, interval=self.interval
                )
                async with ts as stream:
                    reconnect_delay = 1  # 연결 성공 시 리셋
                    logger.info("WebSocket connected")

                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=120)
                        self._last_msg_time = time.time()

                        if msg.get("e") == "error":
                            logger.error("WebSocket error: %s", msg)
                            break

                        await self._process_kline_msg(msg)

            except asyncio.TimeoutError:
                logger.warning("WebSocket 120s timeout, reconnecting...")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("WebSocket error, reconnecting in %ds", reconnect_delay)

            if self._running:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
                # 재연결 시 누락 봉 보충
                await self._fill_missing_candles()

    async def _process_kline_msg(self, msg: dict) -> None:
        """WebSocket kline 메시지를 처리한다."""
        k = msg.get("k")
        if k is None:
            return

        open_time = pd.Timestamp(k["t"], unit="ms", tz="UTC")
        is_closed = k["x"]  # 봉 확정 여부

        candle = {
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "quote_volume": float(k["q"]),
        }

        if self._df is None:
            return

        if is_closed:
            # 봉 확정: DataFrame에 추가
            new_row = pd.DataFrame(
                [candle],
                index=pd.DatetimeIndex([open_time], name="open_time"),
            )

            # close_time은 별도 저장
            new_row["close_time"] = pd.Timestamp(k["T"], unit="ms", tz="UTC")

            if open_time in self._df.index:
                self._df.loc[open_time] = new_row.iloc[0]
            else:
                self._df = pd.concat([self._df, new_row])

            # 최대 봉 수 유지
            if len(self._df) > MAX_CANDLES:
                self._df = self._df.iloc[-MAX_CANDLES:]

            logger.debug(
                "Candle closed: %s close=%.2f vol=%.2f",
                open_time, candle["close"], candle["volume"],
            )

            # 콜백 호출
            if self.on_candle_closed:
                await self.on_candle_closed(self._df)
        else:
            # 진행 중인 봉: 마지막 행 업데이트 (시각화 등에 사용 가능)
            if open_time in self._df.index:
                for col in KLINE_COLUMNS:
                    self._df.at[open_time, col] = candle[col]

            # 실시간 가격 콜백 (트레일링 스탑 등)
            if self.on_price_update:
                await self.on_price_update(candle["close"])

    async def _fill_missing_candles(self) -> None:
        """재연결 후 누락된 봉을 REST API로 보충한다."""
        if self._df is None or self._df.empty:
            await self.load_initial_candles()
            return

        last_time = self._df.index[-1]
        now = pd.Timestamp.now(tz="UTC")

        # 마지막 봉과 현재 시간 차이가 2봉 이상이면 보충
        interval_mins = 5  # 5분봉
        diff_mins = (now - last_time).total_seconds() / 60

        if diff_mins > interval_mins * 2:
            count = min(int(diff_mins / interval_mins) + 5, MAX_CANDLES)
            logger.info("Filling missing candles: %d requested", count)

            klines = await self.client.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=int(last_time.timestamp() * 1000),
                limit=count,
            )

            rows = []
            for k in klines:
                rows.append({
                    "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": pd.Timestamp(k[6], unit="ms", tz="UTC"),
                    "quote_volume": float(k[7]),
                })

            if rows:
                new_df = pd.DataFrame(rows).set_index("open_time")
                self._df = pd.concat([self._df, new_df])
                self._df = self._df[~self._df.index.duplicated(keep="last")]
                self._df = self._df.sort_index().iloc[-MAX_CANDLES:]
                logger.info("Missing candles filled: now %d candles", len(self._df))
