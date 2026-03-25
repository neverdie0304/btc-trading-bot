"""심볼별 캔들 DataFrame 관리 모듈.

각 심볼의 최근 N봉을 메모리에 유지하고,
WebSocket에서 수신한 캔들 데이터를 업데이트한다.
"""

import asyncio
import logging

import pandas as pd
from binance import AsyncClient

from scanner.config import SCANNER_SETTINGS

logger = logging.getLogger(__name__)

KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume"]


class CandleStore:
    """심볼별 캔들 DataFrame 저장소."""

    def __init__(self, max_candles: int | None = None) -> None:
        self._max = max_candles or SCANNER_SETTINGS["max_candles_per_symbol"]
        self._data: dict[str, pd.DataFrame] = {}
        # 현재 형성 중인 캔들의 open_time (close 전까지 덮어쓰기)
        self._current_candle_time: dict[str, int] = {}

    @property
    def symbols(self) -> list[str]:
        """관리 중인 심볼 목록."""
        return list(self._data.keys())

    def get_df(self, symbol: str) -> pd.DataFrame | None:
        """심볼의 캔들 DataFrame을 반환한다. 없으면 None."""
        df = self._data.get(symbol)
        if df is not None:
            return df.copy()
        return None

    def has_symbol(self, symbol: str) -> bool:
        """심볼 데이터 존재 여부."""
        return symbol in self._data

    def remove_symbol(self, symbol: str) -> None:
        """심볼 데이터를 제거한다."""
        self._data.pop(symbol, None)
        self._current_candle_time.pop(symbol, None)

    async def load_initial(
        self,
        client: AsyncClient,
        symbol: str,
        interval: str = "5m",
        count: int | None = None,
    ) -> None:
        """REST API로 심볼의 초기 캔들을 로드한다.

        Args:
            client: Binance AsyncClient.
            symbol: 심볼 (예: BTCUSDT).
            interval: 캔들 간격.
            count: 로드할 봉 수.
        """
        count = count or SCANNER_SETTINGS["initial_candle_count"]

        klines = await client.futures_klines(
            symbol=symbol,
            interval=interval,
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

        if not rows:
            logger.warning("%s: no candle data", symbol)
            return

        df = pd.DataFrame(rows)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)

        # 마지막 봉이 아직 진행 중이면 제거 (확정된 봉만 보관)
        if len(df) > 1:
            df = df.iloc[:-1]

        self._data[symbol] = df.tail(self._max).copy()
        logger.debug(
            "%s: loaded %d initial candles (%s ~ %s)",
            symbol, len(self._data[symbol]),
            self._data[symbol].index[0],
            self._data[symbol].index[-1],
        )

    async def load_all(
        self,
        client: AsyncClient,
        symbols: list[str],
        interval: str = "5m",
    ) -> None:
        """여러 심볼의 초기 캔들을 스로틀링하며 로드한다."""
        throttle = SCANNER_SETTINGS["startup_throttle_per_sec"]
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            try:
                await self.load_initial(client, symbol, interval)
            except Exception:
                logger.exception("%s: initial candle load failed", symbol)

            # 스로틀링
            if (i + 1) % throttle == 0 and i + 1 < total:
                await asyncio.sleep(1.0)

        logger.info("Initial candle load complete: %d/%d symbols", len(self._data), total)

    def update_candle(
        self,
        symbol: str,
        kline: dict,
    ) -> pd.DataFrame | None:
        """WebSocket kline 이벤트로 캔들을 업데이트한다.

        Args:
            symbol: 심볼.
            kline: WebSocket kline 데이터 (k 필드).

        Returns:
            캔들이 확정(close)되면 전체 DataFrame 반환, 아니면 None.
        """
        if symbol not in self._data:
            return None

        is_closed = kline.get("x", False)
        open_time = pd.Timestamp(kline["t"], unit="ms", tz="UTC")

        candle_data = {
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "quote_volume": float(kline.get("q", 0)),
        }

        df = self._data[symbol]

        if is_closed:
            # 확정된 캔들 추가
            close_time = pd.Timestamp(kline["T"], unit="ms", tz="UTC")
            candle_data["close_time"] = close_time
            new_row = pd.DataFrame([candle_data], index=[open_time])
            new_row.index.name = "open_time"

            # 같은 시간의 진행 중 캔들이 있으면 교체
            if open_time in df.index:
                df.drop(open_time, inplace=True)

            df = pd.concat([df, new_row])
            df.sort_index(inplace=True)
            df = df.tail(self._max)
            self._data[symbol] = df
            self._current_candle_time.pop(symbol, None)

            return df.copy()

        # 진행 중인 캔들 — DataFrame에 임시 업데이트 (가격 추적용)
        self._current_candle_time[symbol] = kline["t"]
        return None

    def get_current_price(self, symbol: str, kline: dict) -> float | None:
        """진행 중인 캔들에서 현재가를 추출한다."""
        return float(kline["c"]) if kline else None
