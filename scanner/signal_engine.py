"""시그널 탐지 엔진 모듈.

기존 strategy 함수를 래핑하여 멀티 심볼 시그널 탐지를 수행한다.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from strategy.signals import Signal, compute_indicators, check_long_conditions, check_short_conditions
from strategy.filters import should_filter
from scanner.config import SCANNER_SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    """탐지된 시그널 이벤트."""
    symbol: str
    direction: Signal
    timestamp: pd.Timestamp
    close: float
    atr: float
    atr_median: float
    rsi: float
    volume_ratio: float
    score: float = 0.0  # Prioritizer가 설정


class SignalEngine:
    """멀티 심볼 시그널 탐지 엔진.

    각 심볼의 캔들 DataFrame에 대해 지표 계산 → 시그널 체크 → 필터 적용을 수행한다.
    """

    def process_candle(
        self,
        symbol: str,
        df: pd.DataFrame,
        consecutive_losses: int = 0,
        candles_since_last_loss: int = 999,
    ) -> SignalEvent | None:
        """캔들 확정 시 시그널을 체크한다.

        Args:
            symbol: 심볼.
            df: OHLCV DataFrame (최소 50봉 이상).
            consecutive_losses: 해당 심볼의 연속 패배 수.
            candles_since_last_loss: 마지막 패배 이후 봉 수.

        Returns:
            시그널이 발생하면 SignalEvent, 아니면 None.
        """
        if len(df) < 50:
            return None

        try:
            df_ind = compute_indicators(df)
        except Exception:
            logger.exception("%s: indicator calculation failed", symbol)
            return None

        latest = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]

        # NaN 체크
        if pd.isna(latest["atr"]) or pd.isna(latest["rsi"]) or pd.isna(latest["volume_ratio"]):
            return None

        timestamp = df_ind.index[-1]
        atr = latest["atr"]
        atr_median = latest["atr_median"] if not pd.isna(latest["atr_median"]) else atr

        # ATR/Price 변동성 필터
        min_atr_pct = SCANNER_SETTINGS.get("min_atr_pct")
        if min_atr_pct and latest["close"] > 0:
            atr_pct = atr / latest["close"] * 100
            if atr_pct < min_atr_pct:
                return None

        # 필터 체크
        if should_filter(
            timestamp=timestamp,
            atr=atr,
            atr_median=atr_median,
            consecutive_losses=consecutive_losses,
            candles_since_last_loss=candles_since_last_loss,
        ):
            return None

        # 시그널 체크
        direction = Signal.NO_SIGNAL
        if check_long_conditions(latest, prev):
            direction = Signal.LONG
        elif check_short_conditions(latest, prev):
            direction = Signal.SHORT

        if direction == Signal.NO_SIGNAL:
            return None

        event = SignalEvent(
            symbol=symbol,
            direction=direction,
            timestamp=timestamp,
            close=latest["close"],
            atr=atr,
            atr_median=atr_median,
            rsi=latest["rsi"],
            volume_ratio=latest["volume_ratio"],
        )

        logger.info(
            "[SIGNAL] %s %s | Close: %g | ATR: %g | RSI: %.1f | Vol: %.2fx",
            symbol, direction.value, event.close, atr, event.rsi, event.volume_ratio,
        )

        return event
