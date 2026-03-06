"""6가지 진입 조건을 판별하여 LONG / SHORT / NO_SIGNAL을 반환하는 모듈."""

import logging
from enum import Enum

import pandas as pd

from config.settings import SETTINGS
from indicators.ema import calc_ema_all
from indicators.rsi import calc_rsi
from indicators.vwap import calc_vwap
from indicators.atr import calc_atr
from indicators.volume import calc_volume_ratio

logger = logging.getLogger(__name__)


class Signal(Enum):
    """매매 시그널."""
    LONG = "LONG"
    SHORT = "SHORT"
    NO_SIGNAL = "NO_SIGNAL"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에 모든 지표를 추가한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        지표가 추가된 DataFrame (원본을 수정하지 않는다).
    """
    df = df.copy()

    # EMA
    emas = calc_ema_all(
        df["close"],
        fast=SETTINGS["ema_fast"],
        mid=SETTINGS["ema_mid"],
        slow=SETTINGS["ema_slow"],
    )
    df["ema_fast"] = emas["ema_fast"]
    df["ema_mid"] = emas["ema_mid"]
    df["ema_slow"] = emas["ema_slow"]

    # RSI
    df["rsi"] = calc_rsi(df["close"], SETTINGS["rsi_period"])

    # VWAP
    df["vwap"] = calc_vwap(df)

    # ATR
    df["atr"] = calc_atr(df, SETTINGS["atr_period"])

    # ATR 중앙값 (lookback 기간)
    df["atr_median"] = df["atr"].rolling(
        window=SETTINGS["atr_median_lookback"]
    ).median()

    # Volume ratio
    df["volume_ratio"] = calc_volume_ratio(
        df["volume"], SETTINGS["volume_ma_period"]
    )

    return df


def check_long_conditions(row: pd.Series, prev_row: pd.Series) -> bool:
    """LONG 진입 6가지 조건을 모두 확인한다.

    Args:
        row: 현재 봉의 지표 데이터.
        prev_row: 이전 봉의 지표 데이터.

    Returns:
        모든 조건 충족 시 True.
    """
    # 1. 추세 정배열: EMA9 > EMA21 > EMA50
    if not (row["ema_fast"] > row["ema_mid"] > row["ema_slow"]):
        return False

    # 2. VWAP 위: close > VWAP
    if not (row["close"] > row["vwap"]):
        return False

    # 3. RSI 범위: 40 <= RSI <= 65 AND RSI 상승 중
    if not (SETTINGS["rsi_long_min"] <= row["rsi"] <= SETTINGS["rsi_long_max"]):
        return False
    if not (row["rsi"] > prev_row["rsi"]):
        return False

    # 4. EMA9 돌파: close > EMA9 AND prev_close <= EMA9
    if not (row["close"] > row["ema_fast"] and prev_row["close"] <= prev_row["ema_fast"]):
        return False

    # 5. 거래량 확인: volume > volume_ma20 × 1.2
    if not (row["volume_ratio"] > SETTINGS["volume_threshold"]):
        return False

    # 6. 변동성 확인: ATR > median(ATR, 50) × atr_signal_multiplier
    if pd.isna(row["atr_median"]):
        return False
    atr_mult = SETTINGS.get("atr_signal_multiplier", 1.0)
    if not (row["atr"] > row["atr_median"] * atr_mult):
        return False

    return True


def check_short_conditions(row: pd.Series, prev_row: pd.Series) -> bool:
    """SHORT 진입 6가지 조건을 모두 확인한다.

    Args:
        row: 현재 봉의 지표 데이터.
        prev_row: 이전 봉의 지표 데이터.

    Returns:
        모든 조건 충족 시 True.
    """
    # 1. 추세 역배열: EMA9 < EMA21 < EMA50
    if not (row["ema_fast"] < row["ema_mid"] < row["ema_slow"]):
        return False

    # 2. VWAP 아래: close < VWAP
    if not (row["close"] < row["vwap"]):
        return False

    # 3. RSI 범위: 35 <= RSI <= 60 AND RSI 하락 중
    if not (SETTINGS["rsi_short_min"] <= row["rsi"] <= SETTINGS["rsi_short_max"]):
        return False
    if not (row["rsi"] < prev_row["rsi"]):
        return False

    # 4. EMA9 이탈: close < EMA9 AND prev_close >= EMA9
    if not (row["close"] < row["ema_fast"] and prev_row["close"] >= prev_row["ema_fast"]):
        return False

    # 5. 거래량 확인
    if not (row["volume_ratio"] > SETTINGS["volume_threshold"]):
        return False

    # 6. 변동성 확인
    if pd.isna(row["atr_median"]):
        return False
    atr_mult = SETTINGS.get("atr_signal_multiplier", 1.0)
    if not (row["atr"] > row["atr_median"] * atr_mult):
        return False

    return True


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """전체 DataFrame에 대해 봉별 시그널을 생성한다.

    캔들 종가 확정 시점 기준으로 판단한다.

    Args:
        df: 지표가 포함된 DataFrame (compute_indicators 결과).

    Returns:
        Signal enum 값의 시리즈.
    """
    signals = pd.Series(Signal.NO_SIGNAL, index=df.index)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # NaN 지표가 있으면 스킵
        if pd.isna(row["atr"]) or pd.isna(row["rsi"]) or pd.isna(row["volume_ratio"]):
            continue

        if check_long_conditions(row, prev_row):
            signals.iloc[i] = Signal.LONG
        elif check_short_conditions(row, prev_row):
            signals.iloc[i] = Signal.SHORT

    long_count = (signals == Signal.LONG).sum()
    short_count = (signals == Signal.SHORT).sum()
    logger.info("시그널 생성 완료: LONG %d, SHORT %d, 총 %d봉",
                long_count, short_count, len(df))

    return signals
