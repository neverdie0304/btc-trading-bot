"""EMA(Exponential Moving Average) 계산 모듈."""

import pandas as pd


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    """EMA를 계산한다.

    Args:
        close: 종가 시리즈.
        period: EMA 기간.

    Returns:
        EMA 시리즈.
    """
    return close.ewm(span=period, adjust=False).mean()


def calc_ema_all(close: pd.Series, fast: int = 9, mid: int = 21, slow: int = 50) -> pd.DataFrame:
    """EMA 9, 21, 50을 한 번에 계산한다.

    Args:
        close: 종가 시리즈.
        fast: 단기 EMA 기간.
        mid: 중기 EMA 기간.
        slow: 장기 EMA 기간.

    Returns:
        ema_fast, ema_mid, ema_slow 컬럼을 가진 DataFrame.
    """
    return pd.DataFrame({
        "ema_fast": calc_ema(close, fast),
        "ema_mid": calc_ema(close, mid),
        "ema_slow": calc_ema(close, slow),
    }, index=close.index)
