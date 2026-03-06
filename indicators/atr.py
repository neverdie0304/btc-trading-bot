"""ATR(Average True Range) 및 ADX 계산 모듈. Wilder's smoothing 방식."""

import pandas as pd


def calc_true_range(df: pd.DataFrame) -> pd.Series:
    """True Range를 계산한다.

    Args:
        df: high, low, close 컬럼을 가진 DataFrame.

    Returns:
        True Range 시리즈.
    """
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's smoothing 방식의 ATR을 계산한다.

    Args:
        df: high, low, close 컬럼을 가진 DataFrame.
        period: ATR 기간 (기본값 14).

    Returns:
        ATR 시리즈.
    """
    tr = calc_true_range(df)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX(Average Directional Index)를 계산한다.

    Args:
        df: high, low, close 컬럼을 가진 DataFrame.
        period: ADX 기간 (기본값 14).

    Returns:
        ADX 시리즈 (0~100).
    """
    high = df["high"]
    low = df["low"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calc_atr(df, period)

    alpha = 1 / period
    plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx
