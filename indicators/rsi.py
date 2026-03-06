"""RSI(Relative Strength Index) 계산 모듈. Wilder's smoothing 방식."""

import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothing 방식의 RSI를 계산한다.

    Args:
        close: 종가 시리즈.
        period: RSI 기간 (기본값 14).

    Returns:
        RSI 시리즈 (0~100).
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # 첫 번째 평균: 단순 평균
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
