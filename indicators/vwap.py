"""당일 리셋 VWAP(Volume Weighted Average Price) 계산 모듈."""

import pandas as pd


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """당일 00:00 UTC 기준으로 리셋되는 VWAP을 계산한다.

    Args:
        df: open, high, low, close, volume 컬럼을 가진 OHLCV DataFrame.
            인덱스는 UTC DatetimeIndex여야 한다.

    Returns:
        VWAP 시리즈.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    # 날짜별 그룹으로 cumsum 리셋
    date_groups = df.index.date
    cum_tp_volume = tp_volume.groupby(date_groups).cumsum()
    cum_volume = df["volume"].groupby(date_groups).cumsum()

    vwap = cum_tp_volume / cum_volume

    # volume이 0인 구간은 typical_price로 대체
    vwap = vwap.fillna(typical_price)

    return vwap
