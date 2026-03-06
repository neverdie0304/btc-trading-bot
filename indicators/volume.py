"""거래량 이동평균 대비 비율 계산 모듈."""

import pandas as pd


def calc_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """현재 거래량의 SMA(period) 대비 비율을 계산한다.

    Args:
        volume: 거래량 시리즈.
        period: 이동평균 기간 (기본값 20).

    Returns:
        거래량 비율 시리즈 (1.0 = 평균과 동일).
    """
    volume_ma = volume.rolling(window=period).mean()
    return volume / volume_ma
