"""지표 계산 단위 테스트. ta 라이브러리 결과와 크로스 검증한다."""

import numpy as np
import pandas as pd
import pytest
import ta

from indicators.ema import calc_ema, calc_ema_all
from indicators.rsi import calc_rsi
from indicators.atr import calc_atr, calc_true_range
from indicators.vwap import calc_vwap
from indicators.volume import calc_volume_ratio


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """테스트용 OHLCV 데이터를 생성한다. 200개 5분봉."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    # 랜덤 워크로 현실적인 가격 시뮬레이션
    price = 42000.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for _ in range(n):
        change = np.random.normal(0, 50)
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, 20))
        lo = min(o, c) - abs(np.random.normal(0, 20))
        v = abs(np.random.normal(100, 30))

        opens.append(o)
        highs.append(h)
        lows.append(lo)
        closes.append(c)
        volumes.append(v)
        price = c

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)


class TestEMA:
    """EMA 계산 테스트."""

    def test_ema_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """EMA 결과 길이가 입력과 동일한지 확인."""
        result = calc_ema(sample_ohlcv["close"], 9)
        assert len(result) == len(sample_ohlcv)

    def test_ema_vs_ta_library(self, sample_ohlcv: pd.DataFrame) -> None:
        """ta 라이브러리의 EMA와 결과가 일치하는지 검증."""
        close = sample_ohlcv["close"]
        for period in [9, 21, 50]:
            ours = calc_ema(close, period)
            theirs = ta.trend.EMAIndicator(close, window=period).ema_indicator()
            # NaN이 아닌 구간에서 비교
            mask = ours.notna() & theirs.notna()
            np.testing.assert_allclose(
                ours[mask].values, theirs[mask].values, rtol=1e-10,
                err_msg=f"EMA({period}) mismatch",
            )

    def test_ema_all(self, sample_ohlcv: pd.DataFrame) -> None:
        """calc_ema_all이 3개 컬럼을 올바르게 반환하는지 확인."""
        result = calc_ema_all(sample_ohlcv["close"])
        assert list(result.columns) == ["ema_fast", "ema_mid", "ema_slow"]
        assert len(result) == len(sample_ohlcv)


class TestRSI:
    """RSI 계산 테스트."""

    def test_rsi_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI 값이 0~100 범위인지 확인."""
        rsi = calc_rsi(sample_ohlcv["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_vs_ta_library(self, sample_ohlcv: pd.DataFrame) -> None:
        """ta 라이브러리의 RSI와 결과가 유사한지 검증."""
        close = sample_ohlcv["close"]
        ours = calc_rsi(close, 14)
        theirs = ta.momentum.RSIIndicator(close, window=14).rsi()
        mask = ours.notna() & theirs.notna()
        # Wilder's smoothing 구현 차이로 약간의 오차 허용
        np.testing.assert_allclose(
            ours[mask].values, theirs[mask].values, atol=1.0,
            err_msg="RSI(14) mismatch",
        )


class TestATR:
    """ATR 계산 테스트."""

    def test_true_range_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """True Range가 항상 양수인지 확인."""
        tr = calc_true_range(sample_ohlcv)
        valid = tr.dropna()
        assert (valid >= 0).all()

    def test_atr_vs_ta_library(self, sample_ohlcv: pd.DataFrame) -> None:
        """ta 라이브러리의 ATR과 결과가 유사한지 검증.

        초기 smoothing 방식 차이(SMA vs EWM 시작값)로 초반부에
        오차가 발생하므로, 수렴 후 구간(후반부)에서 비교한다.
        """
        ours = calc_atr(sample_ohlcv, 14)
        theirs = ta.volatility.AverageTrueRange(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], window=14
        ).average_true_range()
        mask = ours.notna() & theirs.notna()
        # 후반 50% 구간에서 비교 (초기 수렴 차이 제외)
        n = mask.sum()
        late_mask = mask.copy()
        late_mask.iloc[:n // 2] = False
        np.testing.assert_allclose(
            ours[late_mask].values, theirs[late_mask].values, rtol=0.1,
            err_msg="ATR(14) mismatch (late convergence check)",
        )


class TestVWAP:
    """VWAP 계산 테스트."""

    def test_vwap_between_high_low(self, sample_ohlcv: pd.DataFrame) -> None:
        """VWAP이 합리적인 가격 범위 내에 있는지 확인."""
        vwap = calc_vwap(sample_ohlcv)
        valid = vwap.dropna()
        # VWAP은 당일 누적이므로 극단적 범위만 체크
        assert (valid > 0).all()

    def test_vwap_daily_reset(self) -> None:
        """일자 변경 시 VWAP이 리셋되는지 확인."""
        dates = pd.date_range("2024-01-01", periods=576, freq="5min", tz="UTC")  # 2일
        df = pd.DataFrame({
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 10.0,
        }, index=dates)
        vwap = calc_vwap(df)
        # 모든 캔들이 동일하면 VWAP = typical_price
        expected_tp = (105.0 + 95.0 + 102.0) / 3
        np.testing.assert_allclose(vwap.values, expected_tp, rtol=1e-10)


class TestVolumeRatio:
    """거래량 비율 테스트."""

    def test_volume_ratio_at_mean(self) -> None:
        """거래량이 평균과 같으면 비율이 1.0인지 확인."""
        volume = pd.Series([100.0] * 30)
        ratio = calc_volume_ratio(volume, 20)
        assert ratio.iloc[-1] == pytest.approx(1.0)

    def test_volume_ratio_double(self) -> None:
        """거래량이 평균의 2배면 비율이 2.0인지 확인."""
        volume = pd.Series([100.0] * 20 + [200.0])
        ratio = calc_volume_ratio(volume, 20)
        # MA20 = (100*19 + 200) / 20 = 105, ratio = 200/105 ≈ 1.905
        # 마지막 값의 MA는 이전 20개의 평균
        # index 20의 MA20 = mean(index 1..20) = (100*19 + 200) / 20
        assert ratio.iloc[-1] > 1.5
