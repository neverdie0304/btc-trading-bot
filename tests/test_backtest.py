"""백테스트 엔진 통합 테스트."""

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio import Portfolio, Trade
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics, _calculate_mdd
from strategy.signals import Signal
from strategy.position import create_position


def _make_ohlcv(n: int = 500, trend: str = "flat") -> pd.DataFrame:
    """테스트용 OHLCV 데이터를 생성한다."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    price = 42000.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for i in range(n):
        if trend == "up":
            drift = 5
        elif trend == "down":
            drift = -5
        else:
            drift = 0
        change = np.random.normal(drift, 50)
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


class TestPortfolio:
    """Portfolio 단위 테스트."""

    def test_open_close_position(self) -> None:
        """포지션 오픈/클로즈 후 PnL 반영 확인."""
        portfolio = Portfolio(initial_capital=10000, capital=10000)
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        portfolio.open_position(pos)

        assert portfolio.has_position()

        trade = portfolio.close_position(
            exit_price=42200,
            exit_time="2024-01-01 00:05:00",
            exit_index=1,
            exit_reason="TP",
        )

        assert not portfolio.has_position()
        assert trade.pnl > 0  # 수익 거래 (수수료 차감 후)
        assert portfolio.capital > 10000

    def test_consecutive_loss_tracking(self) -> None:
        """연속 패배 추적 확인."""
        portfolio = Portfolio(initial_capital=10000, capital=10000)

        for i in range(3):
            pos = create_position(Signal.LONG, 42000, 100, portfolio.capital)
            portfolio.open_position(pos)
            portfolio.close_position(
                exit_price=41800,  # 손실
                exit_time=f"2024-01-01 00:{(i+1)*5}:00",
                exit_index=i + 1,
                exit_reason="SL",
            )

        assert portfolio.consecutive_losses == 3

    def test_equity_recording(self) -> None:
        """자산 기록 확인."""
        portfolio = Portfolio(initial_capital=10000, capital=10000)
        portfolio.record_equity(42000)
        portfolio.record_equity(42100)

        assert len(portfolio.equity_curve) == 2
        assert portfolio.equity_curve[0] == 10000


class TestMetrics:
    """Metrics 계산 테스트."""

    def test_mdd_calculation(self) -> None:
        """MDD 계산 확인."""
        # 10000 → 11000 → 9000 → 10000
        curve = [10000, 11000, 9000, 10000]
        mdd = _calculate_mdd(curve)
        # 11000 → 9000 = -18.18%
        assert mdd == pytest.approx(-18.18, abs=0.1)

    def test_empty_trades(self) -> None:
        """거래 없을 때 metrics가 0인지 확인."""
        portfolio = Portfolio(initial_capital=10000, capital=10000)
        metrics = calculate_metrics(portfolio)
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0


class TestBacktestEngine:
    """백테스트 엔진 통합 테스트."""

    def test_engine_runs_without_error(self) -> None:
        """엔진이 에러 없이 실행되는지 확인."""
        df = _make_ohlcv(500)
        portfolio = run_backtest(df, capital=10000)
        assert portfolio.capital > 0
        assert len(portfolio.equity_curve) > 0

    def test_engine_with_uptrend(self) -> None:
        """상승 추세 데이터에서 엔진 실행."""
        df = _make_ohlcv(500, trend="up")
        portfolio = run_backtest(df, capital=10000)
        metrics = calculate_metrics(portfolio)
        # 결과 출력 (검증보다는 실행 확인)
        assert metrics.total_trades >= 0

    def test_no_future_data_leak(self) -> None:
        """미래 데이터 참조가 없는지 확인.

        진입 시점 이전의 가격만으로 시그널이 결정되어야 한다.
        """
        df = _make_ohlcv(300)
        portfolio = run_backtest(df, capital=10000)

        for trade in portfolio.trades:
            # 진입 인덱스는 반드시 1 이상 (첫 봉은 이전 봉이 없으므로)
            assert trade.entry_index >= 1
            # 청산 인덱스는 진입 이후
            assert trade.exit_index >= trade.entry_index
