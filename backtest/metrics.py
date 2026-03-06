"""백테스트 성과 지표 계산 모듈."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.portfolio import Portfolio, Trade
from strategy.signals import Signal

logger = logging.getLogger(__name__)

# 5분봉 기준 상수
CANDLE_MINUTES = 5
CANDLES_PER_DAY = 60 * 24 // CANDLE_MINUTES  # 288
TRADING_DAYS_PER_YEAR = 365


@dataclass
class BacktestMetrics:
    """백테스트 성과 지표."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r_multiple: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_hold_candles: float = 0.0
    avg_hold_minutes: float = 0.0

    # 롱/숏 별도 통계
    long_trades: int = 0
    long_win_rate: float = 0.0
    short_trades: int = 0
    short_win_rate: float = 0.0

    # 비용
    total_commission: float = 0.0
    total_slippage: float = 0.0


def calculate_metrics(portfolio: Portfolio) -> BacktestMetrics:
    """Portfolio 결과로부터 성과 지표를 계산한다.

    Args:
        portfolio: 백테스트 완료된 Portfolio.

    Returns:
        BacktestMetrics 객체.
    """
    metrics = BacktestMetrics()
    trades = portfolio.trades

    if not trades:
        logger.warning("거래 내역이 없습니다.")
        return metrics

    metrics.total_trades = len(trades)
    metrics.total_commission = portfolio.total_commission
    metrics.total_slippage = portfolio.total_slippage

    # 승패 분리
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    # 승률
    metrics.win_rate = len(wins) / len(trades) * 100

    # Profit Factor
    total_profit = sum(t.pnl for t in wins) if wins else 0
    total_loss = sum(abs(t.pnl) for t in losses) if losses else 0
    metrics.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # 평균 R배수
    r_multiples = [t.r_multiple for t in trades]
    metrics.avg_r_multiple = np.mean(r_multiples)

    # 최대 연속 승/패
    metrics.max_consecutive_wins = _max_consecutive(trades, win=True)
    metrics.max_consecutive_losses = _max_consecutive(trades, win=False)

    # 총 수익률
    metrics.total_return_pct = (
        (portfolio.capital - portfolio.initial_capital) / portfolio.initial_capital * 100
    )

    # CAGR
    if portfolio.equity_curve:
        n_candles = len(portfolio.equity_curve)
        n_days = n_candles / CANDLES_PER_DAY
        n_years = n_days / TRADING_DAYS_PER_YEAR
        if n_years > 0 and portfolio.capital > 0:
            metrics.cagr = (
                (portfolio.capital / portfolio.initial_capital) ** (1 / n_years) - 1
            ) * 100

    # MDD
    metrics.max_drawdown_pct = _calculate_mdd(portfolio.equity_curve)

    # Sharpe Ratio (일별 수익률 기준)
    metrics.sharpe_ratio = _calculate_sharpe(portfolio.equity_curve)

    # Calmar Ratio
    if metrics.max_drawdown_pct != 0:
        metrics.calmar_ratio = abs(metrics.cagr / metrics.max_drawdown_pct)

    # 평균 보유 시간
    hold_candles = [t.hold_candles for t in trades]
    metrics.avg_hold_candles = np.mean(hold_candles)
    metrics.avg_hold_minutes = metrics.avg_hold_candles * CANDLE_MINUTES

    # 롱/숏 별도 통계
    long_trades = [t for t in trades if t.direction == Signal.LONG]
    short_trades = [t for t in trades if t.direction == Signal.SHORT]

    metrics.long_trades = len(long_trades)
    if long_trades:
        metrics.long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) * 100

    metrics.short_trades = len(short_trades)
    if short_trades:
        metrics.short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) * 100

    return metrics


def _max_consecutive(trades: list[Trade], win: bool) -> int:
    """최대 연속 승 또는 패 횟수를 계산한다."""
    max_count = 0
    current = 0

    for t in trades:
        is_win = t.pnl > 0
        if is_win == win:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0

    return max_count


def _calculate_mdd(equity_curve: list[float]) -> float:
    """최대 낙폭(MDD)을 계산한다. 퍼센트로 반환."""
    if not equity_curve:
        return 0.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100

    return float(drawdown.min())


def _calculate_sharpe(equity_curve: list[float], risk_free: float = 0.0) -> float:
    """Sharpe Ratio를 일별 수익률 기준으로 계산한다."""
    if len(equity_curve) < 2:
        return 0.0

    equity = pd.Series(equity_curve)

    daily_equity = equity.iloc[::CANDLES_PER_DAY]

    if len(daily_equity) < 2:
        # 데이터가 부족하면 전체 봉 수익률로 계산
        returns = equity.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        annualization = np.sqrt(CANDLES_PER_DAY * TRADING_DAYS_PER_YEAR)
        return float(
            (returns.mean() - risk_free) / returns.std() * annualization
        )

    daily_returns = daily_equity.pct_change().dropna()

    if daily_returns.std() == 0:
        return 0.0

    return float(
        (daily_returns.mean() - risk_free) / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
