"""백테스트 결과 콘솔 리포트 생성 모듈."""

import logging

from backtest.metrics import BacktestMetrics
from config.settings import SETTINGS

logger = logging.getLogger(__name__)


def print_report(metrics: BacktestMetrics, start: str = "", end: str = "") -> str:
    """백테스트 결과를 포맷팅된 문자열로 생성하고 출력한다.

    Args:
        metrics: BacktestMetrics 객체.
        start: 백테스트 시작 날짜.
        end: 백테스트 종료 날짜.

    Returns:
        포맷팅된 리포트 문자열.
    """
    start = start or SETTINGS["backtest_start"]
    end = end or SETTINGS["backtest_end"]

    report = f"""
{'═' * 50}
  BTC 5min Multi-Confluence Momentum Scalp
  Backtest Report: {start} ~ {end}
{'═' * 50}
  Total Trades:        {metrics.total_trades}
  Win Rate:            {metrics.win_rate:.1f}%
  Profit Factor:       {metrics.profit_factor:.2f}
  Avg R-Multiple:      {metrics.avg_r_multiple:.2f}R
  Total Return:        {metrics.total_return_pct:+.1f}%
  CAGR:                {metrics.cagr:.1f}%
  Max Drawdown:        {metrics.max_drawdown_pct:.1f}%
  Sharpe Ratio:        {metrics.sharpe_ratio:.2f}
  Calmar Ratio:        {metrics.calmar_ratio:.2f}
  Avg Hold Time:       {metrics.avg_hold_minutes:.0f} min ({metrics.avg_hold_candles:.0f} candles)
  {'─' * 48}
  Long Trades:         {metrics.long_trades} (Win: {metrics.long_win_rate:.1f}%)
  Short Trades:        {metrics.short_trades} (Win: {metrics.short_win_rate:.1f}%)
  Max Consec Wins:     {metrics.max_consecutive_wins}
  Max Consec Losses:   {metrics.max_consecutive_losses}
  {'─' * 48}
  Commission Paid:     ${metrics.total_commission:.2f}
  Slippage Cost:       ${metrics.total_slippage:.2f}
{'═' * 50}
"""

    print(report)
    return report
