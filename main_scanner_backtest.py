"""멀티코인 스캐너 백테스트 실행 진입점.

사용법:
    # 2025년 백테스트
    python main_scanner_backtest.py --start 2025-01-01 --end 2025-12-31

    # 데이터만 수집
    python main_scanner_backtest.py --fetch-only --start 2025-01-01 --end 2025-12-31

    # 자본금/포지션 수 변경
    python main_scanner_backtest.py --capital 10000 --max-positions 3
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from scanner.config import SCANNER_SETTINGS
from scanner.data_collector import (
    fetch_all_futures_symbols,
    fetch_daily_volumes,
    get_daily_active_symbols,
    fetch_5m_data_for_symbols,
)
from scanner.backtest_engine import run_multi_backtest, MultiPortfolio, MultiTrade
from strategy.signals import Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_report(portfolio: MultiPortfolio, start: str, end: str) -> None:
    """백테스트 결과 리포트를 출력한다."""
    trades = portfolio.trades
    if not trades:
        print("거래 없음")
        return

    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / total * 100

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0
    total_return = (portfolio.capital / portfolio.initial_capital - 1) * 100

    # MDD
    eq = portfolio.equity_curve
    if eq:
        peak = eq[0]
        mdd = 0
        for e in eq:
            if e > peak:
                peak = e
            dd = (e - peak) / peak * 100
            if dd < mdd:
                mdd = dd
    else:
        mdd = 0

    # 연환산
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    years = days / 365
    cagr = ((portfolio.capital / portfolio.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe (일별 수익률)
    if len(eq) > 288:  # 최소 1일
        eq_arr = np.array(eq)
        daily_eq = eq_arr[::288]  # 5분봉 288개 = 1일
        daily_ret = np.diff(daily_eq) / daily_eq[:-1]
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(365) if np.std(daily_ret) > 0 else 0
    else:
        sharpe = 0

    calmar = abs(cagr / mdd) if mdd != 0 else 0

    # 연속 승/패
    max_consec_w = max_consec_l = 0
    cur_w = cur_l = 0
    for t in trades:
        if t.pnl > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_consec_w = max(max_consec_w, cur_w)
        max_consec_l = max(max_consec_l, cur_l)

    # 롱/숏 분리
    longs = [t for t in trades if t.direction == Signal.LONG]
    shorts = [t for t in trades if t.direction == Signal.SHORT]
    long_wr = len([t for t in longs if t.pnl > 0]) / len(longs) * 100 if longs else 0
    short_wr = len([t for t in shorts if t.pnl > 0]) / len(shorts) * 100 if shorts else 0

    avg_hold = np.mean([t.hold_candles for t in trades]) if trades else 0

    # 심볼별 통계
    sym_trades: dict[str, list] = {}
    for t in trades:
        sym_trades.setdefault(t.symbol, []).append(t)

    top_symbols = sorted(sym_trades.items(), key=lambda x: sum(t.pnl for t in x[1]), reverse=True)

    print()
    print("═" * 55)
    print("  Multi-Coin Scanner Backtest Report")
    print(f"  Period: {start} ~ {end}")
    print("═" * 55)
    print(f"  Total Trades:        {total}")
    print(f"  Win Rate:            {win_rate:.1f}%")
    print(f"  Profit Factor:       {pf:.2f}")
    print(f"  Avg R-Multiple:      {avg_r:.2f}R")
    print(f"  Total Return:        {total_return:+.1f}%")
    print(f"  CAGR:                {cagr:.1f}%")
    print(f"  Max Drawdown:        {mdd:.1f}%")
    print(f"  Sharpe Ratio:        {sharpe:.2f}")
    print(f"  Calmar Ratio:        {calmar:.2f}")
    print(f"  Avg Hold Time:       {avg_hold:.0f} candles ({avg_hold * 5:.0f} min)")
    print("─" * 55)
    print(f"  Long Trades:         {len(longs)} (Win: {long_wr:.1f}%)")
    print(f"  Short Trades:        {len(shorts)} (Win: {short_wr:.1f}%)")
    print(f"  Max Consec Wins:     {max_consec_w}")
    print(f"  Max Consec Losses:   {max_consec_l}")
    print("─" * 55)
    print(f"  Initial Capital:     ${portfolio.initial_capital:,.2f}")
    print(f"  Final Capital:       ${portfolio.capital:,.2f}")
    print(f"  Commission Paid:     ${portfolio.total_commission:,.2f}")
    print(f"  Unique Symbols:      {len(sym_trades)}")
    print("─" * 55)
    print("  Top 10 Symbols by PnL:")
    for sym, tlist in top_symbols[:10]:
        sym_pnl = sum(t.pnl for t in tlist)
        sym_cnt = len(tlist)
        sym_wr = len([t for t in tlist if t.pnl > 0]) / sym_cnt * 100
        print(f"    {sym:>12s}: {sym_pnl:>+10.2f} ({sym_cnt} trades, {sym_wr:.0f}% win)")
    print("─" * 55)
    print("  Bottom 5 Symbols:")
    for sym, tlist in top_symbols[-5:]:
        sym_pnl = sum(t.pnl for t in tlist)
        sym_cnt = len(tlist)
        print(f"    {sym:>12s}: {sym_pnl:>+10.2f} ({sym_cnt} trades)")
    print("═" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Coin Scanner Backtest")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--capital", type=float, default=SETTINGS["initial_capital"])
    parser.add_argument("--max-positions", type=int, default=SCANNER_SETTINGS["max_concurrent_positions"])
    parser.add_argument("--fetch-only", action="store_true", help="데이터 수집만 하고 백테스트 안 함")
    parser.add_argument("--min-volume", type=float, default=SCANNER_SETTINGS["min_24h_quote_volume"])
    args = parser.parse_args()

    t0 = time.time()

    # ── 1단계: 전체 심볼 목록 ──
    logger.info("=" * 60)
    logger.info("  Multi-Coin Scanner Backtest")
    logger.info("  Period: %s ~ %s | Capital: $%.2f | Max Pos: %d",
                args.start, args.end, args.capital, args.max_positions)
    logger.info("=" * 60)

    logger.info("1단계: 전체 USDT-M 선물 심볼 조회...")
    all_symbols = fetch_all_futures_symbols()

    # ── 2단계: 일봉 거래대금 수집 ──
    logger.info("2단계: 일봉 거래대금 수집...")
    daily_vol = fetch_daily_volumes(all_symbols, args.start, args.end)
    if daily_vol.empty:
        logger.error("일봉 데이터 수집 실패")
        return

    # ── 3단계: 일별 활성 심볼 결정 ──
    logger.info("3단계: 일별 활성 심볼 결정...")
    daily_active = get_daily_active_symbols(daily_vol, min_volume=args.min_volume)

    # 고유 심볼
    unique_symbols = set()
    for syms in daily_active.values():
        unique_symbols.update(syms)
    unique_list = sorted(unique_symbols)
    logger.info("수집 대상 고유 심볼: %d개", len(unique_list))

    # ── 4단계: 5분봉 수집 ──
    logger.info("4단계: 5분봉 데이터 수집 (캐시 활용)...")
    all_data = fetch_5m_data_for_symbols(unique_list, args.start, args.end)
    logger.info("5분봉 데이터 로드 완료: %d개 심볼", len(all_data))

    t_fetch = time.time() - t0
    logger.info("데이터 수집 소요: %.1f초", t_fetch)

    if args.fetch_only:
        logger.info("--fetch-only: 데이터 수집만 완료")
        return

    # ── 5단계: 백테스트 실행 ──
    logger.info("5단계: 멀티심볼 백테스트 실행...")
    t1 = time.time()

    portfolio = run_multi_backtest(
        all_data=all_data,
        daily_active=daily_active,
        capital=args.capital,
        max_positions=args.max_positions,
    )

    t_bt = time.time() - t1
    logger.info("백테스트 소요: %.1f초", t_bt)

    # ── 6단계: 결과 출력 ──
    print_report(portfolio, args.start, args.end)

    total_time = time.time() - t0
    logger.info("전체 소요: %.1f초", total_time)


if __name__ == "__main__":
    main()
