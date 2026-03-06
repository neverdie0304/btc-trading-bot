"""1분봉, 3분봉, 5분봉 타임프레임 비교 스크립트.

2025년 BTC 데이터로 각 타임프레임별 백테스트 결과를 비교한다.
"""

import logging
import sys
import time

import numpy as np
import pandas as pd
from tabulate import tabulate
from copy import deepcopy

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from data.fetcher import fetch_klines, validate_data
from data.storage import save_to_parquet, load_from_parquet
from strategy.signals import compute_indicators, Signal
from optimize import generate_relaxed_signals, fast_backtest
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

START = "2025-01-01"
END = "2026-02-25"
INTERVALS = ["1m", "3m", "5m"]


def ensure_data(symbol: str, interval: str) -> pd.DataFrame | None:
    """캐시에 데이터가 없으면 Binance에서 수집한다."""
    df = load_from_parquet(symbol=symbol, interval=interval, start=START, end=END)
    if df is not None and len(df) > 1000:
        print(f"  [{interval}] 캐시 로드: {len(df):,}개 캔들")
        return df

    print(f"  [{interval}] 데이터 수집 중... (Binance API)")
    try:
        df = fetch_klines(symbol=symbol, interval=interval, start=START, end=END)
        if df.empty:
            print(f"  [{interval}] 데이터 수집 실패")
            return None
        df = validate_data(df)
        save_to_parquet(df, symbol=symbol, interval=interval)
        # 날짜 필터링
        start_ts = pd.Timestamp(START, tz="UTC")
        end_ts = pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df.index >= start_ts) & (df.index < end_ts)]
        print(f"  [{interval}] 수집 완료: {len(df):,}개 캔들")
        return df
    except Exception as e:
        print(f"  [{interval}] 수집 실패: {e}")
        return None


def run_backtest_for_interval(df: pd.DataFrame, interval: str) -> dict:
    """특정 타임프레임에 대해 백테스트를 실행하고 결과를 반환한다."""
    params = deepcopy(SETTINGS)
    params["interval"] = interval

    # 쿨다운 캔들 수는 타임프레임에 따라 조정 (1시간 기준)
    if interval == "1m":
        params["cooldown_candles"] = 60   # 60분 = 60봉
    elif interval == "3m":
        params["cooldown_candles"] = 20   # 60분 = 20봉
    else:
        params["cooldown_candles"] = 12   # 60분 = 12봉

    print(f"  [{interval}] 지표 계산 중... ({len(df):,} 캔들)")
    df_ind = compute_indicators(df)

    print(f"  [{interval}] 시그널 생성 중...")
    base_signals = generate_relaxed_signals(df_ind)
    long_count = (base_signals == Signal.LONG).sum()
    short_count = (base_signals == Signal.SHORT).sum()
    print(f"  [{interval}] 후보 시그널: LONG {long_count}, SHORT {short_count}")

    sig_arr = np.zeros(len(df_ind), dtype=np.int8)
    sig_arr[base_signals == Signal.LONG] = 1
    sig_arr[base_signals == Signal.SHORT] = 2

    precomputed = {
        "close": df_ind["close"].values,
        "high": df_ind["high"].values,
        "low": df_ind["low"].values,
        "atr": df_ind["atr"].values,
        "atr_median": df_ind["atr_median"].values,
        "rsi": df_ind["rsi"].values,
        "volume_ratio": df_ind["volume_ratio"].values,
        "timestamps": df_ind.index,
        "weekdays": df_ind.index.weekday.values,
        "hours": df_ind.index.hour.values,
        "signals": sig_arr,
    }

    print(f"  [{interval}] 백테스트 실행 중...")
    t0 = time.time()
    portfolio = fast_backtest(df_ind, base_signals, params, precomputed=precomputed)
    elapsed = time.time() - t0

    metrics = calculate_metrics(portfolio)

    # LONG/SHORT 분리
    long_trades = [t for t in portfolio.trades if t.direction == Signal.LONG]
    short_trades = [t for t in portfolio.trades if t.direction == Signal.SHORT]
    long_wins = sum(1 for t in long_trades if t.pnl > 0)
    short_wins = sum(1 for t in short_trades if t.pnl > 0)

    result = {
        "interval": interval,
        "candles": len(df),
        "trades": metrics.total_trades,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "win_rate": round(metrics.win_rate, 1),
        "long_wr": round(long_wins / len(long_trades) * 100, 1) if long_trades else 0,
        "short_wr": round(short_wins / len(short_trades) * 100, 1) if short_trades else 0,
        "pf": round(metrics.profit_factor, 2),
        "total_return": round(metrics.total_return_pct, 1),
        "cagr": round(metrics.cagr, 1),
        "mdd": round(metrics.max_drawdown_pct, 1),
        "sharpe": round(metrics.sharpe_ratio, 2),
        "calmar": round(metrics.calmar_ratio, 2),
        "avg_r": round(metrics.avg_r_multiple, 2),
        "max_consec_w": metrics.max_consecutive_wins,
        "max_consec_l": metrics.max_consecutive_losses,
        "elapsed": round(elapsed, 1),
    }

    return result


def main():
    symbol = "BTCUSDT"
    print(f"\n{'='*80}")
    print(f"  BTC 타임프레임 비교: 1분봉 vs 3분봉 vs 5분봉")
    print(f"  기간: {START} ~ {END}")
    print(f"  설정: SL={SETTINGS['sl_atr_multiplier']}×ATR, TP={SETTINGS['tp_atr_multiplier']}×ATR, "
          f"Lev={SETTINGS['leverage']}x")
    print(f"{'='*80}\n")

    # 1. 데이터 수집/로드
    print("[ 1단계 ] 데이터 준비")
    data = {}
    for iv in INTERVALS:
        df = ensure_data(symbol, iv)
        if df is None:
            print(f"  [{iv}] 데이터 없음 — 스킵")
            continue
        data[iv] = df
    print()

    if not data:
        print("비교할 데이터가 없습니다.")
        sys.exit(1)

    # 2. 백테스트 실행
    print("[ 2단계 ] 백테스트 실행")
    results = []
    for iv in INTERVALS:
        if iv not in data:
            continue
        result = run_backtest_for_interval(data[iv], iv)
        results.append(result)
        print(f"  [{iv}] 완료 ({result['elapsed']}s)")
    print()

    # 3. 결과 비교
    print("[ 3단계 ] 결과 비교\n")
    print(f"{'='*100}")
    print(f"  BTC 타임프레임 비교 ({START} ~ {END})")
    print(f"{'='*100}\n")

    # 핵심 지표 테이블
    table = []
    for r in results:
        table.append({
            "타임프레임": r["interval"],
            "캔들 수": f"{r['candles']:,}",
            "거래 수": r["trades"],
            "승률%": r["win_rate"],
            "PF": r["pf"],
            "수익률%": r["total_return"],
            "MDD%": r["mdd"],
            "Sharpe": r["sharpe"],
            "Calmar": r["calmar"],
            "평균R": r["avg_r"],
        })

    print(tabulate(table, headers="keys", tablefmt="simple", floatfmt=".2f"))

    # 상세 비교
    print(f"\n{'─'*100}")
    print("  상세 비교\n")

    detail_table = []
    for r in results:
        detail_table.append({
            "타임프레임": r["interval"],
            "LONG": f"{r['long_trades']}건 ({r['long_wr']}%)",
            "SHORT": f"{r['short_trades']}건 ({r['short_wr']}%)",
            "최대연승": r["max_consec_w"],
            "최대연패": r["max_consec_l"],
            "실행시간": f"{r['elapsed']}s",
        })

    print(tabulate(detail_table, headers="keys", tablefmt="simple"))
    print(f"{'─'*100}")

    # 요약
    if len(results) >= 2:
        best = max(results, key=lambda x: x["pf"])
        print(f"\n  → PF 기준 최적 타임프레임: {best['interval']} (PF={best['pf']})")

        best_ret = max(results, key=lambda x: x["total_return"])
        print(f"  → 수익률 기준 최적: {best_ret['interval']} ({best_ret['total_return']}%)")

        best_sharpe = max(results, key=lambda x: x["sharpe"])
        print(f"  → Sharpe 기준 최적: {best_sharpe['interval']} (Sharpe={best_sharpe['sharpe']})")


if __name__ == "__main__":
    main()
