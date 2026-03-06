"""volume_threshold × atr_signal_multiplier 그리드 서치 최적화 스크립트."""

import os
os.environ["TQDM_DISABLE"] = "1"

import logging
import sys
import time
from itertools import product

import pandas as pd
import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_grid_search() -> pd.DataFrame:
    """그리드 서치를 실행하고 결과를 DataFrame으로 반환한다."""

    # 레버리지 20배 설정
    SETTINGS["leverage"] = 20

    # 그리드 파라미터
    vol_thresholds = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
    atr_multipliers = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]

    # 데이터 로드
    data_path = SETTINGS["cache_dir"] + "/BTCUSDT_5m.parquet"
    print(f"데이터 로드 중: {data_path}")
    df = pd.read_parquet(data_path)

    # open_time을 인덱스로 설정
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df.set_index("open_time", inplace=True)

    # 숫자형 변환
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"데이터: {len(df)}봉, {df.index[0]} ~ {df.index[-1]}")
    print(f"레버리지: {SETTINGS['leverage']}x")
    print(f"수수료: maker={SETTINGS['maker_fee']}, taker={SETTINGS['taker_fee']}")
    print(f"슬리피지: {SETTINGS['slippage_rate']}")
    print()

    total_combos = len(vol_thresholds) * len(atr_multipliers)
    print(f"총 {total_combos}개 조합 테스트 시작")
    print("=" * 100)
    print(f"{'vol_thresh':>10} {'atr_mult':>10} {'trades':>8} {'win%':>8} "
          f"{'PF':>8} {'return%':>10} {'MDD%':>8} {'sharpe':>8} {'final$':>12}")
    print("-" * 100)

    results = []
    start_time = time.time()

    for idx, (vt, am) in enumerate(product(vol_thresholds, atr_multipliers)):
        # 설정 변경
        SETTINGS["volume_threshold"] = vt
        SETTINGS["atr_signal_multiplier"] = am

        try:
            portfolio = run_backtest(df.copy(), capital=10000)
            m = calculate_metrics(portfolio)

            row = {
                "volume_threshold": vt,
                "atr_signal_multiplier": am,
                "trades": m.total_trades,
                "win_rate": round(m.win_rate, 1),
                "profit_factor": round(m.profit_factor, 2),
                "total_return_pct": round(m.total_return_pct, 1),
                "max_drawdown_pct": round(m.max_drawdown_pct, 1),
                "sharpe": round(m.sharpe_ratio, 2),
                "cagr": round(m.cagr, 1),
                "avg_r": round(m.avg_r_multiple, 3),
                "long_trades": m.long_trades,
                "short_trades": m.short_trades,
                "long_wr": round(m.long_win_rate, 1),
                "short_wr": round(m.short_win_rate, 1),
                "final_capital": round(portfolio.capital, 0),
                "max_consec_loss": m.max_consecutive_losses,
            }
            results.append(row)

            # 실시간 출력
            print(f"{vt:>10.1f} {am:>10.1f} {m.total_trades:>8} {m.win_rate:>7.1f}% "
                  f"{m.profit_factor:>8.2f} {m.total_return_pct:>9.1f}% "
                  f"{m.max_drawdown_pct:>7.1f}% {m.sharpe_ratio:>8.2f} "
                  f"${portfolio.capital:>11,.0f}", flush=True)

        except Exception as e:
            print(f"{vt:>10.1f} {am:>10.1f}  ERROR: {e}")
            results.append({
                "volume_threshold": vt,
                "atr_signal_multiplier": am,
                "trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe": 0,
                "cagr": 0,
                "avg_r": 0,
                "long_trades": 0,
                "short_trades": 0,
                "long_wr": 0,
                "short_wr": 0,
                "final_capital": 10000,
                "max_consec_loss": 0,
            })

    elapsed = time.time() - start_time
    print("=" * 100)
    print(f"완료: {elapsed:.1f}초 소요")

    result_df = pd.DataFrame(results)

    # Top 10 by total return
    print("\n\n" + "=" * 80)
    print("  TOP 10 - 총 수익률 기준")
    print("=" * 80)
    top_return = result_df.nlargest(10, "total_return_pct")
    for _, r in top_return.iterrows():
        print(f"  vol={r['volume_threshold']:.1f}  atr={r['atr_signal_multiplier']:.1f}  "
              f"trades={r['trades']}  WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  final=${r['final_capital']:,.0f}")

    # Top 10 by profit factor (min 100 trades)
    print("\n" + "=" * 80)
    print("  TOP 10 - Profit Factor 기준 (최소 100거래)")
    print("=" * 80)
    filtered = result_df[result_df["trades"] >= 100]
    top_pf = filtered.nlargest(10, "profit_factor")
    for _, r in top_pf.iterrows():
        print(f"  vol={r['volume_threshold']:.1f}  atr={r['atr_signal_multiplier']:.1f}  "
              f"trades={r['trades']}  WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  final=${r['final_capital']:,.0f}")

    # Top 10 by Sharpe (min 100 trades)
    print("\n" + "=" * 80)
    print("  TOP 10 - Sharpe Ratio 기준 (최소 100거래)")
    print("=" * 80)
    top_sharpe = filtered.nlargest(10, "sharpe")
    for _, r in top_sharpe.iterrows():
        print(f"  vol={r['volume_threshold']:.1f}  atr={r['atr_signal_multiplier']:.1f}  "
              f"trades={r['trades']}  WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  final=${r['final_capital']:,.0f}")

    # CSV 저장
    csv_path = "optimization_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    return result_df


if __name__ == "__main__":
    run_grid_search()
