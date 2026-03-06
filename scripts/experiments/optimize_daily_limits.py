"""daily_max_losses × daily_max_trades 그리드 서치 최적화 스크립트."""

import os
os.environ["TQDM_DISABLE"] = "1"

import logging
import time
from itertools import product

import pandas as pd

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def run_grid_search() -> pd.DataFrame:
    """그리드 서치를 실행하고 결과를 DataFrame으로 반환한다."""

    # 고정 파라미터
    SETTINGS["leverage"] = 20
    SETTINGS["volume_threshold"] = 1.0
    SETTINGS["atr_signal_multiplier"] = 1.0

    # 그리드 파라미터
    max_losses_list = [0, 1, 2, 3, 4, 5]       # 0=무제한
    max_trades_list = [0, 1, 2, 3, 4, 5, 6, 8, 10]  # 0=무제한

    # 데이터 로드
    data_path = SETTINGS["cache_dir"] + "/BTCUSDT_5m.parquet"
    print(f"데이터 로드 중: {data_path}")
    df = pd.read_parquet(data_path)

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df.set_index("open_time", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 백테스트 기간 필터
    start = SETTINGS.get("backtest_start")
    end = SETTINGS.get("backtest_end")
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    print(f"데이터: {len(df)}봉, {df.index[0]} ~ {df.index[-1]}")
    print(f"레버리지: {SETTINGS['leverage']}x, vol={SETTINGS['volume_threshold']}, atr={SETTINGS['atr_signal_multiplier']}")
    print()

    total_combos = len(max_losses_list) * len(max_trades_list)
    print(f"총 {total_combos}개 조합 테스트 시작")
    print("=" * 110)
    print(f"{'max_loss':>10} {'max_trade':>10} {'trades':>8} {'win%':>8} "
          f"{'PF':>8} {'return%':>12} {'MDD%':>8} {'sharpe':>8} {'calmar':>8} {'final$':>14}")
    print("-" * 110)

    results = []
    start_time = time.time()

    for ml, mt in product(max_losses_list, max_trades_list):
        SETTINGS["daily_max_losses"] = ml
        SETTINGS["daily_max_trades"] = mt

        try:
            portfolio = run_backtest(df.copy(), capital=10000)
            m = calculate_metrics(portfolio)

            row = {
                "daily_max_losses": ml,
                "daily_max_trades": mt,
                "trades": m.total_trades,
                "win_rate": round(m.win_rate, 1),
                "profit_factor": round(m.profit_factor, 2),
                "total_return_pct": round(m.total_return_pct, 1),
                "max_drawdown_pct": round(m.max_drawdown_pct, 1),
                "sharpe": round(m.sharpe_ratio, 2),
                "calmar": round(m.calmar_ratio, 2),
                "avg_r": round(m.avg_r_multiple, 3),
                "max_consec_loss": m.max_consecutive_losses,
                "final_capital": round(portfolio.capital, 0),
            }
            results.append(row)

            ml_label = "무제한" if ml == 0 else str(ml)
            mt_label = "무제한" if mt == 0 else str(mt)
            print(f"{ml_label:>10} {mt_label:>10} {m.total_trades:>8} {m.win_rate:>7.1f}% "
                  f"{m.profit_factor:>8.2f} {m.total_return_pct:>11.1f}% "
                  f"{m.max_drawdown_pct:>7.1f}% {m.sharpe_ratio:>8.2f} "
                  f"{m.calmar_ratio:>8.2f} ${portfolio.capital:>13,.0f}", flush=True)

        except Exception as e:
            print(f"{ml:>10} {mt:>10}  ERROR: {e}")
            results.append({
                "daily_max_losses": ml,
                "daily_max_trades": mt,
                "trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_return_pct": 0, "max_drawdown_pct": 0,
                "sharpe": 0, "calmar": 0, "avg_r": 0,
                "max_consec_loss": 0, "final_capital": 10000,
            })

    elapsed = time.time() - start_time
    print("=" * 110)
    print(f"완료: {elapsed:.1f}초 소요")

    result_df = pd.DataFrame(results)

    # Top 10 수익률
    print("\n" + "=" * 90)
    print("  TOP 10 - 총 수익률 기준")
    print("=" * 90)
    for _, r in result_df.nlargest(10, "total_return_pct").iterrows():
        ml_label = "무제한" if r['daily_max_losses'] == 0 else int(r['daily_max_losses'])
        mt_label = "무제한" if r['daily_max_trades'] == 0 else int(r['daily_max_trades'])
        print(f"  loss={ml_label}  trade={mt_label}  "
              f"trades={int(r['trades'])}  WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  calmar={r['calmar']:.2f}")

    # Top 10 Sharpe (min 100 trades)
    print("\n" + "=" * 90)
    print("  TOP 10 - Sharpe Ratio 기준 (최소 100거래)")
    print("=" * 90)
    filtered = result_df[result_df["trades"] >= 100]
    for _, r in filtered.nlargest(10, "sharpe").iterrows():
        ml_label = "무제한" if r['daily_max_losses'] == 0 else int(r['daily_max_losses'])
        mt_label = "무제한" if r['daily_max_trades'] == 0 else int(r['daily_max_trades'])
        print(f"  loss={ml_label}  trade={mt_label}  "
              f"trades={int(r['trades'])}  WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  calmar={r['calmar']:.2f}")

    # Top 10 낮은 MDD (min 100 trades, PF > 1.0)
    print("\n" + "=" * 90)
    print("  TOP 10 - 낮은 MDD 기준 (최소 100거래, PF>1.0)")
    print("=" * 90)
    filtered_pf = filtered[filtered["profit_factor"] > 1.0]
    for _, r in filtered_pf.nsmallest(10, "max_drawdown_pct").iterrows():
        ml_label = "무제한" if r['daily_max_losses'] == 0 else int(r['daily_max_losses'])
        mt_label = "무제한" if r['daily_max_trades'] == 0 else int(r['daily_max_trades'])
        print(f"  loss={ml_label}  trade={mt_label}  "
              f"trades={int(r['trades'])}  WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  calmar={r['calmar']:.2f}")

    csv_path = "daily_limits_optimization.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    return result_df


if __name__ == "__main__":
    run_grid_search()
