"""risk_per_trade 최적화 스크립트."""

import os
os.environ["TQDM_DISABLE"] = "1"

import logging
import time

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
    """risk_per_trade 그리드 서치를 실행한다."""

    # 고정 파라미터
    SETTINGS["leverage"] = 20
    SETTINGS["volume_threshold"] = 1.0
    SETTINGS["atr_signal_multiplier"] = 1.0
    SETTINGS["daily_max_losses"] = 0
    SETTINGS["daily_max_trades"] = 0

    # 그리드 파라미터
    risk_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]

    # 데이터 로드
    data_path = SETTINGS["cache_dir"] + "/BTCUSDT_5m.parquet"
    print(f"데이터 로드 중: {data_path}")
    df = pd.read_parquet(data_path)

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df.set_index("open_time", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    start = SETTINGS.get("backtest_start")
    end = SETTINGS.get("backtest_end")
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    print(f"데이터: {len(df)}봉, {df.index[0]} ~ {df.index[-1]}")
    print(f"레버리지: {SETTINGS['leverage']}x")
    print()

    print(f"총 {len(risk_values)}개 값 테스트 시작")
    print("=" * 110)
    print(f"{'risk%':>8} {'trades':>8} {'win%':>8} {'PF':>8} "
          f"{'return%':>14} {'MDD%':>8} {'sharpe':>8} {'calmar':>8} {'avg_R':>8} {'final$':>16}")
    print("-" * 110)

    results = []
    start_time = time.time()

    for rpt in risk_values:
        SETTINGS["risk_per_trade"] = rpt

        try:
            portfolio = run_backtest(df.copy(), capital=10000)
            m = calculate_metrics(portfolio)

            row = {
                "risk_per_trade": rpt,
                "risk_pct": rpt * 100,
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

            print(f"{rpt*100:>7.1f}% {m.total_trades:>8} {m.win_rate:>7.1f}% "
                  f"{m.profit_factor:>8.2f} {m.total_return_pct:>13.1f}% "
                  f"{m.max_drawdown_pct:>7.1f}% {m.sharpe_ratio:>8.2f} "
                  f"{m.calmar_ratio:>8.2f} {m.avg_r_multiple:>8.3f} "
                  f"${portfolio.capital:>15,.0f}", flush=True)

        except Exception as e:
            print(f"{rpt*100:>7.1f}%  ERROR: {e}")
            results.append({
                "risk_per_trade": rpt, "risk_pct": rpt * 100,
                "trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_return_pct": 0, "max_drawdown_pct": 0,
                "sharpe": 0, "calmar": 0, "avg_r": 0,
                "max_consec_loss": 0, "final_capital": 10000,
            })

    elapsed = time.time() - start_time
    print("=" * 110)
    print(f"완료: {elapsed:.1f}초 소요")

    result_df = pd.DataFrame(results)

    # 요약
    print("\n" + "=" * 90)
    print("  결과 요약 (Sharpe 기준 정렬)")
    print("=" * 90)
    for _, r in result_df.sort_values("sharpe", ascending=False).iterrows():
        print(f"  risk={r['risk_pct']:.1f}%  trades={int(r['trades'])}  "
              f"WR={r['win_rate']}%  PF={r['profit_factor']:.2f}  "
              f"return={r['total_return_pct']:.1f}%  MDD={r['max_drawdown_pct']:.1f}%  "
              f"sharpe={r['sharpe']:.2f}  calmar={r['calmar']:.2f}")

    csv_path = "risk_optimization.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    return result_df


if __name__ == "__main__":
    run_grid_search()
