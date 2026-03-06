"""레버리지 수준별 백테스트 비교.

레버리지: 5, 10, 15, 20, 25, 30, 35
(weekend_filter=True, bad_hours 포함 현재 설정 기준)
"""

import logging
import sys

from tabulate import tabulate

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from data.storage import load_from_parquet
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics

LEVERAGE_LIST = [5, 10, 15, 20, 25, 30, 35]

start = SETTINGS["backtest_start"]
end   = SETTINGS["backtest_end"]

print(f"\n데이터 로드: {start} ~ {end}")
print(f"weekend_filter={SETTINGS['weekend_filter']}, bad_hours={SETTINGS['bad_hours_utc']}")
df = load_from_parquet(start=start, end=end)
if df is None or df.empty:
    print("데이터 없음 — 먼저 python main_backtest.py --fetch-data 실행")
    sys.exit(1)
print(f"캔들 수: {len(df):,}\n")

rows = []
for lev in LEVERAGE_LIST:
    SETTINGS["leverage"] = lev

    portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])
    m = calculate_metrics(portfolio)

    final_bal = SETTINGS["initial_capital"] * (1 + m.total_return_pct / 100)

    ret_str = f"+{m.total_return_pct:.1f}%" if m.total_return_pct >= 0 else f"{m.total_return_pct:.1f}%"
    rows.append([
        f"{lev}x",
        m.total_trades,
        f"{m.win_rate:.1f}%",
        f"{m.profit_factor:.2f}",
        f"{m.avg_r_multiple:.2f}R",
        ret_str,
        f"${final_bal:,.0f}",
        f"{m.max_drawdown_pct:.1f}%",
        f"{m.sharpe_ratio:.2f}",
        f"{m.calmar_ratio:.2f}",
        f"{m.long_trades}({m.long_win_rate:.0f}%)",
        f"{m.short_trades}({m.short_win_rate:.0f}%)",
    ])

# 원본 레버리지 복원
SETTINGS["leverage"] = 20

headers = [
    "레버리지", "거래수", "승률", "PF", "평균R",
    "수익률", "최종잔고", "MDD", "샤프", "칼마",
    "롱(승률)", "숏(승률)",
]

print("=" * 115)
print(f"  레버리지별 백테스트  ({start} ~ {end})")
print("=" * 115)
print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
print()

best_ret    = max(rows, key=lambda r: float(r[5].replace("%","").replace("+","")))
best_pf     = max(rows, key=lambda r: float(r[3]))
best_sharpe = max(rows, key=lambda r: float(r[8]))
best_calmar = max(rows, key=lambda r: float(r[9]))
best_mdd    = min(rows, key=lambda r: float(r[7].replace("%","")))

print(f"  [최고 수익률] {best_ret[0]:>4} → {best_ret[5]}")
print(f"  [최고 PF    ] {best_pf[0]:>4} → {best_pf[3]}")
print(f"  [최저 MDD   ] {best_mdd[0]:>4} → {best_mdd[7]}")
print(f"  [최고 샤프  ] {best_sharpe[0]:>4} → {best_sharpe[8]}")
print(f"  [최고 칼마  ] {best_calmar[0]:>4} → {best_calmar[9]}")
print()
