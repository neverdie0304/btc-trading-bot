"""필터 조합 4가지 비교 백테스트 스크립트.

케이스:
  1. weekend=OFF, bad_hours=OFF
  2. weekend=ON,  bad_hours=OFF
  3. weekend=OFF, bad_hours=ON
  4. weekend=ON,  bad_hours=ON
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

BAD_HOURS = [3, 5, 10, 12, 16, 19, 21]

CASES = [
    {"name": "❌ 둘다 OFF",           "weekend": False, "bad_hours": []},
    {"name": "✅ 주말만 ON",           "weekend": True,  "bad_hours": []},
    {"name": "✅ bad_hours만 ON",      "weekend": False, "bad_hours": BAD_HOURS},
    {"name": "✅ 둘다 ON",             "weekend": True,  "bad_hours": BAD_HOURS},
]

start = SETTINGS["backtest_start"]
end   = SETTINGS["backtest_end"]

print(f"\n데이터 로드: {start} ~ {end}")
df = load_from_parquet(start=start, end=end)
if df is None or df.empty:
    print("데이터 없음 — 먼저 python main_backtest.py --fetch-data 실행")
    sys.exit(1)
print(f"캔들 수: {len(df):,}\n")

rows = []
for case in CASES:
    SETTINGS["weekend_filter"] = case["weekend"]
    SETTINGS["bad_hours_utc"]  = case["bad_hours"]

    portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])
    m = calculate_metrics(portfolio)

    final_bal = SETTINGS["initial_capital"] * (1 + m.total_return_pct / 100)

    rows.append([
        case["name"],
        m.total_trades,
        f"{m.win_rate:.1f}%",
        f"{m.profit_factor:.2f}",
        f"{m.avg_r_multiple:.2f}R",
        f"+{m.total_return_pct:.1f}%" if m.total_return_pct >= 0 else f"{m.total_return_pct:.1f}%",
        f"${final_bal:,.0f}",
        f"{m.max_drawdown_pct:.1f}%",
        f"{m.sharpe_ratio:.2f}",
        f"{m.calmar_ratio:.2f}",
        f"{m.long_trades}({m.long_win_rate:.0f}%)",
        f"{m.short_trades}({m.short_win_rate:.0f}%)",
    ])

headers = [
    "케이스", "거래수", "승률", "PF", "평균R",
    "수익률", "최종잔고", "MDD", "샤프", "칼마",
    "롱(승률)", "숏(승률)",
]

print("=" * 110)
print(f"  필터 조합 비교 백테스트  ({start} ~ {end})")
print("=" * 110)
print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
print()

# 항목별 최고값 표시
best_return = max(rows, key=lambda r: float(r[5].replace("%","").replace("+","")))
best_pf     = max(rows, key=lambda r: float(r[3]))
best_mdd    = min(rows, key=lambda r: float(r[7].replace("%","")))
best_sharpe = max(rows, key=lambda r: float(r[8]))

print("  [최고 수익률] ", best_return[0], "→", best_return[5])
print("  [최고 PF    ] ", best_pf[0],     "→", best_pf[3])
print("  [최저 MDD   ] ", best_mdd[0],    "→", best_mdd[7])
print("  [최고 샤프  ] ", best_sharpe[0], "→", best_sharpe[8])
print()
