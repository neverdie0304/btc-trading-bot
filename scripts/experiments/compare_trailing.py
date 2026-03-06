"""트레일링 스탑 설정별 백테스트 비교.

케이스:
  없음          BE=∞,  STEP=∞   (순수 고정 SL/TP)
  BE@0.5R      BE=0.5, STEP=∞
  BE@1.0R      BE=1.0, STEP=∞   (현재 BE 임계값, STEP 없음)
  BE@1.5R      BE=1.5, STEP=∞
  BE@1.0+ST@1.5  현재 설정
  BE@0.5+ST@1.0
  BE@0.5+ST@1.5
  BE@1.0+ST@2.0
  BE@1.5+ST@2.0
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

_INF = 999.0   # 사실상 미발동

CASES = [
    {"name": "없음 (고정SL/TP)",       "be": _INF, "step": _INF},
    {"name": "BE@0.5R",               "be": 0.5,  "step": _INF},
    {"name": "BE@1.0R",               "be": 1.0,  "step": _INF},
    {"name": "BE@1.5R",               "be": 1.5,  "step": _INF},
    {"name": "BE@0.5+ST@1.0",         "be": 0.5,  "step": 1.0},
    {"name": "BE@0.5+ST@1.5",         "be": 0.5,  "step": 1.5},
    {"name": "BE@1.0+ST@1.5 (현재)",  "be": 1.0,  "step": 1.5},
    {"name": "BE@1.0+ST@2.0",         "be": 1.0,  "step": 2.0},
    {"name": "BE@1.5+ST@2.0",         "be": 1.5,  "step": 2.0},
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
    SETTINGS["trailing_be_threshold"]   = case["be"]
    SETTINGS["trailing_step_threshold"] = case["step"]

    portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])
    m = calculate_metrics(portfolio)

    final_bal = SETTINGS["initial_capital"] * (1 + m.total_return_pct / 100)
    ret_str = f"+{m.total_return_pct:.1f}%" if m.total_return_pct >= 0 else f"{m.total_return_pct:.1f}%"

    rows.append([
        case["name"],
        m.total_trades,
        f"{m.win_rate:.1f}%",
        f"{m.profit_factor:.2f}",
        f"{m.avg_r_multiple:.2f}R",
        ret_str,
        f"${final_bal:,.0f}",
        f"{m.max_drawdown_pct:.1f}%",
        f"{m.sharpe_ratio:.2f}",
        f"{m.calmar_ratio:.2f}",
    ])

# 원복
SETTINGS["trailing_be_threshold"]   = 1.0
SETTINGS["trailing_step_threshold"] = 1.5

headers = [
    "케이스", "거래수", "승률", "PF", "평균R",
    "수익률", "최종잔고", "MDD", "샤프", "칼마",
]

print("=" * 110)
print(f"  트레일링 스탑 설정별 비교  ({start} ~ {end})")
print("=" * 110)
print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
print()

best_ret    = max(rows, key=lambda r: float(r[5].replace("%","").replace("+","")))
best_pf     = max(rows, key=lambda r: float(r[3]))
best_sharpe = max(rows, key=lambda r: float(r[8]))
best_calmar = max(rows, key=lambda r: float(r[9]))
best_mdd    = min(rows, key=lambda r: float(r[7].replace("%","")))

print(f"  [최고 수익률] {best_ret[0]}    → {best_ret[5]}")
print(f"  [최고 PF    ] {best_pf[0]}    → {best_pf[3]}")
print(f"  [최저 MDD   ] {best_mdd[0]}   → {best_mdd[7]}")
print(f"  [최고 샤프  ] {best_sharpe[0]} → {best_sharpe[8]}")
print(f"  [최고 칼마  ] {best_calmar[0]} → {best_calmar[9]}")
print()
