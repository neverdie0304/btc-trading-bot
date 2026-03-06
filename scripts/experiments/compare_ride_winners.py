"""분할익절 후 나머지 물량을 더 멀리 가져가는 전략 비교.

개념:
  현재 TP(tp_atr_multiplier * ATR = 2.5R)에서 일부를 익절하고,
  나머지는 SL을 +2R로 올린 뒤 더 높은 TP까지 가져간다.

NOTE: R 단위 기준 (1R = sl_atr_multiplier × ATR = 1.2 × ATR)
  - 현재 TP = tp_atr_multiplier / sl_atr_multiplier = 3.0 / 1.2 = 2.5R
  - SL→2R = 진입가 + 2.0 × r_unit (이익 보호)
  - 4R / 5R / 6R = 최종 잔여 물량 TP

케이스:
  없음 (현재)        전량 2.5R TP
  3/4@2.5R→4R       3/4 익절 후 1/4는 4R TP, SL→2R
  3/4@2.5R→5R       3/4 익절 후 1/4는 5R TP, SL→2R
  3/4@2.5R→6R       3/4 익절 후 1/4는 6R TP, SL→2R
  3/4@2.5R→4R(SL→2.5R) SL을 현재 TP 수준으로 올려 손익비 보장
  1/2@2.5R→4R       1/2 익절 후 1/2는 4R TP, SL→2R
  1/2@2.5R→5R       1/2 익절 후 1/2는 5R TP, SL→2R
  1/2@2.5R→6R       1/2 익절 후 1/2는 6R TP, SL→2R
  1/4@2.5R→5R       1/4 익절 후 3/4는 5R TP, SL→2R
"""

import logging
import sys
from collections import defaultdict

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
from strategy.signals import compute_indicators, generate_signals

# 현재 TP를 R 단위로 환산 (tp_atr_multiplier / sl_atr_multiplier)
_STD_TP_R = SETTINGS["tp_atr_multiplier"] / SETTINGS["sl_atr_multiplier"]  # = 2.5

CASES = [
    # 베이스라인: 분할 없음 (전량 2.5R TP)
    {
        "name": f"없음 ({_STD_TP_R:.1f}R 전량, 현재)",
        "tp_levels": None,
        "final_tp_r": None,
    },
    # 3/4 @ 2.5R → 나머지 1/4 멀리 가져가기
    {
        "name": f"3/4@{_STD_TP_R:.1f}R + 1/4→4R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.75, 2.0)],
        "final_tp_r": 4.0,
    },
    {
        "name": f"3/4@{_STD_TP_R:.1f}R + 1/4→5R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.75, 2.0)],
        "final_tp_r": 5.0,
    },
    {
        "name": f"3/4@{_STD_TP_R:.1f}R + 1/4→6R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.75, 2.0)],
        "final_tp_r": 6.0,
    },
    {
        "name": f"3/4@{_STD_TP_R:.1f}R + 1/4→4R  (SL→{_STD_TP_R:.1f}R)",
        "tp_levels": [(_STD_TP_R, 0.75, _STD_TP_R)],
        "final_tp_r": 4.0,
    },
    # 1/2 @ 2.5R → 나머지 1/2 멀리 가져가기
    {
        "name": f"1/2@{_STD_TP_R:.1f}R + 1/2→4R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.5, 2.0)],
        "final_tp_r": 4.0,
    },
    {
        "name": f"1/2@{_STD_TP_R:.1f}R + 1/2→5R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.5, 2.0)],
        "final_tp_r": 5.0,
    },
    {
        "name": f"1/2@{_STD_TP_R:.1f}R + 1/2→6R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.5, 2.0)],
        "final_tp_r": 6.0,
    },
    # 1/4만 익절 → 나머지 3/4 더 멀리
    {
        "name": f"1/4@{_STD_TP_R:.1f}R + 3/4→5R  (SL→2R)",
        "tp_levels": [(_STD_TP_R, 0.25, 2.0)],
        "final_tp_r": 5.0,
    },
]

start = SETTINGS["backtest_start"]
end   = SETTINGS["backtest_end"]

print(f"\n데이터 로드: {start} ~ {end}")
df = load_from_parquet(start=start, end=end)
if df is None or df.empty:
    print("데이터 없음 — 먼저 python main_backtest.py --fetch-data 실행")
    sys.exit(1)
print(f"캔들 수: {len(df):,}")

print("지표 & 시그널 계산 중...")
df = compute_indicators(df)
signals = generate_signals(df)
print(f"현재 TP = {_STD_TP_R:.2f}R (tp_atr={SETTINGS['tp_atr_multiplier']}, sl_atr={SETTINGS['sl_atr_multiplier']})\n")


def _lifecycle_stats(portfolio):
    """PARTIAL_TP를 묶어 진입 단위 승률, PF, 평균R을 계산한다."""
    groups: dict = defaultdict(list)
    for t in portfolio.trades:
        groups[t.entry_time].append(t)

    wins, losses, r_list = [], [], []
    for group in groups.values():
        total_pnl = sum(t.pnl for t in group)
        total_size = sum(t.size for t in group)
        if total_pnl > 0:
            wins.append(total_pnl)
        else:
            losses.append(total_pnl)
        if total_size > 0:
            r_list.append(sum(t.r_multiple * t.size for t in group) / total_size)

    n = len(groups)
    win_rate = len(wins) / n * 100 if n else 0.0
    pf = sum(wins) / sum(abs(l) for l in losses) if losses else float("inf")
    avg_r = sum(r_list) / len(r_list) if r_list else 0.0
    return n, win_rate, pf, avg_r


rows = []
for case in CASES:
    portfolio = run_backtest(
        df,
        capital=SETTINGS["initial_capital"],
        pre_signals=signals,
        partial_tp_config=case["tp_levels"],
        final_tp_r=case["final_tp_r"],
    )
    m = calculate_metrics(portfolio)
    n_entries, win_rate, pf, avg_r = _lifecycle_stats(portfolio)

    final_bal = SETTINGS["initial_capital"] * (1 + m.total_return_pct / 100)
    ret_str = f"+{m.total_return_pct:.1f}%" if m.total_return_pct >= 0 else f"{m.total_return_pct:.1f}%"

    rows.append([
        case["name"],
        n_entries,
        f"{win_rate:.1f}%",
        f"{pf:.2f}",
        f"{avg_r:.2f}R",
        ret_str,
        f"${final_bal:,.0f}",
        f"{m.max_drawdown_pct:.1f}%",
        f"{m.sharpe_ratio:.2f}",
        f"{m.calmar_ratio:.2f}",
    ])

headers = [
    "케이스", "진입수", "승률", "PF", "평균R",
    "수익률", "최종잔고", "MDD", "샤프", "칼마",
]

print("=" * 130)
print(f"  잔여 물량 운영 전략 비교  ({start} ~ {end})")
print("=" * 130)
print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
print()

best_ret    = max(rows, key=lambda r: float(r[5].replace("%", "").replace("+", "")))
best_pf     = max(rows, key=lambda r: float(r[3]))
best_sharpe = max(rows, key=lambda r: float(r[8]))
best_calmar = max(rows, key=lambda r: float(r[9]))
best_mdd    = min(rows, key=lambda r: float(r[7].replace("%", "")))

print(f"  [최고 수익률] {best_ret[0]}   → {best_ret[5]}")
print(f"  [최고 PF    ] {best_pf[0]}   → {best_pf[3]}")
print(f"  [최저 MDD   ] {best_mdd[0]}  → {best_mdd[7]}")
print(f"  [최고 샤프  ] {best_sharpe[0]} → {best_sharpe[8]}")
print(f"  [최고 칼마  ] {best_calmar[0]} → {best_calmar[9]}")
print()
