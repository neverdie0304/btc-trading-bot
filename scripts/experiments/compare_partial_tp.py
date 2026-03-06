"""분할 익절(Partial Take-Profit) 설정별 백테스트 비교.

케이스:
  없음            단일 TP @ 3R (현재 기본)
  1/2@2R          50%를 2R에서 익절, 나머지 50%는 3R TP
  1/2@1.5R        50%를 1.5R에서 익절, 나머지 50%는 3R TP
  2/3@2R          67%를 2R에서 익절, 나머지 33%는 3R TP
  1/3@2R+2/3@3R   33%를 2R에서, 67%를 3R에서
  1/4@2+1/4@2.5+1/2@3  (user 제안)
  1/3@2+1/3@2.5+1/3@3  균등 3분할
  1/4@1.5+1/4@2+1/2@3  초기 분할 앞당기기
  1/3@1.5+1/3@2+1/3@3  1.5R부터 균등 3분할
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

CASES = [
    {
        "name": "없음 (3R 전량, 현재)",
        "tp_levels": None,
    },
    {
        "name": "1/2@2R + 1/2@3R",
        "tp_levels": [(2.0, 0.5), (3.0, 0.5)],
    },
    {
        "name": "1/2@1.5R + 1/2@3R",
        "tp_levels": [(1.5, 0.5), (3.0, 0.5)],
    },
    {
        "name": "2/3@2R + 1/3@3R",
        "tp_levels": [(2.0, 2/3), (3.0, 1/3)],
    },
    {
        "name": "1/3@2R + 2/3@3R",
        "tp_levels": [(2.0, 1/3), (3.0, 2/3)],
    },
    {
        "name": "1/4@2R+1/4@2.5R+1/2@3R (user)",
        "tp_levels": [(2.0, 0.25), (2.5, 0.25), (3.0, 0.50)],
    },
    {
        "name": "1/3@2R+1/3@2.5R+1/3@3R",
        "tp_levels": [(2.0, 1/3), (2.5, 1/3), (3.0, 1/3)],
    },
    {
        "name": "1/4@1.5R+1/4@2R+1/2@3R",
        "tp_levels": [(1.5, 0.25), (2.0, 0.25), (3.0, 0.50)],
    },
    {
        "name": "1/3@1.5R+1/3@2R+1/3@3R",
        "tp_levels": [(1.5, 1/3), (2.0, 1/3), (3.0, 1/3)],
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

# 지표 & 시그널 한 번만 계산 (9번 반복 방지)
print("지표 & 시그널 계산 중...")
df = compute_indicators(df)
signals = generate_signals(df)
print(f"시그널: LONG={sum(signals == 'LONG')}건, SHORT={sum(signals == 'SHORT')}건\n")


def _lifecycle_stats(portfolio):
    """PARTIAL_TP를 묶어서 진입 단위의 승률, PF, 평균R을 계산한다."""
    groups: dict = defaultdict(list)
    for t in portfolio.trades:
        groups[t.entry_time].append(t)

    wins, losses = [], []
    r_list = []

    for group in groups.values():
        total_pnl = sum(t.pnl for t in group)
        total_size = sum(t.size for t in group)
        if total_pnl > 0:
            wins.append(total_pnl)
        else:
            losses.append(total_pnl)

        # 가중 평균 R
        if total_size > 0:
            r_list.append(sum(t.r_multiple * t.size for t in group) / total_size)

    n_entries = len(groups)
    win_rate = len(wins) / n_entries * 100 if n_entries else 0.0
    total_profit = sum(wins)
    total_loss = sum(abs(l) for l in losses)
    pf = total_profit / total_loss if total_loss > 0 else float("inf")
    avg_r = sum(r_list) / len(r_list) if r_list else 0.0

    return n_entries, win_rate, pf, avg_r


rows = []
for case in CASES:
    portfolio = run_backtest(
        df,
        capital=SETTINGS["initial_capital"],
        pre_signals=signals,
        partial_tp_config=case["tp_levels"],
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

print("=" * 125)
print(f"  분할 익절 설정별 비교  ({start} ~ {end})")
print("=" * 125)
print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
print()

best_ret    = max(rows, key=lambda r: float(r[5].replace("%", "").replace("+", "")))
best_pf     = max(rows, key=lambda r: float(r[3]))
best_sharpe = max(rows, key=lambda r: float(r[8]))
best_calmar = max(rows, key=lambda r: float(r[9]))
best_mdd    = min(rows, key=lambda r: float(r[7].replace("%", "")))

print(f"  [최고 수익률] {best_ret[0]}    → {best_ret[5]}")
print(f"  [최고 PF    ] {best_pf[0]}    → {best_pf[3]}")
print(f"  [최저 MDD   ] {best_mdd[0]}   → {best_mdd[7]}")
print(f"  [최고 샤프  ] {best_sharpe[0]} → {best_sharpe[8]}")
print(f"  [최고 칼마  ] {best_calmar[0]} → {best_calmar[9]}")
print()
