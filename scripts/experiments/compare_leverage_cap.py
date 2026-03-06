"""레버리지 CAP 유무 백테스트 비교.

케이스:
  1. 레버리지 20x CAP 적용  (구버전)
  2. 레버리지 CAP 없음       (순수 5% 리스크, 신버전)
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
import strategy.position as _pos_module

start = SETTINGS["backtest_start"]
end   = SETTINGS["backtest_end"]

print(f"\n데이터 로드: {start} ~ {end}")
df = load_from_parquet(start=start, end=end)
if df is None or df.empty:
    print("데이터 없음 — 먼저 python main_backtest.py --fetch-data 실행")
    sys.exit(1)
print(f"캔들 수: {len(df):,}\n")

# ── 원본 함수 (CAP 없음, 현재 코드) ──────────────────────────────────────
_no_cap = _pos_module.calculate_position_size


def _with_cap(capital: float, entry_price: float, atr: float):
    """레버리지 20x 상한 적용 버전 (구버전)."""
    sl_distance = SETTINGS["sl_atr_multiplier"] * atr
    risk_amount  = capital * SETTINGS["risk_per_trade"]
    position_size = risk_amount / sl_distance

    leverage = SETTINGS.get("leverage", 20)
    max_size = (capital * leverage) / entry_price
    if position_size > max_size:
        position_size = max_size

    return position_size, sl_distance


CASES = [
    {"name": "20x CAP 적용 (구버전)",          "fn": _with_cap},
    {"name": "CAP 없음 — 순수 5% 리스크 (신버전)", "fn": _no_cap},
]

rows = []
for case in CASES:
    # create_position이 내부에서 module-level로 calculate_position_size를 참조하므로
    # 모듈 전역을 교체하면 몽키패칭이 동작한다.
    _pos_module.calculate_position_size = case["fn"]

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

# 원본 함수 복원
_pos_module.calculate_position_size = _no_cap

headers = [
    "케이스", "거래수", "승률", "PF", "평균R",
    "수익률", "최종잔고", "MDD", "샤프", "칼마",
    "롱(승률)", "숏(승률)",
]

print("=" * 120)
print(f"  레버리지 CAP 비교 백테스트  ({start} ~ {end})")
print("=" * 120)
print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
print()

# 수익률 차이
r0 = float(rows[0][5].replace("%", "").replace("+", ""))
r1 = float(rows[1][5].replace("%", "").replace("+", ""))
diff = r1 - r0
print(f"  CAP 제거 효과: {'+' if diff >= 0 else ''}{diff:.1f}%p (수익률 기준)")
print()
