"""PF 최적화 스크립트.

트레일링 스탑, ADX 필터, SL/TP 비율을 그리드 서치로 최적화한다.
"""

import itertools
import logging
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from tabulate import tabulate

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from data.storage import load_from_parquet
from strategy.signals import compute_indicators, Signal
from indicators.atr import calc_adx
from optimize import generate_relaxed_signals
from backtest.portfolio import Portfolio
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# === 최적화 대상 파라미터 그리드 ===
PARAM_GRID = {
    "trailing_be_threshold": [1.0, 1.5, 2.0, 999.0],  # 999 = OFF
    "adx_min": [0, 20, 25, 30],                         # 0 = OFF
    "sl_atr_multiplier": [0.7, 0.8, 1.0],
    "tp_atr_multiplier": [3.0, 4.0, 5.0],
}
# 총 4 × 4 × 3 × 3 = 144개


def fast_backtest_pf(
    df: pd.DataFrame,
    params: dict,
    precomputed: dict,
    capital: float = 10000,
) -> Portfolio:
    """파라미터별 빠른 백테스트. numpy 배열 직접 접근."""
    from strategy.position import Position

    portfolio = Portfolio(initial_capital=capital, capital=capital)

    p = precomputed
    closes = p["close"]
    highs = p["high"]
    lows = p["low"]
    atrs = p["atr"]
    atr_medians = p["atr_median"]
    rsis = p["rsi"]
    vol_ratios = p["volume_ratio"]
    adx_arr = p["adx"]
    weekdays = p["weekdays"]
    hours = p["hours"]
    sig_arr = p["signals"]

    sl_mult = params["sl_atr_multiplier"]
    tp_mult = params["tp_atr_multiplier"]
    risk_pct = params["risk_per_trade"]
    leverage = params.get("leverage", 1)
    be_thr = params["trailing_be_threshold"]
    # trailing step = be + 0.5 (단, OFF면 무의미)
    step_thr = be_thr + 0.5 if be_thr < 100 else 999.0
    rsi_l_min = params["rsi_long_min"]
    rsi_l_max = params["rsi_long_max"]
    rsi_s_min = params["rsi_short_min"]
    rsi_s_max = params["rsi_short_max"]
    vol_thr = params["volume_threshold"]
    cd_losses = params["cooldown_after_consecutive_losses"]
    cd_candles = params["cooldown_candles"]
    dz_ratio = params["deadzone_atr_ratio"]
    weekend = params.get("weekend_filter", True)
    bad_hours = set(params.get("bad_hours_utc", []))
    adx_min = params.get("adx_min", 0)

    n = len(closes)
    equity = []
    pos = None
    consec_losses = 0
    last_loss_idx = -999

    for i in range(1, n):
        c = closes[i]
        h = highs[i]
        lo = lows[i]

        # 1. SL/TP 체크
        if pos is not None:
            if pos.direction == Signal.LONG:
                sl_hit = lo <= pos.sl_price
                tp_hit = h >= pos.tp_price
            else:
                sl_hit = h >= pos.sl_price
                tp_hit = lo <= pos.tp_price

            if sl_hit or tp_hit:
                exit_p = pos.sl_price if sl_hit else pos.tp_price
                reason = "SL" if sl_hit else "TP"
                portfolio.close_position(exit_p, "", i, reason)
                pos = None
            else:
                # 트레일링 (be_thr < 100일 때만)
                if pos.r_unit > 0 and be_thr < 100:
                    if pos.direction == Signal.LONG:
                        ur = (c - pos.entry_price) / pos.r_unit
                    else:
                        ur = (pos.entry_price - c) / pos.r_unit
                    if ur >= step_thr and pos.trailing_state != "trailing":
                        pos.sl_price = pos.entry_price + (
                            0.5 * pos.r_unit if pos.direction == Signal.LONG
                            else -0.5 * pos.r_unit
                        )
                        pos.trailing_state = "trailing"
                    elif ur >= be_thr and pos.trailing_state == "initial":
                        pos.sl_price = pos.entry_price
                        pos.trailing_state = "break_even"

        # 2. 필터 (인라인)
        filtered = False
        if weekend:
            wd = weekdays[i]
            if wd == 5 or (wd == 6 and hours[i] < 6):
                filtered = True
        if not filtered and bad_hours and hours[i] in bad_hours:
            filtered = True
        if not filtered:
            a = atrs[i]
            am = atr_medians[i]
            if not np.isnan(a) and not np.isnan(am) and a < am * dz_ratio:
                filtered = True
        if not filtered and consec_losses >= cd_losses and (i - last_loss_idx) < cd_candles:
            filtered = True

        # 3. 시그널 필터링
        sig = sig_arr[i]
        if sig == 1:  # LONG
            r = rsis[i]
            if not (rsi_l_min <= r <= rsi_l_max) or vol_ratios[i] <= vol_thr:
                sig = 0
        elif sig == 2:  # SHORT
            r = rsis[i]
            if not (rsi_s_min <= r <= rsi_s_max) or vol_ratios[i] <= vol_thr:
                sig = 0

        # ADX 필터
        if sig != 0 and adx_min > 0:
            adx_val = adx_arr[i]
            if np.isnan(adx_val) or adx_val < adx_min:
                sig = 0

        # 4. 진입
        if sig != 0 and not filtered and pos is None:
            a = atrs[i]
            if not np.isnan(a) and a > 0:
                sl_dist = sl_mult * a
                tp_dist = tp_mult * a
                size = (portfolio.capital * risk_pct) / sl_dist
                max_size = (portfolio.capital * leverage) / c
                if size > max_size:
                    size = max_size

                direction = Signal.LONG if sig == 1 else Signal.SHORT
                if direction == Signal.LONG:
                    sl = c - sl_dist
                    tp = c + tp_dist
                else:
                    sl = c + sl_dist
                    tp = c - tp_dist

                pos = Position(
                    direction=direction, entry_price=c, size=size,
                    sl_price=sl, tp_price=tp, initial_sl=sl,
                    entry_time="", entry_index=i,
                )
                portfolio.open_position(pos)

        # equity 기록
        eq = portfolio.capital
        if pos is not None:
            if pos.direction == Signal.LONG:
                eq += (c - pos.entry_price) * pos.size
            else:
                eq += (pos.entry_price - c) * pos.size
        equity.append(eq)

        # 연속 손실 추적
        if portfolio.trades and portfolio.trades[-1].exit_index == i:
            last_trade = portfolio.trades[-1]
            if last_trade.pnl < 0:
                consec_losses += 1
                last_loss_idx = i
            else:
                consec_losses = 0

    portfolio.equity_curve = equity

    if pos is not None:
        portfolio.close_position(closes[-1], "", n - 1, "END")
        pos = None

    return portfolio


def run_optimization() -> None:
    """그리드 서치 최적화를 실행한다."""
    start = SETTINGS["backtest_start"]
    end = SETTINGS["backtest_end"]

    print(f"데이터 로드 중... ({start} ~ {end})")
    df = load_from_parquet(start=start, end=end)
    if df is None or df.empty:
        print("데이터가 없습니다.")
        sys.exit(1)

    print(f"지표 계산 중... ({len(df)} 캔들)")
    df = compute_indicators(df)

    # ADX 계산
    print("ADX 계산 중...")
    df["adx"] = calc_adx(df, period=14)

    print("완화 시그널 생성 중...")
    base_signals = generate_relaxed_signals(df)
    long_count = (base_signals == Signal.LONG).sum()
    short_count = (base_signals == Signal.SHORT).sum()
    print(f"후보 시그널: LONG {long_count}, SHORT {short_count}")

    # numpy 배열 사전 계산
    sig_arr = np.zeros(len(df), dtype=np.int8)
    sig_arr[base_signals == Signal.LONG] = 1
    sig_arr[base_signals == Signal.SHORT] = 2

    precomputed = {
        "close": df["close"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "atr": df["atr"].values,
        "atr_median": df["atr_median"].values,
        "rsi": df["rsi"].values,
        "volume_ratio": df["volume_ratio"].values,
        "adx": df["adx"].values,
        "timestamps": df.index,
        "weekdays": df.index.weekday.values,
        "hours": df.index.hour.values,
        "signals": sig_arr,
    }

    # 그리드 생성
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    print(f"\n총 {len(combos)}개 파라미터 조합 테스트 시작...\n")

    results = []
    t0 = time.time()

    for idx, combo in enumerate(combos):
        params = deepcopy(SETTINGS)
        for k, v in zip(keys, combo):
            params[k] = v
        # trailing_step도 연동
        if params["trailing_be_threshold"] < 100:
            params["trailing_step_threshold"] = params["trailing_be_threshold"] + 0.5
        else:
            params["trailing_step_threshold"] = 999.0

        portfolio = fast_backtest_pf(df, params, precomputed)
        metrics = calculate_metrics(portfolio)

        # 트레일링 표시
        be_val = combo[0]
        trailing_str = "OFF" if be_val >= 100 else f"{be_val}R"

        results.append({
            "rank": 0,
            "trailing": trailing_str,
            "adx": combo[1] if combo[1] > 0 else "OFF",
            "sl": combo[2],
            "tp": combo[3],
            "rr": f"{combo[3]/combo[2]:.1f}:1",
            "trades": metrics.total_trades,
            "win%": round(metrics.win_rate, 1),
            "PF": round(metrics.profit_factor, 2),
            "return%": round(metrics.total_return_pct, 1),
            "MDD%": round(metrics.max_drawdown_pct, 1),
            "sharpe": round(metrics.sharpe_ratio, 2),
        })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(combos) - idx - 1)
            print(f"  [{idx+1}/{len(combos)}] 완료... (ETA: {eta:.0f}s)")

    # PF 기준 정렬
    results.sort(key=lambda x: x["PF"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    elapsed = time.time() - t0
    print(f"\n최적화 완료! ({elapsed:.1f}s, {len(combos)}개 조합)")

    # 상위 20개 출력
    print("\n" + "=" * 100)
    print("  TOP 20 파라미터 조합 (PF 기준)")
    print("=" * 100)

    top = results[:20]
    print(tabulate(top, headers="keys", tablefmt="simple", floatfmt=".2f"))

    # 최적 파라미터 상세
    best = results[0]
    print(f"\n{'─' * 100}")
    print(f"  BEST: Trailing={best['trailing']}, ADX>{best['adx']}, "
          f"SL={best['sl']}×ATR, TP={best['tp']}×ATR ({best['rr']})")
    print(f"  → {best['trades']}건, 승률 {best['win%']}%, PF {best['PF']}, "
          f"수익 {best['return%']}%, MDD {best['MDD%']}%, Sharpe {best['sharpe']}")
    print(f"{'─' * 100}")

    # PF >= 1.8인 조합들
    high_pf = [r for r in results if r["PF"] >= 1.8]
    if high_pf:
        print(f"\n  PF >= 1.8 조합: {len(high_pf)}개")
    else:
        print(f"\n  PF >= 1.8 조합 없음. 최고 PF: {results[0]['PF']}")


if __name__ == "__main__":
    run_optimization()
