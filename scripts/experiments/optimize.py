"""파라미터 최적화 스크립트.

핵심 파라미터 조합을 그리드 서치로 탐색하여 최적 설정을 찾는다.
"""

import itertools
import logging
import sys
import time
from copy import deepcopy

import pandas as pd
from tabulate import tabulate

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from data.storage import load_from_parquet
from strategy.signals import compute_indicators, Signal
from strategy.filters import should_filter
from strategy.position import create_position, check_sl_tp_hit, update_trailing_stop
from backtest.portfolio import Portfolio
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# === 최적화 대상 파라미터 그리드 ===
PARAM_GRID = {
    "sl_atr_multiplier": [1.0, 1.5, 2.0],
    "tp_atr_multiplier": [2.0, 3.0, 4.0],
    "volume_threshold": [1.0, 1.3],
    "rsi_long_min": [35, 45],
    "rsi_long_max": [65, 70],
    "rsi_short_min": [30, 40],
    "rsi_short_max": [60, 70],
}
# 총 조합: 3 * 3 * 2 * 2 * 2 * 2 * 2 = 288개


def fast_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    params: dict,
    capital: float = 10000,
    precomputed: dict | None = None,
) -> Portfolio:
    """시그널이 미리 계산된 상태에서 파라미터만 바꿔 빠르게 백테스트한다.

    numpy 배열 직접 접근으로 속도를 극대화한다.
    """
    from strategy.position import Position
    import numpy as np

    portfolio = Portfolio(initial_capital=capital, capital=capital)

    # numpy 배열 캐시
    p = precomputed
    closes = p["close"]
    highs = p["high"]
    lows = p["low"]
    atrs = p["atr"]
    atr_medians = p["atr_median"]
    rsis = p["rsi"]
    vol_ratios = p["volume_ratio"]
    timestamps = p["timestamps"]
    weekdays = p["weekdays"]
    hours = p["hours"]
    sig_arr = p["signals"]  # 0=NO, 1=LONG, 2=SHORT

    sl_mult = params["sl_atr_multiplier"]
    tp_mult = params["tp_atr_multiplier"]
    risk_pct = params["risk_per_trade"]
    leverage = params.get("leverage", 1)
    be_thr = params["trailing_be_threshold"]
    step_thr = params["trailing_step_threshold"]
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
                # 트레일링
                if pos.r_unit > 0:
                    if pos.direction == Signal.LONG:
                        ur = (c - pos.entry_price) / pos.r_unit
                    else:
                        ur = (pos.entry_price - c) / pos.r_unit
                    if ur >= step_thr and pos.trailing_state != "trailing":
                        pos.sl_price = pos.entry_price + (0.5 * pos.r_unit if pos.direction == Signal.LONG else -0.5 * pos.r_unit)
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

        # equity 기록 (간소화)
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


def generate_relaxed_signals(df: pd.DataFrame) -> pd.Series:
    """RSI/Volume 조건을 완화한 시그널을 벡터화로 생성한다.

    EMA 정배열/역배열 + VWAP + EMA9 돌파/이탈 + ATR 조건만 체크.
    RSI 범위와 Volume은 fast_backtest에서 파라미터별로 필터링한다.
    """
    prev_close = df["close"].shift(1)
    prev_ema_fast = df["ema_fast"].shift(1)
    prev_rsi = df["rsi"].shift(1)

    valid = df["atr"].notna() & df["rsi"].notna() & df["atr_median"].notna()

    long_mask = (
        valid
        & (df["ema_fast"] > df["ema_mid"])
        & (df["ema_mid"] > df["ema_slow"])
        & (df["close"] > df["vwap"])
        & (df["rsi"] > prev_rsi)
        & (df["close"] > df["ema_fast"])
        & (prev_close <= prev_ema_fast)
        & (df["atr"] > df["atr_median"])
    )

    short_mask = (
        valid
        & (df["ema_fast"] < df["ema_mid"])
        & (df["ema_mid"] < df["ema_slow"])
        & (df["close"] < df["vwap"])
        & (df["rsi"] < prev_rsi)
        & (df["close"] < df["ema_fast"])
        & (prev_close >= prev_ema_fast)
        & (df["atr"] > df["atr_median"])
    )

    import numpy as np
    # 0=NO_SIGNAL, 1=LONG, 2=SHORT
    sig_arr = np.zeros(len(df), dtype=np.int8)
    sig_arr[long_mask.values] = 1
    # short은 long이 아닌 곳에만
    sig_arr[short_mask.values & (sig_arr == 0)] = 2

    signals = pd.Series(Signal.NO_SIGNAL, index=df.index)
    signals.iloc[sig_arr == 1] = Signal.LONG
    signals.iloc[sig_arr == 2] = Signal.SHORT

    return signals


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

    print("완화 시그널 생성 중...")
    base_signals = generate_relaxed_signals(df)
    long_count = (base_signals == Signal.LONG).sum()
    short_count = (base_signals == Signal.SHORT).sum()
    print(f"후보 시그널: LONG {long_count}, SHORT {short_count}")

    # numpy 배열 사전 계산
    import numpy as np
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

        portfolio = fast_backtest(df, base_signals, params, precomputed=precomputed)
        metrics = calculate_metrics(portfolio)

        # 점수 계산: Sharpe * sqrt(거래수) — 너무 적은 거래는 페널티
        score = 0.0
        if metrics.total_trades >= 50 and metrics.max_drawdown_pct < 0:
            score = metrics.sharpe_ratio * (metrics.total_trades ** 0.3)

        results.append({
            "rank": 0,
            "sl_atr": combo[0],
            "tp_atr": combo[1],
            "vol_thr": combo[2],
            "rsi_l": f"{combo[3]}-{combo[4]}",
            "rsi_s": f"{combo[5]}-{combo[6]}",
            "trades": metrics.total_trades,
            "win%": round(metrics.win_rate, 1),
            "PF": round(metrics.profit_factor, 2),
            "return%": round(metrics.total_return_pct, 1),
            "MDD%": round(metrics.max_drawdown_pct, 1),
            "sharpe": round(metrics.sharpe_ratio, 2),
            "score": round(score, 2),
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(combos) - idx - 1)
            print(f"  [{idx+1}/{len(combos)}] 완료... (ETA: {eta:.0f}s)")

    # 정렬 및 랭킹
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    elapsed = time.time() - t0
    print(f"\n최적화 완료! ({elapsed:.1f}s, {len(combos)}개 조합)")

    # 상위 20개 출력
    print("\n" + "=" * 95)
    print("  TOP 20 파라미터 조합")
    print("=" * 95)

    top = results[:20]
    print(tabulate(top, headers="keys", tablefmt="simple", floatfmt=".2f"))

    # 최적 파라미터 상세
    best = results[0]
    print(f"\n{'─' * 95}")
    print(f"  BEST: SL={best['sl_atr']}×ATR, TP={best['tp_atr']}×ATR, "
          f"Vol>{best['vol_thr']}, RSI_L={best['rsi_l']}, RSI_S={best['rsi_s']}")
    print(f"  → {best['trades']}건, 승률 {best['win%']}%, PF {best['PF']}, "
          f"수익 {best['return%']}%, MDD {best['MDD%']}%, Sharpe {best['sharpe']}")
    print(f"{'─' * 95}")


if __name__ == "__main__":
    run_optimization()
