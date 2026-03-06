"""수수료(taker 0.04%)를 이기는 전략 파라미터 탐색 - 다단계 최적화."""

import os
os.environ["TQDM_DISABLE"] = "1"

import logging
import time
import json
from itertools import product
from copy import deepcopy

import pandas as pd
import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from config.settings import SETTINGS
from strategy.signals import compute_indicators, generate_signals, Signal
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  공통 유틸리티
# ═══════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """데이터를 로드하고 전처리한다."""
    data_path = SETTINGS["cache_dir"] + "/BTCUSDT_5m.parquet"
    print(f"데이터 로드: {data_path}")
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
    print(f"  {len(df)}봉, {df.index[0]} ~ {df.index[-1]}")
    return df


def run_single(df, signals=None, capital=10000, **overrides):
    """단일 백테스트 실행. overrides로 SETTINGS를 임시 변경."""
    saved = {}
    for k, v in overrides.items():
        saved[k] = SETTINGS.get(k)
        SETTINGS[k] = v
    try:
        portfolio = run_backtest(df.copy(), capital=capital, pre_signals=signals)
        m = calculate_metrics(portfolio)
        return {
            "trades": m.total_trades,
            "win_rate": round(m.win_rate, 1),
            "pf": round(m.profit_factor, 2),
            "return_pct": round(m.total_return_pct, 1),
            "mdd": round(m.max_drawdown_pct, 1),
            "sharpe": round(m.sharpe_ratio, 2),
            "calmar": round(m.calmar_ratio, 2),
            "avg_r": round(m.avg_r_multiple, 3),
            "max_consec_loss": m.max_consecutive_losses,
            "final": round(portfolio.capital, 0),
            "commission": round(portfolio.total_commission, 0),
        }
    except Exception as e:
        return {"trades": 0, "pf": 0, "return_pct": -100, "mdd": -100,
                "sharpe": 0, "calmar": 0, "error": str(e)}
    finally:
        for k, v in saved.items():
            SETTINGS[k] = v


def print_result(label, r):
    """결과 한 줄 출력."""
    pf = r.get("pf", 0)
    ret = r.get("return_pct", 0)
    # PF > 1 이고 수익이면 마커 표시
    marker = " ★" if pf > 1.05 and ret > 0 else ""
    print(f"  {label:<40} trades={r.get('trades',0):>5}  WR={r.get('win_rate',0):>5.1f}%  "
          f"PF={pf:>5.2f}  ret={ret:>10.1f}%  MDD={r.get('mdd',0):>6.1f}%  "
          f"sharpe={r.get('sharpe',0):>5.2f}  fee=${r.get('commission',0):>8,.0f}{marker}",
          flush=True)


def top_results(results, key="pf", n=5, min_trades=50):
    """상위 N개 결과 반환."""
    filtered = [r for r in results if r.get("trades", 0) >= min_trades]
    return sorted(filtered, key=lambda x: x.get(key, 0), reverse=True)[:n]


# ═══════════════════════════════════════════════
#  Phase 1: SL / TP 그리드
# ═══════════════════════════════════════════════

def phase1_sl_tp(df, signals):
    """SL/TP 배수 최적화. 시그널 재사용으로 빠름."""
    print("\n" + "=" * 80)
    print("  PHASE 1: SL / TP 배수 최적화")
    print("=" * 80)

    sl_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    tp_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

    results = []
    for sl, tp in product(sl_values, tp_values):
        if tp <= sl:
            continue  # TP는 SL보다 커야 함
        r = run_single(df, signals, sl_atr_multiplier=sl, tp_atr_multiplier=tp)
        r["sl"] = sl
        r["tp"] = tp
        results.append(r)
        print_result(f"SL={sl:.1f} TP={tp:.1f}", r)

    print("\n  --- Phase 1 Top 5 (PF) ---")
    for r in top_results(results, "pf"):
        print_result(f"  SL={r['sl']:.1f} TP={r['tp']:.1f}", r)

    print("\n  --- Phase 1 Top 5 (Sharpe) ---")
    for r in top_results(results, "sharpe"):
        print_result(f"  SL={r['sl']:.1f} TP={r['tp']:.1f}", r)

    return results


# ═══════════════════════════════════════════════
#  Phase 2: 시그널 품질 (vol / atr threshold)
# ═══════════════════════════════════════════════

def phase2_signals(df_raw, best_sl, best_tp):
    """시그널 파라미터 최적화. 시그널 재생성 필요."""
    print("\n" + "=" * 80)
    print(f"  PHASE 2: 시그널 품질 최적화 (SL={best_sl}, TP={best_tp})")
    print("=" * 80)

    vol_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    atr_values = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    results = []
    for vol, atr in product(vol_values, atr_values):
        SETTINGS["volume_threshold"] = vol
        SETTINGS["atr_signal_multiplier"] = atr

        df = df_raw.copy()
        df = compute_indicators(df)
        sigs = generate_signals(df)

        r = run_single(df, sigs, sl_atr_multiplier=best_sl, tp_atr_multiplier=best_tp)
        r["vol"] = vol
        r["atr"] = atr
        results.append(r)
        print_result(f"vol={vol:.1f} atr={atr:.1f}", r)

    print("\n  --- Phase 2 Top 5 (PF) ---")
    for r in top_results(results, "pf"):
        print_result(f"  vol={r['vol']:.1f} atr={r['atr']:.1f}", r)

    print("\n  --- Phase 2 Top 5 (Sharpe) ---")
    for r in top_results(results, "sharpe"):
        print_result(f"  vol={r['vol']:.1f} atr={r['atr']:.1f}", r)

    return results


# ═══════════════════════════════════════════════
#  Phase 3: 트레일링 스탑 변형
# ═══════════════════════════════════════════════

def phase3_trailing(df, signals, best_sl, best_tp):
    """트레일링 스탑 파라미터 최적화."""
    print("\n" + "=" * 80)
    print(f"  PHASE 3: 트레일링 스탑 최적화 (SL={best_sl}, TP={best_tp})")
    print("=" * 80)

    # 999 = 사실상 비활성화
    be_values = [0.5, 0.8, 1.0, 1.5, 2.0, 999]
    step_values = [0.8, 1.0, 1.5, 2.0, 2.5, 999]

    results = []
    for be, step in product(be_values, step_values):
        if step <= be and step != 999 and be != 999:
            continue
        be_str = "OFF" if be >= 999 else f"{be:.1f}"
        step_str = "OFF" if step >= 999 else f"{step:.1f}"
        label = f"BE={be_str} STEP={step_str}"
        r = run_single(df, signals,
                       sl_atr_multiplier=best_sl, tp_atr_multiplier=best_tp,
                       trailing_be_threshold=be, trailing_step_threshold=step)
        r["be"] = be
        r["step"] = step
        results.append(r)
        print_result(label, r)

    print("\n  --- Phase 3 Top 5 (PF) ---")
    for r in top_results(results, "pf"):
        be_str = "OFF" if r['be'] >= 999 else f"{r['be']:.1f}"
        step_str = "OFF" if r['step'] >= 999 else f"{r['step']:.1f}"
        label = f"  BE={be_str} STEP={step_str}"
        print_result(label, r)

    return results


# ═══════════════════════════════════════════════
#  Phase 4: EMA 기간
# ═══════════════════════════════════════════════

def phase4_ema(df_raw, best_params):
    """EMA 기간 최적화. 지표 전체 재계산 필요."""
    print("\n" + "=" * 80)
    print("  PHASE 4: EMA 기간 최적화")
    print("=" * 80)

    ema_combos = [
        (5, 13, 34),
        (8, 21, 55),
        (9, 21, 50),   # 기본값
        (10, 20, 50),
        (12, 26, 50),
        (5, 21, 50),
        (9, 21, 100),
        (20, 50, 100),
        (7, 14, 28),
        (5, 10, 20),
    ]

    results = []
    for fast, mid, slow in ema_combos:
        SETTINGS["ema_fast"] = fast
        SETTINGS["ema_mid"] = mid
        SETTINGS["ema_slow"] = slow
        SETTINGS["volume_threshold"] = best_params["vol"]
        SETTINGS["atr_signal_multiplier"] = best_params["atr"]

        df = df_raw.copy()
        df = compute_indicators(df)
        sigs = generate_signals(df)

        r = run_single(df, sigs,
                       sl_atr_multiplier=best_params["sl"],
                       tp_atr_multiplier=best_params["tp"],
                       trailing_be_threshold=best_params["be"],
                       trailing_step_threshold=best_params["step"])
        r["ema"] = f"{fast}/{mid}/{slow}"
        r["ema_fast"] = fast
        r["ema_mid"] = mid
        r["ema_slow"] = slow
        results.append(r)
        print_result(f"EMA {fast}/{mid}/{slow}", r)

    print("\n  --- Phase 4 Top 5 (PF) ---")
    for r in top_results(results, "pf"):
        print_result(f"  EMA {r['ema']}", r)

    return results


# ═══════════════════════════════════════════════
#  Phase 5: 필터 조합
# ═══════════════════════════════════════════════

def phase5_filters(df, signals, best_params):
    """필터 조합 최적화."""
    print("\n" + "=" * 80)
    print("  PHASE 5: 필터 조합 최적화")
    print("=" * 80)

    base_overrides = {
        "sl_atr_multiplier": best_params["sl"],
        "tp_atr_multiplier": best_params["tp"],
        "trailing_be_threshold": best_params["be"],
        "trailing_step_threshold": best_params["step"],
    }

    filter_combos = [
        {"label": "현재 필터", "weekend_filter": True,
         "bad_hours_utc": [3, 5, 10, 12, 16, 19, 21],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12},
        {"label": "필터 전부 OFF", "weekend_filter": False,
         "bad_hours_utc": [],
         "cooldown_after_consecutive_losses": 99, "cooldown_candles": 0},
        {"label": "주말만 ON", "weekend_filter": True,
         "bad_hours_utc": [],
         "cooldown_after_consecutive_losses": 99, "cooldown_candles": 0},
        {"label": "bad_hours만 ON", "weekend_filter": False,
         "bad_hours_utc": [3, 5, 10, 12, 16, 19, 21],
         "cooldown_after_consecutive_losses": 99, "cooldown_candles": 0},
        {"label": "쿨다운만 ON (3연패/12봉)", "weekend_filter": False,
         "bad_hours_utc": [],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12},
        {"label": "쿨다운 강화 (2연패/24봉)", "weekend_filter": False,
         "bad_hours_utc": [],
         "cooldown_after_consecutive_losses": 2, "cooldown_candles": 24},
        {"label": "주말+쿨다운", "weekend_filter": True,
         "bad_hours_utc": [],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12},
        {"label": "주말+bad_hours", "weekend_filter": True,
         "bad_hours_utc": [3, 5, 10, 12, 16, 19, 21],
         "cooldown_after_consecutive_losses": 99, "cooldown_candles": 0},
        {"label": "확장 bad_hours", "weekend_filter": True,
         "bad_hours_utc": [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 16, 19, 20, 21, 22, 23],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12},
        {"label": "최소 bad_hours", "weekend_filter": True,
         "bad_hours_utc": [3, 10, 16],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12},
        {"label": "일일 4매매 제한", "weekend_filter": True,
         "bad_hours_utc": [3, 5, 10, 12, 16, 19, 21],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12,
         "daily_max_trades": 4},
        {"label": "일일 3매매 + 2손절", "weekend_filter": True,
         "bad_hours_utc": [3, 5, 10, 12, 16, 19, 21],
         "cooldown_after_consecutive_losses": 3, "cooldown_candles": 12,
         "daily_max_trades": 3, "daily_max_losses": 2},
    ]

    results = []
    for combo in filter_combos:
        label = combo.pop("label")
        overrides = {**base_overrides, **combo}
        r = run_single(df, signals, **overrides)
        r["filter"] = label
        results.append(r)
        print_result(label, r)
        combo["label"] = label  # restore

    print("\n  --- Phase 5 Top 5 (PF) ---")
    for r in top_results(results, "pf"):
        print_result(f"  {r['filter']}", r)

    return results


# ═══════════════════════════════════════════════
#  Phase 6: RSI 범위
# ═══════════════════════════════════════════════

def phase6_rsi(df_raw, best_params):
    """RSI 범위 최적화."""
    print("\n" + "=" * 80)
    print("  PHASE 6: RSI 범위 최적화")
    print("=" * 80)

    rsi_combos = [
        (35, 70, 40, 60),   # 현재
        (30, 70, 30, 70),   # 넓게
        (40, 65, 35, 60),   # CLAUDE.md 기본
        (30, 75, 25, 75),   # 매우 넓게
        (45, 65, 35, 55),   # 좁게
        (35, 60, 40, 65),   # 약간 보수적
        (30, 65, 35, 70),   # 비대칭
        (25, 75, 25, 75),   # RSI 거의 무시
        (50, 80, 20, 50),   # 극단적 모멘텀
        (35, 70, 35, 70),   # 대칭
    ]

    # 최적 EMA/vol/atr 설정
    SETTINGS["ema_fast"] = best_params.get("ema_fast", 9)
    SETTINGS["ema_mid"] = best_params.get("ema_mid", 21)
    SETTINGS["ema_slow"] = best_params.get("ema_slow", 50)
    SETTINGS["volume_threshold"] = best_params["vol"]
    SETTINGS["atr_signal_multiplier"] = best_params["atr"]

    results = []
    for rsi_l_min, rsi_l_max, rsi_s_min, rsi_s_max in rsi_combos:
        SETTINGS["rsi_long_min"] = rsi_l_min
        SETTINGS["rsi_long_max"] = rsi_l_max
        SETTINGS["rsi_short_min"] = rsi_s_min
        SETTINGS["rsi_short_max"] = rsi_s_max

        df = df_raw.copy()
        df = compute_indicators(df)
        sigs = generate_signals(df)

        r = run_single(df, sigs,
                       sl_atr_multiplier=best_params["sl"],
                       tp_atr_multiplier=best_params["tp"],
                       trailing_be_threshold=best_params["be"],
                       trailing_step_threshold=best_params["step"])
        r["rsi"] = f"L({rsi_l_min}-{rsi_l_max}) S({rsi_s_min}-{rsi_s_max})"
        r["rsi_long_min"] = rsi_l_min
        r["rsi_long_max"] = rsi_l_max
        r["rsi_short_min"] = rsi_s_min
        r["rsi_short_max"] = rsi_s_max
        results.append(r)
        print_result(f"RSI L({rsi_l_min}-{rsi_l_max}) S({rsi_s_min}-{rsi_s_max})", r)

    print("\n  --- Phase 6 Top 5 (PF) ---")
    for r in top_results(results, "pf"):
        print_result(f"  {r['rsi']}", r)

    return results


# ═══════════════════════════════════════════════
#  Phase 7: Risk / Leverage 최종 조정
# ═══════════════════════════════════════════════

def phase7_risk_leverage(df, signals, best_params):
    """리스크/레버리지 최종 최적화."""
    print("\n" + "=" * 80)
    print("  PHASE 7: Risk / Leverage 최적화")
    print("=" * 80)

    risk_values = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    leverage_values = [5, 10, 15, 20]

    base_overrides = {
        "sl_atr_multiplier": best_params["sl"],
        "tp_atr_multiplier": best_params["tp"],
        "trailing_be_threshold": best_params["be"],
        "trailing_step_threshold": best_params["step"],
    }

    results = []
    for risk, lev in product(risk_values, leverage_values):
        overrides = {**base_overrides, "risk_per_trade": risk, "leverage": lev}
        r = run_single(df, signals, **overrides)
        r["risk"] = risk
        r["leverage"] = lev
        results.append(r)
        print_result(f"risk={risk*100:.1f}% lev={lev}x", r)

    print("\n  --- Phase 7 Top 5 (Sharpe) ---")
    for r in top_results(results, "sharpe"):
        print_result(f"  risk={r['risk']*100:.1f}% lev={r['leverage']}x", r)

    print("\n  --- Phase 7 Top 5 (Return, PF>1.0) ---")
    profitable = [r for r in results if r.get("pf", 0) > 1.0]
    for r in sorted(profitable, key=lambda x: x.get("return_pct", 0), reverse=True)[:5]:
        print_result(f"  risk={r['risk']*100:.1f}% lev={r['leverage']}x", r)

    return results


# ═══════════════════════════════════════════════
#  메인 실행
# ═══════════════════════════════════════════════

def main():
    import sys
    resume_phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_time = time.time()

    # 수수료 설정 (현실적)
    SETTINGS["maker_fee"] = 0.0000
    SETTINGS["taker_fee"] = 0.0004
    SETTINGS["slippage_rate"] = 0.0000
    SETTINGS["leverage"] = 20
    SETTINGS["risk_per_trade"] = 0.05

    print("=" * 80)
    print("  수수료 저항 전략 탐색 (taker 0.04% on SL)")
    print("=" * 80)
    print(f"  maker_fee={SETTINGS['maker_fee']}, taker_fee={SETTINGS['taker_fee']}")
    print(f"  leverage={SETTINGS['leverage']}x, risk={SETTINGS['risk_per_trade']*100}%")
    print(f"  Resume from Phase {resume_phase}")

    df_raw = load_data()

    p1 = []
    p2 = []

    if resume_phase <= 1:
        # ─── Phase 1: SL/TP ───
        SETTINGS["volume_threshold"] = 1.0
        SETTINGS["atr_signal_multiplier"] = 1.0
        df1 = df_raw.copy()
        df1 = compute_indicators(df1)
        signals1 = generate_signals(df1)
        print(f"\n  기본 시그널: LONG {(signals1==Signal.LONG).sum()}, SHORT {(signals1==Signal.SHORT).sum()}")

        p1 = phase1_sl_tp(df1, signals1)
        best_p1 = top_results(p1, "pf", n=1, min_trades=50)
        if not best_p1:
            best_p1 = top_results(p1, "return_pct", n=1, min_trades=10)
        b1 = best_p1[0]
        best_sl, best_tp = b1["sl"], b1["tp"]
        print(f"\n  ★ Phase 1 Best: SL={best_sl}, TP={best_tp}, PF={b1['pf']}")
    else:
        # Phase 1 결과 하드코딩 (이전 실행에서 확인됨)
        best_sl, best_tp = 1.2, 3.0
        print(f"\n  ★ Phase 1 (cached): SL={best_sl}, TP={best_tp}")

    if resume_phase <= 2:
        # ─── Phase 2: Signal quality ───
        p2 = phase2_signals(df_raw, best_sl, best_tp)
        best_p2 = top_results(p2, "pf", n=1, min_trades=50)
        if not best_p2:
            best_p2 = top_results(p2, "return_pct", n=1, min_trades=10)
        b2 = best_p2[0]
        best_vol, best_atr = b2["vol"], b2["atr"]
        print(f"\n  ★ Phase 2 Best: vol={best_vol}, atr={best_atr}, PF={b2['pf']}")
    else:
        # Phase 2 결과 하드코딩
        best_vol, best_atr = 1.2, 1.5
        print(f"\n  ★ Phase 2 (cached): vol={best_vol}, atr={best_atr}")

    # ─── Phase 3: Trailing ───
    SETTINGS["volume_threshold"] = best_vol
    SETTINGS["atr_signal_multiplier"] = best_atr
    df3 = df_raw.copy()
    df3 = compute_indicators(df3)
    signals3 = generate_signals(df3)

    p3 = phase3_trailing(df3, signals3, best_sl, best_tp)
    best_p3 = top_results(p3, "pf", n=1, min_trades=50)
    if not best_p3:
        best_p3 = top_results(p3, "return_pct", n=1, min_trades=10)
    b3 = best_p3[0]
    best_be, best_step = b3["be"], b3["step"]
    print(f"\n  ★ Phase 3 Best: BE={best_be}, STEP={best_step}, PF={b3['pf']}")

    best_params = {
        "sl": best_sl, "tp": best_tp,
        "vol": best_vol, "atr": best_atr,
        "be": best_be, "step": best_step,
    }

    # ─── Phase 4: EMA ───
    p4 = phase4_ema(df_raw, best_params)
    best_p4 = top_results(p4, "pf", n=1, min_trades=50)
    if best_p4 and best_p4[0]["pf"] > b3["pf"]:
        b4 = best_p4[0]
        best_params["ema_fast"] = b4["ema_fast"]
        best_params["ema_mid"] = b4["ema_mid"]
        best_params["ema_slow"] = b4["ema_slow"]
        print(f"\n  ★ Phase 4 Best: EMA {b4['ema']}, PF={b4['pf']}")
    else:
        best_params["ema_fast"] = 9
        best_params["ema_mid"] = 21
        best_params["ema_slow"] = 50
        print(f"\n  ★ Phase 4: 기본 EMA 9/21/50 유지")

    # ─── Phase 5: Filters ───
    SETTINGS["ema_fast"] = best_params["ema_fast"]
    SETTINGS["ema_mid"] = best_params["ema_mid"]
    SETTINGS["ema_slow"] = best_params["ema_slow"]
    SETTINGS["volume_threshold"] = best_vol
    SETTINGS["atr_signal_multiplier"] = best_atr
    df5 = df_raw.copy()
    df5 = compute_indicators(df5)
    signals5 = generate_signals(df5)

    p5 = phase5_filters(df5, signals5, best_params)
    best_p5 = top_results(p5, "pf", n=1, min_trades=50)
    if best_p5:
        print(f"\n  ★ Phase 5 Best Filter: {best_p5[0]['filter']}, PF={best_p5[0]['pf']}")

    # ─── Phase 6: RSI ───
    p6 = phase6_rsi(df_raw, best_params)
    best_p6 = top_results(p6, "pf", n=1, min_trades=50)
    if best_p6 and best_p6[0]["pf"] > b3["pf"]:
        b6 = best_p6[0]
        best_params["rsi_long_min"] = b6["rsi_long_min"]
        best_params["rsi_long_max"] = b6["rsi_long_max"]
        best_params["rsi_short_min"] = b6["rsi_short_min"]
        best_params["rsi_short_max"] = b6["rsi_short_max"]
        print(f"\n  ★ Phase 6 Best: {b6['rsi']}, PF={b6['pf']}")
    else:
        print(f"\n  ★ Phase 6: 기존 RSI 유지")

    # ─── Phase 7: Risk/Leverage ───
    # 최적 RSI/EMA/vol/atr 반영한 시그널 재계산
    SETTINGS["ema_fast"] = best_params.get("ema_fast", 9)
    SETTINGS["ema_mid"] = best_params.get("ema_mid", 21)
    SETTINGS["ema_slow"] = best_params.get("ema_slow", 50)
    SETTINGS["volume_threshold"] = best_vol
    SETTINGS["atr_signal_multiplier"] = best_atr
    if "rsi_long_min" in best_params:
        SETTINGS["rsi_long_min"] = best_params["rsi_long_min"]
        SETTINGS["rsi_long_max"] = best_params["rsi_long_max"]
        SETTINGS["rsi_short_min"] = best_params["rsi_short_min"]
        SETTINGS["rsi_short_max"] = best_params["rsi_short_max"]

    df7 = df_raw.copy()
    df7 = compute_indicators(df7)
    signals7 = generate_signals(df7)

    p7 = phase7_risk_leverage(df7, signals7, best_params)

    # ═══════════════════════════════════════════════
    #  최종 결과 요약
    # ═══════════════════════════════════════════════
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  최종 결과 요약")
    print("=" * 80)
    print(f"  소요 시간: {elapsed/60:.1f}분")
    print(f"\n  최적 파라미터:")
    print(json.dumps(best_params, indent=4, default=str))

    # 최종 best risk/leverage
    best_p7_sharpe = top_results(p7, "sharpe", n=1, min_trades=50)
    best_p7_return = [r for r in p7 if r.get("pf", 0) > 1.0]
    best_p7_return = sorted(best_p7_return, key=lambda x: x.get("return_pct", 0), reverse=True)[:1]

    if best_p7_sharpe:
        r = best_p7_sharpe[0]
        print(f"\n  [Sharpe 최적] risk={r['risk']*100:.1f}% lev={r['leverage']}x")
        print_result("    ", r)

    if best_p7_return:
        r = best_p7_return[0]
        print(f"\n  [수익률 최적] risk={r['risk']*100:.1f}% lev={r['leverage']}x")
        print_result("    ", r)

    # 모든 결과 CSV 저장
    all_results = {
        "phase1_sl_tp": p1,
        "phase2_signals": p2,
        "phase3_trailing": p3,
        "phase4_ema": p4,
        "phase5_filters": p5,
        "phase6_rsi": p6,
        "phase7_risk": p7,
    }
    for name, data in all_results.items():
        pd.DataFrame(data).to_csv(f"opt_{name}.csv", index=False)
    print(f"\n  결과 CSV 저장 완료 (opt_phase*.csv)")


if __name__ == "__main__":
    main()
