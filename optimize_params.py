#!/usr/bin/env python3
"""파라미터 최적화 스크립트.

2025년 데이터로 Grid Search → 2026년 1~3월 Validation.
기존 설정을 변경하지 않고 결과만 출력한다.

최적화 전략:
- 지표 계산 + 시그널 후보를 1회 사전 계산 (volume_threshold=1.0 기준)
- 각 파라미터 조합은 시그널 필터링 + SL/TP 시뮬레이션만 수행
"""

import itertools
import logging
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 탐색 공간 ──────────────────────────────────────────────────────

PARAM_GRID = {
    "sl_atr_multiplier": [0.8, 1.0, 1.2, 1.5, 2.0],
    "tp_atr_multiplier": [2.0, 2.5, 3.0, 3.6, 4.0, 5.0],
    "short_enabled": [True, False],
    "volume_threshold": [1.0, 1.2, 1.5, 2.0],
}


@dataclass
class OptResult:
    """최적화 결과."""
    params: dict
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_r: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    calmar: float
    long_trades: int
    long_wr: float
    long_pf: float
    short_trades: int
    short_wr: float
    short_pf: float
    fitness: float = 0.0


@dataclass
class PrecomputedSignal:
    """사전 계산된 시그널."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timestamp_idx: int  # timestamps 리스트 내 인덱스
    close: float
    atr: float
    atr_median: float
    rsi: float
    volume_ratio: float
    score: float


@dataclass
class CandleInfo:
    """SL/TP 체크에 필요한 캔들 정보."""
    high: float
    low: float
    close: float


# ── 워커 공유 데이터 ──

_worker_state = {}


def _worker_init(precomputed_path: str):
    """워커 초기화: 사전 계산된 데이터 로드."""
    import logging as _log
    _log.basicConfig(level=logging.WARNING)

    with open(precomputed_path, "rb") as f:
        data = pickle.load(f)

    _worker_state.update(data)


def _precompute_quarter(all_data_path: str, daily_active_path: str, cache_prefix: str) -> str:
    """분기 데이터에서 지표 + 시그널 후보 + 캔들 HLC를 사전 계산한다."""
    precomputed_path = f"/tmp/{cache_prefix}_precomputed.pkl"
    if os.path.exists(precomputed_path):
        logger.info("사전계산 캐시 로드: %s", cache_prefix)
        return precomputed_path

    logger.info("사전계산 시작: %s", cache_prefix)

    with open(all_data_path, "rb") as f:
        all_data = pickle.load(f)
    with open(daily_active_path, "rb") as f:
        daily_active = pickle.load(f)

    from strategy.signals import compute_indicators, check_long_conditions, check_short_conditions, Signal
    from strategy.filters import should_filter
    from scanner.config import SCANNER_SETTINGS
    from scanner.prioritizer import compute_score
    from scanner.signal_engine import SignalEvent

    # 1. 지표 계산
    indicator_data = {}
    for sym, df in all_data.items():
        try:
            indicator_data[sym] = compute_indicators(df)
        except Exception:
            pass

    # 2. 타임스텝 구성
    all_timestamps = set()
    for df in indicator_data.values():
        all_timestamps.update(df.index.tolist())
    timestamps = sorted(all_timestamps)

    sym_index_map = {}
    for sym, df in indicator_data.items():
        sym_index_map[sym] = {ts: i for i, ts in enumerate(df.index)}

    # 3. 시그널 사전 계산 (volume_threshold=1.0, short=True — 가장 느슨한 조건)
    from config.settings import SETTINGS
    orig_vol = SETTINGS["volume_threshold"]
    SETTINGS["volume_threshold"] = 1.0  # 가장 느슨하게

    signals_by_ti = {}  # ti -> list of PrecomputedSignal
    min_atr_pct = SCANNER_SETTINGS.get("min_atr_pct")

    current_date_str = ""
    active_symbols = []

    for ti, timestamp in enumerate(timestamps):
        date_str = str(timestamp.date()) if hasattr(timestamp, 'date') else str(timestamp)[:10]

        if date_str != current_date_str:
            current_date_str = date_str
            active_symbols = daily_active.get(date_str, [])
            active_symbols = [s for s in active_symbols if s in indicator_data]

        for sym in active_symbols:
            df = indicator_data.get(sym)
            if df is None:
                continue
            idx_map = sym_index_map[sym]
            if timestamp not in idx_map:
                continue
            row_idx = idx_map[timestamp]
            if row_idx < 2:
                continue

            row = df.iloc[row_idx]
            prev = df.iloc[row_idx - 1]

            if pd.isna(row.get("atr")) or pd.isna(row.get("rsi")) or pd.isna(row.get("volume_ratio")):
                continue

            atr = row["atr"]
            atr_med = row["atr_median"] if not pd.isna(row.get("atr_median")) else atr

            # ATR % 필터 (고정 — 파라미터 그리드 외)
            if min_atr_pct and row["close"] > 0:
                atr_pct = atr / row["close"] * 100
                if atr_pct < min_atr_pct:
                    continue

            # 필터 체크 (deadzone, weekend 등 — 파라미터 그리드 외)
            # 쿨다운은 포트폴리오 상태 의존이므로 여기서 스킵
            if should_filter(timestamp, atr, atr_med, 0, 999):
                continue

            direction = None
            if check_long_conditions(row, prev):
                direction = "LONG"
            elif check_short_conditions(row, prev):
                direction = "SHORT"

            if direction is None:
                continue

            sig_direction = Signal.LONG if direction == "LONG" else Signal.SHORT
            event = SignalEvent(
                symbol=sym, direction=sig_direction, timestamp=timestamp,
                close=row["close"], atr=atr, atr_median=atr_med,
                rsi=row["rsi"], volume_ratio=row["volume_ratio"],
            )
            event.score = compute_score(event)

            signals_by_ti.setdefault(ti, []).append(PrecomputedSignal(
                symbol=sym,
                direction=direction,
                timestamp_idx=ti,
                close=row["close"],
                atr=atr,
                atr_median=atr_med,
                rsi=row["rsi"],
                volume_ratio=row["volume_ratio"],
                score=event.score,
            ))

    SETTINGS["volume_threshold"] = orig_vol  # 복원

    # 4. 캔들 HLC 저장 (SL/TP 체크용)
    candle_hlc = {}  # (sym, ti) -> CandleInfo
    for sym, df in indicator_data.items():
        idx_map = sym_index_map[sym]
        for ts, row_idx in idx_map.items():
            ti = timestamps.index(ts) if ts in timestamps[:10] else None  # 느림 — 딕셔너리 사용

    # 더 효율적인 방법: ti_map
    ts_to_ti = {ts: ti for ti, ts in enumerate(timestamps)}

    for sym, df in indicator_data.items():
        idx_map = sym_index_map[sym]
        for ts, row_idx in idx_map.items():
            ti = ts_to_ti.get(ts)
            if ti is None:
                continue
            row = df.iloc[row_idx]
            candle_hlc[(sym, ti)] = CandleInfo(
                high=row["high"], low=row["low"], close=row["close"]
            )

    # 5. daily_active를 ti 기반으로 변환
    daily_active_ti = {}  # date_str -> list of symbols
    current_date_str = ""
    for ti, ts in enumerate(timestamps):
        date_str = str(ts.date()) if hasattr(ts, 'date') else str(ts)[:10]
        if date_str != current_date_str:
            current_date_str = date_str
            daily_active_ti[date_str] = daily_active.get(date_str, [])

    n_signals = sum(len(v) for v in signals_by_ti.values())
    logger.info("사전계산 완료: %d 타임스텝, %d 시그널 후보, %d 캔들 HLC",
                len(timestamps), n_signals, len(candle_hlc))

    precomputed = {
        "timestamps": timestamps,
        "signals_by_ti": signals_by_ti,
        "candle_hlc": candle_hlc,
        "daily_active_ti": daily_active_ti,
        "n_timestamps": len(timestamps),
    }

    with open(precomputed_path, "wb") as f:
        pickle.dump(precomputed, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("사전계산 캐시 저장: %s (시그널 %d개)", cache_prefix, n_signals)
    return precomputed_path


def _compute_metrics(trades: list, equity_curve: list, final_capital: float, capital: float) -> dict:
    """거래 목록과 equity curve에서 성과 지표를 추출한다."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_return_pct": 0, "max_drawdown_pct": 0,
            "sharpe": 0, "calmar": 0,
            "long_trades": 0, "long_wr": 0, "long_pf": 0,
            "short_trades": 0, "short_wr": 0, "short_pf": 0,
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in losses)) if losses else 1
    pf = gp / gl if gl > 0 else 0

    longs = [t for t in trades if t["dir"] == "LONG"]
    shorts = [t for t in trades if t["dir"] == "SHORT"]
    l_wins = [t for t in longs if t["pnl"] > 0]
    s_wins = [t for t in shorts if t["pnl"] > 0]
    l_gp = sum(t["pnl"] for t in l_wins) if l_wins else 0
    l_gl = abs(sum(t["pnl"] for t in longs if t["pnl"] <= 0)) if longs else 1
    s_gp = sum(t["pnl"] for t in s_wins) if s_wins else 0
    s_gl = abs(sum(t["pnl"] for t in shorts if t["pnl"] <= 0)) if shorts else 1

    eq = equity_curve
    peak = eq[0] if eq else capital
    mdd = 0
    for e in eq:
        if e > peak:
            peak = e
        dd = (e - peak) / peak * 100
        if dd < mdd:
            mdd = dd

    daily_step = 288
    daily_returns = []
    for i in range(daily_step, len(eq), daily_step):
        r = (eq[i] - eq[i - daily_step]) / eq[i - daily_step]
        daily_returns.append(r)
    if daily_returns:
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(365)
    else:
        sharpe = 0

    ret = (final_capital / capital - 1) * 100
    calmar = abs(ret / mdd) if mdd != 0 else 0

    r_multiples = [t["r_mul"] for t in trades if t["r_mul"] is not None]

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor": pf,
        "avg_r": np.mean(r_multiples) if r_multiples else 0,
        "total_return_pct": ret,
        "max_drawdown_pct": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "long_trades": len(longs),
        "long_wr": len(l_wins) / len(longs) * 100 if longs else 0,
        "long_pf": l_gp / l_gl if l_gl > 0 else 0,
        "short_trades": len(shorts),
        "short_wr": len(s_wins) / len(shorts) * 100 if shorts else 0,
        "short_pf": s_gp / s_gl if s_gl > 0 else 0,
    }


def _run_fast_backtest(args: tuple) -> dict:
    """사전계산된 시그널로 빠른 백테스트를 실행한다."""
    params, capital, max_positions = args

    signals_by_ti = _worker_state["signals_by_ti"]
    candle_hlc = _worker_state["candle_hlc"]
    daily_active_ti = _worker_state["daily_active_ti"]
    timestamps = _worker_state["timestamps"]
    n_timestamps = _worker_state["n_timestamps"]

    sl_mult = params["sl_atr_multiplier"]
    tp_mult = params["tp_atr_multiplier"]
    vol_thresh = params["volume_threshold"]
    short_enabled = params.get("short_enabled", True)

    # 분할익절 계산
    r_ratio = tp_mult / sl_mult
    partial_tp_enabled = r_ratio >= 2.0
    if partial_tp_enabled:
        partial_r = r_ratio  # 이 R에서 75% 익절
        partial_fraction = 0.75
        final_tp_r = r_ratio + 1.0  # 잔여 물량의 TP
    else:
        partial_r = partial_fraction = final_tp_r = None

    from config.settings import SETTINGS
    leverage = SETTINGS.get("leverage", 20)
    risk_pct = SETTINGS.get("risk_per_trade", 0.05)
    maker_fee = SETTINGS.get("maker_fee", 0.0002)
    taker_fee = SETTINGS.get("taker_fee", 0.0005)
    max_cap_pct = 0.33

    # 포트폴리오 상태
    port_capital = capital
    positions = {}  # sym -> {dir, entry, sl, tp, size, orig_size, r_unit, entry_ti, partial_done, trailing_state, tp_levels}
    trades = []
    equity_curve = []

    current_date_str = ""
    active_set = set()

    for ti in range(n_timestamps):
        ts = timestamps[ti]
        date_str = str(ts.date()) if hasattr(ts, 'date') else str(ts)[:10]

        if date_str != current_date_str:
            current_date_str = date_str
            active_set = set(daily_active_ti.get(date_str, []))
            active_set |= set(positions.keys())

        # 1. SL/TP 체크 for open positions
        closed_syms = []
        for sym in list(positions.keys()):
            candle = candle_hlc.get((sym, ti))
            if candle is None:
                continue

            pos = positions[sym]
            high, low = candle.high, candle.low

            # SL 체크
            hit = False
            exit_price = 0
            reason = ""

            if pos["dir"] == "LONG":
                if low <= pos["sl"]:
                    hit, exit_price, reason = True, pos["sl"], "SL"
                elif high >= pos["tp"]:
                    hit, exit_price, reason = True, pos["tp"], "TP"
            else:
                if high >= pos["sl"]:
                    hit, exit_price, reason = True, pos["sl"], "SL"
                elif low <= pos["tp"]:
                    hit, exit_price, reason = True, pos["tp"], "TP"

            if hit:
                size = pos["size"]
                if pos["dir"] == "LONG":
                    raw = (exit_price - pos["entry"]) * size
                else:
                    raw = (pos["entry"] - exit_price) * size
                fee_rate = taker_fee if reason == "SL" else maker_fee
                comm = (pos["entry"] * size + exit_price * size) * fee_rate
                net = raw - comm
                port_capital += net

                r_mul = raw / (pos["r_unit"] * size) if pos["r_unit"] > 0 and size > 0 else 0
                trades.append({"dir": pos["dir"], "pnl": net, "r_mul": r_mul})
                closed_syms.append(sym)
                continue

            # 분할 익절 체크
            if partial_tp_enabled and not pos["partial_done"] and pos["r_unit"] > 0:
                partial_price = None
                if pos["dir"] == "LONG":
                    partial_price = pos["entry"] + partial_r * pos["r_unit"]
                    if high >= partial_price:
                        pass  # hit
                    else:
                        partial_price = None
                else:
                    partial_price = pos["entry"] - partial_r * pos["r_unit"]
                    if low <= partial_price:
                        pass
                    else:
                        partial_price = None

                if partial_price is not None:
                    close_size = pos["orig_size"] * partial_fraction
                    close_size = min(close_size, pos["size"])
                    if pos["dir"] == "LONG":
                        raw = (partial_price - pos["entry"]) * close_size
                    else:
                        raw = (pos["entry"] - partial_price) * close_size
                    comm = (pos["entry"] * close_size + partial_price * close_size) * maker_fee
                    net = raw - comm
                    port_capital += net

                    r_mul = raw / (pos["r_unit"] * close_size) if pos["r_unit"] > 0 else 0
                    trades.append({"dir": pos["dir"], "pnl": net, "r_mul": r_mul})

                    pos["size"] -= close_size
                    pos["partial_done"] = True

                    # SL을 partial_r로 이동
                    if pos["dir"] == "LONG":
                        new_sl = pos["entry"] + partial_r * pos["r_unit"]
                        pos["sl"] = max(pos["sl"], new_sl)
                    else:
                        new_sl = pos["entry"] - partial_r * pos["r_unit"]
                        pos["sl"] = min(pos["sl"], new_sl)

                    # TP를 final_tp_r로 조정
                    if final_tp_r:
                        if pos["dir"] == "LONG":
                            pos["tp"] = pos["entry"] + final_tp_r * pos["r_unit"]
                        else:
                            pos["tp"] = pos["entry"] - final_tp_r * pos["r_unit"]

                    if pos["size"] <= 1e-10:
                        closed_syms.append(sym)

        for sym in closed_syms:
            positions.pop(sym, None)

        # 2. 새 시그널 확인
        if len(positions) < max_positions and ti in signals_by_ti:
            candidates = []
            for sig in signals_by_ti[ti]:
                if sig.symbol in positions:
                    continue
                if sig.volume_ratio < vol_thresh:
                    continue
                if not short_enabled and sig.direction == "SHORT":
                    continue
                candidates.append(sig)

            candidates.sort(key=lambda s: s.score, reverse=True)
            available = max_positions - len(positions)

            for sig in candidates[:available]:
                used = sum(
                    p["size"] * p["entry"] / leverage
                    for p in positions.values()
                )
                avail_capital = max(0, port_capital - used)
                trade_capital = min(avail_capital, port_capital * max_cap_pct)
                if trade_capital < 10:
                    continue

                sl_dist = sl_mult * sig.atr
                if sl_dist <= 0:
                    continue

                raw_size = (trade_capital * risk_pct) / sl_dist
                max_size = (trade_capital * leverage) / sig.close
                size = min(raw_size, max_size)
                if size <= 0:
                    continue

                r_unit = sl_dist  # 1R = SL distance

                if sig.direction == "LONG":
                    sl_price = sig.close - sl_dist
                    tp_price = sig.close + tp_mult * sig.atr
                else:
                    sl_price = sig.close + sl_dist
                    tp_price = sig.close - tp_mult * sig.atr

                positions[sig.symbol] = {
                    "dir": sig.direction,
                    "entry": sig.close,
                    "sl": sl_price,
                    "tp": tp_price,
                    "size": size,
                    "orig_size": size,
                    "r_unit": r_unit,
                    "entry_ti": ti,
                    "partial_done": False,
                    "trailing_state": "none",
                }

        equity_curve.append(port_capital)

    # 미청산 포지션 정리
    for sym in list(positions.keys()):
        pos = positions[sym]
        last_candle = candle_hlc.get((sym, n_timestamps - 1))
        if last_candle:
            close_price = last_candle.close
            if pos["dir"] == "LONG":
                raw = (close_price - pos["entry"]) * pos["size"]
            else:
                raw = (pos["entry"] - close_price) * pos["size"]
            comm = (pos["entry"] * pos["size"] + close_price * pos["size"]) * maker_fee
            net = raw - comm
            port_capital += net
            r_mul = raw / (pos["r_unit"] * pos["size"]) if pos["r_unit"] > 0 and pos["size"] > 0 else 0
            trades.append({"dir": pos["dir"], "pnl": net, "r_mul": r_mul})

    metrics = _compute_metrics(trades, equity_curve, port_capital, capital)
    metrics["params"] = params
    return metrics


def _generate_param_combos() -> list[dict]:
    """그리드 서치 파라미터 조합을 생성한다."""
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def _load_and_cache_data(start: str, end: str, cache_prefix: str) -> tuple[str, str]:
    """데이터를 로드하고 pickle로 캐시한다."""
    from scanner.data_collector import (
        fetch_all_futures_symbols, fetch_daily_volumes,
        get_daily_active_symbols, fetch_5m_data_for_symbols,
    )

    all_data_path = f"/tmp/{cache_prefix}_all_data.pkl"
    daily_active_path = f"/tmp/{cache_prefix}_daily_active.pkl"

    if os.path.exists(all_data_path) and os.path.exists(daily_active_path):
        logger.info("캐시 로드: %s", cache_prefix)
        return all_data_path, daily_active_path

    logger.info("데이터 수집 시작: %s ~ %s", start, end)
    all_symbols = fetch_all_futures_symbols()
    daily_vol = fetch_daily_volumes(all_symbols, start, end)
    daily_active = get_daily_active_symbols(daily_vol, min_volume=50_000_000)

    unique_symbols = set()
    for syms in daily_active.values():
        unique_symbols.update(syms)
    logger.info("활성 심볼: %d개", len(unique_symbols))

    all_data = fetch_5m_data_for_symbols(sorted(unique_symbols), start, end)
    logger.info("5분봉 로드 완료: %d 심볼", len(all_data))

    with open(all_data_path, "wb") as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(daily_active_path, "wb") as f:
        pickle.dump(daily_active, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("데이터 캐시 저장: %s", cache_prefix)
    return all_data_path, daily_active_path


def main() -> None:
    """메인 실행."""
    t0 = time.time()

    capital = 10000
    max_positions = 3

    # ── 1. Train 데이터 (2025년 전체) ──
    logger.info("=" * 60)
    logger.info("Phase 1: 데이터 로드 + 사전계산")
    logger.info("=" * 60)

    quarters = [
        ("2025-01-01", "2025-03-31", "train_q1"),
        ("2025-04-01", "2025-06-30", "train_q2"),
        ("2025-07-01", "2025-09-30", "train_q3"),
        ("2025-10-01", "2025-12-31", "train_q4"),
    ]

    precomputed_paths = []
    for start, end, prefix in quarters:
        all_data_path, daily_active_path = _load_and_cache_data(start, end, prefix)
        pc_path = _precompute_quarter(all_data_path, daily_active_path, prefix)
        precomputed_paths.append(pc_path)

    # ── 2. Grid Search ──
    combos = _generate_param_combos()
    n_workers = min(mp.cpu_count(), 8)

    logger.info("=" * 60)
    logger.info("Phase 2: Grid Search — %d 조합 × 4 분기 (사전계산 활용)", len(combos))
    logger.info("=" * 60)
    logger.info("병렬 실행: %d workers", n_workers)

    all_results = []

    for qi, pc_path in enumerate(precomputed_paths):
        q_start = time.time()
        logger.info("분기 %d/4 시작", qi + 1)

        worker_args = [(combo, capital, max_positions) for combo in combos]

        with mp.Pool(n_workers, initializer=_worker_init, initargs=(pc_path,)) as pool:
            results = list(pool.imap_unordered(_run_fast_backtest, worker_args))

        for r in results:
            r["quarter"] = qi
        all_results.extend(results)

        q_elapsed = time.time() - q_start
        logger.info("분기 %d/4 완료: %.1f초", qi + 1, q_elapsed)

    logger.info("Grid Search 완료: %d 결과", len(all_results))

    # 조합별 분기 결과 합산
    combo_results: dict[str, list[dict]] = {}
    for r in all_results:
        key = str(sorted(r["params"].items()))
        combo_results.setdefault(key, []).append(r)

    aggregated = []
    for key, quarter_results in combo_results.items():
        params = quarter_results[0]["params"]
        total_trades = sum(r["total_trades"] for r in quarter_results)
        if total_trades == 0:
            continue

        w_sum = total_trades
        wr = sum(r["win_rate"] * r["total_trades"] for r in quarter_results) / w_sum
        avg_r = sum(r["avg_r"] * r["total_trades"] for r in quarter_results) / w_sum
        pf = sum(r["profit_factor"] * r["total_trades"] for r in quarter_results) / w_sum

        compound_return = 1.0
        for r in sorted(quarter_results, key=lambda x: x.get("quarter", 0)):
            compound_return *= (1 + r["total_return_pct"] / 100)
        total_ret = (compound_return - 1) * 100

        worst_mdd = min(r["max_drawdown_pct"] for r in quarter_results)
        sharpe = np.mean([r["sharpe"] for r in quarter_results])
        calmar = abs(total_ret / worst_mdd) if worst_mdd != 0 else 0

        l_trades = sum(r["long_trades"] for r in quarter_results)
        s_trades = sum(r["short_trades"] for r in quarter_results)
        l_wr = sum(r["long_wr"] * r["long_trades"] for r in quarter_results) / l_trades if l_trades else 0
        l_pf = sum(r["long_pf"] * r["long_trades"] for r in quarter_results) / l_trades if l_trades else 0
        s_wr = sum(r["short_wr"] * r["short_trades"] for r in quarter_results) / s_trades if s_trades else 0
        s_pf = sum(r["short_pf"] * r["short_trades"] for r in quarter_results) / s_trades if s_trades else 0

        aggregated.append(OptResult(
            params=params,
            total_trades=total_trades,
            win_rate=wr,
            profit_factor=pf,
            avg_r=avg_r,
            total_return_pct=total_ret,
            max_drawdown_pct=worst_mdd,
            sharpe=sharpe,
            calmar=calmar,
            long_trades=l_trades,
            long_wr=l_wr,
            long_pf=l_pf,
            short_trades=s_trades,
            short_wr=s_wr,
            short_pf=s_pf,
        ))

    # fitness 계산: PF × sqrt(trades) × (1 - |MDD|/100)
    def _fitness(r):
        mdd_factor = max(0, 1 - abs(r.max_drawdown_pct) / 100)
        return r.profit_factor * np.sqrt(max(r.total_trades, 1)) * mdd_factor

    for r in aggregated:
        r.fitness = _fitness(r)

    def _print_ranking(title, sorted_list, highlight_key=None):
        print("\n" + "=" * 120)
        print(f"  {title}")
        print("=" * 120)
        print(f"  {'#':>3} {'SL':>4} {'TP':>4} {'Short':>5} {'Vol':>4} | {'거래':>5} {'승률':>6} {'PF':>6} {'AvgR':>6} {'수익률':>10} {'MDD':>7} {'Sharpe':>7} {'Fitness':>8} | {'L거래':>5} {'L승률':>6} {'LPF':>5} {'S거래':>5} {'S승률':>6} {'SPF':>5}")
        print("  " + "-" * 116)
        for i, r in enumerate(sorted_list[:20], 1):
            p = r.params
            short_str = "ON" if p.get("short_enabled", True) else "OFF"
            marker = ""
            if highlight_key and str(sorted(p.items())) == highlight_key:
                marker = " ★현재"
            print(
                f"  {i:>3} {p['sl_atr_multiplier']:>4.1f} {p['tp_atr_multiplier']:>4.1f} {short_str:>5} {p['volume_threshold']:>4.1f} | "
                f"{r.total_trades:>5} {r.win_rate:>5.1f}% {r.profit_factor:>6.2f} {r.avg_r:>+5.2f}R {r.total_return_pct:>+9.1f}% {r.max_drawdown_pct:>6.1f}% {r.sharpe:>7.2f} {r.fitness:>8.1f} | "
                f"{r.long_trades:>5} {r.long_wr:>5.1f}% {r.long_pf:>5.2f} {r.short_trades:>5} {r.short_wr:>5.1f}% {r.short_pf:>5.2f}{marker}"
            )

    # 현재 설정 key
    from config.settings import SETTINGS as CURRENT
    current_params = {
        "sl_atr_multiplier": CURRENT["sl_atr_multiplier"],
        "tp_atr_multiplier": CURRENT["tp_atr_multiplier"],
        "short_enabled": True,
        "volume_threshold": CURRENT["volume_threshold"],
    }
    current_key = str(sorted(current_params.items()))

    # 3가지 기준 정렬
    by_pf = sorted(aggregated, key=lambda x: x.profit_factor, reverse=True)
    by_return = sorted(aggregated, key=lambda x: x.total_return_pct, reverse=True)
    by_sharpe = sorted(aggregated, key=lambda x: x.sharpe, reverse=True)
    by_fitness = sorted(aggregated, key=lambda x: x.fitness, reverse=True)

    _print_ranking("TRAIN (2025 전체) — PF 기준 Top 20", by_pf, current_key)
    _print_ranking("TRAIN (2025 전체) — 수익률 기준 Top 20", by_return, current_key)
    _print_ranking("TRAIN (2025 전체) — Sharpe 기준 Top 20", by_sharpe, current_key)
    _print_ranking("TRAIN (2025 전체) — Fitness 기준 Top 20  [Fitness = PF × √trades × (1-|MDD|/100)]", by_fitness, current_key)

    # ── 4. Validation (2026년 1~3월) ──
    print("\n" + "=" * 120)
    print("  Phase 3: Validation (2026-01-01 ~ 2026-03-25) — Train 상위 조합")
    print("=" * 120)

    val_data_path, val_daily_path = _load_and_cache_data("2026-01-01", "2026-03-25", "val_2026q1")
    val_pc_path = _precompute_quarter(val_data_path, val_daily_path, "val_2026q1")

    # PF/수익률/샤프/fitness 상위 10 합집합
    seen = set()
    validation_combos = []
    for ranking in [by_pf, by_return, by_sharpe, by_fitness]:
        for r in ranking[:10]:
            key = str(sorted(r.params.items()))
            if key not in seen:
                seen.add(key)
                validation_combos.append(r.params)

    # 현재 설정도 포함
    if current_key not in seen:
        validation_combos.append(current_params)
        seen.add(current_key)

    val_args = [(combo, capital, max_positions) for combo in validation_combos]

    with mp.Pool(n_workers, initializer=_worker_init, initargs=(val_pc_path,)) as pool:
        val_results = list(pool.imap_unordered(_run_fast_backtest, val_args))

    val_map = {str(sorted(r["params"].items())): r for r in val_results}

    # train 결과 매핑
    train_map = {str(sorted(r.params.items())): r for r in aggregated}

    # val_display 구성
    val_display = []
    for combo in validation_combos:
        key = str(sorted(combo.items()))
        vr = val_map.get(key, {})
        train_r = train_map.get(key)
        # Validation fitness
        v_trades = vr.get("total_trades", 0)
        v_pf = vr.get("profit_factor", 0)
        v_mdd = vr.get("max_drawdown_pct", 0)
        v_mdd_factor = max(0, 1 - abs(v_mdd) / 100)
        v_fitness = v_pf * np.sqrt(max(v_trades, 1)) * v_mdd_factor
        vr["fitness"] = v_fitness
        val_display.append((combo, train_r, vr))

    def _print_val_ranking(title, sort_key):
        sorted_list = sorted(val_display, key=lambda x: sort_key(x[2]) if x[2] else 0, reverse=True)
        print(f"\n  {title}")
        print(f"  {'#':>3} {'SL':>4} {'TP':>4} {'Short':>5} {'Vol':>4} | {'Tr PF':>6} {'Tr Ret':>9} {'Tr Sharpe':>9} {'Tr Fit':>7} | {'V거래':>5} {'V승률':>6} {'V PF':>6} {'V Ret':>9} {'V MDD':>7} {'V Sharpe':>8} {'V Fit':>7}")
        print("  " + "-" * 125)
        for i, (combo, train_r, vr) in enumerate(sorted_list, 1):
            short_str = "ON" if combo.get("short_enabled", True) else "OFF"
            t_pf = train_r.profit_factor if train_r else 0
            t_ret = train_r.total_return_pct if train_r else 0
            t_sharpe = train_r.sharpe if train_r else 0
            t_fit = train_r.fitness if train_r else 0
            is_current = combo == current_params
            marker = " ★현재" if is_current else ""
            print(
                f"  {i:>3} {combo['sl_atr_multiplier']:>4.1f} {combo['tp_atr_multiplier']:>4.1f} {short_str:>5} {combo['volume_threshold']:>4.1f} | "
                f"{t_pf:>6.2f} {t_ret:>+8.1f}% {t_sharpe:>9.2f} {t_fit:>7.1f} | "
                f"{vr.get('total_trades', 0):>5} {vr.get('win_rate', 0):>5.1f}% {vr.get('profit_factor', 0):>6.2f} {vr.get('total_return_pct', 0):>+8.1f}% {vr.get('max_drawdown_pct', 0):>6.1f}% {vr.get('sharpe', 0):>8.2f} {vr.get('fitness', 0):>7.1f}"
                f"{marker}"
            )

    _print_val_ranking("Validation — PF 기준", lambda vr: vr.get("profit_factor", 0))
    _print_val_ranking("Validation — 수익률 기준", lambda vr: vr.get("total_return_pct", 0))
    _print_val_ranking("Validation — Sharpe 기준", lambda vr: vr.get("sharpe", 0))
    _print_val_ranking("Validation — Fitness 기준  [Fitness = PF × √trades × (1-|MDD|/100)]", lambda vr: vr.get("fitness", 0))

    elapsed = time.time() - t0
    print(f"\n  총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("=" * 100)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
