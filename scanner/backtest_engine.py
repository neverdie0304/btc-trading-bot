"""멀티심볼 백테스트 엔진.

시간순으로 모든 심볼의 5분봉을 동기화하여 순회하며,
시그널 탐지 → 우선순위 → 리스크 관리 → 진입/청산을 시뮬레이션한다.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from config.settings import SETTINGS
from strategy.signals import Signal, compute_indicators, check_long_conditions, check_short_conditions
from strategy.filters import should_filter
from strategy.position import (
    Position, create_position, check_sl_tp_hit, check_partial_tp_hit, update_trailing_stop,
)
from scanner.config import SCANNER_SETTINGS
from scanner.prioritizer import compute_score
from scanner.signal_engine import SignalEvent

logger = logging.getLogger(__name__)


@dataclass
class MultiTrade:
    """멀티심볼 거래 기록."""
    symbol: str
    direction: Signal
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    exit_reason: str
    pnl: float = 0.0
    r_multiple: float = 0.0
    commission: float = 0.0
    hold_candles: int = 0
    score: float = 0.0


@dataclass
class MultiPortfolio:
    """멀티심볼 포트폴리오."""
    initial_capital: float = 10000.0
    capital: float = 10000.0

    # 열린 포지션: symbol -> (Position, entry_candle_idx)
    positions: dict = field(default_factory=dict)

    # 심볼별 연속 패배 / 마지막 패배 봉
    consecutive_losses: dict = field(default_factory=dict)
    last_loss_candle: dict = field(default_factory=dict)
    candle_counts: dict = field(default_factory=dict)

    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    total_commission: float = 0.0
    daily_pnl: float = 0.0
    _current_date: str = ""

    def open_position(self, symbol: str, position: Position, candle_idx: int) -> None:
        self.positions[symbol] = (position, candle_idx)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: str,
        exit_reason: str,
        candle_idx: int,
        score: float = 0.0,
    ) -> MultiTrade | None:
        if symbol not in self.positions:
            return None

        pos, entry_idx = self.positions.pop(symbol)

        # PnL
        if pos.direction == Signal.LONG:
            raw_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            raw_pnl = (pos.entry_price - pos.entry_price) * pos.size
            raw_pnl = (pos.entry_price - exit_price) * pos.size

        # 수수료
        maker_fee = SETTINGS.get("maker_fee", 0.0)
        taker_fee = SETTINGS.get("taker_fee", 0.0004)
        notional_entry = pos.entry_price * pos.size
        notional_exit = exit_price * pos.size
        entry_fee = notional_entry * maker_fee
        is_limit_exit = exit_reason in ("TP", "PARTIAL_TP")
        exit_fee = notional_exit * (maker_fee if is_limit_exit else taker_fee)
        commission = entry_fee + exit_fee
        slippage = notional_exit * SETTINGS.get("slippage_rate", 0) if not is_limit_exit else 0.0

        net_pnl = raw_pnl - commission - slippage
        r_multiple = raw_pnl / (pos.r_unit * pos.size) if pos.r_unit > 0 else 0.0

        self.capital += net_pnl
        self.daily_pnl += net_pnl
        self.total_commission += commission

        # 연속 패배
        if net_pnl < 0:
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
            self.last_loss_candle[symbol] = candle_idx
        else:
            self.consecutive_losses[symbol] = 0

        trade = MultiTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=str(exit_time),
            exit_reason=exit_reason,
            pnl=net_pnl,
            r_multiple=r_multiple,
            commission=commission,
            hold_candles=candle_idx - entry_idx,
            score=score,
        )
        self.trades.append(trade)
        return trade

    def record_equity(self) -> None:
        """현재 자산을 기록한다 (미실현 손익 제외, 간소화)."""
        self.equity_curve.append(self.capital)

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0


def _precompute_indicators(
    all_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """모든 심볼의 지표를 사전 계산한다."""
    result = {}
    for sym, df in all_data.items():
        try:
            result[sym] = compute_indicators(df)
        except Exception:
            logger.warning("%s: 지표 계산 실패, 스킵", sym)
    logger.info("지표 사전 계산 완료: %d개 심볼", len(result))
    return result


def run_multi_backtest(
    all_data: dict[str, pd.DataFrame],
    daily_active: dict[str, list[str]],
    capital: float | None = None,
    max_positions: int | None = None,
    partial_tp_config: list | None = None,
    final_tp_r: float | None = None,
) -> MultiPortfolio:
    """멀티심볼 백테스트를 실행한다.

    Args:
        all_data: {심볼: 5분봉 DataFrame} (지표 미포함 OK).
        daily_active: {날짜문자열: [활성 심볼 리스트]}.
        capital: 초기 자본.
        max_positions: 최대 동시 포지션.

    Returns:
        MultiPortfolio 결과.
    """
    capital = capital or SETTINGS["initial_capital"]
    max_pos = max_positions or SCANNER_SETTINGS["max_concurrent_positions"]
    max_cap_pct = SCANNER_SETTINGS["max_capital_per_position_pct"]

    portfolio = MultiPortfolio(initial_capital=capital, capital=capital)

    # 지표 사전 계산
    logger.info("지표 사전 계산 중...")
    indicator_data = _precompute_indicators(all_data)

    # 전체 타임스텝 구축 (모든 심볼의 유니온)
    all_timestamps = set()
    for df in indicator_data.values():
        all_timestamps.update(df.index.tolist())
    timestamps = sorted(all_timestamps)
    logger.info("전체 타임스텝: %d개 (%s ~ %s)", len(timestamps), timestamps[0], timestamps[-1])

    # 심볼별 인덱스 맵 (빠른 lookup)
    sym_index_map: dict[str, dict] = {}
    for sym, df in indicator_data.items():
        sym_index_map[sym] = {ts: i for i, ts in enumerate(df.index)}

    # 일별 활성 심볼 → 날짜 lookup 최적화
    current_date_str = ""
    active_symbols: list[str] = []

    logger.info(
        "백테스트 시작: %d 타임스텝, 초기 자본 $%.2f, 최대 %d 포지션",
        len(timestamps), capital, max_pos,
    )

    for ti, timestamp in enumerate(tqdm(timestamps, desc="Multi-Backtest", leave=False)):
        date_str = str(timestamp.date()) if hasattr(timestamp, 'date') else str(timestamp)[:10]

        # 날짜 변경 시 활성 심볼 갱신
        if date_str != current_date_str:
            current_date_str = date_str
            active_symbols = daily_active.get(date_str, [])
            # 포지션 있는 심볼은 항상 포함
            active_set = set(active_symbols) | set(portfolio.positions.keys())
            active_symbols = [s for s in active_set if s in indicator_data]
            portfolio.reset_daily()

        # ── 1. 기존 포지션 SL/TP 체크 ──
        for sym in list(portfolio.positions.keys()):
            if sym not in indicator_data:
                continue

            df = indicator_data[sym]
            idx_map = sym_index_map[sym]
            if timestamp not in idx_map:
                continue

            row_idx = idx_map[timestamp]
            if row_idx < 1:
                continue

            row = df.iloc[row_idx]
            pos, entry_idx = portfolio.positions[sym]

            # SL/TP 체크
            hit, exit_price, reason = check_sl_tp_hit(pos, row["high"], row["low"])
            if hit:
                portfolio.close_position(sym, exit_price, str(timestamp), reason, ti)
                continue

            # 분할 익절
            if pos.tp_levels:
                hit_partials = check_partial_tp_hit(pos, row["high"], row["low"])
                for pt_price, pt_fraction, pt_new_sl_r in hit_partials:
                    if sym in portfolio.positions:
                        # 간소화: 분할익절 PnL 반영
                        close_size = pos.original_size * pt_fraction
                        close_size = min(close_size, pos.size)
                        if pos.direction == Signal.LONG:
                            raw = (pt_price - pos.entry_price) * close_size
                        else:
                            raw = (pos.entry_price - pt_price) * close_size
                        maker_fee = SETTINGS.get("maker_fee", 0.0)
                        comm = (pos.entry_price * close_size + pt_price * close_size) * maker_fee
                        net = raw - comm
                        portfolio.capital += net
                        portfolio.total_commission += comm
                        portfolio.daily_pnl += net
                        pos.size -= close_size

                        portfolio.trades.append(MultiTrade(
                            symbol=sym, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=pt_price,
                            size=close_size, entry_time=pos.entry_time,
                            exit_time=str(timestamp), exit_reason="PARTIAL_TP",
                            pnl=net, r_multiple=raw / (pos.r_unit * close_size) if pos.r_unit > 0 else 0,
                            commission=comm, hold_candles=ti - entry_idx,
                        ))

                        # SL 이동
                        if pt_new_sl_r is not None and pos.size > 1e-10:
                            if pos.direction == Signal.LONG:
                                new_sl = pos.entry_price + pt_new_sl_r * pos.r_unit
                                pos.sl_price = max(pos.sl_price, new_sl)
                            else:
                                new_sl = pos.entry_price - pt_new_sl_r * pos.r_unit
                                pos.sl_price = min(pos.sl_price, new_sl)
                            pos.trailing_state = "trailing"

                        if pos.size <= 1e-10:
                            portfolio.positions.pop(sym, None)

            # 트레일링 스탑
            if sym in portfolio.positions:
                pos, _ = portfolio.positions[sym]
                update_trailing_stop(pos, row["close"])

        # ── 2. 시그널 체크 (활성 심볼만) ──
        signals: list[SignalEvent] = []

        for sym in active_symbols:
            if sym in portfolio.positions:
                continue  # 이미 보유
            if len(portfolio.positions) >= max_pos:
                break  # 슬롯 없음

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

            # NaN 체크
            if pd.isna(row.get("atr")) or pd.isna(row.get("rsi")) or pd.isna(row.get("volume_ratio")):
                continue

            atr = row["atr"]
            atr_med = row["atr_median"] if not pd.isna(row.get("atr_median")) else atr

            # 캔들 카운트 업데이트
            portfolio.candle_counts[sym] = portfolio.candle_counts.get(sym, 0) + 1

            # 필터
            consec = portfolio.consecutive_losses.get(sym, 0)
            last_loss = portfolio.last_loss_candle.get(sym, -999)
            candles_since = ti - last_loss

            if should_filter(timestamp, atr, atr_med, consec, candles_since):
                continue

            # ATR/Price 변동성 필터
            min_atr_pct = SCANNER_SETTINGS.get("min_atr_pct")
            if min_atr_pct and row["close"] > 0:
                atr_filter_mode = SCANNER_SETTINGS.get("atr_filter_mode", "current")
                if atr_filter_mode == "rolling14" and "atr_pct_ma14" in df.columns:
                    atr_pct = row["atr_pct_ma14"] if not pd.isna(row.get("atr_pct_ma14")) else 0
                else:
                    atr_pct = atr / row["close"] * 100
                if atr_pct < min_atr_pct:
                    continue

            # 시그널
            direction = Signal.NO_SIGNAL
            if check_long_conditions(row, prev):
                direction = Signal.LONG
            elif check_short_conditions(row, prev):
                direction = Signal.SHORT

            if direction == Signal.NO_SIGNAL:
                continue

            event = SignalEvent(
                symbol=sym,
                direction=direction,
                timestamp=timestamp,
                close=row["close"],
                atr=atr,
                atr_median=atr_med,
                rsi=row["rsi"],
                volume_ratio=row["volume_ratio"],
            )
            event.score = compute_score(event)
            signals.append(event)

        # ── 3. 우선순위 결정 & 진입 ──
        if signals:
            signals.sort(key=lambda e: e.score, reverse=True)
            available = max_pos - len(portfolio.positions)

            for event in signals[:available]:
                # 자본 배분
                used = sum(
                    p.size * p.entry_price / SETTINGS.get("leverage", 20)
                    for p, _ in portfolio.positions.values()
                )
                avail_capital = max(0, portfolio.capital - used)
                trade_capital = min(avail_capital, portfolio.capital * max_cap_pct)

                if trade_capital < 10:
                    continue

                position = create_position(
                    direction=event.direction,
                    entry_price=event.close,
                    atr=event.atr,
                    capital=trade_capital,
                    entry_time=str(event.timestamp),
                    entry_index=ti,
                    partial_tp_config=partial_tp_config,
                    final_tp_r=final_tp_r,
                )

                portfolio.open_position(event.symbol, position, ti)

                logger.debug(
                    "[ENTRY] %s %s @ %.4f | Score=%.3f | ATR=%.4f | RSI=%.1f",
                    event.symbol, event.direction.value, event.close,
                    event.score, event.atr, event.rsi,
                )

        portfolio.record_equity()

    # ── 마지막 열린 포지션 청산 ──
    for sym in list(portfolio.positions.keys()):
        df = indicator_data.get(sym)
        if df is not None and len(df) > 0:
            last_close = df.iloc[-1]["close"]
            portfolio.close_position(sym, last_close, str(timestamps[-1]), "END", len(timestamps) - 1)

    logger.info(
        "백테스트 완료: %d 거래, 최종 자본 $%.2f (%.1f%%)",
        len(portfolio.trades), portfolio.capital,
        (portfolio.capital / portfolio.initial_capital - 1) * 100,
    )

    return portfolio
