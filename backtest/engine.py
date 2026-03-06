"""백테스트 메인 엔진. 이벤트 드리븐 방식으로 5분봉을 순회한다."""

import logging

import pandas as pd
from tqdm import tqdm

from config.settings import SETTINGS
from strategy.signals import Signal, compute_indicators, generate_signals
from strategy.filters import should_filter
from strategy.position import (
    create_position,
    check_sl_tp_hit,
    check_partial_tp_hit,
    update_trailing_stop,
)
from backtest.portfolio import Portfolio

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    capital: float | None = None,
    pre_signals: pd.Series | None = None,
    partial_tp_config: list | None = None,
    final_tp_r: float | None = None,
) -> Portfolio:
    """백테스트를 실행한다.

    Args:
        df: OHLCV DataFrame (지표 컬럼 포함 가능).
        capital: 초기 자본 (기본값: settings).
        pre_signals: 사전 계산된 시그널 시리즈 (속도 최적화용).
        partial_tp_config: 분할 익절 설정 [(r_level, fraction)] 또는
            [(r_level, fraction, new_sl_r)]. None이면 단일 TP (기본 동작).
        final_tp_r: 마지막 잔여 물량의 TP를 R 단위로 지정.
            None이면 settings의 tp_atr_multiplier 사용.

    Returns:
        백테스트 결과가 담긴 Portfolio 객체.
    """
    capital = capital or SETTINGS["initial_capital"]
    portfolio = Portfolio(initial_capital=capital, capital=capital)

    # 지표 계산 (이미 있으면 스킵)
    if "atr" not in df.columns:
        logger.info("지표 계산 중...")
        df = compute_indicators(df)

    # 시그널 생성 (사전 계산된 시그널이 있으면 스킵)
    if pre_signals is not None:
        signals = pre_signals
    else:
        logger.info("시그널 생성 중...")
        signals = generate_signals(df)

    # 일일 제한 설정
    daily_max_losses = SETTINGS.get("daily_max_losses", 0)
    daily_max_trades = SETTINGS.get("daily_max_trades", 0)

    logger.info("백테스트 시작: %d봉, 초기 자본 $%.2f", len(df), capital)

    # 일일 손절/매매 횟수 추적
    current_date = None
    daily_losses = 0
    daily_trades = 0

    for i in tqdm(range(1, len(df)), desc="Backtesting", leave=False):
        row = df.iloc[i]
        timestamp = df.index[i]
        close = row["close"]
        high = row["high"]
        low = row["low"]

        # 날짜 변경 시 일일 카운터 리셋
        candle_date = timestamp.date()
        if candle_date != current_date:
            current_date = candle_date
            daily_losses = 0
            daily_trades = 0

        # 1. 기존 포지션 SL/TP 체크
        if portfolio.has_position():
            # SL 먼저 체크 (worst case — 분할익절보다 우선)
            hit, exit_price, reason = check_sl_tp_hit(
                portfolio.position, high, low
            )
            if hit:
                trade = portfolio.close_position(
                    exit_price=exit_price,
                    exit_time=str(timestamp),
                    exit_index=i,
                    exit_reason=reason,
                )
                if trade.pnl < 0:
                    daily_losses += 1
            else:
                # 분할 익절 체크 (SL 미히트 시)
                if portfolio.position.tp_levels:
                    hit_partials = check_partial_tp_hit(
                        portfolio.position, high, low
                    )
                    for pt_price, pt_fraction, pt_new_sl_r in hit_partials:
                        if portfolio.has_position():
                            portfolio.partial_close_position(
                                fraction=pt_fraction,
                                exit_price=pt_price,
                                exit_time=str(timestamp),
                                exit_index=i,
                            )
                            # 분할익절 후 SL 이동 (이익 보호)
                            if pt_new_sl_r is not None and portfolio.has_position():
                                pos = portfolio.position
                                if pos.direction == Signal.LONG:
                                    new_sl = pos.entry_price + pt_new_sl_r * pos.r_unit
                                    pos.sl_price = max(pos.sl_price, new_sl)
                                else:
                                    new_sl = pos.entry_price - pt_new_sl_r * pos.r_unit
                                    pos.sl_price = min(pos.sl_price, new_sl)
                                pos.trailing_state = "trailing"

                # 트레일링 스탑 업데이트 (포지션이 남아있으면)
                if portfolio.has_position():
                    update_trailing_stop(portfolio.position, close)

        # 2. 필터 체크
        candles_since_loss = i - portfolio.last_loss_index
        filtered = should_filter(
            timestamp=timestamp,
            atr=row["atr"],
            atr_median=row["atr_median"] if pd.notna(row.get("atr_median")) else 0,
            consecutive_losses=portfolio.consecutive_losses,
            candles_since_last_loss=candles_since_loss,
        )

        # 일일 제한 필터
        if daily_max_losses > 0 and daily_losses >= daily_max_losses:
            filtered = True
        if daily_max_trades > 0 and daily_trades >= daily_max_trades:
            filtered = True

        # 3. 시그널 처리
        signal = signals.iloc[i]

        if signal != Signal.NO_SIGNAL and not filtered:
            if portfolio.has_position():
                # 반대 시그널 처리
                if SETTINGS["reverse_on_opposite_signal"]:
                    if portfolio.position.direction != signal:
                        rev_trade = portfolio.close_position(
                            exit_price=close,
                            exit_time=str(timestamp),
                            exit_index=i,
                            exit_reason="REVERSE",
                        )
                        if rev_trade.pnl < 0:
                            daily_losses += 1
                        # 아래에서 새 포지션 진입
                    else:
                        # 같은 방향 시그널 → 무시
                        portfolio.record_equity(close)
                        continue
                else:
                    # 기존 포지션 있으면 무시
                    portfolio.record_equity(close)
                    continue

            # 새 포지션 진입
            if not portfolio.has_position() and pd.notna(row["atr"]):
                position = create_position(
                    direction=signal,
                    entry_price=close,
                    atr=row["atr"],
                    capital=portfolio.capital,
                    entry_time=str(timestamp),
                    entry_index=i,
                    partial_tp_config=partial_tp_config,
                    final_tp_r=final_tp_r,
                )
                portfolio.open_position(position)
                daily_trades += 1

        # 자산 기록
        portfolio.record_equity(close)

    # 마지막 포지션이 열려있으면 종가로 청산
    if portfolio.has_position():
        last_close = df.iloc[-1]["close"]
        portfolio.close_position(
            exit_price=last_close,
            exit_time=str(df.index[-1]),
            exit_index=len(df) - 1,
            exit_reason="END",
        )

    logger.info(
        "백테스트 완료: %d 거래, 최종 자본 $%.2f",
        len(portfolio.trades), portfolio.capital,
    )

    return portfolio
