"""가상 포트폴리오 관리 모듈. 잔고, 포지션, PnL을 추적한다."""

import logging
from dataclasses import dataclass, field

from config.settings import SETTINGS
from strategy.signals import Signal
from strategy.position import Position

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """완료된 거래 기록."""
    direction: Signal
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    entry_index: int
    exit_index: int
    exit_reason: str       # "SL", "TP", "REVERSE"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    hold_candles: int = 0


@dataclass
class Portfolio:
    """가상 포트폴리오."""
    initial_capital: float = 0.0
    capital: float = 0.0
    position: Position | None = None
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    consecutive_losses: int = 0
    last_loss_index: int = -999
    total_commission: float = 0.0
    total_slippage: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_capital == 0:
            self.initial_capital = SETTINGS["initial_capital"]
        if self.capital == 0:
            self.capital = self.initial_capital

    def has_position(self) -> bool:
        """현재 포지션이 있는지 확인한다."""
        return self.position is not None

    def open_position(self, position: Position) -> None:
        """새 포지션을 연다."""
        self.position = position
        logger.debug(
            "포지션 오픈: %s @ %.2f, size=%.6f",
            position.direction.value, position.entry_price, position.size,
        )

    def close_position(
        self,
        exit_price: float,
        exit_time: str,
        exit_index: int,
        exit_reason: str,
    ) -> Trade:
        """현재 포지션을 닫고 거래 기록을 생성한다.

        Args:
            exit_price: 청산 가격.
            exit_time: 청산 시각.
            exit_index: 청산 봉 인덱스.
            exit_reason: 청산 사유.

        Returns:
            완료된 Trade 객체.
        """
        pos = self.position
        if pos is None:
            raise ValueError("닫을 포지션이 없습니다.")

        # 수수료 및 슬리피지 계산 (maker/taker 분리)
        notional_entry = pos.entry_price * pos.size
        notional_exit = exit_price * pos.size
        maker_fee = SETTINGS.get("maker_fee", 0.0)
        taker_fee = SETTINGS.get("taker_fee", 0.0005)
        # 진입: limit (maker), 청산: TP/TRAILING_SL→limit (maker), SL→stop-market (taker)
        entry_fee = notional_entry * maker_fee
        is_limit_exit = exit_reason in ("TP", "TRAILING_SL")
        exit_fee = notional_exit * (maker_fee if is_limit_exit else taker_fee)
        commission = entry_fee + exit_fee
        slippage = notional_exit * SETTINGS["slippage_rate"] if not is_limit_exit else 0.0

        # PnL 계산
        if pos.direction == Signal.LONG:
            raw_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.size

        net_pnl = raw_pnl - commission - slippage

        # R-multiple
        r_multiple = 0.0
        if pos.r_unit > 0:
            r_multiple = raw_pnl / (pos.r_unit * pos.size)

        # PnL %
        pnl_pct = net_pnl / self.capital * 100

        trade = Trade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_index=pos.entry_index,
            exit_index=exit_index,
            exit_reason=exit_reason,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            commission=commission,
            slippage=slippage,
            hold_candles=exit_index - pos.entry_index,
        )

        # 자본 업데이트
        self.capital += net_pnl
        self.total_commission += commission
        self.total_slippage += slippage
        self.trades.append(trade)

        # 연속 패배 추적
        if net_pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_index = exit_index
        else:
            self.consecutive_losses = 0

        self.position = None

        logger.debug(
            "포지션 종료: %s @ %.2f → %.2f, PnL=%.2f (%.2f%%), R=%.2fR, reason=%s",
            trade.direction.value, trade.entry_price, exit_price,
            net_pnl, pnl_pct, r_multiple, exit_reason,
        )

        return trade

    def partial_close_position(
        self,
        fraction: float,
        exit_price: float,
        exit_time: str,
        exit_index: int,
    ) -> Trade:
        """포지션의 일부를 분할 익절로 청산한다.

        Args:
            fraction: original_size 대비 청산 비율 (0~1).
            exit_price: 청산 가격.
            exit_time: 청산 시각.
            exit_index: 청산 봉 인덱스.

        Returns:
            부분 청산 Trade 객체.
        """
        pos = self.position
        if pos is None:
            raise ValueError("청산할 포지션이 없습니다.")

        # 청산 크기 (original_size 기준 비율, 남은 크기 초과 방지)
        close_size = pos.original_size * fraction
        close_size = min(close_size, pos.size)

        # 분할익절 → limit order → maker fee 0%
        maker_fee = SETTINGS.get("maker_fee", 0.0)
        notional_entry = pos.entry_price * close_size
        notional_exit = exit_price * close_size
        commission = (notional_entry + notional_exit) * maker_fee  # 0%

        # PnL
        if pos.direction == Signal.LONG:
            raw_pnl = (exit_price - pos.entry_price) * close_size
        else:
            raw_pnl = (pos.entry_price - exit_price) * close_size
        net_pnl = raw_pnl - commission

        r_multiple = raw_pnl / (pos.r_unit * close_size) if pos.r_unit > 0 else 0.0
        pnl_pct = net_pnl / self.capital * 100 if self.capital > 0 else 0.0

        trade = Trade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=close_size,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_index=pos.entry_index,
            exit_index=exit_index,
            exit_reason="PARTIAL_TP",
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            commission=commission,
            slippage=0.0,
            hold_candles=exit_index - pos.entry_index,
        )

        # 자본 및 포지션 크기 업데이트
        self.capital += net_pnl
        self.total_commission += commission
        self.trades.append(trade)

        pos.size -= close_size

        # 포지션 완전 청산 시 (마지막 분할익절)
        if pos.size <= 1e-10:
            self.consecutive_losses = 0
            self.position = None

        logger.debug(
            "분할 익절: %s size=%.6f @ %.2f, PnL=%.2f, R=%.2fR",
            pos.direction.value, close_size, exit_price, net_pnl, r_multiple,
        )

        return trade

    def record_equity(self, current_price: float) -> None:
        """현재 자산 가치를 기록한다.

        Args:
            current_price: 현재 종가.
        """
        equity = self.capital
        if self.position is not None:
            pos = self.position
            if pos.direction == Signal.LONG:
                unrealized = (current_price - pos.entry_price) * pos.size
            else:
                unrealized = (pos.entry_price - current_price) * pos.size
            equity += unrealized
        self.equity_curve.append(equity)
