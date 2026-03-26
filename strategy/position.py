"""포지션 사이징, SL/TP 계산 모듈."""

import logging
from dataclasses import dataclass

from config.settings import SETTINGS
from strategy.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """개별 포지션 정보."""
    direction: Signal          # LONG 또는 SHORT
    entry_price: float
    size: float                # 포지션 크기 (BTC 수량)
    sl_price: float            # 손절가
    tp_price: float            # 익절가
    initial_sl: float          # 최초 손절가 (R 계산용)
    entry_time: str = ""       # 진입 시각
    entry_index: int = 0       # 진입 봉 인덱스
    r_unit: float = 0.0        # 1R = |entry - initial_sl|

    def __post_init__(self) -> None:
        self.r_unit = abs(self.entry_price - self.initial_sl)


def calculate_position_size(
    capital: float,
    entry_price: float,
    atr: float,
) -> tuple[float, float]:
    """포지션 크기와 SL 거리를 계산한다.

    Args:
        capital: 현재 가용 자본.
        entry_price: 진입 예정 가격.
        atr: 현재 ATR.

    Returns:
        (position_size, sl_distance) 튜플.
    """
    sl_distance = SETTINGS["sl_atr_multiplier"] * atr
    risk_amount = capital * SETTINGS["risk_per_trade"]
    position_size = risk_amount / sl_distance

    leverage = SETTINGS.get("leverage", 20)
    max_size = (capital * leverage) / entry_price
    if position_size > max_size:
        position_size = max_size

    return position_size, sl_distance


def create_position(
    direction: Signal,
    entry_price: float,
    atr: float,
    capital: float,
    entry_time: str = "",
    entry_index: int = 0,
) -> Position:
    """새로운 포지션을 생성한다.

    Args:
        direction: LONG 또는 SHORT.
        entry_price: 진입 가격.
        atr: 현재 ATR.
        capital: 현재 자본.
        entry_time: 진입 시각 문자열.
        entry_index: 진입 봉 인덱스.

    Returns:
        생성된 Position 객체.
    """
    size, sl_distance = calculate_position_size(capital, entry_price, atr)
    tp_distance = SETTINGS["tp_atr_multiplier"] * atr

    if direction == Signal.LONG:
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    position = Position(
        direction=direction,
        entry_price=entry_price,
        size=size,
        sl_price=sl_price,
        tp_price=tp_price,
        initial_sl=sl_price,
        entry_time=entry_time,
        entry_index=entry_index,
    )

    logger.debug(
        "Position created: %s @ %.2f, size=%.6f, SL=%.2f, TP=%.2f",
        direction.value, entry_price, size, sl_price, tp_price,
    )

    return position


def check_sl_tp_hit(
    position: Position,
    high: float,
    low: float,
) -> tuple[bool, float, str]:
    """현재 봉의 H/L로 SL/TP 히트 여부를 확인한다.

    SL을 먼저 체크한다 (worst case 시나리오).

    Args:
        position: 현재 포지션.
        high: 현재 봉 고가.
        low: 현재 봉 저가.

    Returns:
        (hit, exit_price, reason) 튜플. hit이 False이면 미히트.
    """
    if position.direction == Signal.LONG:
        if low <= position.sl_price:
            return True, position.sl_price, "SL"
        if high >= position.tp_price:
            return True, position.tp_price, "TP"
    else:  # SHORT
        if high >= position.sl_price:
            return True, position.sl_price, "SL"
        if low <= position.tp_price:
            return True, position.tp_price, "TP"

    return False, 0.0, ""
