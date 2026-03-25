"""포지션 사이징, SL/TP 계산, 트레일링 스탑 로직 모듈."""

import logging
from dataclasses import dataclass, field

from config.settings import SETTINGS
from strategy.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """개별 포지션 정보."""
    direction: Signal          # LONG 또는 SHORT
    entry_price: float
    size: float                # 포지션 크기 (BTC 수량)
    sl_price: float            # 현재 손절가
    tp_price: float            # 익절가
    initial_sl: float          # 최초 손절가 (R 계산용)
    entry_time: str = ""       # 진입 시각
    entry_index: int = 0       # 진입 봉 인덱스
    r_unit: float = 0.0        # 1R = |entry - initial_sl|
    trailing_state: str = "initial"  # initial, break_even, trailing
    original_size: float = 0.0       # 최초 포지션 크기 (분할익절 비율 기준)
    tp_levels: list = field(default_factory=list)  # [(r_level, fraction), ...] 미소진 분할익절 레벨

    def __post_init__(self) -> None:
        self.r_unit = abs(self.entry_price - self.initial_sl)
        if self.original_size == 0.0:
            self.original_size = self.size


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

    # 레버리지 상한 적용: 저변동성 구간에서 과도한 노셔널(→ 수수료 급증)을 방지한다.
    # SL 히트 시 리스크는 항상 risk_per_trade이지만, 포지션이 커질수록
    # 노셔널 비례 수수료가 증가해 실질 성과가 낮아진다.
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
    partial_tp_config: list | None = None,
    final_tp_r: float | None = None,
) -> Position:
    """새로운 포지션을 생성한다.

    Args:
        direction: LONG 또는 SHORT.
        entry_price: 진입 가격.
        atr: 현재 ATR.
        capital: 현재 자본.
        entry_time: 진입 시각 문자열.
        entry_index: 진입 봉 인덱스.
        partial_tp_config: 분할 익절 설정 [(r_level, fraction)] 또는
            [(r_level, fraction, new_sl_r)].
            fraction 합계가 1.0이면 전량 분할익절, None이면 단일 TP.
        final_tp_r: 마지막 남은 물량의 TP를 R 단위로 지정.
            None이면 settings의 tp_atr_multiplier 사용.

    Returns:
        생성된 Position 객체.
    """
    size, sl_distance = calculate_position_size(capital, entry_price, atr)

    # 최종 TP 거리: final_tp_r이 지정되면 R 단위 기준, 아니면 ATR 배수
    if final_tp_r is not None:
        tp_distance = final_tp_r * sl_distance  # final_tp_r * r_unit
    else:
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
        tp_levels=list(partial_tp_config) if partial_tp_config else [],
    )

    logger.debug(
        "Position created: %s @ %.2f, size=%.6f, SL=%.2f, TP=%.2f, partialTP=%s",
        direction.value, entry_price, size, sl_price, tp_price,
        f"{len(partial_tp_config)} levels" if partial_tp_config else "none",
    )

    return position


def check_sl_tp_hit(
    position: Position,
    high: float,
    low: float,
) -> tuple[bool, float, str]:
    """현재 봉의 H/L로 SL/TP 히트 여부를 확인한다.

    SL을 먼저 체크한다 (worst case 시나리오).
    분할익절(tp_levels) 사용 중에는 TP 체크를 생략한다.

    Args:
        position: 현재 포지션.
        high: 현재 봉 고가.
        low: 현재 봉 저가.

    Returns:
        (hit, exit_price, reason) 튜플. hit이 False이면 미히트.
    """
    # 트레일링 발동된 SL은 TRAILING_SL로 구분
    sl_reason = "TRAILING_SL" if position.trailing_state != "initial" else "SL"

    if position.direction == Signal.LONG:
        # SL 체크 먼저 (저가가 SL 이하)
        if low <= position.sl_price:
            return True, position.sl_price, sl_reason
        # TP 체크: 분할익절 사용 중이면 생략 (마지막 레벨이 전량 청산)
        if not position.tp_levels and high >= position.tp_price:
            return True, position.tp_price, "TP"
    else:  # SHORT
        # SL 체크 먼저 (고가가 SL 이상)
        if high >= position.sl_price:
            return True, position.sl_price, sl_reason
        # TP 체크: 분할익절 사용 중이면 생략
        if not position.tp_levels and low <= position.tp_price:
            return True, position.tp_price, "TP"

    return False, 0.0, ""


def check_partial_tp_hit(
    position: Position,
    high: float,
    low: float,
) -> list[tuple[float, float, float | None]]:
    """분할 익절 레벨 히트 여부를 확인하고, 히트된 레벨을 제거한다.

    tp_levels 원소 형식:
        (r_level, fraction)              — SL 이동 없음
        (r_level, fraction, new_sl_r)   — 히트 후 SL을 +new_sl_r R로 이동

    Args:
        position: 현재 포지션.
        high: 현재 봉 고가.
        low: 현재 봉 저가.

    Returns:
        [(exit_price, fraction, new_sl_r_or_none), ...] 히트된 레벨 목록.
    """
    if not position.tp_levels:
        return []

    hit_results: list[tuple[float, float, float | None]] = []
    remaining_levels = []

    for level_data in position.tp_levels:
        r_level = level_data[0]
        fraction = level_data[1]
        new_sl_r: float | None = level_data[2] if len(level_data) > 2 else None

        if position.direction == Signal.LONG:
            target = position.entry_price + r_level * position.r_unit
            if high >= target:
                hit_results.append((target, fraction, new_sl_r))
            else:
                remaining_levels.append(level_data)
        else:  # SHORT
            target = position.entry_price - r_level * position.r_unit
            if low <= target:
                hit_results.append((target, fraction, new_sl_r))
            else:
                remaining_levels.append(level_data)

    position.tp_levels = remaining_levels
    return hit_results


def update_trailing_stop(position: Position, close: float) -> None:
    """트레일링 스탑을 업데이트한다 (in-place).

    - 미실현 수익 1.0R 도달 → SL을 진입가(Break-Even)로 이동
    - 미실현 수익 1.5R 도달 → SL을 +0.5R로 이동

    Args:
        position: 현재 포지션.
        close: 현재 봉 종가.
    """
    if position.r_unit == 0:
        return

    if position.direction == Signal.LONG:
        unrealized_r = (close - position.entry_price) / position.r_unit
    else:
        unrealized_r = (position.entry_price - close) / position.r_unit

    be_threshold = SETTINGS["trailing_be_threshold"]
    step_threshold = SETTINGS["trailing_step_threshold"]

    if unrealized_r >= step_threshold and position.trailing_state != "trailing":
        # 1.5R 도달: SL을 +0.5R로 이동
        if position.direction == Signal.LONG:
            new_sl = position.entry_price + 0.5 * position.r_unit
        else:
            new_sl = position.entry_price - 0.5 * position.r_unit
        position.sl_price = new_sl
        position.trailing_state = "trailing"
        logger.debug("Trailing stop updated: +0.5R, SL=%.2f", new_sl)

    elif unrealized_r >= be_threshold and position.trailing_state == "initial":
        # 1.0R 도달: SL을 진입가로 이동 (Break-Even)
        position.sl_price = position.entry_price
        position.trailing_state = "break_even"
        logger.debug("Break-Even move: SL=%.2f", position.entry_price)
