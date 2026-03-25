"""동시 시그널 우선순위 결정 모듈.

여러 심볼에서 동시에 시그널이 발생할 때 점수 기반으로 랭킹한다.
"""

import logging
from strategy.signals import Signal
from scanner.signal_engine import SignalEvent

logger = logging.getLogger(__name__)


def compute_score(event: SignalEvent) -> float:
    """시그널 이벤트의 우선순위 점수를 계산한다.

    점수 = ATR 강도(0.4) + 거래량 비율(0.3) + RSI 여유도(0.3)

    Args:
        event: 시그널 이벤트.

    Returns:
        0.0 ~ 1.0 범위의 점수.
    """
    # ATR 강도: ATR / ATR_median (1.0 = 평균, 높을수록 좋음)
    atr_strength = 0.0
    if event.atr_median > 0:
        ratio = event.atr / event.atr_median
        atr_strength = min(ratio / 3.0, 1.0)  # 3배 이상이면 1.0

    # 거래량 비율: volume_ratio (1.0 = 평균)
    vol_score = min(event.volume_ratio / 3.0, 1.0)

    # RSI 여유도: 목표까지 남은 거리가 클수록 좋음
    if event.direction == Signal.LONG:
        # RSI가 낮을수록 올라갈 여유 많음 (40~65 범위)
        rsi_room = max(0, (70 - event.rsi)) / 30
    else:
        # RSI가 높을수록 내려갈 여유 많음 (35~60 범위)
        rsi_room = max(0, (event.rsi - 30)) / 30

    rsi_score = min(rsi_room, 1.0)

    score = (atr_strength * 0.4) + (vol_score * 0.3) + (rsi_score * 0.3)
    return round(score, 4)


def rank_signals(
    events: list[SignalEvent],
    max_slots: int,
    existing_symbols: set[str] | None = None,
) -> list[SignalEvent]:
    """시그널을 점수 기반으로 랭킹하고 상위 N개를 반환한다.

    Args:
        events: 이번 봉에서 탐지된 시그널 리스트.
        max_slots: 진입 가능한 최대 슬롯 수.
        existing_symbols: 이미 포지션이 있는 심볼 집합 (중복 방지).

    Returns:
        상위 max_slots개의 시그널 (점수 내림차순).
    """
    if not events:
        return []

    existing = existing_symbols or set()

    # 이미 포지션이 있는 심볼 제외
    filtered = [e for e in events if e.symbol not in existing]

    if not filtered:
        return []

    # 점수 계산
    for event in filtered:
        event.score = compute_score(event)

    # 점수 내림차순 정렬
    filtered.sort(key=lambda e: e.score, reverse=True)

    result = filtered[:max_slots]

    if len(events) > 0:
        logger.info(
            "[RANK] 시그널 %d개 중 %d개 선택 (포지션 중복 %d개 제외)",
            len(events), len(result), len(events) - len(filtered),
        )
        for i, e in enumerate(result):
            logger.info(
                "  #%d %s %s — Score: %.3f (ATR=%.2f, Vol=%.1fx, RSI=%.1f)",
                i + 1, e.symbol, e.direction.value,
                e.score, e.atr, e.volume_ratio, e.rsi,
            )

    return result
