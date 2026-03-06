"""매매 금지 구간 필터 모듈."""

import logging

import pandas as pd

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


def is_weekend_filter(timestamp: pd.Timestamp) -> bool:
    """주말 유동성 부족 구간인지 확인한다.

    토요일 00:00 ~ 일요일 06:00 UTC.

    Args:
        timestamp: UTC 시간.

    Returns:
        매매 금지이면 True.
    """
    if not SETTINGS["weekend_filter"]:
        return False

    weekday = timestamp.weekday()  # 0=월 ~ 6=일

    # 토요일 전체
    if weekday == 5:
        return True
    # 일요일 06:00 UTC 이전
    if weekday == 6 and timestamp.hour < 6:
        return True

    return False


def is_news_blackout(timestamp: pd.Timestamp) -> bool:
    """뉴스 발표 전후 블랙아웃 구간인지 확인한다.

    config에 등록된 뉴스 이벤트 전후 N분.

    Args:
        timestamp: UTC 시간.

    Returns:
        매매 금지이면 True.
    """
    blackout_minutes = SETTINGS["news_blackout_minutes"]
    news_events = SETTINGS.get("news_events", [])

    for event_str in news_events:
        event_time = pd.Timestamp(event_str, tz="UTC")
        delta = abs((timestamp - event_time).total_seconds()) / 60
        if delta <= blackout_minutes:
            return True

    return False


def is_deadzone(atr: float, atr_median: float) -> bool:
    """변동성 데드존인지 확인한다.

    ATR < median(ATR) × deadzone_atr_ratio 이면 데드존.

    Args:
        atr: 현재 ATR 값.
        atr_median: ATR 중앙값.

    Returns:
        데드존이면 True.
    """
    if pd.isna(atr) or pd.isna(atr_median):
        return False
    return atr < atr_median * SETTINGS["deadzone_atr_ratio"]


def is_cooldown(
    consecutive_losses: int,
    candles_since_last_loss: int,
) -> bool:
    """연속 패배 후 쿨다운 구간인지 확인한다.

    Args:
        consecutive_losses: 현재 연속 패배 횟수.
        candles_since_last_loss: 마지막 패배 이후 경과 봉 수.

    Returns:
        쿨다운 중이면 True.
    """
    threshold = SETTINGS["cooldown_after_consecutive_losses"]
    cooldown_candles = SETTINGS["cooldown_candles"]

    if consecutive_losses >= threshold and candles_since_last_loss < cooldown_candles:
        return True

    return False


def is_bad_hour(timestamp: pd.Timestamp) -> bool:
    """손실 시간대인지 확인한다.

    config의 bad_hours_utc에 등록된 시간(UTC)이면 매매 금지.

    Args:
        timestamp: UTC 시간.

    Returns:
        매매 금지이면 True.
    """
    bad_hours = SETTINGS.get("bad_hours_utc", [])
    if not bad_hours:
        return False
    return timestamp.hour in bad_hours


def should_filter(
    timestamp: pd.Timestamp,
    atr: float,
    atr_median: float,
    consecutive_losses: int = 0,
    candles_since_last_loss: int = 999,
) -> bool:
    """모든 필터를 종합하여 매매 금지 여부를 판단한다.

    Args:
        timestamp: 현재 봉의 UTC 시간.
        atr: 현재 ATR.
        atr_median: ATR 중앙값.
        consecutive_losses: 연속 패배 횟수.
        candles_since_last_loss: 마지막 패배 이후 봉 수.

    Returns:
        매매 금지이면 True.
    """
    if is_weekend_filter(timestamp):
        return True
    if is_bad_hour(timestamp):
        return True
    if is_news_blackout(timestamp):
        return True
    if is_deadzone(atr, atr_median):
        return True
    if is_cooldown(consecutive_losses, candles_since_last_loss):
        return True

    return False
