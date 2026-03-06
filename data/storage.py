"""캔들 데이터의 Parquet 저장/로드를 담당하는 모듈."""

import logging
from pathlib import Path

import pandas as pd

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


def _get_cache_path(symbol: str, interval: str) -> Path:
    """캐시 파일 경로를 반환한다."""
    cache_dir = Path(SETTINGS["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{interval}.parquet"


def save_to_parquet(
    df: pd.DataFrame,
    symbol: str | None = None,
    interval: str | None = None,
) -> Path:
    """DataFrame을 Parquet 파일로 저장한다.

    기존 캐시가 있으면 병합하여 중복을 제거한다.

    Args:
        df: 저장할 OHLCV DataFrame.
        symbol: 거래 페어.
        interval: 캔들 간격.

    Returns:
        저장된 파일 경로.
    """
    symbol = symbol or SETTINGS["symbol"]
    interval = interval or SETTINGS["interval"]
    path = _get_cache_path(symbol, interval)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        logger.info("기존 캐시와 병합: %d개 캔들", len(df))

    df.to_parquet(path, engine="pyarrow")
    logger.info("Parquet 저장 완료: %s (%d개 캔들)", path, len(df))
    return path


def load_from_parquet(
    symbol: str | None = None,
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame | None:
    """Parquet 캐시에서 데이터를 로드한다.

    Args:
        symbol: 거래 페어.
        interval: 캔들 간격.
        start: 시작 날짜 문자열 (필터링용).
        end: 종료 날짜 문자열 (필터링용).

    Returns:
        OHLCV DataFrame 또는 캐시가 없으면 None.
    """
    symbol = symbol or SETTINGS["symbol"]
    interval = interval or SETTINGS["interval"]
    path = _get_cache_path(symbol, interval)

    if not path.exists():
        logger.info("캐시 파일 없음: %s", path)
        return None

    df = pd.read_parquet(path)
    logger.info("Parquet 로드: %s (%d개 캔들)", path, len(df))

    # 날짜 필터링
    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df.index >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = df[df.index < end_ts]

    if df.empty:
        logger.warning("필터링 후 데이터가 없습니다.")
        return None

    logger.info("필터링 후 %d개 캔들: %s ~ %s", len(df), df.index[0], df.index[-1])
    return df


def get_missing_ranges(
    symbol: str | None = None,
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[tuple[str, str]]:
    """캐시에서 누락된 기간을 계산한다.

    Args:
        symbol: 거래 페어.
        interval: 캔들 간격.
        start: 요청 시작 날짜.
        end: 요청 종료 날짜.

    Returns:
        누락 구간 리스트 [(start, end), ...]. 전체가 누락이면 [(start, end)].
    """
    start = start or SETTINGS["backtest_start"]
    end = end or SETTINGS["backtest_end"]

    cached = load_from_parquet(symbol, interval, start, end)
    if cached is None or cached.empty:
        return [(start, end)]

    cached_start = cached.index[0].strftime("%Y-%m-%d")
    cached_end = cached.index[-1].strftime("%Y-%m-%d")

    missing = []
    if start < cached_start:
        missing.append((start, cached_start))
    if end > cached_end:
        missing.append((cached_end, end))

    return missing
