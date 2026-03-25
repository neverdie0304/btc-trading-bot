"""Binance REST API를 통해 과거 캔들(Kline) 데이터를 수집하는 모듈."""

import logging
import time
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client

from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, SETTINGS

logger = logging.getLogger(__name__)

# Binance kline 컬럼 정의
KLINE_COLUMNS: list[str] = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]

NUMERIC_COLUMNS: list[str] = [
    "open", "high", "low", "close", "volume", "quote_volume",
]


def _create_client() -> Client:
    """Binance API 클라이언트를 생성한다."""
    return Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def fetch_klines(
    symbol: str | None = None,
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
    client: Client | None = None,
) -> pd.DataFrame:
    """Binance에서 과거 캔들 데이터를 수집한다.

    Args:
        symbol: 거래 페어 (기본값: settings의 symbol).
        interval: 캔들 간격 (기본값: settings의 interval).
        start: 시작 날짜 문자열, 예 '2024-01-01'.
        end: 종료 날짜 문자열, 예 '2024-12-31'.
        client: 기존 Binance Client 인스턴스 (없으면 새로 생성).

    Returns:
        OHLCV DataFrame (UTC 시간 인덱스).
    """
    symbol = symbol or SETTINGS["symbol"]
    interval = interval or SETTINGS["interval"]
    start = start or SETTINGS["backtest_start"]
    end = end or SETTINGS["backtest_end"]

    if client is None:
        client = _create_client()

    start_str = f"{start} 00:00:00"
    end_str = f"{end} 23:59:59"

    logger.info(
        "Fetching %s %s klines: %s ~ %s", symbol, interval, start, end
    )

    all_klines: list[list] = []
    batch_count = 0

    # Binance get_historical_klines는 자동으로 페이지네이션을 처리하지만,
    # 대량 요청 시 rate limit을 존중하기 위해 직접 배치로 가져온다.
    start_ts = int(
        datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
    end_ts = int(
        datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )

    current_start = start_ts
    limit = 1000  # Binance API 최대 한 번에 1000개

    while current_start < end_ts:
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ts,
                limit=limit,
            )
        except Exception:
            logger.exception("API 요청 실패, 5초 후 재시도")
            time.sleep(5)
            continue

        if not klines:
            break

        all_klines.extend(klines)
        batch_count += 1

        # 다음 배치 시작점: 마지막 캔들의 close_time + 1ms
        current_start = klines[-1][6] + 1

        if batch_count % 10 == 0:
            logger.info("  %d batches fetched (%d candles)", batch_count, len(all_klines))

        # Rate limit 준수 (weight 5/req, 한도 2400/min → 초당 8회 안전)
        time.sleep(0.12)

    if not all_klines:
        logger.warning("수집된 데이터가 없습니다.")
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)

    # 타입 변환
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # UTC 시간 인덱스 설정
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    # 불필요 컬럼 제거
    df = df.drop(columns=["trades", "taker_buy_base", "taker_buy_quote", "ignore"])

    # 중복 제거 및 정렬
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    logger.info("총 %d개 캔들 수집 완료", len(df))
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """수집된 데이터의 결측치와 시간 연속성을 검증한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        검증 및 보정된 DataFrame.

    Raises:
        ValueError: 데이터가 비어있을 경우.
    """
    if df.empty:
        raise ValueError("검증할 데이터가 없습니다.")

    # 결측치 확인
    null_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    if null_counts.any():
        logger.warning("결측치 발견:\n%s", null_counts[null_counts > 0])
        # forward fill로 보정
        df = df.ffill()
        logger.info("결측치를 forward fill로 보정했습니다.")

    # 시간 연속성 확인 (5분 간격)
    time_diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(minutes=5)
    gaps = time_diffs[time_diffs != expected_diff]

    if not gaps.empty:
        logger.warning(
            "시간 연속성 이상 %d건 발견 (주말/점검 등 정상일 수 있음)",
            len(gaps),
        )
        for ts, gap in gaps.head(5).items():
            logger.warning("  %s: 갭 = %s", ts, gap)

    # OHLC 논리 검증
    invalid_ohlc = df[(df["high"] < df["low"]) | (df["volume"] < 0)]
    if not invalid_ohlc.empty:
        logger.warning("비정상 OHLC 데이터 %d건 발견", len(invalid_ohlc))

    logger.info("데이터 검증 완료: %d개 캔들, %s ~ %s",
                len(df), df.index[0], df.index[-1])
    return df
