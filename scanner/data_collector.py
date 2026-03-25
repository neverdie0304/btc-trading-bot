"""멀티심볼 백테스트용 데이터 수집 모듈.

1단계: 전체 USDT-M 선물의 일봉 거래대금 수집 → 일별 활성 심볼 결정
2단계: 활성 심볼의 5분봉 데이터 수집 및 캐시
"""

import logging
import time
from pathlib import Path

import pandas as pd
from binance.client import Client

from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, SETTINGS
from data.fetcher import fetch_klines
from data.storage import save_to_parquet, load_from_parquet
from scanner.config import SCANNER_SETTINGS

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(SETTINGS["cache_dir"])


def _create_client() -> Client:
    return Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def fetch_all_futures_symbols(client: Client | None = None) -> list[str]:
    """Binance USDT-M 선물의 전체 심볼 목록을 가져온다."""
    if client is None:
        client = _create_client()

    info = client.futures_exchange_info()
    quote = SCANNER_SETTINGS["quote_asset"]
    excluded = set(SCANNER_SETTINGS["excluded_symbols"])

    symbols = []
    for s in info.get("symbols", []):
        sym = s["symbol"]
        if sym.endswith(quote) and sym not in excluded and s.get("status") == "TRADING":
            symbols.append(sym)

    logger.info("전체 USDT-M 선물 심볼: %d개", len(symbols))
    return symbols


def fetch_daily_volumes(
    symbols: list[str],
    start: str,
    end: str,
    client: Client | None = None,
) -> pd.DataFrame:
    """전체 심볼의 일봉 거래대금(quote_volume)을 수집한다.

    캐시 파일이 있으면 로드한다.

    Returns:
        DataFrame: index=날짜, columns=심볼, values=quote_volume
    """
    cache_path = _CACHE_DIR / f"daily_volumes_{start}_{end}.parquet"

    if cache_path.exists():
        logger.info("일봉 거래대금 캐시 로드: %s", cache_path)
        return pd.read_parquet(cache_path)

    if client is None:
        client = _create_client()

    logger.info("일봉 거래대금 수집 시작: %d개 심볼, %s ~ %s", len(symbols), start, end)

    all_data: dict[str, pd.Series] = {}
    failed = []

    for i, sym in enumerate(symbols):
        try:
            klines = client.futures_klines(
                symbol=sym,
                interval="1d",
                startTime=int(pd.Timestamp(f"{start} 00:00:00", tz="UTC").timestamp() * 1000),
                endTime=int(pd.Timestamp(f"{end} 23:59:59", tz="UTC").timestamp() * 1000),
                limit=1000,
            )

            if not klines:
                continue

            dates = []
            volumes = []
            for k in klines:
                dates.append(pd.Timestamp(k[0], unit="ms", tz="UTC").normalize())
                volumes.append(float(k[7]))  # quote_volume

            all_data[sym] = pd.Series(volumes, index=dates, name=sym)

        except Exception:
            failed.append(sym)
            logger.debug("%s: 일봉 수집 실패", sym)

        # rate limit
        if (i + 1) % 20 == 0:
            time.sleep(1)
            if (i + 1) % 100 == 0:
                logger.info("  %d/%d 심볼 수집 완료", i + 1, len(symbols))

    if not all_data:
        logger.error("수집된 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.index.name = "date"
    df = df.sort_index()

    # 캐시 저장
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(
        "일봉 거래대금 수집 완료: %d 심볼, %d일 → %s",
        len(all_data), len(df), cache_path,
    )
    if failed:
        logger.warning("수집 실패: %d개 심볼", len(failed))

    return df


def get_daily_active_symbols(
    daily_volumes: pd.DataFrame,
    min_volume: float | None = None,
    max_volume: float | None = None,
) -> dict[str, list[str]]:
    """일별 거래대금 기준을 충족하는 심볼 목록을 반환한다.

    Args:
        daily_volumes: fetch_daily_volumes()의 결과.
        min_volume: 최소 거래대금 (기본: SCANNER_SETTINGS).
        max_volume: 최대 거래대금 (None이면 무제한).

    Returns:
        {날짜문자열: [심볼 리스트]} 딕셔너리.
    """
    min_vol = min_volume or SCANNER_SETTINGS["min_24h_quote_volume"]
    max_vol = max_volume or SCANNER_SETTINGS.get("max_24h_quote_volume")

    result: dict[str, list[str]] = {}
    unique_symbols = set()

    for date in daily_volumes.index:
        row = daily_volumes.loc[date]
        mask = row >= min_vol
        if max_vol:
            mask = mask & (row <= max_vol)
        active = row[mask].dropna().index.tolist()
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)[:10]
        result[date_str] = active
        unique_symbols.update(active)

    logger.info(
        "일별 활성 심볼: %d일, 고유 심볼 %d개, 일평균 %.1f개",
        len(result),
        len(unique_symbols),
        sum(len(v) for v in result.values()) / max(len(result), 1),
    )

    return result


def fetch_5m_data_for_symbols(
    symbols: list[str],
    start: str,
    end: str,
    client: Client | None = None,
) -> dict[str, pd.DataFrame]:
    """심볼 리스트의 5분봉 데이터를 수집/캐시한다.

    Args:
        symbols: 수집할 심볼 리스트.
        start: 시작 날짜.
        end: 종료 날짜.
        client: Binance Client.

    Returns:
        {심볼: DataFrame} 딕셔너리.
    """
    if client is None:
        client = _create_client()

    result: dict[str, pd.DataFrame] = {}
    total = len(symbols)

    for i, sym in enumerate(symbols):
        # 캐시 확인
        cached = load_from_parquet(symbol=sym, interval="5m", start=start, end=end)
        if cached is not None and len(cached) > 1000:
            result[sym] = cached
            logger.debug("%s: 캐시에서 로드 (%d봉)", sym, len(cached))
        else:
            # 수집
            try:
                logger.info("[%d/%d] %s 5분봉 수집 중...", i + 1, total, sym)
                df = fetch_klines(
                    symbol=sym,
                    interval="5m",
                    start=start,
                    end=end,
                    client=client,
                )
                if not df.empty:
                    save_to_parquet(df, symbol=sym, interval="5m")
                    result[sym] = df
                    logger.info("  %s: %d봉 수집 완료", sym, len(df))
                else:
                    logger.warning("  %s: 데이터 없음", sym)
            except Exception:
                logger.exception("  %s: 수집 실패", sym)

        # 진행 상황
        if (i + 1) % 10 == 0:
            logger.info("5분봉 수집 진행: %d/%d 심볼 완료", i + 1, total)

    logger.info("5분봉 수집 완료: %d/%d 심볼", len(result), total)
    return result
