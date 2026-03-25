"""거래대금 기반 심볼 선정 모듈.

Binance USDT-M 선물에서 24시간 거래대금이 기준치를 초과하는 심볼을 선정한다.
"""

import logging
from dataclasses import dataclass

from binance import AsyncClient

from scanner.config import SCANNER_SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """심볼 거래 규칙."""
    symbol: str
    tick_size: float       # 가격 최소 단위
    step_size: float       # 수량 최소 단위
    quote_volume_24h: float = 0.0


async def fetch_top_symbols(client: AsyncClient) -> list[str]:
    """24시간 거래대금 기준으로 상위 심볼 목록을 반환한다.

    Args:
        client: Binance AsyncClient.

    Returns:
        거래대금 내림차순 정렬된 심볼 문자열 리스트.
    """
    min_volume = SCANNER_SETTINGS["min_24h_quote_volume"]
    quote_asset = SCANNER_SETTINGS["quote_asset"]
    excluded = set(SCANNER_SETTINGS["excluded_symbols"])

    tickers = await client.futures_ticker()

    candidates: list[tuple[str, float]] = []
    for t in tickers:
        symbol = t["symbol"]
        quote_vol = float(t.get("quoteVolume", 0))

        # USDT 마진 페어만
        if not symbol.endswith(quote_asset):
            continue
        # 제외 심볼
        if symbol in excluded:
            continue
        # 거래대금 필터
        if quote_vol < min_volume:
            continue

        candidates.append((symbol, quote_vol))

    # 거래대금 내림차순
    candidates.sort(key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in candidates]

    logger.info(
        "Symbols selected: %d (min: $%.0fM+ / total %d)",
        len(symbols),
        min_volume / 1_000_000,
        len(tickers),
    )
    for i, (s, v) in enumerate(candidates[:10]):
        logger.info("  #%d %s — $%.1fM", i + 1, s, v / 1_000_000)
    if len(candidates) > 10:
        logger.info("  ... and %d more", len(candidates) - 10)

    return symbols


async def fetch_symbol_info(
    client: AsyncClient,
    symbols: list[str],
) -> dict[str, SymbolInfo]:
    """심볼별 거래 규칙(tick_size, step_size)을 조회한다.

    Args:
        client: Binance AsyncClient.
        symbols: 조회할 심볼 리스트.

    Returns:
        {심볼: SymbolInfo} 딕셔너리.
    """
    exchange_info = await client.futures_exchange_info()
    symbol_set = set(symbols)
    result: dict[str, SymbolInfo] = {}

    for s in exchange_info.get("symbols", []):
        sym = s["symbol"]
        if sym not in symbol_set:
            continue

        tick_size = 0.01
        step_size = 0.001
        for f in s.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])

        result[sym] = SymbolInfo(
            symbol=sym,
            tick_size=tick_size,
            step_size=step_size,
        )

    found = len(result)
    if found < len(symbols):
        missing = symbol_set - set(result.keys())
        logger.warning("Symbol info missing: %s", missing)

    logger.info("Symbol info loaded: %d", found)
    return result
