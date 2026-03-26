"""Daily Rotation 백테스트 모듈.

매일 1일봉 데이터로 변동성+거래량 기준 최적 코인을 선정하고,
해당 코인의 5분봉으로 트레이딩하는 전략을 시뮬레이션한다.
"""

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import SETTINGS
from strategy.signals import compute_indicators, generate_signals, Signal
from strategy.filters import should_filter
from strategy.position import create_position, check_sl_tp_hit
from backtest.portfolio import Portfolio

logger = logging.getLogger(__name__)


def load_daily_data(daily_dir: str, min_days: int = 30) -> dict[str, pd.DataFrame]:
    """1일봉 데이터를 모두 로드한다."""
    data = {}
    files = [f for f in os.listdir(daily_dir) if f.endswith("_1d.parquet")]
    for f in files:
        sym = f.replace("_1d.parquet", "")
        try:
            df = pd.read_parquet(os.path.join(daily_dir, f))
            df_2024 = df["2024-01-01":"2024-12-31"]
            if len(df_2024) >= min_days:
                data[sym] = df_2024
        except Exception:
            continue
    return data


def select_daily_coin(
    all_daily: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    min_volume: float = 10_000_000,
    vol_weight: float = 0.7,
    volume_weight: float = 0.3,
) -> dict | None:
    """특정 날짜에 가장 적합한 코인을 선정한다."""
    skip = {"USDCUSDT", "USTCUSDT", "BTCDOMUSDT"}
    scores = []

    for sym, df in all_daily.items():
        if sym in skip:
            continue
        hist = df[df.index <= day]
        if len(hist) < 14:
            continue

        recent = hist.tail(14)
        atr14 = (recent["high"] - recent["low"]).mean()
        last_close = hist.iloc[-1]["close"]

        if last_close == 0:
            continue

        volatility = atr14 / last_close
        avg_volume = hist.tail(7)["quote_volume"].mean()

        if avg_volume < min_volume:
            continue

        scores.append({
            "symbol": sym,
            "volatility": volatility,
            "volume": avg_volume,
        })

    if not scores:
        return None

    scores_df = pd.DataFrame(scores)
    scores_df["vol_rank"] = scores_df["volatility"].rank(pct=True)
    scores_df["volume_rank"] = scores_df["volume"].rank(pct=True)
    scores_df["score"] = (
        scores_df["vol_rank"] * vol_weight + scores_df["volume_rank"] * volume_weight
    )

    top = scores_df.nlargest(1, "score").iloc[0]
    return {
        "symbol": top["symbol"],
        "volatility": top["volatility"],
        "volume": top["volume"],
        "score": top["score"],
    }


def run_rotation_backtest(
    daily_dir: str,
    cache_dir: str,
    capital: float = 10000,
    start_date: str = "2024-01-15",
    end_date: str = "2024-12-31",
) -> Portfolio:
    """Daily Rotation 백테스트를 실행한다.

    Args:
        daily_dir: 1일봉 데이터 디렉토리.
        cache_dir: 5분봉 데이터 캐시 디렉토리.
        capital: 초기 자본.
        start_date: 시작 날짜 (ATR 워밍업 후).
        end_date: 종료 날짜.

    Returns:
        백테스트 결과 Portfolio.
    """
    logger.info("Loading daily data...")
    all_daily = load_daily_data(daily_dir)
    logger.info("Loaded %d symbols for daily selection", len(all_daily))

    portfolio = Portfolio(initial_capital=capital, capital=capital)
    trading_days = pd.date_range(start_date, end_date, freq="D", tz="UTC")

    # 5분봉 데이터 캐시 (메모리)
    candle_cache: dict[str, pd.DataFrame] = {}
    indicator_cache: dict[str, pd.DataFrame] = {}
    signal_cache: dict[str, pd.Series] = {}

    daily_log = []
    skipped_no_data = 0
    skipped_no_pick = 0

    for day in tqdm(trading_days, desc="Rotation Backtest"):
        # 1. 오늘의 코인 선정
        pick = select_daily_coin(all_daily, day)
        if pick is None:
            skipped_no_pick += 1
            # Record equity for the day (288 candles)
            for _ in range(288):
                if portfolio.has_position():
                    # 포지션이 남아있으면 전날 마지막 가격 유지
                    portfolio.equity_curve.append(portfolio.equity_curve[-1] if portfolio.equity_curve else capital)
                else:
                    portfolio.record_equity(0)
            continue

        sym = pick["symbol"]

        # 2. 5분봉 데이터 로드
        if sym not in candle_cache:
            parquet_path = os.path.join(cache_dir, f"{sym}_5m.parquet")
            if not os.path.exists(parquet_path):
                skipped_no_data += 1
                for _ in range(288):
                    portfolio.record_equity(0)
                continue
            df = pd.read_parquet(parquet_path)
            candle_cache[sym] = df
            df_ind = compute_indicators(df.copy())
            indicator_cache[sym] = df_ind
            signal_cache[sym] = generate_signals(df_ind)

        df = indicator_cache[sym]
        signals = signal_cache[sym]

        # 3. 해당 날짜의 5분봉 추출
        day_str = day.strftime("%Y-%m-%d")
        next_day = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        day_mask = (df.index >= day_str) & (df.index < next_day)
        day_indices = np.where(day_mask)[0]

        if len(day_indices) == 0:
            for _ in range(288):
                portfolio.record_equity(0)
            continue

        daily_log.append({"date": day_str, "symbol": sym, "volatility": pick["volatility"]})

        # 4. 하루 동안 백테스트 (기존 포지션은 유지, 새 진입은 오늘 코인으로)
        # 만약 어제와 다른 코인이면 기존 포지션 청산
        if portfolio.has_position():
            # 기존 포지션이 다른 코인이면 강제 청산
            pos = portfolio.position
            if hasattr(pos, "_symbol") and pos._symbol != sym:
                # 전날 마지막 close로 청산
                last_close = df.iloc[day_indices[0]]["close"]
                portfolio.close_position(
                    exit_price=last_close,
                    exit_time=str(df.index[day_indices[0]]),
                    exit_index=day_indices[0],
                    exit_reason="ROTATION",
                )

        for idx in day_indices:
            row = df.iloc[idx]
            timestamp = df.index[idx]
            close = row["close"]
            high = row["high"]
            low = row["low"]

            # SL/TP 체크
            if portfolio.has_position():
                hit, exit_price, reason = check_sl_tp_hit(
                    portfolio.position, high, low
                )
                if hit:
                    portfolio.close_position(
                        exit_price=exit_price,
                        exit_time=str(timestamp),
                        exit_index=idx,
                        exit_reason=reason,
                    )

            # 필터 체크
            candles_since_loss = idx - portfolio.last_loss_index
            atr_val = row.get("atr", 0)
            atr_med = row.get("atr_median", 0)
            if pd.isna(atr_val):
                atr_val = 0
            if pd.isna(atr_med):
                atr_med = 0

            filtered = should_filter(
                timestamp=timestamp,
                atr=atr_val,
                atr_median=atr_med,
                consecutive_losses=portfolio.consecutive_losses,
                candles_since_last_loss=candles_since_loss,
            )

            # 시그널 처리
            signal = signals.iloc[idx]
            if signal != Signal.NO_SIGNAL and not filtered:
                if portfolio.has_position():
                    if SETTINGS["reverse_on_opposite_signal"]:
                        if portfolio.position.direction != signal:
                            portfolio.close_position(
                                exit_price=close,
                                exit_time=str(timestamp),
                                exit_index=idx,
                                exit_reason="REVERSE",
                            )
                    else:
                        portfolio.record_equity(close)
                        continue

                if not portfolio.has_position() and pd.notna(row.get("atr")):
                    position = create_position(
                        direction=signal,
                        entry_price=close,
                        atr=row["atr"],
                        capital=portfolio.capital,
                        entry_time=str(timestamp),
                        entry_index=idx,
                    )
                    # Tag symbol for rotation tracking
                    position._symbol = sym
                    portfolio.open_position(position)

            portfolio.record_equity(close)

    # 마지막 포지션 청산
    if portfolio.has_position():
        # 마지막 캔들의 close로 청산
        last_sym = portfolio.position._symbol if hasattr(portfolio.position, "_symbol") else ""
        if last_sym in indicator_cache:
            last_df = indicator_cache[last_sym]
            last_close = last_df.iloc[-1]["close"]
            portfolio.close_position(
                exit_price=last_close,
                exit_time=str(last_df.index[-1]),
                exit_index=len(last_df) - 1,
                exit_reason="END",
            )

    logger.info("Rotation backtest complete: %d trades", len(portfolio.trades))
    logger.info("Skipped (no pick): %d days, (no data): %d days", skipped_no_pick, skipped_no_data)

    return portfolio, daily_log
