"""백테스트 실행 진입점.

사용법:
    # 데이터 수집
    python main_backtest.py --fetch-data --start 2024-01-01 --end 2024-12-31

    # 백테스트 실행
    python main_backtest.py --run

    # 데이터 수집 + 백테스트
    python main_backtest.py --fetch-data --run --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys

from config.settings import SETTINGS
from data.fetcher import fetch_klines, validate_data
from data.storage import save_to_parquet, load_from_parquet
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics
from analysis.report import print_report
from analysis.visualize import (
    plot_equity_curve,
    plot_trade_overlay,
    plot_monthly_returns,
    plot_r_distribution,
)
from strategy.signals import compute_indicators

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_data(start: str, end: str) -> None:
    """Binance에서 데이터를 수집하고 캐시에 저장한다."""
    logger.info("데이터 수집 시작: %s ~ %s", start, end)

    df = fetch_klines(start=start, end=end)
    if df.empty:
        logger.error("데이터 수집 실패")
        sys.exit(1)

    df = validate_data(df)
    path = save_to_parquet(df)
    logger.info("데이터 저장 완료: %s", path)


def run(start: str, end: str) -> None:
    """백테스트를 실행하고 결과를 출력한다."""
    logger.info("캐시에서 데이터 로드 중...")
    df = load_from_parquet(start=start, end=end)

    if df is None or df.empty:
        logger.error(
            "데이터가 없습니다. --fetch-data로 먼저 데이터를 수집하세요."
        )
        sys.exit(1)

    logger.info("데이터 로드: %d개 캔들", len(df))

    # 백테스트 실행
    portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])

    # 성과 지표 계산
    metrics = calculate_metrics(portfolio)

    # 리포트 출력
    print_report(metrics, start=start, end=end)

    # 시각화
    if portfolio.trades:
        df_with_indicators = compute_indicators(df)
        plot_equity_curve(portfolio, metrics)
        plot_trade_overlay(df_with_indicators, portfolio)
        plot_monthly_returns(portfolio, df)
        plot_r_distribution(portfolio)
        logger.info("차트 파일 생성 완료 (HTML)")
    else:
        logger.warning("거래가 없어 차트를 생성하지 않았습니다.")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(
        description="BTC 5min Multi-Confluence Momentum Scalp — Backtest"
    )
    parser.add_argument(
        "--fetch-data", action="store_true",
        help="Binance에서 캔들 데이터를 수집한다.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="백테스트를 실행한다.",
    )
    parser.add_argument(
        "--start", type=str, default=SETTINGS["backtest_start"],
        help="시작 날짜 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=SETTINGS["backtest_end"],
        help="종료 날짜 (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    if not args.fetch_data and not args.run:
        parser.print_help()
        sys.exit(0)

    if args.fetch_data:
        fetch_data(args.start, args.end)

    if args.run:
        run(args.start, args.end)


if __name__ == "__main__":
    main()
