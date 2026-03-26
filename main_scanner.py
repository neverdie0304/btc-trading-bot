"""멀티 코인 스캐너 실행 진입점.

사용법:
    # Paper 모드 (시뮬레이션)
    python main_scanner.py --mode paper

    # Live 모드 (실제 거래)
    python main_scanner.py --mode live

    # 자본금 지정
    python main_scanner.py --mode paper --capital 1000
"""

import argparse
import asyncio
import logging
import signal

from config.settings import SETTINGS
from scanner.scanner_bot import ScannerBot

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/scanner.log"),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def run(mode: str, capital: float) -> None:
    """스캐너 봇을 실행한다."""
    bot = ScannerBot(mode=mode, capital=capital)

    loop = asyncio.get_event_loop()

    def _signal_handler():
        logger.info("Shutdown signal received...")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Scanner error")
    finally:
        await bot.stop()


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(
        description="Multi-Coin Signal Scanner"
    )
    parser.add_argument(
        "--mode", type=str, choices=["paper", "live"], default="paper",
        help="트레이딩 모드 (기본값: paper)",
    )
    parser.add_argument(
        "--capital", type=float, default=SETTINGS["initial_capital"],
        help="초기 자본금 (기본값: settings)",
    )

    args = parser.parse_args()

    logger.info(
        "Starting scanner: mode=%s, capital=%.2f",
        args.mode, args.capital,
    )

    asyncio.run(run(args.mode, args.capital))


if __name__ == "__main__":
    main()
