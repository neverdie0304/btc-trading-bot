"""멀티 심볼 봇 관리자.

여러 LiveBot 인스턴스를 동시에 관리하고, 상태를 조회한다.
"""

import asyncio
import logging

from binance import AsyncClient

from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, SETTINGS
from main_live import LiveBot
from live.logger_db import TradeLogger

logger = logging.getLogger(__name__)

MAX_BOTS = 3


class BotManager:
    """여러 LiveBot 인스턴스를 관리한다."""

    def __init__(self) -> None:
        self.bots: dict[str, LiveBot] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.client: AsyncClient | None = None
        self.db: TradeLogger | None = None

    async def initialize(self) -> None:
        """공유 Binance 클라이언트를 생성한다."""
        self.client = await AsyncClient.create(
            BINANCE_API_KEY, BINANCE_API_SECRET
        )
        self.db = TradeLogger()
        logger.info("BotManager 초기화 완료")

    async def start_bot(
        self,
        symbol: str,
        mode: str = "paper",
        capital: float = 0,
    ) -> dict:
        """봇을 시작한다."""
        symbol = symbol.upper()

        if symbol in self.bots and not self.bots[symbol]._shutdown:
            return {"success": False, "error": f"{symbol} 봇이 이미 실행 중입니다"}

        if len([s for s, b in self.bots.items() if not b._shutdown]) >= MAX_BOTS:
            return {"success": False, "error": f"최대 {MAX_BOTS}개까지 동시 실행 가능합니다"}

        if capital <= 0:
            capital = SETTINGS["initial_capital"]

        bot = LiveBot(
            mode=mode,
            capital=capital,
            symbol=symbol,
            shared_client=self.client,
        )

        self.bots[symbol] = bot

        async def _run_bot():
            try:
                await bot.start()
            except Exception:
                logger.exception("%s 봇 에러", symbol)
            finally:
                logger.info("%s 봇 종료됨", symbol)

        task = asyncio.create_task(_run_bot())
        self.tasks[symbol] = task

        # 봇이 초기화될 시간을 줌
        await asyncio.sleep(2)

        logger.info("봇 시작: %s (%s)", symbol, mode)
        return {"success": True, "symbol": symbol, "mode": mode}

    async def stop_bot(self, symbol: str) -> dict:
        """봇을 중지한다."""
        symbol = symbol.upper()

        if symbol not in self.bots:
            return {"success": False, "error": f"{symbol} 봇을 찾을 수 없습니다"}

        bot = self.bots[symbol]
        if bot._shutdown:
            return {"success": False, "error": f"{symbol} 봇이 이미 중지되었습니다"}

        await bot.stop()

        # Task도 정리
        task = self.tasks.get(symbol)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("봇 중지: %s", symbol)
        return {"success": True, "symbol": symbol}

    def get_bot_status(self, symbol: str) -> dict | None:
        """특정 봇의 상태를 반환한다."""
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot is None:
            return None
        return bot.get_status()

    def get_all_status(self) -> list[dict]:
        """모든 봇의 상태를 반환한다."""
        statuses = []
        for symbol, bot in self.bots.items():
            statuses.append(bot.get_status())
        return statuses

    def get_recent_trades(self, symbol: str | None = None, limit: int = 30) -> list[dict]:
        """최근 거래 내역을 반환한다."""
        if self.db is None:
            return []
        if symbol:
            return self.db.get_trades_by_symbol(symbol.upper(), limit)
        return self.db.get_recent_trades(limit)

    async def shutdown_all(self) -> None:
        """모든 봇을 중지하고 리소스를 정리한다."""
        logger.info("전체 봇 종료 시작...")

        for symbol in list(self.bots.keys()):
            bot = self.bots[symbol]
            if not bot._shutdown:
                try:
                    await bot.stop()
                except Exception:
                    logger.exception("%s 봇 종료 실패", symbol)

        # Task 정리
        for symbol, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # 공유 클라이언트 종료
        if self.client:
            await self.client.close_connection()

        if self.db:
            self.db.close()

        logger.info("전체 봇 종료 완료")
