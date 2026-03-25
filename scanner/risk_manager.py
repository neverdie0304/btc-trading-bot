"""멀티 심볼 리스크 관리 모듈.

동시 포지션 수, 자본 배분, 일일 손실 한도를 관리한다.
"""

import logging

from config.settings import SETTINGS
from scanner.config import SCANNER_SETTINGS
from strategy.position import Position

logger = logging.getLogger(__name__)


class RiskManager:
    """멀티 심볼 리스크 관리자."""

    def __init__(self, capital: float) -> None:
        self.initial_capital = capital
        self.daily_pnl = 0.0

    @property
    def max_positions(self) -> int:
        return SCANNER_SETTINGS["max_concurrent_positions"]

    def available_slots(self, current_positions: int) -> int:
        """진입 가능한 슬롯 수."""
        return max(0, self.max_positions - current_positions)

    def can_open(
        self,
        symbol: str,
        positions: dict[str, Position],
        balance: float,
    ) -> bool:
        """새 포지션을 열 수 있는지 확인한다.

        Args:
            symbol: 진입할 심볼.
            positions: 현재 열린 포지션 {symbol: Position}.
            balance: 현재 잔고.

        Returns:
            진입 가능하면 True.
        """
        # 동일 심볼 중복 불가
        if symbol in positions:
            logger.debug("%s: already has position", symbol)
            return False

        # 최대 동시 포지션 체크
        if len(positions) >= self.max_positions:
            logger.debug("Max concurrent positions %d reached", self.max_positions)
            return False

        # 총 노출 한도 체크
        max_exposure = balance * SCANNER_SETTINGS["max_total_exposure_pct"]
        current_exposure = sum(
            abs(p.size * p.entry_price) for p in positions.values()
        )
        if current_exposure >= max_exposure:
            logger.debug("Total exposure limit reached: $%.2f / $%.2f", current_exposure, max_exposure)
            return False

        # 일일 손실 한도 체크
        max_loss_pct = SCANNER_SETTINGS["daily_max_loss_pct"]
        if max_loss_pct < 0:
            current_pct = (self.daily_pnl / self.initial_capital) * 100
            if current_pct <= max_loss_pct:
                logger.warning(
                    "Daily max loss reached: %.2f%% (limit: %.2f%%)",
                    current_pct, max_loss_pct,
                )
                return False

        return True

    def get_capital_for_trade(
        self,
        balance: float,
        positions: dict[str, Position],
    ) -> float:
        """단일 거래에 사용할 자본을 계산한다.

        Args:
            balance: 현재 잔고.
            positions: 현재 열린 포지션.

        Returns:
            거래에 사용할 자본 금액.
        """
        max_per_position = balance * SCANNER_SETTINGS["max_capital_per_position_pct"]

        # 이미 배분된 자본 제외
        used = sum(
            abs(p.size * p.entry_price) / SETTINGS.get("leverage", 20)
            for p in positions.values()
        )
        available = max(0, balance - used)

        return min(available, max_per_position)

    def update_daily_pnl(self, pnl: float) -> None:
        """일일 PnL을 업데이트한다."""
        self.daily_pnl += pnl

    def reset_daily(self, new_capital: float) -> None:
        """일일 리셋."""
        self.daily_pnl = 0.0
        self.initial_capital = new_capital
        logger.info("Risk daily reset: capital=$%.2f", new_capital)
