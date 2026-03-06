"""Kill Switch 모듈.

일일 최대 손실, 최대 거래 횟수, 연속 패배 등을 모니터링하고
한도 초과 시 포지션을 강제 청산한다.
"""

import logging
import time

from live.state import LiveState

logger = logging.getLogger(__name__)

# Kill switch 기본 규칙
KILL_SWITCH_RULES = {
    "daily_max_loss_pct": -5.0,       # 일일 -5% 도달 시 전체 정지
    "daily_max_trades": 20,            # 일일 최대 거래 횟수
    "max_consecutive_losses": 5,       # 연속 5패 시 당일 정지
    "max_single_loss_pct": -3.0,       # 단일 거래 -3% 초과 시 경고
    "heartbeat_timeout_sec": 120,      # WebSocket 120초 무응답 시 포지션 청산
}


class KillSwitch:
    """안전 장치. 한도 초과 시 트레이딩을 중단한다."""

    def __init__(self, state: LiveState, rules: dict | None = None) -> None:
        self.state = state
        self.rules = rules or KILL_SWITCH_RULES

    def check(self, last_ws_time: float | None = None) -> tuple[bool, str]:
        """모든 kill switch 조건을 확인한다.

        Args:
            last_ws_time: 마지막 WebSocket 메시지 시각 (epoch).

        Returns:
            (should_stop, reason) 튜플.
        """
        if not self.state.is_active:
            return True, "ALREADY_STOPPED"

        # 1. 일일 손실 한도
        if self.state.initial_balance > 0:
            daily_loss_pct = (self.state.daily_pnl / self.state.initial_balance) * 100
            max_loss = self.rules["daily_max_loss_pct"]
            if daily_loss_pct <= max_loss:
                logger.critical(
                    "KILL SWITCH: 일일 손실 한도 초과 (%.2f%% <= %.2f%%)",
                    daily_loss_pct, max_loss,
                )
                return True, f"DAILY_LOSS_{daily_loss_pct:.1f}%"

        # 2. 일일 거래 횟수
        max_trades = self.rules["daily_max_trades"]
        if len(self.state.trades_today) >= max_trades:
            logger.critical("KILL SWITCH: 일일 최대 거래 횟수 도달 (%d)", max_trades)
            return True, f"MAX_TRADES_{max_trades}"

        # 3. 연속 패배
        max_consec = self.rules["max_consecutive_losses"]
        if self.state.consecutive_losses >= max_consec:
            logger.critical(
                "KILL SWITCH: 연속 %d패 도달", self.state.consecutive_losses,
            )
            return True, f"CONSEC_LOSSES_{self.state.consecutive_losses}"

        # 4. WebSocket heartbeat
        if last_ws_time is not None:
            timeout = self.rules["heartbeat_timeout_sec"]
            elapsed = time.time() - last_ws_time
            if elapsed > timeout:
                logger.critical(
                    "KILL SWITCH: WebSocket 무응답 %.0f초 (한도: %d초)",
                    elapsed, timeout,
                )
                return True, f"WS_TIMEOUT_{elapsed:.0f}s"

        return False, ""

    def check_single_trade(self, pnl: float) -> bool:
        """단일 거래의 손실이 경고 수준인지 확인한다.

        Args:
            pnl: 거래의 PnL.

        Returns:
            경고가 필요하면 True.
        """
        if self.state.initial_balance <= 0:
            return False

        loss_pct = (pnl / self.state.initial_balance) * 100
        threshold = self.rules["max_single_loss_pct"]
        if loss_pct <= threshold:
            logger.warning(
                "경고: 단일 거래 손실 %.2f%% (임계: %.2f%%)",
                loss_pct, threshold,
            )
            return True
        return False
