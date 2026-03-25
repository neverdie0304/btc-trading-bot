"""라이브 트레이딩 상태 관리 모듈.

현재 포지션, 잔고, 미체결 주문, 연속 패배 등을 추적한다.
"""

import json
import logging
from dataclasses import dataclass, field
from strategy.position import Position
from strategy.signals import Signal
from live.logger_db import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class LiveTrade:
    """라이브 거래 기록."""
    trade_db_id: int = 0
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    commission: float = 0.0
    r_multiple: float = 0.0
    entry_order_id: str = ""
    exit_order_id: str = ""
    sl_fill_type: str = ""


@dataclass
class LiveState:
    """라이브 트레이딩 상태."""

    # 심볼
    symbol: str = ""

    # 잔고
    balance: float = 0.0
    initial_balance: float = 0.0

    # 포지션
    position: Position | None = None
    entry_order_id: str | None = None
    sl_order_id: str | None = None
    tp_order_id: str | None = None
    partial_tp_order_id: str | None = None  # 분할 익절 주문 ID (체결 후 None 초기화)

    # 추적
    consecutive_losses: int = 0
    last_loss_candle_count: int = 0
    candle_count: int = 0  # 총 처리한 봉 수
    daily_pnl: float = 0.0
    trades_today: list[LiveTrade] = field(default_factory=list)

    # 진입 LIMIT 주문 추적
    entry_filled: bool = False       # LIMIT 체결 여부 (False = 주문 대기 중)
    entry_limit_candle: int = 0      # LIMIT 주문 발행 시점의 candle_count

    # 세션
    session_start: str = ""
    is_active: bool = True

    def reset_daily(self) -> None:
        """일일 통계를 초기화한다."""
        self.daily_pnl = 0.0
        self.trades_today = []
        logger.info("Daily stats reset")

    def record_win(self) -> None:
        """승리 기록."""
        self.consecutive_losses = 0

    def record_loss(self) -> None:
        """패배 기록."""
        self.consecutive_losses += 1
        self.last_loss_candle_count = self.candle_count

    def candles_since_last_loss(self) -> int:
        """마지막 패배 이후 경과 봉 수."""
        return self.candle_count - self.last_loss_candle_count

    def save_to_db(self, db: TradeLogger) -> None:
        """현재 상태를 DB에 저장한다."""
        state_dict = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "consecutive_losses": self.consecutive_losses,
            "last_loss_candle_count": self.last_loss_candle_count,
            "candle_count": self.candle_count,
            "daily_pnl": self.daily_pnl,
            "is_active": self.is_active,
            "session_start": self.session_start,
            "entry_filled": self.entry_filled,
            "entry_limit_candle": self.entry_limit_candle,
        }

        # 포지션 정보
        if self.position:
            state_dict["position"] = {
                "direction": self.position.direction.value,
                "entry_price": self.position.entry_price,
                "size": self.position.size,
                "original_size": self.position.original_size,
                "sl_price": self.position.sl_price,
                "tp_price": self.position.tp_price,
                "initial_sl": self.position.initial_sl,
                "entry_time": self.position.entry_time,
                "r_unit": self.position.r_unit,
                "trailing_state": self.position.trailing_state,
            }
            state_dict["entry_order_id"] = self.entry_order_id
            state_dict["sl_order_id"] = self.sl_order_id
            state_dict["tp_order_id"] = self.tp_order_id
            state_dict["partial_tp_order_id"] = self.partial_tp_order_id
        else:
            state_dict["position"] = None

        key = f"live_state:{self.symbol}" if self.symbol else "live_state"
        db.save_state(key, json.dumps(state_dict))
        logger.debug("State saved: %s", key)

    @classmethod
    def restore_from_db(cls, db: TradeLogger, symbol: str = "") -> "LiveState | None":
        """DB에서 상태를 복원한다."""
        key = f"live_state:{symbol}" if symbol else "live_state"
        raw = db.load_state(key)
        if raw is None:
            return None

        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("State restore failed: invalid JSON")
            return None

        state = cls(
            balance=d.get("balance", 0),
            initial_balance=d.get("initial_balance", 0),
            consecutive_losses=d.get("consecutive_losses", 0),
            last_loss_candle_count=d.get("last_loss_candle_count", 0),
            candle_count=d.get("candle_count", 0),
            daily_pnl=d.get("daily_pnl", 0),
            is_active=d.get("is_active", True),
            session_start=d.get("session_start", ""),
            entry_filled=d.get("entry_filled", False),
            entry_limit_candle=d.get("entry_limit_candle", 0),
        )

        # 포지션 복원
        pos_data = d.get("position")
        if pos_data:
            state.position = Position(
                direction=Signal(pos_data["direction"]),
                entry_price=pos_data["entry_price"],
                size=pos_data["size"],
                sl_price=pos_data["sl_price"],
                tp_price=pos_data["tp_price"],
                initial_sl=pos_data["initial_sl"],
                entry_time=pos_data.get("entry_time", ""),
                r_unit=pos_data.get("r_unit", 0),
                trailing_state=pos_data.get("trailing_state", "initial"),
                original_size=pos_data.get("original_size", pos_data["size"]),
            )
            state.entry_order_id = d.get("entry_order_id")
            state.sl_order_id = d.get("sl_order_id")
            state.tp_order_id = d.get("tp_order_id")
            state.partial_tp_order_id = d.get("partial_tp_order_id")
            logger.info(
                "Position restored: %s @ %.2f, SL=%.2f, TP=%.2f",
                state.position.direction.value,
                state.position.entry_price,
                state.position.sl_price,
                state.position.tp_price,
            )

        logger.info("State restored: balance=%.2f, candles=%d", state.balance, state.candle_count)
        return state
