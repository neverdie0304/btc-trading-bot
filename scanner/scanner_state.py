"""스캐너 통합 상태 관리 모듈.

멀티 심볼 포지션, 잔고, 심볼별 연속 패배 등을 추적한다.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from strategy.position import Position
from strategy.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class SymbolState:
    """심볼별 상태."""
    symbol: str
    consecutive_losses: int = 0
    last_loss_candle_count: int = 0
    candle_count: int = 0
    trades_count: int = 0


@dataclass
class ScannerState:
    """스캐너 통합 상태."""

    # 잔고
    balance: float = 0.0
    initial_balance: float = 0.0
    daily_pnl: float = 0.0

    # 포지션 (symbol -> Position)
    positions: dict[str, Position] = field(default_factory=dict)

    # 심볼별 상태
    symbol_states: dict[str, SymbolState] = field(default_factory=dict)

    # 심볼별 주문 ID (executor 관리용)
    sl_orders: dict[str, str | None] = field(default_factory=dict)
    tp_orders: dict[str, str | None] = field(default_factory=dict)

    # 활성 심볼
    active_symbols: list[str] = field(default_factory=list)

    # 총 거래 수
    total_trades: int = 0

    def get_symbol_state(self, symbol: str) -> SymbolState:
        """심볼 상태를 가져오거나 새로 생성한다."""
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(symbol=symbol)
        return self.symbol_states[symbol]

    def open_position(self, symbol: str, position: Position) -> None:
        """포지션을 등록한다."""
        self.positions[symbol] = position
        logger.info(
            "[STATE] %s %s position opened @ %.2f (size=%.4f)",
            symbol, position.direction.value, position.entry_price, position.size,
        )

    def close_position(self, symbol: str, pnl: float) -> None:
        """포지션을 닫고 PnL을 반영한다."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        self.balance += pnl
        self.daily_pnl += pnl
        self.total_trades += 1

        sym_state = self.get_symbol_state(symbol)
        sym_state.trades_count += 1

        if pnl < 0:
            sym_state.consecutive_losses += 1
            sym_state.last_loss_candle_count = sym_state.candle_count
        else:
            sym_state.consecutive_losses = 0

        # 주문 ID 정리
        self.sl_orders.pop(symbol, None)
        self.tp_orders.pop(symbol, None)

        logger.info(
            "[STATE] %s position closed: PnL=$%.2f | Balance=$%.2f",
            symbol, pnl, self.balance,
        )

    def increment_candle(self, symbol: str) -> None:
        """심볼의 캔들 카운트를 증가시킨다."""
        self.get_symbol_state(symbol).candle_count += 1

    def save_to_db(self, db_path: str) -> None:
        """상태를 SQLite에 저장한다."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scanner_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            state_data = {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "daily_pnl": self.daily_pnl,
                "total_trades": self.total_trades,
                "positions": {
                    sym: {
                        "direction": pos.direction.value,
                        "entry_price": pos.entry_price,
                        "size": pos.size,
                        "sl_price": pos.sl_price,
                        "tp_price": pos.tp_price,
                        "initial_sl": pos.initial_sl,
                        "entry_time": pos.entry_time,
                        "r_unit": pos.r_unit,
                    }
                    for sym, pos in self.positions.items()
                },
                "symbol_states": {
                    sym: {
                        "consecutive_losses": ss.consecutive_losses,
                        "last_loss_candle_count": ss.last_loss_candle_count,
                        "candle_count": ss.candle_count,
                        "trades_count": ss.trades_count,
                    }
                    for sym, ss in self.symbol_states.items()
                },
                "sl_orders": self.sl_orders,
                "tp_orders": self.tp_orders,
            }

            cursor.execute(
                "INSERT OR REPLACE INTO scanner_state (key, value) VALUES (?, ?)",
                ("state", json.dumps(state_data)),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Scanner state save failed")

    def restore_from_db(self, db_path: str) -> bool:
        """SQLite에서 상태를 복원한다.

        Returns:
            복원 성공 여부.
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM scanner_state WHERE key = 'state'
            """)
            row = cursor.fetchone()
            conn.close()

            if not row:
                return False

            data = json.loads(row[0])
            self.balance = data.get("balance", 0.0)
            self.initial_balance = data.get("initial_balance", 0.0)
            self.daily_pnl = data.get("daily_pnl", 0.0)
            self.total_trades = data.get("total_trades", 0)

            # 포지션 복원
            for sym, pos_data in data.get("positions", {}).items():
                self.positions[sym] = Position(
                    direction=Signal[pos_data["direction"]],
                    entry_price=pos_data["entry_price"],
                    size=pos_data["size"],
                    sl_price=pos_data["sl_price"],
                    tp_price=pos_data["tp_price"],
                    initial_sl=pos_data["initial_sl"],
                    entry_time=pos_data.get("entry_time", ""),
                    r_unit=pos_data.get("r_unit", 0.0),
                )

            # 심볼 상태 복원
            for sym, ss_data in data.get("symbol_states", {}).items():
                self.symbol_states[sym] = SymbolState(
                    symbol=sym,
                    consecutive_losses=ss_data.get("consecutive_losses", 0),
                    last_loss_candle_count=ss_data.get("last_loss_candle_count", 0),
                    candle_count=ss_data.get("candle_count", 0),
                    trades_count=ss_data.get("trades_count", 0),
                )

            # 주문 ID 복원
            self.sl_orders = data.get("sl_orders", {})
            self.tp_orders = data.get("tp_orders", {})

            logger.info(
                "Scanner state restored: balance=$%.2f, positions=%d, trades=%d",
                self.balance, len(self.positions), self.total_trades,
            )
            return True

        except sqlite3.OperationalError:
            # 테이블이 없는 경우
            return False
        except Exception:
            logger.exception("Scanner state restore failed")
            return False
