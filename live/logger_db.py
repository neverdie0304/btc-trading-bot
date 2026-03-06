"""SQLite 기반 거래 로그 모듈.

거래 기록, 일별 요약, 봇 상태를 영구 저장한다.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_PATH = _DB_DIR / "live_trades.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    size REAL NOT NULL,
    sl_price REAL,
    tp_price REAL,
    exit_reason TEXT,
    pnl REAL,
    commission REAL,
    r_multiple REAL,
    entry_order_id TEXT,
    exit_order_id TEXT,
    sl_fill_type TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_summary (
    date TEXT PRIMARY KEY,
    trades_count INTEGER,
    wins INTEGER,
    losses INTEGER,
    total_pnl REAL,
    max_drawdown REAL,
    ending_balance REAL
);

CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);
"""


class TradeLogger:
    """SQLite 거래 로거."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("TradeLogger 초기화: %s", self.db_path)

    def _init_schema(self) -> None:
        """테이블이 없으면 생성한다."""
        self._conn.executescript(SCHEMA_SQL)
        # symbol 컬럼이 없으면 추가 (기존 DB 호환)
        try:
            self._conn.execute("SELECT symbol FROM trades LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE trades ADD COLUMN symbol TEXT DEFAULT ''")
            logger.info("trades 테이블에 symbol 컬럼 추가")
        self._conn.commit()

    def log_trade(
        self,
        timestamp: str,
        direction: str,
        entry_price: float,
        exit_price: float | None,
        size: float,
        sl_price: float | None = None,
        tp_price: float | None = None,
        exit_reason: str | None = None,
        pnl: float | None = None,
        commission: float | None = None,
        r_multiple: float | None = None,
        entry_order_id: str | None = None,
        exit_order_id: str | None = None,
        sl_fill_type: str | None = None,
        notes: str | None = None,
        symbol: str = "",
    ) -> int:
        """거래를 기록한다. 반환: row id."""
        cur = self._conn.execute(
            """INSERT INTO trades
               (timestamp, direction, entry_price, exit_price, size,
                sl_price, tp_price, exit_reason, pnl, commission,
                r_multiple, entry_order_id, exit_order_id, sl_fill_type, notes, symbol)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, direction, entry_price, exit_price, size,
                sl_price, tp_price, exit_reason, pnl, commission,
                r_multiple, entry_order_id, exit_order_id, sl_fill_type, notes, symbol,
            ),
        )
        self._conn.commit()
        logger.info("거래 기록: %s %s %s @ %.2f, PnL=%.2f", symbol, direction, exit_reason or "OPEN", entry_price, pnl or 0)
        return cur.lastrowid

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        commission: float,
        r_multiple: float,
        exit_order_id: str | None = None,
        sl_fill_type: str | None = None,
    ) -> None:
        """기존 거래의 청산 정보를 업데이트한다."""
        self._conn.execute(
            """UPDATE trades
               SET exit_price=?, exit_reason=?, pnl=?, commission=?,
                   r_multiple=?, exit_order_id=?, sl_fill_type=?
               WHERE id=?""",
            (exit_price, exit_reason, pnl, commission, r_multiple,
             exit_order_id, sl_fill_type, trade_id),
        )
        self._conn.commit()

    def update_trade_entry(
        self,
        trade_id: int,
        entry_price: float,
        size: float,
        sl_price: float,
        tp_price: float,
    ) -> None:
        """LIMIT 진입 체결 후 실제 체결가·수량·SL/TP를 업데이트한다."""
        self._conn.execute(
            """UPDATE trades
               SET entry_price=?, size=?, sl_price=?, tp_price=?, notes=NULL
               WHERE id=?""",
            (entry_price, size, sl_price, tp_price, trade_id),
        )
        self._conn.commit()

    def update_daily_summary(
        self,
        date: str,
        trades_count: int,
        wins: int,
        losses: int,
        total_pnl: float,
        max_drawdown: float,
        ending_balance: float,
    ) -> None:
        """일별 요약을 업데이트한다."""
        self._conn.execute(
            """INSERT OR REPLACE INTO daily_summary
               (date, trades_count, wins, losses, total_pnl, max_drawdown, ending_balance)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (date, trades_count, wins, losses, total_pnl, max_drawdown, ending_balance),
        )
        self._conn.commit()

    def save_state(self, key: str, value: str) -> None:
        """봇 상태를 저장한다."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        self._conn.commit()

    def load_state(self, key: str) -> str | None:
        """봇 상태를 로드한다."""
        row = self._conn.execute(
            "SELECT value FROM bot_state WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_today_trades(self) -> list[dict]:
        """오늘 거래 목록을 반환한다."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE date(timestamp) = ?", (today,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_by_symbol(self, symbol: str, limit: int = 50) -> list[dict]:
        """특정 심볼의 최근 거래 목록을 반환한다."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """최근 거래 목록을 반환한다."""
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """DB 연결을 닫는다."""
        self._conn.close()
