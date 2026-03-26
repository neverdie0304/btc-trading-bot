"""Telegram bot for scanner notifications and status queries.

Sends automatic alerts on position entry/exit/trailing stop changes,
and responds to user commands like /status, /positions, /trades.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)


def _fmt_price(price: float) -> str:
    """Format price based on magnitude."""
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    else:
        return f"{price:.8f}"


class TelegramNotifier:
    """Telegram notification bot for the scanner.

    Handles both push notifications (trade alerts) and
    pull commands (/status, /positions, /trades).
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._bot: Bot | None = None
        self._app: Application | None = None
        self._scanner = None  # set by scanner_bot after init
        self._running = False

        if not self._token or not self._chat_id:
            logger.warning("Telegram bot token or chat_id not set, notifications disabled")

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    def set_scanner(self, scanner) -> None:
        """Set reference to ScannerBot for command handlers."""
        self._scanner = scanner

    async def start(self) -> None:
        """Start the Telegram bot polling."""
        if not self.enabled:
            return

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Register command handlers
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("start", self._cmd_help))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        self._bot = self._app.bot
        self._running = True
        logger.info("Telegram bot started")

        await self._send("Scanner bot started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._running or not self._app:
            return

        self._running = False
        try:
            await self._send("Scanner bot stopped")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            logger.exception("Telegram bot stop error")
        logger.info("Telegram bot stopped")

    # ── Push Notifications ──────────────────────────────

    async def notify_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        sl_price: float,
        tp_price: float,
        score: float,
    ) -> None:
        """Send position entry alert."""
        arrow = "\U0001f4c8" if direction == "LONG" else "\U0001f4c9"
        notional = size * entry_price

        sl_pct = abs(sl_price - entry_price) / entry_price * 100
        tp_pct = abs(tp_price - entry_price) / entry_price * 100

        msg = (
            f"{arrow} *{direction} {symbol}*\n"
            f"Entry: ${_fmt_price(entry_price)}\n"
            f"Size: {size:.4f} (${notional:.2f})\n"
            f"SL: ${_fmt_price(sl_price)} (-{sl_pct:.2f}%)\n"
            f"TP: ${_fmt_price(tp_price)} (+{tp_pct:.2f}%)\n"
            f"Score: {score:.1f}"
        )
        await self._send(msg)

    async def notify_exit(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        pnl: float,
        r_multiple: float,
        reason: str,
        balance: float,
    ) -> None:
        """Send position exit alert."""
        icon = "\u2705" if pnl >= 0 else "\u274c"
        sign = "+" if pnl >= 0 else ""

        msg = (
            f"{icon} *CLOSED {symbol} {direction}*\n"
            f"Exit: ${_fmt_price(exit_price)} ({reason})\n"
            f"PnL: {sign}${pnl:.2f} ({sign}{r_multiple:.1f}R)\n"
            f"Balance: ${balance:.2f}"
        )
        await self._send(msg)

    async def notify_trailing(
        self,
        symbol: str,
        old_sl: float,
        new_sl: float,
        state: str,
    ) -> None:
        """Send trailing stop update alert."""
        label = "Break-Even" if state == "break_even" else f"Trailing ({state})"
        msg = (
            f"\U0001f6e1 *{symbol}* SL moved\n"
            f"{label}: ${_fmt_price(old_sl)} \u2192 ${_fmt_price(new_sl)}"
        )
        await self._send(msg)

    async def notify_daily_summary(
        self,
        trades: int,
        wins: int,
        losses: int,
        pnl: float,
        balance: float,
    ) -> None:
        """Send daily summary."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sign = "+" if pnl >= 0 else ""
        pnl_pct = pnl / (balance - pnl) * 100 if (balance - pnl) > 0 else 0

        msg = (
            f"\U0001f4ca *Daily Summary ({date_str})*\n"
            f"Trades: {trades} (W: {wins} / L: {losses})\n"
            f"PnL: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)\n"
            f"Balance: ${balance:.2f}"
        )
        await self._send(msg)

    # ── Command Handlers ────────────────────────────────

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._scanner or not self._scanner.state:
            await update.message.reply_text("Bot not ready yet.")
            return

        s = self._scanner.state
        pnl_pct = (
            s.daily_pnl / s.initial_balance * 100
            if s.initial_balance > 0 else 0
        )
        sign = "+" if s.daily_pnl >= 0 else ""

        msg = (
            f"\U0001f4cb *Status*\n"
            f"Balance: ${s.balance:.2f}\n"
            f"Daily PnL: {sign}${s.daily_pnl:.2f} ({sign}{pnl_pct:.2f}%)\n"
            f"Positions: {len(s.positions)}/{self._scanner.risk_manager.max_positions}\n"
            f"Total Trades: {s.total_trades}\n"
            f"Active Symbols: {len(s.active_symbols)}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        if not self._scanner or not self._scanner.state:
            await update.message.reply_text("Bot not ready yet.")
            return

        positions = self._scanner.state.positions
        if not positions:
            await update.message.reply_text("No open positions.")
            return

        lines = ["\U0001f4c2 *Open Positions*\n"]
        for sym, pos in positions.items():
            lines.append(
                f"*{sym}* {pos.direction.value}\n"
                f"  Entry: ${_fmt_price(pos.entry_price)}\n"
                f"  SL: ${_fmt_price(pos.sl_price)} | TP: ${_fmt_price(pos.tp_price)}\n"
                f"  Size: {pos.size:.4f}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trades command."""
        if not self._scanner or not self._scanner.db:
            await update.message.reply_text("Bot not ready yet.")
            return

        trades = self._scanner.db.get_today_trades()
        if not trades:
            await update.message.reply_text("No trades today.")
            return

        lines = [f"\U0001f4c4 *Today's Trades* ({len(trades)})\n"]
        for t in trades[-10:]:  # last 10
            pnl = t.get("pnl") or 0
            sign = "+" if pnl >= 0 else ""
            reason = t.get("exit_reason") or "OPEN"
            lines.append(
                f"{t['symbol']} {t['direction']} | {reason} | {sign}${pnl:.2f}"
            )

        if len(trades) > 10:
            lines.append(f"\n... and {len(trades) - 10} more")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        msg = (
            "\U0001f916 *Scanner Bot Commands*\n\n"
            "/status - Balance, PnL, positions summary\n"
            "/positions - Open position details\n"
            "/trades - Today's trade history\n"
            "/help - Show this message"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    # ── Internal ────────────────────────────────────────

    async def _send(self, text: str) -> None:
        """Send a message to the configured chat. Failures are logged, not raised."""
        if not self._bot or not self._chat_id:
            return
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("Telegram send failed")
