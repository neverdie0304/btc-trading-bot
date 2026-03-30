"""멀티 코인 스캐너 봇 오케스트레이터.

여러 심볼을 동시에 모니터링하고, 시그널 탐지/우선순위/리스크 관리를 수행한다.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from binance import AsyncClient

from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, SETTINGS
from strategy.position import create_position, Position
from strategy.signals import Signal
from live.state import LiveState
from live.logger_db import TradeLogger
from live.executor import PaperExecutor, LiveExecutor

from scanner.config import SCANNER_SETTINGS
from scanner.symbol_selector import fetch_top_symbols, fetch_symbol_info, SymbolInfo
from scanner.candle_store import CandleStore
from scanner.multi_stream import MultiStream
from scanner.signal_engine import SignalEngine, SignalEvent
from scanner.prioritizer import rank_signals
from scanner.risk_manager import RiskManager
from scanner.scanner_state import ScannerState, SymbolState
from live.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

# 5분봉 마감 후 시그널 배치 대기 시간 (모든 심볼 수집)
_BATCH_DELAY_SEC = 2.0


def _fmt_price(price: float) -> str:
    """가격을 자릿수에 맞게 포맷한다 (저가 코인도 정밀하게 표시)."""
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    else:
        return f"{price:.8f}"


class ScannerBot:
    """멀티 코인 스캐너 봇."""

    def __init__(
        self,
        mode: str = "paper",
        capital: float = 10000,
    ) -> None:
        self.mode = mode
        self.initial_capital = capital
        self.interval = SETTINGS["interval"]

        # 비동기 초기화
        self.client: AsyncClient | None = None
        self.stream: MultiStream | None = None
        self.candle_store: CandleStore | None = None
        self.signal_engine: SignalEngine | None = None
        self.risk_manager: RiskManager | None = None
        self.state: ScannerState | None = None
        self.db: TradeLogger | None = None

        # 심볼 정보
        self.symbol_info: dict[str, SymbolInfo] = {}

        # 심볼별 executor
        self._executors: dict[str, PaperExecutor | LiveExecutor] = {}
        # 심볼별 LiveState (executor가 요구하는 인터페이스)
        self._live_states: dict[str, LiveState] = {}

        # 시그널 배치 수집
        self._signal_batch: list[SignalEvent] = []
        self._batch_task: asyncio.Task | None = None

        # 백그라운드 태스크
        self._symbol_refresh_task: asyncio.Task | None = None
        self._daily_reset_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._shutdown = False

        # Telegram
        self.telegram = TelegramNotifier()
        self.telegram.set_scanner(self)

    async def start(self) -> None:
        """스캐너 봇을 초기화하고 실행한다."""
        logger.info("=" * 60)
        logger.info("  Multi-Coin Signal Scanner")
        logger.info("  Mode: %s | Capital: $%.2f", self.mode.upper(), self.initial_capital)
        logger.info("  Max Positions: %d | Interval: %s",
                     SCANNER_SETTINGS["max_concurrent_positions"], self.interval)
        logger.info("=" * 60)

        # 1. DB
        self.db = TradeLogger(db_path="data/scanner_trades.db")

        # 2. 상태 복원 또는 생성
        self.state = ScannerState()
        restored = self.state.restore_from_db("data/scanner_trades.db")
        if restored and self.state.positions:
            logger.info(
                "State restored: balance=$%.2f, positions=%d",
                self.state.balance, len(self.state.positions),
            )
        else:
            self.state.balance = self.initial_capital
            self.state.initial_balance = self.initial_capital

        # 3. Binance 클라이언트
        self.client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)

        # 3.5 실제 잔고 조회 (live 모드)
        if self.mode == "live":
            await self._fetch_real_balance()

        # 4. 심볼 선정
        symbols = await fetch_top_symbols(self.client)
        if not symbols:
            logger.error("No symbols meet volume criteria. Exiting.")
            return

        self.state.active_symbols = symbols
        self.symbol_info = await fetch_symbol_info(self.client, symbols)

        # 5. 캔들 스토어 초기화
        self.candle_store = CandleStore()
        logger.info("Loading initial candles (%d symbols)...", len(symbols))
        await self.candle_store.load_all(self.client, symbols, self.interval)

        # 6. 시그널 엔진 & 리스크 매니저
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager(self.state.balance)

        # 7. 포지션 있는 심볼의 executor 생성
        for sym in list(self.state.positions.keys()):
            self._ensure_executor(sym)

        # 8. 통합 WebSocket 시작
        self.stream = MultiStream(
            symbols=symbols,
            interval=self.interval,
            on_candle_closed=self._on_candle_closed,
            on_price_update=self._on_price_update,
        )

        # 9. 백그라운드 태스크
        self._symbol_refresh_task = asyncio.create_task(self._symbol_refresh_loop())
        self._daily_reset_task = asyncio.create_task(self._daily_reset_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # 10. Telegram bot
        if self.telegram.enabled:
            await self.telegram.start()

        # 11. WebSocket 시작
        logger.info("Starting WebSocket... (%d symbols)", len(symbols))
        await self.stream.start()

        # 메인 루프 — stream._task가 끝날 때까지 대기
        try:
            while not self._shutdown:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """스캐너를 안전하게 종료한다."""
        logger.info("Shutting down scanner...")
        self._shutdown = True

        if self.stream:
            await self.stream.stop()

        for task in [self._symbol_refresh_task, self._daily_reset_task,
                     self._heartbeat_task, self._batch_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 상태 저장
        if self.state:
            self.state.save_to_db("data/scanner_trades.db")
            logger.info("State saved")

        if self.client:
            await self.client.close_connection()

        if self.db:
            self.db.close()

        if self.telegram.enabled:
            await self.telegram.stop()

        self._print_summary()
        logger.info("Scanner shutdown complete")

    # ── 콜백 ─────────────────────────────────────────────

    async def _on_candle_closed(self, symbol: str, kline: dict) -> None:
        """캔들 확정 콜백.

        CandleStore 업데이트 → 시그널 체크 → 배치 수집.
        """
        if self._shutdown:
            return

        # 캔들 스토어 업데이트
        df = self.candle_store.update_candle(symbol, kline)
        if df is None:
            return

        self.state.increment_candle(symbol)

        # 포지션이 있는 심볼: SL/TP 체크
        if symbol in self.state.positions:
            if self.mode == "paper":
                await self._check_paper_sl_tp(symbol, df)
            elif self.mode == "live":
                await self._check_live_positions()

        # 시그널 체크
        sym_state = self.state.get_symbol_state(symbol)
        candles_since = sym_state.candle_count - sym_state.last_loss_candle_count

        event = self.signal_engine.process_candle(
            symbol=symbol,
            df=df,
            consecutive_losses=sym_state.consecutive_losses,
            candles_since_last_loss=candles_since,
        )

        if event:
            self._signal_batch.append(event)
            # 배치 타이머 시작/리셋
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_signal_batch())

    async def _on_price_update(self, symbol: str, kline: dict) -> None:
        """실시간 가격 업데이트 → SL 히트 체크 / 미체결 LIMIT TP 도달 취소."""
        if self._shutdown or symbol not in self.state.positions:
            return

        price = float(kline["c"])
        pos = self.state.positions[symbol]

        # 라이브 모드: 미체결 LIMIT 주문 — 가격이 TP에 도달하면 즉시 취소
        if self.mode == "live":
            ls = self._live_states.get(symbol)
            if ls and not ls.entry_filled:
                tp_hit = False
                if pos.direction == Signal.LONG and price >= pos.tp_price:
                    tp_hit = True
                elif pos.direction == Signal.SHORT and price <= pos.tp_price:
                    tp_hit = True
                if tp_hit:
                    executor = self._executors.get(symbol)
                    if executor:
                        logger.info("[LIVE] %s price reached TP without fill, cancelling LIMIT", symbol)
                        await executor.cancel_entry_limit()
                        self.state.positions.pop(symbol, None)
                        self.state.save_to_db("data/scanner_trades.db")
                return

        # Paper 모드: SL 히트 체크
        if self.mode == "paper":
            hit = False
            if pos.direction == Signal.LONG and price <= pos.sl_price:
                hit = True
            elif pos.direction == Signal.SHORT and price >= pos.sl_price:
                hit = True

            if hit:
                executor = self._executors.get(symbol)
                live_state = self._live_states.get(symbol)
                bal_before = live_state.balance if live_state else 0
                if executor:
                    await executor.close_position("SL", price)
                pnl = (live_state.balance - bal_before) if live_state else 0
                if symbol in self.state.positions:
                    self.state.close_position(symbol, pnl)
                    self.risk_manager.update_daily_pnl(pnl)
                    self.state.save_to_db("data/scanner_trades.db")
                    logger.info("[EXIT] %s SL hit PnL=$%.2f | Balance=$%.2f", symbol, pnl, self.state.balance)
                    risk_amount = pos.r_unit * pos.size
                    r_mult = pnl / risk_amount if risk_amount else 0
                    await self.telegram.notify_exit(
                        symbol=symbol, direction=pos.direction.value,
                        exit_price=price, pnl=pnl, r_multiple=r_mult,
                        reason="SL", balance=self.state.balance,
                    )

    # ── 시그널 배치 처리 ─────────────────────────────────

    async def _process_signal_batch(self) -> None:
        """시그널 배치를 모아서 우선순위 결정 후 진입."""
        await asyncio.sleep(_BATCH_DELAY_SEC)

        if not self._signal_batch:
            return

        batch = self._signal_batch.copy()
        self._signal_batch.clear()

        # 가용 슬롯 계산
        available = self.risk_manager.available_slots(len(self.state.positions))
        if available <= 0:
            logger.info("[BATCH] No available slots (positions %d/%d)",
                       len(self.state.positions), self.risk_manager.max_positions)
            return

        # 우선순위 랭킹
        existing_symbols = set(self.state.positions.keys())
        ranked = rank_signals(batch, available, existing_symbols)

        # 진입 실행
        for event in ranked:
            if not self.risk_manager.can_open(
                event.symbol, self.state.positions, self.state.balance,
            ):
                continue

            await self._open_position(event)

    async def _open_position(self, event: SignalEvent) -> None:
        """시그널 이벤트 기반으로 포지션을 연다."""
        symbol = event.symbol

        # 라이브 모드: 진입 전 실제 잔고 조회
        if self.mode == "live":
            await self._sync_balance()

        capital = self.risk_manager.get_capital_for_trade(
            self.state.balance, self.state.positions,
        )

        position = create_position(
            direction=event.direction,
            entry_price=event.close,
            atr=event.atr,
            capital=capital,
            entry_time=str(event.timestamp),
        )

        # executor 확보
        executor = self._ensure_executor(symbol)

        success = await executor.open_position(position)
        if success:
            self.state.open_position(symbol, position)
            self.state.save_to_db("data/scanner_trades.db")

            if self.mode == "paper":
                logger.info(
                    ">>> [ENTRY] %s %s @ %s | Size=%.4f | SL=%s | TP=%s | Score=%.3f",
                    symbol, event.direction.value, _fmt_price(event.close),
                    position.size, _fmt_price(position.sl_price),
                    _fmt_price(position.tp_price), event.score,
                )
                await self.telegram.notify_entry(
                    symbol=symbol,
                    direction=event.direction.value,
                    entry_price=event.close,
                    size=position.size,
                    sl_price=position.sl_price,
                    tp_price=position.tp_price,
                    score=event.score,
                )
            else:
                # 라이브: LIMIT 주문 배치됨 (아직 미체결)
                logger.info(
                    ">>> [ORDER] %s %s LIMIT @ %s | Size=%.4f | SL=%s | TP=%s | Score=%.3f",
                    symbol, event.direction.value, _fmt_price(event.close),
                    position.size, _fmt_price(position.sl_price),
                    _fmt_price(position.tp_price), event.score,
                )

    # ── Paper 모드 SL/TP ─────────────────────────────────

    async def _check_paper_sl_tp(self, symbol: str, df: pd.DataFrame) -> None:
        """Paper 모드에서 봉의 H/L로 SL/TP 히트를 체크한다."""
        pos = self.state.positions.get(symbol)
        if not pos:
            return

        latest = df.iloc[-1]
        high = latest["high"]
        low = latest["low"]

        executor = self._executors.get(symbol)
        if not executor:
            return

        live_state = self._live_states.get(symbol)

        # SL/TP 체크
        bal_before = live_state.balance if live_state else 0
        hit = await executor.check_sl_tp(high, low)
        if hit and symbol in self.state.positions:
            # executor._record_exit가 LiveState.balance에 PnL을 반영했으므로 차이로 추출
            pnl = (live_state.balance - bal_before) if live_state else 0
            self.state.close_position(symbol, pnl)
            self.risk_manager.update_daily_pnl(pnl)
            self.state.save_to_db("data/scanner_trades.db")
            logger.info("[EXIT] %s PnL=$%.2f | Balance=$%.2f", symbol, pnl, self.state.balance)
            risk_amount = pos.r_unit * pos.size
            r_mult = pnl / risk_amount if risk_amount else 0
            await self.telegram.notify_exit(
                symbol=symbol, direction=pos.direction.value,
                exit_price=float(latest["close"]), pnl=pnl, r_multiple=r_mult,
                reason="TP" if pnl > 0 else "SL", balance=self.state.balance,
            )

    # ── Executor 관리 ────────────────────────────────────

    def _ensure_executor(self, symbol: str) -> PaperExecutor | LiveExecutor:
        """심볼의 executor를 가져오거나 생성한다."""
        if symbol in self._executors:
            return self._executors[symbol]

        # LiveState 생성 (executor 인터페이스 요구)
        live_state = LiveState(
            symbol=symbol,
            balance=self.state.balance,
            initial_balance=self.state.initial_balance,
        )

        # 복원된 포지션이 있으면 할당
        if symbol in self.state.positions:
            live_state.position = self.state.positions[symbol]
            live_state.entry_filled = True

        self._live_states[symbol] = live_state

        if self.mode == "live":
            info = self.symbol_info.get(symbol)
            executor = LiveExecutor(
                client=self.client,
                state=live_state,
                db=self.db,
                symbol=symbol,
            )
            # tick_size, step_size 설정
            if info:
                executor.tick_size = info.tick_size
                executor.step_size = info.step_size
            self._executors[symbol] = executor
        else:
            executor = PaperExecutor(
                state=live_state,
                db=self.db,
                symbol=symbol,
            )
            self._executors[symbol] = executor

        return executor

    # ── 잔고 조회 ────────────────────────────────────────

    async def _fetch_real_balance(self) -> None:
        """Binance Futures 실제 잔고를 조회한다."""
        try:
            real_balance = 0.0
            account = await self.client.futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] in ("USDT", "USDC"):
                    bal = float(asset.get("walletBalance", 0))
                    if bal > 0:
                        real_balance += bal
                        logger.info("%s balance: $%.2f", asset["asset"], bal)

            logger.info("Binance total balance: $%.2f", real_balance)
            if real_balance > 0:
                self.initial_capital = real_balance
                self.state.balance = real_balance
                self.state.initial_balance = real_balance
        except Exception:
            logger.exception("Balance query failed, using config value")

    async def _sync_balance(self) -> float:
        """Binance에서 실제 잔고를 가져와 state를 업데이트한다."""
        try:
            real_balance = 0.0
            account = await self.client.futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] in ("USDT", "USDC"):
                    bal = float(asset.get("walletBalance", 0))
                    if bal > 0:
                        real_balance += bal
            if real_balance > 0:
                old = self.state.balance
                self.state.balance = real_balance
                if abs(old - real_balance) > 0.01:
                    logger.info("[SYNC] Balance: $%.2f → $%.2f", old, real_balance)
            return real_balance
        except Exception:
            logger.exception("Balance sync failed")
            return self.state.balance

    async def _check_live_positions(self) -> None:
        """라이브 모드에서 거래소 포지션 상태를 확인하고 청산된 포지션을 처리한다."""
        if self.mode != "live" or not self.state.positions:
            return

        for symbol in list(self.state.positions.keys()):
            try:
                # 진입 LIMIT이 아직 미체결인지 확인
                ls = self._live_states.get(symbol)
                executor = self._executors.get(symbol)
                if ls and not ls.entry_filled:
                    # LIMIT 주문 체결 여부를 REST로 확인
                    if executor and hasattr(executor, 'check_pending_entry'):
                        filled = await executor.check_pending_entry()
                        if filled:
                            # 체결됨 → 텔레그램 알림
                            pos = self.state.positions[symbol]
                            logger.info(
                                ">>> [ENTRY FILLED] %s %s @ %s | Size=%.4f",
                                symbol, pos.direction.value,
                                _fmt_price(pos.entry_price), pos.size,
                            )
                            await self.telegram.notify_entry(
                                symbol=symbol,
                                direction=pos.direction.value,
                                entry_price=pos.entry_price,
                                size=pos.size,
                                sl_price=pos.sl_price,
                                tp_price=pos.tp_price,
                                score=0,
                            )
                        else:
                            # 아직 미체결 → 2봉(10분) 이상 지나면 취소
                            candle_count = ls.candle_count if hasattr(ls, 'candle_count') else 0
                            entry_candle = ls.entry_limit_candle if hasattr(ls, 'entry_limit_candle') else 0
                            if candle_count - entry_candle >= 2:
                                logger.info("[LIVE] %s entry LIMIT expired, cancelling", symbol)
                                await executor.cancel_entry_limit()
                                self.state.positions.pop(symbol, None)
                                self.state.save_to_db("data/scanner_trades.db")
                    continue

                # 체결된 포지션: 거래소에서 아직 열려있는지 확인
                exchange_positions = await self.client.futures_position_information(
                    symbol=symbol,
                )
                amt = 0.0
                for ep in exchange_positions:
                    amt = float(ep.get("positionAmt", 0))
                    if amt != 0:
                        break

                if amt == 0:
                    # 거래소에서 포지션이 닫혔음 → 실제 잔고로 PnL 계산
                    pos = self.state.positions[symbol]
                    balance_before = self.state.balance
                    real_balance = await self._sync_balance()
                    real_pnl = real_balance - balance_before

                    self.state.close_position(symbol, 0)  # 수동 PnL 안 더함
                    self.state.balance = real_balance      # 실제 잔고로 덮어쓰기
                    self.state.daily_pnl += real_pnl
                    self.risk_manager.update_daily_pnl(real_pnl)
                    self.state.save_to_db("data/scanner_trades.db")

                    # DB에 거래 기록 업데이트
                    r_mult = real_pnl / (pos.r_unit * pos.size) if pos.r_unit * pos.size > 0 else 0
                    reason = "TP" if real_pnl > 0 else "SL"

                    if self.db and ls:
                        if ls.trades_today:
                            self.db.update_trade_exit(
                                trade_id=ls.trades_today[-1].trade_db_id,
                                exit_price=0,
                                exit_reason=reason,
                                pnl=real_pnl,
                                commission=0,
                                r_multiple=r_mult,
                            )

                    logger.info(
                        "[LIVE EXIT] %s %s closed on exchange | Real PnL=$%.2f (%.1fR) | Balance=$%.2f",
                        symbol, pos.direction.value, real_pnl, r_mult, real_balance,
                    )
                    await self.telegram.notify_exit(
                        symbol=symbol, direction=pos.direction.value,
                        exit_price=0, pnl=real_pnl, r_multiple=r_mult,
                        reason=reason, balance=real_balance,
                    )

                    # executor/live_state 정리
                    if executor:
                        executor._clear_position_state()

            except Exception:
                logger.exception("[LIVE] Position check failed: %s", symbol)

    # ── 백그라운드 태스크 ─────────────────────────────────

    async def _symbol_refresh_loop(self) -> None:
        """주기적으로 심볼 목록을 갱신한다."""
        interval_hours = SCANNER_SETTINGS["symbol_refresh_interval_hours"]
        while not self._shutdown:
            try:
                await asyncio.sleep(interval_hours * 3600)
                if self._shutdown:
                    break

                new_symbols = await fetch_top_symbols(self.client)
                if not new_symbols:
                    continue

                old_set = set(self.state.active_symbols)
                new_set = set(new_symbols)

                added = new_set - old_set
                removed = old_set - new_set

                # 포지션 있는 심볼은 제거하지 않음
                removed -= set(self.state.positions.keys())

                if added:
                    logger.info("Symbols added: %s", added)
                    for sym in added:
                        await self.candle_store.load_initial(
                            self.client, sym, self.interval,
                        )

                if removed:
                    logger.info("Symbols removed: %s", removed)
                    for sym in removed:
                        self.candle_store.remove_symbol(sym)

                self.state.active_symbols = new_symbols

                # WebSocket 스트림 업데이트
                # 포지션 있는 심볼도 포함
                all_symbols = list(new_set | set(self.state.positions.keys()))
                self.stream.update_symbols(all_symbols)

                # 심볼 정보 갱신
                new_info = await fetch_symbol_info(self.client, list(added))
                self.symbol_info.update(new_info)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Symbol refresh error")

    async def _daily_reset_loop(self) -> None:
        """매일 00:00 UTC에 일일 통계를 초기화한다."""
        while not self._shutdown:
            try:
                now = datetime.now(timezone.utc)
                next_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if next_reset <= now:
                    next_reset += timedelta(days=1)

                wait = (next_reset - now).total_seconds()
                logger.info("Next daily reset in %.0f seconds", wait)
                await asyncio.sleep(wait)

                # Send daily summary before reset (at 00:00 UTC, "yesterday" is the completed day)
                if self.db:
                    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
                    day_trades = self.db.get_trades_by_date(yesterday)
                    wins = sum(1 for t in day_trades if (t.get("pnl") or 0) > 0)
                    losses = sum(1 for t in day_trades if (t.get("pnl") or 0) < 0)
                    await self.telegram.notify_daily_summary(
                        trades=len(day_trades), wins=wins, losses=losses,
                        pnl=self.state.daily_pnl, balance=self.state.balance,
                    )

                self.state.daily_pnl = 0.0
                self.risk_manager.reset_daily(self.state.balance)
                for ss in self.state.symbol_states.values():
                    ss.last_loss_candle_count = 0
                logger.info("Daily reset complete")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Daily reset error")
                await asyncio.sleep(60)

    async def _heartbeat_loop(self) -> None:
        """5분마다 상태를 출력하고 저장한다."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)

                # 라이브 모드: 실제 잔고 동기화 + 포지션 체크
                if self.mode == "live":
                    await self._sync_balance()
                    await self._check_live_positions()

                self._log_status()
                self.state.save_to_db("data/scanner_trades.db")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat 에러")

    # ── 로깅 ─────────────────────────────────────────────

    def _log_status(self) -> None:
        """현재 상태를 출력한다."""
        pnl_pct = (
            self.state.daily_pnl / self.state.initial_balance * 100
            if self.state.initial_balance > 0 else 0
        )

        pos_strs = []
        for sym, pos in self.state.positions.items():
            pos_strs.append(
                f"{sym} {pos.direction.value}@{_fmt_price(pos.entry_price)} "
                f"SL={_fmt_price(pos.sl_price)}"
            )
        pos_display = " | ".join(pos_strs) if pos_strs else "None"

        logger.info(
            "[STATUS] Balance: $%.2f | Daily PnL: $%.2f (%.2f%%) | "
            "Positions: %d/%d | Trades: %d | Symbols: %d | %s",
            self.state.balance, self.state.daily_pnl, pnl_pct,
            len(self.state.positions), self.risk_manager.max_positions,
            self.state.total_trades,
            len(self.state.active_symbols),
            pos_display,
        )

    def _print_summary(self) -> None:
        """세션 종료 요약."""
        logger.info("=" * 50)
        logger.info("  Scanner Session Summary")
        logger.info("=" * 50)
        logger.info("  Total Trades: %d", self.state.total_trades)
        logger.info("  Final Balance: $%.2f", self.state.balance)
        logger.info("  Daily PnL: $%.2f", self.state.daily_pnl)
        logger.info("  Open Positions: %d", len(self.state.positions))
        logger.info("=" * 50)
