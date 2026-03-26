"""라이브 트레이딩 실행 진입점.

사용법:
    # 페이퍼 트레이딩 (주문 실행 안 함, WebSocket으로 시뮬레이션)
    python main_live.py --mode paper

    # 실제 트레이딩
    python main_live.py --mode live

    # 자본금 지정
    python main_live.py --mode paper --capital 1000
"""

import argparse
import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from binance import AsyncClient

from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, SETTINGS
from strategy.signals import (
    Signal, compute_indicators, check_long_conditions, check_short_conditions,
)
from strategy.filters import should_filter
from strategy.position import create_position
from live.candle_manager import CandleManager
from live.state import LiveState
from live.logger_db import TradeLogger
from live.kill_switch import KillSwitch
from live.executor import PaperExecutor, LiveExecutor

# 진입 LIMIT 주문이 N봉 이상 미체결이면 취소 (5분봉 × 2 = 10분)
ENTRY_LIMIT_TIMEOUT_CANDLES = 2

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/live_bot.log"),
    ],
)
logger = logging.getLogger(__name__)


class LiveBot:
    """라이브 트레이딩 봇 메인 클래스."""

    def __init__(
        self,
        mode: str = "paper",
        capital: float = 10000,
        symbol: str | None = None,
        shared_client: AsyncClient | None = None,
    ) -> None:
        self.mode = mode
        self.initial_capital = capital
        self.symbol = symbol or SETTINGS["symbol"]
        self.interval = SETTINGS["interval"]
        self._shared_client = shared_client  # 외부에서 전달받은 공유 클라이언트

        # 비동기 클라이언트는 start()에서 초기화
        self.client: AsyncClient | None = None
        self.candle_mgr: CandleManager | None = None
        self.executor: PaperExecutor | LiveExecutor | None = None
        self.state: LiveState | None = None
        self.db: TradeLogger | None = None
        self.kill_switch: KillSwitch | None = None

        self._user_data_task: asyncio.Task | None = None
        self._daily_reset_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._shutdown = False

    async def start(self) -> None:
        """봇을 초기화하고 실행한다."""
        logger.info("=" * 60)
        logger.info("  BTC 5min Live Trading Bot")
        logger.info("  Mode: %s | Symbol: %s | Capital: $%.2f",
                     self.mode.upper(), self.symbol, self.initial_capital)
        logger.info("=" * 60)

        # 1. DB 초기화
        self.db = TradeLogger()

        # 2. 상태 복원 또는 생성
        restored_state = LiveState.restore_from_db(self.db, symbol=self.symbol)
        if restored_state and restored_state.position:
            logger.info("이전 세션 상태 복원")
            self.state = restored_state
        else:
            self.state = LiveState(
                symbol=self.symbol,
                balance=self.initial_capital,
                initial_balance=self.initial_capital,
                session_start=datetime.now(timezone.utc).isoformat(),
                is_active=True,
            )
            logger.info("새 세션 시작")

        # 3. Binance 클라이언트 (공유 또는 새로 생성)
        if self._shared_client:
            self.client = self._shared_client
        else:
            self.client = await AsyncClient.create(
                BINANCE_API_KEY, BINANCE_API_SECRET
            )

        # 3.5. 실제 Binance Futures 잔고 조회
        try:
            real_balance = 0.0
            account = await self.client.futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] in ("USDT", "USDC"):
                    bal = float(asset.get("walletBalance", 0))
                    if bal > 0:
                        real_balance += bal
                        logger.info("%s 잔고: $%.2f", asset["asset"], bal)

            logger.info("Binance 총 잔고: $%.2f", real_balance)
            if real_balance > 0:
                self.initial_capital = real_balance
                self.state.balance = real_balance
                self.state.initial_balance = real_balance
            else:
                logger.warning("잔고가 0입니다. 설정값 $%.2f 사용", self.initial_capital)
        except Exception:
            logger.exception("잔고 조회 실패, 설정값 $%.2f 사용", self.initial_capital)

        # 4. Executor
        if self.mode == "live":
            self.executor = LiveExecutor(
                client=self.client, state=self.state, db=self.db,
                symbol=self.symbol,
            )
            await self.executor.fetch_symbol_info()
            await self.executor.setup_leverage()
            await self.executor.sync_exchange_state()
            if self.state.balance > 0:
                self.state.initial_balance = self.state.balance
                self.initial_capital = self.state.balance
        else:
            self.executor = PaperExecutor(
                state=self.state, db=self.db, symbol=self.symbol,
            )

        # 5. Kill Switch
        self.kill_switch = KillSwitch(self.state)

        # 6. CandleManager
        self.candle_mgr = CandleManager(
            client=self.client,
            symbol=self.symbol,
            interval=self.interval,
            on_candle_closed=self._on_candle_closed,
            on_price_update=self._on_price_update,
        )

        # 초기 캔들 로드
        await self.candle_mgr.load_initial_candles(count=150)

        # 7. 백그라운드 태스크
        self._daily_reset_task = asyncio.create_task(self._daily_reset_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # 8. 라이브 모드: user_data stream
        if self.mode == "live":
            self._user_data_task = asyncio.create_task(self._user_data_loop())

        # 9. WebSocket 시작
        logger.info("WebSocket 시작... (Ctrl+C로 종료)")
        await self.candle_mgr.start_websocket()

        # 10. WebSocket 루프가 끝날 때까지 대기
        if self.candle_mgr._ws_task:
            await self.candle_mgr._ws_task

    async def stop(self) -> None:
        """봇을 안전하게 종료한다."""
        logger.info("봇 종료 중...")
        self._shutdown = True

        if self.candle_mgr:
            await self.candle_mgr.stop()

        # 백그라운드 태스크 취소
        for task in [self._user_data_task, self._daily_reset_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 상태 저장
        if self.state and self.db:
            self.state.save_to_db(self.db)
            logger.info("상태 저장 완료")

        # 클라이언트 종료 (공유 클라이언트가 아닌 경우에만)
        if self.client and not self._shared_client:
            await self.client.close_connection()

        if self.db:
            self.db.close()

        self._print_session_summary()
        logger.info("봇 종료 완료")

    async def _on_price_update(self, price: float) -> None:
        """실시간 가격 업데이트 → 트레일링 스탑 즉시 반영."""
        if self._shutdown or self.state.position is None:
            return
        # 단일 SL/TP 전략 — 실시간 가격 업데이트에서 별도 처리 불필요
        return

    async def _on_candle_closed(self, df: pd.DataFrame) -> None:
        """봉 확정 이벤트 핸들러. 핵심 트레이딩 로직."""
        if self._shutdown:
            return

        self.state.candle_count += 1
        timestamp = df.index[-1]

        # ── 진입 LIMIT 상태 확인 (live 전용) ─────────────────────────
        # WebSocket 누락·레이스컨디션 방어: 매 봉마다 REST API로 직접 확인
        if (self.mode == "live"
                and isinstance(self.executor, LiveExecutor)
                and self.state.position is not None
                and not self.state.entry_filled
                and self.state.entry_order_id):

            filled = await self.executor.check_pending_entry()
            if filled:
                # 체결 확인 → SL/TP 세팅 완료, 상태 저장
                self.state.save_to_db(self.db)
                self._log_status()
                # 이번 봉은 포지션 관리만 완료, 신규 진입 불필요 → return
                return

            # 여전히 미체결 → 타임아웃 체크
            candles_since = self.state.candle_count - self.state.entry_limit_candle
            if candles_since >= ENTRY_LIMIT_TIMEOUT_CANDLES:
                logger.info(
                    "진입 LIMIT 주문 %d봉 미체결 → 취소 (timeout)", candles_since
                )
                await self.executor.cancel_entry_limit()
                self.state.save_to_db(self.db)
                return

        # Kill switch 체크
        should_stop, reason = self.kill_switch.check(
            last_ws_time=self.candle_mgr.last_msg_time if self.candle_mgr else None,
        )
        if should_stop:
            logger.critical("Kill switch 발동: %s", reason)
            await self.executor.emergency_close()
            return

        # 지표 계산
        try:
            df_ind = compute_indicators(df)
        except Exception:
            logger.exception("지표 계산 실패")
            return

        latest = df_ind.iloc[-1]
        prev = df_ind.iloc[-2] if len(df_ind) >= 2 else None

        if prev is None:
            return

        close = latest["close"]
        high = latest["high"]
        low = latest["low"]

        # 1. 포지션이 있으면: SL/TP 체크 (paper만)
        if self.state.position:
            if not self.state.entry_filled:
                # LIMIT 주문 대기 중: 포지션 관리 스킵
                pass
            elif self.mode == "paper":
                # Paper: 봉의 H/L로 SL/TP 체크
                hit = await self.executor.check_sl_tp(high, low)
                if hit:
                    self.state.save_to_db(self.db)
                    self._log_status()
                    return

        # 2. 필터 체크
        atr_val = latest.get("atr", 0)
        atr_med = latest.get("atr_median", 0)
        if pd.isna(atr_val):
            atr_val = 0
        if pd.isna(atr_med):
            atr_med = 0

        filtered = should_filter(
            timestamp=timestamp,
            atr=atr_val,
            atr_median=atr_med,
            consecutive_losses=self.state.consecutive_losses,
            candles_since_last_loss=self.state.candles_since_last_loss(),
        )

        if filtered:
            return

        # 3. 시그널 체크 (현재 봉 + 이전 봉 기준)
        signal = Signal.NO_SIGNAL
        if not (pd.isna(latest.get("atr")) or pd.isna(latest.get("rsi")) or pd.isna(latest.get("volume_ratio"))):
            if check_long_conditions(latest, prev):
                signal = Signal.LONG
            elif SETTINGS.get("short_enabled", True) and check_short_conditions(latest, prev):
                signal = Signal.SHORT

        if signal == Signal.NO_SIGNAL:
            return

        # 4. 포지션이 있으면 무시 (또는 반대 시그널 처리)
        if self.state.position:
            if not self.state.entry_filled:
                # LIMIT 주문 대기 중: 새 시그널 무시
                return
            if SETTINGS["reverse_on_opposite_signal"]:
                if self.state.position.direction != signal:
                    await self.executor.close_position("REVERSE", close)
                else:
                    return
            else:
                return

        # 5. 새 포지션 진입
        if not self.state.position and pd.notna(latest.get("atr")):
            position = create_position(
                direction=signal,
                entry_price=close,
                atr=latest["atr"],
                capital=self.state.balance,
                entry_time=str(timestamp),
            )

            success = await self.executor.open_position(position)
            if success:
                self.state.save_to_db(self.db)
                logger.info(
                    ">>> %s 진입: @ %.2f, size=%.6f, SL=%.2f, TP=%.2f",
                    signal.value, close, position.size,
                    position.sl_price, position.tp_price,
                )
                self._log_status()

    async def _user_data_loop(self) -> None:
        """WebSocket user_data stream으로 주문 이벤트를 수신한다."""
        from binance.streams import BinanceSocketManager

        bm = BinanceSocketManager(self.client)
        while not self._shutdown:
            try:
                ts = bm.futures_user_socket()
                async with ts as stream:
                    logger.info("User data stream 연결")
                    while not self._shutdown:
                        msg = await asyncio.wait_for(stream.recv(), timeout=600)

                        event_type = msg.get("e", "")
                        if event_type == "ORDER_TRADE_UPDATE":
                            if isinstance(self.executor, LiveExecutor):
                                await self.executor.handle_order_update(msg)
                                self.state.save_to_db(self.db)

                        elif event_type == "ACCOUNT_UPDATE":
                            # 잔고 업데이트
                            for balance in msg.get("a", {}).get("B", []):
                                if balance["a"] == "USDC":
                                    self.state.balance = float(balance["wb"])

            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                logger.warning("User data stream 타임아웃, 재연결...")
            except Exception:
                logger.exception("User data stream 에러, 5초 후 재연결")
                await asyncio.sleep(5)

    async def _daily_reset_loop(self) -> None:
        """매일 00:00 UTC에 일일 통계를 초기화한다."""
        while not self._shutdown:
            try:
                now = datetime.now(timezone.utc)
                # 다음 00:00 UTC까지 대기
                next_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if next_reset <= now:
                    next_reset += timedelta(days=1)

                wait_seconds = (next_reset - now).total_seconds()
                logger.info("다음 일일 초기화: %.0f초 후", wait_seconds)
                await asyncio.sleep(wait_seconds)

                # 일별 요약 저장
                today = now.strftime("%Y-%m-%d")
                wins = sum(1 for t in self.state.trades_today if t.pnl > 0)
                losses = sum(1 for t in self.state.trades_today if t.pnl <= 0)
                self.db.update_daily_summary(
                    date=today,
                    trades_count=len(self.state.trades_today),
                    wins=wins,
                    losses=losses,
                    total_pnl=self.state.daily_pnl,
                    max_drawdown=0,
                    ending_balance=self.state.balance,
                )

                # 일일 초기화
                self.state.reset_daily()
                self.state.initial_balance = self.state.balance
                self.state.is_active = True  # kill switch 해제

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("일일 초기화 에러")
                await asyncio.sleep(60)

    async def _heartbeat_loop(self) -> None:
        """주기적으로 상태를 확인하고 로그를 출력한다."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # 5분마다

                if not self.state.is_active:
                    logger.info("[HEARTBEAT] 봇 비활성 상태")
                    continue

                # 상태 로그
                self._log_status()

                # 상태 저장
                self.state.save_to_db(self.db)

                # Live 모드: 거래소 상태 동기화
                if self.mode == "live" and isinstance(self.executor, LiveExecutor):
                    await self.executor.sync_exchange_state()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat 에러")

    def get_status(self) -> dict:
        """현재 봇 상태를 dict로 반환한다."""
        pos_info = None
        if self.state and self.state.position:
            p = self.state.position
            pos_info = {
                "direction": p.direction.value,
                "entry_price": p.entry_price,
                "size": p.size,
                "sl_price": p.sl_price,
                "tp_price": p.tp_price,
            }

        return {
            "symbol": self.symbol,
            "mode": self.mode,
            "running": not self._shutdown,
            "balance": self.state.balance if self.state else 0,
            "initial_balance": self.state.initial_balance if self.state else 0,
            "daily_pnl": self.state.daily_pnl if self.state else 0,
            "trades_today": len(self.state.trades_today) if self.state else 0,
            "consecutive_losses": self.state.consecutive_losses if self.state else 0,
            "candle_count": self.state.candle_count if self.state else 0,
            "position": pos_info,
            "is_active": self.state.is_active if self.state else False,
        }

    def _log_status(self) -> None:
        """현재 상태를 로그로 출력한다."""
        pnl_pct = (self.state.daily_pnl / self.state.initial_balance * 100
                   if self.state.initial_balance > 0 else 0)
        pos_str = "없음"
        if self.state.position:
            pos_str = (
                f"{self.state.position.direction.value} @ "
                f"{self.state.position.entry_price:.2f} "
                f"SL={self.state.position.sl_price:.2f}"
            )

        logger.info(
            "[STATUS] Balance: $%.2f | Daily PnL: $%.2f (%.2f%%) | "
            "Trades: %d | Consec Loss: %d | Position: %s",
            self.state.balance, self.state.daily_pnl, pnl_pct,
            len(self.state.trades_today), self.state.consecutive_losses,
            pos_str,
        )

    def _print_session_summary(self) -> None:
        """세션 종료 시 요약을 출력한다."""
        if not self.state:
            return

        total_trades = len(self.state.trades_today)
        total_pnl = self.state.daily_pnl
        win_pct = (
            sum(1 for t in self.state.trades_today if t.pnl > 0) / max(total_trades, 1) * 100
        )

        logger.info("=" * 50)
        logger.info("  세션 요약")
        logger.info("=" * 50)
        logger.info("  거래 횟수: %d", total_trades)
        logger.info("  승률: %.1f%%", win_pct)
        logger.info("  일일 PnL: $%.2f", total_pnl)
        logger.info("  최종 잔고: $%.2f", self.state.balance)
        logger.info("=" * 50)


async def run(mode: str, capital: float, symbol: str | None) -> None:
    """봇을 실행한다."""
    bot = LiveBot(mode=mode, capital=capital, symbol=symbol)

    loop = asyncio.get_event_loop()

    def _signal_handler():
        logger.info("종료 신호 수신...")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("봇 에러 발생")
    finally:
        await bot.stop()


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(
        description="BTC 5min Live Trading Bot"
    )
    parser.add_argument(
        "--mode", type=str, choices=["paper", "live"], default="paper",
        help="트레이딩 모드 (기본값: paper)",
    )
    parser.add_argument(
        "--capital", type=float, default=SETTINGS["initial_capital"],
        help="초기 자본금 (기본값: settings)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="거래 페어 (기본값: settings의 symbol)",
    )

    args = parser.parse_args()

    # 심볼이 지정되지 않으면 사용자에게 입력 받기
    symbol = args.symbol
    if not symbol:
        default_symbol = SETTINGS["symbol"]
        user_input = input(f"거래할 심볼을 입력하세요 (기본값: {default_symbol}): ").strip().upper()
        symbol = user_input if user_input else default_symbol

    logger.info("Starting bot: mode=%s, symbol=%s, capital=%.2f",
                args.mode, symbol, args.capital)

    asyncio.run(run(args.mode, args.capital, symbol))


if __name__ == "__main__":
    main()
