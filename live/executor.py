"""주문 실행 모듈.

PaperExecutor: 페이퍼 트레이딩 (주문 없이 시뮬레이션)
LiveExecutor: 실제 Binance Futures 주문 실행
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from binance import AsyncClient

# Binance Futures 주문 상수
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

# 트레일링 SL STOP_LIMIT 슬리피지 허용 틱 수
_SLIPPAGE_TICKS = 3

from config.settings import SETTINGS
from strategy.position import Position, check_sl_tp_hit
from strategy.signals import Signal
from live.state import LiveState, LiveTrade
from live.logger_db import TradeLogger

logger = logging.getLogger(__name__)


def _round_price(price: float, tick_size: float = 0.10) -> float:
    """가격을 틱 사이즈에 맞춰 반올림한다."""
    return round(price / tick_size) * tick_size


def _round_qty(qty: float, step_size: float = 0.001) -> float:
    """수량을 스텝 사이즈에 맞춰 내림한다."""
    decimals = max(0, -int(math.log10(step_size)))
    return round(int(qty / step_size) * step_size, decimals)


def _close_side(direction: Signal) -> str:
    """포지션 청산 방향을 반환한다."""
    return SIDE_SELL if direction == Signal.LONG else SIDE_BUY


def _get_order_id(order: dict) -> str:
    """주문 응답에서 ID를 추출한다 (orderId 또는 algoId)."""
    return str(order.get("orderId") or order.get("algoId") or "unknown")


class BaseExecutor(ABC):
    """주문 실행자 기본 클래스."""

    def __init__(self, state: LiveState, db: TradeLogger, symbol: str | None = None) -> None:
        self.state = state
        self.db = db
        self.symbol = symbol or SETTINGS["symbol"]

    @abstractmethod
    async def open_position(self, position: Position) -> bool:
        """포지션을 연다."""
        ...

    @abstractmethod
    async def close_position(self, reason: str, price: float | None = None) -> bool:
        """포지션을 닫는다."""
        ...

    @abstractmethod
    async def update_sl_order(self, new_sl: float) -> bool:
        """SL 주문을 업데이트한다."""
        ...

    @abstractmethod
    async def cancel_all_orders(self) -> None:
        """모든 미체결 주문을 취소한다."""
        ...

    @abstractmethod
    async def emergency_close(self) -> None:
        """비상 청산 (kill switch)."""
        ...

    def _calculate_pnl(self, position: Position, exit_price: float, exit_reason: str) -> tuple[float, float, float]:
        """PnL, commission, r_multiple을 계산한다.

        수수료 모델:
          진입   LIMIT          → maker_fee (0%)
          익절   TAKE_PROFIT    → maker_fee (0%)
          손절   STOP_MARKET    → taker_fee (0.04%)
          트레일 STOP_LIMIT     → taker_fee (보수적)
          긴급   MARKET         → taker_fee
        """
        if position.direction == Signal.LONG:
            raw_pnl = (exit_price - position.entry_price) * position.size
        else:
            raw_pnl = (position.entry_price - exit_price) * position.size

        notional_entry = position.entry_price * position.size
        notional_exit = exit_price * position.size
        maker_fee = SETTINGS.get("maker_fee", 0.0)
        taker_fee = SETTINGS.get("taker_fee", 0.0004)

        entry_fee = notional_entry * maker_fee
        is_maker_exit = exit_reason == "TP"
        exit_fee = notional_exit * (maker_fee if is_maker_exit else taker_fee)
        commission = entry_fee + exit_fee
        slippage = notional_exit * SETTINGS["slippage_rate"]

        net_pnl = raw_pnl - commission - slippage
        r_multiple = raw_pnl / (position.r_unit * position.size) if position.r_unit > 0 else 0.0

        return net_pnl, commission, r_multiple

    def _record_exit(
        self,
        pos: Position,
        exit_price: float,
        reason: str,
        exit_order_id: str = "",
    ) -> None:
        """PnL 계산 + 잔고 업데이트 + DB 기록 + 승패 기록을 일괄 처리한다."""
        net_pnl, commission, r_multiple = self._calculate_pnl(pos, exit_price, reason)

        self.state.balance += net_pnl
        self.state.daily_pnl += net_pnl

        if net_pnl >= 0:
            self.state.record_win()
        else:
            self.state.record_loss()

        if self.state.trades_today:
            is_maker_exit = reason == "TP"
            self.db.update_trade_exit(
                trade_id=self.state.trades_today[-1].trade_db_id,
                exit_price=exit_price,
                exit_reason=reason,
                pnl=net_pnl,
                commission=commission,
                r_multiple=r_multiple,
                exit_order_id=exit_order_id,
                sl_fill_type="maker" if is_maker_exit else "taker",
            )

        logger.info(
            "Position closed: %s @ %g → %g, PnL=%.2f, R=%.2fR, reason=%s",
            pos.direction.value, pos.entry_price, exit_price,
            net_pnl, r_multiple, reason,
        )

    def _clear_position_state(self) -> None:
        """포지션 관련 상태를 모두 초기화한다."""
        self.state.position = None
        self.state.entry_order_id = None
        self.state.sl_order_id = None
        self.state.tp_order_id = None
        self.state.entry_filled = False

    def _cancel_pending_trade_db(self) -> None:
        """DB에서 미완료 거래를 취소 처리한다."""
        if self.state.trades_today and not self.state.trades_today[-1].exit_reason:
            last = self.state.trades_today[-1]
            self.db.update_trade_exit(
                trade_id=last.trade_db_id,
                exit_price=0,
                exit_reason="CANCELLED",
                pnl=0,
                commission=0,
                r_multiple=0,
            )
            self.state.trades_today.pop()


class PaperExecutor(BaseExecutor):
    """페이퍼 트레이딩 실행자.

    실제 주문 없이 봉의 H/L로 SL/TP 히트를 시뮬레이션한다.
    """

    def __init__(self, state: LiveState, db: TradeLogger, symbol: str | None = None) -> None:
        super().__init__(state, db, symbol)
        self._next_order_id = 1

    def _gen_order_id(self) -> str:
        oid = f"PAPER-{self._next_order_id}"
        self._next_order_id += 1
        return oid

    async def open_position(self, position: Position) -> bool:
        """페이퍼 포지션 오픈 (즉시 체결 시뮬레이션)."""
        self.state.position = position
        self.state.entry_filled = True
        entry_oid = self._gen_order_id()
        self.state.entry_order_id = entry_oid
        self.state.sl_order_id = self._gen_order_id()

        self.state.tp_order_id = self._gen_order_id()

        trade_id = self.db.log_trade(
            timestamp=position.entry_time,
            direction=position.direction.value,
            entry_price=position.entry_price,
            exit_price=None,
            size=position.size,
            sl_price=position.sl_price,
            tp_price=position.tp_price,
            entry_order_id=entry_oid,
            symbol=self.symbol,
        )

        self.state.trades_today.append(LiveTrade(
            trade_db_id=trade_id,
            direction=position.direction.value,
            entry_price=position.entry_price,
            size=position.size,
            sl_price=position.sl_price,
            tp_price=position.tp_price,
            entry_order_id=entry_oid,
        ))

        logger.info(
            "[PAPER] Position opened: %s @ %g, size=%.6f, SL=%g, TP=%g",
            position.direction.value, position.entry_price, position.size,
            position.sl_price, position.tp_price,
        )
        return True

    async def close_position(self, reason: str, price: float | None = None) -> bool:
        """페이퍼 포지션 종료."""
        pos = self.state.position
        if pos is None:
            return False

        if price is not None:
            exit_price = price
        elif reason == "SL":
            exit_price = pos.sl_price
        else:
            exit_price = pos.tp_price

        self._record_exit(pos, exit_price, reason, self._gen_order_id())
        self._clear_position_state()
        return True

    async def check_sl_tp(self, high: float, low: float) -> bool:
        """봉의 H/L로 SL/TP 히트를 확인하고 청산한다."""
        if self.state.position is None:
            return False

        hit, exit_price, reason = check_sl_tp_hit(self.state.position, high, low)
        if hit:
            await self.close_position(reason, exit_price)
            return True
        return False

    async def cancel_entry_limit(self) -> None:
        """페이퍼 진입 주문 취소 (상태 초기화)."""
        self._clear_position_state()
        logger.info("[PAPER] Entry order cancelled")

    async def update_sl_order(self, new_sl: float) -> bool:
        """페이퍼 SL 업데이트."""
        if self.state.position is None:
            return False
        self.state.position.sl_price = new_sl
        logger.info("[PAPER] SL updated to %g", new_sl)
        return True

    async def cancel_all_orders(self) -> None:
        """페이퍼 주문 취소 (상태만 정리)."""
        self.state.sl_order_id = None
        self.state.tp_order_id = None
        logger.info("[PAPER] Orders cancelled")

    async def emergency_close(self) -> None:
        """비상 청산."""
        if self.state.position:
            await self.close_position("KILL_SWITCH", self.state.position.sl_price)
        await self.cancel_all_orders()
        self.state.is_active = False
        logger.critical("[PAPER] Emergency close complete")


class LiveExecutor(BaseExecutor):
    """실제 Binance Futures 주문 실행자."""

    def __init__(
        self,
        client: AsyncClient,
        state: LiveState,
        db: TradeLogger,
        symbol: str | None = None,
        tick_size: float = 0.10,
        step_size: float = 0.001,
    ) -> None:
        super().__init__(state, db, symbol)
        self.client = client
        self.tick_size = tick_size
        self.step_size = step_size

    # ── 거래소 초기화 ─────────────────────────────────────────────

    async def fetch_symbol_info(self) -> None:
        """Binance에서 심볼의 tick_size, step_size를 조회한다."""
        try:
            info = await self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == self.symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "PRICE_FILTER":
                            self.tick_size = float(f["tickSize"])
                        elif f["filterType"] == "LOT_SIZE":
                            self.step_size = float(f["stepSize"])
                    logger.info(
                        "%s trading rules: tick=%.8f, step=%.8f",
                        self.symbol, self.tick_size, self.step_size,
                    )
                    return
            logger.warning("%s symbol info not found", self.symbol)
        except Exception:
            logger.exception("Symbol info fetch failed, using defaults")

    async def setup_leverage(self) -> None:
        """레버리지 설정."""
        leverage = SETTINGS.get("leverage", 20)
        try:
            await self.client.futures_change_leverage(
                symbol=self.symbol, leverage=leverage
            )
            logger.info("Leverage set: %dx", leverage)
        except Exception:
            logger.exception("Leverage setup failed")

    async def sync_exchange_state(self) -> None:
        """거래소 상태를 동기화한다 (잔고, 고아 포지션 처리)."""
        try:
            account = await self.client.futures_account()
            for target_asset in ("USDT", "USDC"):
                for asset in account.get("assets", []):
                    if asset["asset"] == target_asset:
                        bal = float(asset["walletBalance"])
                        if bal > 0:
                            self.state.balance = bal
                            break
                if self.state.balance > 0:
                    break

            positions = await self.client.futures_position_information(symbol=self.symbol)
            for p in positions:
                amt = float(p["positionAmt"])
                if amt != 0 and self.state.position is None:
                    direction = Signal.LONG if amt > 0 else Signal.SHORT
                    logger.warning(
                        "Orphan position found! Market closing: %s %s %.6f",
                        self.symbol, direction.value, abs(amt),
                    )
                    try:
                        await self.client.futures_cancel_all_open_orders(symbol=self.symbol)
                        await self.client.futures_create_order(
                            symbol=self.symbol,
                            side=_close_side(direction),
                            type="MARKET",
                            quantity=abs(amt),
                            reduceOnly=True,
                        )
                        logger.info("Orphan position closed")
                    except Exception:
                        logger.exception("Orphan position auto-close failed! Manual check required")

            if self.state.position is not None and not self.state.entry_filled:
                logger.info("Pending LIMIT detected after restart → checking via REST")
                await self.check_pending_entry()

            logger.info("Exchange sync complete: balance=%.2f", self.state.balance)
        except Exception:
            logger.exception("Exchange sync failed")

    # ── 주문 발행 헬퍼 ───────────────────────────────────────────

    async def _place_tp_order(self, position: Position) -> dict | None:
        """TP 주문 (TAKE_PROFIT limit — maker fee 0%)."""
        tp_price = _round_price(position.tp_price, self.tick_size)
        qty = _round_qty(position.size, self.step_size)
        try:
            order = await self.client.futures_create_order(
                symbol=self.symbol,
                side=_close_side(position.direction),
                type="TAKE_PROFIT",
                stopPrice=tp_price,
                price=tp_price,
                timeInForce="GTC",
                quantity=qty,
                reduceOnly=True,
            )
            logger.info("TP order set (LIMIT): %.2f (orderId=%s)", tp_price, _get_order_id(order))
            return order
        except Exception:
            logger.exception("TP order failed")
            return None

    async def _place_sl_order(self, position: Position) -> dict | None:
        """초기 손절 주문 (STOP_MARKET — 시장가로 반드시 체결)."""
        stop_price = _round_price(position.sl_price, self.tick_size)
        qty = _round_qty(position.size, self.step_size)
        try:
            order = await self.client.futures_create_order(
                symbol=self.symbol,
                side=_close_side(position.direction),
                type="STOP_MARKET",
                stopPrice=stop_price,
                quantity=qty,
                reduceOnly=True,
            )
            logger.info("SL order set (STOP_MARKET): trigger=%.2f (orderId=%s)", stop_price, _get_order_id(order))
            return order
        except Exception:
            logger.exception("SL order failed")
            return None

    # ── 체결 후 SL/TP 설정 ────────────────────────────

    async def _setup_sl_tp_after_fill(self, pos: Position, avg_price: float, filled_qty: float) -> None:
        """진입 체결 후 SL/TP 주문을 설정한다."""
        pos.entry_price = avg_price
        pos.size = filled_qty

        _std_tp_r = SETTINGS["tp_atr_multiplier"] / SETTINGS["sl_atr_multiplier"]

        if pos.direction == Signal.LONG:
            pos.sl_price = avg_price - pos.r_unit
            pos.tp_price = avg_price + pos.r_unit * _std_tp_r
        else:
            pos.sl_price = avg_price + pos.r_unit
            pos.tp_price = avg_price - pos.r_unit * _std_tp_r

        pos.initial_sl = pos.sl_price
        self.state.entry_filled = True

        # SL 주문
        sl_order = await self._place_sl_order(pos)
        if sl_order:
            self.state.sl_order_id = str(_get_order_id(sl_order))

        # TP 주문
        tp_order = await self._place_tp_order(pos)
        if tp_order:
            self.state.tp_order_id = str(_get_order_id(tp_order))

        # DB 업데이트
        if self.state.trades_today:
            self.db.update_trade_entry(
                trade_id=self.state.trades_today[-1].trade_db_id,
                entry_price=avg_price,
                size=pos.size,
                sl_price=pos.sl_price,
                tp_price=pos.tp_price,
            )

    # ── 포지션 진입/청산 ─────────────────────────────────────────

    async def open_position(self, position: Position) -> bool:
        """LIMIT 주문 진입 → user_data stream으로 체결 확인 후 SL/TP 설정."""
        side = SIDE_BUY if position.direction == Signal.LONG else SIDE_SELL
        qty = _round_qty(position.size, self.step_size)
        if qty <= 0:
            logger.warning("Order qty 0, entry cancelled")
            return False

        limit_price = _round_price(position.entry_price, self.tick_size)

        # 트레이드별 동적 레버리지 설정
        if self.state.balance > 0:
            notional = qty * limit_price
            required_lev = math.ceil(notional / self.state.balance)
            max_lev = SETTINGS.get("leverage", 20)
            trade_leverage = max(1, min(required_lev, max_lev))
            try:
                await self.client.futures_change_leverage(
                    symbol=self.symbol, leverage=trade_leverage,
                )
                logger.info("Dynamic leverage set: %dx (notional=%.2f, balance=%.2f)", trade_leverage, notional, self.state.balance)
            except Exception:
                logger.warning("Leverage setup failed — using current exchange setting")

        try:
            order = await self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=qty,
                price=limit_price,
            )
            order_id = str(_get_order_id(order))
            logger.info("LIMIT 진입 주문: %s %s qty=%.6f @ %.2f → orderId=%s", side, self.symbol, qty, limit_price, order_id)

            self.state.position = position
            self.state.entry_order_id = order_id
            self.state.entry_filled = False
            self.state.entry_limit_candle = self.state.candle_count

            trade_id = self.db.log_trade(
                timestamp=datetime.now(timezone.utc).isoformat(),
                direction=position.direction.value,
                entry_price=limit_price,
                exit_price=None,
                size=qty,
                sl_price=position.sl_price,
                tp_price=position.tp_price,
                entry_order_id=order_id,
                symbol=self.symbol,
                notes="LIMIT_PENDING",
            )

            self.state.trades_today.append(LiveTrade(
                trade_db_id=trade_id,
                direction=position.direction.value,
                entry_price=limit_price,
                size=qty,
                sl_price=position.sl_price,
                tp_price=position.tp_price,
                entry_order_id=order_id,
            ))

            return True
        except Exception:
            logger.exception("LIMIT entry order failed")
            return False

    async def close_position(self, reason: str, price: float | None = None) -> bool:
        """포지션을 시장가로 청산한다."""
        pos = self.state.position
        if pos is None:
            return False

        qty = _round_qty(pos.size, self.step_size)

        # 레이스 컨디션 방지: await 전에 포지션 상태 초기화
        self._clear_position_state()

        try:
            await self.cancel_all_orders()
            order = await self.client.futures_create_order(
                symbol=self.symbol,
                side=_close_side(pos.direction),
                type="MARKET",
                quantity=qty,
                reduceOnly=True,
            )
            exit_price = float(order.get("avgPrice", price or 0))
            self._record_exit(pos, exit_price, reason, str(_get_order_id(order)))
            return True
        except Exception:
            logger.exception("Close position failed!")
            return False

    async def update_sl_order(self, new_sl: float) -> bool:
        """기존 SL 주문을 취소하고 새 SL로 재설정한다."""
        if self.state.position is None:
            return False

        if self.state.sl_order_id:
            try:
                await self.client.futures_cancel_order(
                    symbol=self.symbol, orderId=self.state.sl_order_id,
                )
                logger.debug("Existing SL order cancelled: %s", self.state.sl_order_id)
            except Exception:
                logger.warning("Existing SL cancel failed, placing new order")

        self.state.position.sl_price = new_sl
        sl_order = await self._place_sl_order(self.state.position)

        if sl_order:
            self.state.sl_order_id = str(_get_order_id(sl_order))
            return True
        return False

    async def check_pending_entry(self) -> bool:
        """미체결 LIMIT 진입 주문의 상태를 REST API로 직접 확인한다."""
        if (self.state.position is None
                or self.state.entry_filled
                or not self.state.entry_order_id):
            return False

        try:
            order = await self.client.futures_get_order(
                symbol=self.symbol,
                orderId=self.state.entry_order_id,
            )
        except Exception:
            logger.exception("Entry order status check failed (REST)")
            return False

        status = order.get("status", "")

        if status == "FILLED":
            avg_price = float(order.get("avgPrice", 0))
            filled_qty = float(order.get("executedQty", 0)) or self.state.position.size
            await self._setup_sl_tp_after_fill(self.state.position, avg_price, filled_qty)
            logger.warning(
                "[REST recovery] Missed entry fill event recovered: %s @ %.4f, SL=%.4f, TP=%.4f",
                self.state.position.direction.value, avg_price,
                self.state.position.sl_price, self.state.position.tp_price,
            )
            return True

        if status in ("CANCELED", "EXPIRED"):
            logger.info("Entry LIMIT order already cancelled (REST check): %s", status)
            self._cancel_pending_trade_db()
            self._clear_position_state()

        return False

    async def cancel_entry_limit(self) -> None:
        """미체결 진입 LIMIT 주문을 취소하고 포지션 상태를 초기화한다."""
        if self.state.entry_order_id:
            try:
                await self.client.futures_cancel_order(
                    symbol=self.symbol, orderId=self.state.entry_order_id,
                )
                logger.info("Entry LIMIT order cancelled: %s", self.state.entry_order_id)
            except Exception:
                logger.warning("Entry LIMIT cancel failed — checking order status")
                try:
                    order = await self.client.futures_get_order(
                        symbol=self.symbol, orderId=self.state.entry_order_id,
                    )
                    if order.get("status") == "FILLED":
                        logger.warning("[cancel_entry_limit] Fill detected during cancel → setting SL/TP")
                        pos = self.state.position
                        avg_price = float(order.get("avgPrice", pos.entry_price))
                        filled_qty = float(order.get("executedQty", 0)) or pos.size
                        await self._setup_sl_tp_after_fill(pos, avg_price, filled_qty)
                        return
                except Exception:
                    logger.exception("Order status check failed, clearing state")

        self._cancel_pending_trade_db()
        self._clear_position_state()

    async def cancel_all_orders(self) -> None:
        """모든 미체결 주문을 취소한다 (일반 + algo 주문)."""
        # 일반 주문 취소
        try:
            await self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            logger.info("Open orders cancelled")
        except Exception:
            logger.exception("Order cancel failed")

        # Algo(조건부) 주문 전체 조회 후 취소
        try:
            algo_orders = await self.client._request_futures_api(
                "get", "openAlgoOrders", signed=True,
                data={"symbol": self.symbol},
            )
            for ao in algo_orders:
                aid = ao.get("algoId")
                if aid:
                    try:
                        await self.client._request_futures_api(
                            "delete", "algoOrder", signed=True,
                            data={"algoId": aid},
                        )
                        logger.info("Algo order cancelled: %s (%s)", aid, ao.get("orderType"))
                    except Exception:
                        logger.debug("Algo order cancel failed: %s", aid)
            if algo_orders:
                logger.info("Algo orders cancelled: %d", len(algo_orders))
        except Exception:
            logger.exception("Algo order query/cancel failed")

        self.state.sl_order_id = None
        self.state.tp_order_id = None

    async def emergency_close(self) -> None:
        """비상 청산: 모든 주문 취소 + 포지션 시장가 청산."""
        logger.critical("Emergency close initiated!")
        if self.state.position:
            await self.close_position("KILL_SWITCH")
        else:
            await self.cancel_all_orders()
        self.state.is_active = False
        logger.critical("Emergency close complete, bot deactivated")

    # ── WebSocket 이벤트 처리 ────────────────────────────────────

    async def handle_order_update(self, event: dict) -> None:
        """WebSocket ORDER_TRADE_UPDATE 이벤트를 처리한다.

        처리 순서:
          1.   진입 LIMIT 체결 → SL + TP(분할 or 단일) 발행
          1.5  분할 TP 체결   → SL 재설정 + 최종 TP 발행
          2.   TP 체결        → SL 취소 + 상태 정리
          3.   SL/트레일링 SL 체결 → TP 취소 + 상태 정리
          4.   진입 LIMIT 외부 취소 → 상태 초기화
          5.   STOP_LIMIT 만료(갭 통과) → 시장가 청산 폴백
        """
        order = event.get("o", {})
        order_id = str(order.get("i", ""))
        status = order.get("X", "")
        order_type = order.get("ot", "")
        symbol = order.get("s", "")

        logger.info(
            "Order event: id=%s type=%s status=%s symbol=%s (entry=%s filled=%s SL=%s TP=%s)",
            order_id, order_type, status, symbol,
            self.state.entry_order_id, self.state.entry_filled,
            self.state.sl_order_id, self.state.tp_order_id,
        )

        if status == "FILLED":
            avg_price = float(order.get("ap", 0))

            # ── 1. 진입 LIMIT 체결
            is_entry = (
                order_id == self.state.entry_order_id
                and not self.state.entry_filled
                and self.state.position is not None
            )
            if is_entry:
                filled_qty = float(order.get("q", 0)) or self.state.position.size
                pos = self.state.position
                await self._setup_sl_tp_after_fill(pos, avg_price, filled_qty)

                logger.info(
                    ">>> Entry LIMIT filled: %s @ %.4f, qty=%.6f, SL=%.4f, TP=%.4f",
                    pos.direction.value, avg_price, filled_qty,
                    pos.sl_price, pos.tp_price,
                )
                return

            if not self.state.entry_filled:
                return

            # ── 2. TP 체결
            is_tp = (
                order_id == self.state.tp_order_id
                or (
                    order_type == "TAKE_PROFIT"
                    and self.state.position is not None
                )
            )
            # ── 3. SL 체결
            is_sl = (
                order_id == self.state.sl_order_id
                or (order_type in ("STOP_MARKET", "STOP") and self.state.position is not None)
            )

            if is_tp and not is_sl:
                pos = self.state.position
                if pos:
                    self._record_exit(pos, avg_price, "TP", order_id)
                    self._clear_position_state()
                    await self.cancel_all_orders()

            elif is_sl:
                pos = self.state.position
                if pos:
                    self._record_exit(pos, avg_price, "SL", order_id)
                    self._clear_position_state()
                    await self.cancel_all_orders()

        # ── 4. 진입 LIMIT 외부 취소
        elif (status == "CANCELED"
              and order_id == self.state.entry_order_id
              and not self.state.entry_filled):
            logger.info("Entry LIMIT externally cancelled: %s", order_id)
            self._cancel_pending_trade_db()
            self._clear_position_state()

        # ── 5. STOP_LIMIT 만료 (갭 통과 폴백)
        elif status == "EXPIRED" and order_id == self.state.sl_order_id:
            logger.warning("SL STOP_LIMIT expired (gap through), executing market close!")
            if self.state.position and self.state.entry_filled:
                await self.close_position("SL_FALLBACK")
