"""고아 포지션 수동 청산 스크립트.

사용법:
    python close_position.py SAHARAUSDT
    python close_position.py          # 전체 포지션 조회만
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance import AsyncClient
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET


async def main():
    client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)

    try:
        # 전체 포지션 조회
        positions = await client.futures_position_information()
        open_positions = [
            p for p in positions if float(p["positionAmt"]) != 0
        ]

        if not open_positions:
            print("열린 포지션 없음")
            return

        print(f"\n{'Symbol':<16} {'Side':<6} {'Amount':<14} {'Entry':>10} {'PnL':>10}")
        print("-" * 60)
        for p in open_positions:
            amt = float(p["positionAmt"])
            side = "LONG" if amt > 0 else "SHORT"
            entry = float(p["entryPrice"])
            pnl = float(p["unRealizedProfit"])
            print(f"{p['symbol']:<16} {side:<6} {abs(amt):<14.6f} {entry:>10.4f} {pnl:>+10.4f}")

        # 심볼 지정 시 해당 포지션 청산
        if len(sys.argv) > 1:
            symbol = sys.argv[1].upper()
            target = next((p for p in open_positions if p["symbol"] == symbol), None)

            if not target:
                print(f"\n{symbol} 포지션을 찾을 수 없습니다.")
                return

            amt = float(target["positionAmt"])
            close_side = "SELL" if amt > 0 else "BUY"
            close_qty = abs(amt)

            confirm = input(f"\n{symbol} {close_qty} 시장가 청산? (y/n): ")
            if confirm.lower() != "y":
                print("취소됨")
                return

            # 미체결 주문 먼저 취소
            try:
                await client.futures_cancel_all_open_orders(symbol=symbol)
                print(f"{symbol} 미체결 주문 취소 완료")
            except Exception as e:
                print(f"주문 취소 실패 (무시): {e}")

            # 시장가 청산
            order = await client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=close_qty,
                reduceOnly=True,
            )
            print(f"\n청산 완료!")
            print(f"  orderId: {order['orderId']}")
            print(f"  status: {order['status']}")
            print(f"  avgPrice: {order.get('avgPrice', '-')}")

    finally:
        await client.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
