"""시그널 시각화 - 일별 캔들차트에 매매 포인트를 표시한다.

현재 백테스트 시스템(strategy.signals + backtest.engine)을 사용하여
일별 5분봉 캔들차트에 LONG/SHORT 진입 및 SL/TP 청산 포인트를 렌더링한다.

사용법:
  python3 visualize_signals.py                        # 2026년 2월 전체
  python3 visualize_signals.py 2026-02-01 2026-02-17  # 날짜 범위 지정
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from datetime import timedelta

from config.settings import SETTINGS
from data.storage import load_from_parquet
from strategy.signals import compute_indicators, Signal
from backtest.engine import run_backtest

logging.basicConfig(level=logging.WARNING)


def draw_candlestick(ax, df, width_minutes=3.0):
    """캔들차트를 그린다."""
    width = timedelta(minutes=width_minutes)

    for ts, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        t = ts.to_pydatetime()

        if c >= o:
            color = "#26a69a"
            body_bottom = o
            body_height = c - o
        else:
            color = "#ef5350"
            body_bottom = c
            body_height = o - c

        ax.plot([t, t], [l, h], color=color, linewidth=0.5, zorder=2)
        rect = Rectangle(
            (t - width / 2, body_bottom), width,
            body_height if body_height > 0 else 0.01,
            facecolor=color, edgecolor=color, linewidth=0.3, zorder=3,
        )
        ax.add_patch(rect)


def render_chart(ax, chunk_df, chunk_trades, df_full, trade_num_offset):
    """일별 차트를 렌더링한다."""
    draw_candlestick(ax, chunk_df, width_minutes=3.0)

    # EMA 라인
    ema_styles = [
        ("ema_fast", "#FF8C00", "EMA 9"),
        ("ema_mid", "#1E90FF", "EMA 21"),
        ("ema_slow", "#9932CC", "EMA 50"),
    ]
    for col, color, label in ema_styles:
        if col in chunk_df.columns:
            ax.plot(
                [t.to_pydatetime() for t in chunk_df.index],
                chunk_df[col], color=color, linewidth=1.2,
                label=label, alpha=0.85, zorder=4,
            )

    # VWAP
    if "vwap" in chunk_df.columns:
        ax.plot(
            [t.to_pydatetime() for t in chunk_df.index],
            chunk_df["vwap"], color="gray", linewidth=1.0,
            linestyle="--", label="VWAP", alpha=0.7, zorder=4,
        )

    # 매매 포인트
    for i, (trade, entry_ts, exit_ts) in enumerate(chunk_trades):
        num = trade_num_offset + i + 1
        pnl = trade.pnl
        pnl_str = f"${pnl:+,.0f}"

        # 진입 마커
        if trade.direction == Signal.LONG:
            marker, color = "^", "#00AA00"
            label = "LONG"
        else:
            marker, color = "v", "#CC0000"
            label = "SHORT"

        ax.scatter(
            entry_ts.to_pydatetime(), trade.entry_price,
            marker=marker, c=color, s=120, zorder=7,
            edgecolors="black", linewidth=0.4,
        )
        ax.annotate(
            f"#{num} {label}",
            xy=(entry_ts.to_pydatetime(), trade.entry_price),
            xytext=(0, 14), textcoords="offset points",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9),
            ha="center", zorder=8,
        )

        # SL/TP 라인
        if exit_ts is not None:
            et = entry_ts.to_pydatetime()
            xt = exit_ts.to_pydatetime()

            # 진입가 점선
            ax.plot(
                [et, xt + timedelta(minutes=15)],
                [trade.entry_price, trade.entry_price],
                color="white", linewidth=0.8, linestyle=":", alpha=0.6, zorder=5,
            )

            # TP 목표 라인
            tp = trade.tp_price if hasattr(trade, "tp_price") else None
            if tp and tp > 0:
                ax.plot(
                    [et, xt + timedelta(minutes=15)], [tp, tp],
                    color="#FFD700", linewidth=0.8, linestyle="--",
                    alpha=0.6, zorder=5,
                )

            # 청산 마커
            if trade.exit_reason == "TP":
                exit_marker, exit_color = "D", "#FFD700"
            else:
                exit_marker, exit_color = "X", "#8B0000"

            ax.scatter(
                xt, trade.exit_price,
                marker=exit_marker, c=exit_color, s=80, zorder=7,
                edgecolors="black", linewidth=0.4,
            )

            # PnL 라벨
            pcolor = "#00AA00" if pnl >= 0 else "#CC0000"
            ax.annotate(
                f"#{num} {trade.exit_reason} {pnl_str}",
                xy=(xt, trade.exit_price),
                xytext=(0, -18), textcoords="offset points",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=pcolor, alpha=0.85),
                ha="center", zorder=9,
            )


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        start_date, end_date = args[0], args[1]
    else:
        start_date = "2026-02-01"
        end_date = "2026-02-24"

    print(f"데이터 로드 중... ({start_date} ~ {end_date})")

    # 지표 warm-up을 위해 한 달 전부터 로드
    warmup_start = pd.Timestamp(start_date) - timedelta(days=30)
    df = load_from_parquet(
        start=warmup_start.strftime("%Y-%m-%d"),
        end=end_date,
    )
    if df is None or df.empty:
        print("데이터가 없습니다. --fetch-data로 먼저 수집하세요.")
        sys.exit(1)

    print(f"데이터: {len(df)}개 캔들")

    # 백테스트 실행
    print("백테스트 실행 중...")
    portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])
    df_ind = compute_indicators(df)

    # 전체 거래에서 날짜 범위 내 거래 추출
    plot_start = pd.Timestamp(start_date, tz="UTC")
    plot_end = pd.Timestamp(end_date + " 23:59:59", tz="UTC")

    all_trades = []
    for trade in portfolio.trades:
        entry_ts = df.index[trade.entry_index] if trade.entry_index < len(df) else None
        exit_ts = df.index[trade.exit_index] if trade.exit_index < len(df) else None
        if entry_ts is None:
            continue
        if plot_start <= entry_ts <= plot_end:
            all_trades.append((trade, entry_ts, exit_ts))

    print(f"기간 내 거래: {len(all_trades)}건")

    # 출력 디렉토리
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "charts_5m"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 일별 차트 생성
    dates = pd.date_range(start_date, end_date, freq="D")
    trade_num = 0

    for di, date in enumerate(dates):
        day_start = pd.Timestamp(date, tz="UTC")
        day_end = day_start + timedelta(days=1)
        date_str = date.strftime("%Y-%m-%d")

        # 해당 일의 캔들
        mask = (df_ind.index >= day_start) & (df_ind.index < day_end)
        chunk_df = df_ind[mask]
        if chunk_df.empty:
            continue

        # 해당 일의 거래
        day_trades = [
            (t, ets, xts) for t, ets, xts in all_trades
            if day_start <= ets < day_end
        ]

        # 차트 생성
        fig, ax = plt.subplots(1, 1, figsize=(32, 11))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        render_chart(ax, chunk_df, day_trades, df, trade_num)

        # PnL 합산
        day_pnl = sum(t.pnl for t, _, _ in day_trades)
        wins = sum(1 for t, _, _ in day_trades if t.pnl > 0)
        losses = len(day_trades) - wins
        pnl_str = f"${day_pnl:+,.0f}"
        pnl_color = "#00AA00" if day_pnl >= 0 else "#CC0000"

        weekday = date.strftime("%a")
        title = (
            f"BTCUSDT 5m  {date_str} ({weekday})  |  "
            f"Trades: {len(day_trades)} ({wins}W/{losses}L)  |  "
            f"PnL: {pnl_str}"
        )
        ax.set_title(title, fontsize=14, fontweight="bold",
                     color=pnl_color if day_trades else "white")
        ax.set_xlabel("Time (UTC)", color="white")
        ax.set_ylabel("Price (USDT)", color="white")
        ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e",
                  edgecolor="gray", labelcolor="white")
        ax.grid(True, alpha=0.15, color="gray")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)

        for spine in ax.spines.values():
            spine.set_color("gray")

        plt.tight_layout()

        filepath = os.path.join(output_dir, f"signals_{date_str}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

        status = f"  [{di+1}/{len(dates)}] {date_str}  trades={len(day_trades)}  PnL={pnl_str}"
        print(status)
        trade_num += len(day_trades)

    # 요약
    total_pnl = sum(t.pnl for t, _, _ in all_trades)
    total_wins = sum(1 for t, _, _ in all_trades if t.pnl > 0)
    print(f"\n완료! {len(dates)}일 차트 → {output_dir}/")
    print(f"총 {len(all_trades)}건 거래, {total_wins}W/{len(all_trades)-total_wins}L, "
          f"PnL: ${total_pnl:+,.0f}")


if __name__ == "__main__":
    main()
