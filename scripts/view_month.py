"""특정 월의 매매 차트를 생성하는 스크립트."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import SETTINGS
from data.storage import load_from_parquet
from strategy.signals import compute_indicators, Signal
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics
from analysis.report import print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 전체 기간 백테스트
df = load_from_parquet(start=SETTINGS["backtest_start"], end=SETTINGS["backtest_end"])
portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])
metrics = calculate_metrics(portfolio)

# 지표 포함 데이터
df_ind = compute_indicators(df)

# 2025년 2월 필터링
month_start = "2025-02-01"
month_end = "2025-02-28"
mask = (df_ind.index >= pd.Timestamp(month_start, tz="UTC")) & \
       (df_ind.index <= pd.Timestamp(month_end + " 23:59:59", tz="UTC"))
plot_df = df_ind[mask]

# 해당 월의 거래 필터링
month_trades = []
for t in portfolio.trades:
    entry_ts = df.index[t.entry_index] if t.entry_index < len(df) else None
    exit_ts = df.index[t.exit_index] if t.exit_index < len(df) else None
    if entry_ts is None:
        continue
    in_month = (pd.Timestamp(month_start, tz="UTC") <= entry_ts <= pd.Timestamp(month_end + " 23:59:59", tz="UTC"))
    if in_month:
        month_trades.append((t, entry_ts, exit_ts))

print(f"\n2025년 2월: {len(month_trades)}건 거래")

# 차트 생성
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=["BTC 5min — 2025년 2월 매매 포인트", "RSI(14)", "Volume"],
)

# 캔들스틱
fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["open"], high=plot_df["high"],
    low=plot_df["low"], close=plot_df["close"], name="BTCUSDT",
), row=1, col=1)

# EMA
for col, color, name in [("ema_fast", "orange", "EMA 9"), ("ema_mid", "dodgerblue", "EMA 21"), ("ema_slow", "purple", "EMA 50")]:
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df[col], mode="lines",
        name=name, line=dict(color=color, width=1),
    ), row=1, col=1)

# VWAP
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["vwap"], mode="lines",
    name="VWAP", line=dict(color="gray", width=1, dash="dash"),
), row=1, col=1)

# 매매 포인트
for trade, entry_ts, exit_ts in month_trades:
    # 진입
    marker_sym = "triangle-up" if trade.direction == Signal.LONG else "triangle-down"
    marker_col = "lime" if trade.direction == Signal.LONG else "red"
    label = "LONG" if trade.direction == Signal.LONG else "SHORT"
    pnl_str = f"+{trade.pnl:.0f}" if trade.pnl > 0 else f"{trade.pnl:.0f}"

    fig.add_trace(go.Scatter(
        x=[entry_ts], y=[trade.entry_price], mode="markers+text",
        marker=dict(symbol=marker_sym, size=14, color=marker_col, line=dict(width=1, color="black")),
        text=[label], textposition="top center", textfont=dict(size=9, color=marker_col),
        showlegend=False,
    ), row=1, col=1)

    # 청산
    if exit_ts is not None:
        exit_col = "lime" if trade.pnl > 0 else "red"
        fig.add_trace(go.Scatter(
            x=[exit_ts], y=[trade.exit_price], mode="markers+text",
            marker=dict(symbol="x", size=10, color=exit_col, line=dict(width=2)),
            text=[f"{trade.exit_reason} {pnl_str}"], textposition="bottom center",
            textfont=dict(size=8, color=exit_col),
            showlegend=False,
        ), row=1, col=1)

    # SL/TP 라인
    if entry_ts and exit_ts:
        fig.add_shape(type="line", x0=entry_ts, x1=exit_ts,
            y0=trade.entry_price, y1=trade.entry_price,
            line=dict(color="white", width=0.5, dash="dot"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["rsi"], mode="lines",
    name="RSI", line=dict(color="orange", width=1),
), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

# Volume
colors = ["green" if c >= o else "red" for c, o in zip(plot_df["close"], plot_df["open"])]
fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df["volume"], marker_color=colors,
    name="Volume", showlegend=False,
), row=3, col=1)

# 레이아웃
wins = sum(1 for t, _, _ in month_trades if t.pnl > 0)
total_pnl = sum(t.pnl for t, _, _ in month_trades)
fig.update_layout(
    title=f"BTC 5min — 2025년 2월 | {len(month_trades)}건 거래, {wins}승 {len(month_trades)-wins}패, PnL: ${total_pnl:+,.0f}",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    height=900,
    width=1600,
)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig.write_html(os.path.join(_project_root, "feb_2025_trades.html"))
print(f"차트 저장: feb_2025_trades.html")
print(f"2월 성적: {wins}승 {len(month_trades)-wins}패, PnL: ${total_pnl:+,.2f}")
