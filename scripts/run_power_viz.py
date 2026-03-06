import sys
import os
import logging
import asyncio

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
from strategy.signals import compute_indicators, Signal
from backtest.engine import run_backtest

logging.basicConfig(level=logging.WARNING)

# 1. Fetch POWERUSDT 5m data with warm-up (500 candles)
from binance.client import AsyncClient

async def fetch_data():
    client = await AsyncClient.create()
    klines = await client.futures_klines(
        symbol='POWERUSDT',
        interval='5m',
        limit=500,
    )
    rows = []
    for k in klines:
        rows.append({
            'open_time': pd.Timestamp(k[0], unit='ms', tz='UTC'),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5]),
            'close_time': pd.Timestamp(k[6], unit='ms', tz='UTC'),
            'quote_volume': float(k[7]),
        })
    await client.close_connection()
    return pd.DataFrame(rows).set_index('open_time')

df = asyncio.run(fetch_data())
print(f"Data loaded: {len(df)} candles, {df.index[0]} ~ {df.index[-1]}")

# 2. Run backtest
portfolio = run_backtest(df, capital=SETTINGS["initial_capital"])
df_ind = compute_indicators(df)

# 3. Today's data and trades
date_str = "2026-02-25"
day_start = pd.Timestamp(date_str, tz="UTC")
day_end = day_start + timedelta(days=1)

all_trades = []
for trade in portfolio.trades:
    entry_ts = df.index[trade.entry_index] if trade.entry_index < len(df) else None
    exit_ts = df.index[trade.exit_index] if trade.exit_index < len(df) else None
    if entry_ts is None:
        continue
    if day_start <= entry_ts < day_end:
        all_trades.append((trade, entry_ts, exit_ts))

print(f"Today's trades: {len(all_trades)}")

# Also check open position
if portfolio.position:
    pos = portfolio.position
    entry_ts = df.index[pos.entry_index] if pos.entry_index < len(df) else None
    if entry_ts and day_start <= entry_ts < day_end:
        # Build a pseudo-trade for the open position
        print(f"  Open position: {pos.direction.value} @ {pos.entry_price}")

# 4. Draw chart (same style as visualize_signals.py)
mask = (df_ind.index >= day_start) & (df_ind.index < day_end)
chunk_df = df_ind[mask]

if chunk_df.empty:
    # If no data for just today, try yesterday or show last 288 candles (1 day)
    print(f"No data for {date_str}, using last available day instead.")
    last_ts = df_ind.index[-1]
    day_start = last_ts.normalize()
    day_end = day_start + timedelta(days=1)
    date_str = day_start.strftime("%Y-%m-%d")
    mask = (df_ind.index >= day_start) & (df_ind.index < day_end)
    chunk_df = df_ind[mask]

    # Recalculate trades for that day
    all_trades = []
    for trade in portfolio.trades:
        entry_ts = df.index[trade.entry_index] if trade.entry_index < len(df) else None
        exit_ts = df.index[trade.exit_index] if trade.exit_index < len(df) else None
        if entry_ts is None:
            continue
        if day_start <= entry_ts < day_end:
            all_trades.append((trade, entry_ts, exit_ts))

    print(f"Adjusted to {date_str}, trades: {len(all_trades)}")

if chunk_df.empty:
    print("No data available at all!")
    sys.exit(1)

print(f"Chart candles: {len(chunk_df)}")

def draw_candlestick(ax, df, width_minutes=3.0):
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
        # Use a small fraction of price for doji candles
        min_height = (h - l) * 0.01 if (h - l) > 0 else 0.0001
        rect = Rectangle(
            (t - width / 2, body_bottom), width,
            body_height if body_height > 0 else min_height,
            facecolor=color, edgecolor=color, linewidth=0.3, zorder=3,
        )
        ax.add_patch(rect)

fig, ax = plt.subplots(1, 1, figsize=(32, 11))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

draw_candlestick(ax, chunk_df, width_minutes=3.0)

# EMA lines
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

# Trade markers
for i, (trade, entry_ts, exit_ts) in enumerate(all_trades):
    num = i + 1
    pnl = trade.pnl if hasattr(trade, 'pnl') and trade.pnl else 0

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

    # SL/TP lines and exit marker
    if exit_ts is not None:
        et = entry_ts.to_pydatetime()
        xt = exit_ts.to_pydatetime()

        # Entry price dotted line
        ax.plot(
            [et, xt + timedelta(minutes=15)],
            [trade.entry_price, trade.entry_price],
            color="white", linewidth=0.8, linestyle=":", alpha=0.6, zorder=5,
        )

        # Exit marker
        exit_reason = trade.exit_reason if hasattr(trade, 'exit_reason') else "?"
        if exit_reason == "TP":
            exit_marker, exit_color = "D", "#FFD700"
        else:
            exit_marker, exit_color = "X", "#8B0000"

        ax.scatter(
            xt, trade.exit_price,
            marker=exit_marker, c=exit_color, s=80, zorder=7,
            edgecolors="black", linewidth=0.4,
        )

        pnl_str = f"${pnl:+,.0f}"
        pcolor = "#00AA00" if pnl >= 0 else "#CC0000"
        ax.annotate(
            f"#{num} {exit_reason} {pnl_str}",
            xy=(xt, trade.exit_price),
            xytext=(0, -18), textcoords="offset points",
            fontsize=7, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=pcolor, alpha=0.85),
            ha="center", zorder=9,
        )

# Title
day_pnl = sum(t.pnl for t, _, _ in all_trades if hasattr(t, 'pnl') and t.pnl)
wins = sum(1 for t, _, _ in all_trades if hasattr(t, 'pnl') and t.pnl and t.pnl > 0)
losses = len(all_trades) - wins
pnl_str = f"${day_pnl:+,.0f}"
pnl_color = "#00AA00" if day_pnl >= 0 else "#CC0000"

weekday = pd.Timestamp(date_str).strftime("%a")
title = (
    f"POWERUSDT 5m  {date_str} ({weekday})  |  "
    f"Trades: {len(all_trades)} ({wins}W/{losses}L)  |  "
    f"PnL: {pnl_str}"
)
ax.set_title(title, fontsize=14, fontweight="bold",
             color=pnl_color if all_trades else "white")
ax.set_xlabel("Time (UTC)", color="white")
ax.set_ylabel("Price (USDC)", color="white")
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

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'charts_5m')
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, f"POWERUSDT_signals_{date_str}.png")
plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\nChart saved: {filepath}")
print(f"Trades: {len(all_trades)}, PnL: {pnl_str}")
for i, (t, ets, xts) in enumerate(all_trades):
    exit_reason = t.exit_reason if hasattr(t, 'exit_reason') else "OPEN"
    pnl_val = t.pnl if hasattr(t, 'pnl') and t.pnl else 0
    print(f"  #{i+1} {t.direction.value} entry={t.entry_price:.4f} exit={getattr(t, 'exit_price', 'N/A')} reason={exit_reason} pnl=${pnl_val:+,.2f}")
