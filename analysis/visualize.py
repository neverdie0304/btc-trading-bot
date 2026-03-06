"""plotly 기반 인터랙티브 차트 시각화 모듈."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.portfolio import Portfolio, Trade
from backtest.metrics import BacktestMetrics
from strategy.signals import Signal

logger = logging.getLogger(__name__)


def plot_equity_curve(
    portfolio: Portfolio,
    metrics: BacktestMetrics,
    output_path: str = "equity_curve.html",
) -> None:
    """Equity Curve 차트를 생성한다.

    Args:
        portfolio: 백테스트 결과 포트폴리오.
        metrics: 성과 지표.
        output_path: 저장 경로.
    """
    equity = np.array(portfolio.equity_curve)
    x = list(range(len(equity)))

    fig = go.Figure()

    # Equity curve
    fig.add_trace(go.Scatter(
        x=x, y=equity,
        mode="lines",
        name="Equity",
        line=dict(color="blue", width=1.5),
    ))

    # MDD 구간 표시
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100

    # MDD가 -2% 이상 깊은 구간만 음영
    in_dd = drawdown < -2
    dd_start = None
    for i in range(len(in_dd)):
        if in_dd[i] and dd_start is None:
            dd_start = i
        elif not in_dd[i] and dd_start is not None:
            fig.add_vrect(
                x0=dd_start, x1=i,
                fillcolor="red", opacity=0.1,
                line_width=0,
            )
            dd_start = None

    fig.update_layout(
        title=f"Equity Curve (Return: {metrics.total_return_pct:+.1f}%, MDD: {metrics.max_drawdown_pct:.1f}%)",
        xaxis_title="Candle Index",
        yaxis_title="Equity ($)",
        template="plotly_white",
    )

    fig.write_html(output_path)
    logger.info("Equity curve 저장: %s", output_path)


def plot_trade_overlay(
    df: pd.DataFrame,
    portfolio: Portfolio,
    output_path: str = "trade_overlay.html",
    window: int = 500,
) -> None:
    """캔들스틱 차트에 매매 포인트를 오버레이한다.

    Args:
        df: OHLCV DataFrame (지표 포함).
        portfolio: 백테스트 결과.
        output_path: 저장 경로.
        window: 표시할 최근 봉 수.
    """
    # 최근 N봉만 표시
    plot_df = df.iloc[-window:].copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=["Price & Signals", "Volume"],
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="BTCUSDT",
    ), row=1, col=1)

    # EMA 라인
    for col, color, name in [
        ("ema_fast", "orange", "EMA 9"),
        ("ema_mid", "blue", "EMA 21"),
        ("ema_slow", "purple", "EMA 50"),
    ]:
        if col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[col],
                mode="lines",
                name=name,
                line=dict(color=color, width=1),
            ), row=1, col=1)

    # VWAP
    if "vwap" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["vwap"],
            mode="lines",
            name="VWAP",
            line=dict(color="gray", width=1, dash="dash"),
        ), row=1, col=1)

    # 매매 포인트
    start_idx = len(df) - window
    for trade in portfolio.trades:
        if trade.entry_index < start_idx:
            continue

        entry_time = df.index[trade.entry_index] if trade.entry_index < len(df) else None
        exit_time = df.index[trade.exit_index] if trade.exit_index < len(df) else None

        if entry_time is not None:
            marker_symbol = "triangle-up" if trade.direction == Signal.LONG else "triangle-down"
            marker_color = "green" if trade.direction == Signal.LONG else "red"
            fig.add_trace(go.Scatter(
                x=[entry_time],
                y=[trade.entry_price],
                mode="markers",
                marker=dict(symbol=marker_symbol, size=12, color=marker_color),
                name=f"{trade.direction.value} Entry",
                showlegend=False,
            ), row=1, col=1)

        if exit_time is not None:
            fig.add_trace(go.Scatter(
                x=[exit_time],
                y=[trade.exit_price],
                mode="markers",
                marker=dict(symbol="x", size=10, color="black"),
                name="Exit",
                showlegend=False,
            ), row=1, col=1)

    # 거래량
    colors = ["green" if c >= o else "red"
              for c, o in zip(plot_df["close"], plot_df["open"])]
    fig.add_trace(go.Bar(
        x=plot_df.index,
        y=plot_df["volume"],
        marker_color=colors,
        name="Volume",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title="BTC 5min — Trade Overlay",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=800,
    )

    fig.write_html(output_path)
    logger.info("Trade overlay 저장: %s", output_path)


def plot_monthly_returns(
    portfolio: Portfolio,
    df: pd.DataFrame,
    output_path: str = "monthly_returns.html",
) -> None:
    """월별 수익률 히트맵을 생성한다.

    Args:
        portfolio: 백테스트 결과.
        df: OHLCV DataFrame.
        output_path: 저장 경로.
    """
    if not portfolio.trades:
        logger.warning("거래 내역이 없어 월별 히트맵을 생성할 수 없습니다.")
        return

    # 월 시작 시점 자본금 계산 (equity_curve에서 추출)
    monthly_start_capital: dict[str, float] = {}
    if portfolio.equity_curve and len(df) > 1:
        for i, eq in enumerate(portfolio.equity_curve):
            ts = df.index[i + 1]  # equity_curve는 index 1부터
            mk = str(ts)[:7]
            if mk not in monthly_start_capital:
                monthly_start_capital[mk] = eq

    # 거래별 월 할당
    monthly_pnl: dict[str, float] = {}
    for trade in portfolio.trades:
        month_key = trade.exit_time[:7]  # "2024-01"
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl

    if not monthly_pnl:
        return

    months = sorted(monthly_pnl.keys())
    values = []
    for m in months:
        base = monthly_start_capital.get(m, portfolio.initial_capital)
        values.append(monthly_pnl[m] / base * 100)

    fig = go.Figure(go.Bar(
        x=months,
        y=values,
        marker_color=["green" if v >= 0 else "red" for v in values],
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Monthly Returns (%)",
        xaxis_title="Month",
        yaxis_title="Return (%)",
        template="plotly_white",
    )

    fig.write_html(output_path)
    logger.info("월별 수익률 차트 저장: %s", output_path)


def plot_r_distribution(
    portfolio: Portfolio,
    output_path: str = "r_distribution.html",
) -> None:
    """R배수 히스토그램을 생성한다.

    Args:
        portfolio: 백테스트 결과.
        output_path: 저장 경로.
    """
    if not portfolio.trades:
        return

    r_multiples = [t.r_multiple for t in portfolio.trades]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["R-Multiple Distribution", "Hold Time Distribution"])

    fig.add_trace(go.Histogram(
        x=r_multiples,
        nbinsx=30,
        name="R-Multiple",
        marker_color="steelblue",
    ), row=1, col=1)

    hold_times = [t.hold_candles * 5 for t in portfolio.trades]  # 분 단위
    fig.add_trace(go.Histogram(
        x=hold_times,
        nbinsx=30,
        name="Hold Time (min)",
        marker_color="coral",
    ), row=1, col=2)

    fig.update_layout(
        title="Trade Distributions",
        template="plotly_white",
        showlegend=False,
    )

    fig.write_html(output_path)
    logger.info("거래 분포 차트 저장: %s", output_path)
