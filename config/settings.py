"""전역 설정 모듈. 전략 파라미터, API 키, 백테스트 설정을 관리한다."""

import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 기준 .env 로드
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

SETTINGS: dict = {
    # 거래 페어
    "symbol": "BTCUSDT",
    "interval": "5m",

    # 지표 파라미터 (실측 최적: EMA 5/13/34, vol=1.0 atr=1.5 조합 최고 수익)
    "ema_fast": 5,
    "ema_mid": 13,
    "ema_slow": 34,
    "rsi_period": 14,
    "atr_period": 14,
    "volume_ma_period": 20,
    "atr_median_lookback": 50,

    # 시그널 조건 (실측 최적: vol=1.0, atr=1.5 → +5380% / RSI 35-70 / 40-60)
    "rsi_long_min": 35,
    "rsi_long_max": 70,
    "rsi_short_min": 40,
    "rsi_short_max": 60,
    "volume_threshold": 1.0,
    "deadzone_atr_ratio": 0.5,
    "atr_signal_multiplier": 1.5,

    # 리스크 관리 (SL=1.2 TP=3.0 / BE=1.0 STEP=1.5 / risk=5% lev=20x)
    "leverage": 20,
    "risk_per_trade": 0.05,
    "sl_atr_multiplier": 1.2,
    "tp_atr_multiplier": 3.0,
    "trailing_be_threshold": 1.0,
    "trailing_step_threshold": 1.5,
    "max_concurrent_positions": 1,
    "reverse_on_opposite_signal": False,

    # 분할 익절 (3/4@2.5R → 나머지 1/4는 4R, SL→2.5R)
    # partial_tp_r = tp_atr_multiplier / sl_atr_multiplier = 2.5R (자동 계산)
    "partial_tp_enabled": True,      # 분할 익절 활성화 여부
    "partial_tp_fraction": 0.75,     # 첫 TP에서 익절할 비율 (75%)
    "final_tp_r": 4.0,               # 잔여 물량 최종 TP (R 단위)

    # 비용 (USDC-M: maker 0%, taker 0.05%)
    "maker_fee": 0.0000,           # limit order (진입, TP 청산) 0%
    "taker_fee": 0.0004,           # stop-market (SL 청산) 0.04%
    "slippage_rate": 0.0000,

    # 필터
    "daily_max_losses": 0,          # 하루 N회 손절 시 당일 매매 중단 (0=무제한)
    "daily_max_trades": 0,           # 하루 최대 매매 횟수 (0=무제한)
    "cooldown_after_consecutive_losses": 3,
    "cooldown_candles": 12,
    "weekend_filter": True,
    "news_blackout_minutes": 30,
    "bad_hours_utc": [3, 5, 10, 12, 16, 19, 21],

    # 뉴스 블랙아웃 날짜 (UTC, 수동 등록)
    "news_events": [],

    # 백테스트
    "initial_capital": 10000,
    "backtest_start": "2020-01-01",
    "backtest_end": "2026-02-25",

    # 데이터 경로
    "cache_dir": str(_PROJECT_ROOT / "data" / "cache"),
}
