# BTC 5min Multi-Confluence Momentum Scalp Bot

Binance Futures BTCUSDT 5분봉 자동매매 시스템.
6가지 조건(EMA 정배열, VWAP, RSI, EMA 돌파, 거래량, ATR)을 동시에 충족할 때만 진입하는 고선택적 모멘텀 스캘핑 전략.

---

## 환경 설정

### 1. 패키지 설치
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정
프로젝트 루트에 `.env` 파일 생성:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

---

## 백테스트

### 데이터 수집
```bash
# 기본 (settings.py의 backtest_start/end 기준)
python main_backtest.py --fetch-data

# 기간 직접 지정
python main_backtest.py --fetch-data --start 2022-01-01 --end 2024-12-31
```

### 백테스트 실행
```bash
python main_backtest.py --run
```

### 데이터 수집 + 백테스트 한 번에
```bash
python main_backtest.py --fetch-data --run --start 2020-01-01 --end 2026-02-25
```

---

## 라이브 트레이딩

### 페이퍼 트레이딩 (실제 주문 없음)
```bash
python main_live.py --mode paper
```

### 실제 트레이딩
```bash
python main_live.py --mode live
```

### 자본금 수동 지정
```bash
python main_live.py --mode paper --capital 500
```

### 대시보드 (FastAPI)
```bash
python main_dashboard.py
```

### 종료
`Ctrl+C` — graceful shutdown (미체결 주문 정리 후 종료)

---

## 주요 설정 (`config/settings.py`)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `ema_fast/mid/slow` | 5 / 13 / 34 | EMA 기간 |
| `atr_signal_multiplier` | 1.5 | 진입 조건: ATR > median × 1.5 |
| `leverage` | 20 | 최대 레버리지 (트레이드별 동적 조절) |
| `risk_per_trade` | 5% | 거래당 리스크 |
| `sl_atr_multiplier` | 1.2 | SL = 1.2 × ATR |
| `tp_atr_multiplier` | 3.0 | TP = 3.0 × ATR |
| `trailing_be_threshold` | 1.0R | 1R 도달 시 SL → Break-Even |
| `trailing_step_threshold` | 1.5R | 1.5R 도달 시 SL → +0.5R |
| `partial_tp_fraction` | 75% | 분할 익절 비율 (2.5R에서 75% 익절) |
| `final_tp_r` | 4.0R | 잔여 25% 최종 TP |
| `maker_fee` | 0% | 진입/TP (limit order) |
| `taker_fee` | 0.04% | SL (stop-market) |
| `weekend_filter` | ON | 토~일 06:00 UTC 매매 금지 |
| `bad_hours_utc` | [3,5,10,12,16,19,21] | 손실 시간대 매매 금지 |

---

## 백테스트 성과 (2020-01-01 ~ 2026-02-25, 수수료 포함)

```
  Total Trades:        407
  Win Rate:            39.6%
  Profit Factor:       1.58
  Total Return:        +5,379.6%
  CAGR:                91.7%
  Max Drawdown:        -45.9%
  Sharpe Ratio:        1.38
  Calmar Ratio:        2.00
  Avg Hold Time:       39 min (8 candles)
```

---

## 프로젝트 구조

```
btc-trading-bot/
├── config/
│   ├── settings.py             # 전략 파라미터 & 전역 설정
│   ├── pairs.py                # 거래 페어 설정
│   └── best_params.json        # 최적화된 파라미터 백업
├── data/
│   ├── fetcher.py              # Binance API 데이터 수집
│   ├── storage.py              # Parquet 저장/로드
│   └── cache/                  # 캔들 데이터 캐시
├── indicators/
│   ├── ema.py                  # EMA 5/13/34
│   ├── rsi.py                  # RSI(14)
│   ├── atr.py                  # ATR(14)
│   ├── vwap.py                 # 당일 리셋 VWAP
│   └── volume.py               # 거래량 MA(20) 대비 비율
├── strategy/
│   ├── signals.py              # 6가지 진입 조건 → LONG/SHORT/NO_SIGNAL
│   ├── filters.py              # 매매 금지 필터 (주말, 뉴스, 데드존)
│   └── position.py             # 포지션 사이징, SL/TP, 트레일링 스탑
├── backtest/
│   ├── engine.py               # 이벤트 드리븐 백테스트 엔진
│   ├── portfolio.py            # 가상 포트폴리오 (잔고, PnL)
│   ├── metrics.py              # 성과 지표 (PF, 승률, MDD, Sharpe)
│   └── rotation.py             # 멀티 심볼 로테이션 백테스트
├── live/
│   ├── executor.py             # 주문 실행 (Paper/Live)
│   ├── state.py                # 봇 상태 관리 & 영속화
│   ├── candle_manager.py       # WebSocket 실시간 캔들 관리
│   ├── logger_db.py            # SQLite 거래 기록
│   └── kill_switch.py          # 일일 손실 한도 자동 중지
├── analysis/
│   ├── report.py               # 콘솔 리포트 출력
│   └── visualize.py            # Plotly 차트 (equity curve, 히트맵)
├── dashboard/
│   ├── app.py                  # FastAPI 모니터링 서버
│   └── bot_manager.py          # 멀티 봇 관리
├── scripts/
│   ├── close_position.py       # 수동 포지션 청산
│   ├── visualize_signals.py    # 일별 시그널 차트 생성
│   ├── view_month.py           # 월별 트레이드 뷰
│   ├── run_power_viz.py        # 알트코인 시각화
│   └── experiments/            # 파라미터 최적화 스크립트
├── tests/
│   ├── test_indicators.py      # 지표 단위 테스트
│   ├── test_signals.py         # 시그널 로직 테스트
│   └── test_backtest.py        # 백테스트 엔진 테스트
├── docs/
│   └── strategy_pinescript.pine # TradingView PineScript
├── main_backtest.py            # 백테스트 실행 진입점
├── main_live.py                # 라이브 봇 실행 진입점
└── main_dashboard.py           # 대시보드 실행 진입점
```

---

## 수수료 모델

| 주문 유형 | 수수료 | 비고 |
|-----------|--------|------|
| 진입 (LIMIT) | maker 0% | Binance USDC-M |
| TP 청산 (TAKE_PROFIT LIMIT) | maker 0% | |
| SL 청산 (STOP_MARKET) | taker 0.04% | |
| 트레일링 SL (STOP_LIMIT) | maker 0% | |

---

## 테스트

```bash
python -m pytest tests/ -v
```
