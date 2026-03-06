# BTC 5min Multi-Confluence Momentum Scalp Bot

Binance Futures BTCUSDT 5분봉 자동매매 시스템.

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

### 비교 분석 스크립트
```bash
python compare_filters.py        # 필터 조합 4가지 비교
python compare_leverage.py       # 레버리지별 (5~35x) 비교
python compare_leverage_cap.py   # 레버리지 CAP 유무 비교
python compare_trailing.py       # 트레일링 스탑 설정별 비교
```

---

## 라이브 트레이딩

### 페이퍼 트레이딩 (실제 주문 없음, 추천 — 먼저 검증)
```bash
python main_live.py --mode paper
```

### 실제 트레이딩
```bash
python main_live.py --mode live
```

### 자본금 수동 지정 (paper 모드 테스트용)
```bash
python main_live.py --mode paper --capital 500
```

### 종료
`Ctrl+C` — graceful shutdown (미체결 주문 정리 후 종료)

---

## 주요 설정 (`config/settings.py`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `ema_fast/mid/slow` | 5/13/34 | EMA 기간 |
| `atr_signal_multiplier` | 1.5 | 신호 발생 조건 (ATR > 1.5 × median) |
| `leverage` | 20 | 최대 레버리지 (동적으로 조절됨) |
| `risk_per_trade` | 0.05 | 거래당 리스크 5% |
| `sl_atr_multiplier` | 1.2 | SL = 1.2 × ATR |
| `tp_atr_multiplier` | 3.0 | TP = 3.0 × ATR |
| `trailing_be_threshold` | 1.0 | 1R 도달 시 SL → 진입가(Break-Even) |
| `trailing_step_threshold` | 1.5 | 1.5R 도달 시 SL → +0.5R |
| `weekend_filter` | True | 토요일~일요일 06:00 UTC 매매 금지 |
| `bad_hours_utc` | [3,5,10,12,16,19,21] | 손실 시간대 매매 금지 |
| `maker_fee` | 0.0000 | 진입/TP(limit) 수수료 |
| `taker_fee` | 0.0004 | SL(stop-market) 수수료 |

---

## 백테스트 성과 (2020-01-01 ~ 2026-02-25, 수수료 포함)

- 거래수: 407건 (약 5일에 1번)
- 승률: 39.6%
- Profit Factor: 1.58
- **총 수익률: +5,380%**
- MDD: -45.9%
- Sharpe: 1.38

---

## 디렉토리 구조

```
btc-trading-bot/
├── config/settings.py       # 전략 파라미터 & 전역 설정
├── data/
│   ├── fetcher.py           # Binance API 데이터 수집
│   ├── storage.py           # Parquet 저장/로드
│   └── cache/               # 캔들 데이터 캐시
├── strategy/
│   ├── signals.py           # 진입 시그널 (LONG/SHORT/NO_SIGNAL)
│   ├── filters.py           # 매매 금지 필터 (주말, 손실시간대 등)
│   └── position.py          # 포지션 사이징, SL/TP, 트레일링 스탑
├── backtest/
│   ├── engine.py            # 백테스트 엔진
│   ├── portfolio.py         # 가상 포트폴리오
│   └── metrics.py           # 성과 지표
├── live/
│   ├── executor.py          # 주문 실행 (Paper/Live)
│   ├── state.py             # 봇 상태 관리
│   ├── logger_db.py         # SQLite 거래 기록
│   ├── candle_manager.py    # 실시간 캔들 관리
│   └── kill_switch.py       # 일일 손실 한도 자동 중지
├── analysis/
│   ├── report.py            # 콘솔 리포트
│   └── visualize.py         # Plotly 차트 (equity curve 등)
├── main_backtest.py         # 백테스트 실행 진입점
├── main_live.py             # 라이브 봇 실행 진입점
└── compare_*.py             # 각종 비교 분석 스크립트
```

---

## 로그 & DB

- 실행 로그: `data/live_bot.log`
- 거래 기록 DB: `data/trades.db` (SQLite)
