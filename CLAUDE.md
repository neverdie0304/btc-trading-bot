# Bitcoin 5분봉 자동매매 시스템 구현 프롬프트

## 프로젝트 개요

Binance API를 활용한 비트코인 5분봉 자동매매 시스템을 Python으로 구현한다.
**백테스트 엔진을 먼저 완성**한 뒤, 이후 라이브 트레이딩 모듈로 확장하는 구조로 설계한다.

---

## 디렉토리 구조

```
btc-trading-bot/
├── config/
│   ├── settings.py          # 전략 파라미터, API 키, 전역 설정
│   └── pairs.py             # 거래 페어 설정 (BTCUSDT 등)
├── data/
│   ├── fetcher.py           # Binance API로 과거 캔들 데이터 수집
│   ├── storage.py           # CSV/Parquet 로컬 저장/로드
│   └── cache/               # 캐시된 캔들 데이터 저장 폴더
├── indicators/
│   ├── ema.py               # EMA 9, 21, 50 계산
│   ├── rsi.py               # RSI(14) 계산
│   ├── vwap.py              # 당일 리셋 VWAP 계산
│   ├── atr.py               # ATR(14) 계산
│   └── volume.py            # 거래량 MA(20) 대비 비율 계산
├── strategy/
│   ├── signals.py           # 6가지 진입 조건 판별 → LONG / SHORT / NO_SIGNAL
│   ├── filters.py           # 매매 금지 구간 필터 (뉴스 시간, 데드존 등)
│   └── position.py          # 포지션 사이징, SL/TP 계산, 트레일링 스탑 로직
├── backtest/
│   ├── engine.py            # 백테스트 메인 엔진 (이벤트 드리븐)
│   ├── portfolio.py         # 가상 포트폴리오 관리 (잔고, 포지션, PnL)
│   └── metrics.py           # 성과 지표 계산 (PF, 승률, MDD, Sharpe 등)
├── analysis/
│   ├── report.py            # 백테스트 결과 리포트 생성
│   └── visualize.py         # 차트 시각화 (equity curve, 매매 포인트 등)
├── live/                    # (2차 구현 - 백테스트 검증 후)
│   ├── executor.py          # 실시간 주문 실행
│   ├── websocket_feed.py    # Binance WebSocket 실시간 데이터
│   └── kill_switch.py       # 일일 최대 손실 도달 시 자동 중지
├── tests/
│   ├── test_indicators.py   # 지표 계산 단위 테스트
│   ├── test_signals.py      # 시그널 로직 단위 테스트
│   └── test_backtest.py     # 백테스트 엔진 통합 테스트
├── main_backtest.py         # 백테스트 실행 진입점
├── main_live.py             # 라이브 트레이딩 실행 진입점 (2차)
├── requirements.txt         # 의존성 패키지
└── README.md                # 프로젝트 설명
```

---

## 기술 스택 & 의존성

```
python>=3.10
pandas>=2.0
numpy>=1.24
python-binance>=1.0.19      # Binance API 클라이언트
ta>=0.11.0                   # 보조지표 라이브러리 (검증용, 직접 구현 우선)
matplotlib>=3.7
plotly>=5.15                 # 인터랙티브 차트
tabulate>=0.9                # 리포트 테이블 출력
tqdm>=4.65                   # 진행바
python-dotenv>=1.0           # 환경변수 관리
pytest>=7.0                  # 테스트
```

---

## 핵심 구현 상세

### 1. 데이터 수집 (`data/fetcher.py`)

```
- Binance REST API로 BTCUSDT 5분봉 과거 데이터를 수집한다.
- python-binance 라이브러리의 client.get_historical_klines() 사용
- 수집 항목: open_time, open, high, low, close, volume, close_time, quote_volume
- 수집 기간: 사용자 지정 (기본값 최근 1년)
- API rate limit 준수: 요청 사이 적절한 sleep 적용
- 수집된 데이터는 data/cache/ 에 Parquet 형식으로 저장
- 이미 캐시된 기간은 스킵하고 누락 구간만 추가 수집
- 모든 시간은 UTC 기준으로 통일
```

### 2. 지표 계산 (`indicators/`)

각 지표는 pandas Series/DataFrame을 입력받아 결과를 반환하는 순수 함수로 구현한다.
ta 라이브러리에 의존하지 않고 직접 계산하되, ta 라이브러리 결과와 크로스체크하는 테스트를 작성한다.

```python
# EMA (ema.py)
- EMA(9), EMA(21), EMA(50) 계산
- pandas ewm 사용, adjust=False
- 입력: close 시리즈 / 출력: EMA 시리즈

# RSI (rsi.py)
- RSI(14) Wilder's smoothing 방식으로 계산
- 입력: close 시리즈 / 출력: RSI 시리즈 (0~100)

# VWAP (vwap.py)
- 당일 00:00 UTC 기준으로 리셋되는 VWAP 계산
- VWAP = cumsum(typical_price × volume) / cumsum(volume)
- typical_price = (high + low + close) / 3
- 입력: OHLCV DataFrame / 출력: VWAP 시리즈

# ATR (atr.py)
- ATR(14) Wilder's smoothing 방식
- True Range = max(H-L, |H-prev_C|, |L-prev_C|)
- 입력: OHLC DataFrame / 출력: ATR 시리즈

# Volume (volume.py)
- 20봉 단순이동평균(SMA) 대비 현재 거래량 비율
- 입력: volume 시리즈 / 출력: 비율 시리즈
```

### 3. 시그널 생성 (`strategy/signals.py`)

6가지 조건을 모두 확인하여 LONG, SHORT, NO_SIGNAL 중 하나를 반환한다.
**반드시 캔들 종가 확정(close) 시점 기준으로 판단**한다 — 진행 중인 캔들로 판단하지 않는다.

```
LONG 진입 조건 (6개 모두 충족):
  1. [추세 정배열]  EMA9 > EMA21 > EMA50
  2. [VWAP 위]     close > VWAP
  3. [RSI 범위]    40 <= RSI <= 65 AND RSI > RSI[1봉 전] (상승 중)
  4. [EMA9 돌파]   close > EMA9 AND prev_close <= EMA9
  5. [거래량 확인]  volume > volume_ma20 × 1.2
  6. [변동성 확인]  ATR(14) > median(ATR, 50봉)

SHORT 진입 조건 (6개 모두 충족):
  1. [추세 역배열]  EMA9 < EMA21 < EMA50
  2. [VWAP 아래]   close < VWAP
  3. [RSI 범위]    35 <= RSI <= 60 AND RSI < RSI[1봉 전] (하락 중)
  4. [EMA9 이탈]   close < EMA9 AND prev_close >= EMA9
  5. [거래량 확인]  volume > volume_ma20 × 1.2
  6. [변동성 확인]  ATR(14) > median(ATR, 50봉)
```

### 4. 필터 (`strategy/filters.py`)

아래 조건에 해당하면 시그널이 있어도 진입하지 않는다:

```
- FOMC/CPI/NFP 등 주요 경제지표 발표 전후 30분
  → config에 날짜 리스트를 수동 등록하는 방식으로 구현
  → 백테스트에서는 해당 시간대를 블랙아웃 처리
- 주말 토요일 00:00 ~ 일요일 06:00 UTC (유동성 부족 구간)
- ATR(14) < median(ATR, 50봉) × 0.5 (데드존 — 변동성 너무 낮음)
- 최근 연속 3패 시 → 다음 12봉(1시간) 동안 매매 금지 (쿨다운)
```

### 5. 포지션 관리 (`strategy/position.py`)

```python
# 포지션 사이징
risk_per_trade = 0.015  # 총 자본의 1.5%
sl_distance = 1.2 * ATR(14)  # 진입가 대비 SL 거리
position_size = (capital * risk_per_trade) / sl_distance

# 손절 / 익절
LONG:
  SL = entry_price - 1.2 * ATR(14)
  TP = entry_price + 2.0 * ATR(14)
SHORT:
  SL = entry_price + 1.2 * ATR(14)
  TP = entry_price - 2.0 * ATR(14)

# 트레일링 스탑 (봉 단위로 업데이트)
- 미실현 수익이 1.0R 도달 → SL을 진입가(Break-Even)로 이동
- 미실현 수익이 1.5R 도달 → SL을 +0.5R로 이동
- TP 도달 또는 트레일링 SL 히트 → 청산

# 추가 규칙
- 동시 포지션 최대 1개 (기존 포지션 있으면 신규 진입 불가)
- 한 방향 포지션 보유 중 반대 시그널 → 기존 청산 후 반대 진입 (선택적으로 config에서 on/off)
```

### 6. 백테스트 엔진 (`backtest/engine.py`)

이벤트 드리븐 방식으로 5분봉을 한 봉씩 순회하며 시뮬레이션한다.

```
핵심 루프:
  for each candle (시간순):
    1. 기존 포지션이 있으면 → 해당 봉의 H/L로 SL/TP 히트 여부 확인
       - SL 먼저 체크 (worst case 시나리오)
       - 히트 시 포지션 청산, PnL 기록
       - 히트 안 했으면 트레일링 스탑 업데이트
    2. 필터 체크 → 매매 금지 구간이면 skip
    3. 시그널 체크 → LONG/SHORT 시그널 발생 시 진입
    4. 포지션 사이징 계산, 진입 기록

비용 모델:
  - 수수료: 편도 0.04% (Binance VIP0 maker/taker 평균)
  - 슬리피지: 편도 0.02% (보수적 가정)
  - 총 왕복 비용: (0.04% + 0.02%) × 2 = 0.12%

주의사항:
  - 미래 데이터 참조(look-ahead bias) 절대 금지
  - 모든 판단은 현재 봉의 close 확정 후 기준
  - SL/TP 히트 판단은 다음 봉의 high/low가 아닌 현재 봉의 high/low 사용
    (진입은 close 시점이므로, SL/TP 체크는 다음 봉부터)
```

### 7. 성과 지표 (`backtest/metrics.py`)

```
계산할 지표:
  - 총 거래 횟수
  - 승률 (%) = 수익 거래 / 전체 거래
  - Profit Factor = 총 이익 / 총 손실 (절대값)
  - 평균 R배수 (실현 수익 / 초기 리스크)
  - 최대 연속 승/패
  - 총 수익률 (%)
  - 연환산 수익률 (CAGR)
  - 최대 낙폭 (MDD %)
  - Sharpe Ratio (무위험 수익률 0% 가정, 일별 수익률 기준)
  - Calmar Ratio = CAGR / MDD
  - 평균 보유 시간 (봉 수 & 분)
  - 롱/숏 별도 통계
```

### 8. 시각화 (`analysis/visualize.py`)

plotly로 인터랙티브 차트를 생성한다.

```
차트 1: Equity Curve
  - X축: 시간, Y축: 누적 자산
  - MDD 구간 음영 표시

차트 2: 매매 포인트 오버레이
  - 5분봉 캔들스틱 차트 위에
  - 롱 진입: 초록 삼각형 ▲, 숏 진입: 빨간 삼각형 ▼
  - 청산 포인트: × 마커
  - EMA 9/21/50, VWAP 라인 오버레이
  - 특정 기간 확대 가능

차트 3: 월별 수익률 히트맵

차트 4: 거래 분포
  - R배수 히스토그램
  - 보유 시간 히스토그램
```

### 9. 리포트 (`analysis/report.py`)

```
백테스트 완료 후 콘솔에 아래 형식으로 출력:

═══════════════════════════════════════════
  BTC 5min Multi-Confluence Momentum Scalp
  Backtest Report: 2024-01-01 ~ 2024-12-31
═══════════════════════════════════════════
  Total Trades:        487
  Win Rate:            62.4%
  Profit Factor:       1.67
  Avg R-Multiple:      0.38R
  Total Return:        +47.2%
  CAGR:                47.2%
  Max Drawdown:        -8.3%
  Sharpe Ratio:        2.14
  Calmar Ratio:        5.69
  Avg Hold Time:       35 min (7 candles)
  ─────────────────────────────────────────
  Long Trades:         281 (Win: 64.1%)
  Short Trades:        206 (Win: 60.2%)
  Max Consec Wins:     11
  Max Consec Losses:   4
  ─────────────────────────────────────────
  Commission Paid:     $584.40
  Slippage Cost:       $292.20
═══════════════════════════════════════════
```

---

## config/settings.py 기본값

```python
SETTINGS = {
    # 거래 페어
    "symbol": "BTCUSDT",
    "interval": "5m",

    # 지표 파라미터
    "ema_fast": 9,
    "ema_mid": 21,
    "ema_slow": 50,
    "rsi_period": 14,
    "atr_period": 14,
    "volume_ma_period": 20,
    "atr_median_lookback": 50,

    # 시그널 조건
    "rsi_long_min": 40,
    "rsi_long_max": 65,
    "rsi_short_min": 35,
    "rsi_short_max": 60,
    "volume_threshold": 1.2,       # 20봉 MA 대비 배수
    "deadzone_atr_ratio": 0.5,     # ATR < median × 이 값이면 데드존

    # 리스크 관리
    "risk_per_trade": 0.015,       # 1.5%
    "sl_atr_multiplier": 1.2,
    "tp_atr_multiplier": 2.0,
    "trailing_be_threshold": 1.0,  # 1R 도달 시 BE로 이동
    "trailing_step_threshold": 1.5,# 1.5R 도달 시 +0.5R로 이동
    "max_concurrent_positions": 1,
    "reverse_on_opposite_signal": False,

    # 비용
    "commission_rate": 0.0004,     # 편도 0.04%
    "slippage_rate": 0.0002,       # 편도 0.02%

    # 필터
    "cooldown_after_consecutive_losses": 3,
    "cooldown_candles": 12,        # 1시간
    "weekend_filter": True,
    "news_blackout_minutes": 30,

    # 백테스트
    "initial_capital": 10000,      # $10,000
    "backtest_start": "2024-01-01",
    "backtest_end": "2024-12-31",
}
```

---

## 구현 순서 (이 순서대로 진행)

```
Phase 1: 데이터 파이프라인
  1. data/fetcher.py — Binance에서 5분봉 데이터 수집
  2. data/storage.py — Parquet 저장/로드
  3. 테스트: 데이터 수집 후 결측치, 시간 연속성 검증

Phase 2: 지표 엔진
  4. indicators/ 전체 구현
  5. tests/test_indicators.py — ta 라이브러리 결과와 크로스 검증

Phase 3: 전략 로직
  6. strategy/signals.py — 시그널 생성
  7. strategy/filters.py — 필터링
  8. strategy/position.py — 포지션 사이징 & SL/TP/트레일링
  9. tests/test_signals.py — 알려진 패턴에서 올바른 시그널 나오는지 검증

Phase 4: 백테스트 엔진
  10. backtest/portfolio.py — 가상 포트폴리오
  11. backtest/engine.py — 메인 백테스트 루프
  12. backtest/metrics.py — 성과 지표
  13. tests/test_backtest.py — 간단한 시나리오로 엔진 검증

Phase 5: 분석 & 시각화
  14. analysis/report.py — 콘솔 리포트
  15. analysis/visualize.py — plotly 차트
  16. main_backtest.py — 전체 파이프라인 실행 스크립트
```

---

## 코드 품질 요구사항

```
- 모든 함수에 타입 힌트(type hints) 적용
- 모든 모듈에 docstring 작성
- 설정값은 하드코딩하지 말고 반드시 config/settings.py에서 불러올 것
- pandas SettingWithCopyWarning 발생하지 않도록 .copy() 적절히 사용
- 로깅: Python logging 모듈 사용 (print 사용 금지)
- 에러 처리: API 호출 실패, 데이터 누락 등에 대한 예외 처리
- 각 Phase 완료 후 해당 테스트를 실행하여 통과 확인
```

---

## .env 파일 (API 키 관리)

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

- .env 파일은 .gitignore에 포함
- config/settings.py에서 python-dotenv로 로드

---

## 최종 확인

구현 완료 후 아래를 실행하여 전체 파이프라인이 동작하는지 확인한다:

```bash
# 데이터 수집
python main_backtest.py --fetch-data --start 2024-01-01 --end 2024-12-31

# 백테스트 실행
python main_backtest.py --run

# 결과: 콘솔 리포트 출력 + plotly 차트 HTML 파일 생성
```