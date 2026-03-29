"""스캐너 전용 설정 모듈.

기존 SETTINGS(지표/시그널 파라미터)는 그대로 사용하고,
스캐너에만 필요한 추가 설정을 정의한다.
"""

SCANNER_SETTINGS: dict = {
    # ── 심볼 선정 ──
    "min_24h_quote_volume": 50_000_000,  # $50M 이상
    "max_24h_quote_volume": None,         # 상한 없음 (None=무제한)
    "quote_asset": "USDT",
    "symbol_refresh_interval_hours": 1,
    "excluded_symbols": [
        "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT",  # 스테이블코인
    ],

    # ── ATR 변동성 필터 ──
    "min_atr_pct": 1.0,           # ATR/Price >= 1.0% 코인만 진입 (None=필터 OFF)

    # ── 포지션 관리 ──
    "max_concurrent_positions": 1,
    "max_capital_per_position_pct": 0.95,  # 포지션당 최대 자본 95%
    "max_total_exposure_pct": 1.00,        # 총 노출 한도 100%

    # ── 리스크 ──
    "daily_max_loss_pct": -5.0,            # 일일 최대 손실 -5%

    # ── 캔들 ──
    "initial_candle_count": 150,           # 초기 로드 봉 수
    "max_candles_per_symbol": 200,         # 심볼당 메모리 유지 봉 수

    # ── WebSocket ──
    "ws_timeout_seconds": 120,             # 메시지 없으면 재연결
    "startup_throttle_per_sec": 10,        # 초기 로드 시 초당 심볼 수
}
