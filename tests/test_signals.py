"""시그널 로직 및 필터 단위 테스트.

현재 설정값 기준:
  risk_per_trade=0.05, sl_atr_multiplier=1.2, tp_atr_multiplier=3.0
  volume_threshold=1.0, atr_signal_multiplier=1.5
  rsi_long: 35~70, rsi_short: 40~60
"""

import pandas as pd
import pytest

from config.settings import SETTINGS
from strategy.signals import Signal, check_long_conditions, check_short_conditions
from strategy.filters import (
    is_weekend_filter,
    is_news_blackout,
    is_deadzone,
    is_cooldown,
    should_filter,
)
from strategy.position import (
    create_position,
    check_sl_tp_hit,
    update_trailing_stop,
    calculate_position_size,
)

# 현재 설정값에서 파생되는 상수
_SL_MULT = SETTINGS["sl_atr_multiplier"]   # 1.2
_TP_MULT = SETTINGS["tp_atr_multiplier"]   # 3.0
_RISK = SETTINGS["risk_per_trade"]         # 0.05
_ATR_SIG_MULT = SETTINGS.get("atr_signal_multiplier", 1.0)  # 1.5


class TestLongConditions:
    """LONG 시그널 조건 테스트."""

    def _make_row(self, **overrides) -> pd.Series:
        """기본 LONG 조건 충족 row를 생성한다.

        atr=200, atr_median=80 → 200 > 80*1.5=120 ✓
        """
        base = {
            "close": 42000, "ema_fast": 41990, "ema_mid": 41900, "ema_slow": 41800,
            "vwap": 41500, "rsi": 55, "atr": 200, "atr_median": 80,
            "volume_ratio": 1.5,
        }
        base.update(overrides)
        return pd.Series(base)

    def _make_prev_row(self, **overrides) -> pd.Series:
        base = {
            "close": 41985, "ema_fast": 41990, "rsi": 50,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_all_conditions_met(self) -> None:
        """6가지 조건 모두 충족 시 True."""
        row = self._make_row()
        prev = self._make_prev_row()
        assert check_long_conditions(row, prev) is True

    def test_fail_ema_order(self) -> None:
        """EMA 정배열 미충족 시 False."""
        row = self._make_row(ema_fast=41800, ema_mid=41900)
        prev = self._make_prev_row()
        assert check_long_conditions(row, prev) is False

    def test_fail_vwap(self) -> None:
        """close < VWAP이면 False."""
        row = self._make_row(close=41000, vwap=41500)
        prev = self._make_prev_row()
        assert check_long_conditions(row, prev) is False

    def test_fail_rsi_range(self) -> None:
        """RSI 범위 초과 시 False."""
        row = self._make_row(rsi=75)
        prev = self._make_prev_row()
        assert check_long_conditions(row, prev) is False

    def test_fail_volume(self) -> None:
        """거래량 부족 시 False."""
        row = self._make_row(volume_ratio=0.8)
        prev = self._make_prev_row()
        assert check_long_conditions(row, prev) is False

    def test_fail_atr_signal(self) -> None:
        """ATR < median * atr_signal_multiplier이면 False."""
        row = self._make_row(atr=100, atr_median=80)
        prev = self._make_prev_row()
        assert check_long_conditions(row, prev) is False


class TestShortConditions:
    """SHORT 시그널 조건 테스트."""

    def _make_row(self, **overrides) -> pd.Series:
        base = {
            "close": 41700, "ema_fast": 41800, "ema_mid": 41900, "ema_slow": 42000,
            "vwap": 42100, "rsi": 45, "atr": 200, "atr_median": 80,
            "volume_ratio": 1.5,
        }
        base.update(overrides)
        return pd.Series(base)

    def _make_prev_row(self, **overrides) -> pd.Series:
        base = {
            "close": 41810, "ema_fast": 41800, "rsi": 50,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_all_conditions_met(self) -> None:
        """6가지 조건 모두 충족 시 True."""
        row = self._make_row()
        prev = self._make_prev_row()
        assert check_short_conditions(row, prev) is True

    def test_fail_ema_order(self) -> None:
        """EMA 역배열 미충족 시 False."""
        row = self._make_row(ema_fast=42100)
        prev = self._make_prev_row()
        assert check_short_conditions(row, prev) is False


class TestFilters:
    """필터 테스트."""

    def test_weekend_saturday(self) -> None:
        ts = pd.Timestamp("2024-01-06 12:00:00", tz="UTC")
        assert is_weekend_filter(ts) is True

    def test_weekend_sunday_early(self) -> None:
        ts = pd.Timestamp("2024-01-07 03:00:00", tz="UTC")
        assert is_weekend_filter(ts) is True

    def test_weekend_sunday_after(self) -> None:
        ts = pd.Timestamp("2024-01-07 07:00:00", tz="UTC")
        assert is_weekend_filter(ts) is False

    def test_weekday_pass(self) -> None:
        ts = pd.Timestamp("2024-01-08 12:00:00", tz="UTC")
        assert is_weekend_filter(ts) is False

    def test_deadzone(self) -> None:
        assert is_deadzone(30, 100) is True
        assert is_deadzone(60, 100) is False

    def test_cooldown(self) -> None:
        assert is_cooldown(3, 5) is True
        assert is_cooldown(3, 15) is False
        assert is_cooldown(2, 5) is False


class TestPosition:
    """포지션 관리 테스트.

    현재 설정: risk=5%, SL=1.2×ATR, TP=3.0×ATR
    ATR=100 → sl_dist=120, tp_dist=300
    capital=10000 → risk_amount=500, size=500/120≈4.167
    """

    def test_position_sizing(self) -> None:
        """포지션 크기가 올바르게 계산되는지 확인."""
        size, sl_dist = calculate_position_size(10000, 42000, 100)
        expected_sl_dist = _SL_MULT * 100  # 120
        expected_risk = 10000 * _RISK       # 500
        assert sl_dist == pytest.approx(expected_sl_dist)
        assert size == pytest.approx(expected_risk / expected_sl_dist)

    def test_long_sl_tp(self) -> None:
        """LONG 포지션 SL/TP 가격 확인."""
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        assert pos.sl_price == pytest.approx(42000 - _SL_MULT * 100)
        assert pos.tp_price == pytest.approx(42000 + _TP_MULT * 100)

    def test_short_sl_tp(self) -> None:
        """SHORT 포지션 SL/TP 가격 확인."""
        pos = create_position(Signal.SHORT, 42000, 100, 10000)
        assert pos.sl_price == pytest.approx(42000 + _SL_MULT * 100)
        assert pos.tp_price == pytest.approx(42000 - _TP_MULT * 100)

    def test_sl_hit_long(self) -> None:
        """LONG SL 히트 확인."""
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        sl = 42000 - _SL_MULT * 100  # 41880
        hit, price, reason = check_sl_tp_hit(pos, 42050, sl - 10)
        assert hit is True
        assert reason == "SL"

    def test_tp_hit_long(self) -> None:
        """LONG TP 히트 확인."""
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        tp = 42000 + _TP_MULT * 100  # 42300
        hit, price, reason = check_sl_tp_hit(pos, tp + 10, 42050)
        assert hit is True
        assert reason == "TP"

    def test_no_hit(self) -> None:
        """SL/TP 미히트 확인."""
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        hit, _, _ = check_sl_tp_hit(pos, 42100, 41950)
        assert hit is False

    def test_trailing_stop_break_even(self) -> None:
        """1R 도달 시 BE로 이동 확인."""
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        r_unit = _SL_MULT * 100  # 120
        update_trailing_stop(pos, 42000 + r_unit + 10)
        assert pos.sl_price == pytest.approx(42000)
        assert pos.trailing_state == "break_even"

    def test_trailing_stop_step(self) -> None:
        """1.5R 도달 시 +0.5R로 이동 확인."""
        pos = create_position(Signal.LONG, 42000, 100, 10000)
        r_unit = _SL_MULT * 100  # 120
        update_trailing_stop(pos, 42000 + r_unit + 10)
        update_trailing_stop(pos, 42000 + 1.5 * r_unit + 10)
        assert pos.sl_price == pytest.approx(42000 + 0.5 * r_unit)
        assert pos.trailing_state == "trailing"
