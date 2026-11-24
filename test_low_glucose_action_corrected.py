"""Pytest unit tests for low_glucose_action.py achieving MC/DC coverage."""

import pytest
from low_glucose_action import (
    GlucoseStatus,
    CurrentTemp,
    Profile,
    RequestedTemp,
    roundBasal,
    determineLowGlucoseAction,
)


# ---------------------------------------------------------------------------
# Helper: build kwargs matching determineLowGlucoseAction's real signature.
# temp_rate / temp_duration are packed into a CurrentTemp instance.
# ---------------------------------------------------------------------------

def make_args(
    bg=65,
    delta=-3.0,
    min_delta=-2.5,
    expected_delta=-1.0,
    threshold=75,
    basal=0.8,
    temp_rate=0.8,
    temp_duration=10,
    max_basal=4.0,
):
    return dict(
        bg=bg,
        glucoseStatus=GlucoseStatus(delta),
        minDelta=min_delta,
        expectedDelta=expected_delta,
        threshold=threshold,
        basal=basal,
        currentTemp=CurrentTemp(temp_rate, temp_duration),
        profile=Profile(max_basal),
    )


# ===========================================================================
# roundBasal
# ===========================================================================

class TestRoundBasal:
    @pytest.fixture
    def profile(self):
        return Profile(maxBasal=4.0)

    def test_negative_rate_returns_zero(self, profile):
        assert roundBasal(-1.0, profile) == 0.0

    def test_rate_above_max_basal_clamped(self, profile):
        assert roundBasal(5.0, profile) == 4.0

    def test_rate_exactly_at_max_basal(self, profile):
        assert roundBasal(4.0, profile) == 4.0

    def test_rounding_down(self, profile):
        assert roundBasal(0.82, profile) == pytest.approx(0.80)

    def test_rounding_up(self, profile):
        assert roundBasal(0.83, profile) == pytest.approx(0.85)

    def test_rounding_half(self, profile):
        assert roundBasal(0.825, profile) == pytest.approx(0.85)

    def test_zero_rate(self, profile):
        assert roundBasal(0.0, profile) == 0.0


# ===========================================================================
# determineLowGlucoseAction
#
# D1: bg < threshold
# D2: C2a or C2b or C2c   (suspend)
#     C2a: delta<=0 and minDelta<=0
#     C2b: delta<expectedDelta and minDelta<expectedDelta
#     C2c: bg < 60
# D3: currentTemp.duration > 15 and
#     roundBasal(basal) == roundBasal(currentTemp.rate)
# ===========================================================================


class TestD1_BelowThreshold:
    """MC/DC for D1: bg < threshold."""

    def test_bg_below_threshold_enters_block(self):
        result = determineLowGlucoseAction(**make_args(bg=65, threshold=75))
        assert result.rate == 0.0
        assert "BG 65 < 75" in result.reason

    def test_bg_equal_threshold_no_action(self):
        result = determineLowGlucoseAction(**make_args(bg=75, threshold=75))
        assert result.rate is None
        assert result.duration is None

    def test_bg_above_threshold_no_action(self):
        result = determineLowGlucoseAction(**make_args(bg=110, threshold=75))
        assert result.rate is None
        assert result.reason == ""


class TestD2_SuspendCondition:
    """MC/DC for D2 = C2a or C2b or C2c."""

    def test_no_suspend_when_all_subconditions_false(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=2.0, min_delta=1.5, expected_delta=-1.0,
            basal=0.8, temp_rate=0.8, temp_duration=10,
        ))
        assert result.rate == 0.8
        assert "setting current basal" in result.reason

    def test_suspend_via_c2a_only(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=-3.0, min_delta=-2.5, expected_delta=-10.0,
        ))
        assert result.rate == 0.0
        assert result.duration == 30
        assert "setting 0 temp" in result.reason

    def test_c2a_false_when_delta_positive(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=1.0, min_delta=-2.5, expected_delta=-10.0,
            temp_duration=10,
        ))
        assert result.rate == 0.8

    def test_c2a_false_when_min_delta_positive(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=-3.0, min_delta=1.0, expected_delta=-10.0,
            temp_duration=10,
        ))
        assert result.rate == 0.8

    def test_suspend_via_c2b_only(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=0.5, min_delta=0.3, expected_delta=1.0,
        ))
        assert result.rate == 0.0
        assert "setting 0 temp" in result.reason

    def test_c2b_false_when_delta_not_less_than_expected(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=0.4, min_delta=0.6, expected_delta=0.5,
            temp_duration=10,
        ))
        assert result.rate == 0.8

    def test_c2b_false_when_min_delta_not_less_than_expected(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=0.5, min_delta=1.0, expected_delta=1.0,
            temp_duration=10,
        ))
        assert result.rate == 0.8

    def test_suspend_via_c2c_only(self):
        result = determineLowGlucoseAction(**make_args(
            bg=55, delta=0.5, min_delta=0.3, expected_delta=0.5,
        ))
        assert result.rate == 0.0
        assert "setting 0 temp" in result.reason

    def test_c2c_false_at_bg_60(self):
        result = determineLowGlucoseAction(**make_args(
            bg=60, delta=0.5, min_delta=0.3, expected_delta=0.5,
            temp_duration=10,
        ))
        assert result.rate == 0.8

    def test_c2a_true_with_min_delta_zero_boundary(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=-1.0, min_delta=0.0, expected_delta=-10.0,
        ))
        assert result.rate == 0.0

    def test_delta_zero_boundary_c2a(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=0.0, min_delta=-1.0, expected_delta=-10.0,
        ))
        assert result.rate == 0.0


class TestD3_CurrentTempAppropriate:
    """MC/DC for D3 = duration>15 AND rates_equal (recovery path only)."""

    def _recovery_args(self, *, temp_duration=20, temp_rate=0.8, basal=0.8):
        # Always rebuild a CurrentTemp; never inject temp_* keys directly.
        return make_args(
            bg=65, delta=2.0, min_delta=1.5, expected_delta=-1.0,
            basal=basal, temp_rate=temp_rate, temp_duration=temp_duration,
        )

    def test_d3_true_keeps_current_temp(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.8, basal=0.8)
        )
        assert result.rate is None
        assert result.duration is None
        assert "~ req" in result.reason

    def test_d3_true_via_rounded_equality(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.78, basal=0.82)
        )
        assert result.rate is None
        assert "~ req" in result.reason

    def test_d3_false_duration_at_15(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=15, temp_rate=0.8, basal=0.8)
        )
        assert result.rate == 0.8
        assert result.duration == 30
        assert "setting current basal" in result.reason

    def test_d3_false_duration_below_15(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=10, temp_rate=0.8, basal=0.8)
        )
        assert result.rate == 0.8
        assert result.duration == 30

    def test_d3_false_rate_differs(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.0, basal=0.8)
        )
        assert result.rate == 0.8
        assert result.duration == 30

    def test_d3_false_rate_slightly_differs_after_rounding(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.9, basal=0.8)
        )
        assert result.rate == 0.8

    def test_d3_true_duration_16(self):
        result = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=16, temp_rate=0.8, basal=0.8)
        )
        assert result.rate is None
        assert "~ req" in result.reason

    def test_d3_rate_operand_flips_decision(self):
        r_true = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.8, basal=0.8)
        )
        assert r_true.rate is None
        r_false = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.5, basal=0.8)
        )
        assert r_false.rate == 0.8

    def test_d3_duration_operand_flips_decision(self):
        r_true = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=20, temp_rate=0.8, basal=0.8)
        )
        assert r_true.rate is None
        r_false = determineLowGlucoseAction(
            **self._recovery_args(temp_duration=15, temp_rate=0.8, basal=0.8)
        )
        assert r_false.rate == 0.8


# ===========================================================================
# Integration / reason-string sanity
# ===========================================================================

class TestReasonAndIntegration:
    def test_suspend_reason_includes_bg_and_threshold(self):
        result = determineLowGlucoseAction(**make_args(
            bg=50, threshold=75, delta=-3.0, min_delta=-2.5,
        ))
        assert "BG 50 < 75" in result.reason
        assert "setting 0 temp" in result.reason

    def test_recovery_reason_includes_delta_values(self):
        result = determineLowGlucoseAction(**make_args(
            bg=65, delta=2.0, min_delta=1.5, expected_delta=-1.0,
            temp_duration=10,
        ))
        assert "delta 2.0" in result.reason
        assert "min delta 1.50" in result.reason

    def test_no_action_reason_empty(self):
        result = determineLowGlucoseAction(**make_args(bg=100, threshold=75))
        assert result.reason == ""
        assert result.rate is None
        assert result.duration is None

    def test_main_scenarios_run(self, capsys):
        import runpy
        runpy.run_module("low_glucose_action", run_name="__main__")
        out = capsys.readouterr().out
        assert "Scenario 1" in out
        assert "Scenario 2" in out
        assert "Scenario 3" in out