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
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def profile():
    return Profile(maxBasal=4.0)


def make_args(
    bg=65,
    delta=-3.0,
    min_delta=-2.5,
    expected_delta=-1.0,
    threshold=75,
    basal=0.8,
    temp_rate=0.8,
    temp_duration=10,
    profile=None,
):
    if profile is None:
        profile = Profile(4.0)
    return dict(
        bg=bg,
        glucoseStatus=GlucoseStatus(delta),
        minDelta=min_delta,
        expectedDelta=expected_delta,
        threshold=threshold,
        basal=basal,
        currentTemp=CurrentTemp(temp_rate, temp_duration),
        profile=profile,
    )


# ===========================================================================
# Tests for roundBasal
# ===========================================================================

class TestRoundBasal:
    def test_negative_rate_returns_zero(self, profile):
        assert roundBasal(-1.0, profile) == 0.0

    def test_rate_above_max_basal_clamped(self, profile):
        assert roundBasal(5.0, profile) == 4.0

    def test_rate_exactly_at_max_basal(self, profile):
        assert roundBasal(4.0, profile) == 4.0

    def test_rounding_down(self, profile):
        # 0.82 * 20 = 16.4 -> floor(16.9) = 16 -> 0.80
        assert roundBasal(0.82, profile) == pytest.approx(0.80)

    def test_rounding_up(self, profile):
        # 0.83 * 20 = 16.6 -> floor(17.1) = 17 -> 0.85
        assert roundBasal(0.83, profile) == pytest.approx(0.85)

    def test_rounding_half(self, profile):
        # 0.825 * 20 = 16.5 -> floor(17.0) = 17 -> 0.85
        assert roundBasal(0.825, profile) == pytest.approx(0.85)

    def test_zero_rate(self, profile):
        assert roundBasal(0.0, profile) == 0.0


# ===========================================================================
# Tests for determineLowGlucoseAction
#
# Decision structure (numbered):
#
# D1: bg < threshold
#     If FALSE -> no action (RequestedTemp with rate=None)
#     If TRUE -> enter block
#
# D2 (suspend condition), composed of three OR-ed sub-conditions:
#     C2a: glucoseStatus.delta <= 0 AND minDelta <= 0
#     C2b: glucoseStatus.delta < expectedDelta AND minDelta < expectedDelta
#     C2c: bg < 60
#     D2 = C2a or C2b or C2c
#     If TRUE -> suspend (rate=0, duration=30, "setting 0 temp")
#     If FALSE -> continue to recovery path
#
# D3: currentTemp.duration > 15 AND
#     roundBasal(basal) == roundBasal(currentTemp.rate)
#     If TRUE -> no action (keep current temp), rate=None
#     If FALSE -> set basal as temp
# ===========================================================================


class TestD1_BelowThreshold:
    """MC/DC for D1: bg < threshold."""

    def test_bg_below_threshold_enters_block(self):
        """D1 TRUE: bg < threshold -> action taken."""
        args = make_args(bg=65, threshold=75, delta=-3.0, min_delta=-2.5)
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.0  # suspend triggered
        assert "BG 65 < 75" in result.reason

    def test_bg_equal_threshold_no_action(self):
        """D1 FALSE (boundary): bg == threshold -> no action."""
        args = make_args(bg=75, threshold=75)
        result = determineLowGlucoseAction(**args)
        assert result.rate is None
        assert result.duration is None

    def test_bg_above_threshold_no_action(self):
        """D1 FALSE: bg > threshold -> no action."""
        args = make_args(bg=110, threshold=75)
        result = determineLowGlucoseAction(**args)
        assert result.rate is None
        assert result.reason == ""


class TestD2_SuspendCondition:
    """MC/DC for D2 = C2a or C2b or C2c.

    To achieve MC/DC for an OR of 3, we need cases where each condition
    independently flips the decision while others are held fixed.
    """

    # --- Baseline "no suspend" case: all three sub-conditions FALSE ---
    #   C2a: delta<=0 AND minDelta<=0        -> (delta=2>0) F
    #   C2b: delta<expected AND minDelta<exp -> (2<-1)F
    #   C2c: bg<60                           -> (65<60) F
    #   D2 = F or F or F = FALSE  -> recovery path
    def test_no_suspend_when_all_subconditions_false(self):
        args = make_args(
            bg=65,
            delta=2.0,
            min_delta=1.5,
            expected_delta=-1.0,
            temp_duration=10,  # ensures D3 FALSE path for basal temp
            temp_rate=0.8,
            basal=0.8,
        )
        result = determineLowGlucoseAction(**args)
        # Recovery path, D3 false -> set basal as temp
        assert result.rate == 0.8
        assert result.duration == 30
        assert "setting current basal" in result.reason

    # --- C2a alone flips D2 to TRUE ---
    #   C2a: delta=-3<=0 AND minDelta=-2.5<=0 -> T
    #   C2b: -3<-1 T AND -2.5<-1 T -> T  (but we want to isolate C2a)
    # To isolate C2a, make C2b FALSE and C2c FALSE:
    #   C2b needs delta<expected AND minDelta<expected to be FALSE.
    #   Keep delta<=0 and minDelta<=0 but set expectedDelta very low so
    #   delta<expected is FALSE.
    def test_suspend_via_c2a_only(self):
        # delta=-3<=0 T, minDelta=-2.5<=0 T -> C2a TRUE
        # expectedDelta=-10: delta(-3) < -10 F -> C2b FALSE
        # bg=65: 65<60 F -> C2c FALSE
        args = make_args(
            bg=65, delta=-3.0, min_delta=-2.5, expected_delta=-10.0
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.0
        assert result.duration == 30
        assert "setting 0 temp" in result.reason

    # --- C2a FALSE (one operand flips) ---
    def test_c2a_false_when_delta_positive(self):
        # delta=1 > 0 -> C2a FALSE (delta<=0 fails)
        # minDelta=-2.5<=0 T but AND fails
        # Make C2b and C2c FALSE too -> no suspend
        args = make_args(
            bg=65, delta=1.0, min_delta=-2.5, expected_delta=-10.0,
            temp_duration=10,
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8  # recovery path

    def test_c2a_false_when_min_delta_positive(self):
        # delta=-3<=0 T, minDelta=1>0 -> C2a FALSE
        # C2b: delta=-3<exp? make exp=-10 -> F
        # C2c: bg=65<60 F
        args = make_args(
            bg=65, delta=-3.0, min_delta=1.0, expected_delta=-10.0,
            temp_duration=10,
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8  # recovery path

    # --- C2b alone flips D2 to TRUE ---
    # Need C2a FALSE, C2c FALSE, C2b TRUE.
    # C2a FALSE: delta<=0 AND minDelta<=0 must be FALSE.
    # C2b TRUE: delta<expected AND minDelta<expected.
    # C2c FALSE: bg>=60.
    # If delta<expected and minDelta<expected, then if expected<=0, both
    # are <= expected <= 0, which would make C2a TRUE too. So we need
    # expected > 0 while still having delta < expected, minDelta < expected,
    # and C2a FALSE.
    # C2a FALSE with delta<expected and minDelta<expected (both potentially
    # positive or negative): we need NOT(delta<=0 AND minDelta<=0), i.e.
    # at least one of delta,minDelta > 0.
    # Let delta=0.5>0, minDelta=0.3>0, expected=1.0.
    #   C2a: 0.5<=0 F -> F
    #   C2b: 0.5<1.0 T AND 0.3<1.0 T -> T
    #   C2c: bg=65<60 F
    #   D2 TRUE
    def test_suspend_via_c2b_only(self):
        args = make_args(
            bg=65, delta=0.5, min_delta=0.3, expected_delta=1.0
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.0
        assert result.duration == 30
        assert "setting 0 temp" in result.reason

    # --- C2b FALSE (each operand flips) ---
    def test_c2b_false_when_delta_not_less_than_expected(self):
        # delta=0.5>0 (C2a F), expected=0.5: delta<expected F -> C2b F
        # minDelta=0.3<0.5 T but AND fails
        # C2c: bg=65<60 F
        args = make_args(
            bg=65, delta=0.5, min_delta=0.3, expected_delta=0.5,
            temp_duration=10,
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8  # recovery path

    def test_c2b_false_when_min_delta_not_less_than_expected(self):
        # delta=0.5<1.0 T, minDelta=1.0<1.0 F -> C2b F
        # C2a: 0.5<=0 F -> F
        # C2c: bg=65<60 F
        args = make_args(
            bg=65, delta=0.5, min_delta=1.0, expected_delta=1.0,
            temp_duration=10,
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8  # recovery path

    # --- C2c alone flips D2 to TRUE ---
    # Need C2a FALSE, C2b FALSE, C2c TRUE (bg<60).
    # bg=55<60 T.
    # C2a F: delta=0.5>0
    # C2b F: delta=0.5<expected? set expected=0.5 -> F
    def test_suspend_via_c2c_only(self):
        args = make_args(
            bg=55, delta=0.5, min_delta=0.3, expected_delta=0.5
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.0
        assert result.duration == 30
        assert "setting 0 temp" in result.reason

    # --- C2c FALSE (boundary) ---
    def test_c2c_false_at_bg_60(self):
        # bg=60: 60<60 F -> C2c F
        # Keep C2a,C2b F as in baseline
        args = make_args(
            bg=60, delta=0.5, min_delta=0.3, expected_delta=0.5,
            temp_duration=10,
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8  # recovery path (not suspended)

    # --- Combined sub-condition coverage: C2a TRUE with other FALSE combos ---
    def test_c2a_true_c2b_false_c2c_false(self):
        # Already covered by test_suspend_via_c2a_only
        pass

    def test_c2a_false_c2b_true_c2c_false(self):
        # Already covered by test_suspend_via_c2b_only
        pass

    def test_c2a_false_c2b_false_c2c_true(self):
        # Already covered by test_suspend_via_c2c_only
        pass

    def test_c2a_true_with_min_delta_zero_boundary(self):
        # minDelta=0<=0 T, delta=-1<=0 T -> C2a T
        args = make_args(
            bg=65, delta=-1.0, min_delta=0.0, expected_delta=-10.0
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.0

    def test_delta_zero_boundary_c2a(self):
        # delta=0<=0 T, minDelta=-1<=0 T -> C2a T
        args = make_args(
            bg=65, delta=0.0, min_delta=-1.0, expected_delta=-10.0
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.0


class TestD3_CurrentTempAppropriate:
    """MC/DC for D3 = (currentTemp.duration > 15) AND
                       (roundBasal(basal) == roundBasal(currentTemp.rate)).

    Only reachable when D1 TRUE and D2 FALSE (recovery path).
    """

    def _recovery_args(self, **overrides):
        base = make_args(
            bg=65,
            delta=2.0,
            min_delta=1.5,
            expected_delta=-1.0,
            basal=0.8,
            temp_rate=0.8,
            temp_duration=20,
        )
        base.update(overrides)
        return base

    # D3 TRUE: both operands TRUE -> no action (rate=None)
    def test_d3_true_keeps_current_temp(self):
        args = self._recovery_args(temp_duration=20, temp_rate=0.8, basal=0.8)
        result = determineLowGlucoseAction(**args)
        assert result.rate is None
        assert result.duration is None
        assert "~ req" in result.reason

    # D3 TRUE via rounded equality (different raw values, same rounded)
    def test_d3_true_via_rounded_equality(self):
        # basal=0.82 -> rounds to 0.80; temp_rate=0.78 -> rounds to 0.80
        args = self._recovery_args(
            temp_duration=20, temp_rate=0.78, basal=0.82
        )
        result = determineLowGlucoseAction(**args)
        assert result.rate is None
        assert "~ req" in result.reason

    # D3 FALSE via duration operand: duration <= 15 (boundary 15)
    def test_d3_false_duration_at_15(self):
        args = self._recovery_args(temp_duration=15, temp_rate=0.8, basal=0.8)
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8
        assert result.duration == 30
        assert "setting current basal" in result.reason

    def test_d3_false_duration_below_15(self):
        args = self._recovery_args(temp_duration=10, temp_rate=0.8, basal=0.8)
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8
        assert result.duration == 30

    # D3 FALSE via rate operand: rates differ after rounding
    def test_d3_false_rate_differs(self):
        args = self._recovery_args(temp_duration=20, temp_rate=0.0, basal=0.8)
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8
        assert result.duration == 30

    def test_d3_false_rate_slightly_differs_after_rounding(self):
        # basal=0.8 -> 0.80; temp_rate=0.9 -> 0.90 (differ)
        args = self._recovery_args(temp_duration=20, temp_rate=0.9, basal=0.8)
        result = determineLowGlucoseAction(**args)
        assert result.rate == 0.8

    # D3 duration boundary exactly 16 (just above 15) with matching rate
    def test_d3_true_duration_16(self):
        args = self._recovery_args(temp_duration=16, temp_rate=0.8, basal=0.8)
        result = determineLowGlucoseAction(**args)
        assert result.rate is None
        assert "~ req" in result.reason

    # D3 FALSE when duration>15 but rate differs (rate operand flips)
    def test_d3_rate_operand_flips_decision(self):
        # duration>15 T, rates equal T -> D3 T
        args_true = self._recovery_args(temp_duration=20, temp_rate=0.8, basal=0.8)
        r_true = determineLowGlucoseAction(**args_true)
        assert r_true.rate is None

        # duration>15 T, rates differ F -> D3 F
        args_false = self._recovery_args(temp_duration=20, temp_rate=0.5, basal=0.8)
        r_false = determineLowGlucoseAction(**args_false)
        assert r_false.rate == 0.8

    # D3 FALSE when rate matches but duration<=15 (duration operand flips)
    def test_d3_duration_operand_flips_decision(self):
        args_true = self._recovery_args(temp_duration=20, temp_rate=0.8, basal=0.8)
        assert determineLowGlucoseAction(**args_true).rate is None

        args_false = self._recovery_args(temp_duration=15, temp_rate=0.8, basal=0.8)
        assert determineLowGlucoseAction(**args_false).rate == 0.8


# ===========================================================================
# Reason-string / integration sanity tests
# ===========================================================================

class TestReasonAndIntegration:
    def test_suspend_reason_includes_bg_and_threshold(self):
        args = make_args(bg=50, threshold=75, delta=-3.0, min_delta=-2.5)
        result = determineLowGlucoseAction(**args)
        assert "BG 50 < 75" in result.reason
        assert "setting 0 temp" in result.reason

    def test_recovery_reason_includes_delta_values(self):
        args = make_args(
            bg=65, delta=2.0, min_delta=1.5, expected_delta=-1.0,
            temp_duration=10,
        )
        result = determineLowGlucoseAction(**args)
        assert "delta 2.0" in result.reason
        assert "min delta 1.50" in result.reason

    def test_no_action_reason_empty(self):
        args = make_args(bg=100, threshold=75)
        result = determineLowGlucoseAction(**args)
        assert result.reason == ""
        assert result.rate is None
        assert result.duration is None

    def test_main_scenarios_run(self, capsys):
        """Smoke-test the __main__ block scenarios."""
        import runpy
        runpy.run_module("low_glucose_action", run_name="__main__")
        out = capsys.readouterr().out
        assert "Scenario 1" in out
        assert "Scenario 2" in out
        assert "Scenario 3" in out