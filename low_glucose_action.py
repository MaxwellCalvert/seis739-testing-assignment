"""Python equivalent of LowGlucoseAction_Simplified.java.

Student exercise: do not change this file. Write tests in a separate file.
"""

from dataclasses import dataclass
import math


@dataclass
class GlucoseStatus:
    delta: float  # mg/dL change per 5 minutes


@dataclass
class CurrentTemp:
    rate: float  # U/hr
    duration: int  # minutes remaining


@dataclass
class Profile:
    maxBasal: float


@dataclass
class RequestedTemp:
    temp: str = "absolute"
    rate: float | None = None
    duration: int | None = None
    reason: str = ""

    def __str__(self):
        if self.rate is None:
            return f"RequestedTemp{{no action, reason='{self.reason}'}}"
        return (f"RequestedTemp{{rate={self.rate:.2f} U/hr, "
                f"duration={self.duration} min, reason='{self.reason}'}}")


def roundBasal(rate, profile):
    """Round to 0.05 U/hr, following Java Math.round for finite inputs."""
    if rate < 0:
        return 0.0
    if rate > profile.maxBasal:
        return profile.maxBasal
    return math.floor(rate * 20.0 + 0.5) / 20.0


def determineLowGlucoseAction(
    bg, glucoseStatus, minDelta, expectedDelta, threshold, basal, currentTemp, profile
):
    rT = RequestedTemp()

    if bg < threshold:
        rT.reason = f"BG {bg:.0f} < {threshold:.0f}"

        # Critical decision: should we suspend insulin delivery?
        if ((glucoseStatus.delta <= 0 and minDelta <= 0) or
                (glucoseStatus.delta < expectedDelta and minDelta < expectedDelta) or
                bg < 60):
            rT.rate = 0.0
            rT.duration = 30
            rT.reason += "; setting 0 temp"
            return rT

        # BG is recovering; record trend values without another decision.
        rT.reason += (f", delta {glucoseStatus.delta:.1f}, "
                      f"min delta {minDelta:.2f}")

        # Check if current temp is already appropriate.
        if (currentTemp.duration > 15 and
                roundBasal(basal, profile) == roundBasal(currentTemp.rate, profile)):
            rT.reason += f", temp {currentTemp.rate:.2f} ~ req {basal:.2f}"
            return rT
        else:
            rT.reason += f"; setting current basal of {basal:.2f} as temp"
            rT.rate = basal
            rT.duration = 30
            return rT

    # BG is not below threshold - no action needed.
    return rT


if __name__ == "__main__":
    profile = Profile(4.0)
    scenarios = [
        (65, GlucoseStatus(-3.0), -2.5, -1.0, 75, 0.8, CurrentTemp(0.8, 10), profile),
        (72, GlucoseStatus(2.0), 1.5, -0.5, 75, 0.8, CurrentTemp(0.0, 20), profile),
        (110, GlucoseStatus(0.5), 0.3, 0.2, 75, 0.8, CurrentTemp(0.8, 15), profile),
    ]
    for i, args in enumerate(scenarios, 1):
        print(f"=== Scenario {i} ===")
        print(determineLowGlucoseAction(*args))
