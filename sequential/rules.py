from __future__ import annotations

from abc import ABC, abstractmethod
from typing import FrozenSet, Sequence, Optional

from sequential.types import EngineCapability, LookResult, StopReason


def _assert_non_empty_look_history(looks: Sequence[LookResult]) -> None:
    if not looks:
        raise ValueError("looks must contain at least one LookResult")


class StoppingRule(ABC):
    reason: StopReason
    requires: FrozenSet[EngineCapability]

    def check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        _assert_non_empty_look_history(looks)
        return self._check(looks)


    @abstractmethod
    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        """Return a stop reason or None if the rule does not trigger."""

class EfficacyRule(StoppingRule):
    """
    Stop when the confidence interval excludes 0.

    This assumes the estimate is an effect or difference where 0 is the reference value
    (for example, no difference between two models).
    """

    reason = StopReason.EFFICACY
    requires = frozenset({"ci"})

    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        look = looks[-1]
        if look.ci_upper < 0 or look.ci_lower > 0:
            return self.reason
        return None

class EquivalenceRule(StoppingRule):
    """
    Stop when the confidence interval is fully inside the equivalence band around 0.

    This assumes the estimate is an effect or difference where 0 is the reference value
    (for example, no meaningful difference between two models).
    """

    requires = frozenset({"ci"})

    def __init__(self, margin: float):
        if margin <= 0.0:
            raise ValueError("margin must be positive")
        self.margin = margin
        self.reason = StopReason.EQUIVALENCE

    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        look = looks[-1]
        if look.ci_lower >= -self.margin and look.ci_upper <= self.margin:
            return self.reason
        return None


class PrecisionRule(StoppingRule):
    """Stop when the current confidence interval halfwidth is at most `max_halfwidth`."""

    requires = frozenset({"ci"})

    def __init__(self, max_halfwidth: float):
        if max_halfwidth <= 0.0:
            raise ValueError("max_halfwidth must be positive")
        self.max_halfwidth = max_halfwidth
        self.reason = StopReason.PRECISION

    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        look = looks[-1]
        if look.halfwidth <= self.max_halfwidth:
            return self.reason
        return None


class ThresholdRule(StoppingRule):
    """
    Stop when the full confidence interval clears a threshold on the estimate scale.

    `direction="greater"` stops when the entire CI is above `threshold`.
    `direction="less"` stops when the entire CI is below `threshold`.
    """

    requires = frozenset({"ci"})

    def __init__(self, threshold: float, direction: str = "greater"):
        if direction not in {"greater", "less"}:
            raise ValueError("direction must be 'greater' or 'less'")
        self.threshold = threshold
        self.direction = direction
        self.reason = StopReason.THRESHOLD

    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        look = looks[-1]
        if self.direction == "greater" and look.ci_lower > self.threshold:
            return self.reason
        if self.direction == "less" and look.ci_upper < self.threshold:
            return self.reason
        return None


class FutilityRule(StoppingRule):
    """Stop only when the native engine marks futility as crossed."""

    requires = frozenset({"futility"})

    reason = StopReason.FUTILITY

    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        look = looks[-1]
        if look.futility_crossed is True:
            return self.reason
        return None


class DiminishingReturnsRule(StoppingRule):
    """Stop when CI halfwidth has not decreased enough over the configured window."""

    requires = frozenset({"ci"})

    def __init__(self, min_delta: float, window: int = 2):
        if window < 2:
            raise ValueError("window must be at least 2")
        if min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        self.min_delta = min_delta
        self.window = window
        self.reason = StopReason.DIMINISHING_RETURNS

    def _check(self, looks: Sequence[LookResult]) -> Optional[StopReason]:
        if len(looks) < self.window:
            return None
        recent_drop = looks[-self.window].halfwidth - looks[-1].halfwidth
        if recent_drop <= self.min_delta:
            return self.reason
        return None
