from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping, Optional, Literal, Protocol, Tuple, TypeAlias

DatasetId: TypeAlias = str


class ScoreLike(Protocol):
    dataset_id: DatasetId
    instance_id: str
    score: float

class SequentialTestType(StrEnum):
    ONE_SIDED = "one_sided"
    TWO_SIDED = "two_sided"


DatasetStreams: TypeAlias = Mapping[DatasetId, Iterable[ScoreLike]]
DatasetSizes: TypeAlias = Mapping[DatasetId, int]

AggregationMode: TypeAlias = Literal["mean_of_means", "pooled_mean"]
SpendingFunction: TypeAlias = Literal["obrien_fleming", "pocock", "wang_tsiatis"]
EngineCapability: TypeAlias = Literal["ci", "efficacy", "futility"]

@dataclass(frozen=True)
class FutilityConfig:
    """
    Native binding futility configuration for R gsDesign.

    Question answered:
        "If the true effect is at least `mdes`, do we still look on track to detect it with
        power `1 - beta`, or is the interim evidence weak enough to stop?"

    Standard group-sequential procedure:
        The futility boundary is fixed before evaluating stopping decisions. The engine uses the
        first planned look as a pilot sample, estimates the full-benchmark standard error through
        the configured aggregator, builds the native gsDesign binding futility design, and then
        freezes that design for all looks. Re-estimating the standard error and redesigning
        boundaries at every look is intentionally not supported here.

    Assumptions:
        Scores are on a scale where `mdes` is meaningful, and the aggregator variance estimate from
        the first planned look is a reasonable planning estimate for the full benchmark.

    Args:
        mdes: Minimum detectable/meaningful effect size in raw score units.
        lower_spending_function: How beta is spent across futility looks. "pocock" spends more
            evenly and can stop earlier; "obrien_fleming" is more conservative early.
    """

    mdes: float
    lower_spending_function: SpendingFunction = "pocock"

    def __post_init__(self) -> None:
        if self.mdes <= 0.0:
            raise ValueError("mdes must be positive.")


@dataclass(frozen=True)
class LookResult:
    """
    Snapshot of one interim look.

    Attributes:
        look_index: Zero-based index of the interim look.
        n_seen: Cumulative number of samples observed at this look.
        total_examples: Total number of examples in the full benchmark.
        estimate: Point estimate of the target metric at this look (based on the chosen aggregator).
        ci_lower: Lower bound of the sequential confidence interval.
        ci_upper: Upper bound of the sequential confidence interval.
        futility_crossed: `None` when native futility was not configured/checked, otherwise whether
            the native binding gsDesign futility boundary was crossed at this look.
        z_stat: Current standardized test statistic, useful for debugging boundary crossings.
        upper_boundary: Upper efficacy boundary used to construct the sequential CI
            and determine efficacy crossing.
        lower_boundary: Lower native futility boundary; futility is crossed
            when z_stat <= lower_boundary.
        samples: The slice of samples contributing to this look (useful for bookkeeping/debugging).
        full_benchmark_mde_estimate: Estimated raw effect size that the full benchmark could detect
            with power `1 - beta` under the current variance estimate. This is explanatory; it does
            not drive stopping.
    """

    look_index: int
    n_seen: int
    total_examples: int
    estimate: float
    ci_lower: float
    ci_upper: float

    futility_crossed: Optional[bool] = None

    z_stat: Optional[float] = None
    upper_boundary: Optional[float] = None
    lower_boundary: Optional[float] = None

    samples: Optional[Tuple[ScoreLike, ...]] = None
    full_benchmark_mde_estimate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.look_index < 0:
            raise ValueError("look_index must be non-negative")
        if self.n_seen < 0:
            raise ValueError("n_seen must be non-negative")
        if self.total_examples < 0:
            raise ValueError("total_examples must be non-negative")
        if self.n_seen > self.total_examples:
            raise ValueError("n_seen must be less than or equal to total_examples")

        for field_name in ("estimate", "ci_lower", "ci_upper"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")

        if self.ci_lower > self.ci_upper:
            raise ValueError("ci_lower must be less than or equal to ci_upper")

        for field_name in ("z_stat", "upper_boundary", "lower_boundary", "full_benchmark_mde_estimate"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite when provided")

        if self.samples is not None:
            samples = tuple(self.samples)
            if len(samples) != self.n_seen:
                raise ValueError("samples length must match n_seen")
            object.__setattr__(self, "samples", samples)

    @property
    def halfwidth(self) -> float:
        return (self.ci_upper - self.ci_lower) / 2.0


class StopReason(StrEnum):
    EFFICACY = "efficacy"
    EQUIVALENCE = "equivalence"
    PRECISION = "precision"
    THRESHOLD = "threshold"
    FUTILITY = "futility"
    DIMINISHING_RETURNS = "diminishing_returns"
    MAX_SAMPLES = "max_samples"


@dataclass(frozen=True)
class SequentialOutcome:
    """
    Result of running a sequential test.

    Attributes:
        stop_reason: Semantic reason the test stopped (enum). Uses `MAX_SAMPLES` when no
            stopping rule fired before the planned sample budget was exhausted.
        stop_look: Index of the look at which we stopped.
        samples_used: Total number of samples consumed when stopping.
        total_available: Total numbers of available samples.
        total_planned_looks: Total number of available looks.
        looks: History of all LookResults up to (and including) the stop.
    """

    stop_reason: StopReason
    stop_look: int
    samples_used: int
    total_available: int
    total_planned_looks: int
    looks: Tuple[LookResult, ...]

    def __post_init__(self) -> None:
        if self.stop_look < 0:
            raise ValueError("stop_look must be non-negative")
        if self.samples_used < 0:
            raise ValueError("samples_used must be non-negative")
        if self.total_available < 0:
            raise ValueError("total_available must be non-negative")
        if self.samples_used > self.total_available:
            raise ValueError("samples_used must be less than or equal to total_available")
        if self.total_planned_looks < 0:
            raise ValueError("total_planned_looks must be non-negative")
        looks = tuple(self.looks)
        if not looks:
            raise ValueError("looks must contain at least one LookResult")
        if self.stop_look >= len(looks):
            raise ValueError("stop_look must refer to an existing look")
        object.__setattr__(self, "looks", looks)
