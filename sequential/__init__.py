"""Sequential testing package split into focused modules."""

from sequential.aggregators import MeanOfMeansAggregator, PooledAggregator, ScoreAggregator
from sequential.engines import (
    BootstrapReferenceEngine,
    SequentialEngine,
    SimpleMeanOfMeansGsDesignEngine,
)
from sequential.rules import (
    DiminishingReturnsRule,
    EfficacyRule,
    EquivalenceRule,
    FutilityRule,
    PrecisionRule,
    StoppingRule,
    ThresholdRule,
)
from sequential.runner import run_sequential_test
from sequential.samplers import NeymanSampler, RoundRobinSampler, Sampler, SamplingPlan
from sequential.types import (
    FutilityConfig,
    LookResult,
    ScoreLike,
    SequentialOutcome,
    SequentialTestType,
    StopReason,
)

__all__ = [
    "ScoreLike",
    "LookResult",
    "SamplingPlan",
    "SequentialOutcome",
    "StopReason",
    "SequentialTestType",
    "FutilityConfig",
    "StoppingRule",
    "EfficacyRule",
    "EquivalenceRule",
    "PrecisionRule",
    "ThresholdRule",
    "FutilityRule",
    "DiminishingReturnsRule",
    "ScoreAggregator",
    "MeanOfMeansAggregator",
    "PooledAggregator",
    "Sampler",
    "RoundRobinSampler",
    "NeymanSampler",
    "SequentialEngine",
    "RgsDesignSequentialEngine",
    "SimpleMeanOfMeansGsDesignEngine",
    "BootstrapReferenceEngine",
    "run_sequential_test",
]
