"""Typed message objects exchanged between TVB and SNN layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TVBStepOutput:
    """TVB-side output at one co-simulation step."""

    time_ms: float
    rate_by_node: FloatArray


@dataclass(frozen=True)
class TVBFeedback:
    """Feedback payload from SNN that will be injected to TVB next step."""

    time_ms: float
    value_by_node: FloatArray
    target_state: str = "Rin"


@dataclass(frozen=True)
class SNNInput:
    """Input payload sent from TVB to SNN."""

    time_ms: float
    drive_by_node: FloatArray


@dataclass(frozen=True)
class SNNStepOutput:
    """SNN-side measured activity at one co-simulation step."""

    time_ms: float
    population_mean_spikes_number_by_node: FloatArray


@dataclass(frozen=True)
class StepTrace:
    """Full trace for one co-simulation step."""

    tvb_output: TVBStepOutput
    snn_input: SNNInput
    snn_output: SNNStepOutput
    tvb_feedback: TVBFeedback

