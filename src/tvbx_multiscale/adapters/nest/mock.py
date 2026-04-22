"""Temporary SNN adapter used for scaffolding and tests.

Replace with a real adapter around NEST run/record interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tvbx_multiscale.contracts.messages import SNNInput, SNNStepOutput


@dataclass
class NESTMockAdapter:
    """Simple deterministic mapping from rate drive to spike ratios."""

    n_nodes: int
    dt_ms: float = 1.0
    saturation_hz: float = 120.0
    _time_ms: float = field(default=0.0, init=False)
    _activity_by_node: np.ndarray = field(default_factory=lambda: np.empty((0,)), init=False)

    def initialize(self) -> None:
        self._time_ms = 0.0
        self._activity_by_node = np.zeros((self.n_nodes,), dtype=np.float64)

    def step(self, input_payload: SNNInput) -> SNNStepOutput:
        # A tiny low-pass filter to emulate neuronal inertia.
        instant_ratio = np.clip(input_payload.drive_by_node / self.saturation_hz, 0.0, 1.0)
        self._activity_by_node = 0.7 * self._activity_by_node + 0.3 * instant_ratio

        self._time_ms = input_payload.time_ms
        return SNNStepOutput(
            time_ms=self._time_ms,
            population_mean_spikes_number_by_node=self._activity_by_node.copy(),
        )

