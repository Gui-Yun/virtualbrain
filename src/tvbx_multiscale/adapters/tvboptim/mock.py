"""Temporary TVB adapter used for scaffolding and tests.

Replace with a real adapter around tvboptim.prepare/solve state evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tvbx_multiscale.contracts.messages import TVBFeedback, TVBStepOutput


@dataclass
class TVBOptimMockAdapter:
    """Small deterministic stand-in for TVB side evolution."""

    n_nodes: int
    dt_ms: float = 1.0
    baseline_rate_hz: float = 5.0
    feedback_blend: float = 0.35
    _time_ms: float = field(default=0.0, init=False)
    _rate_by_node: np.ndarray = field(default_factory=lambda: np.empty((0,)), init=False)

    def initialize(self) -> None:
        self._time_ms = 0.0
        self._rate_by_node = np.full((self.n_nodes,), self.baseline_rate_hz, dtype=np.float64)

    def step(self, feedback: TVBFeedback | None = None) -> TVBStepOutput:
        # Gentle intrinsic drift to avoid a perfectly static series.
        self._rate_by_node = 0.98 * self._rate_by_node + 0.02 * self.baseline_rate_hz

        if feedback is not None:
            self._rate_by_node = (
                (1.0 - self.feedback_blend) * self._rate_by_node
                + self.feedback_blend * feedback.value_by_node
            )

        self._time_ms += self.dt_ms
        return TVBStepOutput(time_ms=self._time_ms, rate_by_node=self._rate_by_node.copy())

