"""Rate-mode TVB<->SNN transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tvbx_multiscale.contracts.messages import SNNInput, SNNStepOutput, TVBFeedback, TVBStepOutput


@dataclass(frozen=True)
class RateTVBToSNN:
    """Convert TVB rate output to SNN drive."""

    interface_weight: float = 1.0
    min_rate_hz: float = 0.0

    def __call__(self, tvb_output: TVBStepOutput) -> SNNInput:
        drive = np.maximum(self.min_rate_hz, tvb_output.rate_by_node * self.interface_weight)
        return SNNInput(time_ms=tvb_output.time_ms, drive_by_node=drive.astype(np.float64, copy=False))


@dataclass(frozen=True)
class RateSNNToTVB:
    """Convert SNN spike ratios per TVB-step to TVB rate feedback.

    In legacy logic this scale is typically `1000 / tvb_dt`.
    """

    spikes_to_rate_scale: float
    target_state: str = "Rin"

    def __call__(self, snn_output: SNNStepOutput, *, tvb_output: TVBStepOutput) -> TVBFeedback:
        values = snn_output.population_mean_spikes_number_by_node * self.spikes_to_rate_scale
        return TVBFeedback(
            time_ms=tvb_output.time_ms,
            value_by_node=values.astype(np.float64, copy=False),
            target_state=self.target_state,
        )

