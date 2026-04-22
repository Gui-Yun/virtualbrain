"""Co-simulation loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from tvbx_multiscale.contracts.interfaces import (
    SNNAdapter,
    SNNToTVBTransform,
    TVBAdapter,
    TVBToSNNTransform,
)
from tvbx_multiscale.contracts.messages import StepTrace, TVBFeedback


@dataclass
class CoSimulationOrchestrator:
    """Simple, explicit, testable TVB<->SNN co-simulation loop."""

    tvb_adapter: TVBAdapter
    snn_adapter: SNNAdapter
    tvb_to_snn: TVBToSNNTransform
    snn_to_tvb: SNNToTVBTransform
    traces: list[StepTrace] = field(default_factory=list)

    def initialize(self) -> None:
        """Initialize both simulators."""
        self.tvb_adapter.initialize()
        self.snn_adapter.initialize()
        self.traces.clear()

    def step(self, feedback: TVBFeedback | None = None) -> StepTrace:
        """Run one co-simulation step.

        The ordering is:
        1. TVB step with previous feedback
        2. Transform TVB output to SNN input
        3. SNN step
        4. Transform SNN output to next TVB feedback
        """
        tvb_output = self.tvb_adapter.step(feedback=feedback)
        snn_input = self.tvb_to_snn(tvb_output)
        snn_output = self.snn_adapter.step(snn_input)
        next_feedback = self.snn_to_tvb(snn_output, tvb_output=tvb_output)
        trace = StepTrace(
            tvb_output=tvb_output,
            snn_input=snn_input,
            snn_output=snn_output,
            tvb_feedback=next_feedback,
        )
        self.traces.append(trace)
        return trace

    def run(self, steps: int, initial_feedback: TVBFeedback | None = None) -> list[StepTrace]:
        """Run multiple co-simulation steps."""
        feedback = initial_feedback
        for _ in range(steps):
            trace = self.step(feedback=feedback)
            feedback = trace.tvb_feedback
        return list(self.traces)

