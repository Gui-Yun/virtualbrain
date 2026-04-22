"""Adapter and transform protocols for the multiscale orchestrator."""

from __future__ import annotations

from typing import Protocol

from .messages import SNNInput, SNNStepOutput, TVBFeedback, TVBStepOutput


class TVBAdapter(Protocol):
    """Protocol for TVB-side adapter."""

    @property
    def dt_ms(self) -> float: ...

    def initialize(self) -> None: ...

    def step(self, feedback: TVBFeedback | None = None) -> TVBStepOutput: ...


class SNNAdapter(Protocol):
    """Protocol for SNN-side adapter."""

    @property
    def dt_ms(self) -> float: ...

    def initialize(self) -> None: ...

    def step(self, input_payload: SNNInput) -> SNNStepOutput: ...


class TVBToSNNTransform(Protocol):
    """Transform TVB output into SNN input."""

    def __call__(self, tvb_output: TVBStepOutput) -> SNNInput: ...


class SNNToTVBTransform(Protocol):
    """Transform SNN output into TVB feedback."""

    def __call__(
        self, snn_output: SNNStepOutput, *, tvb_output: TVBStepOutput
    ) -> TVBFeedback: ...

