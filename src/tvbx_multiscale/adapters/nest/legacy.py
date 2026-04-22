"""Adapter that bridges the new orchestrator with legacy tvb-multiscale NEST interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from tvbx_multiscale.contracts.messages import SNNInput, SNNStepOutput


def _iter_values(maybe_series: Any) -> list[Any]:
    if maybe_series is None:
        return []
    if hasattr(maybe_series, "values"):
        try:
            return list(maybe_series.values)
        except Exception:
            pass
    if isinstance(maybe_series, (list, tuple)):
        return list(maybe_series)
    return [maybe_series]


@dataclass
class NESTLegacyRuntimeAdapter:
    """Runtime adapter over legacy `tvb_multiscale` interface objects.

    The adapter expects:
    - `input_interfaces` objects exposing `nodes_ids` and `set(values)`.
    - `output_interfaces` objects exposing `nodes_ids` and
      `population_mean_spikes_number`.
    """

    nest_instance: Any
    input_interfaces: Iterable[Any]
    output_interfaces: Iterable[Any]
    n_nodes: int
    dt_ms: float
    aggregation: str = "mean"

    @classmethod
    def from_legacy_tvb_nest_interface(
        cls, tvb_nest_interface: Any, dt_ms: float, n_nodes: int | None = None
    ) -> "NESTLegacyRuntimeAdapter":
        input_ifaces = _iter_values(getattr(tvb_nest_interface, "tvb_to_spikeNet_interfaces", None))
        output_ifaces = _iter_values(getattr(tvb_nest_interface, "spikeNet_to_tvb_interfaces", None))
        if n_nodes is None:
            max_node = -1
            for iface in list(input_ifaces) + list(output_ifaces):
                ids = np.asarray(getattr(iface, "nodes_ids", []), dtype=int)
                if ids.size:
                    max_node = max(max_node, int(ids.max()))
            n_nodes = max_node + 1 if max_node >= 0 else 0

        return cls(
            nest_instance=tvb_nest_interface.nest_instance,
            input_interfaces=input_ifaces,
            output_interfaces=output_ifaces,
            n_nodes=int(n_nodes),
            dt_ms=float(dt_ms),
        )

    def initialize(self) -> None:
        # Legacy flow usually prepares NEST externally; keep this adapter non-invasive.
        return None

    def _advance_nest(self) -> None:
        if hasattr(self.nest_instance, "Run"):
            self.nest_instance.Run(self.dt_ms)
            return
        if hasattr(self.nest_instance, "Simulate"):
            self.nest_instance.Simulate(self.dt_ms)
            return
        raise RuntimeError("NEST instance does not expose Run(dt) or Simulate(dt).")

    def step(self, input_payload: SNNInput) -> SNNStepOutput:
        if self.n_nodes <= 0:
            raise RuntimeError("NESTLegacyRuntimeAdapter requires n_nodes > 0.")

        # 1) Push TVB-driven rates/currents to NEST input proxies.
        for interface in self.input_interfaces:
            node_ids = np.asarray(interface.nodes_ids, dtype=int)
            values = np.asarray(input_payload.drive_by_node[node_ids], dtype=np.float64)
            interface.set(values)

        # 2) Advance NEST by one co-simulation step.
        self._advance_nest()

        # 3) Gather spike ratios from output interfaces and aggregate per node.
        accum = np.zeros((self.n_nodes,), dtype=np.float64)
        counts = np.zeros((self.n_nodes,), dtype=np.float64)

        for interface in self.output_interfaces:
            node_ids = np.asarray(interface.nodes_ids, dtype=int)
            values = np.asarray(interface.population_mean_spikes_number, dtype=np.float64).reshape(-1)
            if node_ids.size != values.size:
                raise RuntimeError(
                    f"Interface node/value mismatch: {node_ids.size} nodes vs {values.size} values."
                )
            for node_id, value in zip(node_ids, values):
                accum[int(node_id)] += float(value)
                counts[int(node_id)] += 1.0

        nonzero = counts > 0.0
        if self.aggregation == "mean":
            accum[nonzero] = accum[nonzero] / counts[nonzero]
        elif self.aggregation == "sum":
            pass
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation}")

        return SNNStepOutput(
            time_ms=float(input_payload.time_ms),
            population_mean_spikes_number_by_node=accum,
        )

