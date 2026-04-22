"""Minimal runnable co-simulation smoke experiment."""

from __future__ import annotations

from tvbx_multiscale.adapters.nest import NESTMockAdapter
from tvbx_multiscale.adapters.tvboptim import TVBOptimMockAdapter
from tvbx_multiscale.core import CoSimulationOrchestrator
from tvbx_multiscale.transforms import RateSNNToTVB, RateTVBToSNN


def run_smoke(steps: int = 5, n_nodes: int = 10, tvb_dt_ms: float = 1.0):
    tvb = TVBOptimMockAdapter(n_nodes=n_nodes, dt_ms=tvb_dt_ms)
    snn = NESTMockAdapter(n_nodes=n_nodes, dt_ms=tvb_dt_ms)

    tvb_to_snn = RateTVBToSNN(interface_weight=1.0, min_rate_hz=0.0)
    snn_to_tvb = RateSNNToTVB(spikes_to_rate_scale=1000.0 / tvb_dt_ms)

    orchestrator = CoSimulationOrchestrator(
        tvb_adapter=tvb,
        snn_adapter=snn,
        tvb_to_snn=tvb_to_snn,
        snn_to_tvb=snn_to_tvb,
    )
    orchestrator.initialize()
    return orchestrator.run(steps=steps)


if __name__ == "__main__":
    traces = run_smoke(steps=6, n_nodes=10, tvb_dt_ms=1.0)
    last = traces[-1]
    print(f"steps={len(traces)}")
    print(f"last_time_ms={last.tvb_output.time_ms:.3f}")
    print(f"last_tvb_mean_rate_hz={last.tvb_output.rate_by_node.mean():.3f}")
    print(f"last_snn_mean_ratio={last.snn_output.population_mean_spikes_number_by_node.mean():.5f}")

