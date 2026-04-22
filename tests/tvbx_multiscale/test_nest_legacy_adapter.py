"""Unit tests for NESTLegacyRuntimeAdapter without real NEST dependency."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tvbx_multiscale.adapters.nest import NESTLegacyRuntimeAdapter  # noqa: E402
from tvbx_multiscale.contracts.messages import SNNInput  # noqa: E402


class _FakeNest:
    def __init__(self):
        self.run_calls: list[float] = []

    def Run(self, dt_ms: float) -> None:
        self.run_calls.append(float(dt_ms))


class _FakeInputInterface:
    def __init__(self, nodes_ids):
        self.nodes_ids = np.asarray(nodes_ids, dtype=int)
        self.last_values = None

    def set(self, values):
        self.last_values = np.asarray(values, dtype=np.float64)


class _FakeOutputInterface:
    def __init__(self, nodes_ids, values):
        self.nodes_ids = np.asarray(nodes_ids, dtype=int)
        self._values = np.asarray(values, dtype=np.float64)

    @property
    def population_mean_spikes_number(self):
        return self._values


class NestLegacyAdapterTest(unittest.TestCase):
    def test_step_maps_inputs_and_aggregates_outputs(self):
        nest = _FakeNest()
        in_iface = _FakeInputInterface(nodes_ids=[0, 2])
        out_iface_1 = _FakeOutputInterface(nodes_ids=[0, 1], values=[0.10, 0.20])
        out_iface_2 = _FakeOutputInterface(nodes_ids=[1], values=[0.40])

        adapter = NESTLegacyRuntimeAdapter(
            nest_instance=nest,
            input_interfaces=[in_iface],
            output_interfaces=[out_iface_1, out_iface_2],
            n_nodes=3,
            dt_ms=1.0,
            aggregation="mean",
        )
        adapter.initialize()

        payload = SNNInput(time_ms=5.0, drive_by_node=np.array([10.0, 20.0, 30.0], dtype=np.float64))
        out = adapter.step(payload)

        np.testing.assert_allclose(in_iface.last_values, np.array([10.0, 30.0]))
        self.assertEqual(nest.run_calls, [1.0])
        # Node0: 0.10, Node1: mean(0.20,0.40)=0.30, Node2: no output interface => 0.0
        np.testing.assert_allclose(
            out.population_mean_spikes_number_by_node,
            np.array([0.10, 0.30, 0.0], dtype=np.float64),
        )
        self.assertAlmostEqual(out.time_ms, 5.0)


if __name__ == "__main__":
    unittest.main()

