"""Smoke tests for the new tvbx_multiscale package scaffold."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tvbx_multiscale.experiments.smoke import run_smoke  # noqa: E402


class SmokePipelineTest(unittest.TestCase):
    def test_run_smoke_returns_expected_trace(self):
        traces = run_smoke(steps=4, n_nodes=10, tvb_dt_ms=1.0)

        self.assertEqual(len(traces), 4)
        self.assertAlmostEqual(traces[-1].tvb_output.time_ms, 4.0)

        tvb_rate = traces[-1].tvb_output.rate_by_node
        snn_ratio = traces[-1].snn_output.population_mean_spikes_number_by_node

        self.assertEqual(tvb_rate.shape, (10,))
        self.assertEqual(snn_ratio.shape, (10,))
        self.assertTrue(np.all(np.isfinite(tvb_rate)))
        self.assertTrue(np.all(np.isfinite(snn_ratio)))
        self.assertTrue(np.all(snn_ratio >= 0.0))
        self.assertTrue(np.all(snn_ratio <= 1.0))


if __name__ == "__main__":
    unittest.main()

