"""Integration-smoke tests for TVBOptimRuntimeAdapter.

This test is skipped automatically when jax/tvboptim are not installed in the
active Python environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tvbx_multiscale.adapters.tvboptim import TVBOptimRuntimeAdapter  # noqa: E402
from tvbx_multiscale.contracts.messages import TVBFeedback  # noqa: E402


HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_TVBOPTIM = importlib.util.find_spec("tvboptim") is not None


@unittest.skipUnless(HAS_JAX and HAS_TVBOPTIM, "jax/tvboptim not available in this environment")
class TVBOptimRuntimeAdapterTest(unittest.TestCase):
    def test_step_and_feedback_loop(self):
        import jax
        import jax.numpy as jnp
        from tvboptim.experimental.network_dynamics import Network
        from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
        from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
        from tvboptim.experimental.network_dynamics.graph import DenseGraph
        from tvboptim.experimental.network_dynamics.solvers import Heun

        jax.config.update("jax_enable_x64", True)

        n_nodes = 4
        weights = jnp.ones((n_nodes, n_nodes), dtype=jnp.float64) * 0.01
        weights = weights.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)

        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"instant": LinearCoupling(incoming_states="S", G=0.1)},
            graph=DenseGraph(weights),
        )
        adapter = TVBOptimRuntimeAdapter(network=network, dt_ms=0.1, solver=Heun(), readout_state="S")
        adapter.initialize()

        out1 = adapter.step()
        self.assertEqual(out1.rate_by_node.shape, (n_nodes,))
        self.assertTrue(np.all(np.isfinite(out1.rate_by_node)))

        feedback = TVBFeedback(time_ms=out1.time_ms, value_by_node=np.ones((n_nodes,), dtype=np.float64) * 0.2)
        out2 = adapter.step(feedback=feedback)
        self.assertGreater(out2.time_ms, out1.time_ms)
        self.assertEqual(out2.rate_by_node.shape, (n_nodes,))


if __name__ == "__main__":
    unittest.main()

