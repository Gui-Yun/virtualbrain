"""Runtime adapter that advances a tvboptim Network in co-simulation steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tvbx_multiscale.contracts.messages import TVBFeedback, TVBStepOutput


@dataclass
class TVBOptimRuntimeAdapter:
    """Real TVB-side adapter backed by tvboptim's Network + NativeSolver pipeline.

    This adapter intentionally favors correctness and clean interfaces over runtime
    efficiency in this first version:
    - it runs one small integration window per `step()`,
    - updates an explicit rolling history,
    - rebuilds prepare/solve each step so stateful couplings stay consistent.
    """

    network: Any
    dt_ms: float
    solver: Any | None = None
    readout_state: str = "S"
    feedback_blend: float = 0.35
    readout_scale: float = 1.0
    readout_offset: float = 0.0
    min_rate_hz: float = 0.0
    max_history_steps: int = 4096
    readout_transform: Callable[[np.ndarray], np.ndarray] | None = None

    _time_ms: float = field(default=0.0, init=False)
    _readout_state_index: int = field(default=0, init=False)
    _n_nodes: int = field(default=0, init=False)
    _n_states: int = field(default=0, init=False)
    _history_solution: Any | None = field(default=None, init=False)
    _current_state: Any | None = field(default=None, init=False)
    _tvb_types: Any | None = field(default=None, init=False)
    _tvb_prepare: Any | None = field(default=None, init=False)

    def _ensure_dependencies(self) -> None:
        try:
            import jax.numpy as jnp
            from tvboptim.experimental.network_dynamics.result import NativeSolution
            from tvboptim.experimental.network_dynamics.solve import prepare
            from tvboptim.experimental.network_dynamics.solvers import Heun
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "TVBOptimRuntimeAdapter requires tvboptim and jax. "
                "Install dependencies in the active environment."
            ) from exc

        self._tvb_types = {"NativeSolution": NativeSolution, "jnp": jnp}
        self._tvb_prepare = prepare
        if self.solver is None:
            self.solver = Heun()

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def initialize(self) -> None:
        self._ensure_dependencies()
        jnp = self._tvb_types["jnp"]
        NativeSolution = self._tvb_types["NativeSolution"]

        self._n_nodes = int(self.network.graph.n_nodes)
        self._n_states = int(len(self.network.dynamics.STATE_NAMES))
        state_names = list(self.network.dynamics.STATE_NAMES)
        self._readout_state_index = (
            state_names.index(self.readout_state) if self.readout_state in state_names else 0
        )

        # Ensure all state variables are available in solver output.
        self.network.dynamics.VARIABLES_OF_INTEREST = tuple(range(self._n_states))

        self._time_ms = 0.0
        self._current_state = self.network.initial_state

        # Seed history with the initial state so delayed couplings can query history.
        ts = jnp.asarray([self._time_ms], dtype=jnp.float64)
        ys = jnp.asarray(self._current_state)[None, ...]
        self._history_solution = NativeSolution(ts=ts, ys=ys, dt=self.dt_ms)
        self.network.update_history(self._history_solution)

    def _trim_history(self, ts: Any, ys: Any) -> tuple[Any, Any]:
        if self.max_history_steps <= 0:
            return ts, ys
        if ts.shape[0] <= self.max_history_steps:
            return ts, ys
        return ts[-self.max_history_steps :], ys[-self.max_history_steps :]

    def _apply_feedback(self, feedback: TVBFeedback | None) -> None:
        if feedback is None:
            return

        jnp = self._tvb_types["jnp"]
        values = jnp.asarray(feedback.value_by_node)
        current = self._current_state[self._readout_state_index]
        blended = (1.0 - self.feedback_blend) * current + self.feedback_blend * values

        self._current_state = self._current_state.at[self._readout_state_index].set(blended)
        self._history_solution = self._tvb_types["NativeSolution"](
            ts=self._history_solution.ts,
            ys=self._history_solution.ys.at[-1, self._readout_state_index].set(blended),
            dt=self.dt_ms,
        )
        self.network.update_history(self._history_solution)

    def _append_history_point(self, time_ms: float, state: Any) -> None:
        jnp = self._tvb_types["jnp"]
        NativeSolution = self._tvb_types["NativeSolution"]

        ts = jnp.concatenate([self._history_solution.ts, jnp.asarray([time_ms], dtype=jnp.float64)], axis=0)
        ys = jnp.concatenate([self._history_solution.ys, state[None, ...]], axis=0)
        ts, ys = self._trim_history(ts, ys)
        self._history_solution = NativeSolution(ts=ts, ys=ys, dt=self.dt_ms)
        self.network.update_history(self._history_solution)

    def _state_to_rate(self, state: Any) -> np.ndarray:
        readout = np.asarray(state[self._readout_state_index], dtype=np.float64)
        readout = readout * self.readout_scale + self.readout_offset
        if self.readout_transform is not None:
            readout = np.asarray(self.readout_transform(readout), dtype=np.float64)
        return np.maximum(self.min_rate_hz, readout)

    def step(self, feedback: TVBFeedback | None = None) -> TVBStepOutput:
        if self._tvb_prepare is None:
            raise RuntimeError("Adapter is not initialized. Call initialize() first.")

        self._apply_feedback(feedback)

        t0 = float(self._time_ms)
        t1 = float(self._time_ms + self.dt_ms)

        solve_fn, config = self._tvb_prepare(self.network, self.solver, t0=t0, t1=t1, dt=self.dt_ms)
        result = solve_fn(config)
        if result.ys.shape[0] == 0:
            raise RuntimeError("tvboptim returned an empty trajectory for one-step integration.")

        next_state = result.ys[-1][: self._n_states]
        self._current_state = next_state
        self._time_ms = t1
        self._append_history_point(self._time_ms, self._current_state)

        rate = self._state_to_rate(self._current_state)
        return TVBStepOutput(time_ms=self._time_ms, rate_by_node=rate)
