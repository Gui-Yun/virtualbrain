from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import time

import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking upward until `src/fnn` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "fnn").exists() and (candidate / "notebook").exists():
            return candidate
    raise FileNotFoundError("Could not find a project root containing src/fnn and notebook.")


def ensure_src_on_sys_path(project_root: Path) -> None:
    import sys

    for path in [
        project_root / "notebook",
        project_root / "src" / "fnn",
        project_root / "src" / "tvb-multiscale",
    ]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def set_reproducible_seed(seed: int = 7) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def make_moving_bar_video(
    frame_count: int = 90,
    height: int = 144,
    width: int = 256,
    bar_width: int = 24,
    bar_value: int = 220,
    background_value: int = 32,
    drift_per_frame: int = 5,
) -> np.ndarray:
    """Create a deterministic grayscale movie in the shape expected by FNN."""
    frames = np.full((frame_count, height, width), background_value, dtype=np.uint8)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    baseline = np.clip(background_value + 30.0 * np.sin(2 * np.pi * yy), 0, 255).astype(np.uint8)
    frames[:] = baseline[None, :, :]

    for t in range(frame_count):
        left = (t * drift_per_frame) % (width + bar_width) - bar_width
        right = left + bar_width
        l = max(left, 0)
        r = min(right, width)
        if l < r:
            frames[t, :, l:r] = bar_value

        # Add a weak sinusoidal luminance modulation so the drive is not piecewise flat.
        delta = 16.0 * np.sin(2 * np.pi * t / max(frame_count, 1))
        frames[t] = np.clip(frames[t].astype(np.float32) + delta, 0, 255).astype(np.uint8)

    return frames


def make_block_video(
    seconds_per_block: int = 1,
    fps: int = 30,
    levels: Sequence[int] = (0, 128, 255),
    height: int = 144,
    width: int = 256,
) -> np.ndarray:
    """Reproduce the simple block stimulus pattern used in the official FNN demo."""
    block_frames = seconds_per_block * fps
    frames = [
        np.full((block_frames, height, width), fill_value=level, dtype=np.uint8)
        for level in levels
    ]
    return np.concatenate(frames, axis=0)


def build_random_fnn(
    units: int = 64,
    device: str = "cpu",
    randomize: bool = True,
    weight_scale: float = 0.05,
    bias_scale: float = 0.01,
):
    from fnn.microns.build import network

    model = network(units)
    if randomize:
        randomize_fnn_parameters(model, weight_scale=weight_scale, bias_scale=bias_scale)
    model.eval()
    if device != "cpu":
        model.to(device=device)
    return model


def randomize_fnn_parameters(
    model,
    weight_scale: float = 0.05,
    bias_scale: float = 0.01,
):
    import torch

    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.ndim >= 2:
                param.normal_(mean=0.0, std=weight_scale)
            else:
                param.normal_(mean=0.0, std=bias_scale)
    return model


def load_pretrained_fnn(
    params_dir: Path,
    session: int = 8,
    scan_idx: int = 5,
    cuda: bool = False,
):
    from fnn import microns

    model, ids = microns.scan(
        session=session,
        scan_idx=scan_idx,
        cuda=cuda,
        directory=str(params_dir),
    )
    model.eval()
    return model, ids


def predict_responses(
    model,
    frames: np.ndarray,
    perspectives: Iterable[np.ndarray] | None = None,
    modulations: Iterable[np.ndarray] | None = None,
) -> np.ndarray:
    responses = model.predict(
        stimuli=frames,
        perspectives=perspectives,
        modulations=modulations,
    )
    return np.asarray(responses, dtype=np.float32)


def build_constant_covariates(
    frame_count: int,
    perspective: Sequence[float] = (0.0, 0.0),
    modulation: Sequence[float] = (0.0, 0.0),
):
    perspective_arr = np.asarray(perspective, dtype=np.float32)
    modulation_arr = np.asarray(modulation, dtype=np.float32)
    perspectives = [perspective_arr.copy() for _ in range(frame_count)]
    modulations = [modulation_arr.copy() for _ in range(frame_count)]
    return perspectives, modulations


def summarize_population_drive(
    responses: np.ndarray,
    method: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    if method == "mean":
        drive = responses.mean(axis=1)
    elif method == "median":
        drive = np.median(responses, axis=1)
    elif method == "max":
        drive = responses.max(axis=1)
    else:
        raise ValueError(f"Unsupported drive summary method: {method}")

    drive = drive.astype(np.float64)
    if normalize:
        drive -= drive.min()
        peak = drive.max()
        if peak > 0:
            drive /= peak
    return drive


def frame_times_ms(frame_count: int, fps: float = 30.0) -> np.ndarray:
    dt = 1000.0 / fps
    return np.arange(frame_count, dtype=np.float64) * dt


def build_sampled_temporal_equation(
    sample_times_ms: Sequence[float],
    sample_values: Sequence[float],
):
    """
    Build a TVB-compatible temporal equation backed by a sampled time series.
    """
    from tvb.datatypes import equations

    sample_times_ms = np.asarray(sample_times_ms, dtype=np.float64).reshape(-1)
    sample_values = np.asarray(sample_values, dtype=np.float64).reshape(-1)
    if sample_times_ms.shape != sample_values.shape:
        raise ValueError("Sample times and values must have the same shape.")
    if sample_times_ms.ndim != 1:
        raise ValueError("Sample times must be one-dimensional.")

    class SampledTemporalEquation(equations.TemporalApplicableEquation):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.equation = "sampled_time_series"
            self.parameters = {"mode": "piecewise_linear_interpolation"}
            self.sample_times_ms = sample_times_ms
            self.sample_values = sample_values

        def evaluate(self, var):
            points = np.asarray(var, dtype=np.float64)
            values = np.interp(
                points.reshape(-1),
                self.sample_times_ms,
                self.sample_values,
                left=self.sample_values[0],
                right=self.sample_values[-1],
            )
            return values.reshape(points.shape)

    return SampledTemporalEquation()


def build_visual_weights(
    region_labels: Sequence[str],
    target_labels: Sequence[str] = ("rV1", "rV2", "lV1", "lV2"),
    amplitude: float = 1.0,
) -> np.ndarray:
    weights = np.zeros(len(region_labels), dtype=np.float64)
    targets = set(target_labels)
    for idx, label in enumerate(map(str, region_labels)):
        if label in targets:
            weights[idx] = amplitude
    return weights


def build_sampled_region_stimulus(
    connectivity,
    sample_times_ms: Sequence[float],
    sample_values: Sequence[float],
    target_labels: Sequence[str] = ("rV1", "rV2", "lV1", "lV2"),
    amplitude: float = 0.25,
):
    from tvb.datatypes import equations, patterns

    weights = build_visual_weights(
        region_labels=connectivity.region_labels.tolist(),
        target_labels=target_labels,
        amplitude=amplitude,
    )
    stimulus = patterns.StimuliRegion(
        connectivity=connectivity,
        spatial=equations.DiscreteEquation(),
        temporal=build_sampled_temporal_equation(sample_times_ms, sample_values),
        weight=weights,
    )
    return stimulus


def build_tvb_simulator(
    simulation_length_ms: float,
    dt_ms: float = 1.0,
    monitor_period_ms: float = 10.0,
    stimulus=None,
):
    from tvb.simulator import simulator as simulator_module
    from tvb.simulator.lab import connectivity, coupling, integrators, monitors, models

    conn = connectivity.Connectivity.from_file()
    conn.weights = conn.scaled_weights(mode="region")
    conn.configure()

    integrator = integrators.HeunStochastic(dt=dt_ms)
    integrator.noise.nsig = np.array([0.02, 0.02])

    sim = simulator_module.Simulator(
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([0.0154])),
        model=models.Generic2dOscillator(),
        integrator=integrator,
        monitors=(monitors.Raw(period=monitor_period_ms),),
        stimulus=stimulus,
        simulation_length=simulation_length_ms,
    )
    return sim


def run_tvb(simulator, simulation_length_ms: float):
    simulator.configure()
    return simulator.run(simulation_length=simulation_length_ms)


def unpack_first_monitor(results):
    if not results:
        raise ValueError("Simulator returned no monitor outputs.")
    times, values = results[0]
    return np.asarray(times).reshape(-1), np.asarray(values)


def extract_region_trace(
    times: np.ndarray,
    values: np.ndarray,
    simulator,
    region_label: str,
    variable_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = [str(x) for x in simulator.connectivity.region_labels.tolist()]
    try:
        region_index = labels.index(region_label)
    except ValueError as exc:
        raise ValueError(f"Region label `{region_label}` not found.") from exc

    if variable_name is None:
        variable_index = 0
    else:
        voi = list(simulator.model.variables_of_interest)
        try:
            variable_index = voi.index(variable_name)
        except ValueError as exc:
            raise ValueError(f"Variable `{variable_name}` not found in {voi}.") from exc

    trace = values[:, variable_index, region_index, 0]
    return times, trace


def region_indices_from_labels(connectivity, target_labels: Sequence[str]) -> np.ndarray:
    region_labels = [str(x) for x in connectivity.region_labels.tolist()]
    indices = []
    for label in target_labels:
        try:
            indices.append(region_labels.index(str(label)))
        except ValueError as exc:
            raise ValueError(f"Region label `{label}` not found.") from exc
    return np.asarray(indices, dtype=int)


def extract_mean_region_trace(
    times: np.ndarray,
    values: np.ndarray,
    simulator,
    region_labels: Sequence[str],
    variable_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    traces = []
    for label in region_labels:
        _, trace = extract_region_trace(
            times=times,
            values=values,
            simulator=simulator,
            region_label=label,
            variable_name=variable_name,
        )
        traces.append(trace)
    return times, np.mean(np.asarray(traces), axis=0)


def calibrate_drive_to_reference(
    drive: Sequence[float],
    reference_trace: Sequence[float],
    gain: float = 1.0,
) -> np.ndarray:
    drive = np.asarray(drive, dtype=np.float64)
    reference_trace = np.asarray(reference_trace, dtype=np.float64)

    drive_mean = float(np.mean(drive))
    drive_std = float(np.std(drive))
    ref_mean = float(np.mean(reference_trace))
    ref_std = float(np.std(reference_trace))

    if drive_std <= 1e-12:
        standardized = np.zeros_like(drive)
    else:
        standardized = (drive - drive_mean) / drive_std

    return ref_mean + gain * max(ref_std, 1e-6) * standardized


def map_drive_to_reference_quantiles(
    drive: Sequence[float],
    reference_trace: Sequence[float],
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> np.ndarray:
    drive = np.asarray(drive, dtype=np.float64)
    reference_trace = np.asarray(reference_trace, dtype=np.float64)

    lo = float(np.quantile(reference_trace, lower_quantile))
    hi = float(np.quantile(reference_trace, upper_quantile))
    return lo + drive * (hi - lo)


def _collect_results_from_iterator(simulator, data_iterator, simulation_length_ms: float):
    ts, xs = [], []
    for _ in simulator.monitors:
        ts.append([])
        xs.append([])

    wall_time_start = time.time()
    for data in data_iterator:
        for tl, xl, t_x in zip(ts, xs, data):
            if t_x is not None:
                t, x = t_x
                tl.append(t)
                xl.append(x)

    elapsed_wall_time = time.time() - wall_time_start
    simulator.log.info(
        "%.3f s elapsed, %.3fx real time",
        elapsed_wall_time,
        elapsed_wall_time * 1e3 / simulation_length_ms,
    )

    for i in range(len(ts)):
        ts[i] = np.asarray(ts[i])
        xs[i] = np.asarray(xs[i])

    return list(zip(ts, xs))


def run_tvb_with_soft_replacement(
    simulator,
    simulation_length_ms: float,
    sample_times_ms: Sequence[float],
    sample_values: Sequence[float],
    target_labels: Sequence[str] = ("rV1", "rV2", "lV1", "lV2"),
    blend_alpha: float = 0.35,
    variable_index: int = 0,
):
    sample_times_ms = np.asarray(sample_times_ms, dtype=np.float64).reshape(-1)
    sample_values = np.asarray(sample_values, dtype=np.float64).reshape(-1)
    if sample_times_ms.shape != sample_values.shape:
        raise ValueError("Sample times and values must have the same shape.")
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("blend_alpha must be in [0, 1].")

    simulator.configure()
    target_indices = region_indices_from_labels(simulator.connectivity, target_labels)

    local_coupling = simulator._prepare_local_coupling()
    stimulus = simulator._prepare_stimulus()
    state = simulator.current_state
    start_step = simulator.current_step + 1
    node_coupling = simulator._loop_compute_node_coupling(start_step)
    n_steps = int(np.ceil(simulation_length_ms / simulator.integrator.dt))

    replacement_times = []
    replacement_values = []

    def iterator():
        nonlocal state, node_coupling
        for step in range(start_step, start_step + n_steps):
            simulator._loop_update_stimulus(step, stimulus)
            state = simulator.integrate_next_step(
                state,
                simulator.model,
                node_coupling,
                local_coupling,
                stimulus,
            )

            current_time_ms = step * simulator.integrator.dt
            target_value = float(
                np.interp(
                    current_time_ms,
                    sample_times_ms,
                    sample_values,
                    left=sample_values[0],
                    right=sample_values[-1],
                )
            )

            previous_values = state[variable_index, target_indices, 0]
            state[variable_index, target_indices, 0] = (
                (1.0 - blend_alpha) * previous_values + blend_alpha * target_value
            )

            replacement_times.append(current_time_ms)
            replacement_values.append(target_value)

            simulator._loop_update_history(step, state)
            node_coupling = simulator._loop_compute_node_coupling(step + 1)
            output = simulator._loop_monitor_output(step, state, node_coupling)
            if output is not None:
                yield output

        simulator.current_state = state
        simulator.current_step = simulator.current_step + n_steps

    results = _collect_results_from_iterator(simulator, iterator(), simulation_length_ms)
    return results, np.asarray(replacement_times), np.asarray(replacement_values)


@dataclass
class CouplingSummary:
    frame_times_ms: np.ndarray
    fnn_drive: np.ndarray
    target_labels: tuple[str, ...]
    simulation_length_ms: float


@dataclass
class SoftReplacementSummary:
    replacement_times_ms: np.ndarray
    replacement_values: np.ndarray
    target_labels: tuple[str, ...]
    blend_alpha: float
