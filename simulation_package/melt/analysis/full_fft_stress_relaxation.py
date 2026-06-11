#!/usr/bin/env python3
"""Compute full-FFT stress relaxation curves for ReactiveLJ melt runs.

This standalone script reads every ``virial_tensor_log.gsd`` under an input
root, computes the shear-stress autocovariance from the off-diagonal tensor
components with an unbiased full FFT estimator at every native lag, aggregates
replicates by ``(epsilon, p)``, and writes a direct-lag plot using the same
styling as the current stress-modulus analysis plot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib
import numpy as np
import gsd.hoomd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 1000.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0
DEFAULT_STRESS_MAX_RUNTIME_FRACTION = 1.0 / 3.0
DEFAULT_EXTENSION_STITCH_TIME = 25.0
PLOT_MIN_G = 1.0e-3
DEFAULT_WEAKENING_EXPONENT = 4.0
ConditionKey = Tuple[float, float]

# Match the Liu/O'Connor Green-Kubo estimator: average only shear components
# with alpha != beta (xy, xz, yz).
STRESS_COMPONENT_WEIGHTS = np.array(
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RunSpec:
    virial_path: Path
    metadata_path: Path
    epsilon: float
    weakening_exponent: float
    replicate: int


@dataclass
class ReplicateResult:
    epsilon: float
    weakening_exponent: float
    replicate: int
    virial_path: str
    time: np.ndarray
    modulus: np.ndarray
    counts: np.ndarray
    sample_dt: float
    n_samples: int
    n_segments: int
    longest_segment_samples: int
    max_lag: int


@dataclass
class AggregateResult:
    epsilon: float
    weakening_exponent: float
    time: np.ndarray
    values: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray
    weight_sum: np.ndarray
    n_replicates: np.ndarray


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    melt_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-FFT stress relaxation from virial_tensor_log.gsd files "
            "and plot G(t) by condition."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=melt_dir / "data_generation" / "outputs_clean",
        help="Root directory containing eps_*/rep_*/virial_tensor_log.gsd files.",
    )
    parser.add_argument(
        "--extension-input-root",
        type=Path,
        default=melt_dir / "data_generation" / "outputs_virial_extended",
        help=(
            "Optional mirrored root containing short dense virial extensions. "
            "When present, these curves are stitched into the short-time region."
        ),
    )
    parser.add_argument(
        "--no-extension-input",
        action="store_true",
        help="Ignore --extension-input-root and analyze only --input-root.",
    )
    parser.add_argument(
        "--extension-stitch-time",
        type=float,
        default=DEFAULT_EXTENSION_STITCH_TIME,
        help=(
            "Lag time in tau_LJ at which to switch from dense extension curves "
            "to the original long production curves. Default: 25 tau_LJ, "
            "the first native lag of the base virial logs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "full_fft_stress_relaxation",
        help="Directory for per-condition CSVs, diagnostics, and the SVG plot.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help=(
            "Number of replicate-level FFT workers. 0 uses SLURM_CPUS_PER_TASK "
            "when available, otherwise all visible CPUs."
        ),
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        default=None,
        help="Optional epsilon subset to process.",
    )
    parser.add_argument(
        "--weakening-exponents",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional coordination weakening exponent p subset to process. "
            "Default: p=4, including legacy directories where p is not named."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help=(
            "Optional exact condition filters, formatted like eps=18,p=2 or 18:2. "
            "When provided, --epsilons and --weakening-exponents are ignored."
        ),
    )
    parser.add_argument(
        "--max-replicates-per-epsilon",
        type=int,
        default=0,
        help=(
            "Optional cap per condition for quick checks. "
            "0 processes all discovered replicates."
        ),
    )
    parser.add_argument(
        "--stress-max-runtime-fraction",
        type=float,
        default=DEFAULT_STRESS_MAX_RUNTIME_FRACTION,
        help="Maximum lag as a fraction of each virial-log runtime.",
    )
    parser.add_argument(
        "--max-lag-time",
        type=float,
        default=MAX_ANALYSIS_LAG_TIME,
        help=(
            "Absolute maximum lag time in tau_LJ units. Default matches the "
            "current analysis cap of 1000 tau_R^0."
        ),
    )
    parser.add_argument(
        "--tau-r0",
        type=float,
        default=FALLBACK_TAU_R0,
        help="Reference tau_R^0 used to scale the plot x-axis.",
    )
    parser.add_argument(
        "--min-plot-g",
        type=float,
        default=PLOT_MIN_G,
        help=(
            "After optional plot log-binning, stop plotting each curve at the "
            "first point where G is below this value."
        ),
    )
    parser.add_argument(
        "--plot-name",
        default="stress_modulus_vs_time_lag_by_epsilon.svg",
        help="SVG plot filename written under --output-dir.",
    )
    parser.add_argument(
        "--csv-name",
        default="stress_modulus.csv",
        help="Per-condition aggregate CSV filename.",
    )
    parser.add_argument(
        "--joblib-verbose",
        type=int,
        default=10,
        help="Verbosity passed to joblib. Use 0 for quiet runs.",
    )
    parser.add_argument(
        "--plot-linear-lags",
        type=int,
        default=0,
        help="Number of initial native FFT lag points to plot without log binning.",
    )
    parser.add_argument(
        "--plot-bins-per-decade",
        type=float,
        default=10.0,
        help="Number of logarithmic plot bins per decade after --plot-linear-lags.",
    )
    parser.add_argument(
        "--no-plot-log-binning",
        action="store_true",
        help="Plot every native FFT lag instead of log-binning the plot curve.",
    )
    parser.add_argument(
        "--plot-all-curves",
        action="store_true",
        help=(
            "Plot every replicate curve colored by condition instead of only the "
            "aggregate mean curve for each condition."
        ),
    )
    parser.add_argument(
        "--colormap",
        default="plasma",
        help="Matplotlib colormap for condition curves. Default: plasma.",
    )
    return parser.parse_args()


def resolve_n_jobs(requested: int) -> int:
    if requested > 0:
        return requested
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            parsed = int(slurm_cpus)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def load_metadata(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_prefixed_float(path: Path, prefix: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(prefix)}([-+]?\d+(?:\.\d+)?)$")
    for part in reversed(path.parts):
        match = pattern.match(part)
        if match is None:
            continue
        return float(match.group(1))
    return None


def parse_replicate(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.match(r"^rep_(\d+)$", part)
        if match is not None:
            return int(match.group(1))
    return 0


def value_is_selected(value: float, selected: Iterable[float] | None) -> bool:
    if selected is None:
        return True
    return any(math.isclose(value, selected_value, rel_tol=0.0, abs_tol=1.0e-12) for selected_value in selected)


def parse_condition_token(token: str) -> ConditionKey:
    text = token.strip()
    if not text:
        raise RuntimeError("Empty condition token.")

    if ":" in text and "=" not in text:
        eps_text, p_text = text.split(":", 1)
        return float(eps_text), float(p_text)

    epsilon: float | None = None
    weakening_exponent: float | None = None
    for part in text.split(","):
        if "=" not in part:
            raise RuntimeError(
                f"Could not parse condition '{token}'. Use eps=18,p=2 or 18:2."
            )
        name, value = part.split("=", 1)
        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"eps", "epsilon", "reactive_epsilon"}:
            epsilon = float(value)
        elif normalized in {"p", "weakening_exponent", "weakening"}:
            weakening_exponent = float(value)
        else:
            raise RuntimeError(f"Unknown condition field '{name}' in '{token}'.")

    if epsilon is None or weakening_exponent is None:
        raise RuntimeError(
            f"Condition '{token}' must specify both epsilon and p."
        )
    return epsilon, weakening_exponent


def parse_condition_filters(tokens: List[str] | None) -> Set[ConditionKey] | None:
    if tokens is None:
        return None
    if len(tokens) == 0:
        raise RuntimeError("--conditions was provided without any condition values.")
    return {parse_condition_token(token) for token in tokens}


def condition_is_selected(
    epsilon: float,
    weakening_exponent: float,
    selected_conditions: Set[ConditionKey] | None,
) -> bool:
    if selected_conditions is None:
        return True
    return any(
        math.isclose(epsilon, selected_epsilon, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(
            weakening_exponent,
            selected_weakening_exponent,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for selected_epsilon, selected_weakening_exponent in selected_conditions
    )


def discover_runs(
    input_root: Path,
    selected_epsilons: Iterable[float] | None,
    selected_weakening_exponents: Iterable[float] | None,
    selected_conditions: Set[ConditionKey] | None,
    max_replicates_per_epsilon: int,
) -> List[RunSpec]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ", "archived"}
    runs: List[RunSpec] = []
    for root, dirs, files in os.walk(input_root, topdown=True):
        dirs[:] = sorted(d for d in dirs if d not in excluded_dirs)
        if "virial_tensor_log.gsd" not in files:
            continue

        run_dir = Path(root)
        virial_path = run_dir / "virial_tensor_log.gsd"
        metadata_path = run_dir / "metadata.json"
        metadata = load_metadata(metadata_path)
        epsilon = metadata.get("reactive_epsilon", parse_prefixed_float(run_dir, "eps_"))
        weakening_exponent = metadata.get("weakening_exponent")
        if weakening_exponent is None:
            weakening_exponent = parse_prefixed_float(run_dir, "p_")
        if weakening_exponent is None:
            weakening_exponent = DEFAULT_WEAKENING_EXPONENT
        replicate = metadata.get("replicate", parse_replicate(run_dir))
        if epsilon is None:
            log(f"Skipping {virial_path}: could not infer epsilon")
            continue

        epsilon_f = float(epsilon)
        weakening_exponent_f = float(weakening_exponent)
        if selected_conditions is not None:
            if not condition_is_selected(
                epsilon_f, weakening_exponent_f, selected_conditions
            ):
                continue
        else:
            if not value_is_selected(epsilon_f, selected_epsilons):
                continue
            if not value_is_selected(
                weakening_exponent_f, selected_weakening_exponents
            ):
                continue
        runs.append(
            RunSpec(
                virial_path=virial_path,
                metadata_path=metadata_path,
                epsilon=epsilon_f,
                weakening_exponent=weakening_exponent_f,
                replicate=int(replicate),
            )
        )

    runs.sort(
        key=lambda item: (
            item.epsilon,
            item.weakening_exponent,
            item.replicate,
            str(item.virial_path),
        )
    )
    if max_replicates_per_epsilon > 0:
        kept: List[RunSpec] = []
        counts: Dict[ConditionKey, int] = defaultdict(int)
        for run in runs:
            key = (run.epsilon, run.weakening_exponent)
            if counts[key] >= max_replicates_per_epsilon:
                continue
            kept.append(run)
            counts[key] += 1
        runs = kept
    return runs


def find_virial_key(frame) -> str | None:
    if not hasattr(frame, "log"):
        return None
    for key in frame.log.keys():
        if "virial_tensor" in key:
            return key
    return None


def parse_virial_tensor_components(virial_val) -> Tuple[float, float, float, float, float, float] | None:
    virial_arr = np.asarray(np.squeeze(virial_val), dtype=np.float64)
    if virial_arr.ndim == 0 or virial_arr.shape[-1] < 6:
        return None
    # HOOMD tensor ordering is [xx, xy, xz, yy, yz, zz].
    return (
        float(virial_arr[0]),
        float(virial_arr[1]),
        float(virial_arr[2]),
        float(virial_arr[3]),
        float(virial_arr[4]),
        float(virial_arr[5]),
    )


def load_virial_series(virial_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with gsd.hoomd.open(str(virial_path), "r") as traj:
        n_frames = len(traj)
        if n_frames == 0:
            return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
        virial_key = find_virial_key(traj[0])
        if virial_key is None:
            return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)

        virial = np.empty((n_frames, 6), dtype=np.float64)
        steps = np.empty((n_frames,), dtype=np.int64)
        count = 0
        for frame in traj:
            if not hasattr(frame, "log"):
                continue
            virial_val = frame.log.get(virial_key, None)
            if virial_val is None:
                continue
            parsed = parse_virial_tensor_components(virial_val)
            if parsed is None:
                continue
            virial[count, :] = parsed
            steps[count] = int(frame.configuration.step)
            count += 1

    return virial[:count, :], steps[:count]


def infer_sample_step_delta(
    sample_steps: np.ndarray,
    fallback_step_stride: float,
) -> int:
    if sample_steps.size >= 2:
        diffs = np.diff(sample_steps)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size > 0:
            return int(round(float(np.median(positive_diffs))))
    return int(round(float(fallback_step_stride)))


def infer_sample_dt(sample_steps: np.ndarray, dt: float, fallback_step_stride: float) -> float:
    return dt * float(infer_sample_step_delta(sample_steps, fallback_step_stride))


def split_contiguous_segments(
    sample_steps: np.ndarray,
    expected_step_delta: int,
) -> List[Tuple[int, int]]:
    if sample_steps.size == 0:
        return []
    if sample_steps.size == 1:
        return [(0, 1)]

    diffs = np.diff(sample_steps)
    break_after = np.flatnonzero(diffs != int(expected_step_delta))
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), break_after.astype(np.int64) + 1)
    )
    ends = np.concatenate(
        (break_after.astype(np.int64) + 1, np.array([sample_steps.size], dtype=np.int64))
    )
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def resolve_box_length(metadata: Dict, run_dir: Path, virial_path: Path) -> float:
    for key in ("target_box_length", "initial_box_length", "box_length"):
        value = metadata.get(key)
        if value is None:
            continue
        box_length = float(value)
        if np.isfinite(box_length) and box_length > 0.0:
            return box_length

    trajectory_path = run_dir / "trajectory.gsd"
    if trajectory_path.exists():
        with gsd.hoomd.open(str(trajectory_path), "r") as traj:
            if len(traj) > 0:
                box_length = float(traj[0].configuration.box[0])
                if np.isfinite(box_length) and box_length > 0.0:
                    return box_length

    with gsd.hoomd.open(str(virial_path), "r") as traj:
        if len(traj) > 0:
            box_length = float(traj[0].configuration.box[0])
            if np.isfinite(box_length) and box_length > 0.0:
                return box_length

    raise RuntimeError(f"Could not determine box length for {virial_path}")


def compute_shear_stress_autocovariance_fft(
    virial_arr: np.ndarray,
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return full-grid unbiased shear-stress autocovariance and counts."""
    return compute_segmented_shear_stress_autocovariance_fft(
        virial_arr,
        [(0, int(virial_arr.shape[0]))],
        max_lag,
    )


def compute_segmented_shear_stress_autocovariance_fft(
    virial_arr: np.ndarray,
    segments: List[Tuple[int, int]],
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return autocovariance using only pairs inside contiguous segments."""
    if virial_arr.ndim != 2 or virial_arr.shape[0] <= 1 or virial_arr.shape[1] < 6:
        raise ValueError("virial_arr must have shape (n_samples, >= 6)")

    usable_segments = [
        (int(start), int(end))
        for start, end in segments
        if int(end) - int(start) > 1
    ]
    if not usable_segments:
        raise ValueError("No contiguous segment contains at least two samples")

    n_samples = int(virial_arr.shape[0])
    longest_segment = max(end - start for start, end in usable_segments)
    max_lag = max(1, min(int(max_lag), longest_segment - 1))

    xy = virial_arr[:, 1]
    xz = virial_arr[:, 2]
    yz = virial_arr[:, 4]

    components = np.empty((n_samples, 3), dtype=np.float64)
    components[:, 0] = xy
    components[:, 1] = xz
    components[:, 2] = yz
    components -= np.mean(components, axis=0, keepdims=True)

    acf_sums = np.zeros((max_lag + 1, 3), dtype=np.float64)
    counts = np.zeros((max_lag + 1,), dtype=np.float64)
    for start, end in usable_segments:
        segment_components = components[start:end]
        n_segment = int(segment_components.shape[0])
        segment_max_lag = min(max_lag, n_segment - 1)
        fft_length = 2 * n_segment
        spectrum = np.fft.rfft(segment_components, n=fft_length, axis=0)
        spectrum *= np.conjugate(spectrum)
        segment_sums = np.fft.irfft(spectrum, n=fft_length, axis=0)[
            : segment_max_lag + 1
        ].real
        acf_sums[: segment_max_lag + 1] += segment_sums
        counts[: segment_max_lag + 1] += np.arange(
            n_segment,
            n_segment - segment_max_lag - 1,
            -1,
            dtype=np.float64,
        )

    acf_components = np.divide(
        acf_sums,
        counts[:, None],
        out=np.full_like(acf_sums, np.nan),
        where=counts[:, None] > 0.0,
    )
    return acf_components @ STRESS_COMPONENT_WEIGHTS, counts


def compute_replicate(
    run: RunSpec,
    stress_max_runtime_fraction: float,
    max_lag_time: float,
) -> ReplicateResult | None:
    metadata = load_metadata(run.metadata_path)
    virial_arr, virial_steps = load_virial_series(run.virial_path)
    if virial_arr.shape[0] <= 1:
        log(f"Skipping {run.virial_path}: no usable virial samples")
        return None

    dt = float(metadata.get("dt", 0.005))
    fallback_stride = float(metadata.get("virial_log_steps", metadata.get("frame_steps", 100_000)))
    sample_step_delta = infer_sample_step_delta(virial_steps, fallback_stride)
    sample_dt = dt * float(sample_step_delta)
    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        raise RuntimeError(f"Invalid virial sampling interval for {run.virial_path}: {sample_dt}")

    segments = split_contiguous_segments(virial_steps, sample_step_delta)
    usable_segments = [
        (start, end)
        for start, end in segments
        if end - start > 1
    ]
    if not usable_segments:
        log(f"Skipping {run.virial_path}: no contiguous segment has at least two samples")
        return None

    longest_segment_samples = max(end - start for start, end in usable_segments)
    runtime = sum(
        float(end - start - 1) * sample_dt
        for start, end in usable_segments
    )
    max_time = min(float(stress_max_runtime_fraction) * runtime, float(max_lag_time))
    max_lag = int(np.floor(max_time / sample_dt + 1.0e-12))
    max_lag = max(1, min(max_lag, longest_segment_samples - 1))

    covariance, counts = compute_segmented_shear_stress_autocovariance_fft(
        virial_arr,
        usable_segments,
        max_lag,
    )
    box_length = resolve_box_length(metadata, run.virial_path.parent, run.virial_path)
    temperature = float(metadata.get("temperature", 1.0))
    modulus = (box_length**3 / temperature) * covariance
    lags = np.arange(1, max_lag + 1, dtype=np.float64)

    return ReplicateResult(
        epsilon=run.epsilon,
        weakening_exponent=run.weakening_exponent,
        replicate=run.replicate,
        virial_path=str(run.virial_path),
        time=lags * sample_dt,
        modulus=modulus[1:],
        counts=counts[1:],
        sample_dt=sample_dt,
        n_samples=virial_arr.shape[0],
        n_segments=len(usable_segments),
        longest_segment_samples=longest_segment_samples,
        max_lag=max_lag,
    )


def finite_column_stderr(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    value_sum = np.sum(np.where(finite, values, 0.0), axis=0)
    mean = np.divide(
        value_sum,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    centered = np.where(finite, values - mean, 0.0)
    ss = np.sum(centered * centered, axis=0)
    variance = np.divide(
        ss,
        counts - 1,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=counts > 1,
    )
    stderr = np.divide(
        np.sqrt(variance),
        np.sqrt(counts),
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 1,
    )
    return stderr


def aggregate_one_condition(
    condition: ConditionKey, results: List[ReplicateResult]
) -> AggregateResult:
    epsilon, weakening_exponent = condition
    if not results:
        raise RuntimeError(
            f"No replicate results to aggregate for eps={epsilon:g}, p={weakening_exponent:g}"
        )

    base_idx = max(range(len(results)), key=lambda idx: results[idx].time.size)
    base_time = results[base_idx].time
    values = np.full((len(results), base_time.size), np.nan, dtype=np.float64)
    weights = np.zeros((len(results), base_time.size), dtype=np.float64)

    for row_idx, result in enumerate(results):
        insert_idx = np.searchsorted(base_time, result.time)
        in_bounds = insert_idx < base_time.size
        valid = np.zeros(result.time.size, dtype=bool)
        if np.any(in_bounds):
            matched = np.isclose(
                base_time[insert_idx[in_bounds]],
                result.time[in_bounds],
                rtol=0.0,
                atol=1.0e-9,
            )
            valid[in_bounds] = matched
        if np.any(valid):
            values[row_idx, insert_idx[valid]] = result.modulus[valid]
            weights[row_idx, insert_idx[valid]] = result.counts[valid]

    populated = np.any(np.isfinite(values), axis=0)
    if not np.any(populated):
        raise RuntimeError(
            f"No populated time points after aggregation for eps={epsilon:g}, p={weakening_exponent:g}"
        )

    time = base_time[populated]
    values = values[:, populated]
    weights = weights[:, populated]
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    weight_sum = np.sum(np.where(valid, weights, 0.0), axis=0)
    weighted_value_sum = np.sum(np.where(valid, weights * values, 0.0), axis=0)
    mean = np.divide(
        weighted_value_sum,
        weight_sum,
        out=np.full(time.shape, np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    return AggregateResult(
        epsilon=epsilon,
        weakening_exponent=weakening_exponent,
        time=time,
        values=values,
        mean=mean,
        stderr=finite_column_stderr(values),
        weight_sum=weight_sum,
        n_replicates=np.sum(valid, axis=0).astype(np.int64),
    )


def aggregate_replicates(results: List[ReplicateResult]) -> Dict[ConditionKey, AggregateResult]:
    by_condition: Dict[ConditionKey, List[ReplicateResult]] = defaultdict(list)
    for result in results:
        by_condition[(result.epsilon, result.weakening_exponent)].append(result)
    return {
        condition: aggregate_one_condition(condition, condition_results)
        for condition, condition_results in sorted(by_condition.items())
    }


def stitch_extension_aggregates(
    base_aggregated: Dict[ConditionKey, AggregateResult],
    extension_aggregated: Dict[ConditionKey, AggregateResult],
    stitch_time: float,
) -> Dict[ConditionKey, AggregateResult]:
    """Use dense extension data before stitch_time and base data at/afterward."""
    if stitch_time <= 0.0:
        raise RuntimeError("--extension-stitch-time must be positive.")

    stitched: Dict[ConditionKey, AggregateResult] = {}
    for condition, base in sorted(base_aggregated.items()):
        extension = extension_aggregated.get(condition)
        if extension is None or extension.time.size == 0:
            stitched[condition] = base
            continue

        extension_mask = extension.time < float(stitch_time)
        base_mask = base.time >= float(stitch_time)
        if not np.any(extension_mask):
            stitched[condition] = base
            continue

        time = np.concatenate((extension.time[extension_mask], base.time[base_mask]))
        mean = np.concatenate((extension.mean[extension_mask], base.mean[base_mask]))
        stderr = np.concatenate(
            (extension.stderr[extension_mask], base.stderr[base_mask])
        )
        weight_sum = np.concatenate(
            (extension.weight_sum[extension_mask], base.weight_sum[base_mask])
        )
        n_replicates = np.concatenate(
            (extension.n_replicates[extension_mask], base.n_replicates[base_mask])
        )
        stitched[condition] = AggregateResult(
            epsilon=condition[0],
            weakening_exponent=condition[1],
            time=time,
            values=np.empty((0, time.size), dtype=np.float64),
            mean=mean,
            stderr=stderr,
            weight_sum=weight_sum,
            n_replicates=n_replicates,
        )

    return stitched


def write_timeseries(path: Path, time: np.ndarray, mean: np.ndarray, stderr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((time, mean, stderr))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time,mean,stderr",
        comments="",
        fmt="%.6e",
    )


def write_count_diagnostics(path: Path, result: AggregateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((result.time, result.weight_sum, result.n_replicates))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time,weight_sum,n_replicates",
        comments="",
        fmt=["%.6e", "%.6e", "%d"],
    )


def compute_shared_time_lag_xlim(
    g_time_by_condition: Dict[ConditionKey, np.ndarray],
    g_mean_by_condition: Dict[ConditionKey, np.ndarray],
    tau_r0: float,
) -> Tuple[float, float] | None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    positive_xmins: List[float] = []
    for condition, lag_time in g_time_by_condition.items():
        g_mean = g_mean_by_condition.get(condition)
        if g_mean is None:
            continue
        n = min(lag_time.size, g_mean.size)
        if n == 0:
            continue
        x = lag_time[:n] / tau_r0
        finite_lag = np.isfinite(x) & (x > 0.0)
        if np.any(finite_lag):
            positive_xmins.append(float(np.min(x[finite_lag])))

    if not positive_xmins:
        return None
    return min(positive_xmins), MAX_ANALYSIS_LAG_TAU_R0


def format_condition_label(
    condition: ConditionKey,
    all_conditions: List[ConditionKey],
) -> str:
    epsilon, weakening_exponent = condition
    epsilon_values = {item[0] for item in all_conditions}
    weakening_values = {item[1] for item in all_conditions}
    if len(epsilon_values) == 1 and len(weakening_values) > 1:
        return f"p={weakening_exponent:g}"
    if len(weakening_values) == 1:
        return rf"$\varepsilon_\mathrm{{RLJ}}={epsilon:g}$"
    return rf"$\varepsilon_\mathrm{{RLJ}}={epsilon:g}$, p={weakening_exponent:g}"


def condition_dir_name(condition: ConditionKey, include_p: bool) -> str:
    epsilon, weakening_exponent = condition
    if include_p:
        return f"eps_{epsilon:g}_p_{weakening_exponent:g}"
    return f"eps_{epsilon:g}"


def log_bin_timeseries_for_plot(
    time: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None,
    linear_lags: int,
    bins_per_decade: float,
) -> Tuple[np.ndarray, np.ndarray]:
    time_arr = np.asarray(time, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    n = min(time_arr.size, value_arr.size)
    if weights is None:
        weight_arr = np.ones(n, dtype=np.float64)
    else:
        weight_arr = np.asarray(weights, dtype=np.float64)
        n = min(n, weight_arr.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    time_arr = time_arr[:n]
    value_arr = value_arr[:n]
    weight_arr = weight_arr[:n]
    valid = (
        np.isfinite(time_arr)
        & np.isfinite(value_arr)
        & np.isfinite(weight_arr)
        & (time_arr > 0.0)
        & (weight_arr > 0.0)
    )
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    linear_count = min(max(0, int(linear_lags)), valid_idx.size)
    exact_idx = valid_idx[:linear_count]
    rest_idx = valid_idx[linear_count:]
    output_time: List[np.ndarray] = []
    output_values: List[np.ndarray] = []
    if exact_idx.size > 0:
        output_time.append(time_arr[exact_idx])
        output_values.append(value_arr[exact_idx])

    if rest_idx.size > 0:
        if bins_per_decade <= 0.0:
            output_time.append(time_arr[rest_idx])
            output_values.append(value_arr[rest_idx])
        else:
            t_min = float(time_arr[rest_idx[0]])
            t_max = float(np.max(time_arr[rest_idx]))
            if t_max <= t_min:
                output_time.append(time_arr[rest_idx])
                output_values.append(value_arr[rest_idx])
            else:
                n_bins = max(
                    1,
                    int(np.ceil((np.log10(t_max) - np.log10(t_min)) * bins_per_decade)),
                )
                edges = np.logspace(np.log10(t_min), np.log10(t_max), n_bins + 1)
                edges[0] = t_min
                edges[-1] = np.nextafter(t_max, np.inf)
                bin_indices = np.searchsorted(edges, time_arr[rest_idx], side="right") - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                binned_time: List[float] = []
                binned_values: List[float] = []
                for bin_index in np.unique(bin_indices):
                    idx = rest_idx[bin_indices == bin_index]
                    w = weight_arr[idx]
                    weight_sum = float(np.sum(w))
                    if weight_sum <= 0.0:
                        continue
                    binned_time.append(
                        float(np.exp(np.sum(w * np.log(time_arr[idx])) / weight_sum))
                    )
                    binned_values.append(float(np.sum(w * value_arr[idx]) / weight_sum))
                if binned_time:
                    output_time.append(np.asarray(binned_time, dtype=np.float64))
                    output_values.append(np.asarray(binned_values, dtype=np.float64))

    if not output_time:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.concatenate(output_time), np.concatenate(output_values)


def log_bin_mean_and_stderr_for_plot(
    time: np.ndarray,
    mean: np.ndarray,
    stderr: np.ndarray,
    weights: np.ndarray | None,
    linear_lags: int,
    bins_per_decade: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_arr = np.asarray(time, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    stderr_arr = np.asarray(stderr, dtype=np.float64)
    n = min(time_arr.size, mean_arr.size, stderr_arr.size)
    if weights is None:
        weight_arr = np.ones(n, dtype=np.float64)
    else:
        weight_arr = np.asarray(weights, dtype=np.float64)
        n = min(n, weight_arr.size)
    if n == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty

    time_arr = time_arr[:n]
    mean_arr = mean_arr[:n]
    stderr_arr = stderr_arr[:n]
    weight_arr = weight_arr[:n]
    valid = (
        np.isfinite(time_arr)
        & np.isfinite(mean_arr)
        & np.isfinite(stderr_arr)
        & np.isfinite(weight_arr)
        & (time_arr > 0.0)
        & (weight_arr > 0.0)
    )
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty

    linear_count = min(max(0, int(linear_lags)), valid_idx.size)
    exact_idx = valid_idx[:linear_count]
    rest_idx = valid_idx[linear_count:]
    output_time: List[np.ndarray] = []
    output_mean: List[np.ndarray] = []
    output_stderr: List[np.ndarray] = []
    if exact_idx.size > 0:
        output_time.append(time_arr[exact_idx])
        output_mean.append(mean_arr[exact_idx])
        output_stderr.append(stderr_arr[exact_idx])

    if rest_idx.size > 0:
        if bins_per_decade <= 0.0:
            output_time.append(time_arr[rest_idx])
            output_mean.append(mean_arr[rest_idx])
            output_stderr.append(stderr_arr[rest_idx])
        else:
            t_min = float(time_arr[rest_idx[0]])
            t_max = float(np.max(time_arr[rest_idx]))
            if t_max <= t_min:
                output_time.append(time_arr[rest_idx])
                output_mean.append(mean_arr[rest_idx])
                output_stderr.append(stderr_arr[rest_idx])
            else:
                n_bins = max(
                    1,
                    int(np.ceil((np.log10(t_max) - np.log10(t_min)) * bins_per_decade)),
                )
                edges = np.logspace(np.log10(t_min), np.log10(t_max), n_bins + 1)
                edges[0] = t_min
                edges[-1] = np.nextafter(t_max, np.inf)
                bin_indices = np.searchsorted(edges, time_arr[rest_idx], side="right") - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                binned_time: List[float] = []
                binned_mean: List[float] = []
                binned_stderr: List[float] = []
                for bin_index in np.unique(bin_indices):
                    idx = rest_idx[bin_indices == bin_index]
                    w = weight_arr[idx]
                    weight_sum = float(np.sum(w))
                    if weight_sum <= 0.0:
                        continue
                    binned_time.append(
                        float(np.exp(np.sum(w * np.log(time_arr[idx])) / weight_sum))
                    )
                    binned_mean.append(float(np.sum(w * mean_arr[idx]) / weight_sum))
                    binned_stderr.append(
                        float(np.sqrt(np.sum(w * stderr_arr[idx] ** 2) / weight_sum))
                    )
                if binned_time:
                    output_time.append(np.asarray(binned_time, dtype=np.float64))
                    output_mean.append(np.asarray(binned_mean, dtype=np.float64))
                    output_stderr.append(np.asarray(binned_stderr, dtype=np.float64))

    if not output_time:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty
    return (
        np.concatenate(output_time),
        np.concatenate(output_mean),
        np.concatenate(output_stderr),
    )


def truncate_at_plot_floor(
    time: np.ndarray,
    values: np.ndarray,
    min_plot_g: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the prefix before the first point where G drops below the plot floor."""
    n = min(time.size, values.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    time_arr = np.asarray(time[:n], dtype=np.float64)
    value_arr = np.asarray(values[:n], dtype=np.float64)
    keep = (
        np.isfinite(time_arr)
        & np.isfinite(value_arr)
        & (time_arr > 0.0)
        & (value_arr >= float(min_plot_g))
    )
    invalid_idx = np.flatnonzero(~keep)
    end = int(invalid_idx[0]) if invalid_idx.size > 0 else n
    return time_arr[:end], value_arr[:end]


def write_stress_modulus_by_epsilon_plot(
    path: Path,
    condition_values: List[ConditionKey],
    g_time_by_condition: Dict[ConditionKey, np.ndarray],
    g_mean_by_condition: Dict[ConditionKey, np.ndarray],
    g_stderr_by_condition: Dict[ConditionKey, np.ndarray],
    g_weight_by_condition: Dict[ConditionKey, np.ndarray],
    tau_r0: float,
    min_plot_g: float,
    use_log_binning: bool,
    plot_linear_lags: int,
    plot_bins_per_decade: float,
    colormap: str,
    x_limits: Tuple[float, float] | None = None,
) -> None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    series = []
    for condition in condition_values:
        lag_time = g_time_by_condition.get(condition)
        mean = g_mean_by_condition.get(condition)
        stderr = g_stderr_by_condition.get(condition)
        if lag_time is None or mean is None or stderr is None:
            continue
        if len(lag_time) == 0 or mean.size == 0:
            continue
        if mean.ndim != 1 or mean.shape[0] != len(lag_time):
            continue
        if stderr.ndim != 1 or stderr.shape[0] != len(lag_time):
            continue
        if use_log_binning:
            lag_time, mean, stderr = log_bin_mean_and_stderr_for_plot(
                lag_time,
                mean,
                stderr,
                g_weight_by_condition.get(condition),
                linear_lags=plot_linear_lags,
                bins_per_decade=plot_bins_per_decade,
            )
        lag_time, mean = truncate_at_plot_floor(
            lag_time,
            mean,
            min_plot_g=float(min_plot_g),
        )
        if lag_time.size == 0 or mean.size == 0:
            continue
        series.append((condition, lag_time, mean))

    if not series:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(colormap, len(series))
    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    plot_conditions = [condition for condition, _, _ in series]
    plotted_any = False
    for idx, (condition, lag_time, mean) in enumerate(series):
        color = cmap(idx)
        x = lag_time / tau_r0
        y = mean
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y >= min_plot_g)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        label = format_condition_label(condition, plot_conditions)
        split_points = np.flatnonzero(np.diff(valid_idx) > 1) + 1
        for segment_idx in np.split(valid_idx, split_points):
            if segment_idx.size == 0:
                continue
            ax.plot(
                x[segment_idx],
                y[segment_idx],
                color=color,
                lw=2.0,
                label=label,
            )
            label = None
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=10)
    ax.set_ylabel("G", fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=8)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=1000)
    plt.close(fig)


def write_all_replicate_stress_modulus_plot(
    path: Path,
    results: List[ReplicateResult],
    condition_values: List[ConditionKey],
    tau_r0: float,
    min_plot_g: float,
    use_log_binning: bool,
    plot_linear_lags: int,
    plot_bins_per_decade: float,
    colormap: str,
    x_limits: Tuple[float, float] | None = None,
) -> None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    sorted_results = sorted(
        results,
        key=lambda item: (
            item.epsilon,
            item.weakening_exponent,
            item.replicate,
        ),
    )
    if not sorted_results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(colormap, max(1, len(condition_values)))
    color_by_condition = {
        condition: cmap(idx)
        for idx, condition in enumerate(condition_values)
    }

    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    labeled_conditions = set()
    plotted_any = False
    for result in sorted_results:
        condition = (result.epsilon, result.weakening_exponent)
        lag_time = result.time
        modulus = result.modulus
        if lag_time.size == 0 or modulus.size == 0:
            continue
        if modulus.ndim != 1 or modulus.shape[0] != len(lag_time):
            continue
        if use_log_binning:
            lag_time, modulus = log_bin_timeseries_for_plot(
                lag_time,
                modulus,
                result.counts,
                linear_lags=plot_linear_lags,
                bins_per_decade=plot_bins_per_decade,
            )
        lag_time, modulus = truncate_at_plot_floor(
            lag_time,
            modulus,
            min_plot_g=float(min_plot_g),
        )
        if lag_time.size == 0 or modulus.size == 0:
            continue

        x = lag_time / tau_r0
        y = modulus
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y >= min_plot_g)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        color = color_by_condition.get(condition, "#2b2b2b")
        label = None
        if condition not in labeled_conditions:
            label = format_condition_label(condition, condition_values)
            labeled_conditions.add(condition)
        split_points = np.flatnonzero(np.diff(valid_idx) > 1) + 1
        for segment_idx in np.split(valid_idx, split_points):
            if segment_idx.size == 0:
                continue
            ax.plot(
                x[segment_idx],
                y[segment_idx],
                color=color,
                lw=0.8,
                alpha=0.55,
                label=label,
            )
            label = None
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=10)
    ax.set_ylabel("G", fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=8)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=1000)
    plt.close(fig)


def write_summary(
    path: Path,
    results: List[ReplicateResult],
    aggregated: Dict[ConditionKey, AggregateResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "replicate_count": len(results),
        "conditions": [
            {
                "epsilon": float(epsilon),
                "weakening_exponent": float(weakening_exponent),
            }
            for epsilon, weakening_exponent in sorted(aggregated)
        ],
        "replicates": [
            {
                "epsilon": float(result.epsilon),
                "weakening_exponent": float(result.weakening_exponent),
                "replicate": int(result.replicate),
                "virial_path": result.virial_path,
                "sample_dt": float(result.sample_dt),
                "n_samples": int(result.n_samples),
                "n_segments": int(result.n_segments),
                "longest_segment_samples": int(result.longest_segment_samples),
                "max_lag": int(result.max_lag),
                "max_time": float(result.time[-1]) if result.time.size else None,
            }
            for result in sorted(
                results,
                key=lambda item: (
                    item.epsilon,
                    item.weakening_exponent,
                    item.replicate,
                ),
            )
        ],
        "aggregates": {
            f"eps_{epsilon:g}_p_{weakening_exponent:g}": {
                "epsilon": float(epsilon),
                "weakening_exponent": float(weakening_exponent),
                "n_time_points": int(result.time.size),
                "min_time": float(result.time[0]) if result.time.size else None,
                "max_time": float(result.time[-1]) if result.time.size else None,
                "max_replicates_per_lag": int(np.nanmax(result.n_replicates))
                if result.n_replicates.size
                else 0,
            }
            for (epsilon, weakening_exponent), result in sorted(aggregated.items())
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.stress_max_runtime_fraction <= 0.0:
        raise RuntimeError("--stress-max-runtime-fraction must be > 0.")
    if args.max_lag_time <= 0.0:
        raise RuntimeError("--max-lag-time must be > 0.")
    if args.max_replicates_per_epsilon < 0:
        raise RuntimeError("--max-replicates-per-epsilon must be >= 0.")
    if args.min_plot_g <= 0.0:
        raise RuntimeError("--min-plot-g must be > 0 for log-scale plotting.")
    if args.plot_linear_lags < 0:
        raise RuntimeError("--plot-linear-lags must be >= 0.")
    if args.plot_bins_per_decade <= 0.0:
        raise RuntimeError("--plot-bins-per-decade must be > 0.")
    if args.extension_stitch_time <= 0.0:
        raise RuntimeError("--extension-stitch-time must be > 0.")

    selected_conditions = parse_condition_filters(args.conditions)
    selected_weakening_exponents = args.weakening_exponents
    if selected_conditions is None and selected_weakening_exponents is None:
        selected_weakening_exponents = [DEFAULT_WEAKENING_EXPONENT]

    base_runs = discover_runs(
        input_root=args.input_root,
        selected_epsilons=args.epsilons,
        selected_weakening_exponents=selected_weakening_exponents,
        selected_conditions=selected_conditions,
        max_replicates_per_epsilon=args.max_replicates_per_epsilon,
    )
    if not base_runs:
        raise RuntimeError(f"No virial_tensor_log.gsd files found under {args.input_root}")

    discovered_conditions = {
        (run.epsilon, run.weakening_exponent)
        for run in base_runs
    }
    if selected_conditions is not None:
        missing_conditions = [
            condition
            for condition in sorted(selected_conditions)
            if not condition_is_selected(
                condition[0],
                condition[1],
                discovered_conditions,
            )
        ]
        if missing_conditions:
            formatted = ", ".join(
                f"eps={epsilon:g},p={weakening_exponent:g}"
                for epsilon, weakening_exponent in missing_conditions
            )
            raise RuntimeError(f"No virial logs found for selected condition(s): {formatted}")

    extension_runs: List[RunSpec] = []
    if not args.no_extension_input:
        if args.extension_input_root.exists():
            extension_runs = discover_runs(
                input_root=args.extension_input_root,
                selected_epsilons=args.epsilons,
                selected_weakening_exponents=selected_weakening_exponents,
                selected_conditions=selected_conditions,
                max_replicates_per_epsilon=args.max_replicates_per_epsilon,
            )
        else:
            log(f"No extension input root found at {args.extension_input_root}; using base curves only")

    n_jobs = resolve_n_jobs(args.n_jobs)
    log(
        f"Discovered {len(base_runs)} base virial logs across "
        f"{len(discovered_conditions)} conditions"
    )
    if extension_runs:
        extension_conditions = {
            (run.epsilon, run.weakening_exponent)
            for run in extension_runs
        }
        log(
            f"Discovered {len(extension_runs)} dense extension virial logs across "
            f"{len(extension_conditions)} conditions"
        )
    log(f"Computing full-FFT stress relaxation with n_jobs={n_jobs}")

    computed = Parallel(n_jobs=n_jobs, verbose=args.joblib_verbose)(
        delayed(compute_replicate)(
            run,
            float(args.stress_max_runtime_fraction),
            float(args.max_lag_time),
        )
        for run in base_runs
    )
    base_results = [result for result in computed if result is not None]
    if not base_results:
        raise RuntimeError("No usable stress-relaxation results were produced.")

    extension_results: List[ReplicateResult] = []
    if extension_runs:
        log("Computing full-FFT stress relaxation for dense extension logs")
        extension_computed = Parallel(n_jobs=n_jobs, verbose=args.joblib_verbose)(
            delayed(compute_replicate)(
                run,
                float(args.stress_max_runtime_fraction),
                float(args.max_lag_time),
            )
            for run in extension_runs
        )
        extension_results = [
            result for result in extension_computed if result is not None
        ]

    log("Aggregating replicate curves by condition")
    base_aggregated = aggregate_replicates(base_results)
    extension_aggregated = (
        aggregate_replicates(extension_results) if extension_results else {}
    )
    if extension_aggregated:
        log(
            "Stitching dense extension curves through "
            f"{float(args.extension_stitch_time) / float(args.tau_r0):.3g} tau_R^0"
        )
        aggregated = stitch_extension_aggregates(
            base_aggregated,
            extension_aggregated,
            float(args.extension_stitch_time),
        )
    else:
        aggregated = base_aggregated
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_values = sorted(aggregated)
    p_values = {condition[1] for condition in condition_values}
    include_p_in_output_dirs = len(p_values) > 1 or any(
        not math.isclose(p_value, DEFAULT_WEAKENING_EXPONENT, rel_tol=0.0, abs_tol=1.0e-12)
        for p_value in p_values
    )
    g_time_by_condition = {
        condition: result.time for condition, result in aggregated.items()
    }
    g_mean_by_condition = {
        condition: result.mean for condition, result in aggregated.items()
    }
    g_stderr_by_condition = {
        condition: result.stderr for condition, result in aggregated.items()
    }
    g_weight_by_condition = {
        condition: result.weight_sum for condition, result in aggregated.items()
    }
    x_limits = compute_shared_time_lag_xlim(
        g_time_by_condition,
        g_mean_by_condition,
        float(args.tau_r0),
    )

    for condition, result in aggregated.items():
        epsilon, weakening_exponent = condition
        eps_dir = args.output_dir / condition_dir_name(
            condition,
            include_p=include_p_in_output_dirs,
        )
        write_timeseries(eps_dir / args.csv_name, result.time, result.mean, result.stderr)
        write_count_diagnostics(eps_dir / "stress_modulus_counts.csv", result)
        if extension_aggregated:
            base_result = base_aggregated.get(condition)
            if base_result is not None:
                write_timeseries(
                    eps_dir / "stress_modulus_base.csv",
                    base_result.time,
                    base_result.mean,
                    base_result.stderr,
                )
                write_count_diagnostics(
                    eps_dir / "stress_modulus_base_counts.csv",
                    base_result,
                )
            extension_result = extension_aggregated.get(condition)
            if extension_result is not None:
                write_timeseries(
                    eps_dir / "stress_modulus_extension.csv",
                    extension_result.time,
                    extension_result.mean,
                    extension_result.stderr,
                )
                write_count_diagnostics(
                    eps_dir / "stress_modulus_extension_counts.csv",
                    extension_result,
                )
        log(
            f"Aggregated eps={epsilon:g}, p={weakening_exponent:g}: "
            f"replicates={int(np.nanmax(result.n_replicates))}, "
            f"time_points={result.time.size}"
        )

    plot_path = args.output_dir / args.plot_name
    plot_results = extension_results + base_results if extension_results else base_results
    if args.plot_all_curves:
        write_all_replicate_stress_modulus_plot(
            plot_path,
            plot_results,
            condition_values,
            float(args.tau_r0),
            float(args.min_plot_g),
            not args.no_plot_log_binning,
            int(args.plot_linear_lags),
            float(args.plot_bins_per_decade),
            args.colormap,
            x_limits=x_limits,
        )
    else:
        write_stress_modulus_by_epsilon_plot(
            plot_path,
            condition_values,
            g_time_by_condition,
            g_mean_by_condition,
            g_stderr_by_condition,
            g_weight_by_condition,
            float(args.tau_r0),
            float(args.min_plot_g),
            not args.no_plot_log_binning,
            int(args.plot_linear_lags),
            float(args.plot_bins_per_decade),
            args.colormap,
            x_limits=x_limits,
        )
    write_summary(args.output_dir / "summary.json", plot_results, aggregated)
    log(f"Wrote full-FFT stress relaxation outputs to {args.output_dir}")
    log(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
