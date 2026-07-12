#!/usr/bin/env python3
"""Compute full-FFT stress relaxation curves for ReactiveLJ melt runs.

This standalone script reads every ``virial_tensor_log.gsd`` under an input
root, computes the shear-stress autocorrelation from the off-diagonal tensor
components with an unbiased masked full-FFT estimator at every native lag,
combines dense extensions with the long base trajectories at matching lags,
aggregates replicates by ``(epsilon, p)``, and writes a direct-lag plot using
the same styling as the current stress-modulus analysis plot.
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

USER_TMP_DIR = Path("/tmp") / f"reactive_lj_plot_cache_{os.getuid()}"
USER_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(USER_TMP_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(USER_TMP_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import gsd.hoomd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import ultraplot as uplt


FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 1000.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0
DEFAULT_STRESS_MAX_RUNTIME_FRACTION = 1.0 / 3.0
DEFAULT_EXTENSION_STITCH_TIME = 25.0
PLOT_MIN_G = 1.0e-3
DEFAULT_WEAKENING_EXPONENT = 4.0
POINTS_PER_INCH = 72.0
FIGURE_WIDTH_PT = 237.6
FIGURE_HEIGHT_PT = 144.0
AXES_LEFT_PT = 35.369779
AXES_BOTTOM_PT = 27.66
AXES_WIDTH_PT = 197.730221
AXES_HEIGHT_PT = 108.9
DEFAULT_FIGSIZE = (
    FIGURE_WIDTH_PT / POINTS_PER_INCH,
    FIGURE_HEIGHT_PT / POINTS_PER_INCH,
)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 10
DEFAULT_LABEL_FONTSIZE = 10
STRESS_X_AXIS_LABEL = r"Lag time, $\tau / \tau_R^{(0)}$"
STRESS_Y_AXIS_LABEL = r"$G$"
ConditionKey = Tuple[float, float]
ReplicateKey = Tuple[float, float, int]

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
    regular_grid_samples: int
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
            "When present, these are combined with base estimates at matching "
            "short-time lags."
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
            "Legacy hard-crossover time in tau_LJ. Used only when explicit "
            "--extension-blend-start-time and --extension-blend-end-time are "
            "not provided."
        ),
    )
    parser.add_argument(
        "--extension-blend-start-time",
        type=float,
        default=None,
        help=(
            "Lag time in tau_LJ where the dense-extension weight begins a "
            "raised-cosine taper. Must be used with --extension-blend-end-time."
        ),
    )
    parser.add_argument(
        "--extension-blend-end-time",
        type=float,
        default=None,
        help=(
            "Lag time in tau_LJ where the dense-extension weight reaches zero "
            "and the curve becomes base-only."
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
        "--plot-x-min-time",
        type=float,
        default=None,
        help=(
            "Optional left x-axis limit in tau_LJ units. When omitted, use the "
            "minimum positive lag available in the plotted stress curves."
        ),
    )
    parser.add_argument(
        "--plot-x-max-tau-r0",
        type=float,
        default=MAX_ANALYSIS_LAG_TAU_R0,
        help=(
            "Right x-axis limit in tau/tau_R^0 units. Default keeps the current "
            f"{MAX_ANALYSIS_LAG_TAU_R0:g} tau_R^0 range."
        ),
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
        "--replicate-sem-points",
        type=int,
        default=20,
        help=(
            "Number of logarithmically spaced mean +/- replicate-SEM points "
            "per condition in the diagnostic plot."
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
    """Return full-grid unbiased shear-stress autocorrelation and counts."""
    return compute_segmented_shear_stress_autocovariance_fft(
        virial_arr,
        [(0, int(virial_arr.shape[0]))],
        max_lag,
    )


def compute_masked_shear_stress_autocovariance_fft(
    virial_arr: np.ndarray,
    sample_steps: np.ndarray,
    expected_step_delta: int,
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return autocorrelation on a regular grid while masking missing samples."""
    if virial_arr.ndim != 2 or virial_arr.shape[0] <= 1 or virial_arr.shape[1] < 6:
        raise ValueError("virial_arr must have shape (n_samples, >= 6)")
    if sample_steps.ndim != 1 or sample_steps.size != virial_arr.shape[0]:
        raise ValueError("sample_steps must be one-dimensional and match virial_arr")
    if expected_step_delta <= 0:
        raise ValueError("expected_step_delta must be positive")

    start_step = int(np.min(sample_steps))
    offsets = sample_steps.astype(np.int64, copy=False) - np.int64(start_step)
    misaligned = offsets % int(expected_step_delta) != 0
    if np.any(misaligned):
        first_bad = int(np.flatnonzero(misaligned)[0])
        raise ValueError(
            "Virial samples are not aligned to a single regular timestep grid: "
            f"sample step {int(sample_steps[first_bad])} is offset from start "
            f"{start_step} by a non-multiple of {expected_step_delta}."
        )

    grid_indices = (offsets // int(expected_step_delta)).astype(np.int64, copy=False)
    unique_indices, inverse = np.unique(grid_indices, return_inverse=True)
    if unique_indices.size < 2:
        raise ValueError("Need at least two unique sample times for autocorrelation")

    if unique_indices.size != grid_indices.size:
        coalesced = np.zeros(
            (unique_indices.size, virial_arr.shape[1]),
            dtype=np.float64,
        )
        duplicate_counts = np.zeros((unique_indices.size,), dtype=np.float64)
        np.add.at(coalesced, inverse, virial_arr)
        np.add.at(duplicate_counts, inverse, 1.0)
        observed_virial = coalesced / duplicate_counts[:, None]
        observed_indices = unique_indices
    else:
        observed_virial = virial_arr
        observed_indices = grid_indices

    grid_size = int(np.max(observed_indices)) + 1
    max_lag = max(1, min(int(max_lag), grid_size - 1))

    components = np.empty((observed_virial.shape[0], 3), dtype=np.float64)
    components[:, 0] = observed_virial[:, 1]
    components[:, 1] = observed_virial[:, 2]
    components[:, 2] = observed_virial[:, 4]
    # At isotropic equilibrium the ensemble mean of each shear component is zero.
    # Do not replace that known mean with a finite-record sample mean.

    component_grid = np.zeros((grid_size, 3), dtype=np.float64)
    mask = np.zeros((grid_size,), dtype=np.float64)
    component_grid[observed_indices, :] = components
    mask[observed_indices] = 1.0

    fft_length = 2 * grid_size
    component_spectrum = np.fft.rfft(component_grid, n=fft_length, axis=0)
    component_spectrum *= np.conjugate(component_spectrum)
    acf_sums = np.fft.irfft(component_spectrum, n=fft_length, axis=0)[
        : max_lag + 1
    ].real

    mask_spectrum = np.fft.rfft(mask, n=fft_length)
    mask_spectrum *= np.conjugate(mask_spectrum)
    counts = np.fft.irfft(mask_spectrum, n=fft_length)[: max_lag + 1].real
    counts = np.rint(np.maximum(counts, 0.0)).astype(np.float64, copy=False)

    acf_components = np.divide(
        acf_sums,
        counts[:, None],
        out=np.full_like(acf_sums, np.nan),
        where=counts[:, None] > 0.0,
    )
    return acf_components @ STRESS_COMPONENT_WEIGHTS, counts, grid_size


def compute_segmented_shear_stress_autocovariance_fft(
    virial_arr: np.ndarray,
    segments: List[Tuple[int, int]],
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return autocorrelation using only pairs inside contiguous segments."""
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
    # Preserve the zero-ensemble-mean Green-Kubo estimator used above.

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
    max_lag = max(1, max_lag)

    covariance, counts, regular_grid_samples = (
        compute_masked_shear_stress_autocovariance_fft(
            virial_arr,
            virial_steps,
            sample_step_delta,
            max_lag,
        )
    )
    box_length = resolve_box_length(metadata, run.virial_path.parent, run.virial_path)
    temperature = float(metadata.get("temperature", 1.0))
    modulus = (box_length**3 / temperature) * covariance
    actual_max_lag = covariance.shape[0] - 1
    lags = np.arange(1, actual_max_lag + 1, dtype=np.float64)

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
        regular_grid_samples=regular_grid_samples,
        max_lag=actual_max_lag,
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


def finite_column_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    value_sum = np.sum(np.where(finite, values, 0.0), axis=0)
    return np.divide(
        value_sum,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def align_replicate_results(
    results: List[ReplicateResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not results:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    base_idx = max(range(len(results)), key=lambda idx: results[idx].time.size)
    base_time = np.asarray(results[base_idx].time, dtype=np.float64)
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

    populated = np.any(
        np.isfinite(values) & np.isfinite(weights) & (weights > 0.0),
        axis=0,
    )
    return base_time[populated], values[:, populated], weights[:, populated]


def aggregate_one_condition(
    condition: ConditionKey, results: List[ReplicateResult]
) -> AggregateResult:
    epsilon, weakening_exponent = condition
    if not results:
        raise RuntimeError(
            f"No replicate results to aggregate for eps={epsilon:g}, p={weakening_exponent:g}"
        )

    time, values, weights = align_replicate_results(results)
    if time.size == 0:
        raise RuntimeError(
            f"No populated time points after aggregation for eps={epsilon:g}, p={weakening_exponent:g}"
        )

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


def replicate_key(result: ReplicateResult) -> ReplicateKey:
    return (
        float(result.epsilon),
        float(result.weakening_exponent),
        int(result.replicate),
    )


def extension_taper_weight(
    time: np.ndarray,
    blend_start_time: float,
    blend_end_time: float,
) -> np.ndarray:
    """Return a raised-cosine extension weight from one to zero."""
    time_arr = np.asarray(time, dtype=np.float64)
    if blend_start_time <= 0.0 or blend_end_time <= 0.0:
        raise RuntimeError("Extension blend times must be positive.")
    if blend_start_time > blend_end_time:
        raise RuntimeError("Extension blend start must not exceed blend end.")
    if math.isclose(
        blend_start_time,
        blend_end_time,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return (time_arr < float(blend_end_time)).astype(np.float64)

    weight = np.ones(time_arr.shape, dtype=np.float64)
    weight[time_arr >= float(blend_end_time)] = 0.0
    transition = (
        (time_arr > float(blend_start_time))
        & (time_arr < float(blend_end_time))
    )
    phase = (
        time_arr[transition] - float(blend_start_time)
    ) / (float(blend_end_time) - float(blend_start_time))
    weight[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return weight


def combine_replicate_pair(
    base: ReplicateResult,
    extension: ReplicateResult,
    blend_start_time: float,
    blend_end_time: float,
) -> ReplicateResult:
    """Blend base and extension estimators on the dense lag grid."""
    if replicate_key(base) != replicate_key(extension):
        raise RuntimeError(
            "Cannot combine mismatched replicate results: "
            f"base={replicate_key(base)}, extension={replicate_key(extension)}"
        )
    if extension.time.size == 0 or (
        float(extension.time[-1]) + float(extension.sample_dt)
        < float(blend_end_time)
    ):
        available_time = float(extension.time[-1]) if extension.time.size else 0.0
        raise RuntimeError(
            f"Extension result {replicate_key(extension)} reaches only "
            f"{available_time:g} tau_LJ, before blend end "
            f"{blend_end_time:g} tau_LJ. Increase "
            "--stress-max-runtime-fraction."
        )

    extension_mask = extension.time < float(blend_end_time)
    if not np.any(extension_mask):
        return base

    short_time = np.asarray(extension.time[extension_mask], dtype=np.float64).copy()
    short_modulus = np.asarray(
        extension.modulus[extension_mask],
        dtype=np.float64,
    ).copy()
    short_counts = np.asarray(
        extension.counts[extension_mask],
        dtype=np.float64,
    ).copy()

    base_valid = (
        np.isfinite(base.time)
        & np.isfinite(base.modulus)
        & np.isfinite(base.counts)
        & (base.counts > 0.0)
    )
    if np.count_nonzero(base_valid) < 2:
        raise RuntimeError(
            f"Replicate {replicate_key(base)} has fewer than two valid base lags."
        )
    base_time = np.asarray(base.time[base_valid], dtype=np.float64)
    base_modulus = np.asarray(base.modulus[base_valid], dtype=np.float64)
    base_counts = np.asarray(base.counts[base_valid], dtype=np.float64)

    base_available = (
        (short_time >= base_time[0])
        & (short_time <= base_time[-1])
    )
    interpolated_base_modulus = np.full(short_time.shape, np.nan, dtype=np.float64)
    interpolated_base_counts = np.zeros(short_time.shape, dtype=np.float64)
    interpolated_base_modulus[base_available] = np.interp(
        short_time[base_available],
        base_time,
        base_modulus,
    )
    interpolated_base_counts[base_available] = np.interp(
        short_time[base_available],
        base_time,
        base_counts,
    )

    extension_valid = (
        np.isfinite(short_modulus)
        & np.isfinite(short_counts)
        & (short_counts > 0.0)
    )
    taper = extension_taper_weight(
        short_time,
        blend_start_time,
        blend_end_time,
    )
    extension_weights = np.where(
        extension_valid,
        taper * short_counts,
        0.0,
    )
    base_weights = np.where(
        base_available,
        interpolated_base_counts,
        0.0,
    )
    combined_weights = extension_weights + base_weights
    weighted_extension = np.where(
        extension_valid,
        extension_weights * short_modulus,
        0.0,
    )
    weighted_base = np.where(
        base_available,
        base_weights * interpolated_base_modulus,
        0.0,
    )
    short_modulus = np.divide(
        weighted_extension + weighted_base,
        combined_weights,
        out=np.full(combined_weights.shape, np.nan, dtype=np.float64),
        where=combined_weights > 0.0,
    )
    short_counts = combined_weights

    base_long_mask = base.time >= float(blend_end_time)
    time = np.concatenate((short_time, base.time[base_long_mask]))
    modulus = np.concatenate((short_modulus, base.modulus[base_long_mask]))
    counts = np.concatenate((short_counts, base.counts[base_long_mask]))

    return ReplicateResult(
        epsilon=base.epsilon,
        weakening_exponent=base.weakening_exponent,
        replicate=base.replicate,
        virial_path=f"{extension.virial_path};{base.virial_path}",
        time=time,
        modulus=modulus,
        counts=counts,
        sample_dt=min(base.sample_dt, extension.sample_dt),
        n_samples=base.n_samples + extension.n_samples,
        n_segments=base.n_segments + extension.n_segments,
        longest_segment_samples=max(
            base.longest_segment_samples,
            extension.longest_segment_samples,
        ),
        regular_grid_samples=(
            base.regular_grid_samples + extension.regular_grid_samples
        ),
        max_lag=time.size,
    )


def combine_base_extension_replicates(
    base_results: List[ReplicateResult],
    extension_results: List[ReplicateResult],
    blend_start_time: float,
    blend_end_time: float,
) -> List[ReplicateResult]:
    """Combine matching estimators before computing replicate-level SEM."""
    extension_by_key: Dict[ReplicateKey, ReplicateResult] = {}
    for extension in extension_results:
        key = replicate_key(extension)
        if key in extension_by_key:
            raise RuntimeError(f"Duplicate extension result for {key}")
        extension_by_key[key] = extension

    combined: List[ReplicateResult] = []
    seen_base_keys: Set[ReplicateKey] = set()
    for base in base_results:
        key = replicate_key(base)
        if key in seen_base_keys:
            raise RuntimeError(f"Duplicate base result for {key}")
        seen_base_keys.add(key)
        extension = extension_by_key.get(key)
        if extension is None:
            combined.append(base)
        else:
            combined.append(
                combine_replicate_pair(
                    base,
                    extension,
                    blend_start_time,
                    blend_end_time,
                )
            )
    return combined


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
    plot_x_min_time: float | None = None,
    plot_x_max_tau_r0: float = MAX_ANALYSIS_LAG_TAU_R0,
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

    if plot_x_min_time is not None:
        x_min = float(plot_x_min_time) / tau_r0
        if not np.isfinite(x_min) or x_min <= 0.0:
            return None
    elif positive_xmins:
        x_min = min(positive_xmins)
    else:
        return None

    x_max = float(plot_x_max_tau_r0)
    if not np.isfinite(x_max) or x_max <= x_min:
        x_max = MAX_ANALYSIS_LAG_TAU_R0
    return x_min, x_max


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def format_condition_label(
    condition: ConditionKey,
    all_conditions: List[ConditionKey],
) -> str:
    epsilon, weakening_exponent = condition
    epsilon_values = {item[0] for item in all_conditions}
    weakening_values = {item[1] for item in all_conditions}
    if len(epsilon_values) == 1 and len(weakening_values) > 1:
        return f"p={weakening_exponent:g}"
    if math.isclose(epsilon, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        epsilon_label = "WCA"
    else:
        epsilon_label = rf"$\varepsilon_\mathrm{{RLJ}}={epsilon:g}\varepsilon_0$"
    if len(weakening_values) == 1:
        return epsilon_label
    return f"{epsilon_label}, p={weakening_exponent:g}"


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


def log_bin_replicates_on_common_grid(
    results: List[ReplicateResult],
    linear_lags: int,
    bins_per_decade: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Log-bin replicate curves using shared bin edges."""
    time, values, weights = align_replicate_results(results)
    if time.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    populated_idx = np.flatnonzero(np.any(valid, axis=0) & (time > 0.0))
    if populated_idx.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    linear_count = min(max(0, int(linear_lags)), populated_idx.size)
    exact_idx = populated_idx[:linear_count]
    rest_idx = populated_idx[linear_count:]
    output_time: List[np.ndarray] = []
    output_values: List[np.ndarray] = []

    if exact_idx.size:
        output_time.append(time[exact_idx])
        output_values.append(values[:, exact_idx])

    if rest_idx.size:
        if bins_per_decade <= 0.0:
            output_time.append(time[rest_idx])
            output_values.append(values[:, rest_idx])
        else:
            t_min = float(time[rest_idx[0]])
            t_max = float(np.max(time[rest_idx]))
            if t_max <= t_min:
                output_time.append(time[rest_idx])
                output_values.append(values[:, rest_idx])
            else:
                n_bins = max(
                    1,
                    int(
                        np.ceil(
                            (np.log10(t_max) - np.log10(t_min))
                            * bins_per_decade
                        )
                    ),
                )
                edges = np.logspace(
                    np.log10(t_min),
                    np.log10(t_max),
                    n_bins + 1,
                )
                edges[0] = t_min
                edges[-1] = np.nextafter(t_max, np.inf)
                bin_indices = (
                    np.searchsorted(edges, time[rest_idx], side="right") - 1
                )
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                binned_time: List[float] = []
                binned_values: List[np.ndarray] = []
                for bin_index in np.unique(bin_indices):
                    idx = rest_idx[bin_indices == bin_index]
                    bin_valid = valid[:, idx]
                    bin_weights = np.where(bin_valid, weights[:, idx], 0.0)
                    weight_sum = np.sum(bin_weights, axis=1)
                    weighted_values = np.sum(
                        np.where(bin_valid, bin_weights * values[:, idx], 0.0),
                        axis=1,
                    )
                    replicate_values = np.divide(
                        weighted_values,
                        weight_sum,
                        out=np.full(weight_sum.shape, np.nan, dtype=np.float64),
                        where=weight_sum > 0.0,
                    )
                    total_weights = np.sum(bin_weights, axis=0)
                    total_weight = float(np.sum(total_weights))
                    if total_weight <= 0.0:
                        continue
                    binned_time.append(
                        float(
                            np.exp(
                                np.sum(total_weights * np.log(time[idx]))
                                / total_weight
                            )
                        )
                    )
                    binned_values.append(replicate_values)

                if binned_time:
                    output_time.append(np.asarray(binned_time, dtype=np.float64))
                    output_values.append(
                        np.column_stack(binned_values).astype(
                            np.float64,
                            copy=False,
                        )
                    )

    if not output_time:
        return np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    return np.concatenate(output_time), np.concatenate(output_values, axis=1)


def select_log_spaced_indices(
    time: np.ndarray,
    valid: np.ndarray,
    max_points: int,
) -> np.ndarray:
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0 or max_points <= 0:
        return np.empty((0,), dtype=np.int64)
    if valid_idx.size <= max_points:
        return valid_idx

    positions = np.rint(
        np.linspace(0, valid_idx.size - 1, int(max_points))
    ).astype(np.int64)
    return valid_idx[positions]


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
    """Return the prefix before the first mean value below the plot floor."""
    n = min(time.size, values.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    time_arr = np.asarray(time[:n], dtype=np.float64)
    value_arr = np.asarray(values[:n], dtype=np.float64)
    valid = (
        np.isfinite(time_arr)
        & np.isfinite(value_arr)
        & (time_arr > 0.0)
        & (value_arr >= float(min_plot_g))
    )
    invalid_idx = np.flatnonzero(~valid)
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
    cmap = matplotlib.colormaps.get_cmap(colormap)
    color_positions = np.linspace(0.0, 1.0, max(1, len(series)))
    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    plot_conditions = [condition for condition, _, _ in series]
    plotted_any = False
    for idx, (condition, lag_time, mean) in enumerate(series):
        color = cmap(color_positions[idx])
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
        uplt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(STRESS_X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(STRESS_Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.format(
        xspineloc="both",
        yspineloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        labelsize=DEFAULT_TICK_FONTSIZE,
    )
    ax.legend(frameon=False, ncol=2, fontsize=DEFAULT_TICK_FONTSIZE)
    set_target_axes_position(ax)
    fig.savefig(path)
    uplt.close(fig)


def write_replicate_overlay_with_sem_plot(
    path: Path,
    results: List[ReplicateResult],
    condition_values: List[ConditionKey],
    tau_r0: float,
    min_plot_g: float,
    use_log_binning: bool,
    plot_linear_lags: int,
    plot_bins_per_decade: float,
    sem_points: int,
    colormap: str,
    x_limits: Tuple[float, float] | None = None,
) -> None:
    """Plot sparse binned replicate means with replicate SEM error bars."""
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    by_condition: Dict[ConditionKey, List[ReplicateResult]] = defaultdict(list)
    for result in results:
        by_condition[(result.epsilon, result.weakening_exponent)].append(result)
    if not by_condition:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap(colormap)
    color_positions = np.linspace(0.0, 1.0, max(1, len(condition_values)))
    color_by_condition = {
        condition: cmap(color_positions[idx])
        for idx, condition in enumerate(condition_values)
    }

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    plotted_conditions: List[ConditionKey] = []

    for condition in condition_values:
        condition_results = sorted(
            by_condition.get(condition, []),
            key=lambda result: result.replicate,
        )
        if not condition_results:
            continue

        if use_log_binning:
            lag_time, replicate_values = log_bin_replicates_on_common_grid(
                condition_results,
                linear_lags=plot_linear_lags,
                bins_per_decade=plot_bins_per_decade,
            )
        else:
            lag_time, replicate_values, _ = align_replicate_results(
                condition_results
            )
        if lag_time.size == 0 or replicate_values.size == 0:
            continue

        replicate_mean = finite_column_mean(replicate_values)
        replicate_stderr = finite_column_stderr(replicate_values)
        plotted_time, plotted_mean = truncate_at_plot_floor(
            lag_time,
            replicate_mean,
            min_plot_g=float(min_plot_g),
        )
        if plotted_time.size == 0:
            continue

        n_plot = plotted_time.size
        replicate_stderr = replicate_stderr[:n_plot]
        color = color_by_condition.get(condition, "#2b2b2b")
        x = plotted_time / tau_r0

        sem_valid = (
            np.isfinite(plotted_mean)
            & np.isfinite(replicate_stderr)
            & (plotted_mean >= float(min_plot_g))
            & ((plotted_mean - replicate_stderr) >= float(min_plot_g))
        )
        sem_idx = select_log_spaced_indices(
            plotted_time,
            sem_valid,
            max_points=int(sem_points),
        )
        if sem_idx.size:
            ax.errorbar(
                x[sem_idx],
                plotted_mean[sem_idx],
                yerr=replicate_stderr[sem_idx],
                fmt="o",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                markersize=3.0,
                elinewidth=0.9,
                capsize=2.0,
                capthick=0.9,
                alpha=0.95,
                label=format_condition_label(condition, condition_values),
                zorder=4,
            )
            plotted_conditions.append(condition)

    if not plotted_conditions:
        uplt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(STRESS_X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(STRESS_Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.format(
        xspineloc="both",
        yspineloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        labelsize=DEFAULT_TICK_FONTSIZE,
    )
    ax.legend(frameon=False, ncol=2, fontsize=DEFAULT_TICK_FONTSIZE)
    set_target_axes_position(ax)
    fig.savefig(path)
    uplt.close(fig)


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
    cmap = matplotlib.colormaps.get_cmap(colormap)
    color_positions = np.linspace(0.0, 1.0, max(1, len(condition_values)))
    color_by_condition = {
        condition: cmap(color_positions[idx])
        for idx, condition in enumerate(condition_values)
    }

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
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
        uplt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(STRESS_X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(STRESS_Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.format(
        xspineloc="both",
        yspineloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        labelsize=DEFAULT_TICK_FONTSIZE,
    )
    ax.legend(frameon=False, ncol=2, fontsize=DEFAULT_TICK_FONTSIZE)
    set_target_axes_position(ax)
    fig.savefig(path)
    uplt.close(fig)


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
                "regular_grid_samples": int(result.regular_grid_samples),
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
    if args.plot_x_min_time is not None and args.plot_x_min_time <= 0.0:
        raise RuntimeError("--plot-x-min-time must be > 0 when provided.")
    if args.plot_x_max_tau_r0 <= 0.0:
        raise RuntimeError("--plot-x-max-tau-r0 must be > 0.")
    if args.plot_linear_lags < 0:
        raise RuntimeError("--plot-linear-lags must be >= 0.")
    if args.plot_bins_per_decade <= 0.0:
        raise RuntimeError("--plot-bins-per-decade must be > 0.")
    if args.replicate_sem_points <= 0:
        raise RuntimeError("--replicate-sem-points must be > 0.")
    if args.extension_stitch_time <= 0.0:
        raise RuntimeError("--extension-stitch-time must be > 0.")
    blend_bounds_provided = (
        args.extension_blend_start_time is not None,
        args.extension_blend_end_time is not None,
    )
    if blend_bounds_provided[0] != blend_bounds_provided[1]:
        raise RuntimeError(
            "--extension-blend-start-time and --extension-blend-end-time "
            "must be provided together."
        )
    if all(blend_bounds_provided):
        blend_start_time = float(args.extension_blend_start_time)
        blend_end_time = float(args.extension_blend_end_time)
        if blend_start_time <= 0.0 or blend_end_time <= 0.0:
            raise RuntimeError("Extension blend times must be > 0.")
        if blend_start_time >= blend_end_time:
            raise RuntimeError(
                "--extension-blend-start-time must be less than "
                "--extension-blend-end-time."
            )
    else:
        blend_start_time = float(args.extension_stitch_time)
        blend_end_time = float(args.extension_stitch_time)

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

    log("Aggregating base and extension diagnostics by condition")
    base_aggregated = aggregate_replicates(base_results)
    extension_aggregated = (
        aggregate_replicates(extension_results) if extension_results else {}
    )
    if extension_results:
        if blend_start_time < blend_end_time:
            log(
                "Blending base and dense extension estimators from "
                f"{blend_start_time / float(args.tau_r0):.3g} to "
                f"{blend_end_time / float(args.tau_r0):.3g} tau_R^0"
            )
        else:
            log(
                "Combining base and dense extension estimators below "
                f"{blend_end_time / float(args.tau_r0):.3g} tau_R^0"
            )
        combined_results = combine_base_extension_replicates(
            base_results,
            extension_results,
            blend_start_time,
            blend_end_time,
        )
    else:
        combined_results = base_results
    log("Aggregating combined replicate curves by condition")
    aggregated = aggregate_replicates(combined_results)
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
        plot_x_min_time=args.plot_x_min_time,
        plot_x_max_tau_r0=args.plot_x_max_tau_r0,
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
    plot_results = combined_results
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
    replicate_plot_path = plot_path.with_name(
        f"{plot_path.stem}_replicates_with_sem{plot_path.suffix}"
    )
    write_replicate_overlay_with_sem_plot(
        replicate_plot_path,
        plot_results,
        condition_values,
        float(args.tau_r0),
        float(args.min_plot_g),
        not args.no_plot_log_binning,
        int(args.plot_linear_lags),
        float(args.plot_bins_per_decade),
        int(args.replicate_sem_points),
        args.colormap,
        x_limits=x_limits,
    )
    write_summary(args.output_dir / "summary.json", plot_results, aggregated)
    log(f"Wrote full-FFT stress relaxation outputs to {args.output_dir}")
    log(f"Wrote plot to {plot_path}")
    log(f"Wrote replicate diagnostic plot to {replicate_plot_path}")


if __name__ == "__main__":
    main()
