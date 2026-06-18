#!/usr/bin/env python3
"""Compute and plot monomer MSD curves for ReactiveLJ melt runs.

This standalone script extracts the MSD-only workflow from analyze_trajectories.py.
It discovers ``msd_trajectory.gsd`` files under an input root, computes the FFT
MSD for each replicate, aggregates replicates by ``(epsilon, p)``, and writes a
single log-log MSD plot.
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

import freud
import gsd.hoomd
import matplotlib
import numpy as np
from joblib import Parallel, delayed

matplotlib.use("Agg")
import ultraplot as uplt


FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 1000.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0
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
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
ConditionKey = Tuple[float, float]


@dataclass(frozen=True)
class RunSpec:
    msd_path: Path
    metadata_path: Path
    epsilon: float
    weakening_exponent: float
    replicate: int


@dataclass
class ReplicateResult:
    epsilon: float
    weakening_exponent: float
    replicate: int
    msd_path: str
    time: np.ndarray
    msd: np.ndarray
    counts: np.ndarray
    sample_dt: float
    n_frames: int
    n_particles: int


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
        description="Compute and plot monomer MSD by ReactiveLJ condition."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=melt_dir / "data_generation" / "outputs_clean",
        help="Root directory containing eps_*/rep_*/msd_trajectory.gsd files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "msd_plot",
        help="Directory for per-condition CSVs, diagnostics, and the SVG plot.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help=(
            "Number of replicate-level MSD workers. 0 uses SLURM_CPUS_PER_TASK "
            "when available, otherwise all visible CPUs."
        ),
    )
    parser.add_argument(
        "--analysis-stride",
        type=int,
        default=1,
        help="Stride over saved MSD frames during analysis.",
    )
    parser.add_argument(
        "--msd-max-lag-frames",
        type=int,
        default=0,
        help=(
            "Maximum lag in saved MSD frames; 0 uses the 1000 tau_R^0 physical "
            "cap, subject to available runtime."
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
        "--max-replicates-per-condition",
        type=int,
        default=0,
        help="Optional cap for quick checks. 0 processes all discovered replicates.",
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
            "minimum positive lag available in the plotted MSD curves."
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
        "--plot-name",
        default="monomer_msd_vs_time_lag_by_condition.svg",
        help="SVG plot filename written under --output-dir.",
    )
    parser.add_argument(
        "--csv-name",
        default="msd.csv",
        help="Per-condition aggregate CSV filename.",
    )
    parser.add_argument(
        "--joblib-verbose",
        type=int,
        default=10,
        help="Verbosity passed to joblib. Use 0 for quiet runs.",
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
        if match is not None:
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
    return any(
        math.isclose(value, selected_value, rel_tol=0.0, abs_tol=1.0e-12)
        for selected_value in selected
    )


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
        raise RuntimeError(f"Condition '{token}' must specify both epsilon and p.")
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
    max_replicates_per_condition: int,
) -> List[RunSpec]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ", "archived"}
    runs: List[RunSpec] = []
    for root, dirs, files in os.walk(input_root, topdown=True):
        dirs[:] = sorted(d for d in dirs if d not in excluded_dirs)
        if "msd_trajectory.gsd" not in files:
            continue

        run_dir = Path(root)
        metadata_path = run_dir / "metadata.json"
        metadata = load_metadata(metadata_path)
        msd_filename = str(metadata.get("msd_trajectory_file", "msd_trajectory.gsd"))
        msd_path = run_dir / msd_filename
        if not msd_path.exists():
            continue

        epsilon = metadata.get("reactive_epsilon", parse_prefixed_float(run_dir, "eps_"))
        weakening_exponent = metadata.get("weakening_exponent")
        if weakening_exponent is None:
            weakening_exponent = parse_prefixed_float(run_dir, "p_")
        if weakening_exponent is None:
            weakening_exponent = DEFAULT_WEAKENING_EXPONENT
        replicate = metadata.get("replicate", parse_replicate(run_dir))
        if epsilon is None:
            log(f"Skipping {msd_path}: could not infer epsilon")
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
                msd_path=msd_path,
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
            str(item.msd_path),
        )
    )
    if max_replicates_per_condition > 0:
        kept: List[RunSpec] = []
        counts: Dict[ConditionKey, int] = defaultdict(int)
        for run in runs:
            key = (run.epsilon, run.weakening_exponent)
            if counts[key] >= max_replicates_per_condition:
                continue
            kept.append(run)
            counts[key] += 1
        runs = kept
    return runs


def unwrap_positions_with_freud(frame) -> np.ndarray:
    positions = np.asarray(frame.particles.position, dtype=np.float64)
    images = getattr(frame.particles, "image", None)
    if images is None:
        return positions
    box = freud.box.Box.from_box(np.asarray(frame.configuration.box, dtype=np.float32))
    return np.asarray(
        box.unwrap(
            np.asarray(positions, dtype=np.float32),
            np.asarray(images, dtype=np.int32),
        ),
        dtype=np.float64,
    )


def load_msd_positions(
    msd_path: Path,
    analysis_stride: int,
    progress_label: str | None = None,
) -> np.ndarray | None:
    with gsd.hoomd.open(str(msd_path), "r") as traj:
        n_frames = len(traj)
        if n_frames < 2:
            return None

        n_analyzed = (n_frames + analysis_stride - 1) // analysis_stride
        progress_interval = max(1, n_analyzed // 10)
        positions: List[np.ndarray] = []
        for analyzed_idx, frame_idx in enumerate(
            range(0, n_frames, analysis_stride), start=1
        ):
            positions.append(
                unwrap_positions_with_freud(traj[frame_idx]).astype(
                    np.float32, copy=False
                )
            )
            if progress_label is not None and (
                analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
            ):
                progress_pct = 100.0 * analyzed_idx / n_analyzed
                log(
                    f"{progress_label}: frame progress {analyzed_idx}/{n_analyzed} "
                    f"({progress_pct:.1f}%)"
                )

    if len(positions) < 2:
        return None
    return np.stack(positions, axis=0)


def compute_msd_fft(
    positions: np.ndarray,
    sample_dt: float,
    max_lag_frames: int,
    runtime: float | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if positions.ndim != 3 or positions.shape[0] < 2:
        return None, None

    n_frames = int(positions.shape[0])
    if runtime is None or not np.isfinite(runtime) or runtime <= 0.0:
        runtime = float(n_frames - 1) * float(sample_dt)

    max_lag_time = min(runtime, MAX_ANALYSIS_LAG_TIME)
    runtime_lag = int(np.floor(max_lag_time / float(sample_dt) + 1.0e-12))
    runtime_lag = max(1, min(runtime_lag, n_frames - 1))
    requested_lag = (
        n_frames - 1
        if max_lag_frames <= 0
        else min(int(max_lag_frames), n_frames - 1)
    )
    max_lag = max(1, min(requested_lag, runtime_lag))

    pos = np.asarray(positions, dtype=np.float64)
    _, n_particles, n_dim = pos.shape
    coord = pos.reshape(n_frames, n_particles * n_dim)

    fft = np.fft.rfft(coord, n=2 * n_frames, axis=0)
    acf = np.fft.irfft(fft * np.conjugate(fft), n=2 * n_frames, axis=0)[
        :n_frames
    ].real
    counts = np.arange(n_frames, 0, -1, dtype=np.float64)[:, None]
    acf /= counts
    acf = acf.reshape(n_frames, n_particles, n_dim).sum(axis=2)

    r2 = np.sum(pos * pos, axis=2, dtype=np.float64)
    cumsum = np.vstack(
        [
            np.zeros((1, n_particles), dtype=np.float64),
            np.cumsum(r2, axis=0, dtype=np.float64),
        ]
    )
    lags = np.arange(n_frames, dtype=np.int64)
    s1 = (cumsum[n_frames - lags] + (cumsum[n_frames] - cumsum[lags])) / counts

    msd = np.mean(s1 - 2.0 * acf, axis=1)
    msd = np.maximum(msd[1 : max_lag + 1], 0.0)
    msd_time = np.arange(1, max_lag + 1, dtype=np.float64) * float(sample_dt)
    return msd_time, msd


def compute_replicate(
    run: RunSpec,
    analysis_stride: int,
    msd_max_lag_frames: int,
) -> ReplicateResult | None:
    metadata = load_metadata(run.metadata_path)
    positions = load_msd_positions(
        run.msd_path,
        analysis_stride,
        progress_label=f"eps={run.epsilon:g} p={run.weakening_exponent:g} rep={run.replicate}",
    )
    if positions is None:
        log(f"Skipping {run.msd_path}: fewer than two usable frames")
        return None

    dt = float(metadata.get("dt", 0.005))
    frame_steps = int(
        metadata.get("trajectory_frame_steps", metadata.get("frame_steps", 100_000))
    )
    msd_frame_steps = int(metadata.get("msd_frame_steps", frame_steps))
    sample_dt = dt * float(msd_frame_steps) * float(analysis_stride)
    runtime = sample_dt * float(max(positions.shape[0] - 1, 0))
    msd_time, msd = compute_msd_fft(
        positions,
        sample_dt,
        msd_max_lag_frames,
        runtime=runtime,
    )
    if msd_time is None or msd is None:
        log(f"Skipping {run.msd_path}: MSD calculation produced no data")
        return None

    n_frames = int(positions.shape[0])
    n_particles = int(positions.shape[1])
    counts = (
        np.arange(n_frames - 1, n_frames - 1 - len(msd), -1, dtype=np.float64)
        * float(n_particles)
    )
    return ReplicateResult(
        epsilon=run.epsilon,
        weakening_exponent=run.weakening_exponent,
        replicate=run.replicate,
        msd_path=str(run.msd_path),
        time=msd_time,
        msd=msd,
        counts=counts,
        sample_dt=sample_dt,
        n_frames=n_frames,
        n_particles=n_particles,
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
    return np.divide(
        np.sqrt(variance),
        np.sqrt(counts),
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 1,
    )


def aggregate_one_condition(
    condition: ConditionKey,
    results: List[ReplicateResult],
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
            values[row_idx, insert_idx[valid]] = result.msd[valid]
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


def aggregate_replicates(
    results: List[ReplicateResult],
) -> Dict[ConditionKey, AggregateResult]:
    by_condition: Dict[ConditionKey, List[ReplicateResult]] = defaultdict(list)
    for result in results:
        by_condition[(result.epsilon, result.weakening_exponent)].append(result)
    return {
        condition: aggregate_one_condition(condition, condition_results)
        for condition, condition_results in sorted(by_condition.items())
    }


def write_timeseries(
    path: Path,
    time: np.ndarray,
    mean: np.ndarray,
    stderr: np.ndarray,
) -> None:
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
    msd_time_by_condition: Dict[ConditionKey, np.ndarray],
    msd_mean_by_condition: Dict[ConditionKey, np.ndarray],
    tau_r0: float,
    plot_x_min_time: float | None = None,
    plot_x_max_tau_r0: float = MAX_ANALYSIS_LAG_TAU_R0,
) -> Tuple[float, float] | None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    positive_xmins: List[float] = []
    for condition, msd_time in msd_time_by_condition.items():
        msd_mean = msd_mean_by_condition.get(condition)
        if msd_mean is None:
            continue
        n = min(msd_time.size, msd_mean.size)
        if n == 0:
            continue
        x = msd_time[:n] / tau_r0
        y = msd_mean[:n]
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if np.any(mask):
            positive_xmins.append(float(np.min(x[mask])))

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


def format_epsilon_legend_value(epsilon: float) -> str:
    if math.isclose(epsilon, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return r"\mathrm{None}"
    return f"{epsilon:g}"


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
        return rf"$\varepsilon_\mathrm{{RLJ}}={format_epsilon_legend_value(epsilon)}$"
    return (
        rf"$\varepsilon_\mathrm{{RLJ}}={format_epsilon_legend_value(epsilon)}$, "
        rf"p={weakening_exponent:g}"
    )


def condition_dir_name(condition: ConditionKey, include_p: bool) -> str:
    epsilon, weakening_exponent = condition
    if include_p:
        return f"eps_{epsilon:g}_p_{weakening_exponent:g}"
    return f"eps_{epsilon:g}"


def write_msd_by_condition_plot(
    path: Path,
    condition_values: List[ConditionKey],
    msd_time_by_condition: Dict[ConditionKey, np.ndarray],
    msd_mean_by_condition: Dict[ConditionKey, np.ndarray],
    tau_r0: float,
    colormap: str,
    x_limits: Tuple[float, float] | None = None,
) -> None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    series = []
    for condition in condition_values:
        msd_time = msd_time_by_condition.get(condition)
        msd_mean = msd_mean_by_condition.get(condition)
        if msd_time is None or msd_mean is None:
            continue
        if len(msd_time) == 0 or msd_mean.size == 0:
            continue
        if msd_mean.ndim != 1 or msd_mean.shape[0] != len(msd_time):
            continue
        x = np.asarray(msd_time, dtype=np.float64) / tau_r0
        y = np.asarray(msd_mean, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if np.any(mask):
            series.append((condition, x[mask], y[mask]))

    if not series:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap(colormap)
    color_positions = np.linspace(0.0, 1.0, max(1, len(series)))
    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    plot_conditions = [condition for condition, _, _ in series]
    for idx, (condition, x, y) in enumerate(series):
        ax.plot(
            x,
            y,
            color=cmap(color_positions[idx]),
            lw=2.0,
            label=format_condition_label(condition, plot_conditions),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel("MSD", fontsize=DEFAULT_LABEL_FONTSIZE)
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
                "msd_path": result.msd_path,
                "sample_dt": float(result.sample_dt),
                "n_frames": int(result.n_frames),
                "n_particles": int(result.n_particles),
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
    if args.analysis_stride < 1:
        raise RuntimeError("--analysis-stride must be >= 1.")
    if args.msd_max_lag_frames < 0:
        raise RuntimeError("--msd-max-lag-frames must be >= 0.")
    if args.max_replicates_per_condition < 0:
        raise RuntimeError("--max-replicates-per-condition must be >= 0.")
    if args.plot_x_min_time is not None and args.plot_x_min_time <= 0.0:
        raise RuntimeError("--plot-x-min-time must be > 0 when provided.")
    if args.plot_x_max_tau_r0 <= 0.0:
        raise RuntimeError("--plot-x-max-tau-r0 must be > 0.")

    selected_conditions = parse_condition_filters(args.conditions)
    selected_weakening_exponents = args.weakening_exponents
    if selected_conditions is None and selected_weakening_exponents is None:
        selected_weakening_exponents = [DEFAULT_WEAKENING_EXPONENT]

    runs = discover_runs(
        input_root=args.input_root,
        selected_epsilons=args.epsilons,
        selected_weakening_exponents=selected_weakening_exponents,
        selected_conditions=selected_conditions,
        max_replicates_per_condition=args.max_replicates_per_condition,
    )
    if not runs:
        raise RuntimeError(f"No msd_trajectory.gsd files found under {args.input_root}")

    discovered_conditions = {
        (run.epsilon, run.weakening_exponent)
        for run in runs
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
            raise RuntimeError(f"No MSD trajectories found for selected condition(s): {formatted}")

    n_jobs = resolve_n_jobs(args.n_jobs)
    log(
        f"Discovered {len(runs)} MSD trajectories across "
        f"{len(discovered_conditions)} conditions"
    )
    log(f"Computing MSD curves with n_jobs={n_jobs}")

    computed = Parallel(n_jobs=n_jobs, verbose=args.joblib_verbose)(
        delayed(compute_replicate)(
            run,
            int(args.analysis_stride),
            int(args.msd_max_lag_frames),
        )
        for run in runs
    )
    results = [result for result in computed if result is not None]
    if not results:
        raise RuntimeError("No usable MSD results were produced.")

    log("Aggregating replicate curves by condition")
    aggregated = aggregate_replicates(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_values = sorted(aggregated)
    p_values = {condition[1] for condition in condition_values}
    include_p_in_output_dirs = len(p_values) > 1 or any(
        not math.isclose(
            p_value,
            DEFAULT_WEAKENING_EXPONENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for p_value in p_values
    )
    msd_time_by_condition = {
        condition: result.time for condition, result in aggregated.items()
    }
    msd_mean_by_condition = {
        condition: result.mean for condition, result in aggregated.items()
    }
    x_limits = compute_shared_time_lag_xlim(
        msd_time_by_condition,
        msd_mean_by_condition,
        float(args.tau_r0),
        plot_x_min_time=args.plot_x_min_time,
        plot_x_max_tau_r0=args.plot_x_max_tau_r0,
    )

    for condition, result in aggregated.items():
        epsilon, weakening_exponent = condition
        condition_dir = args.output_dir / condition_dir_name(
            condition,
            include_p=include_p_in_output_dirs,
        )
        write_timeseries(condition_dir / args.csv_name, result.time, result.mean, result.stderr)
        write_count_diagnostics(condition_dir / "msd_counts.csv", result)
        log(
            f"Aggregated eps={epsilon:g}, p={weakening_exponent:g}: "
            f"replicates={int(np.nanmax(result.n_replicates))}, "
            f"time_points={result.time.size}"
        )

    plot_path = args.output_dir / args.plot_name
    write_msd_by_condition_plot(
        plot_path,
        condition_values,
        msd_time_by_condition,
        msd_mean_by_condition,
        float(args.tau_r0),
        args.colormap,
        x_limits=x_limits,
    )
    write_summary(args.output_dir / "summary.json", results, aggregated)
    log(f"Wrote MSD outputs to {args.output_dir}")
    log(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
