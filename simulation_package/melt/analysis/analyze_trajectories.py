"""Analyze ReactiveLJ trajectories (Block 2 metrics).

This script loops over GSD files produced by data_generation, computes the
analysis metrics described in agents.md, and averages results over replicates
for each ReactiveLJ attraction strength.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import freud
import gsd.hoomd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ultraplot as uplt

# Ensure local imports resolve when running from repo root.
sys.path.append(os.path.dirname(__file__))

FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 1000.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0
DEFAULT_STRESS_MAX_RUNTIME_FRACTION = 1.0 / 3.0
DEFAULT_WEAKENING_EXPONENT = 4.0
POINTS_PER_INCH = 72.0
STANDARD_FIGURE_WIDTH_PT = 237.6
STANDARD_FIGURE_HEIGHT_PT = 144.0
STANDARD_AXES_LEFT_PT = 35.369779
STANDARD_AXES_BOTTOM_PT = 27.66
STANDARD_AXES_WIDTH_PT = 197.730221
STANDARD_AXES_HEIGHT_PT = 108.9
BOND_TAU_EPSILON_PLOT = "sticky_bond_lifetime_vs_epsilon.svg"
MIN_RESOLVED_BOND_TAU_EPSILON = 12.0
LEGACY_BOND_TAU_PLOT_FILES = (
    "ln_bond_tau_vs_epsilon.png",
    "ln_bond_tau_vs_epsilon.svg",
    "bond_tau_vs_epsilon.png",
)
INTRA_INTER_RATIO_EPSILON_PLOT = "intra_to_inter_bond_ratio_vs_epsilon.svg"
INTRA_INTER_RATIO_LABEL = r"$\psi$"
INTRA_INTER_RATIO_FIGSIZE = (3.3, 3.3 / 2.0)
ANALYSIS_CHOICES = (
    "all",
    "msd",
    "stress_modulus",
    "bond_statistics",
    "cluster_distribution",
    "gelation_epsilon",
)

from analysis_utils import (
    MultiTauSetCorrelationAccumulator,
    UnionFind,
    autocorr_fft,
    compute_r_thresh,
    extract_semilog_linear_region,
    find_sticker_neighbor_pairs,
    fit_exponential,
    fit_exponential_semilog_linear_region,
)
from make_exchange_rate_plot import (
    DEFAULT_OUTPUT_NAME as EXCHANGE_RATE_EPSILON_PLOT,
    DUMP_INTERVAL_TAU_LJ as EXCHANGE_RATE_DUMP_INTERVAL_TAU_LJ,
    write_exchange_rate_plot,
)
from make_gelation_epsilon_plot import (
    DEFAULT_OUTPUT_NAME as GELATION_EPSILON_PLOT,
    summarize_replicate_points as summarize_gelation_epsilon_points,
    write_gelation_epsilon_plot,
)

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ReactiveLJ trajectories.")
    parser.add_argument(
        "--input-root",
        default="../data_generation/outputs_clean",
        help="Root directory containing eps_*/rep_*/trajectory.gsd sticker trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for analysis results.",
    )
    parser.add_argument(
        "--analysis-stride",
        type=int,
        default=1,
        help="Stride over saved GSD frames during analysis (1 = use every saved frame).",
    )
    parser.add_argument(
        "--msd-max-lag-frames",
        type=int,
        default=0,
        help=(
            "Maximum lag (in frames) used for MSD calculation; 0 uses the "
            "same 1000 tau_R^0 physical cap, subject to available runtime."
        ),
    )
    parser.add_argument(
        "--stress-max-runtime-fraction",
        type=float,
        default=DEFAULT_STRESS_MAX_RUNTIME_FRACTION,
        help=(
            "Maximum stress-modulus lag as a fraction of the virial-log runtime "
            "(default: 1/3)."
        ),
    )
    parser.add_argument(
        "--weakening-exponents",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Only analyze runs whose weakening exponent p matches these values. "
            "Runs without metadata or path p_* default to p=4."
        ),
    )
    parser.add_argument(
        "--plot-x-min-time",
        type=float,
        default=None,
        help=(
            "Optional left x-axis limit in tau_LJ for shared MSD/stress plots. "
            "When omitted, use the shortest resolved stress-modulus lag."
        ),
    )
    parser.add_argument(
        "--plot-x-max-tau-r0",
        type=float,
        default=MAX_ANALYSIS_LAG_TAU_R0,
        help=(
            "Right x-axis limit for shared MSD/stress plots in tau/tau_R^0 units."
        ),
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=ANALYSIS_CHOICES,
        default=["all"],
        help=(
            "Which analysis families to run. Use `all` for the full pipeline "
            "(default), or choose one or more of `msd`, `stress_modulus`, "
            "`bond_statistics`, `cluster_distribution`, and "
            "`gelation_epsilon` to rerun only those outputs."
        ),
    )
    parser.add_argument(
        "--bond-lifetime-plot-only",
        action="store_true",
        help=(
            "When rerunning `bond_statistics`, regenerate only the sticky-bond "
            "lifetime plot and supporting bond-correlation CSVs."
        ),
    )
    return parser.parse_args()


def resolve_analysis_selection(
    args: argparse.Namespace,
) -> Tuple[bool, bool, bool, bool, bool, bool]:
    requested: Set[str] = set(args.analyses)
    if "all" in requested:
        return True, True, True, True, True, True
    return (
        False,
        ("msd" in requested),
        ("stress_modulus" in requested),
        ("bond_statistics" in requested),
        ("cluster_distribution" in requested),
        ("gelation_epsilon" in requested),
    )


def gsd_frame_count_or_none(gsd_path: str, input_root: str) -> int | None:
    try:
        with gsd.hoomd.open(gsd_path, "r") as traj:
            return len(traj)
    except (RuntimeError, OSError) as exc:
        rel_path = os.path.relpath(gsd_path, input_root)
        log(f"Skipping corrupt GSD file {rel_path}: {exc}")
        return None


def is_corrupt_gsd_error(exc: BaseException) -> bool:
    return "Corrupt GSD file" in str(exc)


def parse_prefixed_float_from_path(path: str, prefix: str) -> float | None:
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if not part.startswith(prefix):
            continue
        try:
            return float(part[len(prefix) :])
        except ValueError:
            continue
    return None


def infer_weakening_exponent(run_dir: str, metadata: Dict) -> float:
    value = metadata.get("weakening_exponent")
    if value is None:
        value = parse_prefixed_float_from_path(run_dir, "p_")
    if value is None:
        value = DEFAULT_WEAKENING_EXPONENT
    return float(value)


def value_matches_selection(value: float, selected_values: List[float] | None) -> bool:
    if selected_values is None:
        return True
    return any(
        np.isclose(value, selected, rtol=0.0, atol=1.0e-12)
        for selected in selected_values
    )


def discover_runs(
    input_root: str,
    run_full_suite: bool,
    run_msd: bool,
    run_stress_modulus: bool,
    selected_weakening_exponents: List[float] | None,
) -> List[Tuple[str, str]]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ", "archived"}
    runs = []
    skipped_short = 0
    skipped_corrupt = 0
    skipped_weakening_exponent = 0
    for root, dirs, files in os.walk(input_root, topdown=True):
        # Keep test trajectories out of production analysis sweeps.
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        if "trajectory.gsd" in files:
            gsd_path = os.path.join(root, "trajectory.gsd")
            metadata_path = os.path.join(root, "metadata.json")
            metadata: Dict = {}
            required_frames = None
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                frame_steps = int(
                    metadata.get(
                        "trajectory_frame_steps",
                        metadata.get("frame_steps", 100_000),
                    )
                )
                sample_dt = float(metadata.get("dt", 0.005)) * float(frame_steps)
                required_frames = int(
                    np.floor(MAX_ANALYSIS_LAG_TIME / sample_dt + 1.0e-12)
                ) + 1
            weakening_exponent = infer_weakening_exponent(root, metadata)
            if not value_matches_selection(
                weakening_exponent,
                selected_weakening_exponents,
            ):
                skipped_weakening_exponent += 1
                continue

            n_frames = gsd_frame_count_or_none(gsd_path, input_root)
            if n_frames is None:
                skipped_corrupt += 1
                continue

            validation_paths = []
            if run_full_suite or run_msd:
                msd_path = os.path.join(
                    root,
                    str(metadata.get("msd_trajectory_file", "msd_trajectory.gsd")),
                )
                if os.path.exists(msd_path):
                    validation_paths.append(msd_path)
            if run_full_suite or run_stress_modulus:
                virial_path = os.path.join(root, "virial_tensor_log.gsd")
                if os.path.exists(virial_path):
                    validation_paths.append(virial_path)

            has_corrupt_required_input = False
            for validation_path in validation_paths:
                if gsd_frame_count_or_none(validation_path, input_root) is None:
                    skipped_corrupt += 1
                    has_corrupt_required_input = True
                    break
            if has_corrupt_required_input:
                continue

            if required_frames is not None and n_frames < required_frames:
                skipped_short += 1
                rel_path = os.path.relpath(gsd_path, input_root)
                log(
                    f"Skipping incomplete trajectory {rel_path}: "
                    f"{n_frames} frames < {required_frames}"
                )
                continue
            runs.append((gsd_path, metadata_path))
    if skipped_short > 0:
        log(
            f"Skipped {skipped_short} trajectories shorter than the "
            f"{MAX_ANALYSIS_LAG_TAU_R0:g} tau_R^0 lag requirement"
        )
    if skipped_corrupt > 0:
        log(f"Skipped {skipped_corrupt} runs with corrupt GSD input files")
    if skipped_weakening_exponent > 0:
        selected = ", ".join(f"{value:g}" for value in selected_weakening_exponents or [])
        log(
            f"Skipped {skipped_weakening_exponent} runs outside selected "
            f"weakening exponent(s): {selected}"
        )
    return runs


def fit_mean_timeseries_exponential(
    time: np.ndarray | None,
    values: np.ndarray | None,
    use_semilog_linear_region: bool = False,
) -> float:
    if time is None or values is None:
        return float("nan")
    time_arr = np.asarray(time, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    if time_arr.ndim != 1 or time_arr.size < 2 or value_arr.size == 0:
        return float("nan")
    if value_arr.ndim == 1:
        mean_series = value_arr
    elif value_arr.ndim == 2:
        mean_series = np.nanmean(value_arr, axis=0)
    else:
        return float("nan")
    mask = np.isfinite(time_arr) & np.isfinite(mean_series)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    if use_semilog_linear_region:
        return fit_exponential_semilog_linear_region(time_arr[mask], mean_series[mask])
    return fit_exponential(time_arr[mask], mean_series[mask])


def scalar_as_array(value: float) -> np.ndarray:
    if not np.isfinite(value):
        return np.array([], dtype=np.float64)
    return np.asarray([float(value)], dtype=np.float64)


def resolved_bond_tau_data(
    epsilon_values: List[float],
    tau_s_data: List[np.ndarray],
) -> Tuple[List[float], List[np.ndarray]]:
    resolved_eps: List[float] = []
    resolved_tau_s: List[np.ndarray] = []
    for epsilon, values in zip(epsilon_values, tau_s_data):
        epsilon_float = float(epsilon)
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if epsilon_float >= MIN_RESOLVED_BOND_TAU_EPSILON and arr.size:
            resolved_eps.append(epsilon_float)
            resolved_tau_s.append(arr)
    return resolved_eps, resolved_tau_s


def exchange_rate_points_from_summary_rows(
    summary_rows: List[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilons: List[float] = []
    turnover_assoc: List[float] = []
    turnover_dissoc: List[float] = []
    for row in summary_rows:
        epsilon = float(row["epsilon"])
        assoc = float(row["rate_assoc_mean"]) * EXCHANGE_RATE_DUMP_INTERVAL_TAU_LJ
        dissoc = float(row["rate_dissoc_mean"]) * EXCHANGE_RATE_DUMP_INTERVAL_TAU_LJ
        if np.isfinite(epsilon) and np.isfinite(assoc) and np.isfinite(dissoc):
            epsilons.append(epsilon)
            turnover_assoc.append(assoc)
            turnover_dissoc.append(dissoc)

    if not epsilons:
        raise ValueError("No finite exchange-rate rows found in summary data")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    assoc_arr = np.asarray(turnover_assoc, dtype=np.float64)[order]
    dissoc_arr = np.asarray(turnover_dissoc, dtype=np.float64)[order]
    return epsilon_arr, assoc_arr, dissoc_arr


def finite_column_stderr(values: np.ndarray) -> np.ndarray:
    stderr = np.zeros(values.shape[1], dtype=np.float64)
    for idx in range(values.shape[1]):
        finite = values[np.isfinite(values[:, idx]), idx]
        if finite.size > 1:
            stderr[idx] = float(np.std(finite, ddof=1) / np.sqrt(finite.size))
    return stderr


def unwrap_positions_with_freud(
    frame, particle_ids: np.ndarray | None = None
) -> np.ndarray:
    if particle_ids is None:
        positions = np.asarray(frame.particles.position, dtype=np.float64)
    else:
        positions = np.asarray(frame.particles.position[particle_ids], dtype=np.float64)
    images = getattr(frame.particles, "image", None)
    if images is None:
        return positions
    if particle_ids is None:
        images = np.asarray(images, dtype=np.int32)
    else:
        images = np.asarray(images[particle_ids], dtype=np.int32)
    box = freud.box.Box.from_box(np.asarray(frame.configuration.box, dtype=np.float32))
    return np.asarray(
        box.unwrap(
            np.asarray(positions, dtype=np.float32),
            images,
        ),
        dtype=np.float64,
    )


def find_virial_key(frame) -> str | None:
    if not hasattr(frame, "log"):
        return None
    for key in frame.log.keys():
        if "virial_tensor" in key:
            return key
    return None


def parse_virial_tensor_components(
    virial_val,
) -> Tuple[float, float, float, float, float, float] | None:
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


def load_virial_series_from_gsd(
    virial_gsd_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    virial_samples: List[Tuple[float, float, float, float, float, float]] = []
    virial_steps: List[int] = []
    with gsd.hoomd.open(virial_gsd_path, "r") as virial_traj:
        if len(virial_traj) == 0:
            return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
        virial_key = find_virial_key(virial_traj[0])
        if virial_key is None:
            return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
        for frame in virial_traj:
            if not hasattr(frame, "log"):
                continue
            virial_val = frame.log.get(virial_key, None)
            if virial_val is None:
                continue
            parsed = parse_virial_tensor_components(virial_val)
            if parsed is None:
                continue
            virial_samples.append(parsed)
            virial_steps.append(int(frame.configuration.step))
    if not virial_samples:
        return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.asarray(virial_samples, dtype=np.float64), np.asarray(
        virial_steps, dtype=np.int64
    )


def infer_sample_dt(
    sample_steps: np.ndarray, dt: float, fallback_step_stride: float
) -> float:
    if sample_steps.size >= 2:
        diffs = np.diff(sample_steps)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size > 0:
            return dt * float(np.median(positive_diffs))
    return dt * float(fallback_step_stride)


def multitau_autocovariance_with_counts(
    series: np.ndarray, p: int = 16, m: int = 2, S: int = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if p % m != 0:
        raise ValueError("p must be divisible by m")

    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be 1D")
    if x.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    p_m = p // m
    lags: List[float] = []
    cov: List[float] = []
    counts: List[float] = []

    level_data = x.copy()
    lag_scale = 1

    for level in range(S):
        n_level = level_data.size
        if n_level == 0:
            break

        j_start = 0 if level == 0 else p_m
        j_stop = min(p, n_level)
        for j in range(j_start, j_stop):
            span = n_level - j
            cov_ij = float(np.dot(level_data[:span], level_data[j:]) / span)
            lags.append(float(j * lag_scale))
            cov.append(cov_ij)
            counts.append(float(span))

        if level == S - 1:
            break

        n_next = n_level // m
        if n_next == 0:
            break
        trimmed = level_data[: n_next * m]
        level_data = np.mean(trimmed.reshape(n_next, m), axis=1)
        lag_scale *= m

    return (
        np.asarray(lags, dtype=np.float64),
        np.asarray(cov, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def compute_stress_autocovariance_multitau(
    tensor_arr: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if tensor_arr.ndim != 2 or tensor_arr.shape[0] <= 1 or tensor_arr.shape[1] < 6:
        return None, None, None

    xy = tensor_arr[:, 1]
    xz = tensor_arr[:, 2]
    yz = tensor_arr[:, 4]

    # Match the Liu/O'Connor Green-Kubo estimator: average only shear
    # components with alpha != beta (xy, xz, yz).
    weighted_series = ((1.0 / 3.0, (xy, xz, yz)),)

    g_lags = None
    g_cov = None
    g_counts = None
    for weight, series_group in weighted_series:
        for series in series_group:
            centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
            lags_i, cov_i, counts_i = multitau_autocovariance_with_counts(centered)
            if g_lags is None:
                g_lags = lags_i
                g_cov = np.zeros_like(cov_i, dtype=np.float64)
                g_counts = counts_i
            elif not np.array_equal(lags_i, g_lags):
                raise RuntimeError(
                    'Multi-tau lag grids do not match across stress components.'
                )
            g_cov += weight * cov_i
            if not np.array_equal(counts_i, g_counts):
                raise RuntimeError(
                    "Multi-tau sample counts do not match across stress components."
                )

    return g_lags, g_cov, g_counts


def compute_stress_modulus_from_virial(
    gsd_path: str,
    metadata: Dict,
    box_length: float,
    frame_steps: int,
    analysis_stride: int,
    stress_max_runtime_fraction: float,
) -> Tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
]:
    G_t = None
    G_autocorr_t = None
    G_time = None
    G_counts = None
    virial_source = None

    virial_arr = np.empty((0, 6), dtype=np.float64)
    virial_steps = np.empty((0,), dtype=np.int64)
    virial_log_path = os.path.join(os.path.dirname(gsd_path), "virial_tensor_log.gsd")
    if os.path.exists(virial_log_path):
        virial_arr, virial_steps = load_virial_series_from_gsd(virial_log_path)
        if virial_arr.size > 0:
            virial_source = "virial_tensor_log.gsd"

    if virial_arr.shape[0] <= 1:
        return G_time, G_t, G_autocorr_t, G_counts, virial_source

    g_lags, g_cov, g_counts = compute_stress_autocovariance_multitau(virial_arr)
    if g_lags is None or g_cov is None or g_counts is None:
        return G_time, G_t, G_autocorr_t, G_counts, virial_source

    cov0 = g_cov[0] if g_cov.size > 0 and g_cov[0] != 0.0 else np.nan
    G_autocorr_t = g_cov / cov0 if g_cov.size > 0 else np.empty((0,), dtype=np.float64)

    skip = 1 if len(g_lags) > 1 and g_lags[0] == 0 else 0
    g_lags = g_lags[skip:]
    g_cov = g_cov[skip:]
    G_autocorr_t = G_autocorr_t[skip:]
    g_counts = g_counts[skip:]

    dt_default_stride = (
        metadata.get("virial_log_steps", frame_steps)
        if virial_source == "virial_tensor_log.gsd"
        else frame_steps * analysis_stride
    )
    virial_dt = infer_sample_dt(
        virial_steps,
        float(metadata.get("dt", 0.005)),
        float(dt_default_stride),
    )
    g_time = g_lags * virial_dt

    if virial_steps.size >= 2:
        runtime = (
            float(np.max(virial_steps) - np.min(virial_steps))
            * float(metadata.get("dt", 0.005))
        )
    else:
        runtime = float(max(virial_arr.shape[0] - 1, 0)) * virial_dt

    max_g_time = min(stress_max_runtime_fraction * runtime, MAX_ANALYSIS_LAG_TIME)
    if np.isfinite(max_g_time) and max_g_time > 0.0:
        lag_mask = g_time <= max_g_time
        if np.any(lag_mask):
            g_time = g_time[lag_mask]
            g_cov = g_cov[lag_mask]
            G_autocorr_t = G_autocorr_t[lag_mask]
            g_counts = g_counts[lag_mask]

    volume = box_length**3
    kT = metadata.get("temperature", 1.0)
    G_t = (volume / kT) * g_cov
    G_time = g_time
    G_counts = g_counts
    return G_time, G_t, G_autocorr_t, G_counts, virial_source


def compute_msd_fft(
    positions: np.ndarray,
    sample_dt: float,
    max_lag_frames: int,
    runtime: float | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Compute MSD for all lags with an FFT-based autocorrelation formula."""
    if positions.ndim != 3 or positions.shape[0] < 2:
        return None, None

    n_frames = int(positions.shape[0])
    if runtime is None or not np.isfinite(runtime) or runtime <= 0.0:
        runtime = float(n_frames - 1) * float(sample_dt)

    max_lag_time = min(runtime, MAX_ANALYSIS_LAG_TIME)
    runtime_lag = int(np.floor(max_lag_time / float(sample_dt) + 1.0e-12))
    runtime_lag = max(1, min(runtime_lag, n_frames - 1))
    requested_lag = (n_frames - 1) if max_lag_frames <= 0 else min(int(max_lag_frames), n_frames - 1)
    max_lag = max(1, min(requested_lag, runtime_lag))

    pos = np.asarray(positions, dtype=np.float64)
    _, n_particles, n_dim = pos.shape
    coord = pos.reshape(n_frames, n_particles * n_dim)

    fft = np.fft.rfft(coord, n=2 * n_frames, axis=0)
    acf = np.fft.irfft(fft * np.conjugate(fft), n=2 * n_frames, axis=0)[:n_frames].real
    counts = np.arange(n_frames, 0, -1, dtype=np.float64)[:, None]
    acf /= counts
    acf = acf.reshape(n_frames, n_particles, n_dim).sum(axis=2)

    r2 = np.sum(pos * pos, axis=2, dtype=np.float64)
    cumsum = np.vstack(
        [np.zeros((1, n_particles), dtype=np.float64), np.cumsum(r2, axis=0, dtype=np.float64)]
    )
    lags = np.arange(n_frames, dtype=np.int64)
    s1 = (cumsum[n_frames - lags] + (cumsum[n_frames] - cumsum[lags])) / counts

    msd = np.mean(s1 - 2.0 * acf, axis=1)
    msd = np.maximum(msd[1 : max_lag + 1], 0.0)
    msd_time = np.arange(1, max_lag + 1, dtype=np.float64) * float(sample_dt)
    return msd_time, msd


def load_msd_positions(
    msd_gsd_path: str,
    analysis_stride: int,
    progress_label: str | None = None,
) -> np.ndarray | None:
    with gsd.hoomd.open(msd_gsd_path, "r") as traj:
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
                unwrap_positions_with_freud(traj[frame_idx]).astype(np.float32, copy=False)
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


def sticker_tags_from_metadata(metadata: Dict) -> np.ndarray:
    n_chains = int(metadata.get("n_chains", 0))
    chain_length = int(metadata.get("chain_length", 0))
    stickers_per_chain = int(metadata.get("stickers_per_chain", 0))
    if n_chains <= 0 or chain_length <= 0 or stickers_per_chain <= 0:
        return np.empty((0,), dtype=np.int32)

    segment = chain_length / stickers_per_chain
    offsets = np.rint((np.arange(stickers_per_chain) + 0.5) * segment).astype(np.int32) - 1
    offsets = np.clip(offsets, 0, chain_length - 1)
    offsets = np.unique(offsets)
    if offsets.size != stickers_per_chain:
        raise RuntimeError(
            "Could not reconstruct sticker tags from metadata; non-unique offsets detected."
        )

    chain_starts = np.arange(n_chains, dtype=np.int32) * chain_length
    return (chain_starts[:, None] + offsets[None, :]).reshape(-1)


def analyze_replicate(
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    msd_max_lag_frames: int,
    stress_max_runtime_fraction: float,
    run_full_suite: bool,
    run_msd: bool,
    run_stress_modulus: bool,
    run_bond_statistics: bool,
    run_cluster_distribution: bool,
    run_gelation_epsilon: bool,
    progress_label: str | None = None,
) -> Dict:
    with gsd.hoomd.open(gsd_path, "r") as traj:
        n_frames = len(traj)
        if n_frames == 0:
            raise RuntimeError(f"No frames found in {gsd_path}")
        n_analyzed = (n_frames + analysis_stride - 1) // analysis_stride
        progress_interval = max(1, n_analyzed // 10)

        first = traj[0]
        box_length = float(first.configuration.box[0])
        dt = float(metadata.get("dt", 0.005))
        frame_steps = int(
            metadata.get("trajectory_frame_steps", metadata.get("frame_steps", 100_000))
        )
        if not run_full_suite:
            result: Dict[str, np.ndarray | str | None] = {}
            if run_stress_modulus:
                G_time, G_t, G_autocorr_t, G_counts, virial_source = compute_stress_modulus_from_virial(
                    gsd_path,
                    metadata,
                    box_length,
                    frame_steps,
                    analysis_stride,
                    stress_max_runtime_fraction,
                )
                result.update(
                    {
                        "G_time": G_time,
                        "G_t": G_t,
                        "G_autocorr_t": G_autocorr_t,
                        "G_counts": G_counts,
                        "virial_source": virial_source,
                    }
                )

            if run_msd:
                msd_time = None
                msd = None
                msd_path = os.path.join(
                    os.path.dirname(gsd_path),
                    str(metadata.get("msd_trajectory_file", "msd_trajectory.gsd")),
                )
                if os.path.exists(msd_path):
                    msd_positions = load_msd_positions(
                        msd_path,
                        analysis_stride,
                        progress_label=(
                            f"{progress_label} [MSD]" if progress_label is not None else None
                        ),
                    )
                    if msd_positions is not None:
                        msd_frame_steps = int(metadata.get("msd_frame_steps", frame_steps))
                        msd_dt = dt * msd_frame_steps * analysis_stride
                        runtime = msd_dt * float(max(msd_positions.shape[0] - 1, 0))
                        msd_time, msd = compute_msd_fft(
                            msd_positions, msd_dt, msd_max_lag_frames, runtime=runtime
                        )
                msd_counts = None
                if msd is not None and msd_positions is not None:
                    n_frames = int(msd_positions.shape[0])
                    n_particles = int(msd_positions.shape[1])
                    msd_counts = (
                        np.arange(n_frames - 1, n_frames - 1 - len(msd), -1, dtype=np.float64)
                        * float(n_particles)
                    )
                result.update({"msd_time": msd_time, "msd": msd, "msd_counts": msd_counts})

            if run_bond_statistics or run_cluster_distribution or run_gelation_epsilon:
                chain_length = int(metadata.get("chain_length", 1))
                n_chains = int(
                    metadata.get("n_chains", first.particles.N // chain_length)
                )
                expected_sticker_tags = sticker_tags_from_metadata(metadata)
                if expected_sticker_tags.size != first.particles.N:
                    raise RuntimeError(
                        f"Sticker-only trajectory size mismatch in {gsd_path}: "
                        f"expected {expected_sticker_tags.size} particles, got "
                        f"{first.particles.N}."
                    )
                sticker_chain_ids = (expected_sticker_tags // chain_length).astype(
                    np.int32, copy=False
                )
                n_stickers = int(first.particles.N)
                stickers_per_chain = float(metadata.get("stickers_per_chain", 4))
                p_c = (
                    float(1.0 / (stickers_per_chain - 1.0))
                    if stickers_per_chain > 1.0
                    else float("nan")
                )
                frame_dt = dt * frame_steps * analysis_stride
                corr_lag_frames = max(
                    1, int(np.floor(MAX_ANALYSIS_LAG_TIME / frame_dt + 1.0e-12))
                )
                reactive_sigma = float(metadata.get("reactive_sigma", 1.0))
                r_thresh = float(compute_r_thresh(reactive_sigma))
                r_cut = metadata.get("reactive_r_cut")
                if r_cut is None:
                    r_cut = 1.5 * reactive_sigma
                weakening_outer = metadata.get("weakening_outer")
                if weakening_outer is None:
                    weakening_outer = 1.5 * reactive_sigma
                pair_cutoff = max(r_thresh, float(r_cut), float(weakening_outer))

                epsilon_series: List[float] = []
                cluster_hist = np.zeros(n_chains + 1, dtype=np.float64)
                cluster_count_total = 0
                cluster_frames = 0
                frac_bond0_series: List[float] = []
                frac_bond1_series: List[float] = []
                frac_bond_gt1_series: List[float] = []
                intra_bond_total = 0
                inter_bond_total = 0
                bond_frames = 0
                bond_corr = (
                    MultiTauSetCorrelationAccumulator(corr_lag_frames)
                    if run_bond_statistics
                    else None
                )

                for analyzed_idx, frame_idx in enumerate(
                    range(0, n_frames, analysis_stride), start=1
                ):
                    frame = traj[frame_idx]
                    positions = frame.particles.position
                    pair_i, pair_j, pair_dist = find_sticker_neighbor_pairs(
                        positions, box_length, pair_cutoff
                    )
                    uf = UnionFind(n_chains)
                    degrees = np.zeros(n_stickers, dtype=np.int32)
                    intra = 0
                    inter = 0
                    bond_ids = np.empty((0,), dtype=np.int64)

                    if pair_dist.size > 0:
                        bond_mask = pair_dist < r_thresh
                        bond_i = pair_i[bond_mask]
                        bond_j = pair_j[bond_mask]
                        if bond_i.size > 0:
                            if run_bond_statistics or run_gelation_epsilon:
                                np.add.at(degrees, bond_i, 1)
                                np.add.at(degrees, bond_j, 1)
                            chain_i = sticker_chain_ids[bond_i]
                            chain_j = sticker_chain_ids[bond_j]
                            inter_mask = chain_i != chain_j
                            if run_cluster_distribution and np.any(inter_mask):
                                chain_pairs = np.stack(
                                    (chain_i[inter_mask], chain_j[inter_mask]), axis=1
                                )
                                chain_pairs.sort(axis=1)
                                unique_chain_pairs = np.unique(chain_pairs, axis=0)
                                for chain_a, chain_b in unique_chain_pairs:
                                    uf.union(int(chain_a), int(chain_b))
                            if run_bond_statistics:
                                inter = int(np.count_nonzero(inter_mask))
                                intra = int(bond_i.size - inter)
                                low = np.minimum(bond_i, bond_j).astype(
                                    np.int64, copy=False
                                )
                                high = np.maximum(bond_i, bond_j).astype(
                                    np.int64, copy=False
                                )
                                bond_ids = low * np.int64(n_stickers) + high
                                bond_ids.sort()

                    if run_bond_statistics or run_gelation_epsilon:
                        open_count = int(np.count_nonzero(degrees == 0))

                    if run_gelation_epsilon:
                        p_open = (
                            float(open_count) / float(n_stickers)
                            if n_stickers > 0
                            else float("nan")
                        )
                        p = 1.0 - p_open
                        epsilon_series.append(
                            (p - p_c) / p_c if np.isfinite(p_c) else float("nan")
                        )

                    if run_bond_statistics:
                        count1 = int(np.count_nonzero(degrees == 1))
                        count_gt1 = n_stickers - open_count - count1
                        frac_bond0_series.append(open_count / n_stickers)
                        frac_bond1_series.append(count1 / n_stickers)
                        frac_bond_gt1_series.append(count_gt1 / n_stickers)
                        intra_bond_total += intra
                        inter_bond_total += inter
                        bond_frames += 1
                        bond_corr.update(bond_ids)

                    if run_cluster_distribution:
                        sizes = uf.cluster_sizes()
                        cluster_hist += np.bincount(
                            sizes, minlength=n_chains + 1
                        ).astype(np.float64, copy=False)
                        cluster_count_total += len(sizes)
                        cluster_frames += 1

                    if progress_label is not None and (
                        analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
                    ):
                        progress_pct = 100.0 * analyzed_idx / n_analyzed
                        selected_labels = []
                        if run_bond_statistics:
                            selected_labels.append("bond_statistics")
                        if run_cluster_distribution:
                            selected_labels.append("cluster_distribution")
                        if run_gelation_epsilon:
                            selected_labels.append("gelation_epsilon")
                        analysis_label = "+".join(selected_labels)
                        log(
                            f"{progress_label} [{analysis_label}]: frame progress "
                            f"{analyzed_idx}/{n_analyzed} ({progress_pct:.1f}%)"
                        )

                if run_bond_statistics and bond_corr is not None:
                    cs_valid_length = bond_corr.valid_length()
                    cs = bond_corr.correlation()[:cs_valid_length]
                    cs_time = (
                        bond_corr.lag_indices[:cs_valid_length].astype(np.float64)
                        * frame_dt
                    )
                    result.update(
                        {
                            "intra_bonds_mean": intra_bond_total / max(bond_frames, 1),
                            "inter_bonds_mean": inter_bond_total / max(bond_frames, 1),
                            "intra_inter_ratio": (
                                intra_bond_total / inter_bond_total
                                if inter_bond_total > 0
                                else float("nan")
                            ),
                            "cs_time": cs_time,
                            "cs": cs,
                            "frac_bond0_series": np.array(
                                frac_bond0_series, dtype=np.float64
                            ),
                            "frac_bond1_series": np.array(
                                frac_bond1_series, dtype=np.float64
                            ),
                            "frac_bond_gt1_series": np.array(
                                frac_bond_gt1_series, dtype=np.float64
                            ),
                        }
                    )

                if run_gelation_epsilon:
                    result["epsilon_mean"] = (
                        float(np.mean(epsilon_series)) if epsilon_series else float("nan")
                    )
                if run_cluster_distribution:
                    result.update(
                        {
                            "cluster_hist": cluster_hist,
                            "cluster_count_total": cluster_count_total,
                            "cluster_frames": cluster_frames,
                        }
                    )

            return result

        chain_length = int(metadata.get("chain_length", 1))
        n_chains = int(metadata.get("n_chains", first.particles.N // chain_length))
        expected_sticker_tags = sticker_tags_from_metadata(metadata)
        if expected_sticker_tags.size != first.particles.N:
            raise RuntimeError(
                f"Sticker-only trajectory size mismatch in {gsd_path}: "
                f"expected {expected_sticker_tags.size} particles, got {first.particles.N}."
            )
        sticker_chain_ids = (expected_sticker_tags // chain_length).astype(
            np.int32, copy=False
        )
        n_stickers = int(first.particles.N)

        r_thresh = float(compute_r_thresh(metadata.get("reactive_sigma", 1.0)))
        reactive_sigma = float(metadata.get("reactive_sigma", 1.0))
        r_cut = metadata.get("reactive_r_cut")
        if r_cut is None:
            r_cut = 1.5 * reactive_sigma
        weakening_inner = metadata.get("weakening_inner")
        if weakening_inner is None:
            weakening_inner = 1.3 * reactive_sigma
        weakening_outer = metadata.get("weakening_outer")
        if weakening_outer is None:
            weakening_outer = 1.5 * reactive_sigma
        r_cut = float(r_cut)
        weakening_inner = float(weakening_inner)
        weakening_outer = float(weakening_outer)
        pair_cutoff = max(r_thresh, r_cut, weakening_outer)

        frame_dt = dt * frame_steps * analysis_stride
        corr_lag_frames = max(
            1, int(np.floor(MAX_ANALYSIS_LAG_TIME / frame_dt + 1.0e-12))
        )
        reactive_epsilon = float(metadata.get("reactive_epsilon", float("nan")))
        stickers_per_chain = float(metadata.get("stickers_per_chain", 4))
        p_c = (
            float(1.0 / (stickers_per_chain - 1.0))
            if stickers_per_chain > 1.0
            else float("nan")
        )

        bond_corr = MultiTauSetCorrelationAccumulator(corr_lag_frames)
        open_corr = MultiTauSetCorrelationAccumulator(corr_lag_frames)

        p_open_series: List[float] = []
        p_series: List[float] = []
        epsilon_series: List[float] = []
        frac_bond0_series: List[float] = []
        frac_bond1_series: List[float] = []
        frac_bond_gt1_series: List[float] = []
        cexc_series: List[float] = []

        cluster_hist = np.zeros(n_chains + 1, dtype=np.float64)
        cluster_count_total = 0
        largest_cluster_fraction_sum = 0.0
        mean_cluster_size_sum = 0.0
        cluster_frames = 0

        intra_bond_total = 0
        inter_bond_total = 0
        bond_frames = 0

        assoc_events_total = 0
        dissoc_events_total = 0
        free_sticker_time = 0.0

        prev_bond_ids: np.ndarray | None = None
        prev_open_count: int | None = None
        prev_is_open: np.ndarray | None = None

        for analyzed_idx, frame_idx in enumerate(
            range(0, n_frames, analysis_stride), start=1
        ):
            frame = traj[frame_idx]
            positions = frame.particles.position

            pair_i, pair_j, pair_dist = find_sticker_neighbor_pairs(
                positions, box_length, pair_cutoff
            )
            degrees = np.zeros(n_stickers, dtype=np.int32)
            uf = UnionFind(n_chains)
            intra = 0
            inter = 0
            bond_ids = np.empty((0,), dtype=np.int64)

            if n_stickers > 1:
                cexc_series.append(
                    compute_cexc_mean(
                        n_stickers,
                        pair_i,
                        pair_j,
                        pair_dist,
                        r_cut,
                        weakening_inner,
                        weakening_outer,
                    )
                )

            if pair_dist.size > 0:
                bond_mask = pair_dist < r_thresh
                bond_i = pair_i[bond_mask]
                bond_j = pair_j[bond_mask]
                if bond_i.size > 0:
                    np.add.at(degrees, bond_i, 1)
                    np.add.at(degrees, bond_j, 1)

                    chain_i = sticker_chain_ids[bond_i]
                    chain_j = sticker_chain_ids[bond_j]
                    inter_mask = chain_i != chain_j
                    inter = int(np.count_nonzero(inter_mask))
                    intra = int(bond_i.size - inter)
                    if np.any(inter_mask):
                        chain_pairs = np.stack(
                            (chain_i[inter_mask], chain_j[inter_mask]), axis=1
                        )
                        chain_pairs.sort(axis=1)
                        unique_chain_pairs = np.unique(chain_pairs, axis=0)
                        for chain_a, chain_b in unique_chain_pairs:
                            uf.union(int(chain_a), int(chain_b))

                    low = np.minimum(bond_i, bond_j).astype(np.int64, copy=False)
                    high = np.maximum(bond_i, bond_j).astype(np.int64, copy=False)
                    bond_ids = low * np.int64(n_stickers) + high
                    bond_ids.sort()

            is_open = degrees == 0
            open_count = int(np.count_nonzero(is_open))

            p_open = (
                float(open_count) / float(n_stickers)
                if n_stickers > 0
                else float("nan")
            )
            p = 1.0 - p_open
            p_open_series.append(p_open)
            p_series.append(p)

            epsilon_val = (p - p_c) / p_c if np.isfinite(p_c) else float("nan")
            epsilon_series.append(epsilon_val)

            count1 = int(np.count_nonzero(degrees == 1))
            count_gt1 = n_stickers - open_count - count1
            frac_bond0_series.append(open_count / n_stickers)
            frac_bond1_series.append(count1 / n_stickers)
            frac_bond_gt1_series.append(count_gt1 / n_stickers)

            sizes = uf.cluster_sizes()
            cluster_hist += np.bincount(sizes, minlength=n_chains + 1).astype(
                np.float64, copy=False
            )
            cluster_count_total += len(sizes)
            largest_cluster_fraction_sum += float(np.max(sizes)) / n_chains
            mean_cluster_size_sum += float(np.mean(sizes))
            cluster_frames += 1

            intra_bond_total += intra
            inter_bond_total += inter
            bond_frames += 1

            bond_corr.update(bond_ids)
            open_corr.update(np.flatnonzero(is_open).astype(np.int64, copy=False))

            if (
                prev_bond_ids is not None
                and prev_open_count is not None
                and prev_is_open is not None
            ):
                new_bonds = np.setdiff1d(bond_ids, prev_bond_ids, assume_unique=True)
                assoc = 0
                dissoc = 0
                if new_bonds.size > 0:
                    new_i = new_bonds // n_stickers
                    new_j = new_bonds % n_stickers
                    assoc = int(
                        np.count_nonzero((~prev_is_open[new_i]) | (~prev_is_open[new_j]))
                    )
                    dissoc = int(new_bonds.size - assoc)
                assoc_events_total += assoc
                dissoc_events_total += dissoc
                free_sticker_time += prev_open_count * frame_dt

            prev_bond_ids = bond_ids
            prev_open_count = open_count
            prev_is_open = is_open.copy()

            if progress_label is not None and (
                analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
            ):
                progress_pct = 100.0 * analyzed_idx / n_analyzed
                log(
                    f"{progress_label}: frame progress {analyzed_idx}/{n_analyzed} "
                    f"({progress_pct:.1f}%)"
                )

        cs_valid_length = bond_corr.valid_length()
        cs = bond_corr.correlation()[:cs_valid_length]
        cb_valid_length = open_corr.valid_length()
        cb = open_corr.correlation()[:cb_valid_length]

        cs_time = bond_corr.lag_indices[:cs_valid_length].astype(np.float64) * frame_dt
        tb_time = open_corr.lag_indices[:cb_valid_length].astype(np.float64) * frame_dt
        cs_time, cs = truncate_lag_time(cs_time, cs, MAX_ANALYSIS_LAG_TIME)
        tb_time, cb = truncate_lag_time(tb_time, cb, MAX_ANALYSIS_LAG_TIME)

        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)
        tau_b = fit_exponential_semilog_linear_region(tb_time, cb)

        p_arr = np.array(p_series, dtype=np.float64)
        cp_full = autocorr_fft(p_arr, subtract_mean=True)
        cp = cp_full[1 : corr_lag_frames + 1]
        cp_time = np.arange(1, len(cp) + 1, dtype=np.float64) * frame_dt
        cp_time, cp = truncate_lag_time(cp_time, cp, MAX_ANALYSIS_LAG_TIME)
        tau_c = fit_exponential(cp_time, cp)

        G_time, G_t, G_autocorr_t, G_counts, virial_source = compute_stress_modulus_from_virial(
            gsd_path,
            metadata,
            box_length,
            frame_steps,
            analysis_stride,
            stress_max_runtime_fraction,
        )

        msd_time = None
        msd = None
        msd_counts = None
        msd_path = os.path.join(
            os.path.dirname(gsd_path),
            str(metadata.get("msd_trajectory_file", "msd_trajectory.gsd")),
        )
        if os.path.exists(msd_path):
            msd_positions = load_msd_positions(
                msd_path,
                analysis_stride,
                progress_label=(
                    f"{progress_label} [MSD]" if progress_label is not None else None
                ),
            )
            if msd_positions is not None:
                msd_frame_steps = int(metadata.get("msd_frame_steps", frame_steps))
                msd_dt = dt * msd_frame_steps * analysis_stride
                runtime = msd_dt * float(max(msd_positions.shape[0] - 1, 0))
                msd_time, msd = compute_msd_fft(
                    msd_positions, msd_dt, msd_max_lag_frames, runtime=runtime
                )
                if msd is not None:
                    n_frames = int(msd_positions.shape[0])
                    n_particles = int(msd_positions.shape[1])
                    msd_counts = (
                        np.arange(n_frames - 1, n_frames - 1 - len(msd), -1, dtype=np.float64)
                        * float(n_particles)
                    )

        result = {
            "p_open_mean": (
                float(np.mean(p_open_series)) if p_open_series else float("nan")
            ),
            "p_mean": float(np.mean(p_series)) if p_series else float("nan"),
            "epsilon_mean": (
                float(np.mean(epsilon_series)) if epsilon_series else float("nan")
            ),
            "intra_bonds_mean": intra_bond_total / max(bond_frames, 1),
            "inter_bonds_mean": inter_bond_total / max(bond_frames, 1),
            "intra_inter_ratio": (
                intra_bond_total / inter_bond_total
                if inter_bond_total > 0
                else float("nan")
            ),
            "mean_cluster_size": mean_cluster_size_sum / max(cluster_frames, 1),
            "largest_cluster_fraction": largest_cluster_fraction_sum
            / max(cluster_frames, 1),
            "rate_assoc": (
                float(assoc_events_total / free_sticker_time)
                if free_sticker_time > 0.0
                else float("nan")
            ),
            "rate_dissoc": (
                float(dissoc_events_total / free_sticker_time)
                if free_sticker_time > 0.0
                else float("nan")
            ),
            "tau_s": tau_s,
            "tau_b": tau_b,
            "tau_c": tau_c,
            "stickers_per_chain": stickers_per_chain,
            "p_c": p_c,
            "cluster_hist": cluster_hist,
            "cluster_count_total": cluster_count_total,
            "cluster_frames": cluster_frames,
            "cs_time": cs_time,
            "cs": cs,
            "tb_time": tb_time,
            "tb_corr": cb,
            "cp_time": cp_time,
            "cp": cp,
            "G_time": G_time,
            "G_t": G_t,
            "G_autocorr_t": G_autocorr_t,
            "G_counts": G_counts,
            "virial_source": virial_source,
            "msd_time": msd_time,
            "msd": msd,
            "msd_counts": msd_counts,
            "frac_bond0_series": np.array(frac_bond0_series, dtype=np.float64),
            "frac_bond1_series": np.array(frac_bond1_series, dtype=np.float64),
            "frac_bond_gt1_series": np.array(frac_bond_gt1_series, dtype=np.float64),
            "cexc_series": np.array(cexc_series, dtype=np.float64),
        }

        return result

def mean_and_stderr(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, stderr


def aggregate_replicate_timeseries(
    replicate_results: List[Dict],
    key_time: str,
    key_val: str,
    key_weight: str | None = None,
) -> Tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    series = []
    for res in replicate_results:
        if res.get(key_val) is None:
            continue
        weights = res.get(key_weight) if key_weight is not None else None
        series.append((res[key_time], res[key_val], weights))
    if not series:
        return None, None, None, None

    normalized = []
    for time_arr, value_arr, weight_arr in series:
        time_f = np.asarray(time_arr, dtype=np.float64)
        value_f = np.asarray(value_arr, dtype=np.float64)
        if weight_arr is None:
            weight_f = np.ones_like(value_f, dtype=np.float64)
        else:
            weight_f = np.asarray(weight_arr, dtype=np.float64)
        n = min(len(time_f), len(value_f), len(weight_f))
        if n > 0:
            normalized.append((time_f[:n], value_f[:n], weight_f[:n]))
    if not normalized:
        return None, None, None, None

    base_idx = max(range(len(normalized)), key=lambda idx: normalized[idx][0].size)
    base_time = normalized[base_idx][0]
    values = np.full((len(normalized), base_time.size), np.nan, dtype=np.float64)
    weights = np.zeros((len(normalized), base_time.size), dtype=np.float64)
    for row_idx, (time_arr, value_arr, weight_arr) in enumerate(normalized):
        insert_idx = np.searchsorted(base_time, time_arr)
        in_bounds = insert_idx < base_time.size
        valid = np.zeros(time_arr.size, dtype=bool)
        if np.any(in_bounds):
            matched = np.isclose(
                base_time[insert_idx[in_bounds]],
                time_arr[in_bounds],
                rtol=0.0,
                atol=1.0e-12,
            )
            valid[in_bounds] = matched
        if np.any(valid):
            values[row_idx, insert_idx[valid]] = value_arr[valid]
            weights[row_idx, insert_idx[valid]] = weight_arr[valid]

    populated = np.any(np.isfinite(values), axis=0)
    if not np.any(populated):
        return None, None, None, None

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
    stderr = finite_column_stderr(values)
    return time, values, mean, stderr


def aggregate_cluster_distribution(
    replicate_results: List[Dict],
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    normalized: List[Tuple[np.ndarray, float]] = []
    max_len = 0
    for res in replicate_results:
        hist = res.get("cluster_hist")
        total = res.get("cluster_count_total")
        if hist is None or total is None:
            continue
        hist_arr = np.asarray(hist, dtype=np.float64)
        total_float = float(total)
        if hist_arr.ndim != 1 or hist_arr.size == 0 or total_float <= 0.0:
            continue
        normalized.append((hist_arr / total_float, total_float))
        max_len = max(max_len, hist_arr.size)

    if not normalized or max_len == 0:
        return None, None, None

    values = np.full((len(normalized), max_len), np.nan, dtype=np.float64)
    weights = np.zeros((len(normalized), max_len), dtype=np.float64)
    for row_idx, (distribution, total) in enumerate(normalized):
        values[row_idx, : distribution.size] = distribution
        weights[row_idx, : distribution.size] = total

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    weight_sum = np.sum(np.where(valid, weights, 0.0), axis=0)
    weighted_value_sum = np.sum(np.where(valid, weights * values, 0.0), axis=0)
    mean = np.divide(
        weighted_value_sum,
        weight_sum,
        out=np.full((max_len,), np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    stderr = finite_column_stderr(values)
    cluster_size = np.arange(max_len, dtype=np.int64)
    return cluster_size, mean, stderr


def write_properties_csv(path: str, properties: Dict[str, Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("property,mean,stderr\n")
        for name, stats in properties.items():
            handle.write(f"{name},{stats['mean']},{stats['stderr']}\n")


def write_timeseries(
    path: str, time: np.ndarray, mean: np.ndarray, stderr: np.ndarray
) -> None:
    header = "time,mean,stderr\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for t, m, s in zip(time, mean, stderr):
            handle.write(f"{t:.6e},{m:.6e},{s:.6e}\n")


def write_autocorr_fit_plot(
    path: str,
    time: np.ndarray,
    values: np.ndarray,
    title: str,
    y_label: str,
    use_semilog_linear_region: bool = False,
) -> None:
    """Plot mean + IQR of replicate autocorrelations and an exponential fit."""
    finite_cols = np.any(np.isfinite(values), axis=0)
    if not np.any(finite_cols):
        return
    time = time[finite_cols]
    values = values[:, finite_cols]
    mean = np.nanmean(values, axis=0)
    q1 = np.nanpercentile(values, 25.0, axis=0)
    q3 = np.nanpercentile(values, 75.0, axis=0)

    if use_semilog_linear_region:
        tau_fit = fit_exponential_semilog_linear_region(time, mean)
        fit_time, _ = extract_semilog_linear_region(time, mean)
    else:
        tau_fit = fit_exponential(time, mean)
        fit_time = time

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, mean, color="#2b2b2b", lw=2.0, label="Mean")

    if np.isfinite(tau_fit) and fit_time.size > 0:
        fit_curve = np.exp(-fit_time / tau_fit)
        ax.plot(
            fit_time,
            fit_curve,
            color="#e77500",
            lw=2.0,
            label=f"Exponential fit (tau={tau_fit:.3g})",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compute_cexc_mean(
    n_stickers: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_dist: np.ndarray,
    r_cut: float,
    weakening_inner: float,
    weakening_outer: float,
    smooth_eps: float = 1e-6,
) -> float:
    if n_stickers < 2 or pair_i.size == 0:
        return float("nan")

    coordination = np.zeros(n_stickers, dtype=np.float64)
    w_ij = np.zeros_like(pair_dist)
    mask_inner = pair_dist <= weakening_inner
    mask_outer = (pair_dist > weakening_inner) & (pair_dist < weakening_outer)
    if np.any(mask_inner):
        w_ij[mask_inner] = 1.0
    if np.any(mask_outer):
        angle = np.pi * (pair_dist[mask_outer] - weakening_inner) / (
            weakening_outer - weakening_inner
        )
        w_ij[mask_outer] = 0.5 * (1.0 + np.cos(angle))

    if np.any(w_ij):
        np.add.at(coordination, pair_i, w_ij)
        np.add.at(coordination, pair_j, w_ij)

    mask_cut = pair_dist < r_cut
    if not np.any(mask_cut):
        return float("nan")

    i_cut = pair_i[mask_cut]
    j_cut = pair_j[mask_cut]
    w_cut = w_ij[mask_cut]
    raw = (coordination[i_cut] - w_cut) + (coordination[j_cut] - w_cut)
    cexc_vals = 0.5 * (raw + np.sqrt(raw * raw + smooth_eps * smooth_eps))
    return float(np.mean(cexc_vals))


def analyze_replicate_job(
    epsilon: float,
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    msd_max_lag_frames: int,
    stress_max_runtime_fraction: float,
    run_full_suite: bool,
    run_msd: bool,
    run_stress_modulus: bool,
    run_bond_statistics: bool,
    run_cluster_distribution: bool,
    run_gelation_epsilon: bool,
    rep_label: str,
    rel_path: str,
) -> Tuple[float, Dict | None]:
    log(f"{rep_label}: start ({rel_path})")
    try:
        result = analyze_replicate(
            gsd_path,
            metadata,
            analysis_stride,
            msd_max_lag_frames,
            stress_max_runtime_fraction,
            run_full_suite=run_full_suite,
            run_msd=run_msd,
            run_stress_modulus=run_stress_modulus,
            run_bond_statistics=run_bond_statistics,
            run_cluster_distribution=run_cluster_distribution,
            run_gelation_epsilon=run_gelation_epsilon,
            progress_label=rep_label,
        )
    except RuntimeError as exc:
        if is_corrupt_gsd_error(exc):
            log(
                f"{rep_label}: skipping corrupt GSD input while analyzing "
                f"{rel_path}: {exc}"
            )
            return epsilon, None
        raise
    log(f"{rep_label}: done")
    return epsilon, result

def truncate_lag_time(
    time: np.ndarray | None, values: np.ndarray | None, max_time: float
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if time is None or values is None:
        return None, None
    if len(time) == 0 or not np.isfinite(max_time) or max_time <= 0.0:
        return None, None
    mask = np.asarray(time, dtype=np.float64) <= float(max_time)
    if not np.any(mask):
        return None, None
    end = int(np.count_nonzero(mask))
    return time[:end], values[..., :end]


def write_fraction_violin_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str | None,
    y_label: str,
    figsize: Tuple[float, float] = (6.8, 4.4),
    dpi: int = 220,
    x_label: str = "epsilon",
    tick_label_size: float | None = None,
    axis_label_size: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, positions=epsilon_values, widths=0.6, showextrema=False)
    for body in parts.get("bodies", []):
        body.set_facecolor("#9e9e9e")
        body.set_edgecolor("#6f6f6f")
        body.set_alpha(0.5)

    median_positions: List[float] = []
    median_values: List[float] = []
    for eps, values in zip(epsilon_values, data):
        if values.size == 0:
            continue
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color="#2b2b2b", lw=2.0)
        ax.scatter([eps], [med], color="#2b2b2b", s=18, zorder=3)
        median_positions.append(float(eps))
        median_values.append(med)

    if median_positions:
        ax.plot(
            median_positions,
            median_values,
            color="#2b2b2b",
            lw=1.8,
            alpha=0.9,
            zorder=2,
        )

    if title:
        ax.set_title(title)
    if axis_label_size is None:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        ax.set_xlabel(x_label, fontsize=axis_label_size)
        ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    if tick_label_size is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_cexc_vs_epsilon_plot(
    path: str, epsilon_values: List[float], data: List[np.ndarray]
) -> None:
    medians = []
    q1s = []
    q3s = []
    for values in data:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            medians.append(float("nan"))
            q1s.append(float("nan"))
            q3s.append(float("nan"))
            continue
        medians.append(float(np.median(finite)))
        q1s.append(float(np.percentile(finite, 25.0)))
        q3s.append(float(np.percentile(finite, 75.0)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(epsilon_values, medians, color="#2b2b2b", marker="o", lw=2.0)
    ax.fill_between(epsilon_values, q1s, q3s, color="#9e9e9e", alpha=0.35)
    ax.set_title("Mean C_exc vs epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Mean C_exc (reactive pairs)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_cluster_distribution_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    cluster_distribution_by_eps: Dict[float, np.ndarray],
) -> None:
    def set_standard_axes_position(ax) -> None:
        ax.set_position(
            [
                STANDARD_AXES_LEFT_PT / STANDARD_FIGURE_WIDTH_PT,
                STANDARD_AXES_BOTTOM_PT / STANDARD_FIGURE_HEIGHT_PT,
                STANDARD_AXES_WIDTH_PT / STANDARD_FIGURE_WIDTH_PT,
                STANDARD_AXES_HEIGHT_PT / STANDARD_FIGURE_HEIGHT_PT,
            ]
        )

    def format_rlj_legend_label(epsilon: float) -> str:
        if np.isclose(float(epsilon), 0.0, rtol=0.0, atol=1.0e-12):
            return "WCA"
        return rf"$\varepsilon_\mathrm{{RLJ}}={float(epsilon):g}$"

    def align_terminal_log_xtick_labels(fig, ax) -> None:
        fig.canvas.draw()
        labels = [label for label in ax.get_xticklabels() if label.get_text()]
        if labels:
            labels[-1].set_ha("right")

    series = []
    for eps in epsilon_values:
        distribution = cluster_distribution_by_eps.get(eps)
        if distribution is None:
            continue
        cluster_size = np.arange(distribution.size, dtype=np.float64)
        prob = np.asarray(distribution, dtype=np.float64)
        mask = np.isfinite(cluster_size) & np.isfinite(prob) & (cluster_size > 0.0) & (
            prob > 0.0
        )
        if not np.any(mask):
            continue
        series.append((eps, cluster_size[mask], prob[mask]))

    if not series:
        return

    cmap = plt.get_cmap("plasma", len(series))
    fig, ax = uplt.subplots(
        figsize=(
            STANDARD_FIGURE_WIDTH_PT / POINTS_PER_INCH,
            STANDARD_FIGURE_HEIGHT_PT / POINTS_PER_INCH,
        ),
        dpi=600,
        tight=False,
    )
    set_standard_axes_position(ax)
    max_x = 1.0
    max_y = 1.0e-6
    min_y = np.inf
    for idx, (eps, cluster_size, prob) in enumerate(series):
        color = mcolors.to_hex(cmap(idx))
        ax.scatter(
            cluster_size,
            prob,
            s=8.0,
            color=color,
            label=format_rlj_legend_label(eps),
            edgecolors="black",
            linewidths=0.35,
        )
        max_x = max(max_x, float(np.max(cluster_size)))
        max_y = max(max_y, float(np.max(prob)))
        min_y = min(min_y, float(np.min(prob)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cluster size, $M$", fontsize=10)
    ax.set_ylabel(r"Probability, $P(M)$", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.tick_params(axis="both", which="both", labelsize=10)
    ax.set_xlim(left=1.0, right=max_x * 1.08)
    if np.isfinite(min_y) and min_y > 0.0:
        ax.set_ylim(bottom=min_y * 0.8, top=max_y * 1.2)
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.legend(frameon=False, fontsize=10, ncol=1)
    set_standard_axes_position(ax)
    fig.savefig(path)
    uplt.close(fig)


def write_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str | None,
    y_label: str,
    log_transform: bool = False,
    log_y: bool = False,
    figsize: Tuple[float, float] = (6.8, 4.4),
    dpi: int = 220,
    x_label: str = "epsilon",
    tick_label_size: float | None = None,
    axis_label_size: float | None = None,
) -> None:
    processed: List[Tuple[float, np.ndarray]] = []
    for eps, values in zip(epsilon_values, data):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if log_transform:
            arr = arr[arr > 0.0]
            if arr.size:
                arr = np.log(arr)
        elif log_y:
            arr = arr[arr > 0.0]
        if arr.size == 0:
            continue
        processed.append((float(eps), arr))

    if not processed:
        return

    fig, ax = plt.subplots(figsize=figsize)
    violin_positions = [eps for eps, values in processed if values.size > 1]
    violin_data = [values for _, values in processed if values.size > 1]
    if violin_data:
        parts = ax.violinplot(
            violin_data,
            positions=violin_positions,
            widths=0.6,
            showextrema=False,
        )
        for body in parts.get("bodies", []):
            body.set_facecolor("#9e9e9e")
            body.set_edgecolor("#6f6f6f")
            body.set_alpha(0.5)

    positions: List[float] = []
    medians: List[float] = []
    for eps, values in processed:
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color="#2b2b2b", lw=2.0)
        ax.scatter([eps], [med], color="#2b2b2b", s=18, zorder=3)
        positions.append(eps)
        medians.append(med)

    if positions:
        ax.plot(
            positions,
            medians,
            color="#2b2b2b",
            lw=1.8,
            alpha=0.9,
            zorder=2,
        )

    if title:
        ax.set_title(title)
    if axis_label_size is None:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        ax.set_xlabel(x_label, fontsize=axis_label_size)
        ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    if tick_label_size is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_dual_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data_left: List[np.ndarray],
    data_right: List[np.ndarray],
    title: str,
    y_label: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
) -> None:
    del left_label, right_label
    left_positions: List[float] = []
    left_violin_data: List[np.ndarray] = []
    left_medians: List[float] = []
    right_positions: List[float] = []
    right_violin_data: List[np.ndarray] = []
    right_medians: List[float] = []
    offset = 0.16
    width = 0.28

    for eps, left_vals, right_vals in zip(epsilon_values, data_left, data_right):
        left_arr = np.asarray(left_vals, dtype=np.float64)
        left_arr = left_arr[np.isfinite(left_arr)]
        right_arr = np.asarray(right_vals, dtype=np.float64)
        right_arr = right_arr[np.isfinite(right_arr)]

        if left_arr.size:
            left_positions.append(float(eps) - offset)
            left_violin_data.append(left_arr)
            left_medians.append(float(np.median(left_arr)))
        if right_arr.size:
            right_positions.append(float(eps) + offset)
            right_violin_data.append(right_arr)
            right_medians.append(float(np.median(right_arr)))

    if not left_violin_data and not right_violin_data:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    if left_violin_data:
        left_parts = ax.violinplot(
            left_violin_data, positions=left_positions, widths=width, showextrema=False
        )
        for body in left_parts.get("bodies", []):
            body.set_facecolor(left_color)
            body.set_edgecolor(left_color)
            body.set_alpha(0.35)
        for x, values in zip(left_positions, left_violin_data):
            q1 = float(np.percentile(values, 25.0))
            q3 = float(np.percentile(values, 75.0))
            med = float(np.median(values))
            ax.vlines(x, q1, q3, color=left_color, lw=2.0)
            ax.scatter([x], [med], color=left_color, s=18, zorder=3)
        ax.plot(left_positions, left_medians, color=left_color, lw=1.8, alpha=0.9, zorder=2)

    if right_violin_data:
        right_parts = ax.violinplot(
            right_violin_data, positions=right_positions, widths=width, showextrema=False
        )
        for body in right_parts.get("bodies", []):
            body.set_facecolor(right_color)
            body.set_edgecolor(right_color)
            body.set_alpha(0.35)
        for x, values in zip(right_positions, right_violin_data):
            q1 = float(np.percentile(values, 25.0))
            q3 = float(np.percentile(values, 75.0))
            med = float(np.median(values))
            ax.vlines(x, q1, q3, color=right_color, lw=2.0)
            ax.scatter([x], [med], color=right_color, s=18, zorder=3)
        ax.plot(
            right_positions,
            right_medians,
            color=right_color,
            lw=1.8,
            alpha=0.9,
            zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_log_tau_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    tau_data: List[np.ndarray],
    title: str | None = None,
    figsize: Tuple[float, float] = (6.8, 4.4),
    dpi: int = 220,
    x_label: str = r"$\varepsilon_\mathrm{reactiveLJ}$",
    tick_label_size: float | None = None,
    axis_label_size: float | None = None,
) -> None:
    write_scalar_violin_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        data=tau_data,
        title=title,
        y_label=r"$\tau_s$",
        log_transform=False,
        log_y=True,
        figsize=figsize,
        dpi=dpi,
        x_label=x_label,
        tick_label_size=tick_label_size,
        axis_label_size=axis_label_size,
    )


def write_tau_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    tau_data: List[np.ndarray],
    title: str = "Bond Correlation Decay Tau vs epsilon",
    y_label: str = "bond persistence time",
    ) -> None:
    write_scalar_violin_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        data=tau_data,
        title=title,
        y_label=y_label,
        log_transform=False,
    )


def remove_legacy_bond_tau_plots(output_dir: str) -> None:
    for filename in LEGACY_BOND_TAU_PLOT_FILES:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def compute_shared_time_lag_xlim(
    msd_time_by_eps: Dict[float, np.ndarray],
    msd_mean_by_eps: Dict[float, np.ndarray],
    g_time_by_eps: Dict[float, np.ndarray],
    g_mean_by_eps: Dict[float, np.ndarray],
    tau_r0: float,
    plot_x_min_time: float | None = None,
    plot_x_max_tau_r0: float = MAX_ANALYSIS_LAG_TAU_R0,
) -> Tuple[float, float] | None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    stress_xmins: List[float] = []
    fallback_xmins: List[float] = []

    for eps, msd_time in msd_time_by_eps.items():
        msd_mean = msd_mean_by_eps.get(eps)
        if msd_mean is None:
            continue
        time_arr = np.asarray(msd_time, dtype=np.float64)
        mean_arr = np.asarray(msd_mean, dtype=np.float64)
        n = min(time_arr.size, mean_arr.size)
        if n == 0:
            continue
        x = time_arr[:n] / tau_r0
        y = mean_arr[:n]
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if np.any(mask):
            fallback_xmins.append(float(np.min(x[mask])))

    for eps, lag_time in g_time_by_eps.items():
        g_mean = g_mean_by_eps.get(eps)
        if g_mean is None:
            continue
        time_arr = np.asarray(lag_time, dtype=np.float64)
        mean_arr = np.asarray(g_mean, dtype=np.float64)
        n = min(time_arr.size, mean_arr.size)
        if n == 0:
            continue
        x = time_arr[:n] / tau_r0
        finite_lag = np.isfinite(x) & (x > 0.0)
        if np.any(finite_lag):
            stress_xmins.append(float(np.min(x[finite_lag])))

    if plot_x_min_time is not None:
        x_min = float(plot_x_min_time) / tau_r0
        if not np.isfinite(x_min) or x_min <= 0.0:
            return None
    elif stress_xmins:
        x_min = min(stress_xmins)
    elif fallback_xmins:
        x_min = min(fallback_xmins)
    else:
        return None

    x_max = float(plot_x_max_tau_r0)
    if not np.isfinite(x_max) or x_max <= x_min:
        x_max = MAX_ANALYSIS_LAG_TAU_R0
    return x_min, x_max


def write_msd_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    msd_time_by_eps: Dict[float, np.ndarray],
    msd_mean_by_eps: Dict[float, np.ndarray],
    tau_r0: float,
    x_limits: Tuple[float, float] | None = None,
) -> None:
    def set_standard_axes_position(ax) -> None:
        ax.set_position(
            [
                STANDARD_AXES_LEFT_PT / STANDARD_FIGURE_WIDTH_PT,
                STANDARD_AXES_BOTTOM_PT / STANDARD_FIGURE_HEIGHT_PT,
                STANDARD_AXES_WIDTH_PT / STANDARD_FIGURE_WIDTH_PT,
                STANDARD_AXES_HEIGHT_PT / STANDARD_FIGURE_HEIGHT_PT,
            ]
        )

    def format_rlj_legend_label(epsilon: float) -> str:
        if np.isclose(float(epsilon), 0.0, rtol=0.0, atol=1.0e-12):
            return r"$\varepsilon_\mathrm{RLJ}=\mathrm{None}$"
        return rf"$\varepsilon_\mathrm{{RLJ}}={float(epsilon):g}$"

    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    series = []
    for eps in epsilon_values:
        msd_time = msd_time_by_eps.get(eps)
        msd_mean = msd_mean_by_eps.get(eps)
        if msd_time is None or msd_mean is None:
            continue
        if len(msd_time) == 0 or msd_mean.size == 0:
            continue
        if msd_mean.ndim != 1 or msd_mean.shape[0] != len(msd_time):
            continue
        x = np.asarray(msd_time, dtype=np.float64) / tau_r0
        y = np.asarray(msd_mean, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if not np.any(mask):
            continue
        series.append((eps, x[mask], y[mask]))

    if not series:
        return

    cmap = plt.get_cmap("plasma", len(series))
    fig, ax = uplt.subplots(
        figsize=(
            STANDARD_FIGURE_WIDTH_PT / POINTS_PER_INCH,
            STANDARD_FIGURE_HEIGHT_PT / POINTS_PER_INCH,
        ),
        dpi=1000,
        tight=False,
    )
    set_standard_axes_position(ax)
    for idx, (eps, x, y) in enumerate(series):
        ax.plot(x, y, color=cmap(idx), lw=2.0, label=format_rlj_legend_label(eps))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=10)
    ax.set_ylabel("MSD", fontsize=10)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.format(
        xspineloc="both",
        yspineloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=8)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    align_terminal_log_xtick_labels(fig, ax)
    set_standard_axes_position(ax)
    fig.savefig(path)
    uplt.close(fig)


def write_stress_modulus_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    g_time_by_eps: Dict[float, np.ndarray],
    g_mean_by_eps: Dict[float, np.ndarray],
    tau_r0: float,
    x_limits: Tuple[float, float] | None = None,
) -> None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    def align_terminal_log_xtick_labels(fig, ax) -> None:
        fig.canvas.draw()
        labels = [label for label in ax.get_xticklabels() if label.get_text()]
        if labels:
            labels[-1].set_ha("right")

    series = []
    for eps in epsilon_values:
        lag_time = g_time_by_eps.get(eps)
        mean = g_mean_by_eps.get(eps)
        if lag_time is None or mean is None:
            continue
        if len(lag_time) == 0 or mean.size == 0:
            continue
        if mean.ndim != 1 or mean.shape[0] != len(lag_time):
            continue
        series.append((eps, lag_time, mean))

    if not series:
        return

    cmap = plt.get_cmap("plasma", len(series))
    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    for idx, (eps, lag_time, mean) in enumerate(series):
        color = cmap(idx)
        stop_mask = (~np.isfinite(mean)) | (mean < 1.0e-3)
        stop_idx = np.flatnonzero(stop_mask)
        end_idx = int(stop_idx[0]) if stop_idx.size else int(mean.size)
        if end_idx <= 0:
            continue

        x = lag_time[:end_idx] / tau_r0
        y = mean[:end_idx]
        finite_positive = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if not np.any(finite_positive):
            continue

        ax.plot(
            x[finite_positive],
            y[finite_positive],
            color=color,
            lw=2.0,
            label=f"eps={eps:g}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=10)
    ax.set_ylabel("G", fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=8)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.legend(frameon=False, ncol=2, fontsize=8)
    align_terminal_log_xtick_labels(fig, ax)
    fig.savefig(path, dpi=1000)
    plt.close(fig)

def main() -> None:
    args = parse_args()
    if args.stress_max_runtime_fraction <= 0.0:
        raise RuntimeError(
            "--stress-max-runtime-fraction must be > 0, "
            f"got {args.stress_max_runtime_fraction}."
        )
    if args.plot_x_min_time is not None and args.plot_x_min_time <= 0.0:
        raise RuntimeError("--plot-x-min-time must be > 0 when provided.")
    if args.plot_x_max_tau_r0 <= 0.0:
        raise RuntimeError("--plot-x-max-tau-r0 must be > 0.")
    if args.weakening_exponents is not None:
        for value in args.weakening_exponents:
            if not np.isfinite(value):
                raise RuntimeError(
                    f"--weakening-exponents values must be finite, got {value}."
                )
    (
        run_full_suite,
        run_msd,
        run_stress_modulus,
        run_bond_statistics,
        run_cluster_distribution,
        run_gelation_epsilon,
    ) = resolve_analysis_selection(args)
    if args.bond_lifetime_plot_only and run_full_suite:
        raise RuntimeError(
            "--bond-lifetime-plot-only requires a selected-analysis rerun, "
            "for example `--analyses bond_statistics`."
        )
    if args.bond_lifetime_plot_only and not run_bond_statistics:
        raise RuntimeError(
            "--bond-lifetime-plot-only requires `--analyses bond_statistics`."
        )
    if run_full_suite:
        log("Requested analyses: all")
    else:
        selected = []
        if run_msd:
            selected.append("msd")
        if run_stress_modulus:
            selected.append("stress_modulus")
        if run_bond_statistics:
            selected.append("bond_statistics")
        if run_cluster_distribution:
            selected.append("cluster_distribution")
        if run_gelation_epsilon:
            selected.append("gelation_epsilon")
        log(f"Requested analyses: {', '.join(selected)}")
    if args.weakening_exponents is not None:
        selected_p = ", ".join(f"{value:g}" for value in args.weakening_exponents)
        log(f"Selected weakening exponent(s): {selected_p}")
    log(f"Scanning trajectories under {args.input_root}")
    runs = discover_runs(
        args.input_root,
        run_full_suite,
        run_msd,
        run_stress_modulus,
        args.weakening_exponents,
    )
    if not runs:
        raise RuntimeError(f"No trajectories found under {args.input_root}")
    log(f"Discovered {len(runs)} trajectory/metadata pairs")

    # Group runs by epsilon
    grouped: Dict[float, List[Tuple[str, Dict]]] = defaultdict(list)
    for gsd_path, metadata_path in runs:
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"Missing metadata.json for {gsd_path}")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        epsilon = float(metadata["reactive_epsilon"])
        grouped[epsilon].append((gsd_path, metadata))
    log(f"Grouped runs into {len(grouped)} epsilon values")

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    summary_json: Dict[str, Dict] = {}
    epsilon_values: List[float] = []
    frac_bond0_data: List[np.ndarray] = []
    frac_bond1_data: List[np.ndarray] = []
    frac_bond_gt1_data: List[np.ndarray] = []
    cexc_data: List[np.ndarray] = []
    tau_s_data: List[np.ndarray] = []
    tau_b_data: List[np.ndarray] = []
    cluster_distribution_by_eps: Dict[float, np.ndarray] = {}
    g_time_by_eps: Dict[float, np.ndarray] = {}
    g_mean_by_eps: Dict[float, np.ndarray] = {}
    msd_time_by_eps: Dict[float, np.ndarray] = {}
    msd_mean_by_eps: Dict[float, np.ndarray] = {}
    tau_r0_reference = FALLBACK_TAU_R0
    scalar_violin_data: Dict[str, List[np.ndarray]] = {
        "p_open_mean": [],
        "p_mean": [],
        "epsilon_mean": [],
        "intra_inter_ratio": [],
        "tau_c": [],
        "rate_assoc": [],
        "rate_dissoc": [],
        "stickers_per_chain": [],
        "p_c": [],
    }

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is None:
        raise RuntimeError(
            "SLURM_CPUS_PER_TASK is not set. Run under Slurm with --cpus-per-task "
            "or export SLURM_CPUS_PER_TASK."
        )
    try:
        n_jobs = int(slurm_cpus)
    except ValueError as exc:
        raise RuntimeError(
            f"SLURM_CPUS_PER_TASK must be an integer, got '{slurm_cpus}'."
        ) from exc
    if n_jobs < 1:
        raise RuntimeError(
            f"SLURM_CPUS_PER_TASK must be >= 1, got '{slurm_cpus}'."
        )

    sorted_groups = sorted(grouped.items())
    jobs = []
    for eps_idx, (epsilon, entries) in enumerate(sorted_groups, start=1):
        log(
            f"Queueing epsilon group {eps_idx}/{len(sorted_groups)}: "
            f"eps={epsilon:g}, replicates={len(entries)}"
        )
        for rep_idx, (gsd_path, metadata) in enumerate(entries, start=1):
            rel_path = os.path.relpath(gsd_path, args.input_root)
            rep_label = f"eps={epsilon:g} rep={rep_idx}/{len(entries)}"
            jobs.append(
                delayed(analyze_replicate_job)(
                    epsilon,
                    gsd_path,
                    metadata,
                    args.analysis_stride,
                    args.msd_max_lag_frames,
                    args.stress_max_runtime_fraction,
                    run_full_suite,
                    run_msd,
                    run_stress_modulus,
                    run_bond_statistics,
                    run_cluster_distribution,
                    run_gelation_epsilon,
                    rep_label,
                    rel_path,
                )
            )

    n_jobs = min(n_jobs, 49, len(jobs))
    log(f"Starting parallel analysis with {n_jobs} workers on {len(jobs)} runs")
    results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    log(f"Completed parallel analysis for {len(results)} runs")

    grouped_results: Dict[float, List[Dict]] = defaultdict(list)
    skipped_corrupt_runtime = 0
    for epsilon, result in results:
        if result is None:
            skipped_corrupt_runtime += 1
            continue
        grouped_results[epsilon].append(result)
    if skipped_corrupt_runtime > 0:
        log(
            f"Skipped {skipped_corrupt_runtime} runs that encountered corrupt "
            "GSD input during analysis"
        )
    if not grouped_results:
        raise RuntimeError("No valid trajectory analysis results remain after skips.")

    sorted_results = sorted(grouped_results.items())
    if not run_full_suite:
        msd_eps: List[float] = []
        stress_eps: List[float] = []
        bond_statistics_eps: List[float] = []
        intra_inter_ratio_data: List[np.ndarray] = []
        tau_s_data: List[np.ndarray] = []
        frac_bond0_data: List[np.ndarray] = []
        frac_bond1_data: List[np.ndarray] = []
        frac_bond_gt1_data: List[np.ndarray] = []
        cluster_eps: List[float] = []
        gelation_epsilon_eps: List[float] = []
        gelation_epsilon_data: List[np.ndarray] = []
        compact_eps_plot_kwargs = {
            "figsize": (3.3 / 2.0, 3.3 / 2.0),
            "dpi": 1000,
            "x_label": r"$\varepsilon$",
            "tick_label_size": 8,
            "axis_label_size": 10,
        }
        bond_tau_plot_kwargs = dict(compact_eps_plot_kwargs)
        bond_tau_plot_kwargs["x_label"] = r"$\varepsilon_\mathrm{reactiveLJ}$"
        intra_inter_ratio_plot_kwargs = dict(compact_eps_plot_kwargs)
        intra_inter_ratio_plot_kwargs["figsize"] = INTRA_INTER_RATIO_FIGSIZE
        for eps_idx, (epsilon, replicate_results) in enumerate(sorted_results, start=1):
            eps_dir = os.path.join(args.output_dir, f"eps_{epsilon:g}")
            os.makedirs(eps_dir, exist_ok=True)
            log(
                f"Aggregating selected analyses for epsilon group "
                f"{eps_idx}/{len(sorted_results)}: "
                f"eps={epsilon:g}, replicates={len(replicate_results)}"
            )

            if run_stress_modulus:
                G_time, G_values, G_mean, G_stderr = aggregate_replicate_timeseries(
                    replicate_results, "G_time", "G_t", key_weight="G_counts"
                )
                if (
                    G_time is not None
                    and G_values is not None
                    and G_mean is not None
                    and G_stderr is not None
                ):
                    stress_eps.append(epsilon)
                    g_time_by_eps[epsilon] = G_time
                    g_mean_by_eps[epsilon] = G_mean
                    write_timeseries(
                        os.path.join(eps_dir, "stress_modulus.csv"),
                        G_time,
                        G_mean,
                        G_stderr,
                    )

            if run_msd:
                msd_time, msd_values, msd_mean, msd_stderr = aggregate_replicate_timeseries(
                    replicate_results, "msd_time", "msd", key_weight="msd_counts"
                )
                if (
                    msd_time is not None
                    and msd_values is not None
                    and msd_mean is not None
                    and msd_stderr is not None
                ):
                    msd_eps.append(epsilon)
                    msd_time_by_eps[epsilon] = msd_time
                    msd_mean_by_eps[epsilon] = msd_mean
                    write_timeseries(
                        os.path.join(eps_dir, "msd.csv"),
                        msd_time,
                        msd_mean,
                        msd_stderr,
                    )

            if run_bond_statistics:
                cs_time, cs_values, cs_mean, cs_stderr = aggregate_replicate_timeseries(
                    replicate_results, "cs_time", "cs"
                )
                tau_s_fit = fit_mean_timeseries_exponential(
                    cs_time, cs_values, use_semilog_linear_region=True
                )
                if (
                    cs_time is not None
                    and cs_values is not None
                    and cs_mean is not None
                    and cs_stderr is not None
                ):
                    write_timeseries(
                        os.path.join(eps_dir, "bond_correlation.csv"),
                        cs_time,
                        cs_mean,
                        cs_stderr,
                    )

                bond_statistics_eps.append(epsilon)
                tau_s_data.append(scalar_as_array(tau_s_fit))
                if not args.bond_lifetime_plot_only:
                    ratio_values = np.asarray(
                        [
                            float(res.get("intra_inter_ratio", float("nan")))
                            for res in replicate_results
                        ],
                        dtype=np.float64,
                    )
                    frac_bond0_all = np.concatenate(
                        [
                            np.asarray(res["frac_bond0_series"], dtype=np.float64)
                            for res in replicate_results
                            if res.get("frac_bond0_series") is not None
                        ]
                    )
                    frac_bond1_all = np.concatenate(
                        [
                            np.asarray(res["frac_bond1_series"], dtype=np.float64)
                            for res in replicate_results
                            if res.get("frac_bond1_series") is not None
                        ]
                    )
                    frac_bond_gt1_all = np.concatenate(
                        [
                            np.asarray(res["frac_bond_gt1_series"], dtype=np.float64)
                            for res in replicate_results
                            if res.get("frac_bond_gt1_series") is not None
                        ]
                    )
                    intra_inter_ratio_data.append(ratio_values)
                    frac_bond0_data.append(frac_bond0_all)
                    frac_bond1_data.append(frac_bond1_all)
                    frac_bond_gt1_data.append(frac_bond_gt1_all)

            if run_cluster_distribution:
                cluster_size, cluster_mean, cluster_stderr = aggregate_cluster_distribution(
                    replicate_results
                )
                if (
                    cluster_size is not None
                    and cluster_mean is not None
                    and cluster_stderr is not None
                ):
                    cluster_eps.append(epsilon)
                    cluster_distribution_by_eps[epsilon] = cluster_mean
                    with open(
                        os.path.join(eps_dir, "cluster_distribution.csv"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write("cluster_size,mean,stderr\n")
                        for size, mean_val, stderr_val in zip(
                            cluster_size, cluster_mean, cluster_stderr
                        ):
                            handle.write(
                                f"{int(size)},{mean_val:.6e},{stderr_val:.6e}\n"
                            )

            if run_gelation_epsilon:
                epsilon_values_arr = np.asarray(
                    [
                        float(res.get("epsilon_mean", float("nan")))
                        for res in replicate_results
                    ],
                    dtype=np.float64,
                )
                epsilon_values_arr = epsilon_values_arr[np.isfinite(epsilon_values_arr)]
                if epsilon_values_arr.size > 0:
                    gelation_epsilon_eps.append(epsilon)
                    gelation_epsilon_data.append(epsilon_values_arr)

            log(f"Updated selected outputs for eps={epsilon:g} in {eps_dir}")

        if run_stress_modulus and stress_eps:
            shared_time_lag_xlim = compute_shared_time_lag_xlim(
                msd_time_by_eps,
                msd_mean_by_eps,
                g_time_by_eps,
                g_mean_by_eps,
                tau_r0_reference,
                plot_x_min_time=args.plot_x_min_time,
                plot_x_max_tau_r0=args.plot_x_max_tau_r0,
            )
            write_stress_modulus_by_epsilon_plot(
                os.path.join(args.output_dir, "stress_modulus_vs_time_lag_by_epsilon.svg"),
                stress_eps,
                g_time_by_eps,
                g_mean_by_eps,
                tau_r0_reference,
                x_limits=shared_time_lag_xlim,
            )
        if run_msd and msd_eps:
            shared_time_lag_xlim = compute_shared_time_lag_xlim(
                msd_time_by_eps,
                msd_mean_by_eps,
                g_time_by_eps,
                g_mean_by_eps,
                tau_r0_reference,
                plot_x_min_time=args.plot_x_min_time,
                plot_x_max_tau_r0=args.plot_x_max_tau_r0,
            )
            write_msd_vs_epsilon_plot(
                os.path.join(args.output_dir, "monomer_msd_vs_time_lag_by_epsilon.svg"),
                msd_eps,
                msd_time_by_eps,
                msd_mean_by_eps,
                tau_r0_reference,
                x_limits=shared_time_lag_xlim,
            )
        if run_cluster_distribution and cluster_eps:
            write_cluster_distribution_by_epsilon_plot(
                os.path.join(args.output_dir, "cluster_size_distribution_by_epsilon.svg"),
                cluster_eps,
                cluster_distribution_by_eps,
            )
        if run_gelation_epsilon and gelation_epsilon_eps:
            gelation_epsilon, gelation_mean, gelation_stderr = (
                summarize_gelation_epsilon_points(
                    gelation_epsilon_eps,
                    gelation_epsilon_data,
                )
            )
            write_gelation_epsilon_plot(
                os.path.join(args.output_dir, GELATION_EPSILON_PLOT),
                gelation_epsilon,
                gelation_mean,
                gelation_stderr,
            )
        if run_bond_statistics and bond_statistics_eps:
            remove_legacy_bond_tau_plots(args.output_dir)
            resolved_eps, resolved_tau_s = resolved_bond_tau_data(
                bond_statistics_eps,
                tau_s_data,
            )
            write_log_tau_vs_epsilon_plot(
                os.path.join(args.output_dir, BOND_TAU_EPSILON_PLOT),
                resolved_eps,
                resolved_tau_s,
                title=None,
                **bond_tau_plot_kwargs,
            )
            if not args.bond_lifetime_plot_only:
                write_fraction_violin_plot(
                    os.path.join(args.output_dir, "sticker_fraction_bond0_violin.png"),
                    bond_statistics_eps,
                    frac_bond0_data,
                    title=None,
                    y_label="Fraction of stickers",
                    **compact_eps_plot_kwargs,
                )
                write_fraction_violin_plot(
                    os.path.join(args.output_dir, "sticker_fraction_bond1_violin.png"),
                    bond_statistics_eps,
                    frac_bond1_data,
                    title=None,
                    y_label="Fraction of stickers",
                    **compact_eps_plot_kwargs,
                )
                write_fraction_violin_plot(
                    os.path.join(args.output_dir, "sticker_fraction_bond_gt1_violin.png"),
                    bond_statistics_eps,
                    frac_bond_gt1_data,
                    title=None,
                    y_label="Fraction of stickers",
                    **compact_eps_plot_kwargs,
                )
                write_scalar_violin_vs_epsilon_plot(
                    os.path.join(args.output_dir, INTRA_INTER_RATIO_EPSILON_PLOT),
                    bond_statistics_eps,
                    intra_inter_ratio_data,
                    title=None,
                    y_label=INTRA_INTER_RATIO_LABEL,
                    **intra_inter_ratio_plot_kwargs,
                )
        log("Selected-analysis rerun complete")
        return

    for eps_idx, (epsilon, replicate_results) in enumerate(sorted_results, start=1):
        log(
            f"Aggregating epsilon group {eps_idx}/{len(sorted_results)}: "
            f"eps={epsilon:g}, replicates={len(replicate_results)}"
        )
        epsilon_values.append(epsilon)

        # Aggregate scalar metrics
        scalar_keys = [
            "p_open_mean",
            "p_mean",
            "epsilon_mean",
            "intra_bonds_mean",
            "inter_bonds_mean",
            "intra_inter_ratio",
            "mean_cluster_size",
            "largest_cluster_fraction",
            "rate_assoc",
            "rate_dissoc",
        ]

        scalar_summary = {}
        for key in scalar_keys:
            vals = [res[key] for res in replicate_results]
            mean, stderr = mean_and_stderr(vals)
            scalar_summary[key] = {"mean": mean, "stderr": stderr}

        cluster_size, cluster_mean, cluster_stderr = aggregate_cluster_distribution(
            replicate_results
        )
        if (
            cluster_size is not None
            and cluster_mean is not None
            and cluster_stderr is not None
        ):
            cluster_distribution_by_eps[epsilon] = cluster_mean

        # Time series averages
        cs_time, cs_values, cs_mean, cs_stderr = aggregate_replicate_timeseries(
            replicate_results, "cs_time", "cs"
        )
        tb_time, tb_values, tb_mean, tb_stderr = aggregate_replicate_timeseries(
            replicate_results, "tb_time", "tb_corr"
        )
        cp_time, cp_values, cp_mean, cp_stderr = aggregate_replicate_timeseries(
            replicate_results, "cp_time", "cp"
        )
        G_time, G_values, G_mean, G_stderr = aggregate_replicate_timeseries(
            replicate_results, "G_time", "G_t", key_weight="G_counts"
        )
        msd_time, msd_values, msd_mean, msd_stderr = aggregate_replicate_timeseries(
            replicate_results, "msd_time", "msd", key_weight="msd_counts"
        )
        if G_time is not None and G_mean is not None:
            g_time_by_eps[epsilon] = G_time
            g_mean_by_eps[epsilon] = G_mean
        if msd_time is not None and msd_mean is not None:
            msd_time_by_eps[epsilon] = msd_time
            msd_mean_by_eps[epsilon] = msd_mean

        tau_s_fit = fit_mean_timeseries_exponential(
            cs_time, cs_values, use_semilog_linear_region=True
        )
        tau_b_fit = fit_mean_timeseries_exponential(
            tb_time, tb_values, use_semilog_linear_region=True
        )
        tau_c_fit = fit_mean_timeseries_exponential(cp_time, cp_values)
        scalar_summary["tau_s"] = {"mean": tau_s_fit, "stderr": 0.0}
        scalar_summary["tau_b"] = {"mean": tau_b_fit, "stderr": 0.0}
        scalar_summary["tau_c"] = {"mean": tau_c_fit, "stderr": 0.0}

        tau_s_data.append(scalar_as_array(tau_s_fit))
        tau_b_data.append(scalar_as_array(tau_b_fit))
        for key in scalar_violin_data:
            if key == "tau_c":
                scalar_violin_data[key].append(scalar_as_array(tau_c_fit))
                continue
            scalar_violin_data[key].append(
                np.asarray(
                    [float(res.get(key, float("nan"))) for res in replicate_results],
                    dtype=np.float64,
                )
            )

        f_mean, f_stderr = mean_and_stderr(
            [float(res.get("stickers_per_chain", float("nan"))) for res in replicate_results]
        )
        pc_mean, pc_stderr = mean_and_stderr(
            [float(res.get("p_c", float("nan"))) for res in replicate_results]
        )
        epsilon_properties = {
            "p_open": scalar_summary["p_open_mean"],
            "bonding_probability_p": scalar_summary["p_mean"],
            "gelation_epsilon": scalar_summary["epsilon_mean"],
            "stickers_per_chain_f": {"mean": f_mean, "stderr": f_stderr},
            "gel_point_p_c": {"mean": pc_mean, "stderr": pc_stderr},
            "intra_to_inter_bond_ratio": scalar_summary["intra_inter_ratio"],
            "bond_persistence_time_tau_s": scalar_summary["tau_s"],
            "brachiation_time_tau_b": scalar_summary["tau_b"],
            "fluctuation_relaxation_time_tau_c": scalar_summary["tau_c"],
            "associative_exchange_rate_R_a": scalar_summary["rate_assoc"],
            "passive_dimerization_rate_R_d": scalar_summary["rate_dissoc"],
        }

        # Fraction of sticker degrees across all frames/replicates
        frac_bond0_all = np.concatenate(
            [res["frac_bond0_series"] for res in replicate_results]
        )
        frac_bond1_all = np.concatenate(
            [res["frac_bond1_series"] for res in replicate_results]
        )
        frac_bond_gt1_all = np.concatenate(
            [res["frac_bond_gt1_series"] for res in replicate_results]
        )
        cexc_all = np.concatenate([res["cexc_series"] for res in replicate_results])
        frac_bond0_data.append(frac_bond0_all)
        frac_bond1_data.append(frac_bond1_all)
        frac_bond_gt1_data.append(frac_bond_gt1_all)
        cexc_data.append(cexc_all)

        # Output per-epsilon directory
        eps_dir = os.path.join(args.output_dir, f"eps_{epsilon:g}")
        os.makedirs(eps_dir, exist_ok=True)
        log(f"Writing outputs for eps={epsilon:g} to {eps_dir}")

        if (
            cluster_size is not None
            and cluster_mean is not None
            and cluster_stderr is not None
        ):
            with open(
                os.path.join(eps_dir, "cluster_distribution.csv"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("cluster_size,mean,stderr\n")
                for size, mean_val, stderr_val in zip(
                    cluster_size, cluster_mean, cluster_stderr
                ):
                    handle.write(
                        f"{int(size)},{mean_val:.6e},{stderr_val:.6e}\n"
                    )

        if cs_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "bond_correlation.csv"),
                cs_time,
                cs_mean,
                cs_stderr,
            )
        if tb_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "open_sticker_correlation.csv"),
                tb_time,
                tb_mean,
                tb_stderr,
            )
        if cp_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "connectivity_correlation.csv"),
                cp_time,
                cp_mean,
                cp_stderr,
            )
        if cs_time is not None and cs_values is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "bond_correlation_fit.png"),
                cs_time,
                cs_values,
                title=f"Bond Correlation Decay (eps={epsilon:g})",
                y_label="C_s(t)",
                use_semilog_linear_region=True,
            )
        if tb_time is not None and tb_values is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "open_sticker_correlation_fit.png"),
                tb_time,
                tb_values,
                title=f"Open-Sticker Correlation Decay (eps={epsilon:g})",
                y_label=r"C_b(t)",
                use_semilog_linear_region=True,
            )
        if cp_time is not None and cp_values is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "connectivity_correlation_fit.png"),
                cp_time,
                cp_values,
                title=f"Connectivity Correlation Decay (eps={epsilon:g})",
                y_label="C_p(t)",
            )
        if G_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "stress_modulus.csv"), G_time, G_mean, G_stderr
            )
        if msd_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "msd.csv"), msd_time, msd_mean, msd_stderr
            )
        with open(os.path.join(eps_dir, "properties.json"), "w", encoding="utf-8") as handle:
            json.dump(epsilon_properties, handle, indent=2)
        write_properties_csv(os.path.join(eps_dir, "properties.csv"), epsilon_properties)

        summary_rows.append(
            {
                "epsilon": epsilon,
                **{f"{k}_mean": v["mean"] for k, v in scalar_summary.items()},
                **{f"{k}_stderr": v["stderr"] for k, v in scalar_summary.items()},
            }
        )
        summary_json[f"{epsilon:g}"] = scalar_summary
        log(f"Finished epsilon group eps={epsilon:g}")

    # Write summary JSON and CSV
    with open(
        os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary_json, handle, indent=2)
    log(f"Wrote {os.path.join(args.output_dir, 'summary.json')}")

    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(
            os.path.join(args.output_dir, "summary.csv"), "w", encoding="utf-8"
        ) as handle:
            handle.write(",".join(keys) + "\n")
            for row in summary_rows:
                handle.write(",".join(str(row[k]) for k in keys) + "\n")
        log(f"Wrote {os.path.join(args.output_dir, 'summary.csv')}")

    if epsilon_values:
        compact_eps_plot_kwargs = {
            "figsize": (3.3 / 2.0, 3.3 / 2.0),
            "dpi": 1000,
            "x_label": r"$\varepsilon$",
            "tick_label_size": 8,
            "axis_label_size": 10,
        }
        bond_tau_plot_kwargs = dict(compact_eps_plot_kwargs)
        bond_tau_plot_kwargs["x_label"] = r"$\varepsilon_\mathrm{reactiveLJ}$"
        intra_inter_ratio_plot_kwargs = dict(compact_eps_plot_kwargs)
        intra_inter_ratio_plot_kwargs["figsize"] = INTRA_INTER_RATIO_FIGSIZE
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond0_violin.png"),
            epsilon_values,
            frac_bond0_data,
            title=None,
            y_label="Fraction of stickers",
            **compact_eps_plot_kwargs,
        )
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond1_violin.png"),
            epsilon_values,
            frac_bond1_data,
            title=None,
            y_label="Fraction of stickers",
            **compact_eps_plot_kwargs,
        )
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond_gt1_violin.png"),
            epsilon_values,
            frac_bond_gt1_data,
            title=None,
            y_label="Fraction of stickers",
            **compact_eps_plot_kwargs,
        )
        write_cexc_vs_epsilon_plot(
            os.path.join(args.output_dir, "cexc_vs_epsilon.png"),
            epsilon_values,
            cexc_data,
        )
        write_cluster_distribution_by_epsilon_plot(
            os.path.join(args.output_dir, "cluster_size_distribution_by_epsilon.svg"),
            epsilon_values,
            cluster_distribution_by_eps,
        )
        shared_time_lag_xlim = compute_shared_time_lag_xlim(
            msd_time_by_eps,
            msd_mean_by_eps,
            g_time_by_eps,
            g_mean_by_eps,
            tau_r0_reference,
            plot_x_min_time=args.plot_x_min_time,
            plot_x_max_tau_r0=args.plot_x_max_tau_r0,
        )
        write_msd_vs_epsilon_plot(
            os.path.join(args.output_dir, "monomer_msd_vs_time_lag_by_epsilon.svg"),
            epsilon_values,
            msd_time_by_eps,
            msd_mean_by_eps,
            tau_r0_reference,
            x_limits=shared_time_lag_xlim,
        )
        remove_legacy_bond_tau_plots(args.output_dir)
        resolved_eps, resolved_tau_s = resolved_bond_tau_data(
            epsilon_values,
            tau_s_data,
        )
        write_log_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, BOND_TAU_EPSILON_PLOT),
            resolved_eps,
            resolved_tau_s,
            title=None,
            **bond_tau_plot_kwargs,
        )
        write_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, "brachiation_tau_vs_epsilon.png"),
            epsilon_values,
            tau_b_data,
            title="Brachiation Time vs epsilon",
            y_label="brachiation time",
        )
        scalar_violin_specs = [
            (
                "p_open_mean",
                "p_open_vs_epsilon.png",
                "Open Sticker Fraction vs epsilon",
                "p_open",
            ),
            (
                "p_mean",
                "bonding_probability_vs_epsilon.png",
                "Bonding Probability vs epsilon",
                "p",
            ),
            (
                "intra_inter_ratio",
                INTRA_INTER_RATIO_EPSILON_PLOT,
                "Intra/Inter Bond Ratio vs epsilon",
                INTRA_INTER_RATIO_LABEL,
            ),
            (
                "tau_c",
                "fluctuation_relaxation_tau_c_vs_epsilon.png",
                "Fluctuation Relaxation Time vs epsilon",
                "tau_c",
            ),
            (
                "stickers_per_chain",
                "stickers_per_chain_vs_epsilon.png",
                "Stickers per Chain vs epsilon",
                "f",
            ),
        ]
        for key, filename, title, y_label in scalar_violin_specs:
            plot_kwargs = {}
            plot_title = title
            if key == "intra_inter_ratio":
                plot_title = None
                plot_kwargs = intra_inter_ratio_plot_kwargs
            write_scalar_violin_vs_epsilon_plot(
                os.path.join(args.output_dir, filename),
                epsilon_values,
                scalar_violin_data[key],
                title=plot_title,
                y_label=y_label,
                **plot_kwargs,
            )
        gelation_epsilon, gelation_mean, gelation_stderr = (
            summarize_gelation_epsilon_points(
                epsilon_values,
                scalar_violin_data["epsilon_mean"],
            )
        )
        write_gelation_epsilon_plot(
            os.path.join(args.output_dir, GELATION_EPSILON_PLOT),
            gelation_epsilon,
            gelation_mean,
            gelation_stderr,
        )
        exchange_epsilon, exchange_assoc, exchange_dissoc = (
            exchange_rate_points_from_summary_rows(summary_rows)
        )
        write_exchange_rate_plot(
            os.path.join(args.output_dir, EXCHANGE_RATE_EPSILON_PLOT),
            exchange_epsilon,
            exchange_assoc,
            exchange_dissoc,
        )
        write_stress_modulus_by_epsilon_plot(
            os.path.join(args.output_dir, "stress_modulus_vs_time_lag_by_epsilon.svg"),
            epsilon_values,
            g_time_by_eps,
            g_mean_by_eps,
            tau_r0_reference,
            x_limits=shared_time_lag_xlim,
        )

    log("Analysis complete")


if __name__ == "__main__":
    main()
