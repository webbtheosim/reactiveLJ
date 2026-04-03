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
import numpy as np
import freud
import gsd.hoomd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports resolve when running from repo root.
sys.path.append(os.path.dirname(__file__))

FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 100.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0
REQUIRED_TRAJECTORY_FRAMES = 2020
DEFAULT_STRESS_MAX_RUNTIME_FRACTION = 0.2
ANALYSIS_CHOICES = ("all", "msd", "stress_modulus")

from analysis_utils import (
    CorrelationAccumulator,
    UnionFind,
    autocorr_fft,
    compute_r_thresh,
    extract_semilog_linear_region,
    find_sticker_neighbor_pairs,
    fit_exponential,
    fit_exponential_semilog_linear_region,
    multitau_autocovariance,
)

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ReactiveLJ trajectories.")
    parser.add_argument(
        "--input-root",
        default="../data_generation/outputs",
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
        "--max-lag-frames",
        type=int,
        default=100,
        help="Maximum lag (in frames) used for correlation functions.",
    )
    parser.add_argument(
        "--msd-max-lag-frames",
        type=int,
        default=0,
        help=(
            "Maximum lag (in frames) used for MSD calculation; 0 uses the "
            "same truncated window as G(t), capped at 100 tau_R^0."
        ),
    )
    parser.add_argument(
        "--stress-max-runtime-fraction",
        type=float,
        default=DEFAULT_STRESS_MAX_RUNTIME_FRACTION,
        help=(
            "Maximum stress-modulus lag as a fraction of the virial-log runtime "
            "(default: 0.2)."
        ),
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=ANALYSIS_CHOICES,
        default=["all"],
        help=(
            "Which analysis families to run. Use `all` for the full pipeline "
            "(default), or choose one or more of `msd` and `stress_modulus` "
            "to rerun only those outputs."
        ),
    )
    return parser.parse_args()


def resolve_analysis_selection(args: argparse.Namespace) -> Tuple[bool, bool, bool]:
    requested: Set[str] = set(args.analyses)
    if "all" in requested:
        return True, True, True
    return False, ("msd" in requested), ("stress_modulus" in requested)


def discover_runs(input_root: str) -> List[Tuple[str, str]]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ", "archived"}
    runs = []
    skipped_short = 0
    for root, dirs, files in os.walk(input_root, topdown=True):
        # Keep test trajectories out of production analysis sweeps.
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        if "trajectory.gsd" in files:
            gsd_path = os.path.join(root, "trajectory.gsd")
            with gsd.hoomd.open(gsd_path, "r") as traj:
                n_frames = len(traj)
            if n_frames < REQUIRED_TRAJECTORY_FRAMES:
                skipped_short += 1
                rel_path = os.path.relpath(gsd_path, input_root)
                log(
                    f"Skipping incomplete trajectory {rel_path}: "
                    f"{n_frames} frames < {REQUIRED_TRAJECTORY_FRAMES}"
                )
                continue
            metadata_path = os.path.join(root, "metadata.json")
            runs.append((gsd_path, metadata_path))
    if skipped_short > 0:
        log(
            f"Skipped {skipped_short} trajectories with fewer than "
            f"{REQUIRED_TRAJECTORY_FRAMES} frames"
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


def compute_stress_autocovariance_multitau(
    tensor_arr: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if tensor_arr.ndim != 2 or tensor_arr.shape[0] <= 1 or tensor_arr.shape[1] < 6:
        return None, None

    xx = tensor_arr[:, 0]
    xy = tensor_arr[:, 1]
    xz = tensor_arr[:, 2]
    yy = tensor_arr[:, 3]
    yz = tensor_arr[:, 4]
    zz = tensor_arr[:, 5]

    # Rotationally averaged deviatoric-stress autocovariance following
    # ref. 57 eq. 33: 1/5 over the three shear components plus 1/30 over
    # the three normal-stress differences.
    weighted_series = (
        (1.0 / 5.0, (xy, xz, yz)),
        (1.0 / 30.0, (xx - yy, xx - zz, yy - zz)),
    )

    g_lags = None
    g_cov = None
    for weight, series_group in weighted_series:
        for series in series_group:
            centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
            lags_i, cov_i = multitau_autocovariance(centered)
            if g_lags is None:
                g_lags = lags_i
                g_cov = np.zeros_like(cov_i, dtype=np.float64)
            elif not np.array_equal(lags_i, g_lags):
                raise RuntimeError(
                    'Multi-tau lag grids do not match across stress components.'
                )
            g_cov += weight * cov_i

    return g_lags, g_cov


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
    str | None,
]:
    G_t = None
    G_autocorr_t = None
    G_time = None
    virial_source = None

    virial_arr = np.empty((0, 6), dtype=np.float64)
    virial_steps = np.empty((0,), dtype=np.int64)
    virial_log_path = os.path.join(os.path.dirname(gsd_path), "virial_tensor_log.gsd")
    if os.path.exists(virial_log_path):
        virial_arr, virial_steps = load_virial_series_from_gsd(virial_log_path)
        if virial_arr.size > 0:
            virial_source = "virial_tensor_log.gsd"

    if virial_arr.shape[0] <= 1:
        return G_time, G_t, G_autocorr_t, virial_source

    g_lags, g_cov = compute_stress_autocovariance_multitau(virial_arr)
    if g_lags is None or g_cov is None:
        return G_time, G_t, G_autocorr_t, virial_source

    cov0 = g_cov[0] if g_cov.size > 0 and g_cov[0] != 0.0 else np.nan
    G_autocorr_t = g_cov / cov0 if g_cov.size > 0 else np.empty((0,), dtype=np.float64)

    skip = 1 if len(g_lags) > 1 and g_lags[0] == 0 else 0
    g_lags = g_lags[skip:]
    g_cov = g_cov[skip:]
    G_autocorr_t = G_autocorr_t[skip:]

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

    volume = box_length**3
    kT = metadata.get("temperature", 1.0)
    G_t = (volume / kT) * g_cov
    G_time = g_time
    return G_time, G_t, G_autocorr_t, virial_source


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

    max_lag_time = min(0.5 * runtime, MAX_ANALYSIS_LAG_TIME)
    half_runtime_lag = int(np.floor(max_lag_time / float(sample_dt) + 1.0e-12))
    half_runtime_lag = max(1, min(half_runtime_lag, n_frames - 1))
    requested_lag = (n_frames - 1) if max_lag_frames <= 0 else min(int(max_lag_frames), n_frames - 1)
    max_lag = max(1, min(requested_lag, half_runtime_lag))

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
    max_lag_frames: int,
    msd_max_lag_frames: int,
    stress_max_runtime_fraction: float,
    run_full_suite: bool,
    run_msd: bool,
    run_stress_modulus: bool,
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
                G_time, G_t, G_autocorr_t, virial_source = compute_stress_modulus_from_virial(
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
                result.update({"msd_time": msd_time, "msd": msd})

            return result

        type_names = first.particles.types
        if "sticky" not in type_names:
            raise RuntimeError("Sticker type 'sticky' not found in trajectory.")
        sticker_type = type_names.index("sticky")

        typeid = np.asarray(first.particles.typeid, dtype=np.int32)
        chain_length = int(metadata.get("chain_length", 1))
        n_chains = int(metadata.get("n_chains", first.particles.N // chain_length))
        trajectory_subset = str(metadata.get("trajectory_particle_subset", ""))
        if trajectory_subset != "sticky_only":
            raise RuntimeError(
                f"Expected sticker-only trajectory metadata for {gsd_path}, "
                f"found trajectory_particle_subset={trajectory_subset!r}."
            )

        expected_sticker_tags = sticker_tags_from_metadata(metadata)
        if expected_sticker_tags.size != first.particles.N:
            raise RuntimeError(
                f"Sticker-only trajectory size mismatch in {gsd_path}: "
                f"expected {expected_sticker_tags.size} particles, got {first.particles.N}."
            )
        if not np.all(typeid == sticker_type):
            raise RuntimeError(
                f"Sticker-only trajectory {gsd_path} contains non-sticky particle types."
            )
        sticker_ids = np.arange(first.particles.N, dtype=np.int32)
        sticker_chain_ids = (expected_sticker_tags // chain_length).astype(
            np.int32, copy=False
        )
        n_stickers = int(sticker_ids.size)

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
        reactive_epsilon = float(metadata.get("reactive_epsilon", float("nan")))
        stickers_per_chain = float(metadata.get("stickers_per_chain", 4))
        p_c = (
            float(1.0 / (stickers_per_chain - 1.0))
            if stickers_per_chain > 1.0
            else float("nan")
        )

        bond_corr = CorrelationAccumulator(max_lag_frames)
        open_corr = CorrelationAccumulator(max_lag_frames)

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

        rate_assoc_sum = 0.0
        rate_dissoc_sum = 0.0
        rate_count = 0

        prev_bonds: set[int] | None = None
        prev_open_count: int | None = None
        prev_partners: Dict[int, set] | None = None

        for analyzed_idx, frame_idx in enumerate(
            range(0, n_frames, analysis_stride), start=1
        ):
            frame = traj[frame_idx]
            positions = frame.particles.position

            pair_i, pair_j, pair_dist = find_sticker_neighbor_pairs(
                positions, sticker_ids, box_length, pair_cutoff
            )
            degrees = np.zeros(n_stickers, dtype=np.int32)
            uf = UnionFind(n_chains)
            intra = 0
            inter = 0
            partner_map: Dict[int, set] = defaultdict(set)
            bonds: set[int] = set()

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
                bond_global_i = sticker_ids[bond_i]
                bond_global_j = sticker_ids[bond_j]
                bond_ids = np.empty(bond_i.size, dtype=np.int64)
                for bond_idx, (i_local, j_local, i_global, j_global) in enumerate(
                    zip(bond_i, bond_j, bond_global_i, bond_global_j)
                ):
                    degrees[i_local] += 1
                    degrees[j_local] += 1

                    chain_i = int(sticker_chain_ids[i_local])
                    chain_j = int(sticker_chain_ids[j_local])
                    if chain_i != chain_j:
                        uf.union(chain_i, chain_j)
                        inter += 1
                    else:
                        intra += 1

                    i_global_int = int(i_global)
                    j_global_int = int(j_global)
                    partner_map[i_global_int].add(j_global_int)
                    partner_map[j_global_int].add(i_global_int)
                    if i_global_int < j_global_int:
                        bond_key = i_global_int * n_stickers + j_global_int
                    else:
                        bond_key = j_global_int * n_stickers + i_global_int
                    bond_ids[bond_idx] = bond_key
                    bonds.add(bond_key)
                bond_ids.sort()
            else:
                bond_ids = np.empty((0,), dtype=np.int64)

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
            for size in sizes:
                cluster_hist[size] += 1
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
                prev_bonds is not None
                and prev_open_count is not None
                and prev_partners is not None
            ):
                new_bonds = bonds - prev_bonds
                n_m = prev_open_count
                if n_m > 0:
                    assoc = 0
                    dissoc = 0
                    for bond_key in new_bonds:
                        i = bond_key // n_stickers
                        j = bond_key % n_stickers
                        i_prev = prev_partners.get(i, set())
                        j_prev = prev_partners.get(j, set())
                        if (i_prev and (j not in i_prev)) or (
                            j_prev and (i not in j_prev)
                        ):
                            assoc += 1
                        else:
                            dissoc += 1
                    rate_assoc_sum += assoc / (n_m * frame_dt)
                    rate_dissoc_sum += dissoc / (n_m * frame_dt)
                    rate_count += 1

            prev_bonds = bonds
            prev_open_count = open_count
            prev_partners = partner_map

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

        cs_time = np.arange(1, len(cs) + 1, dtype=np.float64) * frame_dt
        tb_time = np.arange(1, len(cb) + 1, dtype=np.float64) * frame_dt

        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)
        tau_b = fit_exponential_semilog_linear_region(tb_time, cb)

        p_arr = np.array(p_series, dtype=np.float64)
        cp_full = autocorr_fft(p_arr, subtract_mean=True)
        cp = cp_full[1 : max_lag_frames + 1]
        cp_time = np.arange(1, len(cp) + 1, dtype=np.float64) * frame_dt
        tau_c = fit_exponential(cp_time, cp)

        G_time, G_t, G_autocorr_t, virial_source = compute_stress_modulus_from_virial(
            gsd_path,
            metadata,
            box_length,
            frame_steps,
            analysis_stride,
            stress_max_runtime_fraction,
        )

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
            "rate_assoc": rate_assoc_sum / max(rate_count, 1),
            "rate_dissoc": rate_dissoc_sum / max(rate_count, 1),
            "tau_s": tau_s,
            "tau_b": tau_b,
            "tau_c": tau_c,
            "stickers_per_chain": stickers_per_chain,
            "p_c": p_c,
            "cluster_hist": cluster_hist,
            "cluster_count_total": cluster_count_total,
            "cs_time": cs_time,
            "cs": cs,
            "tb_time": tb_time,
            "tb_corr": cb,
            "cp_time": cp_time,
            "cp": cp,
            "G_time": G_time,
            "G_t": G_t,
            "G_autocorr_t": G_autocorr_t,
            "virial_source": virial_source,
            "msd_time": msd_time,
            "msd": msd,
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
) -> Tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    series = [
        (res[key_time], res[key_val])
        for res in replicate_results
        if res.get(key_val) is not None
    ]
    if not series:
        return None, None, None, None

    normalized = []
    for time_arr, value_arr in series:
        time_f = np.asarray(time_arr, dtype=np.float64)
        value_f = np.asarray(value_arr, dtype=np.float64)
        n = min(len(time_f), len(value_f))
        if n > 0:
            normalized.append((time_f[:n], value_f[:n]))
    if not normalized:
        return None, None, None, None

    base_idx = max(range(len(normalized)), key=lambda idx: normalized[idx][0].size)
    base_time = normalized[base_idx][0]
    values = np.full((len(normalized), base_time.size), np.nan, dtype=np.float64)
    for row_idx, (time_arr, value_arr) in enumerate(normalized):
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

    populated = np.any(np.isfinite(values), axis=0)
    if not np.any(populated):
        return None, None, None, None

    time = base_time[populated]
    values = values[:, populated]
    mean = np.nanmean(values, axis=0)
    stderr = finite_column_stderr(values)
    return time, values, mean, stderr


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
    max_lag_frames: int,
    msd_max_lag_frames: int,
    stress_max_runtime_fraction: float,
    run_full_suite: bool,
    run_msd: bool,
    run_stress_modulus: bool,
    rep_label: str,
    rel_path: str,
) -> Tuple[float, Dict]:
    log(f"{rep_label}: start ({rel_path})")
    result = analyze_replicate(
        gsd_path,
        metadata,
        analysis_stride,
        max_lag_frames,
        msd_max_lag_frames,
        stress_max_runtime_fraction,
        run_full_suite=run_full_suite,
        run_msd=run_msd,
        run_stress_modulus=run_stress_modulus,
        progress_label=rep_label,
    )
    log(f"{rep_label}: done")
    return epsilon, result


def truncate_lag(
    time: np.ndarray | None, values: np.ndarray | None, max_lag: int
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if time is None or values is None:
        return None, None
    n = min(max_lag, len(time))
    return time[:n], values[:, :n]


def write_fraction_violin_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
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

    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
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


def write_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str,
    y_label: str,
    log_transform: bool = False,
) -> None:
    processed: List[Tuple[float, np.ndarray]] = []
    for eps, values in zip(epsilon_values, data):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if log_transform:
            arr = arr[arr > 0.0]
            if arr.size:
                arr = np.log(arr)
        if arr.size == 0:
            continue
        processed.append((float(eps), arr))

    if not processed:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
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

    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
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
    path: str, epsilon_values: List[float], tau_data: List[np.ndarray]
) -> None:
    write_scalar_violin_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        data=tau_data,
        title="ln(Bond Correlation Decay Tau) vs epsilon",
        y_label="ln(bond persistence time)",
        log_transform=True,
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


def write_pooled_timeseries_mean_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    values_by_eps: Dict[float, np.ndarray],
    title: str,
    y_label: str,
) -> None:
    pooled_eps: List[float] = []
    pooled_means: List[float] = []
    for eps in epsilon_values:
        values = values_by_eps.get(eps)
        if values is None:
            continue
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        pooled_eps.append(float(eps))
        pooled_means.append(float(np.mean(finite)))

    if not pooled_eps:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(pooled_eps, pooled_means, color="#2b2b2b", marker="o", lw=2.0)
    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(pooled_eps)
    ax.set_xticklabels([f"{eps:g}" for eps in pooled_eps])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_msd_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    msd_values_by_eps: Dict[float, np.ndarray],
) -> None:
    write_pooled_timeseries_mean_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        values_by_eps=msd_values_by_eps,
        title="Pooled Mean Monomer MSD vs epsilon",
        y_label=r"mean $g_1(t)$",
    )


def write_stress_modulus_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    g_values_by_eps: Dict[float, np.ndarray],
) -> None:
    write_pooled_timeseries_mean_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        values_by_eps=g_values_by_eps,
        title="Pooled Mean Stress Modulus vs epsilon",
        y_label="mean G(t)",
    )

def main() -> None:
    args = parse_args()
    if args.stress_max_runtime_fraction <= 0.0:
        raise RuntimeError(
            "--stress-max-runtime-fraction must be > 0, "
            f"got {args.stress_max_runtime_fraction}."
        )
    run_full_suite, run_msd, run_stress_modulus = resolve_analysis_selection(args)
    if run_full_suite:
        log("Requested analyses: all")
    else:
        selected = []
        if run_msd:
            selected.append("msd")
        if run_stress_modulus:
            selected.append("stress_modulus")
        log(f"Requested analyses: {', '.join(selected)}")
    log(f"Scanning trajectories under {args.input_root}")
    runs = discover_runs(args.input_root)
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
    g_values_by_eps: Dict[float, np.ndarray] = {}
    msd_values_by_eps: Dict[float, np.ndarray] = {}
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
                    args.max_lag_frames,
                    args.msd_max_lag_frames,
                    args.stress_max_runtime_fraction,
                    run_full_suite,
                    run_msd,
                    run_stress_modulus,
                    rep_label,
                    rel_path,
                )
            )

    log(f"Starting parallel analysis with {n_jobs} workers on {len(jobs)} runs")
    results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    log(f"Completed parallel analysis for {len(results)} runs")

    grouped_results: Dict[float, List[Dict]] = defaultdict(list)
    for epsilon, result in results:
        grouped_results[epsilon].append(result)

    sorted_results = sorted(grouped_results.items())
    if not run_full_suite:
        msd_eps: List[float] = []
        stress_eps: List[float] = []
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
                    replicate_results, "G_time", "G_t"
                )
                if (
                    G_time is not None
                    and G_values is not None
                    and G_mean is not None
                    and G_stderr is not None
                ):
                    stress_eps.append(epsilon)
                    g_values_by_eps[epsilon] = G_values
                    write_timeseries(
                        os.path.join(eps_dir, "stress_modulus.csv"),
                        G_time,
                        G_mean,
                        G_stderr,
                    )

            if run_msd:
                msd_time, msd_values, msd_mean, msd_stderr = aggregate_replicate_timeseries(
                    replicate_results, "msd_time", "msd"
                )
                if (
                    msd_time is not None
                    and msd_values is not None
                    and msd_mean is not None
                    and msd_stderr is not None
                ):
                    msd_eps.append(epsilon)
                    msd_values_by_eps[epsilon] = msd_values
                    write_timeseries(
                        os.path.join(eps_dir, "msd.csv"),
                        msd_time,
                        msd_mean,
                        msd_stderr,
                    )

            log(f"Updated selected outputs for eps={epsilon:g} in {eps_dir}")

        if run_stress_modulus and stress_eps:
            write_stress_modulus_by_epsilon_plot(
                os.path.join(args.output_dir, "stress_modulus_mean_vs_epsilon.png"),
                stress_eps,
                g_values_by_eps,
            )
        if run_msd and msd_eps:
            write_msd_vs_epsilon_plot(
                os.path.join(args.output_dir, "monomer_msd_mean_vs_epsilon.png"),
                msd_eps,
                msd_values_by_eps,
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

        # Cluster distribution
        cluster_hists = [res["cluster_hist"] for res in replicate_results]
        cluster_counts = [res["cluster_count_total"] for res in replicate_results]
        max_len = max(len(hist) for hist in cluster_hists)
        padded = []
        for hist in cluster_hists:
            pad = np.zeros(max_len, dtype=np.float64)
            pad[: len(hist)] = hist
            padded.append(pad)
        cluster_p = [pad / max(count, 1) for pad, count in zip(padded, cluster_counts)]
        cluster_p_arr = np.vstack(cluster_p)
        cluster_mean = np.mean(cluster_p_arr, axis=0)
        cluster_stderr = (
            np.std(cluster_p_arr, axis=0, ddof=1) / np.sqrt(cluster_p_arr.shape[0])
            if cluster_p_arr.shape[0] > 1
            else np.zeros_like(cluster_mean)
        )

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
            replicate_results, "G_time", "G_t"
        )
        msd_time, msd_values, msd_mean, msd_stderr = aggregate_replicate_timeseries(
            replicate_results, "msd_time", "msd"
        )
        if G_time is not None and G_values is not None:
            g_values_by_eps[epsilon] = G_values
        if msd_time is not None and msd_values is not None:
            msd_values_by_eps[epsilon] = msd_values

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

        # Cluster distribution CSV
        with open(
            os.path.join(eps_dir, "cluster_distribution.csv"), "w", encoding="utf-8"
        ) as handle:
            handle.write("cluster_size,mean,stderr\n")
            for size, (m, s) in enumerate(zip(cluster_mean, cluster_stderr)):
                handle.write(f"{size},{m:.6e},{s:.6e}\n")

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
            cs_time_fit, cs_values_fit = truncate_lag(
                cs_time, cs_values, args.max_lag_frames
            )
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "bond_correlation_fit.png"),
                cs_time_fit,
                cs_values_fit,
                title=f"Bond Correlation Decay (eps={epsilon:g})",
                y_label="C_s(t)",
                use_semilog_linear_region=True,
            )
        tb_time_fit, tb_values_fit = truncate_lag(
            tb_time, tb_values, args.max_lag_frames
        )
        if tb_time_fit is not None and tb_values_fit is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "open_sticker_correlation_fit.png"),
                tb_time_fit,
                tb_values_fit,
                title=f"Open-Sticker Correlation Decay (eps={epsilon:g})",
                y_label=r"C_b(t)",
                use_semilog_linear_region=True,
            )
        cp_time_fit, cp_values_fit = truncate_lag(
            cp_time, cp_values, args.max_lag_frames
        )
        if cp_time_fit is not None and cp_values_fit is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "connectivity_correlation_fit.png"),
                cp_time_fit,
                cp_values_fit,
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
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond0_violin.png"),
            epsilon_values,
            frac_bond0_data,
            title="Sticker Fraction with 0 Bonds",
            y_label="Fraction of stickers",
        )
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond1_violin.png"),
            epsilon_values,
            frac_bond1_data,
            title="Sticker Fraction with 1 Bond",
            y_label="Fraction of stickers",
        )
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond_gt1_violin.png"),
            epsilon_values,
            frac_bond_gt1_data,
            title="Sticker Fraction with >1 Bonds",
            y_label="Fraction of stickers",
        )
        write_cexc_vs_epsilon_plot(
            os.path.join(args.output_dir, "cexc_vs_epsilon.png"),
            epsilon_values,
            cexc_data,
        )
        write_msd_vs_epsilon_plot(
            os.path.join(args.output_dir, "monomer_msd_mean_vs_epsilon.png"),
            epsilon_values,
            msd_values_by_eps,
        )
        write_log_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, "ln_bond_tau_vs_epsilon.png"),
            epsilon_values,
            tau_s_data,
        )
        write_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, "bond_tau_vs_epsilon.png"),
            epsilon_values,
            tau_s_data,
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
                "epsilon_mean",
                "gelation_epsilon_vs_epsilon.png",
                "Degree of Gelation vs epsilon",
                "gelation epsilon",
            ),
            (
                "intra_inter_ratio",
                "intra_to_inter_bond_ratio_vs_epsilon.png",
                "Intra/Inter Bond Ratio vs epsilon",
                "intra/inter bond ratio",
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
            write_scalar_violin_vs_epsilon_plot(
                os.path.join(args.output_dir, filename),
                epsilon_values,
                scalar_violin_data[key],
                title=title,
                y_label=y_label,
            )
        write_dual_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "exchange_rate_comparison_vs_epsilon.png"),
            epsilon_values,
            scalar_violin_data["rate_assoc"],
            scalar_violin_data["rate_dissoc"],
            title="Associative Exchange vs Passive Dimerization Rates",
            y_label="rate (1/time)",
            left_label="Associative exchange rate (R_a)",
            right_label="Passive dimerization rate (R_d)",
            left_color="#e77500",
            right_color="#121212",
        )
        write_stress_modulus_by_epsilon_plot(
            os.path.join(args.output_dir, "stress_modulus_mean_vs_epsilon.png"),
            epsilon_values,
            g_values_by_eps,
        )

    log("Analysis complete")


if __name__ == "__main__":
    main()
