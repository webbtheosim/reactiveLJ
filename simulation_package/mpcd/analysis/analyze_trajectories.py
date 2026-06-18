"""Analyze MPCD-solvated ReactiveLJ polymer-solution trajectories.

The MPCD system is a dilute/semidilute polymer solution in solvent, so the
primary observables are local sticker kinetics, finite cluster statistics,
clumping diagnostics, degree of gelation, and sampled monomer diffusion.
Equilibrium stress-modulus outputs from the melt pipeline are intentionally
omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import gsd.hoomd
import numba
import numpy as np
from joblib import Parallel, delayed

_CACHE_ROOT = os.path.join("/tmp", f"mpcd-analysis-cache-{os.getuid()}")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import ultraplot as uplt

# Ensure local imports resolve when running from repo root.
sys.path.append(os.path.dirname(__file__))

from analysis_utils import (
    UnionFind,
    compute_r_thresh,
    compute_sticker_neighbor_pairs,
    extract_semilog_linear_region,
    fit_exponential_semilog_linear_region,
)
from make_brachiation_tau_plot import (
    DEFAULT_OUTPUT_NAME as BRACHIATION_TAU_EPSILON_PLOT,
    summarize_replicate_points as summarize_brachiation_tau_points,
    write_brachiation_tau_plot,
)
from make_largest_cluster_size_plot import (
    DEFAULT_OUTPUT_NAME as LARGEST_CLUSTER_SIZE_EPSILON_PLOT,
    summarize_replicate_points as summarize_largest_cluster_size_points,
    write_largest_cluster_size_plot,
)


FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 1000.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0
REPLICATE_CACHE_VERSION = 10
TIME_AXIS_MATCH_RTOL = 1.0e-8
TIME_AXIS_MATCH_ATOL = 1.0e-12

PLOT_DPI = 1000
TICK_FONTSIZE = 8
LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 8
INTRA_INTER_RATIO_EPSILON_PLOT = "intra_to_inter_bond_ratio_vs_epsilon.svg"
INTRA_INTER_RATIO_LABEL = r"$\psi$"
INTRA_INTER_RATIO_FIGSIZE = (3.3, 3.3 / 2.0)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MPCD-solvated ReactiveLJ polymer-solution trajectories."
    )
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
        help="Stride over saved GSD frames during sticker/cluster analysis.",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=0,
        help=(
            "Maximum lag in saved frames used for sticker correlation functions; "
            "0 uses the same 1000 tau_R^0 physical cap as the melt analysis, "
            "subject to available runtime."
        ),
    )
    parser.add_argument(
        "--msd-max-lag-frames",
        type=int,
        default=0,
        help=(
            "Maximum lag in sampled MSD frames; 0 uses the same 1000 tau_R^0 "
            "physical cap as the melt analysis, subject to available runtime."
        ),
    )
    return parser.parse_args()


def discover_runs(input_root: str) -> List[Tuple[str, str]]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ"}
    runs: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(input_root):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        if "trajectory.gsd" in files:
            metadata_path = os.path.join(root, "metadata.json")
            runs.append((os.path.join(root, "trajectory.gsd"), metadata_path))
    return sorted(runs)


def file_signature(path: str) -> Dict[str, int | str | None]:
    if not os.path.exists(path):
        return {"path": path, "exists": None, "size": None, "mtime_ns": None}
    stat = os.stat(path)
    return {
        "path": path,
        "exists": "1",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def frame_step_or_index(frame, fallback_index: int) -> int:
    frame_step = getattr(frame.configuration, "step", None)
    return int(frame_step) if frame_step is not None else int(fallback_index)


def checkpoint_deduplicated_frame_indices(traj, gsd_path: str) -> np.ndarray:
    """Keep the resumed branch when append-mode checkpoint restarts overlap."""
    kept_indices: List[int] = []
    kept_steps: List[int] = []
    discarded_frames = 0
    overlap_events = 0
    first_overlap_frame: int | None = None
    first_overlap_step: int | None = None

    for physical_idx in range(len(traj)):
        step = frame_step_or_index(traj[physical_idx], physical_idx)
        dropped_this_frame = 0
        while kept_steps and kept_steps[-1] >= step:
            kept_steps.pop()
            kept_indices.pop()
            discarded_frames += 1
            dropped_this_frame += 1
        if dropped_this_frame:
            overlap_events += 1
            if first_overlap_frame is None:
                first_overlap_frame = physical_idx
                first_overlap_step = step
        kept_steps.append(step)
        kept_indices.append(physical_idx)

    if discarded_frames:
        log(
            "Affected GSD file: "
            f"{gsd_path} has checkpoint-overlap frames; using resumed branch "
            f"and ignoring {discarded_frames} earlier frame(s) across "
            f"{overlap_events} overlap event(s) "
            f"(logical_frames={len(kept_indices)}/{len(traj)}, "
            f"first_overlap_physical_frame={first_overlap_frame}, "
            f"first_overlap_step={first_overlap_step})."
        )

    return np.asarray(kept_indices, dtype=np.int64)


def build_replicate_cache_key(
    gsd_path: str,
    metadata_path: str,
    analysis_stride: int,
    max_lag_frames: int,
    msd_max_lag_frames: int,
) -> str:
    msd_gsd_path = os.path.join(os.path.dirname(gsd_path), "msd_trajectory.gsd")
    payload = {
        "version": REPLICATE_CACHE_VERSION,
        "files": {
            "trajectory": file_signature(gsd_path),
            "metadata": file_signature(metadata_path),
            "msd_trajectory": file_signature(msd_gsd_path),
        },
        "analysis_args": {
            "analysis_stride": int(analysis_stride),
            "max_lag_frames": int(max_lag_frames),
            "msd_max_lag_frames": int(msd_max_lag_frames),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_replicate_result(cache_path: str) -> Dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError):
        return None


def save_cached_replicate_result(cache_path: str, result: Dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def sticker_tags_from_metadata(metadata: Dict) -> np.ndarray:
    n_chains = int(metadata.get("n_chains", 0))
    chain_length = int(metadata.get("chain_length", 0))
    stickers_per_chain = int(metadata.get("stickers_per_chain", 0))
    if n_chains <= 0 or chain_length <= 0 or stickers_per_chain <= 0:
        return np.empty((0,), dtype=np.int32)

    segment = chain_length / stickers_per_chain
    offsets = np.rint((np.arange(stickers_per_chain) + 0.5) * segment).astype(np.int32)
    offsets = np.clip(offsets, 0, chain_length - 1)
    if np.unique(offsets).size != offsets.size:
        offsets = np.rint(
            np.linspace(0, chain_length - 1, stickers_per_chain + 2)[1:-1]
        ).astype(np.int32)
        offsets = np.clip(offsets, 0, chain_length - 1)
    offsets = np.unique(offsets)
    if offsets.size != stickers_per_chain:
        raise RuntimeError(
            "Could not reconstruct sticker tags from metadata; non-unique offsets detected."
        )

    chain_starts = np.arange(n_chains, dtype=np.int32) * chain_length
    return (chain_starts[:, None] + offsets[None, :]).reshape(-1)


def infer_sample_dt(sampled_steps: np.ndarray, dt: float, default_step_stride: float) -> float:
    steps = np.asarray(sampled_steps, dtype=np.float64)
    if steps.size >= 2:
        diffs = np.diff(steps)
        positive = diffs[diffs > 0.0]
        if positive.size:
            return float(np.mean(positive) * dt)
    return float(default_step_stride * dt)


def split_contiguous_segments(
    sampled_steps: np.ndarray,
    expected_step_delta: int,
) -> List[Tuple[int, int]]:
    if sampled_steps.size == 0:
        return []
    if sampled_steps.size == 1:
        return [(0, 1)]

    diffs = np.diff(np.asarray(sampled_steps, dtype=np.int64))
    break_after = np.flatnonzero(diffs != int(expected_step_delta))
    starts = np.concatenate((np.array([0], dtype=np.int64), break_after + 1))
    ends = np.concatenate(
        (break_after.astype(np.int64) + 1, np.array([sampled_steps.size], dtype=np.int64))
    )
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def segment_bound_arrays(
    segments: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    usable = [
        (int(start), int(end))
        for start, end in segments
        if int(end) - int(start) >= 2
    ]
    if not usable:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    return (
        np.asarray([start for start, _end in usable], dtype=np.int64),
        np.asarray([end for _start, end in usable], dtype=np.int64),
    )


def build_lag_time_axis(sampled_steps: np.ndarray, dt: float, lag_count: int) -> np.ndarray:
    if lag_count <= 0:
        return np.empty((0,), dtype=np.float64)
    steps = np.asarray(sampled_steps, dtype=np.float64)
    if steps.size < 2:
        return np.arange(1, lag_count + 1, dtype=np.float64) * dt

    diffs = np.diff(steps)
    positive = diffs[diffs > 0.0]
    if positive.size and np.allclose(positive, positive[0], rtol=0.0, atol=1.0e-12):
        return np.arange(1, lag_count + 1, dtype=np.float64) * float(positive[0] * dt)

    lag_times = np.empty(lag_count, dtype=np.float64)
    for lag in range(1, lag_count + 1):
        lag_step = steps[lag:] - steps[:-lag]
        positive_lag = lag_step[lag_step > 0.0]
        lag_times[lag - 1] = (
            float(np.mean(positive_lag) * dt)
            if positive_lag.size
            else float(lag * infer_sample_dt(steps, dt, 1.0))
        )
    return lag_times


def pack_open_mask(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(mask.astype(np.uint8, copy=False), bitorder="little")
    remainder = packed.size % 8
    if remainder != 0:
        packed = np.pad(packed, (0, 8 - remainder), mode="constant")
    return np.ascontiguousarray(packed.view(np.uint64))


@numba.njit(cache=True)
def popcount_uint64(value: np.uint64) -> int:
    count = 0
    while value != np.uint64(0):
        value &= value - np.uint64(1)
        count += 1
    return count


@numba.njit(cache=True)
def bitset_intersection_count(left: np.ndarray, right: np.ndarray) -> int:
    total = 0
    for idx in range(left.shape[0]):
        total += popcount_uint64(left[idx] & right[idx])
    return total


@numba.njit(cache=True)
def sorted_intersection_count(
    bonds_flat: np.ndarray,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    count = 0
    i = left_start
    j = right_start
    while i < left_end and j < right_end:
        left = bonds_flat[i]
        right = bonds_flat[j]
        if left == right:
            count += 1
            i += 1
            j += 1
        elif left < right:
            i += 1
        else:
            j += 1
    return count


@numba.njit(cache=True)
def compute_subsampled_bond_correlation(
    bonds_flat: np.ndarray,
    bond_offsets: np.ndarray,
    sample_indices: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    max_lag_frames: int,
) -> np.ndarray:
    max_segment_length = 0
    for segment_idx in range(segment_starts.shape[0]):
        segment_length = segment_ends[segment_idx] - segment_starts[segment_idx]
        if segment_length > max_segment_length:
            max_segment_length = segment_length
    lag_count = min(max_lag_frames, max_segment_length - 1)
    corr = np.zeros(lag_count, dtype=np.float64)
    if lag_count <= 0:
        return corr

    numerators = np.zeros(lag_count, dtype=np.float64)
    denominators = np.zeros(lag_count, dtype=np.float64)

    for segment_idx in range(segment_starts.shape[0]):
        segment_start = segment_starts[segment_idx]
        segment_end = segment_ends[segment_idx]
        for pos in range(segment_start + 1, segment_end):
            current_frame = sample_indices[pos]
            current_start = bond_offsets[current_frame]
            current_end = bond_offsets[current_frame + 1]
            max_back = min(lag_count, pos - segment_start)
            for lag in range(1, max_back + 1):
                previous_frame = sample_indices[pos - lag]
                previous_start = bond_offsets[previous_frame]
                previous_end = bond_offsets[previous_frame + 1]
                previous_count = previous_end - previous_start
                if previous_count <= 0:
                    continue
                denominators[lag - 1] += previous_count
                numerators[lag - 1] += sorted_intersection_count(
                    bonds_flat,
                    previous_start,
                    previous_end,
                    current_start,
                    current_end,
                )

    for idx in range(lag_count):
        if denominators[idx] > 0.0:
            corr[idx] = numerators[idx] / denominators[idx]
    return corr


@numba.njit(cache=True)
def compute_subsampled_open_correlation(
    open_bitsets: np.ndarray,
    open_counts: np.ndarray,
    sample_indices: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    max_lag_frames: int,
) -> np.ndarray:
    max_segment_length = 0
    for segment_idx in range(segment_starts.shape[0]):
        segment_length = segment_ends[segment_idx] - segment_starts[segment_idx]
        if segment_length > max_segment_length:
            max_segment_length = segment_length
    lag_count = min(max_lag_frames, max_segment_length - 1)
    corr = np.zeros(lag_count, dtype=np.float64)
    if lag_count <= 0:
        return corr

    numerators = np.zeros(lag_count, dtype=np.float64)
    denominators = np.zeros(lag_count, dtype=np.float64)

    for segment_idx in range(segment_starts.shape[0]):
        segment_start = segment_starts[segment_idx]
        segment_end = segment_ends[segment_idx]
        for pos in range(segment_start + 1, segment_end):
            current_frame = sample_indices[pos]
            max_back = min(lag_count, pos - segment_start)
            for lag in range(1, max_back + 1):
                previous_frame = sample_indices[pos - lag]
                previous_count = open_counts[previous_frame]
                if previous_count <= 0:
                    continue
                denominators[lag - 1] += previous_count
                numerators[lag - 1] += bitset_intersection_count(
                    open_bitsets[previous_frame],
                    open_bitsets[current_frame],
                )

    for idx in range(lag_count):
        if denominators[idx] > 0.0:
            corr[idx] = numerators[idx] / denominators[idx]
    return corr


def unwrap_positions(frame) -> np.ndarray:
    positions = np.asarray(frame.particles.position, dtype=np.float64)
    images = getattr(frame.particles, "image", None)
    if images is None:
        return positions
    box_length = float(frame.configuration.box[0])
    return positions + np.asarray(images, dtype=np.float64) * box_length


def load_msd_positions(
    msd_gsd_path: str,
    analysis_stride: int,
    progress_label: str | None = None,
) -> Tuple[np.ndarray | None, np.ndarray]:
    with gsd.hoomd.open(msd_gsd_path, "r") as traj:
        frame_indices = checkpoint_deduplicated_frame_indices(traj, msd_gsd_path)
        n_frames = int(frame_indices.size)
        if n_frames < 2:
            return None, np.empty((0,), dtype=np.int64)

        sampled_frame_indices = frame_indices[::analysis_stride]
        n_analyzed = int(sampled_frame_indices.size)
        progress_interval = max(1, n_analyzed // 10)
        positions: List[np.ndarray] = []
        steps: List[int] = []
        for analyzed_idx, physical_frame_idx in enumerate(sampled_frame_indices, start=1):
            frame = traj[int(physical_frame_idx)]
            positions.append(unwrap_positions(frame).astype(np.float32, copy=False))
            steps.append(frame_step_or_index(frame, int(physical_frame_idx)))
            if progress_label is not None and (
                analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
            ):
                progress_pct = 100.0 * analyzed_idx / n_analyzed
                log(
                    f"{progress_label}: MSD frame progress {analyzed_idx}/{n_analyzed} "
                    f"({progress_pct:.1f}%)"
                )

    if len(positions) < 2:
        return None, np.empty((0,), dtype=np.int64)
    return np.stack(positions, axis=0), np.asarray(steps, dtype=np.int64)


def compute_msd(
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
    requested_lag = n_frames - 1 if max_lag_frames <= 0 else min(max_lag_frames, n_frames - 1)
    max_lag = max(1, min(requested_lag, runtime_lag))

    pos = np.asarray(positions, dtype=np.float32)
    n_particles = int(pos.shape[1])
    msd_values = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        diff = pos[lag:] - pos[:-lag]
        total_sq = np.einsum("ijk,ijk->", diff, diff, dtype=np.float64)
        msd_values[lag - 1] = total_sq / float((n_frames - lag) * n_particles)

    msd_time = np.arange(1, max_lag + 1, dtype=np.float64) * float(sample_dt)
    return msd_time, msd_values


def compute_segmented_msd(
    positions: np.ndarray,
    sample_dt: float,
    max_lag_frames: int,
    segments: List[Tuple[int, int]],
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if positions.ndim != 3 or positions.shape[0] < 2:
        return None, None

    usable_segments = [
        (int(start), int(end))
        for start, end in segments
        if int(end) - int(start) >= 2
    ]
    if not usable_segments:
        return None, None

    longest_segment = max(end - start for start, end in usable_segments)
    runtime_lag = int(np.floor(MAX_ANALYSIS_LAG_TIME / float(sample_dt) + 1.0e-12))
    runtime_lag = max(1, min(runtime_lag, longest_segment - 1))
    requested_lag = (
        longest_segment - 1
        if max_lag_frames <= 0
        else min(max_lag_frames, longest_segment - 1)
    )
    max_lag = max(1, min(requested_lag, runtime_lag))

    pos = np.asarray(positions, dtype=np.float32)
    n_particles = int(pos.shape[1])
    msd_values = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        total_sq = 0.0
        pair_count = 0
        for start, end in usable_segments:
            if end - start <= lag:
                continue
            diff = pos[start + lag : end] - pos[start : end - lag]
            total_sq += float(np.einsum("ijk,ijk->", diff, diff, dtype=np.float64))
            pair_count += (end - start - lag) * n_particles
        msd_values[lag - 1] = total_sq / float(pair_count) if pair_count > 0 else float("nan")

    msd_time = np.arange(1, max_lag + 1, dtype=np.float64) * float(sample_dt)
    return msd_time, msd_values


def estimate_diffusion_coefficient(time: np.ndarray | None, msd: np.ndarray | None) -> float:
    if time is None or msd is None:
        return float("nan")
    n = min(len(time), len(msd))
    if n < 4:
        return float("nan")
    t = np.asarray(time[:n], dtype=np.float64)
    y = np.asarray(msd[:n], dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0.0)
    t = t[mask]
    y = y[mask]
    if t.size < 4:
        return float("nan")
    start = t.size // 2
    slope, _ = np.polyfit(t[start:], y[start:], 1)
    return float(slope / 6.0) if np.isfinite(slope) and slope > 0.0 else float("nan")


def compute_clump_fraction(degrees: np.ndarray, bond_i: np.ndarray, bond_j: np.ndarray) -> float:
    if degrees.size == 0:
        return float("nan")
    high_degree = degrees >= 2
    clumped = high_degree.copy()
    if bond_i.size:
        adjacent_to_high = high_degree[bond_i] | high_degree[bond_j]
        if np.any(adjacent_to_high):
            clumped[bond_i[adjacent_to_high]] = True
            clumped[bond_j[adjacent_to_high]] = True
    return float(np.count_nonzero(clumped) / degrees.size)


def analyze_replicate(
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    msd_max_lag_frames: int,
    progress_label: str | None = None,
) -> Dict:
    with gsd.hoomd.open(gsd_path, "r") as traj:
        frame_indices = checkpoint_deduplicated_frame_indices(traj, gsd_path)
        n_frames = int(frame_indices.size)
        if n_frames == 0:
            raise RuntimeError(f"No frames found in {gsd_path}")

        analysis_sample_indices = np.arange(0, n_frames, analysis_stride, dtype=np.int64)
        n_analyzed = int(analysis_sample_indices.size)
        progress_interval = max(1, n_analyzed // 10)

        first = traj[int(frame_indices[0])]
        trajectory_subset = str(metadata.get("trajectory_particle_subset", ""))
        if trajectory_subset != "sticky_only":
            raise RuntimeError(
                f"Expected sticker-only trajectory metadata for {gsd_path}, "
                f"found trajectory_particle_subset={trajectory_subset!r}."
            )

        type_names = first.particles.types
        if "sticky" not in type_names:
            raise RuntimeError("Sticker type 'sticky' not found in trajectory.")
        sticker_type = type_names.index("sticky")
        typeid = np.asarray(first.particles.typeid, dtype=np.int32)
        if not np.all(typeid == sticker_type):
            raise RuntimeError(f"Sticker-only trajectory {gsd_path} contains non-sticky particles.")

        chain_length = int(metadata.get("chain_length", 1))
        n_chains = int(metadata.get("n_chains", 0))
        expected_sticker_tags = sticker_tags_from_metadata(metadata)
        if expected_sticker_tags.size != first.particles.N:
            raise RuntimeError(
                f"Sticker-only trajectory size mismatch in {gsd_path}: "
                f"expected {expected_sticker_tags.size}, got {first.particles.N}."
            )
        n_stickers = int(first.particles.N)
        stickers_per_chain = float(metadata.get("stickers_per_chain", 4))
        p_c = (
            float(1.0 / (stickers_per_chain - 1.0))
            if stickers_per_chain > 1.0
            else float("nan")
        )
        sticker_chain_ids = (expected_sticker_tags // chain_length).astype(np.int32, copy=False)
        bond_code_scale = np.int64(max(n_stickers, 1))

        box_length = float(first.configuration.box[0])
        reactive_sigma = float(metadata.get("reactive_sigma", 1.0))
        r_thresh = float(compute_r_thresh(reactive_sigma))

        dt = float(metadata.get("dt", 0.005))
        frame_steps_default = int(
            metadata.get("trajectory_frame_steps", metadata.get("frame_steps", 10_000))
        )

        p_open_series = np.empty((n_analyzed,), dtype=np.float64)
        bonding_probability_series = np.empty((n_analyzed,), dtype=np.float64)
        gelation_epsilon_series = np.empty((n_analyzed,), dtype=np.float64)
        bonded_pair_count_series = np.empty((n_analyzed,), dtype=np.float64)
        frac_bond0_series = np.empty((n_analyzed,), dtype=np.float64)
        frac_bond1_series = np.empty((n_analyzed,), dtype=np.float64)
        frac_bond_gt1_series = np.empty((n_analyzed,), dtype=np.float64)
        clump_fraction_series = np.empty((n_analyzed,), dtype=np.float64)
        largest_cluster_timestep_series = np.empty((n_analyzed,), dtype=np.float64)
        largest_cluster_size_series = np.empty((n_analyzed,), dtype=np.float64)

        cluster_hist = np.zeros(n_chains + 1, dtype=np.float64)
        cluster_count_total = 0
        number_average_cluster_size_sum = 0.0
        weight_average_cluster_size_sum = 0.0
        largest_cluster_size_sum = 0.0
        largest_cluster_fraction_sum = 0.0
        cluster_frames = 0

        intra_bond_total = 0
        inter_bond_total = 0
        bond_frames = 0

        assoc_events_total = 0
        passive_events_total = 0
        transition_time_total = 0.0
        free_sticker_time = 0.0

        n_open_words = (n_stickers + 63) // 64
        open_bitsets = np.zeros((n_frames, n_open_words), dtype=np.uint64)
        open_counts = np.zeros((n_frames,), dtype=np.int32)
        bond_chunks: List[np.ndarray] = []
        bond_offsets = np.zeros((n_frames + 1,), dtype=np.int64)
        frame_step_numbers = np.zeros((n_frames,), dtype=np.int64)

        prev_bond_ids: np.ndarray | None = None
        prev_open_count: int | None = None
        prev_bonded_mask: np.ndarray | None = None
        prev_frame_step: int | None = None

        analyzed_idx = 0
        for frame_idx, physical_frame_idx in enumerate(frame_indices):
            frame = traj[int(physical_frame_idx)]
            positions = np.asarray(frame.particles.position, dtype=np.float32)
            pair_i, pair_j, pair_dist = compute_sticker_neighbor_pairs(
                positions, box_length, r_thresh
            )

            degrees = np.zeros(n_stickers, dtype=np.int32)
            bond_i = np.empty((0,), dtype=np.int64)
            bond_j = np.empty((0,), dtype=np.int64)
            if pair_i.size > 0:
                bond_mask = pair_dist < r_thresh
                bond_i = pair_i[bond_mask]
                bond_j = pair_j[bond_mask]
                if bond_i.size > 0:
                    np.add.at(degrees, bond_i, 1)
                    np.add.at(degrees, bond_j, 1)

            frame_step = frame_step_or_index(frame, int(frame_idx * frame_steps_default))

            if bond_i.size > 0:
                low = np.minimum(bond_i, bond_j).astype(np.uint32, copy=False)
                high = np.maximum(bond_i, bond_j).astype(np.uint32, copy=False)
                bond_ids = np.ascontiguousarray(np.sort(low * np.uint32(n_stickers) + high))
            else:
                bond_ids = np.empty((0,), dtype=np.uint32)

            open_mask = degrees == 0
            open_count = int(np.count_nonzero(open_mask))
            bonded_mask = degrees > 0
            bond_chunks.append(bond_ids)
            bond_offsets[frame_idx + 1] = bond_offsets[frame_idx] + bond_ids.size
            open_bitsets[frame_idx] = pack_open_mask(open_mask)
            open_counts[frame_idx] = open_count
            frame_step_numbers[frame_idx] = frame_step

            if (
                prev_bond_ids is not None
                and prev_open_count is not None
                and prev_bonded_mask is not None
                and prev_frame_step is not None
            ):
                step_delta = frame_step - prev_frame_step
                if step_delta == frame_steps_default:
                    delta_t = dt * float(step_delta)
                    new_codes = np.setdiff1d(bond_ids, prev_bond_ids, assume_unique=True)
                    assoc = 0
                    passive = 0
                    if new_codes.size > 0:
                        new_i = (new_codes.astype(np.int64, copy=False) // bond_code_scale).astype(
                            np.int64, copy=False
                        )
                        new_j = (new_codes.astype(np.int64, copy=False) % bond_code_scale).astype(
                            np.int64, copy=False
                        )
                        assoc_mask = prev_bonded_mask[new_i] | prev_bonded_mask[new_j]
                        assoc = int(np.count_nonzero(assoc_mask))
                        passive = int(new_codes.size - assoc)
                    assoc_events_total += assoc
                    passive_events_total += passive
                    transition_time_total += delta_t
                    free_sticker_time += prev_open_count * delta_t

            prev_bond_ids = bond_ids
            prev_open_count = open_count
            prev_bonded_mask = bonded_mask
            prev_frame_step = frame_step

            if frame_idx % analysis_stride != 0:
                continue

            p_open = open_count / n_stickers
            bonding_probability = 1.0 - p_open
            p_open_series[analyzed_idx] = p_open
            bonding_probability_series[analyzed_idx] = bonding_probability
            gelation_epsilon_series[analyzed_idx] = (
                (bonding_probability - p_c) / p_c if np.isfinite(p_c) else float("nan")
            )
            bonded_pair_count_series[analyzed_idx] = float(bond_i.size)

            count1 = int(np.count_nonzero(degrees == 1))
            count_gt1 = n_stickers - open_count - count1
            frac_bond0_series[analyzed_idx] = p_open
            frac_bond1_series[analyzed_idx] = count1 / n_stickers
            frac_bond_gt1_series[analyzed_idx] = count_gt1 / n_stickers
            clump_fraction_series[analyzed_idx] = compute_clump_fraction(degrees, bond_i, bond_j)

            intra = 0
            inter = 0
            chain_pair_codes = np.empty((0,), dtype=np.int64)
            if bond_i.size > 0:
                chain_i = sticker_chain_ids[bond_i]
                chain_j = sticker_chain_ids[bond_j]
                same_chain = chain_i == chain_j
                intra = int(np.count_nonzero(same_chain))
                inter = int(bond_i.size - intra)
                inter_mask = ~same_chain
                if np.any(inter_mask):
                    chain_low = np.minimum(chain_i[inter_mask], chain_j[inter_mask]).astype(
                        np.int64, copy=False
                    )
                    chain_high = np.maximum(chain_i[inter_mask], chain_j[inter_mask]).astype(
                        np.int64, copy=False
                    )
                    chain_pair_codes = np.unique(chain_low * n_chains + chain_high)

            if chain_pair_codes.size == 0:
                sizes = np.ones(n_chains, dtype=np.int32)
            else:
                uf = UnionFind(n_chains)
                chain_src = (chain_pair_codes // n_chains).astype(np.int32, copy=False)
                chain_dst = (chain_pair_codes % n_chains).astype(np.int32, copy=False)
                for chain_a, chain_b in zip(chain_src.tolist(), chain_dst.tolist()):
                    uf.union(chain_a, chain_b)
                sizes = uf.cluster_sizes()

            cluster_hist += np.bincount(sizes, minlength=n_chains + 1).astype(np.float64)
            cluster_count_total += int(sizes.size)
            largest_cluster_size = float(np.max(sizes))
            number_average_cluster_size_sum += float(np.mean(sizes))
            weight_average_cluster_size_sum += float(np.sum(sizes.astype(np.float64) ** 2) / n_chains)
            largest_cluster_size_sum += largest_cluster_size
            largest_cluster_fraction_sum += largest_cluster_size / n_chains
            cluster_frames += 1
            largest_cluster_timestep_series[analyzed_idx] = float(frame_step)
            largest_cluster_size_series[analyzed_idx] = largest_cluster_size

            intra_bond_total += intra
            inter_bond_total += inter
            bond_frames += 1

            analyzed_idx += 1
            if progress_label is not None and (
                analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
            ):
                progress_pct = 100.0 * analyzed_idx / n_analyzed
                log(
                    f"{progress_label}: frame progress {analyzed_idx}/{n_analyzed} "
                    f"({progress_pct:.1f}%)"
                )

        if analyzed_idx != n_analyzed:
            raise RuntimeError(
                f"Analyzed-frame count mismatch for {gsd_path}: expected {n_analyzed}, got {analyzed_idx}."
            )

        bonds_flat = (
            np.ascontiguousarray(np.concatenate(bond_chunks))
            if bond_offsets[-1] > 0
            else np.empty((0,), dtype=np.uint32)
        )
        sampled_steps = frame_step_numbers[analysis_sample_indices]
        sample_step_delta = int(frame_steps_default * analysis_stride)
        sample_dt = dt * float(sample_step_delta)
        sample_segments = split_contiguous_segments(sampled_steps, sample_step_delta)
        segment_starts, segment_ends = segment_bound_arrays(sample_segments)
        longest_segment_samples = (
            int(np.max(segment_ends - segment_starts))
            if segment_starts.size
            else 0
        )
        physical_lag_frames = (
            max(1, int(np.floor(MAX_ANALYSIS_LAG_TIME / sample_dt + 1.0e-12)))
            if sample_dt > 0.0
            else n_analyzed - 1
        )
        requested_lag_frames = (
            physical_lag_frames
            if max_lag_frames <= 0
            else min(max_lag_frames, physical_lag_frames)
        )
        lag_count = min(requested_lag_frames, max(0, longest_segment_samples - 1))
        if lag_count > 0:
            lag_time = np.arange(1, lag_count + 1, dtype=np.float64) * sample_dt
            cs = compute_subsampled_bond_correlation(
                bonds_flat,
                bond_offsets,
                analysis_sample_indices,
                segment_starts,
                segment_ends,
                lag_count,
            )
            cb = compute_subsampled_open_correlation(
                open_bitsets,
                open_counts,
                analysis_sample_indices,
                segment_starts,
                segment_ends,
                lag_count,
            )
            cs_time = lag_time[: len(cs)]
            cb_time = lag_time[: len(cb)]
        else:
            cs = np.empty((0,), dtype=np.float64)
            cb = np.empty((0,), dtype=np.float64)
            cs_time = np.empty((0,), dtype=np.float64)
            cb_time = np.empty((0,), dtype=np.float64)

        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)
        # Brachiation time: decay time of the open-sticker correlation C_b(t),
        # matching the melt analysis' operational definition.
        tau_b = fit_exponential_semilog_linear_region(cb_time, cb)

    msd_time = None
    msd = None
    diffusion_coefficient = float("nan")
    msd_path = os.path.join(
        os.path.dirname(gsd_path), str(metadata.get("msd_trajectory_file", "msd_trajectory.gsd"))
    )
    if os.path.exists(msd_path):
        msd_positions, msd_steps = load_msd_positions(
            msd_path,
            analysis_stride,
            progress_label=f"{progress_label} [MSD]" if progress_label is not None else None,
        )
        if msd_positions is not None:
            msd_frame_steps = int(metadata.get("msd_frame_steps", metadata.get("frame_steps", 10_000)))
            msd_step_delta = int(msd_frame_steps * analysis_stride)
            msd_dt = float(metadata.get("dt", 0.005)) * float(msd_step_delta)
            msd_segments = split_contiguous_segments(msd_steps, msd_step_delta)
            msd_time, msd = compute_segmented_msd(
                msd_positions,
                msd_dt,
                msd_max_lag_frames,
                msd_segments,
            )
            diffusion_coefficient = estimate_diffusion_coefficient(msd_time, msd)

    associative_rate_per_free = (
        float(assoc_events_total / free_sticker_time) if free_sticker_time > 0.0 else float("nan")
    )
    passive_rate_per_free = (
        float(passive_events_total / free_sticker_time) if free_sticker_time > 0.0 else float("nan")
    )
    associative_events_per_time = (
        float(assoc_events_total / transition_time_total)
        if transition_time_total > 0.0
        else float("nan")
    )
    passive_events_per_time = (
        float(passive_events_total / transition_time_total)
        if transition_time_total > 0.0
        else float("nan")
    )
    finite_gelation_epsilon = gelation_epsilon_series[np.isfinite(gelation_epsilon_series)]

    return {
        "bonded_pair_count_mean": float(np.mean(bonded_pair_count_series)),
        "p_open_mean": float(np.mean(p_open_series)),
        "bonding_probability_mean": float(np.mean(bonding_probability_series)),
        "gelation_epsilon_mean": (
            float(np.mean(finite_gelation_epsilon))
            if finite_gelation_epsilon.size
            else float("nan")
        ),
        "stickers_per_chain": stickers_per_chain,
        "p_c": p_c,
        "frac_bond0_mean": float(np.mean(frac_bond0_series)),
        "frac_bond1_mean": float(np.mean(frac_bond1_series)),
        "frac_bond_gt1_mean": float(np.mean(frac_bond_gt1_series)),
        "clump_fraction_mean": float(np.mean(clump_fraction_series)),
        "intra_bonds_mean": intra_bond_total / max(bond_frames, 1),
        "inter_bonds_mean": inter_bond_total / max(bond_frames, 1),
        "intra_inter_ratio": intra_bond_total / inter_bond_total if inter_bond_total > 0 else float("nan"),
        "number_average_cluster_size": number_average_cluster_size_sum / max(cluster_frames, 1),
        "weight_average_cluster_size": weight_average_cluster_size_sum / max(cluster_frames, 1),
        "largest_cluster_size": largest_cluster_size_sum / max(cluster_frames, 1),
        "largest_cluster_fraction": largest_cluster_fraction_sum / max(cluster_frames, 1),
        "rate_assoc": associative_rate_per_free,
        "rate_passive": passive_rate_per_free,
        "assoc_events_per_time": associative_events_per_time,
        "passive_events_per_time": passive_events_per_time,
        "tau_s": tau_s,
        "tau_b": tau_b,
        "diffusion_coefficient": diffusion_coefficient,
        "cluster_hist": cluster_hist,
        "cluster_count_total": cluster_count_total,
        "largest_cluster_timestep": largest_cluster_timestep_series,
        "largest_cluster_size_series": largest_cluster_size_series,
        "cs_time": cs_time,
        "cs": cs,
        "cb_time": cb_time,
        "cb": cb,
        "msd_time": msd_time,
        "msd": msd,
        "bonded_pair_count_series": bonded_pair_count_series,
        "frac_bond0_series": frac_bond0_series,
        "frac_bond1_series": frac_bond1_series,
        "frac_bond_gt1_series": frac_bond_gt1_series,
        "clump_fraction_series": clump_fraction_series,
    }


def summarize_finite_values(values: List[float]) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "stderr": float("nan"), "n_finite": 0}
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "stderr": stderr, "n_finite": int(arr.size)}


def write_properties_csv(path: str, properties: Dict[str, Dict[str, float | int]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("property,mean,stderr,n_finite\n")
        for name, stats in properties.items():
            handle.write(
                f"{name},{stats['mean']},{stats['stderr']},{int(stats.get('n_finite', 0))}\n"
            )


def write_timeseries(
    path: str,
    time: np.ndarray,
    mean: np.ndarray,
    stderr: np.ndarray,
    time_label: str = "time",
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{time_label},mean,stderr\n")
        for t, m, s in zip(time, mean, stderr):
            handle.write(f"{t:.6e},{m:.6e},{s:.6e}\n")


def aggregate_timeseries(
    replicate_results: List[Dict],
    key_time: str,
    key_val: str,
    epsilon: float,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    series = [
        (res.get(key_time), res.get(key_val))
        for res in replicate_results
        if res.get(key_time) is not None and res.get(key_val) is not None
    ]
    if not series:
        return None, None, None, None

    n = min(min(len(t) for t, _ in series), min(len(v) for _, v in series))
    if n == 0:
        return None, None, None, None
    time = np.asarray(series[0][0][:n], dtype=np.float64)
    for series_idx, (candidate_time, _) in enumerate(series[1:], start=2):
        candidate = np.asarray(candidate_time[:n], dtype=np.float64)
        if not np.allclose(candidate, time, rtol=TIME_AXIS_MATCH_RTOL, atol=TIME_AXIS_MATCH_ATOL):
            max_abs_diff = float(np.max(np.abs(candidate - time))) if n > 0 else 0.0
            raise RuntimeError(
                f"Inconsistent time axis for {key_val} in eps={epsilon:g}: "
                f"replicate 1 vs replicate {series_idx} differ "
                f"(max_abs_diff={max_abs_diff:.3e})."
            )
    values = np.stack([np.asarray(v[:n], dtype=np.float64) for _, v in series], axis=0)
    mean = np.mean(values, axis=0)
    stderr = (
        np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        if values.shape[0] > 1
        else np.zeros_like(mean)
    )
    return time, values, mean, stderr


def aggregate_cluster_distribution(
    replicate_results: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cluster_hists = [np.asarray(res["cluster_hist"], dtype=np.float64) for res in replicate_results]
    cluster_counts = [float(res["cluster_count_total"]) for res in replicate_results]
    max_len = max(len(hist) for hist in cluster_hists)
    distributions = []
    for hist, count in zip(cluster_hists, cluster_counts):
        padded = np.zeros(max_len, dtype=np.float64)
        padded[: len(hist)] = hist
        distributions.append(padded / max(count, 1.0))
    values = np.vstack(distributions)
    mean = np.mean(values, axis=0)
    stderr = (
        np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        if values.shape[0] > 1
        else np.zeros_like(mean)
    )
    cluster_size = np.arange(max_len, dtype=np.int64)
    return cluster_size, mean, stderr


def finite_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def style_axes(ax) -> None:
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2)


def write_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    y_label: str,
    ylim: Tuple[float, float] | None = None,
    *,
    figsize: Tuple[float, float] = (3.3, 3.3),
    dpi: int = PLOT_DPI,
    x_label: str = r"$\varepsilon_\mathrm{reactiveLJ}$",
    tick_label_size: float | None = TICK_FONTSIZE,
    axis_label_size: float | None = LABEL_FONTSIZE,
    body_facecolor: str = "#e77500",
    body_edgecolor: str = "#121212",
    body_alpha: float = 0.45,
    median_color: str = "#121212",
    iqr_lw: float = 1.7,
    median_lw: float = 1.4,
    median_marker_size: float = 14.0,
) -> None:
    processed: List[Tuple[float, np.ndarray]] = []
    for eps, values in zip(epsilon_values, data):
        arr = finite_array(values)
        if arr.size == 0:
            continue
        processed.append((float(eps), arr))
    if not processed:
        return

    fig, ax = plt.subplots(figsize=figsize)
    violin_positions = [eps for eps, values in processed if values.size > 1]
    violin_data = [values for _, values in processed if values.size > 1]
    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions, widths=0.6, showextrema=False)
        for body in parts.get("bodies", []):
            body.set_facecolor(body_facecolor)
            body.set_edgecolor(body_edgecolor)
            body.set_alpha(body_alpha)

    positions: List[float] = []
    medians = []
    for eps, values in processed:
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color=median_color, lw=iqr_lw)
        ax.scatter([eps], [med], color=median_color, s=median_marker_size, zorder=3)
        positions.append(eps)
        medians.append(med)
    ax.plot(positions, medians, color=median_color, lw=median_lw, alpha=0.9)
    if axis_label_size is None:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        ax.set_xlabel(x_label, fontsize=axis_label_size)
        ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes(ax)
    if tick_label_size is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_median_iqr_line_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    y_label: str,
    ylim: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (3.3, 3.3),
    dpi: int = PLOT_DPI,
    yscale: str = "linear",
    show_iqr: bool = True,
) -> None:
    valid_eps: List[float] = []
    medians: List[float] = []
    q1s: List[float] = []
    q3s: List[float] = []
    for eps, values in zip(epsilon_values, data):
        arr = finite_array(values)
        if yscale == "log":
            arr = arr[arr > 0.0]
        if arr.size == 0:
            continue
        valid_eps.append(float(eps))
        medians.append(float(np.median(arr)))
        q1s.append(float(np.percentile(arr, 25.0)))
        q3s.append(float(np.percentile(arr, 75.0)))
    if not valid_eps:
        return

    x = np.asarray(valid_eps, dtype=np.float64)
    fig, ax = plt.subplots(figsize=figsize)
    if show_iqr:
        ax.fill_between(x, q1s, q3s, color="#9e9e9e", alpha=0.35)
    ax.plot(x, medians, color="#121212", lw=1.7, marker="o", ms=3.5)
    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    if yscale != "linear":
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_dual_rate_plot(
    path: str,
    epsilon_values: List[float],
    assoc_data: List[np.ndarray],
    passive_data: List[np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for data, color, label, offset in (
        (assoc_data, "#e77500", r"Associative exchange rate ($R_a$)", -0.16),
        (passive_data, "#121212", r"Passive dimerization rate ($R_d$)", 0.16),
    ):
        xs: List[float] = []
        medians: List[float] = []
        for eps, values in zip(epsilon_values, data):
            arr = finite_array(values)
            if arr.size == 0:
                continue
            xs.append(float(eps) + offset)
            medians.append(float(np.median(arr)))
        if not xs:
            continue
        ax.plot(xs, medians, color=color, lw=1.8, alpha=0.9, zorder=2, label=label)
        ax.scatter(xs, medians, color=color, s=18, zorder=3)
    ax.set_title("Associative Exchange vs Passive Dimerization Rates")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("rate (1/time)")
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_autocorr_fit_plot(
    path: str,
    time: np.ndarray,
    values: np.ndarray,
    y_label: str,
) -> None:
    if values.ndim != 2 or values.shape[1] == 0:
        return
    median = np.median(values, axis=0)
    q1 = np.percentile(values, 25.0, axis=0)
    q3 = np.percentile(values, 75.0, axis=0)
    tau_fit = fit_exponential_semilog_linear_region(time, median)
    fit_time, _ = extract_semilog_linear_region(time, median)

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, median, color="#121212", lw=1.7, label="median")
    if np.isfinite(tau_fit) and fit_time.size > 0:
        ax.plot(
            fit_time,
            np.exp(-fit_time / tau_fit),
            color="#e77500",
            lw=1.6,
            label=rf"$\tau={tau_fit:.3g}$",
        )
    ax.set_xlabel("time lag", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_cluster_distribution_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    cluster_distribution_by_eps: Dict[float, np.ndarray],
) -> None:
    def set_target_axes_position(ax) -> None:
        figure_width_pt = 237.6
        figure_height_pt = 144.0
        axes_left_pt = 51.541515
        axes_bottom_pt = 41.816
        axes_width_pt = 175.258485
        axes_height_pt = 88.344535
        ax.set_position(
            [
                axes_left_pt / figure_width_pt,
                axes_bottom_pt / figure_height_pt,
                axes_width_pt / figure_width_pt,
                axes_height_pt / figure_height_pt,
            ]
        )

    def format_epsilon_legend_label(epsilon: float) -> str:
        if np.isclose(float(epsilon), 0.0, rtol=0.0, atol=1.0e-12):
            return r"$\varepsilon_\mathrm{RLJ}=\mathrm{None}$"
        return rf"$\varepsilon_\mathrm{{RLJ}}={epsilon:g}$"

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
        figsize=(237.6 / 72.0, 144.0 / 72.0),
        dpi=1000,
        tight=False,
    )
    set_target_axes_position(ax)
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
            label=format_epsilon_legend_label(eps),
            linewidths=0.0,
        )
        max_x = max(max_x, float(np.max(cluster_size)))
        max_y = max(max_y, float(np.max(prob)))
        min_y = min(min_y, float(np.min(prob)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M", fontsize=10)
    ax.set_ylabel("P(M)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", labelsize=8)
    ax.set_xlim(left=1.0, right=max_x * 1.08)
    if np.isfinite(min_y) and min_y > 0.0:
        ax.set_ylim(bottom=min_y * 0.8, top=max_y * 1.2)
    ax.legend(frameon=False, fontsize=7, ncol=1)
    set_target_axes_position(ax)
    fig.savefig(path, format="svg")
    uplt.close(fig)


def write_msd_vs_time_lag_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    msd_time_by_eps: Dict[float, np.ndarray],
    msd_mean_by_eps: Dict[float, np.ndarray],
    tau_r0: float,
    x_limits: Tuple[float, float] | None = None,
) -> None:
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
    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    for idx, (eps, x, y) in enumerate(series):
        ax.plot(x, y, color=cmap(idx), lw=2.0, label=f"eps={eps:g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=10)
    ax.set_ylabel("MSD", fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=8)
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=1000)
    plt.close(fig)


def write_largest_cluster_vs_timestep_plot(
    path: str,
    epsilon_values: List[float],
    timestep_data: List[np.ndarray],
    size_data: List[np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    colors = plt.cm.Greys(np.linspace(0.35, 0.9, max(len(epsilon_values), 2)))
    any_series = False
    for idx, (eps, tvals, svals) in enumerate(zip(epsilon_values, timestep_data, size_data)):
        if tvals.size == 0 or svals.size == 0:
            continue
        n = min(len(tvals), len(svals))
        ax.plot(tvals[:n], svals[:n], color=colors[idx], lw=1.4, label=f"{eps:g}")
        any_series = True
    if not any_series:
        plt.close(fig)
        return
    ax.set_xlabel("timestep", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("largest cluster size (chains)", fontsize=LABEL_FONTSIZE)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, title=r"$\varepsilon$", title_fontsize=LEGEND_FONTSIZE)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def analyze_replicate_job(
    epsilon: float,
    gsd_path: str,
    metadata_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    msd_max_lag_frames: int,
    cache_root: str,
    rep_label: str,
    rel_path: str,
) -> Tuple[float, Dict]:
    cache_key = build_replicate_cache_key(
        gsd_path, metadata_path, analysis_stride, max_lag_frames, msd_max_lag_frames
    )
    cache_path = os.path.join(cache_root, f"{cache_key}.pkl")
    cached = load_cached_replicate_result(cache_path)
    if cached is not None:
        log(f"{rep_label}: cache hit ({rel_path})")
        return epsilon, cached

    log(f"{rep_label}: start ({rel_path})")
    result = analyze_replicate(
        gsd_path,
        metadata,
        analysis_stride,
        max_lag_frames,
        msd_max_lag_frames,
        progress_label=rep_label,
    )
    save_cached_replicate_result(cache_path, result)
    log(f"{rep_label}: done")
    return epsilon, result


def main() -> None:
    args = parse_args()
    if args.analysis_stride < 1:
        raise ValueError("--analysis-stride must be >= 1")
    if args.max_lag_frames < 0:
        raise ValueError("--max-lag-frames must be >= 0")
    if args.msd_max_lag_frames < 0:
        raise ValueError("--msd-max-lag-frames must be >= 0")

    log(f"Scanning trajectories under {args.input_root}")
    runs = discover_runs(args.input_root)
    if not runs:
        raise RuntimeError(f"No trajectories found under {args.input_root}")
    log(f"Discovered {len(runs)} trajectory/metadata pairs")

    grouped: Dict[float, List[Tuple[str, str, Dict]]] = defaultdict(list)
    tau_r0_reference = FALLBACK_TAU_R0
    for gsd_path, metadata_path in runs:
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"Missing metadata.json for {gsd_path}")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        epsilon = float(metadata["reactive_epsilon"])
        tau_r0 = float(metadata.get("tau_R0", float("nan")))
        if np.isfinite(tau_r0) and tau_r0 > 0.0:
            tau_r0_reference = tau_r0
        grouped[epsilon].append((gsd_path, metadata_path, metadata))
    log(f"Grouped runs into {len(grouped)} epsilon values")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_root = os.path.join(args.output_dir, ".replicate_cache")

    n_jobs_requested = int(os.environ.get("SLURM_CPUS_PER_TASK", max(1, os.cpu_count() or 1)))
    n_jobs_requested = max(1, n_jobs_requested)

    jobs = []
    sorted_groups = sorted(grouped.items())
    for eps_idx, (epsilon, entries) in enumerate(sorted_groups, start=1):
        log(
            f"Queueing epsilon group {eps_idx}/{len(sorted_groups)}: "
            f"eps={epsilon:g}, replicates={len(entries)}"
        )
        for rep_idx, (gsd_path, metadata_path, metadata) in enumerate(entries, start=1):
            rel_path = os.path.relpath(gsd_path, args.input_root)
            rep_label = f"eps={epsilon:g} rep={rep_idx}/{len(entries)}"
            jobs.append(
                delayed(analyze_replicate_job)(
                    epsilon,
                    gsd_path,
                    metadata_path,
                    metadata,
                    args.analysis_stride,
                    args.max_lag_frames,
                    args.msd_max_lag_frames,
                    cache_root,
                    rep_label,
                    rel_path,
                )
            )

    n_jobs = min(n_jobs_requested, max(1, len(jobs)))
    log(f"Starting parallel analysis with {n_jobs} workers on {len(jobs)} runs")
    results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    log(f"Completed parallel analysis for {len(results)} runs")

    grouped_results: Dict[float, List[Dict]] = defaultdict(list)
    for epsilon, result in results:
        grouped_results[epsilon].append(result)

    epsilon_values: List[float] = []
    summary_rows: List[Dict[str, float | int]] = []
    summary_json: Dict[str, Dict] = {}
    cluster_distribution_by_eps: Dict[float, np.ndarray] = {}

    scalar_keys = [
        "bonded_pair_count_mean",
        "p_open_mean",
        "bonding_probability_mean",
        "gelation_epsilon_mean",
        "stickers_per_chain",
        "p_c",
        "frac_bond0_mean",
        "frac_bond1_mean",
        "frac_bond_gt1_mean",
        "clump_fraction_mean",
        "intra_bonds_mean",
        "inter_bonds_mean",
        "intra_inter_ratio",
        "number_average_cluster_size",
        "weight_average_cluster_size",
        "largest_cluster_size",
        "largest_cluster_fraction",
        "rate_assoc",
        "rate_passive",
        "assoc_events_per_time",
        "passive_events_per_time",
        "tau_s",
        "tau_b",
        "diffusion_coefficient",
    ]

    scalar_violin_data: Dict[str, List[np.ndarray]] = {key: [] for key in scalar_keys}
    bonded_pair_data: List[np.ndarray] = []
    frac_bond0_data: List[np.ndarray] = []
    frac_bond1_data: List[np.ndarray] = []
    frac_bond_gt1_data: List[np.ndarray] = []
    clump_fraction_data: List[np.ndarray] = []
    tau_s_data: List[np.ndarray] = []
    tau_b_data: List[np.ndarray] = []
    assoc_rate_data: List[np.ndarray] = []
    passive_rate_data: List[np.ndarray] = []
    diffusion_data: List[np.ndarray] = []
    largest_cluster_timestep_data: List[np.ndarray] = []
    largest_cluster_size_mean_data: List[np.ndarray] = []
    msd_time_by_eps: Dict[float, np.ndarray] = {}
    msd_mean_by_eps: Dict[float, np.ndarray] = {}

    for eps_idx, (epsilon, replicate_results) in enumerate(sorted(grouped_results.items()), start=1):
        log(
            f"Aggregating epsilon group {eps_idx}/{len(grouped_results)}: "
            f"eps={epsilon:g}, replicates={len(replicate_results)}"
        )
        epsilon_values.append(epsilon)
        eps_dir = os.path.join(args.output_dir, f"eps_{epsilon:g}")
        os.makedirs(eps_dir, exist_ok=True)

        scalar_summary = {
            key: summarize_finite_values([float(res.get(key, float("nan"))) for res in replicate_results])
            for key in scalar_keys
        }
        for key in scalar_keys:
            scalar_violin_data[key].append(
                np.asarray([float(res.get(key, float("nan"))) for res in replicate_results], dtype=np.float64)
            )

        epsilon_properties = {
            "bonded_pair_count": scalar_summary["bonded_pair_count_mean"],
            "p_open": scalar_summary["p_open_mean"],
            "bonding_probability_p": scalar_summary["bonding_probability_mean"],
            "gelation_epsilon": scalar_summary["gelation_epsilon_mean"],
            "stickers_per_chain_f": scalar_summary["stickers_per_chain"],
            "gel_point_p_c": scalar_summary["p_c"],
            "fraction_stickers_bond0": scalar_summary["frac_bond0_mean"],
            "fraction_stickers_bond1": scalar_summary["frac_bond1_mean"],
            "fraction_stickers_bond_gt1": scalar_summary["frac_bond_gt1_mean"],
            "clump_fraction": scalar_summary["clump_fraction_mean"],
            "intra_bonds": scalar_summary["intra_bonds_mean"],
            "inter_bonds": scalar_summary["inter_bonds_mean"],
            "intra_to_inter_bond_ratio": scalar_summary["intra_inter_ratio"],
            "number_average_cluster_size": scalar_summary["number_average_cluster_size"],
            "weight_average_cluster_size": scalar_summary["weight_average_cluster_size"],
            "largest_cluster_size": scalar_summary["largest_cluster_size"],
            "largest_cluster_fraction": scalar_summary["largest_cluster_fraction"],
            "bond_persistence_time_tau_s": scalar_summary["tau_s"],
            "brachiation_time_tau_b": scalar_summary["tau_b"],
            "associative_exchange_rate_R_a": scalar_summary["rate_assoc"],
            "passive_dimerization_rate_R_d": scalar_summary["rate_passive"],
            "associative_exchange_events_per_time": scalar_summary["assoc_events_per_time"],
            "passive_dimerization_events_per_time": scalar_summary["passive_events_per_time"],
            "sampled_monomer_diffusion_coefficient": scalar_summary["diffusion_coefficient"],
        }

        with open(os.path.join(eps_dir, "properties.json"), "w", encoding="utf-8") as handle:
            json.dump(epsilon_properties, handle, indent=2)
        write_properties_csv(os.path.join(eps_dir, "properties.csv"), epsilon_properties)

        cluster_size, cluster_mean, cluster_stderr = aggregate_cluster_distribution(replicate_results)
        cluster_distribution_by_eps[epsilon] = cluster_mean
        with open(os.path.join(eps_dir, "cluster_distribution.csv"), "w", encoding="utf-8") as handle:
            handle.write("cluster_size,mean,stderr\n")
            for size, mean, stderr in zip(cluster_size, cluster_mean, cluster_stderr):
                handle.write(f"{int(size)},{mean:.6e},{stderr:.6e}\n")

        cs_time, cs_values, cs_mean, cs_stderr = aggregate_timeseries(
            replicate_results, "cs_time", "cs", epsilon
        )
        if cs_time is not None and cs_values is not None:
            write_timeseries(os.path.join(eps_dir, "bond_correlation.csv"), cs_time, cs_mean, cs_stderr)
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "bond_correlation_fit.png"),
                cs_time,
                cs_values,
                r"$C_s(t)$",
            )

        cb_time, cb_values, cb_mean, cb_stderr = aggregate_timeseries(
            replicate_results, "cb_time", "cb", epsilon
        )
        if cb_time is not None and cb_values is not None:
            write_timeseries(
                os.path.join(eps_dir, "open_sticker_correlation.csv"), cb_time, cb_mean, cb_stderr
            )
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "open_sticker_correlation_fit.png"),
                cb_time,
                cb_values,
                r"$C_b(t)$",
            )

        msd_time, _, msd_mean, msd_stderr = aggregate_timeseries(
            replicate_results, "msd_time", "msd", epsilon
        )
        if msd_time is not None and msd_mean is not None:
            msd_time_by_eps[epsilon] = msd_time
            msd_mean_by_eps[epsilon] = msd_mean
            write_timeseries(os.path.join(eps_dir, "sampled_monomer_msd.csv"), msd_time, msd_mean, msd_stderr)

        (
            largest_cluster_timestep,
            _,
            largest_cluster_size_mean,
            largest_cluster_size_stderr,
        ) = aggregate_timeseries(
            replicate_results,
            "largest_cluster_timestep",
            "largest_cluster_size_series",
            epsilon,
        )
        if largest_cluster_timestep is not None and largest_cluster_size_mean is not None:
            write_timeseries(
                os.path.join(eps_dir, "largest_cluster_size.csv"),
                largest_cluster_timestep,
                largest_cluster_size_mean,
                largest_cluster_size_stderr,
                time_label="timestep",
            )
            largest_cluster_timestep_data.append(largest_cluster_timestep)
            largest_cluster_size_mean_data.append(largest_cluster_size_mean)
        else:
            largest_cluster_timestep_data.append(np.empty((0,), dtype=np.float64))
            largest_cluster_size_mean_data.append(np.empty((0,), dtype=np.float64))

        bonded_pair_data.append(
            np.concatenate([np.asarray(res["bonded_pair_count_series"], dtype=np.float64) for res in replicate_results])
        )
        frac_bond0_data.append(
            np.concatenate([np.asarray(res["frac_bond0_series"], dtype=np.float64) for res in replicate_results])
        )
        frac_bond1_data.append(
            np.concatenate([np.asarray(res["frac_bond1_series"], dtype=np.float64) for res in replicate_results])
        )
        frac_bond_gt1_data.append(
            np.concatenate([np.asarray(res["frac_bond_gt1_series"], dtype=np.float64) for res in replicate_results])
        )
        clump_fraction_data.append(
            np.concatenate([np.asarray(res["clump_fraction_series"], dtype=np.float64) for res in replicate_results])
        )
        tau_s_data.append(scalar_violin_data["tau_s"][-1])
        tau_b_data.append(scalar_violin_data["tau_b"][-1])
        assoc_rate_data.append(scalar_violin_data["rate_assoc"][-1])
        passive_rate_data.append(scalar_violin_data["rate_passive"][-1])
        diffusion_data.append(scalar_violin_data["diffusion_coefficient"][-1])

        summary_rows.append(
            {
                "epsilon": epsilon,
                **{f"{key}_mean": value["mean"] for key, value in scalar_summary.items()},
                **{f"{key}_stderr": value["stderr"] for key, value in scalar_summary.items()},
                **{f"{key}_n_finite": value["n_finite"] for key, value in scalar_summary.items()},
            }
        )
        summary_json[f"{epsilon:g}"] = scalar_summary
        log(f"Finished epsilon group eps={epsilon:g}")

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(os.path.join(args.output_dir, "summary.csv"), "w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for row in summary_rows:
                handle.write(",".join(str(row[k]) for k in keys) + "\n")

    if epsilon_values:
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "bonded_pairs_violin_vs_epsilon.png"),
            epsilon_values,
            bonded_pair_data,
            "sticker-sticker bonded pairs",
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "p_open_vs_epsilon.png"),
            epsilon_values,
            scalar_violin_data["p_open_mean"],
            r"$p_\mathrm{open}$",
            ylim=(0.0, 1.0),
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "bonding_probability_vs_epsilon.png"),
            epsilon_values,
            scalar_violin_data["bonding_probability_mean"],
            r"$p$",
            ylim=(0.0, 1.0),
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "gelation_epsilon_vs_epsilon.svg"),
            epsilon_values,
            scalar_violin_data["gelation_epsilon_mean"],
            "gelation epsilon",
            figsize=(3.3, 2.0),
            dpi=1000,
            x_label=r"$\varepsilon_\mathrm{reactiveLJ}$",
            tick_label_size=8,
            axis_label_size=10,
            body_facecolor="#9e9e9e",
            body_edgecolor="#6f6f6f",
            body_alpha=0.5,
            median_color="#2b2b2b",
            iqr_lw=2.0,
            median_lw=1.8,
            median_marker_size=18.0,
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond0_violin.png"),
            epsilon_values,
            frac_bond0_data,
            "fraction of stickers",
            ylim=(0.0, 1.0),
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond1_violin.png"),
            epsilon_values,
            frac_bond1_data,
            "fraction of stickers",
            ylim=(0.0, 1.0),
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond_gt1_violin.png"),
            epsilon_values,
            frac_bond_gt1_data,
            "fraction of stickers",
            ylim=(0.0, 1.0),
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "clump_fraction_violin_vs_epsilon.png"),
            epsilon_values,
            clump_fraction_data,
            "clump fraction",
            ylim=(0.0, 1.0),
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, INTRA_INTER_RATIO_EPSILON_PLOT),
            epsilon_values,
            scalar_violin_data["intra_inter_ratio"],
            INTRA_INTER_RATIO_LABEL,
            figsize=INTRA_INTER_RATIO_FIGSIZE,
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "number_average_cluster_size_vs_epsilon.png"),
            epsilon_values,
            scalar_violin_data["number_average_cluster_size"],
            "number-average cluster size",
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "weight_average_cluster_size_vs_epsilon.png"),
            epsilon_values,
            scalar_violin_data["weight_average_cluster_size"],
            "weight-average cluster size",
        )
        largest_cluster_epsilon, largest_cluster_mean = (
            summarize_largest_cluster_size_points(
                epsilon_values,
                scalar_violin_data["largest_cluster_size"],
            )
        )
        write_largest_cluster_size_plot(
            os.path.join(args.output_dir, LARGEST_CLUSTER_SIZE_EPSILON_PLOT),
            largest_cluster_epsilon,
            largest_cluster_mean,
        )
        write_cluster_distribution_by_epsilon_plot(
            os.path.join(args.output_dir, "cluster_size_distribution_by_epsilon.svg"),
            epsilon_values,
            cluster_distribution_by_eps,
        )
        write_dual_rate_plot(
            os.path.join(args.output_dir, "exchange_rate_comparison_vs_epsilon.png"),
            epsilon_values,
            assoc_rate_data,
            passive_rate_data,
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "bond_tau_vs_epsilon.svg"),
            epsilon_values,
            tau_s_data,
            r"$\tau_s$",
            figsize=(3.3, 1.5),
            yscale="log",
        )
        brachiation_epsilon, brachiation_tau = summarize_brachiation_tau_points(
            epsilon_values,
            tau_b_data,
        )
        write_brachiation_tau_plot(
            os.path.join(args.output_dir, BRACHIATION_TAU_EPSILON_PLOT),
            brachiation_epsilon,
            brachiation_tau,
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "sampled_monomer_diffusion_vs_epsilon.png"),
            epsilon_values,
            diffusion_data,
            "sampled monomer diffusion coefficient",
        )
        write_msd_vs_time_lag_by_epsilon_plot(
            os.path.join(args.output_dir, "sampled_monomer_msd_vs_time_lag_by_epsilon.svg"),
            epsilon_values,
            msd_time_by_eps,
            msd_mean_by_eps,
            tau_r0_reference,
        )
        write_largest_cluster_vs_timestep_plot(
            os.path.join(args.output_dir, "largest_cluster_size_vs_timestep_by_epsilon.png"),
            epsilon_values,
            largest_cluster_timestep_data,
            largest_cluster_size_mean_data,
        )

    log("MPCD polymer-solution analysis complete")


if __name__ == "__main__":
    main()
