#!/usr/bin/env python3
"""Plot MPCD sticker lifetime and brachiation time versus effective dump interval.

This follows Liu and O'Connor 2024 Figure 2b:
* tau_s is extracted from the bond-lifetime correlation f(t) = <s(0)s(t)>
* tau_b is extracted from the analogous open-sticker correlation

The script reuses the existing ReactiveLJ bond/open-state definitions from the
main MPCD analysis and measures both quantities for multiple effective dump
intervals by subsampling the stored trajectory frames with integer strides.
Physical lag times are derived directly from the stored GSD frame step numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

USER_TMP_DIR = Path("/tmp") / f"reactive_lj_plot_cache_{os.getuid()}"
USER_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(USER_TMP_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(USER_TMP_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import gsd.hoomd
import numba
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from analysis_utils import (
    compute_r_thresh,
    fit_exponential,
    fit_exponential_semilog_linear_region,
)


DEFAULT_EPSILONS = (0.0, 6.0, 12.0, 15.0, 18.0)
DEFAULT_BOND_PERSISTENCE_EPSILONS = (6.0, 12.0, 15.0, 18.0)
DEFAULT_STRIDES = (1, 2, 3, 4, 6, 8, 12, 16, 20)
DEFAULT_MAX_LAG_FRAMES = 100
DEFAULT_FIGSIZE = (3.1, 2.0)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 8
TAU_S_PLOT_EXCLUDED_EPSILONS = (0.0,)


@dataclass(frozen=True)
class RunEntry:
    epsilon: float
    rep_label: str
    gsd_path: Path
    metadata_path: Path


def default_n_jobs() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, 8))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Measure tau_s and tau_b versus effective dump interval by "
            "subsampling existing MPCD trajectories."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=script_dir.parent / "data_generation" / "outputs",
        help="Directory containing eps_*/rep_*/trajectory.gsd and metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "tau_s_tau_b_vs_dump_interval.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=script_dir / "results" / "tau_s_tau_b_vs_dump_interval.csv",
        help="Output CSV path with aggregated values.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="Stickiness values to analyze.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=list(DEFAULT_STRIDES),
        help=(
            "Integer frame strides used to emulate coarser dump intervals. "
            "The physical dump interval is inferred from GSD frame step numbers."
        ),
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=DEFAULT_MAX_LAG_FRAMES,
        help="Maximum number of subsampled lags used for the correlation fits.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=default_n_jobs(),
        help="Number of parallel workers to use across replicate trajectories.",
    )
    return parser.parse_args()


def parse_epsilon(path: Path) -> float | None:
    if not path.is_dir() or not path.name.startswith("eps_"):
        return None
    try:
        return float(path.name.split("_", maxsplit=1)[1])
    except ValueError:
        return None


def discover_runs(input_root: Path, requested_epsilons: Iterable[float]) -> list[RunEntry]:
    requested = tuple(float(eps) for eps in requested_epsilons)
    entries: list[RunEntry] = []
    for eps_dir in sorted(input_root.glob("eps_*")):
        epsilon = parse_epsilon(eps_dir)
        if epsilon is None:
            continue
        if not any(np.isclose(epsilon, requested_eps, rtol=0.0, atol=1.0e-12) for requested_eps in requested):
            continue
        for rep_dir in sorted(eps_dir.glob("rep_*")):
            gsd_path = rep_dir / "trajectory.gsd"
            metadata_path = rep_dir / "metadata.json"
            if gsd_path.is_file() and metadata_path.is_file():
                entries.append(
                    RunEntry(
                        epsilon=epsilon,
                        rep_label=rep_dir.name,
                        gsd_path=gsd_path,
                        metadata_path=metadata_path,
                    )
                )
    return entries


def format_epsilon_label(epsilon: float) -> str:
    return rf"{epsilon:g}$\mathrm{{k}}_\mathrm{{B}}T$"


def build_epsilon_color_map(epsilons: list[float]) -> dict[float, tuple[float, ...]]:
    reference_epsilons = sorted(
        {float(eps) for eps in DEFAULT_BOND_PERSISTENCE_EPSILONS}
        | {float(eps) for eps in epsilons}
    )
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(reference_epsilons)))
    return {
        epsilon: tuple(color)
        for epsilon, color in zip(reference_epsilons, colors)
    }


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, stderr


def infer_dump_interval_tau_lj(
    sampled_steps: np.ndarray,
    dt: float,
) -> float:
    if sampled_steps.size < 2:
        raise RuntimeError("Need at least two sampled frames to infer dump interval.")
    step_diffs = np.diff(sampled_steps.astype(np.float64, copy=False))
    positive = step_diffs[step_diffs > 0.0]
    if positive.size == 0:
        raise RuntimeError("Sampled GSD frames do not have increasing step numbers.")
    return float(np.mean(positive) * dt)


def frame_step_or_index(frame, fallback_index: int) -> int:
    frame_step = getattr(frame.configuration, "step", None)
    return int(frame_step) if frame_step is not None else int(fallback_index)


def checkpoint_deduplicated_frame_indices(traj, gsd_path: Path) -> np.ndarray:
    """Keep the resumed branch when append-mode checkpoint restarts overlap."""
    kept_indices: list[int] = []
    kept_steps: list[int] = []
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
        print(
            "Affected GSD file: "
            f"{gsd_path} has checkpoint-overlap frames; using resumed branch "
            f"and ignoring {discarded_frames} earlier frame(s) across "
            f"{overlap_events} overlap event(s) "
            f"(logical_frames={len(kept_indices)}/{len(traj)}, "
            f"first_overlap_physical_frame={first_overlap_frame}, "
            f"first_overlap_step={first_overlap_step}).",
            flush=True,
        )

    return np.asarray(kept_indices, dtype=np.int64)


def split_contiguous_segments(
    sampled_steps: np.ndarray,
    expected_step_delta: int,
) -> list[tuple[int, int]]:
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
    segments: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
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


def build_lag_time_axis(
    sampled_steps: np.ndarray,
    dt: float,
    lag_count: int,
) -> np.ndarray:
    if lag_count <= 0:
        return np.empty((0,), dtype=np.float64)
    if sampled_steps.size < 2:
        raise RuntimeError("Need at least two sampled frames to build a lag-time axis.")

    sampled_steps_f = sampled_steps.astype(np.float64, copy=False)
    first_diffs = np.diff(sampled_steps_f)
    positive = first_diffs[first_diffs > 0.0]
    if positive.size == 0:
        raise RuntimeError("Sampled GSD frames do not have increasing step numbers.")

    if np.allclose(positive, positive[0], rtol=0.0, atol=1.0e-12):
        return np.arange(1, lag_count + 1, dtype=np.float64) * float(positive[0] * dt)

    lag_times = np.empty(lag_count, dtype=np.float64)
    for lag in range(1, lag_count + 1):
        lag_step = sampled_steps_f[lag:] - sampled_steps_f[:-lag]
        positive_lag = lag_step[lag_step > 0.0]
        if positive_lag.size == 0:
            raise RuntimeError(
                "Sampled GSD frames do not have increasing step numbers at one or more lags."
            )
        lag_times[lag - 1] = float(np.mean(positive_lag) * dt)
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
def compute_bond_correlation(
    bonds_flat: np.ndarray,
    bond_offsets: np.ndarray,
    sample_indices: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    max_lag_frames: int,
) -> np.ndarray:
    lag_count = max_lag_frames
    corr = np.zeros(lag_count, dtype=np.float64)
    if lag_count <= 0:
        return corr

    numerators = np.zeros(lag_count, dtype=np.float64)
    denominators = np.zeros(lag_count, dtype=np.float64)

    for segment_idx in range(segment_starts.shape[0]):
        start_pos = segment_starts[segment_idx]
        end_pos = segment_ends[segment_idx]
        for pos in range(start_pos + 1, end_pos):
            current_frame = sample_indices[pos]
            current_start = bond_offsets[current_frame]
            current_end = bond_offsets[current_frame + 1]
            max_back = min(lag_count, pos - start_pos)
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
def compute_open_correlation(
    open_bitsets: np.ndarray,
    open_counts: np.ndarray,
    sample_indices: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    max_lag_frames: int,
) -> np.ndarray:
    lag_count = max_lag_frames
    corr = np.zeros(lag_count, dtype=np.float64)
    if lag_count <= 0:
        return corr

    numerators = np.zeros(lag_count, dtype=np.float64)
    denominators = np.zeros(lag_count, dtype=np.float64)

    for segment_idx in range(segment_starts.shape[0]):
        start_pos = segment_starts[segment_idx]
        end_pos = segment_ends[segment_idx]
        for pos in range(start_pos + 1, end_pos):
            current_frame = sample_indices[pos]
            max_back = min(lag_count, pos - start_pos)
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


def build_frame_states(
    positions: np.ndarray,
    sticker_ids: np.ndarray,
    n_stickers: int,
    box_length: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return sorted bond IDs and a packed open-sticker bitset for one frame."""
    sticker_positions = positions[sticker_ids]
    wrapped = np.mod(sticker_positions + 0.5 * box_length, box_length)
    tree = cKDTree(wrapped, boxsize=box_length)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    bonded_mask = np.zeros(n_stickers, dtype=bool)
    if pairs.size == 0:
        open_mask = ~bonded_mask
        return np.empty(0, dtype=np.uint32), pack_open_mask(open_mask), n_stickers

    bonded_mask[pairs.reshape(-1)] = True
    open_mask = ~bonded_mask

    bond_ids = (
        pairs[:, 0].astype(np.uint32, copy=False) * np.uint32(n_stickers)
        + pairs[:, 1].astype(np.uint32, copy=False)
    )
    bond_ids = np.ascontiguousarray(np.sort(bond_ids))
    open_bitset = pack_open_mask(open_mask)
    open_count = int(np.count_nonzero(open_mask))
    return bond_ids, open_bitset, open_count


def analyze_run(entry: RunEntry, strides: list[int], max_lag_frames: int) -> list[dict[str, float | str]]:
    with entry.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    with gsd.hoomd.open(str(entry.gsd_path), "r") as traj:
        frame_indices = checkpoint_deduplicated_frame_indices(traj, entry.gsd_path)
        n_frames = int(frame_indices.size)
        if n_frames < 3:
            return []

        first = traj[int(frame_indices[0])]
        type_names = first.particles.types
        if "sticky" not in type_names:
            raise RuntimeError(f"Sticker type 'sticky' not found in {entry.gsd_path}")
        sticker_type = type_names.index("sticky")
        trajectory_subset = str(metadata.get("trajectory_particle_subset", ""))
        if trajectory_subset == "sticky_only":
            typeid = np.asarray(first.particles.typeid, dtype=np.int32)
            if not np.all(typeid == sticker_type):
                raise RuntimeError(
                    f"Sticker-only trajectory {entry.gsd_path} contains non-sticky particles."
                )
            sticker_ids = np.arange(first.particles.N, dtype=np.int32)
        else:
            sticker_ids = np.where(first.particles.typeid == sticker_type)[0]
        n_stickers = int(sticker_ids.size)
        if n_stickers == 0:
            raise RuntimeError(f"No stickers found in {entry.gsd_path}")

        box_length = float(first.configuration.box[0])
        reactive_sigma = float(metadata.get("reactive_sigma", 1.0))
        r_thresh = compute_r_thresh(reactive_sigma)
        epsilon = float(metadata.get("reactive_epsilon", entry.epsilon))
        dt = float(metadata.get("dt", 0.005))
        frame_steps_default = int(
            metadata.get("trajectory_frame_steps", metadata.get("frame_steps", 10_000))
        )
        n_open_words = (n_stickers + 63) // 64
        open_bitsets = np.zeros((n_frames, n_open_words), dtype=np.uint64)
        open_counts = np.zeros(n_frames, dtype=np.int32)
        bond_chunks: list[np.ndarray] = []
        bond_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        frame_step_numbers = np.zeros(n_frames, dtype=np.int64)

        for frame_idx, physical_frame_idx in enumerate(frame_indices):
            frame = traj[int(physical_frame_idx)]
            positions = frame.particles.position
            bond_ids, open_bitset, open_count = build_frame_states(
                positions=positions,
                sticker_ids=sticker_ids,
                n_stickers=n_stickers,
                box_length=box_length,
                cutoff=r_thresh,
            )
            bond_chunks.append(bond_ids)
            bond_offsets[frame_idx + 1] = bond_offsets[frame_idx] + bond_ids.size
            open_bitsets[frame_idx] = open_bitset
            open_counts[frame_idx] = open_count
            frame_step_numbers[frame_idx] = frame_step_or_index(
                frame,
                int(physical_frame_idx),
            )

    if bond_offsets[-1] > 0:
        bonds_flat = np.ascontiguousarray(np.concatenate(bond_chunks))
    else:
        bonds_flat = np.empty(0, dtype=np.uint32)

    rows: list[dict[str, float | str]] = []
    for stride in sorted(strides):
        sample_indices = np.arange(0, n_frames, stride, dtype=np.int64)
        sampled_steps = frame_step_numbers[sample_indices]
        expected_step_delta = int(frame_steps_default * stride)
        sample_segments = split_contiguous_segments(sampled_steps, expected_step_delta)
        segment_starts, segment_ends = segment_bound_arrays(sample_segments)
        longest_segment_samples = (
            int(np.max(segment_ends - segment_starts))
            if segment_starts.size
            else 0
        )
        lag_count = min(max_lag_frames, max(0, longest_segment_samples - 1))
        if lag_count < 2:
            continue

        cs = compute_bond_correlation(
            bonds_flat,
            bond_offsets,
            sample_indices,
            segment_starts,
            segment_ends,
            lag_count,
        )
        cb = compute_open_correlation(
            open_bitsets,
            open_counts,
            sample_indices,
            segment_starts,
            segment_ends,
            lag_count,
        )
        frame_dt = dt * float(expected_step_delta)
        lag_time = np.arange(1, lag_count + 1, dtype=np.float64) * frame_dt
        cs_time = lag_time[: len(cs)]
        cb_time = lag_time[: len(cb)]
        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)
        tau_b = fit_exponential_semilog_linear_region(cb_time, cb)
        rows.append(
            {
                "epsilon": epsilon,
                "rep_label": entry.rep_label,
                "stride": float(stride),
                "dump_interval_tau_lj": frame_dt,
                "tau_s": tau_s,
                "tau_b": tau_b,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon",
        "stride",
        "dump_interval_tau_lj",
        "n_replicates",
        "tau_s_mean",
        "tau_s_stderr",
        "tau_b_mean",
        "tau_b_stderr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_aggregates(
    measurements: list[dict[str, float | str]],
    epsilons: list[float],
    strides: list[int],
) -> list[dict[str, float | int]]:
    grouped: dict[tuple[float, int], dict[str, list[float]]] = defaultdict(
        lambda: {"dump_interval_tau_lj": [], "tau_s": [], "tau_b": []}
    )
    for row in measurements:
        epsilon = float(row["epsilon"])
        stride = int(row["stride"])
        bucket = grouped[(epsilon, stride)]
        bucket["dump_interval_tau_lj"].append(float(row["dump_interval_tau_lj"]))
        bucket["tau_s"].append(float(row["tau_s"]))
        bucket["tau_b"].append(float(row["tau_b"]))

    aggregates: list[dict[str, float | int]] = []
    for epsilon in epsilons:
        for stride in strides:
            bucket = grouped.get((epsilon, stride))
            if bucket is None:
                continue
            dump_interval = float(np.mean(bucket["dump_interval_tau_lj"]))
            tau_s_mean, tau_s_stderr = mean_and_stderr(bucket["tau_s"])
            tau_b_mean, tau_b_stderr = mean_and_stderr(bucket["tau_b"])
            aggregates.append(
                {
                    "epsilon": epsilon,
                    "stride": stride,
                    "dump_interval_tau_lj": dump_interval,
                    "n_replicates": len(bucket["tau_s"]),
                    "tau_s_mean": tau_s_mean,
                    "tau_s_stderr": tau_s_stderr,
                    "tau_b_mean": tau_b_mean,
                    "tau_b_stderr": tau_b_stderr,
                }
            )
    return aggregates


def make_plot(path: Path, aggregates: list[dict[str, float | int]], epsilons: list[float]) -> None:
    by_epsilon: dict[float, list[dict[str, float | int]]] = defaultdict(list)
    for row in aggregates:
        by_epsilon[float(row["epsilon"])].append(row)

    color_map = build_epsilon_color_map(epsilons)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)

    plotted_dump_intervals: list[float] = []
    for epsilon in epsilons:
        color = color_map[float(epsilon)]
        rows = sorted(by_epsilon.get(epsilon, []), key=lambda item: float(item["dump_interval_tau_lj"]))
        if not rows:
            continue

        dump_interval = np.asarray([float(row["dump_interval_tau_lj"]) for row in rows], dtype=np.float64)
        tau_s_mean = np.asarray([float(row["tau_s_mean"]) for row in rows], dtype=np.float64)
        tau_s_stderr = np.asarray([float(row["tau_s_stderr"]) for row in rows], dtype=np.float64)
        tau_b_mean = np.asarray([float(row["tau_b_mean"]) for row in rows], dtype=np.float64)
        tau_b_stderr = np.asarray([float(row["tau_b_stderr"]) for row in rows], dtype=np.float64)

        plot_tau_s = not any(
            np.isclose(float(epsilon), excluded, rtol=0.0, atol=1.0e-12)
            for excluded in TAU_S_PLOT_EXCLUDED_EPSILONS
        )
        tau_s_mask = np.isfinite(dump_interval) & np.isfinite(tau_s_mean) & (tau_s_mean > 0.0)
        tau_s_mask &= plot_tau_s
        if np.any(tau_s_mask):
            plotted_dump_intervals.extend(dump_interval[tau_s_mask].tolist())
            ax.errorbar(
                dump_interval[tau_s_mask],
                tau_s_mean[tau_s_mask],
                yerr=tau_s_stderr[tau_s_mask],
                color=color,
                linewidth=1.5,
                linestyle="-",
                marker="o",
                markersize=3.2,
                capsize=2.0,
                label=rf"$\tau_s$, {format_epsilon_label(epsilon)}",
            )

        tau_b_mask = np.isfinite(dump_interval) & np.isfinite(tau_b_mean) & (tau_b_mean > 0.0)
        if np.any(tau_b_mask):
            plotted_dump_intervals.extend(dump_interval[tau_b_mask].tolist())
            ax.errorbar(
                dump_interval[tau_b_mask],
                tau_b_mean[tau_b_mask],
                yerr=tau_b_stderr[tau_b_mask],
                color=color,
                linewidth=1.5,
                linestyle="-",
                marker="^",
                markersize=3.2,
                capsize=2.0,
                alpha=0.5,
                label=rf"$\tau_b$, {format_epsilon_label(epsilon)}",
            )

    if plotted_dump_intervals:
        x_min = min(plotted_dump_intervals)
        x_max = max(plotted_dump_intervals)
        x_ref = np.geomspace(x_min, x_max, 256)
        ax.plot(
            x_ref,
            2.0 * x_ref,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=r"$y = 2\Delta t$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dump interval, $\Delta t$ $(\tau_\mathrm{LJ})$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\tau_s$, $\tau_b$ $(\tau_\mathrm{LJ})$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.legend(
        fontsize=DEFAULT_LEGEND_FONTSIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        ncol=1,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    strides = sorted({int(stride) for stride in args.strides if int(stride) > 0})
    if not strides:
        raise ValueError("At least one positive stride is required.")
    if args.max_lag_frames < 2:
        raise ValueError("--max-lag-frames must be at least 2.")

    epsilons = [float(eps) for eps in args.epsilons]
    runs = discover_runs(args.input_root, epsilons)
    if not runs:
        raise FileNotFoundError(
            f"No matching eps_*/rep_* trajectories found under {args.input_root}"
        )

    print(
        f"Analyzing {len(runs)} replicate trajectories for epsilons "
        f"{', '.join(f'{eps:g}' for eps in epsilons)} with strides "
        f"{', '.join(str(stride) for stride in strides)}",
        flush=True,
    )

    jobs = min(max(1, int(args.n_jobs)), len(runs))
    nested_rows = Parallel(n_jobs=jobs, backend="loky")(
        delayed(analyze_run)(entry, strides, int(args.max_lag_frames)) for entry in runs
    )
    measurements = [row for rows in nested_rows for row in rows]
    if not measurements:
        raise RuntimeError("No tau measurements were produced.")

    aggregates = build_aggregates(measurements, epsilons, strides)
    if not aggregates:
        raise RuntimeError("No aggregated tau measurements were produced.")

    write_csv(args.csv_output, aggregates)
    make_plot(args.output, aggregates, epsilons)

    print(f"Wrote aggregate tau data to {args.csv_output}", flush=True)
    print(f"Wrote tau_s/tau_b plot to {args.output}", flush=True)


if __name__ == "__main__":
    main()
