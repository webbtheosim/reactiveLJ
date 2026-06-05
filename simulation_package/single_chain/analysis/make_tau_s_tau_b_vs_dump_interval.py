#!/usr/bin/env python3
"""Plot sticker persistence (tau_s) and brachiation (tau_b) vs dump interval.

All times are reported in units of tau / tau_R^(0), consistent with Liu & O'Connor
(2024). The script reads tau_R0 from each run's metadata (fallback to 4041.0).

Bond definitions follow the single-chain analysis: only paired bonds (stickers
with exactly one bond) are counted. Bonds in 3+ sticker clusters are excluded.
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

USER_TMP_DIR = Path("/tmp") / f"single_chain_plot_cache_{os.getuid()}"
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

from analysis_utils import compute_r_thresh, fit_exponential_semilog_linear_region


DEFAULT_TAU_R0 = 4041.0
DEFAULT_TARGET_DUMP_INTERVAL_TAU_R0 = 2.0e4
DEFAULT_MAX_LAG_FRAMES = 100
DEFAULT_STRIDE_COUNT = 12
DEFAULT_FIGSIZE = (3.1, 2.0)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 8


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
            "Measure tau_s and tau_b vs effective dump interval by subsampling "
            "single-chain trajectories."
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
        default=script_dir / "tau_s_tau_b_vs_dump_interval.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=script_dir / "tau_s_tau_b_vs_dump_interval.csv",
        help="Output CSV path with aggregated values.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=None,
        help="Stickiness values to analyze (defaults to all discovered).",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit integer frame strides used to emulate coarser dump intervals. "
            "When omitted, the script builds a log-spaced set of strides from 1 "
            "up to the target dump interval."
        ),
    )
    parser.add_argument(
        "--target-dump-interval-tau-r0",
        type=float,
        default=DEFAULT_TARGET_DUMP_INTERVAL_TAU_R0,
        help=(
            "Target dump interval in units of tau_R^0 used to determine the "
            "maximum stride (default 2e4)."
        ),
    )
    parser.add_argument(
        "--stride-count",
        type=int,
        default=DEFAULT_STRIDE_COUNT,
        help="Number of log-spaced strides used when --strides is omitted.",
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


def discover_runs(
    input_root: Path, requested_epsilons: Iterable[float] | None
) -> list[RunEntry]:
    requested: tuple[float, ...] | None = None
    if requested_epsilons is not None:
        requested = tuple(float(eps) for eps in requested_epsilons)
    entries: list[RunEntry] = []
    for eps_dir in sorted(input_root.glob("eps_*")):
        epsilon = parse_epsilon(eps_dir)
        if epsilon is None:
            continue
        if requested is not None and not any(
            np.isclose(epsilon, requested_eps, rtol=0.0, atol=1.0e-12)
            for requested_eps in requested
        ):
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
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(epsilons)))
    return {epsilon: tuple(color) for epsilon, color in zip(epsilons, colors)}


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, stderr


def estimate_base_dump_interval_tau_r0(entries: list[RunEntry]) -> float:
    intervals: list[float] = []
    for entry in entries:
        with entry.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        dt = float(metadata.get("dt", 0.005))
        frame_steps = int(metadata.get("frame_steps", 10_000))
        tau_r0 = float(metadata.get("tau_R0", DEFAULT_TAU_R0))
        if tau_r0 > 0.0:
            intervals.append(dt * frame_steps / tau_r0)
    if not intervals:
        raise RuntimeError("Failed to infer base dump interval from metadata.")
    return min(intervals)


def build_default_strides(
    base_interval_tau_r0: float, target_interval_tau_r0: float, stride_count: int
) -> list[int]:
    if base_interval_tau_r0 <= 0.0:
        return [1]
    max_stride = max(1, int(np.ceil(target_interval_tau_r0 / base_interval_tau_r0)))
    stride_count = max(1, stride_count)
    raw = np.geomspace(1, max_stride, num=stride_count)
    strides = sorted({int(round(val)) for val in raw} | {1, max_stride})
    return [stride for stride in strides if stride > 0]


def infer_dump_interval_tau_r0(
    sampled_steps: np.ndarray, dt: float, tau_r0: float
) -> float:
    if sampled_steps.size < 2:
        raise RuntimeError("Need at least two sampled frames to infer dump interval.")
    step_diffs = np.diff(sampled_steps.astype(np.float64, copy=False))
    positive = step_diffs[step_diffs > 0.0]
    if positive.size == 0:
        raise RuntimeError("Sampled GSD frames do not have increasing step numbers.")
    return float(np.mean(positive) * dt / tau_r0)


def build_lag_time_axis_tau_r0(
    sampled_steps: np.ndarray,
    dt: float,
    tau_r0: float,
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
        return np.arange(1, lag_count + 1, dtype=np.float64) * float(
            positive[0] * dt / tau_r0
        )

    lag_times = np.empty(lag_count, dtype=np.float64)
    for lag in range(1, lag_count + 1):
        lag_step = sampled_steps_f[lag:] - sampled_steps_f[:-lag]
        positive_lag = lag_step[lag_step > 0.0]
        if positive_lag.size == 0:
            raise RuntimeError(
                "Sampled GSD frames do not have increasing step numbers at one or more lags."
            )
        lag_times[lag - 1] = float(np.mean(positive_lag) * dt / tau_r0)
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
    max_lag_frames: int,
) -> np.ndarray:
    n_samples = sample_indices.shape[0]
    lag_count = min(max_lag_frames, n_samples - 1)
    corr = np.zeros(lag_count, dtype=np.float64)
    if lag_count <= 0:
        return corr

    numerators = np.zeros(lag_count, dtype=np.float64)
    denominators = np.zeros(lag_count, dtype=np.float64)

    for pos in range(1, n_samples):
        current_frame = sample_indices[pos]
        current_start = bond_offsets[current_frame]
        current_end = bond_offsets[current_frame + 1]
        max_back = min(lag_count, pos)
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
    max_lag_frames: int,
) -> np.ndarray:
    n_samples = sample_indices.shape[0]
    lag_count = min(max_lag_frames, n_samples - 1)
    corr = np.zeros(lag_count, dtype=np.float64)
    if lag_count <= 0:
        return corr

    numerators = np.zeros(lag_count, dtype=np.float64)
    denominators = np.zeros(lag_count, dtype=np.float64)

    for pos in range(1, n_samples):
        current_frame = sample_indices[pos]
        max_back = min(lag_count, pos)
        for lag in range(1, max_back + 1):
            previous_frame = sample_indices[pos - lag]
            previous_count = open_counts[previous_frame]
            if previous_count <= 0:
                continue
            denominators[lag - 1] += previous_count
            numerators[lag - 1] += bitset_intersection_count(
                open_bitsets[previous_frame],
                open_bitsets[current_frame]
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
    """Return paired-bond IDs and a packed open-sticker bitset for one frame."""
    sticker_positions = positions[sticker_ids]
    wrapped = np.mod(sticker_positions + 0.5 * box_length, box_length)
    tree = cKDTree(wrapped, boxsize=box_length)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    bonded_mask = np.zeros(n_stickers, dtype=bool)
    if pairs.size == 0:
        open_mask = ~bonded_mask
        return np.empty(0, dtype=np.uint32), pack_open_mask(open_mask), n_stickers

    degrees = np.zeros(n_stickers, dtype=np.int32)
    np.add.at(degrees, pairs[:, 0], 1)
    np.add.at(degrees, pairs[:, 1], 1)
    paired_mask = (degrees[pairs[:, 0]] == 1) & (degrees[pairs[:, 1]] == 1)
    if not np.any(paired_mask):
        open_mask = ~bonded_mask
        return np.empty(0, dtype=np.uint32), pack_open_mask(open_mask), n_stickers

    paired_pairs = pairs[paired_mask]
    bonded_mask[paired_pairs.reshape(-1)] = True
    open_mask = ~bonded_mask

    bond_ids = (
        paired_pairs[:, 0].astype(np.uint32, copy=False) * np.uint32(n_stickers)
        + paired_pairs[:, 1].astype(np.uint32, copy=False)
    )
    bond_ids = np.ascontiguousarray(np.sort(bond_ids))
    open_bitset = pack_open_mask(open_mask)
    open_count = int(np.count_nonzero(open_mask))
    return bond_ids, open_bitset, open_count


def analyze_run(
    entry: RunEntry, strides: list[int], max_lag_frames: int
) -> list[dict[str, float | str]]:
    with entry.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    with gsd.hoomd.open(str(entry.gsd_path), "r") as traj:
        n_frames = len(traj)
        if n_frames < 3:
            return []

        first = traj[0]
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
        if "analysis_bond_cutoff" in metadata:
            r_thresh = float(metadata["analysis_bond_cutoff"])
        else:
            reactive_sigma = float(metadata.get("reactive_sigma", 1.0))
            r_thresh = compute_r_thresh(reactive_sigma)
        dt = float(metadata.get("dt", 0.005))
        tau_r0 = float(metadata.get("tau_R0", DEFAULT_TAU_R0))
        if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
            tau_r0 = DEFAULT_TAU_R0

        n_open_words = (n_stickers + 63) // 64
        open_bitsets = np.zeros((n_frames, n_open_words), dtype=np.uint64)
        open_counts = np.zeros(n_frames, dtype=np.int32)
        bond_chunks: list[np.ndarray] = []
        bond_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        frame_step_numbers = np.zeros(n_frames, dtype=np.int64)

        for frame_idx, frame in enumerate(traj):
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
            frame_step_numbers[frame_idx] = int(frame.configuration.step)

    if bond_offsets[-1] > 0:
        bonds_flat = np.ascontiguousarray(np.concatenate(bond_chunks))
    else:
        bonds_flat = np.empty(0, dtype=np.uint32)

    rows: list[dict[str, float | str]] = []
    for stride in sorted(strides):
        sample_indices = np.arange(0, n_frames, stride, dtype=np.int64)
        lag_count = min(max_lag_frames, sample_indices.size - 1)
        if lag_count < 2:
            continue
        sampled_steps = frame_step_numbers[sample_indices]

        cs = compute_bond_correlation(
            bonds_flat,
            bond_offsets,
            sample_indices,
            max_lag_frames,
        )
        cb = compute_open_correlation(
            open_bitsets,
            open_counts,
            sample_indices,
            max_lag_frames,
        )
        dump_interval_tau_r0 = infer_dump_interval_tau_r0(sampled_steps, dt, tau_r0)
        lag_time = build_lag_time_axis_tau_r0(sampled_steps, dt, tau_r0, lag_count)
        cs_time = lag_time[: len(cs)]
        cb_time = lag_time[: len(cb)]
        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)
        tau_b = fit_exponential_semilog_linear_region(cb_time, cb)
        rows.append(
            {
                "epsilon": float(entry.epsilon),
                "rep_label": entry.rep_label,
                "stride": float(stride),
                "dump_interval_tau_r0": dump_interval_tau_r0,
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
        "dump_interval_tau_r0",
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
        lambda: {"dump_interval_tau_r0": [], "tau_s": [], "tau_b": []}
    )
    for row in measurements:
        epsilon = float(row["epsilon"])
        stride = int(row["stride"])
        bucket = grouped[(epsilon, stride)]
        bucket["dump_interval_tau_r0"].append(float(row["dump_interval_tau_r0"]))
        bucket["tau_s"].append(float(row["tau_s"]))
        bucket["tau_b"].append(float(row["tau_b"]))

    aggregates: list[dict[str, float | int]] = []
    for epsilon in epsilons:
        for stride in strides:
            bucket = grouped.get((epsilon, stride))
            if bucket is None:
                continue
            dump_interval = float(np.mean(bucket["dump_interval_tau_r0"]))
            tau_s_mean, tau_s_stderr = mean_and_stderr(bucket["tau_s"])
            tau_b_mean, tau_b_stderr = mean_and_stderr(bucket["tau_b"])
            aggregates.append(
                {
                    "epsilon": epsilon,
                    "stride": stride,
                    "dump_interval_tau_r0": dump_interval,
                    "n_replicates": len(bucket["tau_s"]),
                    "tau_s_mean": tau_s_mean,
                    "tau_s_stderr": tau_s_stderr,
                    "tau_b_mean": tau_b_mean,
                    "tau_b_stderr": tau_b_stderr,
                }
            )
    return aggregates


def make_plot(
    path: Path, aggregates: list[dict[str, float | int]], epsilons: list[float]
) -> None:
    by_epsilon: dict[float, list[dict[str, float | int]]] = defaultdict(list)
    for row in aggregates:
        by_epsilon[float(row["epsilon"])].append(row)

    color_map = build_epsilon_color_map(epsilons)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)

    plotted_dump_intervals: list[float] = []
    for epsilon in epsilons:
        color = color_map[float(epsilon)]
        rows = sorted(
            by_epsilon.get(epsilon, []),
            key=lambda item: float(item["dump_interval_tau_r0"]),
        )
        if not rows:
            continue

        dump_interval = np.asarray(
            [float(row["dump_interval_tau_r0"]) for row in rows], dtype=np.float64
        )
        tau_s_mean = np.asarray([float(row["tau_s_mean"]) for row in rows], dtype=np.float64)
        tau_s_stderr = np.asarray(
            [float(row["tau_s_stderr"]) for row in rows], dtype=np.float64
        )
        tau_b_mean = np.asarray([float(row["tau_b_mean"]) for row in rows], dtype=np.float64)
        tau_b_stderr = np.asarray(
            [float(row["tau_b_stderr"]) for row in rows], dtype=np.float64
        )

        tau_s_mask = np.isfinite(dump_interval) & np.isfinite(tau_s_mean) & (tau_s_mean > 0.0)
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
    ax.set_xlabel(r"Dump interval, $\Delta t / \tau_R^{(0)}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(
        r"$\tau_s$, $\tau_b$ $(\tau/\tau_R^{(0)})$", fontsize=DEFAULT_LABEL_FONTSIZE
    )
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
    if args.max_lag_frames < 2:
        raise ValueError("--max-lag-frames must be at least 2.")
    if args.target_dump_interval_tau_r0 <= 0.0:
        raise ValueError("--target-dump-interval-tau-r0 must be positive.")

    runs = discover_runs(args.input_root, args.epsilons)
    if not runs:
        raise FileNotFoundError(
            f"No matching eps_*/rep_* trajectories found under {args.input_root}"
        )
    epsilons = (
        [float(eps) for eps in args.epsilons]
        if args.epsilons is not None
        else sorted({entry.epsilon for entry in runs})
    )

    if args.strides is None:
        base_interval = estimate_base_dump_interval_tau_r0(runs)
        strides = build_default_strides(
            base_interval, args.target_dump_interval_tau_r0, int(args.stride_count)
        )
    else:
        strides = sorted({int(stride) for stride in args.strides if int(stride) > 0})
    if not strides:
        raise ValueError("At least one positive stride is required.")

    print(
        "Analyzing {} replicate trajectories for epsilons {} with strides {}".format(
            len(runs),
            ", ".join(f"{eps:g}" for eps in epsilons),
            ", ".join(str(stride) for stride in strides),
        ),
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
