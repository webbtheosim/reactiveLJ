"""Analyze single-chain ReactiveLJ trajectories across linker strengths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import gsd.hoomd
import numpy as np
from joblib import Parallel, delayed

_CACHE_ROOT = os.path.join("/tmp", f"single-chain-analysis-cache-{os.getuid()}")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports resolve when running from repo root.
sys.path.append(os.path.dirname(__file__))

from analysis_utils import (
    CorrelationAccumulator,
    compute_r_thresh,
    extract_semilog_linear_region,
    find_sticker_bonds,
    fit_exponential_semilog_linear_region,
)


PLOT_DPI = 1000
TICK_FONTSIZE = 8
LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 8


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze single-chain ReactiveLJ trajectories.")
    parser.add_argument(
        "--input-root",
        default="../data_generation/outputs",
        help="Root directory containing eps_*/rep_*/trajectory.gsd",
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
        help="Stride for frames in analysis (1 = use every frame).",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=50,
        help="Maximum lag (in frames) used for bond autocorrelation.",
    )
    return parser.parse_args()


def discover_runs(input_root: str) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    for root, _, files in os.walk(input_root):
        if "trajectory.gsd" in files and "metadata.json" in files:
            runs.append(
                (os.path.join(root, "trajectory.gsd"), os.path.join(root, "metadata.json"))
            )
    return sorted(runs)


def mean_and_stderr(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, stderr


def write_properties_csv(path: str, properties: Dict[str, Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("property,mean,stderr\n")
        for name, stats in properties.items():
            handle.write(f"{name},{stats['mean']},{stats['stderr']}\n")


def write_timeseries(path: str, time: np.ndarray, mean: np.ndarray, stderr: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("time,mean,stderr\n")
        for t, m, s in zip(time, mean, stderr):
            handle.write(f"{t:.6e},{m:.6e},{s:.6e}\n")


def write_timeseries_quantiles(
    path: str,
    time: np.ndarray,
    median: np.ndarray,
    q1: np.ndarray,
    q3: np.ndarray,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("time,median,q1,q3\n")
        for t, m, lo, hi in zip(time, median, q1, q3):
            handle.write(f"{t:.6e},{m:.6e},{lo:.6e},{hi:.6e}\n")


def finite_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def style_axes(ax) -> None:
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2)


def write_autocorr_fit_plot(
    path: str,
    time: np.ndarray,
    values: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    del title  # Plot titles intentionally omitted for publication styling.
    median = np.median(values, axis=0)
    q1 = np.percentile(values, 25.0, axis=0)
    q3 = np.percentile(values, 75.0, axis=0)

    tau_fit = fit_exponential_semilog_linear_region(time, median)
    fit_time, _ = extract_semilog_linear_region(time, median)

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, median, color="#121212", lw=1.7, label="median")

    if np.isfinite(tau_fit) and fit_time.size > 0:
        fit_curve = np.exp(-fit_time / tau_fit)
        ax.plot(
            fit_time,
            fit_curve,
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


def write_rg_summary_plot(
    path: str,
    time: np.ndarray,
    rg_values: np.ndarray,
    epsilon: float,
) -> None:
    del epsilon
    median = np.median(rg_values, axis=0)
    q1 = np.percentile(rg_values, 25.0, axis=0)
    q3 = np.percentile(rg_values, 75.0, axis=0)

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, median, color="#121212", lw=1.7, label="median")
    ax.set_xlabel("time", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$R_\mathrm{g}$", fontsize=LABEL_FONTSIZE)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str,
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
    del title  # Plot titles intentionally omitted for publication styling.
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
    medians: List[float] = []
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
    del title  # Plot titles intentionally omitted for publication styling.
    left_positions: List[float] = []
    left_violin_data: List[np.ndarray] = []
    left_medians: List[float] = []
    right_positions: List[float] = []
    right_violin_data: List[np.ndarray] = []
    right_medians: List[float] = []
    offset = 0.16
    width = 0.28

    for eps, left_vals, right_vals in zip(epsilon_values, data_left, data_right):
        left_arr = finite_array(left_vals)
        right_arr = finite_array(right_vals)

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

    fig, ax = plt.subplots(figsize=(3.6, 3.3))

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
            ax.vlines(x, q1, q3, color=left_color, lw=1.7)
            ax.scatter([x], [med], color=left_color, s=14, zorder=3)
        ax.plot(
            left_positions,
            left_medians,
            color=left_color,
            lw=1.4,
            alpha=0.9,
            zorder=2,
            label=left_label,
        )

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
            ax.vlines(x, q1, q3, color=right_color, lw=1.7)
            ax.scatter([x], [med], color=right_color, s=14, zorder=3)
        ax.plot(
            right_positions,
            right_medians,
            color=right_color,
            lw=1.4,
            alpha=0.9,
            zorder=2,
            label=right_label,
        )

    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_rg_time_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    rg_time_by_eps: Dict[float, np.ndarray],
    rg_median_by_eps: Dict[float, np.ndarray],
) -> None:
    if not epsilon_values:
        return

    colors = plt.cm.Greys(np.linspace(0.25, 0.85, max(len(epsilon_values), 2)))

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    for idx, eps in enumerate(epsilon_values):
        time = rg_time_by_eps.get(eps)
        med = rg_median_by_eps.get(eps)
        if time is None or med is None:
            continue
        ax.plot(time, med, lw=1.7, color=colors[idx], label=f"eps={eps:g}")

    ax.set_xlabel("time", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$R_\mathrm{g}$", fontsize=LABEL_FONTSIZE)
    ax.legend(frameon=False, ncol=2, fontsize=LEGEND_FONTSIZE)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
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
    if not epsilon_values:
        return

    medians: List[float] = []
    q1s: List[float] = []
    q3s: List[float] = []
    valid_eps: List[float] = []

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
    med = np.asarray(medians, dtype=np.float64)
    q1 = np.asarray(q1s, dtype=np.float64)
    q3 = np.asarray(q3s, dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)
    if show_iqr:
        ax.fill_between(x, q1, q3, color="#9e9e9e", alpha=0.35)
    ax.plot(x, med, color="#121212", lw=1.7, marker="o", ms=3.5)
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


def aggregate_timeseries(
    replicate_results: List[Dict],
    key_time: str,
    key_val: str,
    epsilon: float,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    series = [
        (res[key_time], res[key_val])
        for res in replicate_results
        if res.get(key_val) is not None and res.get(key_time) is not None
    ]
    if not series:
        return None, None, None, None

    lengths_t = [len(t) for t, _ in series]
    lengths_v = [len(v) for _, v in series]
    if len(set(lengths_t)) != 1 or len(set(lengths_v)) != 1:
        raise RuntimeError(
            f"Replicate series length mismatch for eps={epsilon:g}, key={key_val}, "
            f"time_lengths={lengths_t}, value_lengths={lengths_v}"
        )

    n = lengths_t[0]
    time = series[0][0][:n]
    values = np.stack([v[:n] for _, v in series], axis=0)
    mean = np.mean(values, axis=0)
    stderr = (
        np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        if values.shape[0] > 1
        else np.zeros_like(mean)
    )
    return time, values, mean, stderr


def analyze_replicate(
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    progress_label: str | None = None,
) -> Dict:
    with gsd.hoomd.open(gsd_path, "r") as traj:
        n_frames = len(traj)
        if n_frames == 0:
            raise RuntimeError(f"No frames found in {gsd_path}")

        n_analyzed = (n_frames + analysis_stride - 1) // analysis_stride
        progress_interval = max(1, n_analyzed // 10)

        first = traj[0]
        type_names = first.particles.types
        if "sticky" not in type_names:
            raise RuntimeError("Sticker type 'sticky' not found in trajectory.")
        sticker_type = type_names.index("sticky")

        typeid = first.particles.typeid
        sticker_ids = np.where(typeid == sticker_type)[0]
        if sticker_ids.size == 0:
            raise RuntimeError("No sticker particles found in trajectory.")

        n_particles = int(first.particles.N)
        chain_length = int(metadata.get("chain_length", n_particles))
        if chain_length != n_particles:
            log(
                f"Warning: metadata chain_length={chain_length} differs from particle count={n_particles}. "
                "Using particle count for end-to-end and loop normalization."
            )
            chain_length = n_particles

        n_stickers = int(sticker_ids.size)
        sticker_idx_map = np.full(n_particles, -1, dtype=np.int32)
        sticker_idx_map[sticker_ids] = np.arange(n_stickers, dtype=np.int32)

        box_length = float(first.configuration.box[0])
        if "analysis_bond_cutoff" in metadata:
            r_thresh = float(metadata["analysis_bond_cutoff"])
        else:
            r_thresh = compute_r_thresh(float(metadata.get("reactive_sigma", 1.0)))

        dt = float(metadata.get("dt", 0.005))
        frame_steps = int(metadata.get("frame_steps", 10_000))
        frame_dt = dt * frame_steps * analysis_stride

        bond_corr = CorrelationAccumulator(max_lag_frames)

        raw_bond_count_series: List[float] = []
        bond_count_series: List[float] = []
        rg_series: List[float] = []
        end_to_end_series: List[float] = []
        frac_bond0_series: List[float] = []
        frac_bond1_series: List[float] = []
        frac_bond_gt1_series: List[float] = []
        loop_length_series: List[int] = []

        assoc_rate_sum = 0.0
        dissoc_rate_sum = 0.0
        rate_count = 0

        assoc_events_total = 0
        total_transition_time = 0.0
        free_sticker_time = 0.0

        prev_bonds: set | None = None
        prev_open_count: int | None = None
        prev_has_partner: np.ndarray | None = None

        for analyzed_idx, frame_idx in enumerate(range(0, n_frames, analysis_stride), start=1):
            frame = traj[frame_idx]
            positions = frame.particles.position
            images = getattr(frame.particles, "image", None)

            raw_bonds = find_sticker_bonds(positions, sticker_ids, box_length, r_thresh)
            raw_bond_count_series.append(float(len(raw_bonds)))
            paired_bonds: set = set()
            degrees = np.zeros(n_stickers, dtype=np.int32)
            if raw_bonds:
                bond_array = np.asarray(list(raw_bonds), dtype=np.int64)
                i_global = bond_array[:, 0]
                j_global = bond_array[:, 1]
                i_idx = sticker_idx_map[i_global]
                j_idx = sticker_idx_map[j_global]

                raw_degrees = np.zeros(n_stickers, dtype=np.int32)
                np.add.at(raw_degrees, i_idx, 1)
                np.add.at(raw_degrees, j_idx, 1)
                paired_mask = (raw_degrees[i_idx] == 1) & (raw_degrees[j_idx] == 1)
                if np.any(paired_mask):
                    paired_array = bond_array[paired_mask]
                    paired_bonds = set(
                        zip(paired_array[:, 0].tolist(), paired_array[:, 1].tolist())
                    )
                    paired_i_idx = sticker_idx_map[paired_array[:, 0]]
                    paired_j_idx = sticker_idx_map[paired_array[:, 1]]
                    np.add.at(degrees, paired_i_idx, 1)
                    np.add.at(degrees, paired_j_idx, 1)
                    loop_length_series.extend(
                        np.abs(paired_array[:, 0] - paired_array[:, 1]).tolist()
                    )

            bond_corr.update(paired_bonds)
            bond_count_series.append(float(len(paired_bonds)))

            current_has_partner = degrees > 0

            count0 = int(np.sum(degrees == 0))
            count1 = int(np.sum(degrees == 1))
            count_gt1 = n_stickers - count0 - count1
            frac_bond0_series.append(count0 / n_stickers)
            frac_bond1_series.append(count1 / n_stickers)
            frac_bond_gt1_series.append(count_gt1 / n_stickers)

            if images is None:
                unwrapped = positions.astype(np.float64)
            else:
                unwrapped = positions.astype(np.float64) + images.astype(np.float64) * box_length

            com = np.mean(unwrapped, axis=0)
            delta = unwrapped - com
            rg_val = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
            rg_series.append(rg_val)

            ree_val = float(np.linalg.norm(unwrapped[-1] - unwrapped[0]))
            end_to_end_series.append(ree_val)

            open_count = count0

            if prev_bonds is not None and prev_open_count is not None and prev_has_partner is not None:
                new_bonds = paired_bonds - prev_bonds
                if new_bonds:
                    new_bond_array = np.asarray(list(new_bonds), dtype=np.int64)
                    new_i_idx = sticker_idx_map[new_bond_array[:, 0]]
                    new_j_idx = sticker_idx_map[new_bond_array[:, 1]]
                    assoc_mask = prev_has_partner[new_i_idx] | prev_has_partner[new_j_idx]
                    assoc = int(np.count_nonzero(assoc_mask))
                    dissoc = int(new_bond_array.shape[0] - assoc)
                else:
                    assoc = 0
                    dissoc = 0

                assoc_events_total += assoc
                total_transition_time += frame_dt

                n_m = prev_open_count
                free_sticker_time += n_m * frame_dt
                if n_m > 0:
                    assoc_rate_sum += assoc / (n_m * frame_dt)
                    dissoc_rate_sum += dissoc / (n_m * frame_dt)
                    rate_count += 1

            prev_bonds = paired_bonds
            prev_open_count = open_count
            prev_has_partner = current_has_partner

            if progress_label is not None and (
                analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
            ):
                progress_pct = 100.0 * analyzed_idx / n_analyzed
                log(
                    f"{progress_label}: frame progress {analyzed_idx}/{n_analyzed} "
                    f"({progress_pct:.1f}%)"
                )

        cs = bond_corr.correlation()
        cs_time = np.arange(1, len(cs) + 1, dtype=np.float64) * frame_dt
        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)

        rg_arr = np.asarray(rg_series, dtype=np.float64)
        ree_arr = np.asarray(end_to_end_series, dtype=np.float64)
        bond_count_arr = np.asarray(bond_count_series, dtype=np.float64)

        if total_transition_time > 0.0:
            swap_rate = float(assoc_events_total / total_transition_time)
        else:
            swap_rate = float("nan")

        if free_sticker_time > 0.0:
            swap_rate_per_free = float(assoc_events_total / free_sticker_time)
        else:
            swap_rate_per_free = float("nan")

        result = {
            "bonded_pairs_mean": float(np.mean(bond_count_arr)) if bond_count_arr.size else float("nan"),
            "rate_assoc": assoc_rate_sum / max(rate_count, 1),
            "rate_dissoc": dissoc_rate_sum / max(rate_count, 1),
            "swap_rate": swap_rate,
            "swap_rate_per_free_sticker": swap_rate_per_free,
            "tau_s": tau_s,
            "rg_mean": float(np.mean(rg_arr)) if rg_arr.size else float("nan"),
            "end_to_end_mean": float(np.mean(ree_arr)) if ree_arr.size else float("nan"),
            "raw_bond_count_series": np.asarray(raw_bond_count_series, dtype=np.float64),
            "bond_count_series": bond_count_arr,
            "rg_time": np.arange(len(rg_arr), dtype=np.float64) * frame_dt,
            "rg_series": rg_arr,
            "end_to_end_series": ree_arr,
            "frac_bond0_series": np.asarray(frac_bond0_series, dtype=np.float64),
            "frac_bond1_series": np.asarray(frac_bond1_series, dtype=np.float64),
            "frac_bond_gt1_series": np.asarray(frac_bond_gt1_series, dtype=np.float64),
            "loop_length_series": np.asarray(loop_length_series, dtype=np.int32),
            "cs_time": cs_time,
            "cs": cs,
            "chain_length": chain_length,
        }
        return result


def analyze_replicate_job(
    epsilon: float,
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    rep_label: str,
    rel_path: str,
) -> Tuple[float, Dict]:
    log(f"{rep_label}: start ({rel_path})")
    result = analyze_replicate(
        gsd_path,
        metadata,
        analysis_stride,
        max_lag_frames,
        progress_label=rep_label,
    )
    log(f"{rep_label}: done")
    return epsilon, result


def main() -> None:
    args = parse_args()
    log(f"Scanning trajectories under {args.input_root}")
    runs = discover_runs(args.input_root)
    if not runs:
        raise RuntimeError(f"No trajectories found under {args.input_root}")
    log(f"Discovered {len(runs)} trajectory/metadata pairs")

    grouped: Dict[float, List[Tuple[str, Dict]]] = defaultdict(list)
    for gsd_path, metadata_path in runs:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        epsilon = float(metadata["reactive_epsilon"])
        grouped[epsilon].append((gsd_path, metadata))
    log(f"Grouped runs into {len(grouped)} epsilon values")

    os.makedirs(args.output_dir, exist_ok=True)

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", max(1, os.cpu_count() or 1)))
    if n_jobs < 1:
        n_jobs = 1

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

    epsilon_values: List[float] = []
    summary_rows: List[Dict[str, float]] = []
    summary_json: Dict[str, Dict] = {}

    bonded_pairs_data: List[np.ndarray] = []
    rg_data: List[np.ndarray] = []
    end_to_end_data: List[np.ndarray] = []
    valence0_data: List[np.ndarray] = []
    valence1_data: List[np.ndarray] = []
    valence_gt1_data: List[np.ndarray] = []
    loop_length_data: List[np.ndarray] = []

    assoc_rate_data: List[np.ndarray] = []
    dissoc_rate_data: List[np.ndarray] = []
    swap_rate_data: List[np.ndarray] = []
    swap_rate_per_free_data: List[np.ndarray] = []
    tau_s_data: List[np.ndarray] = []

    rg_time_by_eps: Dict[float, np.ndarray] = {}
    rg_median_by_eps: Dict[float, np.ndarray] = {}

    sorted_results = sorted(grouped_results.items())
    for eps_idx, (epsilon, replicate_results) in enumerate(sorted_results, start=1):
        log(
            f"Aggregating epsilon group {eps_idx}/{len(sorted_results)}: "
            f"eps={epsilon:g}, replicates={len(replicate_results)}"
        )
        epsilon_values.append(epsilon)

        scalar_keys = [
            "bonded_pairs_mean",
            "rate_assoc",
            "rate_dissoc",
            "swap_rate",
            "swap_rate_per_free_sticker",
            "tau_s",
            "rg_mean",
            "end_to_end_mean",
        ]
        scalar_summary: Dict[str, Dict[str, float]] = {}
        for key in scalar_keys:
            vals = [float(res.get(key, float("nan"))) for res in replicate_results]
            mean, stderr = mean_and_stderr(vals)
            scalar_summary[key] = {"mean": mean, "stderr": stderr}

        epsilon_properties = {
            "bonded_pair_count": scalar_summary["bonded_pairs_mean"],
            "associative_exchange_rate_R_a": scalar_summary["rate_assoc"],
            "passive_dimerization_rate_R_d": scalar_summary["rate_dissoc"],
            "swap_rate": scalar_summary["swap_rate"],
            "swap_rate_per_free_sticker": scalar_summary["swap_rate_per_free_sticker"],
            "bond_persistence_time_tau_s": scalar_summary["tau_s"],
            "radius_of_gyration_R_g": scalar_summary["rg_mean"],
            "end_to_end_distance": scalar_summary["end_to_end_mean"],
        }

        eps_dir = os.path.join(args.output_dir, f"eps_{epsilon:g}")
        os.makedirs(eps_dir, exist_ok=True)

        with open(os.path.join(eps_dir, "properties.json"), "w", encoding="utf-8") as handle:
            json.dump(epsilon_properties, handle, indent=2)
        write_properties_csv(os.path.join(eps_dir, "properties.csv"), epsilon_properties)

        cs_time, cs_values, cs_mean, cs_stderr = aggregate_timeseries(
            replicate_results, "cs_time", "cs", epsilon
        )
        if cs_time is not None and cs_values is not None:
            write_timeseries(
                os.path.join(eps_dir, "bond_correlation.csv"),
                cs_time,
                cs_mean,
                cs_stderr,
            )
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "bond_correlation_fit.png"),
                cs_time,
                cs_values,
                title=f"Bond Correlation Decay (eps={epsilon:g})",
                y_label="C_s(t)",
            )

        rg_time, rg_values, rg_mean, rg_stderr = aggregate_timeseries(
            replicate_results, "rg_time", "rg_series", epsilon
        )
        if rg_time is not None and rg_values is not None:
            rg_median = np.median(rg_values, axis=0)
            rg_q1 = np.percentile(rg_values, 25.0, axis=0)
            rg_q3 = np.percentile(rg_values, 75.0, axis=0)
            write_timeseries(
                os.path.join(eps_dir, "rg_time_mean.csv"),
                rg_time,
                rg_mean,
                rg_stderr,
            )
            write_timeseries_quantiles(
                os.path.join(eps_dir, "rg_time_quantiles.csv"),
                rg_time,
                rg_median,
                rg_q1,
                rg_q3,
            )
            write_rg_summary_plot(
                os.path.join(eps_dir, "rg_time_summary.png"),
                rg_time,
                rg_values,
                epsilon,
            )
            rg_time_by_eps[epsilon] = rg_time
            rg_median_by_eps[epsilon] = rg_median

        chain_length = int(replicate_results[0].get("chain_length", 0))
        loop_all = np.concatenate([res["loop_length_series"] for res in replicate_results])
        loop_hist_path = os.path.join(eps_dir, "loop_length_distribution.csv")
        with open(loop_hist_path, "w", encoding="utf-8") as handle:
            handle.write("loop_length,probability\n")
            if loop_all.size > 0 and chain_length > 1:
                counts = np.bincount(loop_all.astype(np.int64), minlength=chain_length)
                total = float(np.sum(counts[1:]))
                if total > 0.0:
                    for loop_length in range(1, chain_length):
                        prob = counts[loop_length] / total
                        handle.write(f"{loop_length},{prob:.6e}\n")

        bonded_pairs_all = np.concatenate([res["bond_count_series"] for res in replicate_results])
        rg_all = np.concatenate([res["rg_series"] for res in replicate_results])
        ree_all = np.concatenate([res["end_to_end_series"] for res in replicate_results])
        frac0_all = np.concatenate([res["frac_bond0_series"] for res in replicate_results])
        frac1_all = np.concatenate([res["frac_bond1_series"] for res in replicate_results])
        frac_gt1_all = np.concatenate([res["frac_bond_gt1_series"] for res in replicate_results])

        bonded_pairs_data.append(bonded_pairs_all)
        rg_data.append(rg_all)
        end_to_end_data.append(ree_all)
        valence0_data.append(frac0_all)
        valence1_data.append(frac1_all)
        valence_gt1_data.append(frac_gt1_all)
        loop_length_data.append(loop_all.astype(np.float64))

        assoc_rate_data.append(
            np.asarray([float(res.get("rate_assoc", float("nan"))) for res in replicate_results], dtype=np.float64)
        )
        dissoc_rate_data.append(
            np.asarray([float(res.get("rate_dissoc", float("nan"))) for res in replicate_results], dtype=np.float64)
        )
        swap_rate_data.append(
            np.asarray([float(res.get("swap_rate", float("nan"))) for res in replicate_results], dtype=np.float64)
        )
        swap_rate_per_free_data.append(
            np.asarray(
                [float(res.get("swap_rate_per_free_sticker", float("nan"))) for res in replicate_results],
                dtype=np.float64,
            )
        )
        tau_s_data.append(
            np.asarray([float(res.get("tau_s", float("nan"))) for res in replicate_results], dtype=np.float64)
        )

        summary_rows.append(
            {
                "epsilon": epsilon,
                **{f"{k}_mean": v["mean"] for k, v in scalar_summary.items()},
                **{f"{k}_stderr": v["stderr"] for k, v in scalar_summary.items()},
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
            bonded_pairs_data,
            title="Bonded Pair Count vs epsilon",
            y_label="sticker-sticker bonded pairs",
        )
        write_dual_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "exchange_rate_comparison_vs_epsilon.png"),
            epsilon_values,
            assoc_rate_data,
            dissoc_rate_data,
            title="Associative Exchange vs Passive Dimerization Rates",
            y_label="rate (1/time)",
            left_label="Associative exchange rate (R_a)",
            right_label="Passive dimerization rate (R_d)",
            left_color="#e77500",
            right_color="#121212",
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "rg_violin_vs_epsilon.png"),
            epsilon_values,
            rg_data,
            title="R_g Distribution vs epsilon",
            y_label=r"$R_\mathrm{g}$",
        )
        write_rg_time_vs_epsilon_plot(
            os.path.join(args.output_dir, "rg_time_median_vs_epsilon.png"),
            epsilon_values,
            rg_time_by_eps,
            rg_median_by_eps,
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "swap_rate_vs_epsilon.png"),
            epsilon_values,
            swap_rate_data,
            y_label="swap rate (1/time)",
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "swap_rate_per_free_sticker_vs_epsilon.png"),
            epsilon_values,
            swap_rate_per_free_data,
            y_label="swap rate / free sticker (1/time)",
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "tau_s_vs_epsilon.png"),
            epsilon_values,
            tau_s_data,
            y_label="bond persistence time",
            yscale="log",
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond0_violin.png"),
            epsilon_values,
            valence0_data,
            y_label="fraction of stickers",
            ylim=(0.0, 1.0),
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond1_violin.png"),
            epsilon_values,
            valence1_data,
            y_label="fraction of stickers",
            ylim=(0.0, 1.0),
        )
        write_median_iqr_line_vs_epsilon_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond_gt1_violin.png"),
            epsilon_values,
            valence_gt1_data,
            y_label="fraction of stickers",
            ylim=(0.0, 1.0),
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "loop_length_violin_vs_epsilon.png"),
            epsilon_values,
            loop_length_data,
            title="Loop Length Distribution vs epsilon",
            y_label="loop length (index separation)",
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "end_to_end_violin_vs_epsilon.png"),
            epsilon_values,
            end_to_end_data,
            title="End-to-End Distance vs epsilon",
            y_label="end-to-end distance",
        )

    log("Single-chain analysis complete")


if __name__ == "__main__":
    main()
