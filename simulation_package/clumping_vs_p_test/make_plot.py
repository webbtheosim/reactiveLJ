#!/usr/bin/env python3
"""Create clump, paired, and excess-coordination violin plots for the p sweep."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

USER_TMP_DIR = Path("/tmp") / f"reactive_lj_plot_cache_{os.getuid()}"
USER_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(USER_TMP_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(USER_TMP_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import gsd.hoomd
import matplotlib
import matplotlib.cbook as cbook
import matplotlib.mlab as mlab
import matplotlib.ticker as mticker
import numpy as np
from joblib import Parallel, delayed

matplotlib.use("Agg")
import ultraplot as uplt


FIGSIZE = (3.3 / 2.0, 3.3 / 2.0)
DPI = 1000
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 8
VIOLIN_FILL = "#e77500"
OUTLINE_COLOR = "#121212"
POINTS_PER_INCH = 72.0
FIGURE_WIDTH_PT = 118.8
FIGURE_HEIGHT_PT = 118.8
AXES_LEFT_PT = 49.349
AXES_BOTTOM_PT = 39.224
AXES_WIDTH_PT = 58.651
AXES_HEIGHT_PT = 57.5412
WIDE_FIGURE_WIDTH_PT = 237.6
WIDE_FIGURE_HEIGHT_PT = 72.0
WIDE_AXES_LEFT_PT = 35.369779
WIDE_AXES_BOTTOM_PT = 22.0
WIDE_AXES_WIDTH_PT = 197.730221
WIDE_AXES_HEIGHT_PT = 40.0


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Read the clumping-test GSD files and plot the framewise clump "
            "fraction distribution versus weakening exponent p."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=script_dir / "outputs",
        help="Root directory containing p_*/rep_*/trajectory.gsd outputs.",
    )
    parser.add_argument(
        "--clump-output",
        type=Path,
        default=script_dir / "results" / "clump_fraction_violin.svg",
        help="Output SVG path for the clump-fraction violin plot.",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=script_dir / "results" / "paired_fraction_violin.svg",
        help="Output SVG path for the paired-fraction violin plot.",
    )
    parser.add_argument(
        "--excess-coordination-output",
        type=Path,
        default=script_dir / "results" / "excess_coordination_violin.svg",
        help="Output SVG path for the excess-coordination violin plot.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Parallel jobs over trajectories (0 uses SLURM_CPUS_PER_TASK or 1).",
    )
    parser.add_argument(
        "--p-values",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of p values to include, for example: --p-values 2 4 8.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "clump", "paired", "excess-coordination"),
        default="all",
        help="Optional plot selection. Default writes all plots.",
    )
    return parser.parse_args()


def lj_inflection_cutoff(sigma: float = 1.0) -> float:
    return float(sigma) * (26.0 / 7.0) ** (1.0 / 6.0)


def resolve_n_jobs(requested_n_jobs: int) -> int:
    if requested_n_jobs > 0:
        return requested_n_jobs
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return 1


def infer_p_value(run_dir: Path, metadata: dict | None) -> int:
    if metadata is not None:
        if "clumping_test_p" in metadata:
            return int(metadata["clumping_test_p"])
        if "weakening_exponent" in metadata:
            return int(round(float(metadata["weakening_exponent"])))

    for part in run_dir.parts:
        match = re.fullmatch(r"p_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer p value for run directory {run_dir}")


def discover_runs(
    input_root: Path,
    selected_p_values: set[int] | None = None,
) -> list[tuple[int, Path, dict | None]]:
    runs: list[tuple[int, Path, dict | None]] = []
    for trajectory_path in sorted(input_root.rglob("trajectory.gsd")):
        run_dir = trajectory_path.parent
        metadata_path = run_dir / "metadata.json"
        metadata = None
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        p_value = infer_p_value(run_dir, metadata)
        if selected_p_values is not None and p_value not in selected_p_values:
            continue
        runs.append((p_value, trajectory_path, metadata))
    if not runs:
        raise FileNotFoundError(f"No trajectory.gsd files found under {input_root}")
    if selected_p_values is not None:
        discovered_p_values = {p_value for p_value, _, _ in runs}
        missing_p_values = sorted(selected_p_values - discovered_p_values)
        if missing_p_values:
            raise FileNotFoundError(
                "No trajectory.gsd files found for p value(s): "
                + ", ".join(str(value) for value in missing_p_values)
            )
    return sorted(runs, key=lambda item: (item[0], str(item[1])))


def compute_frame_metrics(
    positions: np.ndarray,
    box_length: float,
    cutoff_sq: float,
) -> tuple[float, float, float]:
    n_stickers = int(positions.shape[0])
    if n_stickers == 0:
        raise ValueError("Cannot compute frame metrics for a frame with zero stickers.")
    if n_stickers == 1:
        return 0.0, 0.0, 0.0

    delta = positions[:, None, :] - positions[None, :, :]
    delta -= box_length * np.rint(delta / box_length)
    dist_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    bonded = (dist_sq <= cutoff_sq) & (dist_sq > 0.0)
    bond_count = bonded.sum(axis=1)
    multi_bonded = bond_count >= 2
    isolated_dimer_members = np.zeros(n_stickers, dtype=bool)
    singly_bonded = bond_count == 1
    if np.any(singly_bonded):
        singly_indices = np.flatnonzero(singly_bonded)
        neighbor_indices = np.argmax(bonded[singly_indices], axis=1)
        isolated_mask = bond_count[neighbor_indices] == 1
        isolated_dimer_members[singly_indices] = isolated_mask
    if np.any(multi_bonded):
        clumped = multi_bonded | bonded[:, multi_bonded].any(axis=1)
    else:
        clumped = multi_bonded
    excess_coordination = np.maximum(bond_count - 1, 0).sum(dtype=np.int64) / n_stickers
    return (
        float(np.count_nonzero(clumped) / n_stickers),
        float(np.count_nonzero(isolated_dimer_members) / n_stickers),
        float(excess_coordination),
    )


def analyze_trajectory(
    p_value: int,
    trajectory_path: Path,
    metadata: dict | None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    sigma = 1.0
    if metadata is not None and metadata.get("reactive_sigma") is not None:
        sigma = float(metadata["reactive_sigma"])
    cutoff = lj_inflection_cutoff(sigma)
    cutoff_sq = cutoff * cutoff

    clump_fractions: list[float] = []
    paired_fractions: list[float] = []
    excess_coordination_values: list[float] = []
    with gsd.hoomd.open(str(trajectory_path), "r") as trajectory:
        for frame in trajectory:
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_length = float(frame.configuration.box[0])
            clump_fraction, paired_fraction, excess_coordination = compute_frame_metrics(
                positions=positions,
                box_length=box_length,
                cutoff_sq=cutoff_sq,
            )
            clump_fractions.append(clump_fraction)
            paired_fractions.append(paired_fraction)
            excess_coordination_values.append(excess_coordination)

    if not clump_fractions:
        raise RuntimeError(f"Trajectory contains no frames: {trajectory_path}")
    return (
        p_value,
        np.asarray(clump_fractions, dtype=np.float64),
        np.asarray(paired_fractions, dtype=np.float64),
        np.asarray(excess_coordination_values, dtype=np.float64),
    )


def paired_fraction_transform_epsilon(grouped_data: list[tuple[int, np.ndarray]]) -> float:
    finite_values = np.concatenate(
        [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for _, values in grouped_data]
    )
    below_one = finite_values[finite_values < (1.0 - 1.0e-12)]
    if below_one.size == 0:
        return 1.0e-3
    smallest_gap = float(np.min(1.0 - below_one))
    return max(0.5 * smallest_gap, 1.0e-6)


def stretch_near_one(values: np.ndarray, epsilon: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.minimum(arr, 1.0 - epsilon)
    return -np.log10(1.0 - clipped)


def paired_fraction_tick_spec(epsilon: float) -> tuple[np.ndarray, list[str]]:
    tick_values = np.asarray([0.0, 0.9, 0.99, 1.0 - epsilon], dtype=np.float64)
    tick_positions = stretch_near_one(tick_values, epsilon)
    tick_labels = ["0", "0.9", "0.99", "1"]
    return tick_positions, tick_labels


def symlog_tick_labels(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in np.asarray(values, dtype=np.float64):
        if np.isclose(value, 0.0):
            labels.append("0")
            continue
        exponent = int(np.round(np.log10(abs(float(value)))))
        if exponent == 0:
            labels.append("1")
        elif exponent == 1:
            labels.append("10")
        else:
            labels.append(rf"$10^{{{exponent}}}$")
    return labels


def make_violin_plot(
    grouped_data: list[tuple[int, np.ndarray]],
    output_path: Path,
    y_label: str,
    y_limits: tuple[float, float] | None = None,
    *,
    figure_width_pt: float = FIGURE_WIDTH_PT,
    figure_height_pt: float = FIGURE_HEIGHT_PT,
    axes_left_pt: float = AXES_LEFT_PT,
    axes_bottom_pt: float = AXES_BOTTOM_PT,
    axes_width_pt: float = AXES_WIDTH_PT,
    axes_height_pt: float = AXES_HEIGHT_PT,
    y_ticks: np.ndarray | None = None,
    y_tick_labels: list[str] | None = None,
) -> None:
    def set_target_axes_position(ax) -> None:
        ax.set_position(
            [
                axes_left_pt / figure_width_pt,
                axes_bottom_pt / figure_height_pt,
                axes_width_pt / figure_width_pt,
                axes_height_pt / figure_height_pt,
            ]
        )

    category_values = [item[0] for item in grouped_data]
    positions = np.arange(len(grouped_data), dtype=np.float64)
    datasets = [item[1] for item in grouped_data]
    finite_values = np.concatenate(
        [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in datasets]
    )

    fig, axs = uplt.subplots(
        figsize=(figure_width_pt / POINTS_PER_INCH, figure_height_pt / POINTS_PER_INCH),
        dpi=DPI,
        tight=False,
    )
    ax = axs[0]
    set_target_axes_position(ax)
    def kde_method(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
        values = cbook._unpack_to_numpy(values)
        if np.all(values[0] == values):
            return (values[0] == coords).astype(float)
        kde = mlab.GaussianKDE(values, "scott")
        return kde.evaluate(coords)

    violin_stats = cbook.violin_stats(datasets, kde_method, points=100)
    max_width = 0.8
    for position, stats in zip(positions, violin_stats):
        vals = np.asarray(stats["vals"], dtype=np.float64)
        coords = np.asarray(stats["coords"], dtype=np.float64)
        if vals.size == 0 or coords.size == 0:
            continue
        peak = float(np.max(vals))
        if not np.isfinite(peak) or peak <= 0.0:
            continue
        half_width = 0.5 * max_width * vals / peak
        left = position - half_width
        right = position + half_width
        ax.fill_betweenx(
            coords,
            left,
            right,
            facecolor=VIOLIN_FILL,
            edgecolor=OUTLINE_COLOR,
            linewidth=0.8,
            alpha=1.0,
            zorder=2,
        )

    ax.set_xlabel(r"$p$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in category_values])
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif finite_values.size > 0:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if np.isclose(y_min, y_max):
            pad = 0.05 * max(abs(y_max), 1.0)
        else:
            pad = 0.04 * (y_max - y_min)
        y_lower = y_min - pad
        y_upper = y_max + pad
        ax.set_ylim(y_lower, y_upper)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        if y_tick_labels is not None:
            ax.set_yticklabels(y_tick_labels)
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE, colors=OUTLINE_COLOR)
    ax.tick_params(axis="x", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_color(OUTLINE_COLOR)
        spine.set_linewidth(0.8)

    set_target_axes_position(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    uplt.close(fig)


def make_mean_bar_plot(
    grouped_data: list[tuple[int, np.ndarray]],
    output_path: Path,
    y_label: str,
    *,
    figure_width_pt: float,
    figure_height_pt: float,
    axes_left_pt: float,
    axes_bottom_pt: float,
    axes_width_pt: float,
    axes_height_pt: float,
) -> None:
    def set_target_axes_position(ax) -> None:
        ax.set_position(
            [
                axes_left_pt / figure_width_pt,
                axes_bottom_pt / figure_height_pt,
                axes_width_pt / figure_width_pt,
                axes_height_pt / figure_height_pt,
            ]
        )

    category_values = [item[0] for item in grouped_data]
    means = np.asarray(
        [float(np.mean(np.asarray(values, dtype=np.float64))) for _, values in grouped_data],
        dtype=np.float64,
    )
    x = np.arange(len(grouped_data), dtype=np.float64)

    positive_means = means[means > 0.0]
    linthresh = 1.0e-4
    if positive_means.size > 0:
        linthresh = min(linthresh, float(np.min(positive_means)))
    upper = float(np.max(means)) if means.size else 1.0
    upper = max(upper * 1.2, linthresh * 10.0)

    fig, axs = uplt.subplots(
        figsize=(figure_width_pt / POINTS_PER_INCH, figure_height_pt / POINTS_PER_INCH),
        dpi=DPI,
        tight=False,
    )
    ax = axs[0]
    set_target_axes_position(ax)
    ax.bar(
        x,
        means,
        width=0.62,
        color=VIOLIN_FILL,
        edgecolor=OUTLINE_COLOR,
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel(r"$p$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in category_values])
    ax.set_xlim(-0.5, len(grouped_data) - 0.5)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10.0)
    ax.set_ylim(0.0, upper)
    tick_values = np.asarray([0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0], dtype=np.float64)
    tick_values = tick_values[tick_values <= upper * 1.0001]
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_values))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(symlog_tick_labels(tick_values)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE, colors=OUTLINE_COLOR)
    ax.tick_params(axis="x", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_color(OUTLINE_COLOR)
        spine.set_linewidth(0.8)

    set_target_axes_position(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    uplt.close(fig)


def main() -> None:
    args = parse_args()
    selected_p_values = None if args.p_values is None else set(args.p_values)
    runs = discover_runs(args.input_root, selected_p_values=selected_p_values)
    n_jobs = min(resolve_n_jobs(args.n_jobs), len(runs))

    analyzed = Parallel(n_jobs=n_jobs)(
        delayed(analyze_trajectory)(p_value, trajectory_path, metadata)
        for p_value, trajectory_path, metadata in runs
    )

    clump_grouped: dict[int, list[np.ndarray]] = {}
    paired_grouped: dict[int, list[np.ndarray]] = {}
    excess_coordination_grouped: dict[int, list[np.ndarray]] = {}
    for (
        p_value,
        clump_fractions,
        paired_fractions,
        excess_coordination_values,
    ) in analyzed:
        clump_grouped.setdefault(p_value, []).append(clump_fractions)
        paired_grouped.setdefault(p_value, []).append(paired_fractions)
        excess_coordination_grouped.setdefault(p_value, []).append(
            excess_coordination_values
        )

    clump_grouped_data = [
        (p_value, np.concatenate(clump_grouped[p_value]))
        for p_value in sorted(clump_grouped)
    ]
    paired_grouped_data = [
        (p_value, np.concatenate(paired_grouped[p_value]))
        for p_value in sorted(paired_grouped)
    ]
    excess_coordination_grouped_data = [
        (p_value, np.concatenate(excess_coordination_grouped[p_value]))
        for p_value in sorted(excess_coordination_grouped)
    ]

    if args.only in {"all", "clump"}:
        for p_value, fractions in clump_grouped_data:
            print(
                f"p={p_value} frames={fractions.size} "
                f"mean_clump_fraction={np.mean(fractions):.6f}",
                flush=True,
            )
        make_violin_plot(
            clump_grouped_data,
            args.clump_output,
            y_label="Clump fraction",
            y_limits=(0.0, 1.0),
        )
        print(f"Wrote clump-fraction plot to {args.clump_output}", flush=True)

    if args.only in {"all", "paired"}:
        for p_value, fractions in paired_grouped_data:
            print(
                f"p={p_value} frames={fractions.size} "
                f"mean_paired_fraction={np.mean(fractions):.6f}",
                flush=True,
            )
        paired_epsilon = paired_fraction_transform_epsilon(paired_grouped_data)
        paired_plot_data = [
            (p_value, stretch_near_one(fractions, paired_epsilon))
            for p_value, fractions in paired_grouped_data
        ]
        paired_y_ticks, paired_y_tick_labels = paired_fraction_tick_spec(paired_epsilon)
        make_violin_plot(
            paired_plot_data,
            args.paired_output,
            y_label=r"$\phi_{\mathrm{pair}}$",
            figure_width_pt=WIDE_FIGURE_WIDTH_PT,
            figure_height_pt=WIDE_FIGURE_HEIGHT_PT,
            axes_left_pt=WIDE_AXES_LEFT_PT,
            axes_bottom_pt=WIDE_AXES_BOTTOM_PT,
            axes_width_pt=WIDE_AXES_WIDTH_PT,
            axes_height_pt=WIDE_AXES_HEIGHT_PT,
            y_ticks=paired_y_ticks,
            y_tick_labels=paired_y_tick_labels,
        )
        print(f"Wrote paired-fraction plot to {args.paired_output}", flush=True)

    if args.only in {"all", "excess-coordination"}:
        for p_value, values in excess_coordination_grouped_data:
            print(
                f"p={p_value} frames={values.size} "
                f"mean_excess_coordination={np.mean(values):.6f}",
                flush=True,
            )
        make_mean_bar_plot(
            excess_coordination_grouped_data,
            args.excess_coordination_output,
            y_label=r"$\langle C_{ij}^{\mathrm{exc}} \rangle$",
            figure_width_pt=WIDE_FIGURE_WIDTH_PT,
            figure_height_pt=WIDE_FIGURE_HEIGHT_PT,
            axes_left_pt=WIDE_AXES_LEFT_PT,
            axes_bottom_pt=WIDE_AXES_BOTTOM_PT,
            axes_width_pt=WIDE_AXES_WIDTH_PT,
            axes_height_pt=WIDE_AXES_HEIGHT_PT,
        )
        print(
            f"Wrote excess-coordination plot to {args.excess_coordination_output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
