#!/usr/bin/env python3
"""Plot cluster-size distributions versus epsilon from cached per-epsilon CSV files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

USER_TMP_DIR = Path("/tmp") / f"reactive_lj_plot_cache_{os.getuid()}"
USER_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(USER_TMP_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(USER_TMP_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ultraplot as uplt


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
DEFAULT_DPI = 600
DEFAULT_TICK_FONTSIZE = 10
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "cluster_size_distribution_by_epsilon.svg"
X_AXIS_LABEL = r"Cluster size, $M$"
Y_AXIS_LABEL = r"Probability, $P(M)$"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot cluster-size distributions versus epsilon from cached CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results",
        help="Directory containing eps_*/cluster_distribution.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / DEFAULT_OUTPUT_NAME,
        help="Output SVG path.",
    )
    return parser.parse_args()


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
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


def load_cached_distributions(results_dir: Path) -> tuple[list[float], dict[float, np.ndarray]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    epsilon_values: list[float] = []
    distributions: dict[float, np.ndarray] = {}

    for csv_path in sorted(results_dir.glob("eps_*/cluster_distribution.csv")):
        eps_dir = csv_path.parent.name
        eps_text = eps_dir.removeprefix("eps_")
        epsilon = float(eps_text)
        cluster_sizes: list[int] = []
        probabilities: list[float] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"cluster_size", "mean"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{csv_path} must contain columns {sorted(required)}; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                cluster_sizes.append(int(row["cluster_size"]))
                probabilities.append(float(row["mean"]))

        if not cluster_sizes:
            continue
        max_size = max(cluster_sizes)
        distribution = np.zeros(max_size + 1, dtype=np.float64)
        for size, prob in zip(cluster_sizes, probabilities):
            distribution[size] = prob
        epsilon_values.append(epsilon)
        distributions[epsilon] = distribution

    if not epsilon_values:
        raise ValueError(f"No cached cluster distributions found under {results_dir}")

    epsilon_values.sort()
    return epsilon_values, distributions


def write_cluster_distribution_plot(
    output: Path | str,
    epsilon_values: list[float],
    cluster_distribution_by_eps: dict[float, np.ndarray],
) -> None:
    series: list[tuple[float, np.ndarray, np.ndarray]] = []
    for eps in epsilon_values:
        distribution = cluster_distribution_by_eps.get(eps)
        if distribution is None:
            continue
        cluster_size = np.arange(distribution.size, dtype=np.float64)
        prob = np.asarray(distribution, dtype=np.float64)
        mask = np.isfinite(cluster_size) & np.isfinite(prob) & (cluster_size > 0.0) & (prob > 0.0)
        if not np.any(mask):
            continue
        series.append((eps, cluster_size[mask], prob[mask]))

    if not series:
        raise ValueError("No finite positive cluster-distribution values found for plotting.")

    cmap = plt.get_cmap("plasma", len(series))
    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
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
            edgecolors="black",
            linewidths=0.35,
            label=format_rlj_legend_label(eps),
        )
        max_x = max(max_x, float(np.max(cluster_size)))
        max_y = max(max_y, float(np.max(prob)))
        min_y = min(min_y, float(np.min(prob)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
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
    align_terminal_log_xtick_labels(fig, ax)
    ax.legend(frameon=False, fontsize=DEFAULT_LEGEND_FONTSIZE, ncol=1)
    set_target_axes_position(ax)
    fig.savefig(output)
    uplt.close(fig)

    print(f"Wrote cluster distribution plot to {output}", flush=True)


def main() -> None:
    args = parse_args()
    epsilon_values, distributions = load_cached_distributions(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_cluster_distribution_plot(args.output, epsilon_values, distributions)


if __name__ == "__main__":
    main()
