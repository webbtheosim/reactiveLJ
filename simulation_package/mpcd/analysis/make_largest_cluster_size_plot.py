#!/usr/bin/env python3
"""Plot largest cluster size versus ReactiveLJ epsilon."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

USER_TMP_DIR = Path("/tmp") / f"mpcd_analysis_plot_cache_{os.getuid()}"
USER_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(USER_TMP_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(USER_TMP_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import ultraplot as uplt


DEFAULT_FIGSIZE = (3.3, 1.5)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "largest_cluster_size_vs_epsilon.svg"
BAR_COLOR = "#e77500"
EDGE_COLOR = "#121212"
POINTS_PER_INCH = 72.0
FIGURE_WIDTH_PT = 237.6
FIGURE_HEIGHT_PT = 144.0
AXES_LEFT_PT = 51.541515
AXES_BOTTOM_PT = 41.816
AXES_WIDTH_PT = 175.258485
AXES_HEIGHT_PT = 88.344535


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot largest cluster size versus ReactiveLJ epsilon."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=script_dir / "results" / "summary.csv",
        help="Path to the cached analysis summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / DEFAULT_OUTPUT_NAME,
        help="Output SVG path.",
    )
    return parser.parse_args()


def epsilon_category_labels(epsilon: np.ndarray) -> list[str]:
    return ["None" if np.isclose(value, 0.0) else f"{value:g}" for value in epsilon]


def summarize_replicate_points(
    epsilon_values: list[float] | np.ndarray,
    data: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    epsilons: list[float] = []
    means: list[float] = []
    for epsilon, values in zip(epsilon_values, data):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        epsilons.append(float(epsilon))
        means.append(float(np.mean(arr)))

    if not epsilons:
        raise ValueError("No finite largest-cluster-size rows found")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    mean_arr = np.asarray(means, dtype=np.float64)[order]
    return epsilon_arr, mean_arr


def load_summary_points(summary_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    epsilons: list[float] = []
    means: list[float] = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"epsilon", "largest_cluster_size_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{summary_csv} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            epsilon = float(row["epsilon"])
            mean = float(row["largest_cluster_size_mean"])
            if np.isfinite(epsilon) and np.isfinite(mean):
                epsilons.append(epsilon)
                means.append(mean)

    if not epsilons:
        raise ValueError(f"No finite largest-cluster-size rows found in {summary_csv}")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    mean_arr = np.asarray(means, dtype=np.float64)[order]
    return epsilon_arr, mean_arr


def write_largest_cluster_size_plot(
    output: Path | str,
    epsilon: np.ndarray,
    largest_cluster_size: np.ndarray,
) -> None:
    def set_target_axes_position(ax) -> None:
        ax.set_position(
            [
                AXES_LEFT_PT / FIGURE_WIDTH_PT,
                AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
                AXES_WIDTH_PT / FIGURE_WIDTH_PT,
                AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
            ]
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon = np.asarray(epsilon, dtype=np.float64)
    largest_cluster_size = np.asarray(largest_cluster_size, dtype=np.float64)
    mask = np.isfinite(epsilon) & np.isfinite(largest_cluster_size)
    epsilon = epsilon[mask]
    largest_cluster_size = largest_cluster_size[mask]
    if epsilon.size == 0:
        raise ValueError("No finite largest-cluster-size points to plot")

    order = np.argsort(epsilon)
    epsilon = epsilon[order]
    largest_cluster_size = largest_cluster_size[order]

    fig, ax = uplt.subplots(
        figsize=(FIGURE_WIDTH_PT / POINTS_PER_INCH, FIGURE_HEIGHT_PT / POINTS_PER_INCH),
        dpi=DEFAULT_DPI,
        tight=False,
    )
    set_target_axes_position(ax)
    x = np.arange(epsilon.size, dtype=np.float64)
    ax.bar(
        x,
        largest_cluster_size,
        width=0.62,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
    )
    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\langle \max(M) \rangle_t$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_category_labels(epsilon))
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.tick_params(axis="x", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(0.8)
    set_target_axes_position(ax)
    fig.savefig(output_path, format="svg")
    uplt.close(fig)

    print(f"Wrote largest-cluster-size plot to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    epsilon, largest_cluster_size = load_summary_points(args.summary_csv)
    write_largest_cluster_size_plot(args.output, epsilon, largest_cluster_size)


if __name__ == "__main__":
    main()
