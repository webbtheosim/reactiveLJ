#!/usr/bin/env python3
"""Plot gelation epsilon versus ReactiveLJ epsilon."""

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
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")
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
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 10
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "gelation_epsilon_vs_epsilon.svg"
BAR_COLOR = "#e77500"
EDGE_COLOR = "black"
X_AXIS_LABEL = r"Sticker strength, $\varepsilon_\mathrm{RLJ}/\varepsilon_0$"
Y_AXIS_LABEL = r"Degree of gelation, $\mathcal{E}$"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot degree of gelation versus ReactiveLJ epsilon."
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
    return ["WCA" if np.isclose(value, 0.0) else f"{value:g}" for value in epsilon]


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def summarize_replicate_points(
    epsilon_values: list[float] | np.ndarray,
    data: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilons: list[float] = []
    means: list[float] = []
    stderrs: list[float] = []
    for epsilon, values in zip(epsilon_values, data):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        epsilons.append(float(epsilon))
        means.append(float(np.mean(arr)))
        if arr.size > 1:
            stderrs.append(float(np.std(arr, ddof=1) / np.sqrt(arr.size)))
        else:
            stderrs.append(0.0)

    if not epsilons:
        raise ValueError("No finite gelation-epsilon rows found")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    mean_arr = np.asarray(means, dtype=np.float64)[order]
    stderr_arr = np.asarray(stderrs, dtype=np.float64)[order]
    return epsilon_arr, mean_arr, stderr_arr


def load_summary_points(summary_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    epsilons: list[float] = []
    means: list[float] = []
    stderrs: list[float] = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"epsilon", "epsilon_mean_mean", "epsilon_mean_stderr"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{summary_csv} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            epsilon = float(row["epsilon"])
            mean = float(row["epsilon_mean_mean"])
            stderr = float(row["epsilon_mean_stderr"])
            if np.isfinite(epsilon) and np.isfinite(mean):
                epsilons.append(epsilon)
                means.append(mean)
                stderrs.append(stderr if np.isfinite(stderr) else 0.0)

    if not epsilons:
        raise ValueError(f"No finite gelation-epsilon rows found in {summary_csv}")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    mean_arr = np.asarray(means, dtype=np.float64)[order]
    stderr_arr = np.asarray(stderrs, dtype=np.float64)[order]
    return epsilon_arr, mean_arr, stderr_arr


def write_gelation_epsilon_plot(
    output: Path | str,
    epsilon: np.ndarray,
    gelation_mean: np.ndarray,
    gelation_stderr: np.ndarray | None = None,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon = np.asarray(epsilon, dtype=np.float64)
    gelation_mean = np.asarray(gelation_mean, dtype=np.float64)
    if gelation_stderr is None:
        gelation_stderr = np.zeros_like(gelation_mean)
    else:
        gelation_stderr = np.asarray(gelation_stderr, dtype=np.float64)

    mask = np.isfinite(epsilon) & np.isfinite(gelation_mean)
    mask &= np.isfinite(gelation_stderr)
    epsilon = epsilon[mask]
    gelation_mean = gelation_mean[mask]
    gelation_stderr = gelation_stderr[mask]
    if epsilon.size == 0:
        raise ValueError("No finite gelation-epsilon points to plot")

    order = np.argsort(epsilon)
    epsilon = epsilon[order]
    gelation_mean = gelation_mean[order]
    gelation_stderr = gelation_stderr[order]

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    x = np.arange(epsilon.size, dtype=np.float64)
    ax.bar(
        x,
        gelation_mean,
        width=0.62,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_category_labels(epsilon))
    ax.set_xlim(-0.5, epsilon.size - 0.5)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.format(
        xspineloc="both",
        yspineloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    set_target_axes_position(ax)
    fig.savefig(output_path)
    uplt.close(fig)

    print(f"Wrote gelation-epsilon plot to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    epsilon, gelation_mean, gelation_stderr = load_summary_points(args.summary_csv)
    write_gelation_epsilon_plot(args.output, epsilon, gelation_mean, gelation_stderr)


if __name__ == "__main__":
    main()
