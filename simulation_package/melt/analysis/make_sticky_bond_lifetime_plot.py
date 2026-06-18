#!/usr/bin/env python3
"""Plot sticky-bond lifetime versus epsilon from cached summary data."""

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
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "sticky_bond_lifetime_vs_epsilon.svg"
DEFAULT_TAU_R0 = 4041.0
MIN_RESOLVED_EPSILON = 12.0
BAR_COLOR = "#e77500"
BAR_EDGE_COLOR = "black"
BAR_WIDTH = 0.62
LEGACY_OUTPUT_NAMES = (
    "ln_bond_tau_vs_epsilon.png",
    "ln_bond_tau_vs_epsilon.svg",
    "bond_tau_vs_epsilon.png",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot sticky-bond lifetime versus epsilon from summary.csv."
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
    parser.add_argument(
        "--min-resolved-epsilon",
        type=float,
        default=MIN_RESOLVED_EPSILON,
        help=(
            "Only plot tau_s values at or above this epsilon. "
            "The default excludes unresolved low-epsilon lifetimes."
        ),
    )
    return parser.parse_args()


def load_summary_points(
    summary_csv: Path,
    min_resolved_epsilon: float = MIN_RESOLVED_EPSILON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    category_epsilons: list[float] = []
    epsilons: list[float] = []
    tau_s_values: list[float] = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"epsilon", "tau_s_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{summary_csv} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            epsilon = float(row["epsilon"])
            tau_s = float(row["tau_s_mean"])
            if np.isfinite(epsilon):
                category_epsilons.append(epsilon)
            if (
                np.isfinite(epsilon)
                and epsilon >= min_resolved_epsilon
                and np.isfinite(tau_s)
                and tau_s > 0.0
            ):
                epsilons.append(epsilon)
                tau_s_values.append(tau_s)

    if not epsilons:
        raise ValueError(
            f"No resolved finite positive tau_s values found in {summary_csv} "
            f"for epsilon >= {min_resolved_epsilon:g}"
        )
    if not category_epsilons:
        raise ValueError(f"No finite epsilon categories found in {summary_csv}")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    category_order = np.argsort(np.asarray(category_epsilons, dtype=np.float64))
    category_arr = np.asarray(category_epsilons, dtype=np.float64)[category_order]
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    tau_s_arr = np.asarray(tau_s_values, dtype=np.float64)[order]
    return category_arr, epsilon_arr, tau_s_arr


def remove_legacy_outputs(output_path: Path) -> None:
    for filename in LEGACY_OUTPUT_NAMES:
        candidate = output_path.with_name(filename)
        if candidate.exists():
            candidate.unlink()


def bar_axis_floor(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("Need at least one finite positive value to set a log-scale bar axis.")

    floor = float(10.0 ** np.floor(np.log10(np.min(positive))))
    if np.isclose(np.min(positive), floor):
        floor /= 10.0
    return floor


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def epsilon_category_labels(epsilon: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in np.asarray(epsilon, dtype=np.float64):
        if np.isclose(value, 0.0, rtol=0.0, atol=1.0e-12):
            labels.append("None")
        else:
            labels.append(f"{value:g}")
    return labels


def main() -> None:
    args = parse_args()
    category_epsilon, epsilon, tau_s = load_summary_points(
        args.summary_csv,
        args.min_resolved_epsilon,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(args.output)

    category_x = np.arange(category_epsilon.size, dtype=np.float64)
    epsilon_to_position = {
        float(eps): float(position) for eps, position in zip(category_epsilon, category_x)
    }
    x = np.asarray([epsilon_to_position[float(value)] for value in epsilon], dtype=np.float64)
    tau_s_normalized = tau_s / DEFAULT_TAU_R0
    y_floor = bar_axis_floor(tau_s_normalized)
    bottoms = np.full(tau_s_normalized.shape, y_floor, dtype=np.float64)

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    ax.bar(
        x,
        tau_s_normalized - bottoms,
        bottom=bottoms,
        width=BAR_WIDTH,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.5,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(y_floor, float(np.max(tau_s_normalized) * 1.3))
    ax.set_xlim(-0.5, category_epsilon.size - 0.5)
    ax.set_xlabel(r"$\varepsilon_\mathrm{RLJ}/\varepsilon_0$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\tau_s/\tau_R^{(0)}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(category_x)
    ax.set_xticklabels(epsilon_category_labels(category_epsilon))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.format(
        xspineloc="both",
        yspineloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    set_target_axes_position(ax)
    fig.savefig(args.output)
    uplt.close(fig)

    print(f"Wrote sticky-bond lifetime plot to {args.output}", flush=True)


if __name__ == "__main__":
    main()
