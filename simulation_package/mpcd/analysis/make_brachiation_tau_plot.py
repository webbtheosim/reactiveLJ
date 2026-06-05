#!/usr/bin/env python3
"""Plot brachiation time versus ReactiveLJ epsilon."""

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
import matplotlib.pyplot as plt


DEFAULT_FIGSIZE = (3.3, 1.5)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "brachiation_tau_vs_epsilon.svg"
BAR_COLOR = "#e77500"
EDGE_COLOR = "#121212"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot brachiation time versus ReactiveLJ epsilon."
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
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size == 0:
            continue
        epsilons.append(float(epsilon))
        means.append(float(np.mean(arr)))

    if not epsilons:
        raise ValueError("No finite positive brachiation-time rows found")

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
        required = {"epsilon", "tau_b_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{summary_csv} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            epsilon = float(row["epsilon"])
            mean = float(row["tau_b_mean"])
            if np.isfinite(epsilon) and np.isfinite(mean) and mean > 0.0:
                epsilons.append(epsilon)
                means.append(mean)

    if not epsilons:
        raise ValueError(f"No finite positive brachiation-time rows found in {summary_csv}")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    mean_arr = np.asarray(means, dtype=np.float64)[order]
    return epsilon_arr, mean_arr


def write_brachiation_tau_plot(
    output: Path | str,
    epsilon: np.ndarray,
    tau_b: np.ndarray,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon = np.asarray(epsilon, dtype=np.float64)
    tau_b = np.asarray(tau_b, dtype=np.float64)
    mask = np.isfinite(epsilon) & np.isfinite(tau_b) & (tau_b > 0.0)
    epsilon = epsilon[mask]
    tau_b = tau_b[mask]
    if epsilon.size == 0:
        raise ValueError("No finite positive brachiation-time points to plot")

    order = np.argsort(epsilon)
    epsilon = epsilon[order]
    tau_b = tau_b[order]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    x = np.arange(epsilon.size, dtype=np.float64)

    ax.set_yscale("log")
    ax.plot(x, tau_b, alpha=0.0, linewidth=0.0)
    ax.relim()
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    ax.cla()

    ax.bar(
        x,
        tau_b - y_min,
        bottom=y_min,
        width=0.62,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
    )
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\tau_b$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_category_labels(epsilon))
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Wrote brachiation-time plot to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    epsilon, tau_b = load_summary_points(args.summary_csv)
    write_brachiation_tau_plot(args.output, epsilon, tau_b)


if __name__ == "__main__":
    main()
