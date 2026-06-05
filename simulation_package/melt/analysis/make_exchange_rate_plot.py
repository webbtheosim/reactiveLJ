#!/usr/bin/env python3
"""Plot snapshot-resolved bond-turnover metrics versus epsilon."""

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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FIGSIZE = (3.3, 2)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "exchange_rate_comparison_vs_epsilon.svg"
LEGACY_OUTPUT_NAMES = ("exchange_rate_comparison_vs_epsilon.png",)
ASSOCIATIVE_COLOR = "#e77500"
PASSIVE_COLOR = "#121212"
DUMP_INTERVAL_TAU_LJ = 1000.0


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot snapshot-resolved apparent bond-turnover metrics versus epsilon."
        )
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


def load_summary_points(
    summary_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    epsilons: list[float] = []
    turnover_assoc: list[float] = []
    turnover_dissoc: list[float] = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"epsilon", "rate_assoc_mean", "rate_dissoc_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{summary_csv} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            epsilon = float(row["epsilon"])
            assoc = float(row["rate_assoc_mean"]) * DUMP_INTERVAL_TAU_LJ
            dissoc = float(row["rate_dissoc_mean"]) * DUMP_INTERVAL_TAU_LJ
            if np.isfinite(epsilon) and np.isfinite(assoc) and np.isfinite(dissoc):
                epsilons.append(epsilon)
                turnover_assoc.append(assoc)
                turnover_dissoc.append(dissoc)

    if not epsilons:
        raise ValueError(f"No finite exchange-rate rows found in {summary_csv}")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    assoc_arr = np.asarray(turnover_assoc, dtype=np.float64)[order]
    dissoc_arr = np.asarray(turnover_dissoc, dtype=np.float64)[order]
    return epsilon_arr, assoc_arr, dissoc_arr


def remove_legacy_outputs(output_path: Path) -> None:
    for filename in LEGACY_OUTPUT_NAMES:
        candidate = output_path.with_name(filename)
        if candidate.exists():
            candidate.unlink()


def epsilon_category_labels(epsilon: np.ndarray) -> list[str]:
    return ["None" if np.isclose(value, 0.0) else f"{value:g}" for value in epsilon]


def write_exchange_rate_plot(
    output: Path | str,
    epsilon: np.ndarray,
    turnover_assoc: np.ndarray,
    turnover_dissoc: np.ndarray,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(output_path)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    x = np.arange(epsilon.size, dtype=np.float64)
    width = 0.36
    all_values = np.concatenate((turnover_assoc, turnover_dissoc))
    positive_values = all_values[np.isfinite(all_values) & (all_values > 0.0)]
    if positive_values.size != all_values.size:
        raise ValueError("Exchange-rate bars require finite positive values on a log axis")
    ax.set_yscale("log")
    ax.plot(np.arange(positive_values.size), positive_values, alpha=0.0, linewidth=0.0)
    ax.relim()
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    ax.cla()
    ax.bar(
        x - width / 2.0,
        turnover_assoc - y_min,
        width=width,
        bottom=y_min,
        color=ASSOCIATIVE_COLOR,
        edgecolor=PASSIVE_COLOR,
        linewidth=0.7,
        label="assoc.",
    )
    ax.bar(
        x + width / 2.0,
        turnover_dissoc - y_min,
        width=width,
        bottom=y_min,
        color=PASSIVE_COLOR,
        edgecolor=PASSIVE_COLOR,
        linewidth=0.7,
        label="dissoc.",
    )
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\nu_\mathrm{app}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_category_labels(epsilon))
    ax.set_xlim(-0.5, epsilon.size - 0.5)
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=DEFAULT_LEGEND_FONTSIZE)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Wrote exchange-rate comparison plot to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    epsilon, turnover_assoc, turnover_dissoc = load_summary_points(args.summary_csv)
    write_exchange_rate_plot(args.output, epsilon, turnover_assoc, turnover_dissoc)


if __name__ == "__main__":
    main()
