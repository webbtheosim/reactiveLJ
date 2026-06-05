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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FIGSIZE = (2, 2)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "sticky_bond_lifetime_vs_epsilon.svg"
MIN_RESOLVED_EPSILON = 12.0
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
) -> tuple[np.ndarray, np.ndarray]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

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

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    tau_s_arr = np.asarray(tau_s_values, dtype=np.float64)[order]
    return epsilon_arr, tau_s_arr


def remove_legacy_outputs(output_path: Path) -> None:
    for filename in LEGACY_OUTPUT_NAMES:
        candidate = output_path.with_name(filename)
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    args = parse_args()
    epsilon, tau_s = load_summary_points(
        args.summary_csv,
        args.min_resolved_epsilon,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(args.output)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    ax.plot(
        epsilon,
        tau_s,
        color="#2b2b2b",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\tau_s$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(epsilon)
    ax.set_xticklabels([f"{value:g}" for value in epsilon])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)

    print(f"Wrote sticky-bond lifetime plot to {args.output}", flush=True)


if __name__ == "__main__":
    main()
