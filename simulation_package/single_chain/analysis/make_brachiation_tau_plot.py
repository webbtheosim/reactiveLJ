#!/usr/bin/env python3
"""Plot single-chain persistence times versus epsilon from cached summary data."""

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
import ultraplot as uplt


POINTS_PER_INCH = 72.0
FIGURE_WIDTH_PT = 237.6
FIGURE_HEIGHT_PT = 144.0
AXES_LEFT_PT = 35.55
AXES_BOTTOM_PT = 27.66
AXES_WIDTH_PT = 197.55
AXES_HEIGHT_PT = 109.609905
DEFAULT_FIGSIZE = (
    FIGURE_WIDTH_PT / POINTS_PER_INCH,
    FIGURE_HEIGHT_PT / POINTS_PER_INCH,
)
DEFAULT_DPI = 600
DEFAULT_TICK_FONTSIZE = 10
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "brachiation_tau_vs_epsilon.svg"
DEFAULT_TAU_R0 = 4041.0
TAU_S_EXCLUDED_EPSILONS = (0.0, 6.0)
X_AXIS_LABEL = r"Sticker strength, $\varepsilon_\mathrm{RLJ}/\varepsilon_0$"
Y_AXIS_LABEL = r"Persistence time $\tau_\alpha/\tau_R^{(0)}$"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot single-chain persistence times versus epsilon from summary.csv."
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
        "--tau-r0",
        type=float,
        default=DEFAULT_TAU_R0,
        help="Reference Rouse time used for normalization.",
    )
    return parser.parse_args()


def epsilon_is_excluded(epsilon: float, excluded_values: tuple[float, ...]) -> bool:
    return any(np.isclose(float(epsilon), excluded) for excluded in excluded_values)


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def load_summary_rows(summary_csv: Path) -> list[tuple[float, float, float]]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    rows: list[tuple[float, float, float]] = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"epsilon", "tau_s_mean", "tau_b_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{summary_csv} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            rows.append(
                (
                    float(row["epsilon"]),
                    float(row["tau_s_mean"]),
                    float(row["tau_b_mean"]),
                )
            )
    if not rows:
        raise ValueError(f"No persistence-time rows found in {summary_csv}")
    return rows


def write_tau_bar_plot(
    output: Path | str,
    summary_rows: list[tuple[float, float, float]],
    tau_r0: float,
) -> None:
    rows: list[tuple[float, float, float]] = []
    for epsilon, tau_s, tau_b in summary_rows:
        tau_s_value = float("nan")
        if (
            not epsilon_is_excluded(epsilon, TAU_S_EXCLUDED_EPSILONS)
            and np.isfinite(tau_s)
            and tau_s > 0.0
        ):
            tau_s_value = tau_s / tau_r0
        tau_b_value = float("nan")
        if np.isfinite(tau_b) and tau_b > 0.0:
            tau_b_value = tau_b / tau_r0
        if np.isfinite(epsilon) and (np.isfinite(tau_s_value) or np.isfinite(tau_b_value)):
            rows.append((epsilon, tau_s_value, tau_b_value))

    if not rows:
        raise ValueError("No finite positive persistence-time values found for plotting.")

    rows.sort(key=lambda item: item[0])
    epsilon_values = np.asarray([row[0] for row in rows], dtype=np.float64)
    tau_s_values = np.asarray([row[1] for row in rows], dtype=np.float64)
    tau_b_values = np.asarray([row[2] for row in rows], dtype=np.float64)

    positions = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    positive_values = np.concatenate(
        (tau_s_values[np.isfinite(tau_s_values)], tau_b_values[np.isfinite(tau_b_values)])
    )
    y_floor = 10.0 ** np.floor(np.log10(np.min(positive_values)))

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    tau_s_specs: list[tuple[float, float]] = []
    tau_b_specs: list[tuple[float, float]] = []
    for center, tau_s_value, tau_b_value in zip(positions, tau_s_values, tau_b_values):
        tau_s_resolved = np.isfinite(tau_s_value)
        tau_b_resolved = np.isfinite(tau_b_value)
        if tau_s_resolved and tau_b_resolved:
            tau_s_specs.append((center - width / 2.0, float(tau_s_value)))
            tau_b_specs.append((center + width / 2.0, float(tau_b_value)))
        elif tau_s_resolved:
            tau_s_specs.append((center, float(tau_s_value)))
        elif tau_b_resolved:
            tau_b_specs.append((center, float(tau_b_value)))

    if tau_s_specs:
        tau_s_positions = np.asarray([spec[0] for spec in tau_s_specs], dtype=np.float64)
        tau_s_plot_values = np.asarray([spec[1] for spec in tau_s_specs], dtype=np.float64)
        ax.bar(
            tau_s_positions,
            tau_s_plot_values,
            width=width,
            bottom=y_floor,
            color="#e77500",
            edgecolor="black",
            linewidth=0.5,
            label=r"$\tau_s$",
            zorder=3,
        )
    if tau_b_specs:
        tau_b_positions = np.asarray([spec[0] for spec in tau_b_specs], dtype=np.float64)
        tau_b_plot_values = np.asarray([spec[1] for spec in tau_b_specs], dtype=np.float64)
        ax.bar(
            tau_b_positions,
            tau_b_plot_values,
            width=width,
            bottom=y_floor,
            color="#121212",
            edgecolor="black",
            linewidth=0.5,
            label=r"$\tau_b$",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            "None" if np.isclose(float(epsilon), 0.0, rtol=0.0, atol=1.0e-12) else f"{epsilon:g}"
            for epsilon in epsilon_values
        ],
        fontsize=DEFAULT_TICK_FONTSIZE,
    )
    ax.set_yscale("log")
    ax.set_ylim(y_floor, float(np.max(positive_values) * 1.3))
    ax.set_xlabel(X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.tick_params(axis="x", which="both", length=0)
    ax.legend(frameon=False, fontsize=DEFAULT_LEGEND_FONTSIZE, loc="best")
    set_target_axes_position(ax)
    fig.savefig(output)
    uplt.close(fig)

    print(f"Wrote brachiation tau plot to {output}", flush=True)


def main() -> None:
    args = parse_args()
    summary_rows = load_summary_rows(args.summary_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_tau_bar_plot(args.output, summary_rows, args.tau_r0)


if __name__ == "__main__":
    main()
