#!/usr/bin/env python3
"""Plot single-chain exchange rates versus epsilon from cached summary data."""

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
DEFAULT_OUTPUT_NAME = "exchange_rate_comparison_vs_epsilon.svg"
ASSOCIATIVE_COLOR = "#e77500"
PASSIVE_COLOR = "#121212"
X_AXIS_LABEL = r"Sticker strength, $\varepsilon_\mathrm{RLJ}/\varepsilon_0$"
Y_AXIS_LABEL = r"Turnover rate, $\nu_\alpha$"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot single-chain exchange rates versus epsilon from summary.csv."
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
    labels: list[str] = []
    for value in np.asarray(epsilon, dtype=np.float64):
        if np.isclose(value, 0.0, rtol=0.0, atol=1.0e-12):
            labels.append("WCA")
        else:
            labels.append(f"{value:g}")
    return labels


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def load_summary_points(summary_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    epsilons: list[float] = []
    assoc_rates: list[float] = []
    dissoc_rates: list[float] = []
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
            assoc = float(row["rate_assoc_mean"])
            dissoc = float(row["rate_dissoc_mean"])
            if np.isfinite(epsilon):
                epsilons.append(epsilon)
                assoc_rates.append(assoc)
                dissoc_rates.append(dissoc)

    if not epsilons:
        raise ValueError(f"No finite exchange-rate rows found in {summary_csv}")

    order = np.argsort(np.asarray(epsilons, dtype=np.float64))
    epsilon_arr = np.asarray(epsilons, dtype=np.float64)[order]
    assoc_arr = np.asarray(assoc_rates, dtype=np.float64)[order]
    dissoc_arr = np.asarray(dissoc_rates, dtype=np.float64)[order]
    return epsilon_arr, assoc_arr, dissoc_arr


def write_exchange_rate_plot(
    output: Path | str,
    epsilon: np.ndarray,
    assoc_rates: np.ndarray,
    dissoc_rates: np.ndarray,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions: list[float] = []
    labels: list[str] = []
    assoc_values: list[float] = []
    dissoc_values: list[float] = []
    positive_values: list[float] = []

    for eps, assoc, dissoc in zip(epsilon, assoc_rates, dissoc_rates):
        assoc_ok = np.isfinite(assoc) and assoc > 0.0
        dissoc_ok = np.isfinite(dissoc) and dissoc > 0.0
        if not (assoc_ok or dissoc_ok):
            continue
        positions.append(float(len(positions)))
        labels.append("WCA" if np.isclose(float(eps), 0.0, rtol=0.0, atol=1.0e-12) else f"{eps:g}")
        assoc_values.append(float(assoc) if assoc_ok else float("nan"))
        dissoc_values.append(float(dissoc) if dissoc_ok else float("nan"))
        if assoc_ok:
            positive_values.append(float(assoc))
        if dissoc_ok:
            positive_values.append(float(dissoc))

    if not positions:
        raise ValueError("No finite positive exchange-rate values found for plotting.")

    position_array = np.asarray(positions, dtype=np.float64)
    assoc_array = np.asarray(assoc_values, dtype=np.float64)
    dissoc_array = np.asarray(dissoc_values, dtype=np.float64)
    positive_array = np.asarray(positive_values, dtype=np.float64)
    width = 0.36
    y_floor = 10.0 ** np.floor(np.log10(np.min(positive_array)))
    y_ceiling = 10.0 ** np.ceil(np.log10(np.max(positive_array) * 1.2))

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    assoc_mask = np.isfinite(assoc_array)
    if np.any(assoc_mask):
        ax.bar(
            position_array[assoc_mask] - width / 2.0,
            assoc_array[assoc_mask],
            width=width,
            bottom=y_floor,
            color=ASSOCIATIVE_COLOR,
            edgecolor="black",
            linewidth=0.5,
            label="assoc.",
            zorder=3,
        )
    dissoc_mask = np.isfinite(dissoc_array)
    if np.any(dissoc_mask):
        ax.bar(
            position_array[dissoc_mask] + width / 2.0,
            dissoc_array[dissoc_mask],
            width=width,
            bottom=y_floor,
            color=PASSIVE_COLOR,
            edgecolor="black",
            linewidth=0.5,
            label="dissoc.",
            zorder=3,
        )
    ax.set_xticks(position_array)
    ax.set_xticklabels(labels, fontsize=DEFAULT_TICK_FONTSIZE)
    ax.set_yscale("log")
    ax.set_ylim(y_floor, y_ceiling)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
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
    ax.xaxis.get_offset_text().set_fontsize(DEFAULT_TICK_FONTSIZE)
    ax.yaxis.get_offset_text().set_fontsize(DEFAULT_TICK_FONTSIZE)
    ax.legend(
        frameon=False,
        fontsize=DEFAULT_LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=2,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    set_target_axes_position(ax)
    fig.savefig(output_path)
    uplt.close(fig)

    print(f"Wrote exchange-rate comparison plot to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    epsilon, assoc_rates, dissoc_rates = load_summary_points(args.summary_csv)
    write_exchange_rate_plot(args.output, epsilon, assoc_rates, dissoc_rates)


if __name__ == "__main__":
    main()
