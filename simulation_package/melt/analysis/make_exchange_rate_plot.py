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
DEFAULT_DPI = 600
DEFAULT_TICK_FONTSIZE = 10
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 10
DEFAULT_OUTPUT_NAME = "exchange_rate_comparison_vs_epsilon.svg"
LEGACY_OUTPUT_NAMES = ("exchange_rate_comparison_vs_epsilon.png",)
ASSOCIATIVE_COLOR = "#e77500"
PASSIVE_COLOR = "#121212"
DUMP_INTERVAL_TAU_LJ = 1000.0
DEFAULT_TAU_R0 = 4041.0
X_AXIS_LABEL = r"Sticker strength, $\varepsilon_\mathrm{RLJ}/\varepsilon_0$"
Y_AXIS_LABEL = r"Turnover rate, $\nu_\alpha$"


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
            labels.append("WCA")
        else:
            labels.append(f"{value:g}")
    return labels


def write_exchange_rate_plot(
    output: Path | str,
    epsilon: np.ndarray,
    turnover_assoc: np.ndarray,
    turnover_dissoc: np.ndarray,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(output_path)

    epsilon = np.asarray(epsilon, dtype=np.float64)
    turnover_assoc = np.asarray(turnover_assoc, dtype=np.float64)
    turnover_dissoc = np.asarray(turnover_dissoc, dtype=np.float64)
    normalization_factor = DUMP_INTERVAL_TAU_LJ / DEFAULT_TAU_R0
    assoc_normalized = turnover_assoc / normalization_factor
    dissoc_normalized = turnover_dissoc / normalization_factor
    x = np.arange(epsilon.size, dtype=np.float64)
    width = 0.36
    all_values = np.concatenate((assoc_normalized, dissoc_normalized))
    positive_values = all_values[np.isfinite(all_values) & (all_values > 0.0)]
    if positive_values.size == 0:
        raise ValueError("Exchange-rate bars require at least one finite positive value.")
    if positive_values.size != all_values.size:
        raise ValueError("Exchange-rate bars require finite positive values on a log axis.")
    y_floor = 10.0 ** np.floor(np.log10(np.min(positive_values)))
    y_ceiling = 10.0 ** np.ceil(np.log10(np.max(positive_values) * 1.2))
    assoc_bottoms = np.full(assoc_normalized.shape, y_floor, dtype=np.float64)
    dissoc_bottoms = np.full(dissoc_normalized.shape, y_floor, dtype=np.float64)

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    ax.bar(
        x - width / 2.0,
        assoc_normalized - assoc_bottoms,
        width=width,
        bottom=assoc_bottoms,
        color=ASSOCIATIVE_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="assoc.",
        zorder=3,
    )
    ax.bar(
        x + width / 2.0,
        dissoc_normalized - dissoc_bottoms,
        width=width,
        bottom=dissoc_bottoms,
        color=PASSIVE_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="dissoc.",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(y_floor, y_ceiling)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.set_xlabel(X_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_category_labels(epsilon), fontsize=DEFAULT_TICK_FONTSIZE)
    ax.set_xlim(-0.5, epsilon.size - 0.5)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", labelsize=DEFAULT_TICK_FONTSIZE)
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
    epsilon, turnover_assoc, turnover_dissoc = load_summary_points(args.summary_csv)
    write_exchange_rate_plot(args.output, epsilon, turnover_assoc, turnover_dissoc)


if __name__ == "__main__":
    main()
