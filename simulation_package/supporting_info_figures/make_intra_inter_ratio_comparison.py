#!/usr/bin/env python3
"""Compare melt and semidilute intra/inter bond ratios with grouped bars."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_CACHE_ROOT = Path("/tmp") / f"supporting-info-figures-{os.getuid()}"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker as mticker
import ultraplot as uplt


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

MELT_SOURCE_SVG = (
    PACKAGE_DIR
    / "melt"
    / "analysis"
    / "results"
    / "intra_to_inter_bond_ratio_vs_epsilon.svg"
)
MPCD_SOURCE_SVG = (
    PACKAGE_DIR
    / "mpcd"
    / "analysis"
    / "results"
    / "intra_to_inter_bond_ratio_vs_epsilon.svg"
)

OUTPUT_PATH = SCRIPT_DIR / "intra_to_inter_bond_ratio_melt_mpcd_comparison.svg"

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
BAR_WIDTH = 0.36
MELT_COLOR = "#e77500"
SEMIDILUTE_COLOR = "#121212"
EDGE_COLOR = "black"


def load_ratio_series(source_svg: Path) -> dict[float, float]:
    """Load epsilon-to-mean ratio data from the summary beside a source SVG."""
    if not source_svg.exists():
        raise FileNotFoundError(f"Missing source SVG: {source_svg}")

    summary_path = source_svg.parent / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    ratios: dict[float, float] = {}
    for eps_key, metrics in summary.items():
        ratio = metrics.get("intra_inter_ratio")
        if ratio is None:
            raise KeyError(f"Missing intra_inter_ratio in {summary_path} for eps={eps_key}")
        epsilon = float(eps_key)
        mean = float(ratio["mean"])
        if np.isfinite(epsilon) and np.isfinite(mean) and mean > 0.0:
            ratios[epsilon] = mean
    if not ratios:
        raise ValueError(f"No finite positive intra/inter ratios found in {summary_path}")
    return ratios


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def category_labels(epsilon: np.ndarray) -> list[str]:
    return ["None" if np.isclose(value, 0.0) else f"{value:g}" for value in epsilon]


def bar_axis_floor(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("Need at least one finite positive value to set a log-scale bar axis.")
    floor = float(10.0 ** np.floor(np.log10(np.min(positive))))
    if np.isclose(np.min(positive), floor):
        floor /= 10.0
    return floor


def build_aligned_series(
    melt_ratios: dict[float, float],
    semidilute_ratios: dict[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilon = np.asarray(
        sorted(set(melt_ratios.keys()) | set(semidilute_ratios.keys())),
        dtype=np.float64,
    )
    melt = np.asarray([melt_ratios.get(float(value), float("nan")) for value in epsilon], dtype=np.float64)
    semidilute = np.asarray(
        [semidilute_ratios.get(float(value), float("nan")) for value in epsilon],
        dtype=np.float64,
    )
    return epsilon, melt, semidilute


def main() -> None:
    melt_ratios = load_ratio_series(MELT_SOURCE_SVG)
    semidilute_ratios = load_ratio_series(MPCD_SOURCE_SVG)
    epsilon, melt_mean, semidilute_mean = build_aligned_series(melt_ratios, semidilute_ratios)
    x = np.arange(epsilon.size, dtype=np.float64)
    positive_values = np.concatenate(
        (
            melt_mean[np.isfinite(melt_mean) & (melt_mean > 0.0)],
            semidilute_mean[np.isfinite(semidilute_mean) & (semidilute_mean > 0.0)],
        )
    )
    y_floor = bar_axis_floor(positive_values)

    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    melt_bottom = np.full(melt_mean.shape, y_floor, dtype=np.float64)
    semidilute_bottom = np.full(semidilute_mean.shape, y_floor, dtype=np.float64)
    ax.bar(
        x - BAR_WIDTH / 2.0,
        melt_mean - melt_bottom,
        bottom=melt_bottom,
        width=BAR_WIDTH,
        color=MELT_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        label="Melt",
        zorder=3,
    )
    ax.bar(
        x + BAR_WIDTH / 2.0,
        semidilute_mean - semidilute_bottom,
        bottom=semidilute_bottom,
        width=BAR_WIDTH,
        color=SEMIDILUTE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        label="Semidilute",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(y_floor, float(np.max(positive_values) * 1.3))
    ax.set_xlim(-0.5, epsilon.size - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels(epsilon))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(r"$\varepsilon_\mathrm{RLJ}/\varepsilon_0$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\psi$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", which="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    ax.legend(frameon=False, fontsize=DEFAULT_TICK_FONTSIZE, loc="best")
    set_target_axes_position(ax)
    fig.savefig(OUTPUT_PATH)
    uplt.close(fig)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
