#!/usr/bin/env python3
"""Compare melt and MPCD intra/inter bond ratios on log-log axes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

_CACHE_ROOT = Path("/tmp") / f"supporting-info-figures-{os.getuid()}"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

MELT_COLOR = "#fa4616"
MPCD_COLOR = "#0021a5"


def load_ratio_series(source_svg: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load epsilon, mean, and stderr from the summary beside a source SVG."""
    if not source_svg.exists():
        raise FileNotFoundError(f"Missing source SVG: {source_svg}")

    summary_path = source_svg.parent / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    eps_values: list[float] = []
    means: list[float] = []
    stderrs: list[float] = []
    for eps_key, metrics in sorted(summary.items(), key=lambda item: float(item[0])):
        ratio = metrics.get("intra_inter_ratio")
        if ratio is None:
            raise KeyError(f"Missing intra_inter_ratio in {summary_path} for eps={eps_key}")
        eps_values.append(float(eps_key))
        means.append(float(ratio["mean"]))
        stderrs.append(float(ratio.get("stderr", 0.0)))

    return (
        np.asarray(eps_values, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(stderrs, dtype=np.float64),
    )


def positive_log_data(
    label: str,
    eps: np.ndarray,
    means: np.ndarray,
    stderrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(eps) & np.isfinite(means) & np.isfinite(stderrs) & (eps > 0.0) & (means > 0.0)
    skipped = eps.size - int(np.count_nonzero(mask))
    if skipped:
        skipped_eps = ", ".join(f"{value:g}" for value in eps[~mask])
        print(f"{label}: skipped {skipped} non-loggable point(s): epsilon={skipped_eps}")
    return eps[mask], means[mask], stderrs[mask]


def add_series(
    ax,
    label: str,
    color: str,
    eps: np.ndarray,
    means: np.ndarray,
) -> None:
    ax.plot(
        eps,
        means,
        color=color,
        marker="o",
        markersize=4.0,
        linewidth=1.8,
        label=label,
    )


def configure_log_ticks(ax, eps_values: Iterable[float]) -> None:
    positive_eps = sorted({float(eps) for eps in eps_values if float(eps) > 0.0})
    ax.set_xticks(positive_eps)
    ax.set_xticklabels([f"{eps:g}" for eps in positive_eps])


def main() -> None:
    melt_eps, melt_mean, melt_stderr = positive_log_data("Melt", *load_ratio_series(MELT_SOURCE_SVG))
    mpcd_eps, mpcd_mean, mpcd_stderr = positive_log_data(
        "Semidilute", *load_ratio_series(MPCD_SOURCE_SVG)
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    add_series(ax, "Melt", MELT_COLOR, melt_eps, melt_mean)
    add_series(ax, "Semidilute", MPCD_COLOR, mpcd_eps, mpcd_mean)

    ax.set_xscale("log")
    ax.set_yscale("log")
    configure_log_ticks(ax, np.concatenate([melt_eps, mpcd_eps]))
    ax.set_xlabel(r"$\varepsilon_\mathrm{reactiveLJ}$", fontsize=10)
    ax.set_ylabel(r"$\psi$", fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=8)
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
