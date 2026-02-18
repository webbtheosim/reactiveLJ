#!/usr/bin/env python3
"""Build runtime comparison violin plots for ReactiveLJ vs Tersoff runs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EPSILONS_DEFAULT = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
PRODUCTION_RUNTIME_RE = re.compile(
    r"Production_runtime_seconds=([0-9]+(?:\.[0-9]+)?)"
)
REQUESTED_EPSILON_RE = re.compile(r"Requested_epsilon=([0-9]+(?:\.[0-9]+)?)")
EPSILON_LINE_RE = re.compile(r"epsilon=([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a combined runtime violin plot for ReactiveLJ and Tersoff runs."
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(EPSILONS_DEFAULT),
        help="Epsilon values in array-order mapping.",
    )
    parser.add_argument(
        "--reactive-log-glob",
        default="../logs/generate_data_*.out",
        help="Glob for ReactiveLJ log files.",
    )
    parser.add_argument(
        "--tersoff-log-glob",
        default="logs/generate_tersoff_data_*.out",
        help="Glob for Tersoff log files.",
    )
    parser.add_argument(
        "--output-path",
        default="plots/runtime_violin_comparison.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--samples-csv",
        default="outputs/runtime_samples.csv",
        help="Optional CSV dump of parsed runtime samples.",
    )
    return parser.parse_args()


def _extract_production_runtime_seconds(path: str) -> float | None:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    match = PRODUCTION_RUNTIME_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _epsilon_from_log_text(path: str, epsilons: list[float]) -> float | None:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "Stage=reactive_equil epsilon_ramp done" not in line:
            continue
        if idx <= 0:
            return None
        prev_line = lines[idx - 1]
        match = EPSILON_LINE_RE.search(prev_line)
        if match is None:
            return None
        value = float(match.group(1))
        for eps in epsilons:
            if abs(value - eps) < 1e-8:
                return float(eps)
        return None

    # Tersoff logs do not have the ReactiveLJ epsilon ramp; use explicit value.
    requested_matches = REQUESTED_EPSILON_RE.findall(text)
    if requested_matches:
        value = float(requested_matches[-1])
        for eps in epsilons:
            if abs(value - eps) < 1e-8:
                return float(eps)
    return None


def collect_samples(
    log_glob: str,
    epsilons: list[float],
) -> dict[float, list[float]]:
    samples: dict[float, list[float]] = defaultdict(list)

    for path in sorted(glob.glob(log_glob)):
        runtime = _extract_production_runtime_seconds(path)
        if runtime is None:
            continue

        epsilon = _epsilon_from_log_text(path, epsilons)
        if epsilon is None:
            continue

        samples[float(epsilon)].append(runtime)

    return samples


def dump_samples_csv(
    path: str,
    reactive_samples: dict[float, list[float]],
    tersoff_samples: dict[float, list[float]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "epsilon", "production_runtime_seconds"])
        for epsilon, values in sorted(reactive_samples.items()):
            for runtime in values:
                writer.writerow(["ReactiveLJ", epsilon, runtime])
        for epsilon, values in sorted(tersoff_samples.items()):
            for runtime in values:
                writer.writerow(["Tersoff", epsilon, runtime])


def _draw_violin(ax, data: list[list[float]], positions: np.ndarray, color: str) -> None:
    valid_data = []
    valid_positions = []
    for vals, pos in zip(data, positions):
        if len(vals) == 0:
            continue
        valid_data.append(vals)
        valid_positions.append(pos)

    if not valid_data:
        return

    violin = ax.violinplot(
        valid_data,
        positions=np.array(valid_positions),
        widths=0.30,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.75)


def _median_series(data: list[list[float]]) -> np.ndarray:
    medians = []
    for values in data:
        if len(values) == 0:
            medians.append(np.nan)
        else:
            medians.append(float(np.median(values)))
    return np.asarray(medians, dtype=np.float64)


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    reactive_glob = os.path.abspath(os.path.join(script_dir, args.reactive_log_glob))
    tersoff_glob = os.path.abspath(os.path.join(script_dir, args.tersoff_log_glob))
    output_path = os.path.abspath(os.path.join(script_dir, args.output_path))
    samples_csv = os.path.abspath(os.path.join(script_dir, args.samples_csv))

    epsilons = [float(e) for e in args.epsilons]

    reactive_samples = collect_samples(
        log_glob=reactive_glob,
        epsilons=epsilons,
    )
    tersoff_samples = collect_samples(
        log_glob=tersoff_glob,
        epsilons=epsilons,
    )

    dump_samples_csv(samples_csv, reactive_samples=reactive_samples, tersoff_samples=tersoff_samples)

    bases = np.arange(len(epsilons), dtype=float) + 1.0
    reactive_positions = bases
    tersoff_positions = bases

    reactive_data = [reactive_samples.get(eps, []) for eps in epsilons]
    tersoff_data = [tersoff_samples.get(eps, []) for eps in epsilons]

    fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=600)

    _draw_violin(ax, reactive_data, reactive_positions, color="#e77500")
    _draw_violin(ax, tersoff_data, tersoff_positions, color="#121212")

    reactive_medians = _median_series(reactive_data)
    tersoff_medians = _median_series(tersoff_data)
    ax.plot(
        bases,
        reactive_medians,
        color="#e77500",
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        zorder=4,
    )
    ax.plot(
        bases,
        tersoff_medians,
        color="#121212",
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        zorder=4,
    )

    ax.set_xticks(bases)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilons], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_xlabel(r"ReactiveLJ $\varepsilon$", fontsize=10)
    ax.set_ylabel("Production Runtime (s)", fontsize=10)
    ax.set_title("Production Runtime Comparison", fontsize=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    legend_handles = [
        Patch(facecolor="#e77500", edgecolor="#e77500", alpha=0.75, label="ReactiveLJ"),
        Patch(facecolor="#121212", edgecolor="#121212", alpha=0.75, label="Tersoff analog"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, frameon=True, loc="best")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    reactive_count = sum(len(v) for v in reactive_samples.values())
    tersoff_count = sum(len(v) for v in tersoff_samples.values())
    print(f"Parsed runtime samples: ReactiveLJ={reactive_count}, Tersoff={tersoff_count}")
    print(f"Wrote sample table: {samples_csv}")
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
