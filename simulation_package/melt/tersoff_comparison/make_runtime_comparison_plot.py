#!/usr/bin/env python3
"""Build runtime comparison bar plots for ReactiveLJ vs Tersoff runs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np

import ultraplot as uplt


EPSILONS_DEFAULT = (6.0, 12.0, 15.0, 18.0)
SECONDS_PER_DAY = 86400.0
TIME_STEP_DEFAULT = 0.005
PRODUCTION_RUNTIME_RE = re.compile(
    r"Production_runtime_seconds=([0-9]+(?:\.[0-9]+)?)"
)
PRODUCTION_STEPS_RE = re.compile(r"Stage=production start steps=([0-9]+)")
REQUESTED_EPSILON_RE = re.compile(r"Requested_epsilon=([0-9]+(?:\.[0-9]+)?)")
EPSILON_LINE_RE = re.compile(r"epsilon=([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped runtime bar plot for ReactiveLJ and Tersoff runs."
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
        default="logs/generate_reactive_lj_data_*.out",
        help="Glob for ReactiveLJ comparison log files.",
    )
    parser.add_argument(
        "--tersoff-log-glob",
        default="logs/generate_tersoff_data_*.out",
        help="Glob for Liu/O'Connor Tersoff comparison log files.",
    )
    parser.add_argument(
        "--output-path",
        default="plots/runtime_bar_comparison.svg",
        help="Output plot path.",
    )
    parser.add_argument(
        "--samples-csv",
        default="outputs/runtime_samples.csv",
        help="Optional CSV dump of parsed runtime samples.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=TIME_STEP_DEFAULT,
        help="Production timestep used to convert timesteps/day to reduced time units/day.",
    )
    parser.add_argument(
        "--metric",
        choices=["tau_per_day", "timesteps_per_day"],
        default="tau_per_day",
        help="Throughput metric to plot on the y axis.",
    )
    return parser.parse_args()


def _read_log_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _extract_production_runtime_seconds(text: str) -> float | None:
    match = PRODUCTION_RUNTIME_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _extract_production_steps(text: str) -> int | None:
    matches = PRODUCTION_STEPS_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def _epsilon_from_log_text(text: str, epsilons: list[float]) -> float | None:
    requested_matches = REQUESTED_EPSILON_RE.findall(text)
    if requested_matches:
        value = float(requested_matches[-1])
        for eps in epsilons:
            if abs(value - eps) < 1e-8:
                return float(eps)

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
    return None


def collect_samples(
    log_glob: str,
    epsilons: list[float],
    time_step: float,
) -> dict[float, list[dict[str, Any]]]:
    samples: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for path in sorted(glob.glob(log_glob)):
        text = _read_log_text(path)
        runtime = _extract_production_runtime_seconds(text)
        if runtime is None:
            continue

        production_steps = _extract_production_steps(text)
        if production_steps is None:
            continue

        epsilon = _epsilon_from_log_text(text, epsilons)
        if epsilon is None:
            continue

        timesteps_per_day = production_steps * SECONDS_PER_DAY / runtime
        tau_per_day = timesteps_per_day * time_step

        samples[float(epsilon)].append(
            {
                "log_path": path,
                "production_runtime_seconds": runtime,
                "production_steps": production_steps,
                "timesteps_per_day": timesteps_per_day,
                "tau_per_day": tau_per_day,
            }
        )

    return samples


def dump_samples_csv(
    path: str,
    reactive_samples: dict[float, list[dict[str, Any]]],
    tersoff_samples: dict[float, list[dict[str, Any]]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "epsilon",
                "production_steps",
                "production_runtime_seconds",
                "timesteps_per_day",
                "tau_per_day",
                "log_path",
            ]
        )
        for epsilon, values in sorted(reactive_samples.items()):
            for sample in values:
                writer.writerow(
                    [
                        "ReactiveLJ",
                        epsilon,
                        sample["production_steps"],
                        sample["production_runtime_seconds"],
                        sample["timesteps_per_day"],
                        sample["tau_per_day"],
                        sample["log_path"],
                    ]
                )
        for epsilon, values in sorted(tersoff_samples.items()):
            for sample in values:
                writer.writerow(
                    [
                        "Tersoff",
                        epsilon,
                        sample["production_steps"],
                        sample["production_runtime_seconds"],
                        sample["timesteps_per_day"],
                        sample["tau_per_day"],
                        sample["log_path"],
                    ]
                )


def _summary_series(data: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    errors = []
    for values in data:
        if len(values) == 0:
            centers.append(np.nan)
            errors.append(np.nan)
        else:
            arr = np.asarray(values, dtype=np.float64)
            centers.append(float(np.median(arr)))
            if arr.size == 1:
                errors.append(0.0)
            else:
                q25, q75 = np.percentile(arr, [25, 75])
                errors.append(float((q75 - q25) / 2.0))
    return np.asarray(centers, dtype=np.float64), np.asarray(errors, dtype=np.float64)


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
        time_step=args.time_step,
    )
    tersoff_samples = collect_samples(
        log_glob=tersoff_glob,
        epsilons=epsilons,
        time_step=args.time_step,
    )

    dump_samples_csv(samples_csv, reactive_samples=reactive_samples, tersoff_samples=tersoff_samples)

    bases = np.arange(len(epsilons), dtype=float)

    reactive_data = [
        [sample[args.metric] for sample in reactive_samples.get(eps, [])]
        for eps in epsilons
    ]
    tersoff_data = [
        [sample[args.metric] for sample in tersoff_samples.get(eps, [])]
        for eps in epsilons
    ]

    fig, ax = uplt.subplots(figsize=(3.3, 2.0), dpi=600)

    reactive_medians, reactive_errors = _summary_series(reactive_data)
    tersoff_medians, tersoff_errors = _summary_series(tersoff_data)

    width = 0.36
    ax.bar(
        bases - width / 2,
        reactive_medians,
        width=width,
        yerr=reactive_errors,
        color="#e77500",
        edgecolor="#8f4a00",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "capsize": 2.0},
        label="ReactiveLJ",
        zorder=3,
    )
    ax.bar(
        bases + width / 2,
        tersoff_medians,
        width=width,
        yerr=tersoff_errors,
        color="#121212",
        edgecolor="#121212",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "capsize": 2.0},
        label="Tersoff analog",
        zorder=3,
    )

    ax.set_xticks(bases)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilons], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_xlabel(r"$\varepsilon_\mathrm{RLJ}$", fontsize=10)
    ylabel = {
        "tau_per_day": r"Simulation Throughput ($\tau$/day)",
        "timesteps_per_day": "Simulation Throughput (timesteps/day)",
    }[args.metric]
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("Production Throughput Comparison", fontsize=12)
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.yaxis.label.set_rotation(90)
    ax.yaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("bottom")

    ax.legend(fontsize=8, frameon=True, loc="best")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    uplt.close(fig)

    reactive_count = sum(len(v) for v in reactive_samples.values())
    tersoff_count = sum(len(v) for v in tersoff_samples.values())
    print(f"Parsed runtime samples: ReactiveLJ={reactive_count}, Tersoff={tersoff_count}")
    print(f"Wrote sample table: {samples_csv}")
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
