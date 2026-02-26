#!/usr/bin/env python3
"""Plot median/IQR relative total-energy drift for validation runs."""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gsd.hoomd
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "gsd is required for energy-conservation plotting. "
        "Activate the HOOMD environment before running this script."
    ) from exc


EPSILONS_DEFAULT = (3.0, 18.0)
STEP_KEY = "configuration/step"
POTENTIAL_KEY = "log/md/compute/ThermodynamicQuantities/potential_energy"
KINETIC_KEY = "log/md/compute/ThermodynamicQuantities/kinetic_energy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read energy-conservation trajectories and plot median/IQR "
            "relative total-energy drift (DeltaE/E0) versus timestep."
        )
    )
    parser.add_argument(
        "--input-root",
        default="../energy_conservation",
        help="Root directory containing eps_*/rep_*/trajectory.gsd files.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(EPSILONS_DEFAULT),
        help="Epsilon values to include.",
    )
    parser.add_argument(
        "--output-path",
        default="results/energy_conservation_deltaE_over_E0.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def discover_gsd_paths(input_root: str, epsilon: float) -> list[str]:
    eps_dir = os.path.join(input_root, f"eps_{epsilon:g}")
    patterns = [
        os.path.join(eps_dir, "rep_*", "trajectory.gsd"),
        os.path.join(eps_dir, "*.gsd"),
        os.path.join(eps_dir, "**", "trajectory.gsd"),
    ]
    paths: set[str] = set()
    for pattern in patterns:
        paths.update(glob.glob(pattern, recursive=True))
    return sorted(paths)


def load_energy_trace(gsd_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = gsd.hoomd.read_log(gsd_path)

    missing = [
        key for key in (STEP_KEY, POTENTIAL_KEY, KINETIC_KEY) if key not in data
    ]
    if missing:
        raise RuntimeError(
            f"Missing required log keys in {gsd_path}: {missing}. "
            "Ensure production logger records ThermodynamicQuantities energies."
        )

    step = np.asarray(data[STEP_KEY], dtype=np.int64).reshape(-1)
    potential = np.asarray(data[POTENTIAL_KEY], dtype=np.float64).reshape(-1)
    kinetic = np.asarray(data[KINETIC_KEY], dtype=np.float64).reshape(-1)
    if not (step.size == potential.size == kinetic.size):
        raise RuntimeError(
            f"Inconsistent log lengths in {gsd_path}: "
            f"step={step.size}, potential={potential.size}, kinetic={kinetic.size}"
        )

    total = potential + kinetic
    finite_mask = np.isfinite(potential) & np.isfinite(kinetic) & np.isfinite(total)
    if not np.any(finite_mask):
        raise RuntimeError(f"No finite energy samples found in {gsd_path}")

    step_finite = step[finite_mask]
    total_finite = total[finite_mask]
    e0 = float(total_finite[0])
    if abs(e0) < 1e-14:
        raise RuntimeError(
            f"Initial total energy E0 is too small in {gsd_path}; cannot compute DeltaE/E0."
        )
    delta_e_over_e0 = (total_finite - e0) / e0
    return step_finite, delta_e_over_e0


def aggregate_traces_by_timestep(
    traces: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values_by_timestep: dict[int, list[float]] = defaultdict(list)
    for timesteps, energies in traces:
        for timestep, energy in zip(timesteps, energies):
            values_by_timestep[int(timestep)].append(float(energy))

    if not values_by_timestep:
        raise RuntimeError("No timestep-aligned samples were collected.")

    sorted_steps = np.asarray(sorted(values_by_timestep.keys()), dtype=np.int64)
    median = np.empty(sorted_steps.shape[0], dtype=np.float64)
    q1 = np.empty(sorted_steps.shape[0], dtype=np.float64)
    q3 = np.empty(sorted_steps.shape[0], dtype=np.float64)

    for idx, step in enumerate(sorted_steps):
        vals = np.asarray(values_by_timestep[int(step)], dtype=np.float64)
        median[idx] = np.median(vals)
        q1[idx] = np.percentile(vals, 25.0)
        q3[idx] = np.percentile(vals, 75.0)

    return sorted_steps, median, q1, q3


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.abspath(os.path.join(script_dir, args.input_root))
    output_path = os.path.abspath(os.path.join(script_dir, args.output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    epsilons = [float(eps) for eps in args.epsilons]
    energy_label = r"$\Delta E / E_0$"
    median_label = r"Median $\Delta E / E_0$"
    title = r"Energy Conservation Check ($\Delta E / E_0$)"

    n_cols = max(1, len(epsilons))
    fig, axes = plt.subplots(1, n_cols, figsize=(5.4 * n_cols, 4.2), dpi=220)
    if n_cols == 1:
        axes = [axes]

    for axis, epsilon in zip(axes, epsilons):
        gsd_paths = discover_gsd_paths(input_root=input_root, epsilon=epsilon)
        if not gsd_paths:
            axis.text(
                0.5,
                0.5,
                f"No trajectories found\nfor eps={epsilon:g}",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            axis.set_title(f"eps = {epsilon:g}")
            axis.set_xlabel("Timestep")
            axis.grid(alpha=0.2)
            continue

        traces: list[tuple[np.ndarray, np.ndarray]] = []
        for gsd_path in gsd_paths:
            timesteps, energies = load_energy_trace(gsd_path=gsd_path)
            traces.append((timesteps, energies))

        steps, med, q1, q3 = aggregate_traces_by_timestep(traces)
        axis.fill_between(steps, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
        axis.plot(steps, med, color="#121212", lw=2.0, label=median_label)
        axis.set_title(f"eps = {epsilon:g} (n={len(gsd_paths)})")
        axis.set_xlabel("Timestep")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        print(
            f"eps={epsilon:g}: {len(gsd_paths)} trajectories, "
            f"using keys=({STEP_KEY}, {POTENTIAL_KEY}, {KINETIC_KEY})",
            flush=True,
        )

    axes[0].set_ylabel(energy_label)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Wrote plot: {output_path}", flush=True)


if __name__ == "__main__":
    main()
