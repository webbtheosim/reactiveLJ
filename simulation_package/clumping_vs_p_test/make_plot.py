#!/usr/bin/env python3
"""Create the clump-fraction violin plot for the small-system p sweep."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

USER_TMP_DIR = Path("/tmp") / f"reactive_lj_plot_cache_{os.getuid()}"
USER_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(USER_TMP_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(USER_TMP_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import gsd.hoomd
import matplotlib
import numpy as np
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGSIZE = (3.3, 3.3)
DPI = 1000
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 8
VIOLIN_FILL = "#e77500"
OUTLINE_COLOR = "#121212"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Read the clumping-test GSD files and plot the framewise clump "
            "fraction distribution versus weakening exponent p."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=script_dir / "outputs",
        help="Root directory containing p_*/rep_*/trajectory.gsd outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "clump_fraction_violin.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Parallel jobs over trajectories (0 uses SLURM_CPUS_PER_TASK or 1).",
    )
    return parser.parse_args()


def lj_inflection_cutoff(sigma: float = 1.0) -> float:
    return float(sigma) * (26.0 / 7.0) ** (1.0 / 6.0)


def resolve_n_jobs(requested_n_jobs: int) -> int:
    if requested_n_jobs > 0:
        return requested_n_jobs
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return 1


def infer_p_value(run_dir: Path, metadata: dict | None) -> int:
    if metadata is not None:
        if "clumping_test_p" in metadata:
            return int(metadata["clumping_test_p"])
        if "weakening_exponent" in metadata:
            return int(round(float(metadata["weakening_exponent"])))

    for part in run_dir.parts:
        match = re.fullmatch(r"p_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer p value for run directory {run_dir}")


def discover_runs(input_root: Path) -> list[tuple[int, Path, dict | None]]:
    runs: list[tuple[int, Path, dict | None]] = []
    for trajectory_path in sorted(input_root.rglob("trajectory.gsd")):
        run_dir = trajectory_path.parent
        metadata_path = run_dir / "metadata.json"
        metadata = None
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        p_value = infer_p_value(run_dir, metadata)
        runs.append((p_value, trajectory_path, metadata))
    if not runs:
        raise FileNotFoundError(f"No trajectory.gsd files found under {input_root}")
    return sorted(runs, key=lambda item: (item[0], str(item[1])))


def compute_frame_clump_fraction(
    positions: np.ndarray,
    box_length: float,
    cutoff_sq: float,
) -> float:
    n_stickers = int(positions.shape[0])
    if n_stickers == 0:
        raise ValueError("Cannot compute clump fraction for a frame with zero stickers.")
    if n_stickers == 1:
        return 0.0

    delta = positions[:, None, :] - positions[None, :, :]
    delta -= box_length * np.rint(delta / box_length)
    dist_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    bonded = (dist_sq <= cutoff_sq) & (dist_sq > 0.0)
    multi_bonded = bonded.sum(axis=1) >= 2
    if np.any(multi_bonded):
        clumped = multi_bonded | bonded[:, multi_bonded].any(axis=1)
    else:
        clumped = multi_bonded
    return float(np.count_nonzero(clumped) / n_stickers)


def analyze_trajectory(
    p_value: int,
    trajectory_path: Path,
    metadata: dict | None,
) -> tuple[int, np.ndarray]:
    sigma = 1.0
    if metadata is not None and metadata.get("reactive_sigma") is not None:
        sigma = float(metadata["reactive_sigma"])
    cutoff = lj_inflection_cutoff(sigma)
    cutoff_sq = cutoff * cutoff

    fractions: list[float] = []
    with gsd.hoomd.open(str(trajectory_path), "r") as trajectory:
        for frame in trajectory:
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_length = float(frame.configuration.box[0])
            fractions.append(
                compute_frame_clump_fraction(
                    positions=positions,
                    box_length=box_length,
                    cutoff_sq=cutoff_sq,
                )
            )

    if not fractions:
        raise RuntimeError(f"Trajectory contains no frames: {trajectory_path}")
    return p_value, np.asarray(fractions, dtype=np.float64)


def make_violin_plot(
    grouped_data: list[tuple[int, np.ndarray]],
    output_path: Path,
) -> None:
    positions = [item[0] for item in grouped_data]
    datasets = [item[1] for item in grouped_data]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    violin = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin["bodies"]:
        body.set_facecolor(VIOLIN_FILL)
        body.set_edgecolor(OUTLINE_COLOR)
        body.set_linewidth(0.8)
        body.set_alpha(1.0)

    ax.set_xlabel(r"$p$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Clump fraction", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in positions])
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE, colors=OUTLINE_COLOR)
    for spine in ax.spines.values():
        spine.set_color(OUTLINE_COLOR)
        spine.set_linewidth(0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.input_root)
    n_jobs = min(resolve_n_jobs(args.n_jobs), len(runs))

    analyzed = Parallel(n_jobs=n_jobs)(
        delayed(analyze_trajectory)(p_value, trajectory_path, metadata)
        for p_value, trajectory_path, metadata in runs
    )

    grouped: dict[int, list[np.ndarray]] = {}
    for p_value, fractions in analyzed:
        grouped.setdefault(p_value, []).append(fractions)

    grouped_data = [
        (p_value, np.concatenate(grouped[p_value]))
        for p_value in sorted(grouped)
    ]

    for p_value, fractions in grouped_data:
        print(
            f"p={p_value} frames={fractions.size} "
            f"mean_clump_fraction={np.mean(fractions):.6f}",
            flush=True,
        )

    make_violin_plot(grouped_data, args.output)
    print(f"Wrote clump-fraction plot to {args.output}", flush=True)


if __name__ == "__main__":
    main()
