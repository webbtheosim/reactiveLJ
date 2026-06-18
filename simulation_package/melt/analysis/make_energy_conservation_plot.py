#!/usr/bin/env python3
"""Plot median/IQR relative total-energy drift for validation runs."""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from datetime import datetime
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
from joblib import Parallel, delayed

matplotlib.use("Agg")
import ultraplot as uplt

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
MEDIAN_COLOR = "#121212"
LEGACY_COMBINED_OUTPUT_NAMES = (
    "energy_conservation_deltaE_over_E0.png",
    "energy_conservation_deltaE_over_E0.svg",
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
        default="results/energy_conservation_deltaE_over_E0.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=20,
        help=(
            "Number of trajectory-loading workers. 0 uses SLURM_CPUS_PER_TASK "
            "when available, otherwise all visible CPUs."
        ),
    )
    parser.add_argument(
        "--joblib-verbose",
        type=int,
        default=10,
        help="Verbosity passed to joblib. Use 0 for quiet runs.",
    )
    return parser.parse_args()


def resolve_n_jobs(requested: int) -> int:
    if requested > 0:
        return requested
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            parsed = int(slurm_cpus)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


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


def load_energy_trace_with_progress(
    gsd_path: str,
    input_root: str,
    epsilon: float,
    trajectory_index: int,
    total_trajectories: int,
) -> tuple[np.ndarray, np.ndarray]:
    rel_path = os.path.relpath(gsd_path, input_root)
    log(
        f"eps={epsilon:g}: starting trajectory "
        f"{trajectory_index}/{total_trajectories} ({rel_path})"
    )
    trace = load_energy_trace(gsd_path=gsd_path)
    log(
        f"eps={epsilon:g}: finished trajectory "
        f"{trajectory_index}/{total_trajectories} ({rel_path})"
    )
    return trace


def aggregate_traces_by_timestep(
    traces: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    values_by_timestep: dict[int, list[float]] = defaultdict(list)
    for timesteps, energies in traces:
        for timestep, energy in zip(timesteps, energies):
            values_by_timestep[int(timestep)].append(float(energy))

    if not values_by_timestep:
        raise RuntimeError("No timestep-aligned samples were collected.")

    sorted_steps = np.asarray(sorted(values_by_timestep.keys()), dtype=np.int64)
    median = np.empty(sorted_steps.shape[0], dtype=np.float64)

    for idx, step in enumerate(sorted_steps):
        vals = np.asarray(values_by_timestep[int(step)], dtype=np.float64)
        median[idx] = np.median(vals)

    return sorted_steps, median


def set_target_axes_position(ax) -> None:
    ax.set_position(
        [
            AXES_LEFT_PT / FIGURE_WIDTH_PT,
            AXES_BOTTOM_PT / FIGURE_HEIGHT_PT,
            AXES_WIDTH_PT / FIGURE_WIDTH_PT,
            AXES_HEIGHT_PT / FIGURE_HEIGHT_PT,
        ]
    )


def format_epsilon_title_value(epsilon: float) -> str:
    if np.isclose(epsilon, 0.0, rtol=0.0, atol=1.0e-12):
        return r"\mathrm{None}"
    return f"{epsilon:g}"


def epsilon_output_tag(epsilon: float) -> str:
    if np.isclose(epsilon, 0.0, rtol=0.0, atol=1.0e-12):
        return "none"
    return f"{epsilon:g}".replace(".", "p").replace("-", "m")


def normalize_output_path(output_path: Path) -> Path:
    if output_path.suffix.lower() == ".svg":
        return output_path
    if output_path.suffix:
        return output_path.with_suffix(".svg")
    return output_path.with_name(f"{output_path.name}.svg")


def resolve_output_path(base_output_path: Path, epsilon: float, multiple: bool) -> Path:
    if not multiple:
        return base_output_path
    return base_output_path.with_name(
        f"{base_output_path.stem}_eps_{epsilon_output_tag(epsilon)}{base_output_path.suffix}"
    )


def remove_legacy_combined_output(output_path: Path, multiple: bool) -> None:
    if multiple and output_path.exists():
        output_path.unlink()
    png_variant = output_path.with_suffix(".png")
    if multiple and png_variant.exists():
        png_variant.unlink()
    for filename in LEGACY_COMBINED_OUTPUT_NAMES:
        legacy_path = output_path.with_name(filename)
        if multiple and legacy_path.exists():
            legacy_path.unlink()


def remove_legacy_output_variants(output_path: Path) -> None:
    png_variant = output_path.with_suffix(".png")
    if png_variant != output_path and png_variant.exists():
        png_variant.unlink()


def write_energy_plot(
    output_path: Path,
    epsilon: float,
    steps: np.ndarray,
    median: np.ndarray,
) -> None:
    remove_legacy_output_variants(output_path)
    fig, ax = uplt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, tight=False)
    set_target_axes_position(ax)
    ax.plot(steps, median, color=MEDIAN_COLOR, lw=1.4, zorder=3)
    ax.set_xlabel("Timestep", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Delta E / E_0$", fontsize=DEFAULT_LABEL_FONTSIZE)
    y_formatter = mticker.ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.text(
        0.5,
        0.96,
        rf"$\varepsilon_\mathrm{{RLJ}}={format_epsilon_title_value(epsilon)}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=DEFAULT_LABEL_FONTSIZE,
    )
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", labelsize=DEFAULT_TICK_FONTSIZE)
    ax.yaxis.get_offset_text().set_fontsize(DEFAULT_TICK_FONTSIZE)
    set_target_axes_position(ax)
    fig.savefig(output_path)
    uplt.close(fig)
    log(f"Wrote plot: {output_path}")


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    input_root = Path(os.path.abspath(os.path.join(script_dir, args.input_root)))
    output_path = normalize_output_path(
        Path(os.path.abspath(os.path.join(script_dir, args.output_path)))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilons = [float(eps) for eps in args.epsilons]
    configured_n_jobs = resolve_n_jobs(args.n_jobs)
    log(f"Energy-conservation input root: {input_root}")
    log(f"Output path: {output_path}")
    log("Selected epsilons: " + ", ".join(f"{epsilon:g}" for epsilon in epsilons))
    log(f"Configured worker count: n_jobs={configured_n_jobs}")
    multiple_outputs = len(epsilons) > 1
    remove_legacy_combined_output(output_path, multiple_outputs)

    for epsilon in epsilons:
        gsd_paths = discover_gsd_paths(input_root=str(input_root), epsilon=epsilon)
        log(f"eps={epsilon:g}: discovered {len(gsd_paths)} trajectory file(s)")
        if not gsd_paths:
            log(f"eps={epsilon:g}: no trajectories found, skipping plot")
            continue

        effective_n_jobs = min(configured_n_jobs, len(gsd_paths))
        log(
            f"eps={epsilon:g}: loading {len(gsd_paths)} trajectory file(s) "
            f"with n_jobs={effective_n_jobs}"
        )
        traces = Parallel(
            n_jobs=effective_n_jobs,
            verbose=args.joblib_verbose,
            prefer="threads",
        )(
            delayed(load_energy_trace_with_progress)(
                gsd_path,
                str(input_root),
                epsilon,
                idx,
                len(gsd_paths),
            )
            for idx, gsd_path in enumerate(gsd_paths, start=1)
        )

        steps, med = aggregate_traces_by_timestep(traces)
        log(
            f"eps={epsilon:g}: aggregated {steps.size} timestep(s) "
            f"from {len(traces)} trajectory file(s)"
        )
        epsilon_output_path = resolve_output_path(output_path, epsilon, multiple_outputs)
        write_energy_plot(epsilon_output_path, epsilon, steps, med)
        log(
            f"eps={epsilon:g}: {len(gsd_paths)} trajectories, "
            f"using keys=({STEP_KEY}, {POTENTIAL_KEY}, {KINETIC_KEY})",
        )


if __name__ == "__main__":
    main()
