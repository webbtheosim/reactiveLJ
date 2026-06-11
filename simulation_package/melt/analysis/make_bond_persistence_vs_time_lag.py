#!/usr/bin/env python3
"""Plot bond autocorrelation versus normalized time lag for each stickiness."""

from __future__ import annotations

import argparse
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
import matplotlib.pyplot as plt

from analysis_utils import extract_semilog_linear_region


DEFAULT_TAU_R = 4041.0
DEFAULT_FIGWIDTH_PT = 214.5419
DEFAULT_FIGWIDTH_IN = DEFAULT_FIGWIDTH_PT / 72.0
DEFAULT_FIGSIZE = (DEFAULT_FIGWIDTH_IN, DEFAULT_FIGWIDTH_IN * 2.0 / 3.0)
DEFAULT_DPI = 1000
DEFAULT_TICK_FONTSIZE = 8
DEFAULT_LABEL_FONTSIZE = 10
DEFAULT_LEGEND_FONTSIZE = 8


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot the mean bond autocorrelation versus time lag normalized by tau_R "
            "for each linker stickiness."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=script_dir / "results",
        help="Directory containing eps_*/bond_correlation.csv outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "bond_persistence_vs_time_lag.svg",
        help="Output svg path.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit epsilon list to plot. Defaults to all positive epsilons found.",
    )
    parser.add_argument(
        "--include-nonpositive-epsilon",
        action="store_true",
        help="Include epsilon <= 0 when auto-discovering result folders.",
    )
    parser.add_argument(
        "--tau-r",
        type=float,
        default=DEFAULT_TAU_R,
        help=(
            "Rouse time used to normalize the lag axis. The default follows "
            "Liu and O'Connor 2024 for the N=40 unsticky melt."
        ),
    )
    return parser.parse_args()


def parse_epsilon(path: Path) -> float | None:
    if not path.is_dir() or not path.name.startswith("eps_"):
        return None
    try:
        return float(path.name.split("_", maxsplit=1)[1])
    except ValueError:
        return None


def discover_epsilon_dirs(input_root: Path) -> list[tuple[float, Path]]:
    epsilon_dirs: list[tuple[float, Path]] = []
    for candidate in input_root.glob("eps_*"):
        epsilon = parse_epsilon(candidate)
        if epsilon is None:
            continue
        csv_path = candidate / "bond_correlation.csv"
        if csv_path.is_file():
            epsilon_dirs.append((epsilon, candidate))
    return sorted(epsilon_dirs, key=lambda item: item[0])


def select_epsilon_dirs(
    discovered: list[tuple[float, Path]],
    requested_epsilons: list[float] | None,
    include_nonpositive_epsilon: bool,
) -> list[tuple[float, Path]]:
    if requested_epsilons is None:
        if include_nonpositive_epsilon:
            return discovered
        return [(epsilon, path) for epsilon, path in discovered if epsilon > 0.0]

    selected: list[tuple[float, Path]] = []
    for requested in requested_epsilons:
        match = next(
            (
                (epsilon, path)
                for epsilon, path in discovered
                if np.isclose(epsilon, requested, rtol=0.0, atol=1.0e-12)
            ),
            None,
        )
        if match is None:
            raise FileNotFoundError(
                f"Could not find bond_correlation.csv for epsilon={requested:g} in {discovered}"
            )
        selected.append(match)
    return selected


def load_bond_correlation(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        raise ValueError(f"No rows found in {csv_path}")

    time = np.asarray(data[:, 0], dtype=np.float64)
    mean = np.asarray(data[:, 1], dtype=np.float64)
    stderr = np.asarray(data[:, 2], dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(mean) & np.isfinite(stderr)
    time = time[valid]
    mean = mean[valid]
    stderr = stderr[valid]
    if time.size == 0:
        raise ValueError(f"No finite time/mean/stderr data found in {csv_path}")
    return time, mean, stderr


def trim_for_log_plot(
    time_lag: np.ndarray,
    mean_corr: np.ndarray,
    stderr_corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cutoff = np.flatnonzero(
        (mean_corr <= 0.0) | ((mean_corr - stderr_corr) <= 0.0)
    )
    if cutoff.size == 0:
        return time_lag, mean_corr
    stop = int(cutoff[0])
    return time_lag[:stop], mean_corr[:stop]


def build_plot_arrays(
    time_lag: np.ndarray,
    mean_corr: np.ndarray,
    stderr_corr: np.ndarray,
    tau_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_filtered, mean_filtered = trim_for_log_plot(
        time_lag, mean_corr, stderr_corr
    )
    time_filtered, mean_filtered = extract_semilog_linear_region(
        time_filtered, mean_filtered
    )
    if time_filtered.size == 0 or not np.any(mean_filtered > 0.0):
        raise ValueError("No positive correlation values available for log-scale plotting.")

    x_values = np.concatenate(([0.0], time_filtered / tau_r))
    y_values = np.concatenate(([1.0], mean_filtered))
    return x_values, y_values


def format_epsilon_label(epsilon: float) -> str:
    return rf"{epsilon:g}$\mathrm{{k}}_\mathrm{{B}}T$"


def apply_wraparound_tick_style(ax: plt.Axes) -> None:
    for spine in ("bottom", "top", "left", "right"):
        ax.spines[spine].set_visible(True)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=DEFAULT_TICK_FONTSIZE,
    )


def main() -> None:
    args = parse_args()

    discovered = discover_epsilon_dirs(args.input_root)
    if not discovered:
        raise FileNotFoundError(
            f"No eps_*/bond_correlation.csv files found under {args.input_root}"
        )

    selected = select_epsilon_dirs(
        discovered,
        requested_epsilons=args.epsilons,
        include_nonpositive_epsilon=args.include_nonpositive_epsilon,
    )
    if not selected:
        raise ValueError("No epsilon datasets selected for plotting.")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected)))
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)

    for color, (epsilon, eps_dir) in zip(colors, selected):
        time_lag, mean_corr, stderr_corr = load_bond_correlation(
            eps_dir / "bond_correlation.csv"
        )
        x_values, y_values = build_plot_arrays(
            time_lag, mean_corr, stderr_corr, args.tau_r
        )
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.3,
            label=format_epsilon_label(epsilon),
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau / \tau_R^0$", fontsize=DEFAULT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$f(t)$", fontsize=DEFAULT_LABEL_FONTSIZE)
    apply_wraparound_tick_style(ax)
    legend = ax.legend(
        fontsize=DEFAULT_LEGEND_FONTSIZE,
        loc="upper right",
        frameon=False,
        handlelength=0.0,
        handletextpad=0.0,
        borderpad=0.0,
        labelspacing=0.25,
        ncol=1,
    )
    legend_handles = getattr(legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = legend.legendHandles
    for handle, text in zip(legend_handles, legend.get_texts()):
        text.set_color(handle.get_color())
        handle.set_visible(False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)

    print(
        f"Wrote bond autocorrelation plot for epsilons "
        f"{', '.join(f'{epsilon:g}' for epsilon, _ in selected)} to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
