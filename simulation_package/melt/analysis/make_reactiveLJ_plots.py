#!/usr/bin/env python3
"""Generate ReactiveLJ multi-curve plots without running Tersoff fitting."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPSILONS_DEFAULT = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
REACTIVE_R_CUT_MULT = 1.5
REACTIVE_WEAKENING_DISTANCE_MULT = 0.2
REACTIVE_WEAKENING_INNER_MULT = REACTIVE_R_CUT_MULT - REACTIVE_WEAKENING_DISTANCE_MULT
THIRD_BEAD_DISTANCES_SIGMA = (1.3, 1.4, 1.5)


@dataclass(frozen=True)
class ReactivePlotConfig:
    sigma: float
    reactive_r_cut: float
    weakening_inner: float
    weakening_outer: float
    weakening_exponent: float
    smooth_elbow: bool
    smooth_kappa: float
    smooth_beta: float
    third_bead_multiplier: float
    r_min: float
    n_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the ReactiveLJ-only curve plots from find_tersoff_params.py "
            "without fitting Tersoff parameters."
        )
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(EPSILONS_DEFAULT),
        help="ReactiveLJ epsilon values to plot.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="ReactiveLJ sigma.",
    )
    parser.add_argument(
        "--reactive-r-cut",
        type=float,
        default=1.5,
        help="ReactiveLJ r_cut in units of sigma (fixed at 1.5 in this script).",
    )
    parser.add_argument(
        "--weakening-inner",
        type=float,
        default=None,
        help="ReactiveLJ weakening_inner in units of sigma (fixed at 1.3).",
    )
    parser.add_argument(
        "--weakening-outer",
        type=float,
        default=None,
        help="ReactiveLJ weakening_outer in units of sigma (fixed at 1.5).",
    )
    parser.add_argument(
        "--weakening-exponent",
        type=float,
        default=4.0,
        help="ReactiveLJ weakening exponent.",
    )
    parser.add_argument(
        "--smooth-elbow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ReactiveLJ elbow smoothing.",
    )
    parser.add_argument(
        "--smooth-kappa",
        type=float,
        default=0.05,
        help="ReactiveLJ smooth_kappa.",
    )
    parser.add_argument(
        "--smooth-beta",
        type=float,
        default=1.0,
        help="ReactiveLJ smooth_beta.",
    )
    parser.add_argument(
        "--r-min",
        type=float,
        default=0.85,
        help="Minimum r sampled for curves.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=500,
        help="Number of r points in each curve.",
    )
    parser.add_argument(
        "--third-bead-min-sigma",
        type=float,
        default=1.3,
        help="Minimum third-bead distance in units of sigma.",
    )
    parser.add_argument(
        "--third-bead-max-sigma",
        type=float,
        default=1.5,
        help="Maximum third-bead distance in units of sigma.",
    )
    parser.add_argument(
        "--n-third-bead-distances",
        type=int,
        default=9,
        help="Number of equally spaced third-bead distances between min and max.",
    )
    parser.add_argument(
        "--third-bead-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to third-bead crowding contribution.",
    )
    parser.add_argument(
        "--plot-dir",
        default="reactive_lj_curves",
        help="Directory for output plots, relative to this script.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=-14.0,
        help="Lower y-limit for plots.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=10.0,
        help="Upper y-limit for plots.",
    )
    return parser.parse_args()


def shifted_lj_energy(
    r: np.ndarray | float,
    epsilon: float,
    sigma: float,
    r_cut: float,
) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    r_safe = np.maximum(r_arr, 1e-12)
    sr = sigma / r_safe
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    sigma_over_rcut = sigma / r_cut
    sigma_over_rcut_6 = sigma_over_rcut**6
    energy_shift = 4.0 * epsilon * (sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6)
    return 4.0 * epsilon * (sr12 - sr6) - energy_shift


def shifted_lj_force_magnitude(
    r: np.ndarray | float,
    epsilon: float,
    sigma: float,
    smooth_r_min: float,
) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    r_safe = np.maximum(r_arr, smooth_r_min)
    inv_r = 1.0 / r_safe
    sr = sigma * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    return 24.0 * epsilon * inv_r * (2.0 * sr12 - sr6)


def reactive_weight(distance: float, weakening_inner: float, weakening_outer: float) -> float:
    if distance >= weakening_outer:
        return 0.0
    if distance <= weakening_inner:
        return 1.0
    fraction = (distance - weakening_inner) / (weakening_outer - weakening_inner)
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def reactive_c_exc(raw: float) -> float:
    eps = 1e-6
    return 0.5 * (raw + math.sqrt(raw * raw + eps * eps))


def reactive_lj_curve_with_third_bead(
    r: np.ndarray,
    epsilon: float,
    cfg: ReactivePlotConfig,
    third_bead_distance: float,
) -> np.ndarray:
    base_energy = shifted_lj_energy(r=r, epsilon=epsilon, sigma=cfg.sigma, r_cut=cfg.reactive_r_cut)

    w_third = reactive_weight(
        distance=third_bead_distance,
        weakening_inner=cfg.weakening_inner,
        weakening_outer=cfg.weakening_outer,
    )
    raw = cfg.third_bead_multiplier * w_third
    c_exc = reactive_c_exc(raw)
    weakening = (1.0 + c_exc) ** (-cfg.weakening_exponent)
    pair_energy = np.maximum(base_energy, 0.0) + weakening * np.minimum(base_energy, 0.0)

    if not cfg.smooth_elbow:
        return pair_energy

    sigma_over_rcut = cfg.sigma / cfg.reactive_r_cut
    sigma_over_rcut_6 = sigma_over_rcut**6
    quadratic_rhs = sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6
    discriminant = 1.0 + 4.0 * quadratic_rhs
    if discriminant <= 0.0:
        raise ValueError("Invalid ReactiveLJ geometry: elbow discriminant is non-positive.")

    sr6_at_zero = 0.5 * (1.0 + math.sqrt(discriminant))
    sr_root = sr6_at_zero ** (1.0 / 6.0)
    r_elbow = cfg.sigma / sr_root
    smooth_delta_tol = 1e-6 * cfg.sigma
    smooth_r_min = 1e-7 * cfg.sigma

    one_minus_w = max(0.0, 1.0 - weakening)
    delta = cfg.smooth_kappa * cfg.sigma * (one_minus_w**cfg.smooth_beta)
    r1 = r_elbow - delta
    r2 = r_elbow + delta
    width = r2 - r1

    if r1 <= smooth_r_min:
        raise ValueError("Invalid ReactiveLJ smoothing geometry: r1 <= smooth_r_min.")
    if r2 >= cfg.reactive_r_cut:
        raise ValueError("Invalid ReactiveLJ smoothing geometry: r2 >= r_cut.")

    if delta <= smooth_delta_tol or width <= smooth_delta_tol:
        return pair_energy

    mask = (r > r1) & (r < r2)
    if not np.any(mask):
        return pair_energy

    u1 = float(shifted_lj_energy(r1, epsilon=epsilon, sigma=cfg.sigma, r_cut=cfg.reactive_r_cut))
    du1 = -float(shifted_lj_force_magnitude(r1, epsilon=epsilon, sigma=cfg.sigma, smooth_r_min=smooth_r_min))
    u2 = weakening * float(
        shifted_lj_energy(r2, epsilon=epsilon, sigma=cfg.sigma, r_cut=cfg.reactive_r_cut)
    )
    du2 = -weakening * float(
        shifted_lj_force_magnitude(r2, epsilon=epsilon, sigma=cfg.sigma, smooth_r_min=smooth_r_min)
    )

    t = (r[mask] - r1) / width
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    pair_energy[mask] = h00 * u1 + h10 * width * du1 + h01 * u2 + h11 * width * du2
    return pair_energy


def reactive_curve_set(
    r: np.ndarray,
    epsilon: float,
    cfg: ReactivePlotConfig,
    third_distances: np.ndarray,
) -> np.ndarray:
    curves = [
        reactive_lj_curve_with_third_bead(
            r=r,
            epsilon=epsilon,
            cfg=cfg,
            third_bead_distance=float(distance),
        )
        for distance in third_distances
    ]
    return np.asarray(curves, dtype=np.float64)


def make_colored_curve_plot(
    path: str,
    epsilon: float,
    r: np.ndarray,
    third_distances: np.ndarray,
    curves: np.ndarray,
    sigma: float,
    y_min: float,
    y_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.0), dpi=600)
    plot_mask = (r >= 0.9) & (r <= 1.5)
    if not np.any(plot_mask):
        raise ValueError("No points remain after filtering plot range to 0.9 <= r <= 1.5.")

    r_plot = r[plot_mask]
    curves_plot = curves[:, plot_mask]
    scaled_distances = third_distances / sigma

    cmap = plt.get_cmap("plasma_r")
    norm = matplotlib.colors.Normalize(
        vmin=float(np.min(scaled_distances)),
        vmax=float(np.max(scaled_distances)),
    )
    for idx, dist in enumerate(scaled_distances):
        color = cmap(norm(float(dist)))
        ax.plot(r_plot, curves_plot[idx], color=color, linewidth=1.3)

    ax.set_ylim(bottom=float(y_min), top=float(y_max))

    ax.set_xlabel(r"$r_{AB}/\sigma$", fontsize=10)
    ax.set_ylabel(r"$U(r_{AB})$", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(scaled_distances)
    cbar.set_label(r"$r_{AC}/\sigma$")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.n_third_bead_distances < 2:
        raise ValueError("--n-third-bead-distances must be >= 2")
    if args.third_bead_max_sigma <= args.third_bead_min_sigma:
        raise ValueError("--third-bead-max-sigma must be greater than --third-bead-min-sigma")

    if not np.isclose(float(args.reactive_r_cut), REACTIVE_R_CUT_MULT):
        raise ValueError(
            f"--reactive-r-cut is fixed to {REACTIVE_R_CUT_MULT}*sigma in this script."
        )
    if args.weakening_inner is not None and not np.isclose(
        float(args.weakening_inner),
        REACTIVE_WEAKENING_INNER_MULT,
    ):
        raise ValueError(
            "--weakening-inner is fixed to "
            f"{REACTIVE_WEAKENING_INNER_MULT}*sigma in this script."
        )
    if args.weakening_outer is not None and not np.isclose(
        float(args.weakening_outer),
        REACTIVE_R_CUT_MULT,
    ):
        raise ValueError(
            f"--weakening-outer is fixed to {REACTIVE_R_CUT_MULT}*sigma in this script."
        )

    sigma = float(args.sigma)
    cfg = ReactivePlotConfig(
        sigma=sigma,
        reactive_r_cut=REACTIVE_R_CUT_MULT * sigma,
        weakening_inner=REACTIVE_WEAKENING_INNER_MULT * sigma,
        weakening_outer=REACTIVE_R_CUT_MULT * sigma,
        weakening_exponent=float(args.weakening_exponent),
        smooth_elbow=bool(args.smooth_elbow),
        smooth_kappa=float(args.smooth_kappa),
        smooth_beta=float(args.smooth_beta),
        third_bead_multiplier=float(args.third_bead_multiplier),
        r_min=float(args.r_min),
        n_points=int(args.n_points),
    )

    r_grid = np.linspace(cfg.r_min, cfg.reactive_r_cut, cfg.n_points)
    # Use a compact, fixed third-bead set for clearer overlays.
    third_distances = np.asarray(THIRD_BEAD_DISTANCES_SIGMA, dtype=np.float64) * cfg.sigma

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.abspath(os.path.join(script_dir, args.plot_dir))
    os.makedirs(plot_dir, exist_ok=True)

    for epsilon in args.epsilons:
        epsilon = float(epsilon)
        curves = reactive_curve_set(
            r=r_grid,
            epsilon=epsilon,
            cfg=cfg,
            third_distances=third_distances,
        )
        plot_path = os.path.join(plot_dir, f"reactive_lj_curves_eps_{epsilon:g}.png")
        make_colored_curve_plot(
            path=plot_path,
            epsilon=epsilon,
            r=r_grid,
            third_distances=third_distances,
            curves=curves,
            sigma=cfg.sigma,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        print(f"epsilon={epsilon:g} wrote ReactiveLJ plot: {plot_path}", flush=True)

    print(f"Wrote ReactiveLJ plots: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
