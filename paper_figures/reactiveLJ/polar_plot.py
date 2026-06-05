#!/usr/bin/env python3
"""Plot the three-bead ReactiveLJ energy landscape in midpoint polar coordinates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


SIGMA_DEFAULT = 1.0
EPSILON_DEFAULT = 1.0
REACTIVE_R_CUT_MULT = 1.5
# HOOMD ReactiveLJ defaults this inner weakening radius to 1.3 * sigma.
# This is the "r_min" requested here, where the crowding-driven weakening saturates.
WEAKENING_INNER_MULT = 1.3
# HOOMD ReactiveLJ defaults this outer weakening radius to 1.5 * sigma.
# This is the "r_max" requested here, where the crowding contribution decays to zero.
WEAKENING_OUTER_MULT = 1.5
SMOOTH_KAPPA_DEFAULT = 0.05
SMOOTH_BETA_DEFAULT = 1.0
FIGSIZE_INCHES = (3.3, 3.3)
FIG_DPI = 1000
PLOT_R_MIN_MULT = 0.0
PLOT_R_MAX_MULT = 2.2
BEAD_DIAMETER_MULT = 1.0
MASK_OVERLAP_DISTANCE_MULT = 1.0
BEAD_FACE_COLOR = "#e77500"
BEAD_EDGE_COLOR = "#121212"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a midpoint-centered polar heatmap of the total three-bead "
            "ReactiveLJ energy as bead C moves around a fixed A-B pair."
        )
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA_DEFAULT,
        help="ReactiveLJ sigma.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help="ReactiveLJ epsilon. Energy values scale linearly with epsilon.",
    )
    parser.add_argument(
        "--weakening-exponent",
        type=float,
        default=4.0,
        help="ReactiveLJ weakening exponent p.",
    )
    parser.add_argument(
        "--n-r",
        type=int,
        default=240,
        help="Number of radial samples between r_min and r_max.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=720,
        help="Number of angular samples between 0 and 2*pi.",
    )
    parser.add_argument(
        "--smooth-elbow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the ReactiveLJ Hermite smoothing around the LJ elbow.",
    )
    parser.add_argument(
        "--smooth-kappa",
        type=float,
        default=SMOOTH_KAPPA_DEFAULT,
        help="ReactiveLJ smooth_kappa parameter.",
    )
    parser.add_argument(
        "--smooth-beta",
        type=float,
        default=SMOOTH_BETA_DEFAULT,
        help="ReactiveLJ smooth_beta parameter.",
    )
    parser.add_argument(
        "--quantity",
        choices=("total_barrier", "ab_weakening"),
        default="total_barrier",
        help=(
            "Field to plot: total barrier relative to the isolated A-B dimer, "
            "or only the A-B weakening penalty."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output SVG path. Defaults to this script directory.",
    )
    parser.add_argument(
        "--colorbar-vmin",
        type=float,
        default=None,
        help=(
            "Optional fixed colorbar minimum. If omitted, use the data-driven "
            "default (clamped to 0 when all values are nonnegative)."
        ),
    )
    parser.add_argument(
        "--colorbar-vmax",
        type=float,
        default=None,
        help="Optional fixed colorbar maximum. If omitted, use the data-driven maximum.",
    )
    return parser.parse_args()


def format_exponent_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "m").replace(".", "p")


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


def coordination_weight(
    distance: np.ndarray | float,
    weakening_inner: float,
    weakening_outer: float,
) -> np.ndarray:
    distance_arr = np.asarray(distance, dtype=np.float64)
    weight = np.zeros_like(distance_arr)
    weight = np.where(distance_arr <= weakening_inner, 1.0, weight)
    mask = (distance_arr > weakening_inner) & (distance_arr < weakening_outer)
    if np.any(mask):
        fraction = (distance_arr[mask] - weakening_inner) / (weakening_outer - weakening_inner)
        weight[mask] = 0.5 * (1.0 + np.cos(np.pi * fraction))
    return weight


def reactive_c_exc(raw: np.ndarray | float) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float64)
    eps = 1e-6
    return 0.5 * (raw_arr + np.sqrt(raw_arr * raw_arr + eps * eps))


def reactive_pair_energy(
    pair_distance: np.ndarray,
    raw_crowding: np.ndarray,
    epsilon: float,
    sigma: float,
    r_cut: float,
    weakening_exponent: float,
    smooth_elbow: bool,
    smooth_kappa: float,
    smooth_beta: float,
) -> np.ndarray:
    pair_distance = np.asarray(pair_distance, dtype=np.float64)
    raw_crowding = np.asarray(raw_crowding, dtype=np.float64)
    cutoff_mask = pair_distance < r_cut

    c_exc = reactive_c_exc(raw_crowding)
    weakening = np.power(1.0 + c_exc, -weakening_exponent)

    base_energy = shifted_lj_energy(pair_distance, epsilon=epsilon, sigma=sigma, r_cut=r_cut)
    pair_energy = np.maximum(base_energy, 0.0) + weakening * np.minimum(base_energy, 0.0)
    pair_energy = np.where(cutoff_mask, pair_energy, 0.0)

    if not smooth_elbow:
        return pair_energy

    sigma_over_rcut = sigma / r_cut
    sigma_over_rcut_6 = sigma_over_rcut**6
    quadratic_rhs = sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6
    discriminant = 1.0 + 4.0 * quadratic_rhs
    if discriminant <= 0.0:
        raise ValueError("Invalid ReactiveLJ geometry: elbow discriminant is non-positive.")

    sr6_at_zero = 0.5 * (1.0 + math.sqrt(discriminant))
    sr_root = sr6_at_zero ** (1.0 / 6.0)
    r_elbow = sigma / sr_root
    smooth_delta_tol = 1e-6 * sigma
    smooth_r_min = 1e-7 * sigma

    one_minus_w = np.maximum(0.0, 1.0 - weakening)
    delta = smooth_kappa * sigma * np.power(one_minus_w, smooth_beta)
    r1 = r_elbow - delta
    r2 = r_elbow + delta
    width = r2 - r1

    if np.any(r1 <= smooth_r_min):
        raise ValueError("Invalid ReactiveLJ smoothing geometry: r1 <= smooth_r_min.")
    if np.any(r2 >= r_cut):
        raise ValueError("Invalid ReactiveLJ smoothing geometry: r2 >= r_cut.")

    mask = (
        cutoff_mask
        & (delta > smooth_delta_tol)
        & (width > smooth_delta_tol)
        & (pair_distance > r1)
        & (pair_distance < r2)
    )
    if not np.any(mask):
        return pair_energy

    u1 = shifted_lj_energy(r1, epsilon=epsilon, sigma=sigma, r_cut=r_cut)
    du1 = -shifted_lj_force_magnitude(r1, epsilon=epsilon, sigma=sigma, smooth_r_min=smooth_r_min)
    u2 = weakening * shifted_lj_energy(r2, epsilon=epsilon, sigma=sigma, r_cut=r_cut)
    du2 = -weakening * shifted_lj_force_magnitude(
        r2,
        epsilon=epsilon,
        sigma=sigma,
        smooth_r_min=smooth_r_min,
    )

    t = (pair_distance[mask] - r1[mask]) / width[mask]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    pair_energy = np.array(pair_energy, copy=True)
    pair_energy[mask] = (
        h00 * u1[mask]
        + h10 * width[mask] * du1[mask]
        + h01 * u2[mask]
        + h11 * width[mask] * du2[mask]
    )
    return pair_energy


def total_system_energy(
    radius: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    epsilon: float,
    weakening_exponent: float,
    smooth_elbow: bool,
    smooth_kappa: float,
    smooth_beta: float,
) -> np.ndarray:
    r_cut = REACTIVE_R_CUT_MULT * sigma
    weakening_inner = WEAKENING_INNER_MULT * sigma
    weakening_outer = WEAKENING_OUTER_MULT * sigma

    # The shifted LJ minimum is unchanged from standard LJ:
    # r_eq = 2^(1/6) * sigma.
    r_eq = (2.0 ** (1.0 / 6.0)) * sigma
    half_separation = 0.5 * r_eq

    x_c = radius * np.cos(theta)
    y_c = radius * np.sin(theta)

    d_ab = np.full_like(radius, r_eq)
    d_ac = np.sqrt((x_c + half_separation) ** 2 + y_c**2)
    d_bc = np.sqrt((x_c - half_separation) ** 2 + y_c**2)

    w_ab = coordination_weight(d_ab, weakening_inner=weakening_inner, weakening_outer=weakening_outer)
    w_ac = coordination_weight(d_ac, weakening_inner=weakening_inner, weakening_outer=weakening_outer)
    w_bc = coordination_weight(d_bc, weakening_inner=weakening_inner, weakening_outer=weakening_outer)

    # Leave-one-out crowding from the HOOMD ReactiveLJ implementation:
    # raw_ij = (C_i - w_ij) + (C_j - w_ij).
    raw_ab = w_ac + w_bc
    raw_ac = w_ab + w_bc
    raw_bc = w_ab + w_ac

    u_ab = reactive_pair_energy(
        pair_distance=d_ab,
        raw_crowding=raw_ab,
        epsilon=epsilon,
        sigma=sigma,
        r_cut=r_cut,
        weakening_exponent=weakening_exponent,
        smooth_elbow=smooth_elbow,
        smooth_kappa=smooth_kappa,
        smooth_beta=smooth_beta,
    )
    u_ac = reactive_pair_energy(
        pair_distance=d_ac,
        raw_crowding=raw_ac,
        epsilon=epsilon,
        sigma=sigma,
        r_cut=r_cut,
        weakening_exponent=weakening_exponent,
        smooth_elbow=smooth_elbow,
        smooth_kappa=smooth_kappa,
        smooth_beta=smooth_beta,
    )
    u_bc = reactive_pair_energy(
        pair_distance=d_bc,
        raw_crowding=raw_bc,
        epsilon=epsilon,
        sigma=sigma,
        r_cut=r_cut,
        weakening_exponent=weakening_exponent,
        smooth_elbow=smooth_elbow,
        smooth_kappa=smooth_kappa,
        smooth_beta=smooth_beta,
    )

    # Mask the blow-up region where bead C overlaps either fixed bead.
    # For equal-diameter beads, center overlap begins when the center-center
    # distance is less than one bead diameter, sigma.
    overlap_mask = (d_ac < MASK_OVERLAP_DISTANCE_MULT * sigma) | (
        d_bc < MASK_OVERLAP_DISTANCE_MULT * sigma
    )
    return u_ab + u_ac + u_bc, overlap_mask


def ab_weakening_energy(
    radius: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    epsilon: float,
    weakening_exponent: float,
    smooth_elbow: bool,
    smooth_kappa: float,
    smooth_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_cut = REACTIVE_R_CUT_MULT * sigma
    weakening_inner = WEAKENING_INNER_MULT * sigma
    weakening_outer = WEAKENING_OUTER_MULT * sigma

    r_eq = (2.0 ** (1.0 / 6.0)) * sigma
    half_separation = 0.5 * r_eq

    x_c = radius * np.cos(theta)
    y_c = radius * np.sin(theta)

    d_ac = np.sqrt((x_c + half_separation) ** 2 + y_c**2)
    d_bc = np.sqrt((x_c - half_separation) ** 2 + y_c**2)

    w_ac = coordination_weight(d_ac, weakening_inner=weakening_inner, weakening_outer=weakening_outer)
    w_bc = coordination_weight(d_bc, weakening_inner=weakening_inner, weakening_outer=weakening_outer)
    raw_ab = w_ac + w_bc

    u_ab = reactive_pair_energy(
        pair_distance=np.full_like(radius, r_eq),
        raw_crowding=raw_ab,
        epsilon=epsilon,
        sigma=sigma,
        r_cut=r_cut,
        weakening_exponent=weakening_exponent,
        smooth_elbow=smooth_elbow,
        smooth_kappa=smooth_kappa,
        smooth_beta=smooth_beta,
    )

    overlap_mask = (d_ac < MASK_OVERLAP_DISTANCE_MULT * sigma) | (
        d_bc < MASK_OVERLAP_DISTANCE_MULT * sigma
    )
    return u_ab, overlap_mask


def baseline_system_energy(
    sigma: float,
    epsilon: float,
    weakening_exponent: float,
    smooth_elbow: bool,
    smooth_kappa: float,
    smooth_beta: float,
) -> float:
    r_eq = (2.0 ** (1.0 / 6.0)) * sigma
    r_cut = REACTIVE_R_CUT_MULT * sigma
    baseline = reactive_pair_energy(
        pair_distance=np.asarray(r_eq, dtype=np.float64),
        raw_crowding=np.asarray(0.0, dtype=np.float64),
        epsilon=epsilon,
        sigma=sigma,
        r_cut=r_cut,
        weakening_exponent=weakening_exponent,
        smooth_elbow=smooth_elbow,
        smooth_kappa=smooth_kappa,
        smooth_beta=smooth_beta,
    )
    return float(baseline)


def draw_bead(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    radius: float,
    sigma: float,
) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=True)
    x = x_center + radius * np.cos(phi)
    y = y_center + radius * np.sin(phi)
    theta = np.unwrap(np.arctan2(y, x))
    r = np.sqrt(x * x + y * y) / sigma
    ax.fill(
        theta,
        r,
        facecolor=BEAD_FACE_COLOR,
        edgecolor=BEAD_EDGE_COLOR,
        linewidth=0.8,
        zorder=6,
    )


def build_grid(
    sigma: float,
    epsilon: float,
    weakening_exponent: float,
    n_r: int,
    n_theta: int,
    smooth_elbow: bool,
    smooth_kappa: float,
    smooth_beta: float,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_min = PLOT_R_MIN_MULT * sigma
    r_max = PLOT_R_MAX_MULT * sigma

    radius_edges = np.linspace(r_min, r_max, n_r + 1, dtype=np.float64)
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1, dtype=np.float64)

    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_grid, radius_grid = np.meshgrid(theta_centers, radius_centers)

    baseline = baseline_system_energy(
        sigma=sigma,
        epsilon=epsilon,
        weakening_exponent=weakening_exponent,
        smooth_elbow=smooth_elbow,
        smooth_kappa=smooth_kappa,
        smooth_beta=smooth_beta,
    )
    if quantity == "total_barrier":
        energy, overlap_mask = total_system_energy(
            radius=radius_grid,
            theta=theta_grid,
            sigma=sigma,
            epsilon=epsilon,
            weakening_exponent=weakening_exponent,
            smooth_elbow=smooth_elbow,
            smooth_kappa=smooth_kappa,
            smooth_beta=smooth_beta,
        )
        field = energy - baseline
    elif quantity == "ab_weakening":
        u_ab, overlap_mask = ab_weakening_energy(
            radius=radius_grid,
            theta=theta_grid,
            sigma=sigma,
            epsilon=epsilon,
            weakening_exponent=weakening_exponent,
            smooth_elbow=smooth_elbow,
            smooth_kappa=smooth_kappa,
            smooth_beta=smooth_beta,
        )
        field = u_ab - baseline
    else:
        raise ValueError(f"Unsupported quantity: {quantity}")
    return theta_edges, radius_edges / sigma, np.ma.masked_where(overlap_mask, field)


def plot_energy_landscape(
    theta_edges: np.ndarray,
    radius_edges_scaled: np.ndarray,
    energy: np.ndarray,
    output_path: Path,
    sigma: float,
    quantity: str,
    colorbar_vmin: float | None = None,
    colorbar_vmax: float | None = None,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_INCHES, dpi=FIG_DPI, constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")

    theta_edge_grid, radius_edge_grid = np.meshgrid(theta_edges, radius_edges_scaled)
    finite_values = energy.compressed()
    e_min = float(np.min(finite_values))
    e_max = float(np.max(finite_values))
    resolved_vmin = (
        float(colorbar_vmin)
        if colorbar_vmin is not None
        else (0.0 if e_min >= -1e-8 else e_min)
    )
    resolved_vmax = float(colorbar_vmax) if colorbar_vmax is not None else e_max
    if resolved_vmax <= resolved_vmin:
        raise ValueError("--colorbar-vmax must be greater than --colorbar-vmin.")
    norm = mcolors.Normalize(vmin=resolved_vmin, vmax=resolved_vmax)

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="white")
    mesh = ax.pcolormesh(
        theta_edge_grid,
        radius_edge_grid,
        energy,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )

    ax.set_ylim(float(radius_edges_scaled[0]), float(radius_edges_scaled[-1]))
    ax.set_xticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
    ax.set_yticks([1.0, 2.0])
    ax.set_rlabel_position(135)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.35, linewidth=0.4)

    r_eq = (2.0 ** (1.0 / 6.0)) * sigma
    bead_radius = 0.5 * BEAD_DIAMETER_MULT * sigma
    draw_bead(ax, x_center=-0.5 * r_eq, y_center=0.0, radius=bead_radius, sigma=sigma)
    draw_bead(ax, x_center=0.5 * r_eq, y_center=0.0, radius=bead_radius, sigma=sigma)

    # Radial coordinate is measured from the midpoint between A and B. The plot
    # is zoomed to the requested annulus r_min <= r <= r_max.
    ax.set_xlabel(r"$\theta$", fontsize=10, labelpad=2)
    ax.set_ylabel(r"$r_{\mathrm{mid}\rightarrow C}/\sigma$", fontsize=10, labelpad=20)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.92)
    cbar.ax.tick_params(labelsize=8)
    if quantity == "ab_weakening":
        cbar.set_label(r"$\Delta U_{AB}$", fontsize=10)
    else:
        cbar.set_label("Energy barrier", fontsize=10)

    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_r < 2:
        raise ValueError("--n-r must be at least 2.")
    if args.n_theta < 4:
        raise ValueError("--n-theta must be at least 4.")

    script_dir = Path(__file__).resolve().parent
    p_tag = format_exponent_tag(float(args.weakening_exponent))
    quantity_tag = "" if args.quantity == "total_barrier" else f"_{args.quantity}"
    output_path = args.output or (script_dir / f"polar_plot{quantity_tag}_p{p_tag}.svg")

    theta_edges, radius_edges_scaled, energy = build_grid(
        sigma=float(args.sigma),
        epsilon=float(args.epsilon),
        weakening_exponent=float(args.weakening_exponent),
        n_r=int(args.n_r),
        n_theta=int(args.n_theta),
        smooth_elbow=bool(args.smooth_elbow),
        smooth_kappa=float(args.smooth_kappa),
        smooth_beta=float(args.smooth_beta),
        quantity=args.quantity,
    )
    plot_energy_landscape(
        theta_edges=theta_edges,
        radius_edges_scaled=radius_edges_scaled,
        energy=energy,
        output_path=output_path,
        sigma=float(args.sigma),
        quantity=args.quantity,
        colorbar_vmin=args.colorbar_vmin,
        colorbar_vmax=args.colorbar_vmax,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
