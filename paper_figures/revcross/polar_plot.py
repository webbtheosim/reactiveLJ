#!/usr/bin/env python3
"""Plot the three-bead RevCross energy landscape in midpoint polar coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


SIGMA_DEFAULT = 1.0
EPSILON_DEFAULT = 1.0
N_EXPONENT_DEFAULT = 10.0
LAMBDA3_DEFAULT = 1.0
LAMBDA3_DEFAULTS = (1.0, 2.0)
R_CUT_MULT_DEFAULT = 1.3
FIGSIZE_INCHES = (3.3, 3.3)
FIG_DPI = 1000
PLOT_R_MIN_MULT = 0.0
PLOT_R_MAX_MULT = 2.2
BEAD_DIAMETER_MULT = 1.0
MASK_OVERLAP_DISTANCE_MULT = 1.0
A_BEAD_FACE_COLOR = "#e77500"
B_BEAD_FACE_COLOR = "#5b7bd5"
BEAD_EDGE_COLOR = "#121212"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a midpoint-centered polar heatmap of the total three-bead "
            "RevCross energy for a moving A bead around a fixed A-B pair."
        )
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA_DEFAULT,
        help="RevCross sigma.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help="RevCross epsilon. Energy values scale linearly with epsilon.",
    )
    parser.add_argument(
        "--n",
        type=float,
        default=N_EXPONENT_DEFAULT,
        help="RevCross generalized-LJ exponent n.",
    )
    parser.add_argument(
        "--lambda3",
        type=float,
        nargs="+",
        default=list(LAMBDA3_DEFAULTS),
        help=(
            "RevCross three-body factor lambda3 values to plot. "
            "lambda3=1 is barrierless swapping. Defaults to 1 and 2."
        ),
    )
    parser.add_argument(
        "--r-cut",
        type=float,
        default=None,
        help=(
            "RevCross A-B cutoff distance. Defaults to "
            f"{R_CUT_MULT_DEFAULT:g} * sigma."
        ),
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
        "--output",
        type=Path,
        default=None,
        help="Optional output SVG path. Defaults to this script directory.",
    )
    return parser.parse_args()


def format_float_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "m").replace(".", "p")


def revcross_r_min(sigma: float, n: float) -> float:
    return sigma * (2.0 ** (1.0 / n))


def revcross_pair_energy(
    distance: np.ndarray | float,
    sigma: float,
    epsilon: float,
    n: float,
    r_cut: float,
) -> np.ndarray:
    distance_arr = np.asarray(distance, dtype=np.float64)
    r_safe = np.maximum(distance_arr, 1e-12)
    ratio_n = np.power(sigma / r_safe, n)
    energy = 4.0 * epsilon * (ratio_n * ratio_n - ratio_n)
    return np.where(distance_arr < r_cut, energy, 0.0)


def same_type_wca_energy(
    distance: np.ndarray | float,
    sigma: float,
    epsilon: float,
    n: float,
) -> np.ndarray:
    distance_arr = np.asarray(distance, dtype=np.float64)
    r_min = revcross_r_min(sigma=sigma, n=n)
    r_safe = np.maximum(distance_arr, 1e-12)
    ratio_n = np.power(sigma / r_safe, n)
    energy = 4.0 * epsilon * (ratio_n * ratio_n - ratio_n) + epsilon
    return np.where(distance_arr < r_min, energy, 0.0)


def revcross_hat_pair_energy(
    distance: np.ndarray | float,
    sigma: float,
    epsilon: float,
    n: float,
    r_cut: float,
) -> np.ndarray:
    distance_arr = np.asarray(distance, dtype=np.float64)
    r_min = revcross_r_min(sigma=sigma, n=n)
    pair_energy = revcross_pair_energy(
        distance_arr,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
    )
    hat = np.where(distance_arr <= r_min, 1.0, -pair_energy / epsilon)
    return np.where(distance_arr < r_cut, hat, 0.0)


def revcross_three_body_energy(
    d_ba_fixed: np.ndarray,
    d_ba_moving: np.ndarray,
    sigma: float,
    epsilon: float,
    n: float,
    r_cut: float,
    lambda3: float,
) -> np.ndarray:
    h_fixed = revcross_hat_pair_energy(
        d_ba_fixed,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
    )
    h_moving = revcross_hat_pair_energy(
        d_ba_moving,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
    )
    return lambda3 * epsilon * h_fixed * h_moving


def total_system_energy(
    radius: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    epsilon: float,
    n: float,
    r_cut: float,
    lambda3: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_eq = revcross_r_min(sigma=sigma, n=n)
    half_separation = 0.5 * r_eq

    x_c = radius * np.cos(theta)
    y_c = radius * np.sin(theta)

    # Fixed A is on the left, fixed B is on the right, and the moving bead is A.
    d_ab = np.full_like(radius, r_eq)
    d_a_moving_b = np.sqrt((x_c - half_separation) ** 2 + y_c**2)
    d_a_fixed_a_moving = np.sqrt((x_c + half_separation) ** 2 + y_c**2)

    # RevCross is active only for complementary A-B pairs. Same-type A-A
    # interactions use the repulsive WCA branch; there is no B-B pair here.
    u_ab = revcross_pair_energy(
        d_ab,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
    )
    u_moving_b = revcross_pair_energy(
        d_a_moving_b,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
    )
    u_three_body = revcross_three_body_energy(
        d_ba_fixed=d_ab,
        d_ba_moving=d_a_moving_b,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
        lambda3=lambda3,
    )
    u_same_type = same_type_wca_energy(
        d_a_fixed_a_moving,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
    )

    mask_distance = MASK_OVERLAP_DISTANCE_MULT * sigma
    overlap_mask = (d_a_fixed_a_moving < mask_distance) | (
        d_a_moving_b < mask_distance
    )
    return u_ab + u_moving_b + u_three_body + u_same_type, overlap_mask


def baseline_system_energy(
    sigma: float,
    epsilon: float,
    n: float,
    r_cut: float,
) -> float:
    r_eq = revcross_r_min(sigma=sigma, n=n)
    baseline = revcross_pair_energy(
        np.asarray(r_eq, dtype=np.float64),
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
    )
    return float(baseline)


def draw_bead(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    radius: float,
    sigma: float,
    face_color: str,
) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=True)
    x = x_center + radius * np.cos(phi)
    y = y_center + radius * np.sin(phi)
    theta = np.unwrap(np.arctan2(y, x))
    r = np.sqrt(x * x + y * y) / sigma
    ax.fill(
        theta,
        r,
        facecolor=face_color,
        edgecolor=BEAD_EDGE_COLOR,
        linewidth=0.8,
        zorder=6,
    )


def build_grid(
    sigma: float,
    epsilon: float,
    n: float,
    r_cut: float,
    lambda3: float,
    n_r: int,
    n_theta: int,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    r_min = PLOT_R_MIN_MULT * sigma
    r_max = PLOT_R_MAX_MULT * sigma

    radius_edges = np.linspace(r_min, r_max, n_r + 1, dtype=np.float64)
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1, dtype=np.float64)

    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_grid, radius_grid = np.meshgrid(theta_centers, radius_centers)

    energy, overlap_mask = total_system_energy(
        radius=radius_grid,
        theta=theta_grid,
        sigma=sigma,
        epsilon=epsilon,
        n=n,
        r_cut=r_cut,
        lambda3=lambda3,
    )
    baseline = baseline_system_energy(sigma=sigma, epsilon=epsilon, n=n, r_cut=r_cut)
    field = energy - baseline
    return theta_edges, radius_edges / sigma, np.ma.masked_where(overlap_mask, field)


def plot_energy_landscape(
    theta_edges: np.ndarray,
    radius_edges_scaled: np.ndarray,
    energy: np.ma.MaskedArray,
    output_path: Path,
    sigma: float,
    n: float,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_INCHES, dpi=FIG_DPI, constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")

    theta_edge_grid, radius_edge_grid = np.meshgrid(theta_edges, radius_edges_scaled)
    finite_values = energy.compressed()
    e_min = float(np.min(finite_values))
    e_max = float(np.max(finite_values))
    vmin = 0.0 if e_min >= -1e-8 else e_min
    if abs(e_max - vmin) < 1e-12:
        e_max = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=e_max)

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

    r_eq = revcross_r_min(sigma=sigma, n=n)
    bead_radius = 0.5 * BEAD_DIAMETER_MULT * sigma
    draw_bead(
        ax,
        x_center=-0.5 * r_eq,
        y_center=0.0,
        radius=bead_radius,
        sigma=sigma,
        face_color=A_BEAD_FACE_COLOR,
    )
    draw_bead(
        ax,
        x_center=0.5 * r_eq,
        y_center=0.0,
        radius=bead_radius,
        sigma=sigma,
        face_color=B_BEAD_FACE_COLOR,
    )

    ax.set_xlabel(r"$\theta$", fontsize=10, labelpad=2)
    ax.set_ylabel(r"$r_{\mathrm{mid}\rightarrow A_2}/\sigma$", fontsize=10, labelpad=20)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.92)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$\Delta E = E_{A_1BA_2} - E_{A_1B}$", fontsize=10)

    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.sigma <= 0.0:
        raise ValueError("--sigma must be positive.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be positive.")
    if args.n <= 0.0:
        raise ValueError("--n must be positive.")
    lambda_values = [float(value) for value in args.lambda3]
    if any(value < 0.0 for value in lambda_values):
        raise ValueError("--lambda3 values must be non-negative.")
    if args.n_r < 2:
        raise ValueError("--n-r must be at least 2.")
    if args.n_theta < 4:
        raise ValueError("--n-theta must be at least 4.")

    sigma = float(args.sigma)
    r_cut = float(args.r_cut) if args.r_cut is not None else R_CUT_MULT_DEFAULT * sigma
    if r_cut <= revcross_r_min(sigma=sigma, n=float(args.n)):
        raise ValueError("--r-cut must be greater than the RevCross pair minimum.")

    script_dir = Path(__file__).resolve().parent
    if args.output is not None and len(lambda_values) != 1:
        raise ValueError("--output can only be used when exactly one --lambda3 value is requested.")

    for lambda3 in lambda_values:
        lambda_tag = format_float_tag(lambda3)
        output_path = args.output or (script_dir / f"polar_plot_lambda{lambda_tag}.svg")

        theta_edges, radius_edges_scaled, energy = build_grid(
            sigma=sigma,
            epsilon=float(args.epsilon),
            n=float(args.n),
            r_cut=r_cut,
            lambda3=lambda3,
            n_r=int(args.n_r),
            n_theta=int(args.n_theta),
        )
        plot_energy_landscape(
            theta_edges=theta_edges,
            radius_edges_scaled=radius_edges_scaled,
            energy=energy,
            output_path=output_path,
            sigma=sigma,
            n=float(args.n),
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
