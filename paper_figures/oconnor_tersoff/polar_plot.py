#!/usr/bin/env python3
"""Plot the three-bead O'Connor Tersoff landscape in midpoint polar coordinates."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


SIGMA_DEFAULT = 1.0
U_DEFAULT = 3.0
R0_MULT = 1.0
PLOT_R_MIN_MULT = 0.0
PLOT_R_MAX_MULT = 2.2
BEAD_DIAMETER_MULT = 2.0 ** (-1.0 / 6.0)
MASK_OVERLAP_DISTANCE_MULT = 1.0
FIGSIZE_INCHES = (3.3, 3.3)
FIG_DPI = 1000
BEAD_FACE_COLOR = "#e77500"
BEAD_EDGE_COLOR = "#121212"


@dataclass(frozen=True)
class TersoffParameters:
    model: str
    sigma: float
    u: float
    r0: float
    alpha: float
    beta: float
    n: float
    r_v: float
    d_v: float
    r_zeta: float
    d_zeta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create midpoint-centered polar heatmaps of the total three-bead "
            "O'Connor Tersoff energy as bead C moves around a fixed A-B pair."
        )
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA_DEFAULT,
        help="Length unit sigma.",
    )
    parser.add_argument(
        "--u",
        type=float,
        default=U_DEFAULT,
        help="Sticky cohesive energy U/kBT. The paper's Figure 1d uses U/kBT = 3.",
    )
    parser.add_argument(
        "--model",
        choices=("both", "ex0", "exu2"),
        default="both",
        help="Which paper Tersoff model to plot.",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to this script directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default="polar_plot",
        help="Prefix for generated SVG files.",
    )
    return parser.parse_args()


def format_float_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "m").replace(".", "p")


def paper_tersoff_parameters(model: str, sigma: float, u: float) -> TersoffParameters:
    """Return Table 1 parameters from Liu and O'Connor for one barrier model."""
    if u <= 0.0:
        raise ValueError("--u must be positive.")
    if sigma <= 0.0:
        raise ValueError("--sigma must be positive.")

    sqrt_u_over_3 = math.sqrt(u / 3.0)
    r_v = (1.0 + 0.1 * sqrt_u_over_3) * sigma
    d_v = 0.1 * sqrt_u_over_3 * sigma

    if model == "ex0":
        r_zeta = (1.0 - 0.193877 * sqrt_u_over_3) * sigma
        d_zeta = 0.393877 * sqrt_u_over_3 * sigma
    elif model == "exu2":
        r_zeta = (1.0 + 0.05 * sqrt_u_over_3) * sigma
        d_zeta = 0.15 * sqrt_u_over_3 * sigma
    else:
        raise ValueError(f"Unsupported Tersoff model: {model}")

    return TersoffParameters(
        model=model,
        sigma=sigma,
        u=u,
        r0=R0_MULT * sigma,
        alpha=6.0 / (sqrt_u_over_3 * sigma),
        beta=31.449697,
        n=1.451724,
        r_v=r_v,
        d_v=d_v,
        r_zeta=r_zeta,
        d_zeta=d_zeta,
    )


def tersoff_cutoff(
    distance: np.ndarray | float,
    center: float,
    half_width: float,
) -> np.ndarray:
    distance_arr = np.asarray(distance, dtype=np.float64)
    values = np.ones_like(distance_arr)

    r_inner = center - half_width
    r_outer = center + half_width
    shell = (distance_arr > r_inner) & (distance_arr < r_outer)

    values = np.where(distance_arr >= r_outer, 0.0, values)
    if np.any(shell):
        x = (distance_arr[shell] - center) / half_width
        values[shell] = 0.5 - 0.5 * np.sin(0.5 * np.pi * x)
    return values


def _safe_exp(value: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(value, -80.0, 80.0))


def morse_terms(distance: np.ndarray, params: TersoffParameters) -> tuple[np.ndarray, np.ndarray]:
    dr = np.asarray(distance, dtype=np.float64) - params.r0
    repulsive = params.u * _safe_exp(-2.0 * params.alpha * dr)
    attractive = -2.0 * params.u * _safe_exp(-params.alpha * dr)
    return repulsive, attractive


def bond_order(zeta: np.ndarray, params: TersoffParameters) -> np.ndarray:
    zeta_arr = np.maximum(np.asarray(zeta, dtype=np.float64), 0.0)
    beta_zeta_n = np.power(params.beta * zeta_arr, params.n)
    return np.power(1.0 + beta_zeta_n, -0.5 / params.n)


def directed_pair_energy(
    pair_distance: np.ndarray,
    zeta: np.ndarray,
    params: TersoffParameters,
) -> np.ndarray:
    f_v = tersoff_cutoff(pair_distance, center=params.r_v, half_width=params.d_v)
    f_r, f_a = morse_terms(pair_distance, params=params)
    b_ij = bond_order(zeta, params=params)
    return 0.5 * f_v * (f_r + b_ij * f_a)


def total_system_energy(
    radius: np.ndarray,
    theta: np.ndarray,
    params: TersoffParameters,
) -> tuple[np.ndarray, np.ndarray]:
    half_separation = 0.5 * params.r0

    x_c = radius * np.cos(theta)
    y_c = radius * np.sin(theta)

    d_ab = np.full_like(radius, params.r0)
    d_ac = np.sqrt((x_c + half_separation) ** 2 + y_c**2)
    d_bc = np.sqrt((x_c - half_separation) ** 2 + y_c**2)

    z_ab = tersoff_cutoff(d_ac, center=params.r_zeta, half_width=params.d_zeta)
    z_ba = tersoff_cutoff(d_bc, center=params.r_zeta, half_width=params.d_zeta)
    z_ac = tersoff_cutoff(d_ab, center=params.r_zeta, half_width=params.d_zeta)
    z_ca = tersoff_cutoff(d_bc, center=params.r_zeta, half_width=params.d_zeta)
    z_bc = tersoff_cutoff(d_ab, center=params.r_zeta, half_width=params.d_zeta)
    z_cb = tersoff_cutoff(d_ac, center=params.r_zeta, half_width=params.d_zeta)

    energy = (
        directed_pair_energy(d_ab, z_ab, params=params)
        + directed_pair_energy(d_ab, z_ba, params=params)
        + directed_pair_energy(d_ac, z_ac, params=params)
        + directed_pair_energy(d_ac, z_ca, params=params)
        + directed_pair_energy(d_bc, z_bc, params=params)
        + directed_pair_energy(d_bc, z_cb, params=params)
    )

    mask_distance = MASK_OVERLAP_DISTANCE_MULT * params.sigma
    overlap_mask = (d_ac < mask_distance) | (d_bc < mask_distance)
    return energy, overlap_mask


def baseline_system_energy(params: TersoffParameters) -> float:
    distance = np.asarray(params.r0, dtype=np.float64)
    zeta = np.asarray(0.0, dtype=np.float64)
    baseline = 2.0 * directed_pair_energy(distance, zeta, params=params)
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
    params: TersoffParameters,
    n_r: int,
    n_theta: int,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    sigma = params.sigma
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
        params=params,
    )
    field = energy - baseline_system_energy(params)
    return theta_edges, radius_edges / sigma, np.ma.masked_where(overlap_mask, field)


def plot_energy_landscape(
    theta_edges: np.ndarray,
    radius_edges_scaled: np.ndarray,
    energy: np.ma.MaskedArray,
    output_path: Path,
    params: TersoffParameters,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_INCHES, dpi=FIG_DPI, constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")

    theta_edge_grid, radius_edge_grid = np.meshgrid(theta_edges, radius_edges_scaled)
    finite_values = energy.compressed()
    e_min = float(np.min(finite_values))
    e_max = float(np.max(finite_values))
    vmin = 0.0 if e_min >= -1e-8 else e_min
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

    bead_radius = 0.5 * BEAD_DIAMETER_MULT * params.r0
    draw_bead(
        ax,
        x_center=-0.5 * params.r0,
        y_center=0.0,
        radius=bead_radius,
        sigma=params.sigma,
    )
    draw_bead(
        ax,
        x_center=0.5 * params.r0,
        y_center=0.0,
        radius=bead_radius,
        sigma=params.sigma,
    )

    ax.set_xlabel(r"$\theta$", fontsize=10, labelpad=2)
    ax.set_ylabel(r"$r_{\mathrm{mid}\rightarrow C}/\sigma$", fontsize=10, labelpad=20)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.92)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Energy barrier", fontsize=10)

    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_r < 2:
        raise ValueError("--n-r must be at least 2.")
    if args.n_theta < 4:
        raise ValueError("--n-theta must be at least 4.")

    output_dir = args.output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ("ex0", "exu2") if args.model == "both" else (args.model,)
    u_tag = format_float_tag(float(args.u))

    for model in models:
        params = paper_tersoff_parameters(
            model=model,
            sigma=float(args.sigma),
            u=float(args.u),
        )
        theta_edges, radius_edges_scaled, energy = build_grid(
            params=params,
            n_r=int(args.n_r),
            n_theta=int(args.n_theta),
        )
        output_path = output_dir / f"{args.output_prefix}_{model}_u{u_tag}.svg"
        plot_energy_landscape(
            theta_edges=theta_edges,
            radius_edges_scaled=radius_edges_scaled,
            energy=energy,
            output_path=output_path,
            params=params,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
