#!/usr/bin/env python3
"""Fit Liu/O'Connor Tersoff parameters to ReactiveLJ targets."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from functools import partial

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


EPSILONS_DEFAULT = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
REACTIVE_R_CUT_MULT = 1.5
REACTIVE_WEAKENING_DISTANCE_MULT = 0.2
REACTIVE_WEAKENING_INNER_MULT = REACTIVE_R_CUT_MULT - REACTIVE_WEAKENING_DISTANCE_MULT

# Ordered for optimizer vectors / CSV columns.
PARAM_ORDER = (
    "A1",
    "A2",
    "lambda1",
    "lambda2",
    "lambda3",
    "dimer_r",
    "cutoff_thickness",
    "r_cut",
    "alpha",
    "n",
    "gamma",
    "c",
    "d",
    "m",
)
PARAM_INDEX = {name: idx for idx, name in enumerate(PARAM_ORDER)}


@dataclass(frozen=True)
class FitConfig:
    sigma: float
    reactive_r_cut: float
    weakening_inner: float
    weakening_outer: float
    weakening_exponent: float
    smooth_elbow: bool
    smooth_kappa: float
    smooth_beta: float
    third_bead_multiplier: float
    n_points: int
    adam_steps: int
    adam_lr: float
    plateau_min_steps: int
    plateau_patience: int
    plateau_min_delta: float
    beta1: float
    beta2: float
    adam_eps: float
    restarts: int
    surrogate_cos_theta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the Liu/O'Connor Tersoff subset to match ReactiveLJ pair curves "
            "at fixed third-bead distances using Adam with JAX autodiff gradients."
        )
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(EPSILONS_DEFAULT),
        help="ReactiveLJ epsilon values to fit.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="ReactiveLJ sigma (and Tersoff length scale baseline).",
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
        help="Enable/disable ReactiveLJ elbow smoothing for target curves.",
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
        "--n-points",
        type=int,
        default=600,
        help="Number of r points in each fitted curve.",
    )
    parser.add_argument(
        "--adam-steps",
        type=int,
        default=1200,
        help="Maximum Adam steps per fit attempt (early stopping may stop sooner).",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--plateau-min-steps",
        type=int,
        default=300,
        help="Minimum Adam steps before plateau early stopping is allowed.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=200,
        help="Stop when no meaningful improvement is seen for this many steps.",
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=1e-8,
        help="Minimum absolute loss decrease required to reset plateau patience.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of random-restart fits per epsilon.",
    )
    parser.add_argument(
        "--n-third-bead-distances",
        type=int,
        default=10,
        help=(
            "Number of third-bead distances sampled by uniform spacing in "
            "distance^weakening_exponent between fixed ReactiveLJ bounds [1.3, 1.5] * sigma."
        ),
    )
    parser.add_argument(
        "--third-bead-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for the leave-one-out crowding raw value from the third bead. "
            "Use 1.0 for one-sided contribution, 2.0 for symmetric contribution."
        ),
    )
    parser.add_argument(
        "--surrogate-cos-theta",
        type=float,
        default=0.0,
        help=(
            "Deprecated and ignored. The Liu/O'Connor Tersoff subset has g(theta)=1 "
            "and no angular dependence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for fitted CSV files.",
    )
    parser.add_argument(
        "--plot-dir",
        default="plots",
        help="Directory for overlay curve plots.",
    )
    parser.add_argument(
        "--params-csv",
        default=None,
        help="Output CSV path for fitted parameters (defaults to <output-dir>/tersoff_fitted_params.csv).",
    )
    return parser.parse_args()


def _safe_exp(x: np.ndarray) -> np.ndarray:
    # Keep exponentials in a numerically stable range during optimization.
    return np.exp(np.clip(x, -20.0, 20.0))


def shifted_lj_energy(r: np.ndarray | float, epsilon: float, sigma: float, r_cut: float) -> np.ndarray:
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
    # Match the smooth positive-part used in ReactiveLJForceCompute.
    eps = 1e-6
    return 0.5 * (raw + math.sqrt(raw * raw + eps * eps))


def reactive_lj_curve_with_third_bead(
    r: np.ndarray,
    epsilon: float,
    cfg: FitConfig,
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
    cfg: FitConfig,
    third_distances: np.ndarray,
) -> np.ndarray:
    curves = [
        reactive_lj_curve_with_third_bead(
            r=r,
            epsilon=epsilon,
            cfg=cfg,
            third_bead_distance=float(dist),
        )
        for dist in third_distances
    ]
    return np.asarray(curves, dtype=np.float64)


def compute_third_bead_distances(
    r_inner: float, r_outer: float, exponent: float, count: int
) -> np.ndarray:
    """Sample third-bead distances via uniform spacing in distance^exponent."""
    if count <= 1:
        return np.asarray([r_inner], dtype=np.float64)
    powered = np.linspace(r_outer**exponent, r_inner**exponent, count)
    return np.asarray(powered ** (1.0 / exponent), dtype=np.float64)


def tersoff_cutoff(r: np.ndarray, r_cut: float, cutoff_thickness: float, alpha: float) -> np.ndarray:
    r_inner = r_cut - cutoff_thickness
    f_c = np.ones_like(r)

    mask_outer = r >= r_cut
    mask_shell = (r > r_inner) & (r < r_cut)

    if np.any(mask_shell):
        x = (r[mask_shell] - r_inner) / cutoff_thickness
        x3 = x * x * x
        denom = x3 - 1.0
        f_c[mask_shell] = np.exp((-alpha) * x3 / denom)

    f_c[mask_outer] = 0.0
    return f_c


def tersoff_surrogate_curve(
    r: np.ndarray,
    params: dict[str, float],
    r_cut: float,
    rik_distance: float,
    surrogate_cos_theta: float,
) -> np.ndarray:
    a1 = params["A1"]
    a2 = params["A2"]
    lam1 = params["lambda1"]
    lam2 = params["lambda2"]
    dimer_r = params["dimer_r"]
    cutoff_thickness = params["cutoff_thickness"]
    alpha = params["alpha"]
    n = params["n"]
    gamma = params["gamma"]

    f_c_ij = tersoff_cutoff(r, r_cut=r_cut, cutoff_thickness=cutoff_thickness, alpha=alpha)
    f_r = a1 * _safe_exp(lam1 * (dimer_r - r))
    f_a = a2 * _safe_exp(lam2 * (dimer_r - r))

    rik = np.full_like(r, rik_distance)
    f_c_ik = tersoff_cutoff(rik, r_cut=r_cut, cutoff_thickness=cutoff_thickness, alpha=alpha)

    # Match LiuOConnorTersoffForceCompute: for one third bead, the directed
    # bond-order coordination excluding the ij pair is zeta_ij = f_C(r_ik).
    # The Liu/O'Connor subset has lambda3=0 and g(theta)=1.
    del surrogate_cos_theta
    zeta = np.maximum(f_c_ik, 0.0)
    n_eff = max(n, 1e-6)
    gamma_n = gamma**n_eff
    zeta_n = np.power(np.where(zeta > 0.0, zeta, 1.0), n_eff)
    bij_raw = np.power(1.0 + gamma_n * zeta_n, -0.5 / n_eff)
    bij = np.where(zeta > 0.0, bij_raw, 1.0)

    # Match LiuOConnorTersoffForceCompute's directed energy.
    return 0.5 * f_c_ij * (f_r - bij * f_a)


def vector_to_params(vec: np.ndarray) -> dict[str, float]:
    return {name: float(vec[i]) for i, name in enumerate(PARAM_ORDER)}


def params_to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array([params[name] for name in PARAM_ORDER], dtype=np.float64)


def build_bounds(
    sigma: float,
    reactive_r_cut: float,
    cutoff_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            1.0,    # A1
            1.0,    # A2 (positive, HOOMD uses f_R - b_ij f_A)
            0.5,    # lambda1
            0.5,    # lambda2
            0.0,    # lambda3
            0.5 * sigma,   # dimer_r
            cutoff_thickness,  # cutoff_thickness (fixed)
            reactive_r_cut,    # r_cut (fixed)
            -3.0,   # alpha (fixed)
            0.5,    # n
            0.3,    # gamma
            0.0,    # c (fixed)
            1.0,    # d (fixed)
            0.0,    # m (fixed)
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            500.0,          # A1
            500.0,          # A2 (positive, HOOMD uses f_R - b_ij f_A)
            30.0,           # lambda1
            30.0,           # lambda2
            0.0,            # lambda3 (fixed: Liu/O'Connor subset)
            1.5 * sigma,    # dimer_r
            cutoff_thickness,  # cutoff_thickness (fixed)
            reactive_r_cut,    # r_cut (fixed)
            -3.0,           # alpha (fixed)
            8.0,            # n
            50.0,           # gamma
            0.0,            # c (fixed: g(theta)=1)
            1.0,            # d (fixed and unused)
            0.0,            # m (fixed and unused)
        ],
        dtype=np.float64,
    )
    return lower, upper


def build_default_params(
    epsilon: float,
    sigma: float,
    reactive_r_cut: float,
    cutoff_thickness: float,
) -> dict[str, float]:
    initial_scale = min(500.0, max(1.0, 20.0 * float(epsilon)))
    return {
        "A1": initial_scale,
        "A2": initial_scale,
        "lambda1": 3.0,
        "lambda2": 2.0,
        "lambda3": 0.0,
        "dimer_r": sigma,
        "cutoff_thickness": cutoff_thickness,
        "r_cut": reactive_r_cut,
        "alpha": -3.0,
        "n": 1.0,
        "gamma": 1.0,
        "c": 0.0,
        "d": 1.0,
        "m": 0.0,
    }


def _safe_exp_jax(x):
    return jnp.exp(jnp.clip(x, -20.0, 20.0))


def tersoff_cutoff_jax(r, r_cut, cutoff_thickness, alpha):
    r_inner = r_cut - cutoff_thickness
    x = (r - r_inner) / cutoff_thickness
    x3 = x * x * x
    denom = x3 - 1.0
    denom = jnp.where(jnp.abs(denom) < 1e-12, -1e-12, denom)
    shell_val = _safe_exp_jax((-alpha) * x3 / denom)

    f_c = jnp.where(r >= r_cut, 0.0, 1.0)
    mask_shell = (r > r_inner) & (r < r_cut)
    f_c = jnp.where(mask_shell, shell_val, f_c)
    return f_c


def tersoff_surrogate_curve_jax(r, vec, rik_distance, surrogate_cos_theta):
    a1 = vec[PARAM_INDEX["A1"]]
    a2 = vec[PARAM_INDEX["A2"]]
    lam1 = vec[PARAM_INDEX["lambda1"]]
    lam2 = vec[PARAM_INDEX["lambda2"]]
    dimer_r = vec[PARAM_INDEX["dimer_r"]]
    cutoff_thickness = vec[PARAM_INDEX["cutoff_thickness"]]
    r_cut = vec[PARAM_INDEX["r_cut"]]
    alpha = vec[PARAM_INDEX["alpha"]]
    n = vec[PARAM_INDEX["n"]]
    gamma = vec[PARAM_INDEX["gamma"]]

    f_c_ij = tersoff_cutoff_jax(r, r_cut=r_cut, cutoff_thickness=cutoff_thickness, alpha=alpha)
    f_r = a1 * _safe_exp_jax(lam1 * (dimer_r - r))
    f_a = a2 * _safe_exp_jax(lam2 * (dimer_r - r))

    rik = jnp.full_like(r, rik_distance)
    f_c_ik = tersoff_cutoff_jax(rik, r_cut=r_cut, cutoff_thickness=cutoff_thickness, alpha=alpha)

    del surrogate_cos_theta
    zeta = jnp.maximum(f_c_ik, 0.0)
    n_eff = jnp.maximum(n, 1e-6)
    gamma_n = gamma**n_eff
    zeta_safe = jnp.where(zeta > 0.0, zeta, 1.0)
    zeta_n = jnp.power(zeta_safe, n_eff)
    bij_raw = jnp.power(1.0 + gamma_n * zeta_n, -0.5 / n_eff)
    bij = jnp.where(zeta > 0.0, bij_raw, 1.0)

    return 0.5 * f_c_ij * (f_r - bij * f_a)


def objective_jax(
    vec,
    target_curves,
    third_distances,
    r_grid,
    surrogate_cos_theta,
):
    pred_curves = jax.vmap(
        lambda rik_distance: tersoff_surrogate_curve_jax(
            r_grid,
            vec,
            rik_distance,
            surrogate_cos_theta,
        )
    )(third_distances)

    diff = pred_curves - target_curves
    per_curve_mse = jnp.mean(diff * diff, axis=1)
    fit_loss = jnp.mean(per_curve_mse)
    return fit_loss, pred_curves


def run_adam_jax(
    init_vec: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_curves: np.ndarray,
    third_distances: np.ndarray,
    r_grid: np.ndarray,
    cfg: FitConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    x = jnp.asarray(np.clip(init_vec.copy(), lower, upper), dtype=jnp.float64)
    lower_j = jnp.asarray(lower, dtype=jnp.float64)
    upper_j = jnp.asarray(upper, dtype=jnp.float64)
    target_j = jnp.asarray(target_curves, dtype=jnp.float64)
    third_j = jnp.asarray(third_distances, dtype=jnp.float64)
    r_j = jnp.asarray(r_grid, dtype=jnp.float64)

    objective_bound = partial(
        objective_jax,
        target_curves=target_j,
        third_distances=third_j,
        r_grid=r_j,
        surrogate_cos_theta=float(cfg.surrogate_cos_theta),
    )

    loss_and_grad = jax.jit(jax.value_and_grad(lambda v: objective_bound(v)[0]))
    curves_from_vec = jax.jit(lambda v: objective_bound(v)[1])

    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)

    best_loss = math.inf
    best_x = np.asarray(x, dtype=np.float64)
    best_curve = None
    best_step = 0

    for step in range(1, cfg.adam_steps + 1):
        loss, grad = loss_and_grad(x)

        m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (grad * grad)

        m_hat = m / (1.0 - cfg.beta1**step)
        v_hat = v / (1.0 - cfg.beta2**step)

        x = x - cfg.adam_lr * m_hat / (jnp.sqrt(v_hat) + cfg.adam_eps)
        x = jnp.clip(x, lower_j, upper_j)

        loss_value = float(loss)
        if np.isfinite(loss_value) and loss_value < (best_loss - cfg.plateau_min_delta):
            best_loss = loss_value
            best_x = np.asarray(x, dtype=np.float64)
            best_curve = np.asarray(curves_from_vec(x), dtype=np.float64)
            best_step = step

        if (
            step >= cfg.plateau_min_steps
            and best_step > 0
            and (step - best_step) >= cfg.plateau_patience
        ):
            break

    if best_curve is None:
        loss_final, curves_final = objective_bound(x)
        best_loss = float(loss_final)
        best_x = np.asarray(x, dtype=np.float64)
        best_curve = np.asarray(curves_final, dtype=np.float64)

    return best_x, best_loss, best_curve


def write_fit_csv(path: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    fieldnames = [
        "reactive_epsilon",
        "loss",
        "rmse",
        "target_reactive_r_cut",
        "target_weakening_inner",
        "target_weakening_outer",
        "r_cut",
        "third_bead_min",
        "third_bead_max",
        "n_third_bead_distances",
        "A1",
        "A2",
        "lambda1",
        "lambda2",
        "lambda3",
        "dimer_r",
        "cutoff_thickness",
        "alpha",
        "n",
        "gamma",
        "c",
        "d",
        "m",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_curve_csv(
    path: str,
    r: np.ndarray,
    third_distances: np.ndarray,
    reactive: np.ndarray,
    tersoff: np.ndarray,
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["third_bead_distance", "r", "reactive_lj", "tersoff_fit"])
        for curve_idx, third_distance in enumerate(third_distances):
            rows = np.column_stack(
                [
                    np.full_like(r, fill_value=float(third_distance), dtype=np.float64),
                    r,
                    reactive[curve_idx],
                    tersoff[curve_idx],
                ]
            )
            writer.writerows(rows)


def make_colored_curve_plot(
    path: str,
    epsilon: float,
    r: np.ndarray,
    third_distances: np.ndarray,
    curves: np.ndarray,
    sigma: float,
    model_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 1.5), dpi=300)
    plot_r_min = 0.95 * sigma
    plot_r_max = 1.5 * sigma
    plot_mask = (r >= plot_r_min) & (r <= plot_r_max)
    if not np.any(plot_mask):
        raise ValueError(
            f"No points remain after filtering plot range to {plot_r_min:g} <= r <= {plot_r_max:g}."
        )
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

    ax.set_xlabel("r")
    ax.set_ylabel("U(r)")
    ax.set_ylim(-14.0, 14.0)
    ax.grid(alpha=0.25, linewidth=0.5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$\frac{r_3}{\sigma}$")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    plot_dir = os.path.abspath(os.path.join(script_dir, args.plot_dir))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    params_csv = args.params_csv
    if params_csv is None:
        params_csv = os.path.join(output_dir, "tersoff_fitted_params.csv")
    else:
        params_csv = os.path.abspath(params_csv)

    if args.n_third_bead_distances < 2:
        raise ValueError("--n-third-bead-distances must be >= 2")
    if args.adam_steps < 1:
        raise ValueError("--adam-steps must be >= 1")
    if args.plateau_min_steps < 0:
        raise ValueError("--plateau-min-steps must be >= 0")
    if args.plateau_patience < 1:
        raise ValueError("--plateau-patience must be >= 1")
    if args.plateau_min_delta < 0.0:
        raise ValueError("--plateau-min-delta must be >= 0")

    sigma = float(args.sigma)
    reactive_r_cut_sigma = REACTIVE_R_CUT_MULT
    weakening_inner_sigma = REACTIVE_WEAKENING_INNER_MULT
    weakening_outer_sigma = REACTIVE_R_CUT_MULT
    third_bead_min_sigma = REACTIVE_WEAKENING_INNER_MULT
    third_bead_max_sigma = REACTIVE_R_CUT_MULT

    reactive_r_cut = reactive_r_cut_sigma * sigma
    weakening_inner = weakening_inner_sigma * sigma
    weakening_outer = weakening_outer_sigma * sigma
    cutoff_thickness = reactive_r_cut - weakening_inner

    cfg = FitConfig(
        sigma=sigma,
        reactive_r_cut=reactive_r_cut,
        weakening_inner=weakening_inner,
        weakening_outer=weakening_outer,
        weakening_exponent=float(args.weakening_exponent),
        smooth_elbow=bool(args.smooth_elbow),
        smooth_kappa=float(args.smooth_kappa),
        smooth_beta=float(args.smooth_beta),
        third_bead_multiplier=float(args.third_bead_multiplier),
        n_points=int(args.n_points),
        adam_steps=int(args.adam_steps),
        adam_lr=float(args.adam_lr),
        plateau_min_steps=int(args.plateau_min_steps),
        plateau_patience=int(args.plateau_patience),
        plateau_min_delta=float(args.plateau_min_delta),
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        restarts=int(args.restarts),
        surrogate_cos_theta=float(args.surrogate_cos_theta),
    )

    print(
        "ReactiveLJ target (fixed) "
        f"r_cut/sigma={reactive_r_cut_sigma:g} "
        f"weakening_inner/sigma={weakening_inner_sigma:g} "
        f"weakening_outer/sigma={weakening_outer_sigma:g} "
        f"third_bead_range/sigma=[{third_bead_min_sigma:g}, {third_bead_max_sigma:g}]",
        flush=True,
    )
    print("Optimizer backend: jax-adam", flush=True)
    print(
        "Early stopping: "
        f"min_steps={cfg.plateau_min_steps} "
        f"patience={cfg.plateau_patience} "
        f"min_delta={cfg.plateau_min_delta:g} "
        f"max_steps={cfg.adam_steps}",
        flush=True,
    )
    print(
        f"Fitting and plotting range: r in [{0.95 * sigma:g}, {cfg.reactive_r_cut:g}]",
        flush=True,
    )

    r_grid = np.linspace(0.95 * sigma, cfg.reactive_r_cut, cfg.n_points)
    third_distances = compute_third_bead_distances(
        r_inner=third_bead_min_sigma * cfg.sigma,
        r_outer=third_bead_max_sigma * cfg.sigma,
        exponent=cfg.weakening_exponent,
        count=int(args.n_third_bead_distances),
    )

    rows: list[dict[str, float]] = []

    for epsilon in args.epsilons:
        epsilon = float(epsilon)
        target_curves = reactive_curve_set(
            r_grid,
            epsilon=epsilon,
            cfg=cfg,
            third_distances=third_distances,
        )
        reactive_plot_path = os.path.join(
            plot_dir, f"reactive_lj_curves_eps_{epsilon:g}.svg"
        )
        make_colored_curve_plot(
            path=reactive_plot_path,
            epsilon=epsilon,
            r=r_grid,
            third_distances=third_distances,
            curves=target_curves,
            sigma=cfg.sigma,
            model_label="ReactiveLJ",
        )
        print(
            f"epsilon={epsilon:g} wrote pre-fit ReactiveLJ curves: {reactive_plot_path}",
            flush=True,
        )

        default_params = build_default_params(
            epsilon=epsilon,
            sigma=cfg.sigma,
            reactive_r_cut=cfg.reactive_r_cut,
            cutoff_thickness=cutoff_thickness,
        )
        default_vec = params_to_vector(default_params)
        lower, upper = build_bounds(
            sigma=cfg.sigma,
            reactive_r_cut=cfg.reactive_r_cut,
            cutoff_thickness=cutoff_thickness,
        )

        best_vec = None
        best_loss = math.inf
        best_curve = None

        rng = np.random.default_rng(int(1000 * epsilon) + 17)
        for restart_idx in range(cfg.restarts):
            if restart_idx == 0:
                init_vec = default_vec.copy()
            else:
                noise = rng.normal(loc=0.0, scale=0.15, size=default_vec.shape)
                init_vec = np.clip(default_vec * (1.0 + noise), lower, upper)

            fit_vec, fit_loss, fit_curve = run_adam_jax(
                init_vec=init_vec,
                lower=lower,
                upper=upper,
                target_curves=target_curves,
                third_distances=third_distances,
                r_grid=r_grid,
                cfg=cfg,
            )

            if fit_loss < best_loss:
                best_loss = fit_loss
                best_vec = fit_vec
                best_curve = fit_curve

        assert best_vec is not None and best_curve is not None

        rmse = float(np.sqrt(np.mean(np.square(best_curve - target_curves))))
        fit_params = vector_to_params(best_vec)

        row = {
            "reactive_epsilon": epsilon,
            "loss": best_loss,
            "rmse": rmse,
            "target_reactive_r_cut": cfg.reactive_r_cut,
            "target_weakening_inner": cfg.weakening_inner,
            "target_weakening_outer": cfg.weakening_outer,
            "r_cut": fit_params["r_cut"],
            "third_bead_min": float(third_distances[0]),
            "third_bead_max": float(third_distances[-1]),
            "n_third_bead_distances": int(third_distances.size),
        }
        row.update(fit_params)
        rows.append(row)

        curve_csv = os.path.join(output_dir, f"curve_fit_eps_{epsilon:g}.csv")
        write_curve_csv(curve_csv, r_grid, third_distances, target_curves, best_curve)

        tersoff_plot_path = os.path.join(
            plot_dir, f"tersoff_curves_eps_{epsilon:g}.svg"
        )
        make_colored_curve_plot(
            path=tersoff_plot_path,
            epsilon=epsilon,
            r=r_grid,
            third_distances=third_distances,
            curves=best_curve,
            sigma=cfg.sigma,
            model_label="Liu/O'Connor Tersoff fit",
        )

        print(
            f"epsilon={epsilon:g} fit_loss={best_loss:.6e} rmse={rmse:.6e} "
            f"A1={fit_params['A1']:.4g} A2={fit_params['A2']:.4g} "
            f"tersoff_plot={tersoff_plot_path}",
            flush=True,
        )

    write_fit_csv(params_csv, rows)
    print(f"Wrote fitted parameters: {params_csv}", flush=True)
    print(f"Wrote curve plots: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
