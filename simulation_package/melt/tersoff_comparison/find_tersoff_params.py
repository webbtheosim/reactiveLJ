#!/usr/bin/env python3
"""Fit Liu/O'Connor Tersoff parameters to ReactiveLJ weakening targets."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from functools import partial

import numpy as np

import ultraplot as uplt

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


EPSILONS_DEFAULT = (6.0, 12.0, 15.0, 18.0)
REACTIVE_R_CUT_MULT = 1.5
REACTIVE_WEAKENING_DISTANCE_MULT = 0.2
REACTIVE_WEAKENING_INNER_MULT = REACTIVE_R_CUT_MULT - REACTIVE_WEAKENING_DISTANCE_MULT
WEAKENING_PLOT_FIGSIZE = (3.3, 4.6)
REACTIVE_COLOR = "#e77500"
TERSOFF_COLOR = "#121212"

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
            "Fit the Liu/O'Connor Tersoff subset to match ReactiveLJ "
            "minimum-energy weakening across fixed third-bead distances using "
            "Adam with JAX autodiff gradients."
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
        help="Directory for weakening comparison plots.",
    )
    parser.add_argument(
        "--params-csv",
        default=None,
        help="Output CSV path for fitted parameters (defaults to <output-dir>/tersoff_fitted_params.csv).",
    )
    return parser.parse_args()


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


def reference_distance_index(third_distances: np.ndarray) -> int:
    return int(np.argmax(third_distances))


def minimum_energies(curves: np.ndarray) -> np.ndarray:
    return np.min(curves, axis=1)


def minimum_point_weakening(
    curves: np.ndarray,
    reference_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    minima = minimum_energies(curves)
    return minima - minima[reference_index], minima


def compute_weakening_rmse(
    predicted: np.ndarray,
    target: np.ndarray,
    reference_index: int,
) -> float:
    mask = np.arange(target.size) != int(reference_index)
    return float(np.sqrt(np.mean(np.square(predicted[mask] - target[mask]))))


def compute_third_bead_distances(
    r_inner: float, r_outer: float, exponent: float, count: int
) -> np.ndarray:
    """Sample third-bead distances via uniform spacing in distance^exponent."""
    if count <= 1:
        return np.asarray([r_inner], dtype=np.float64)
    powered = np.linspace(r_outer**exponent, r_inner**exponent, count)
    return np.asarray(powered ** (1.0 / exponent), dtype=np.float64)


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


def minimum_point_weakening_jax(curves, reference_index):
    minima = jnp.min(curves, axis=1)
    return minima - minima[reference_index]


def objective_jax(
    vec,
    target_weakening,
    third_distances,
    reference_index,
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

    pred_weakening = minimum_point_weakening_jax(
        pred_curves,
        reference_index=reference_index,
    )
    diff = pred_weakening - target_weakening
    mask = jnp.arange(diff.size) != reference_index
    fit_loss = jnp.sum(jnp.where(mask, diff * diff, 0.0)) / jnp.sum(mask)
    return fit_loss, pred_curves


def run_adam_jax(
    init_vec: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_weakening: np.ndarray,
    third_distances: np.ndarray,
    reference_index: int,
    r_grid: np.ndarray,
    cfg: FitConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    x = jnp.asarray(np.clip(init_vec.copy(), lower, upper), dtype=jnp.float64)
    lower_j = jnp.asarray(lower, dtype=jnp.float64)
    upper_j = jnp.asarray(upper, dtype=jnp.float64)
    target_j = jnp.asarray(target_weakening, dtype=jnp.float64)
    third_j = jnp.asarray(third_distances, dtype=jnp.float64)
    r_j = jnp.asarray(r_grid, dtype=jnp.float64)

    objective_bound = partial(
        objective_jax,
        target_weakening=target_j,
        third_distances=third_j,
        reference_index=int(reference_index),
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
        "weakening_rmse",
        "target_reference_third_bead_distance",
        "target_reference_min_energy",
        "fit_reference_min_energy",
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


def write_weakening_csv(
    path: str,
    third_distances: np.ndarray,
    sigma: float,
    reactive_weakening: np.ndarray,
    tersoff_weakening: np.ndarray,
    reactive_minima: np.ndarray,
    tersoff_minima: np.ndarray,
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "third_bead_distance",
                "third_bead_distance_over_sigma",
                "reactive_lj_min_energy",
                "tersoff_fit_min_energy",
                "reactive_lj_weakening",
                "tersoff_fit_weakening",
            ]
        )
        for idx, third_distance in enumerate(third_distances):
            writer.writerow(
                [
                    float(third_distance),
                    float(third_distance / sigma),
                    float(reactive_minima[idx]),
                    float(tersoff_minima[idx]),
                    float(reactive_weakening[idx]),
                    float(tersoff_weakening[idx]),
                ]
            )


def write_average_weakening_csv(
    path: str,
    third_distances: np.ndarray,
    sigma: float,
    reactive_by_epsilon: np.ndarray,
    tersoff_by_epsilon: np.ndarray,
) -> None:
    reactive_mean = np.mean(reactive_by_epsilon, axis=0)
    tersoff_mean = np.mean(tersoff_by_epsilon, axis=0)
    reactive_std = np.std(reactive_by_epsilon, axis=0)
    tersoff_std = np.std(tersoff_by_epsilon, axis=0)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "third_bead_distance",
                "third_bead_distance_over_sigma",
                "reactive_lj_mean_weakening",
                "tersoff_fit_mean_weakening",
                "reactive_lj_std_weakening",
                "tersoff_fit_std_weakening",
            ]
        )
        for idx, third_distance in enumerate(third_distances):
            writer.writerow(
                [
                    float(third_distance),
                    float(third_distance / sigma),
                    float(reactive_mean[idx]),
                    float(tersoff_mean[idx]),
                    float(reactive_std[idx]),
                    float(tersoff_std[idx]),
                ]
            )


def make_per_epsilon_weakening_plot(
    path: str,
    epsilons: list[float],
    third_distances: np.ndarray,
    sigma: float,
    reactive_by_epsilon: np.ndarray,
    tersoff_by_epsilon: np.ndarray,
) -> None:
    n_panels = len(epsilons)
    if reactive_by_epsilon.shape[0] != n_panels or tersoff_by_epsilon.shape[0] != n_panels:
        raise ValueError("Per-epsilon weakening arrays must align with the epsilon list.")

    ncols = 1
    nrows = max(1, math.ceil(n_panels / ncols))
    fig, axes = uplt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=WEAKENING_PLOT_FIGSIZE,
        dpi=600,
        tight=False,
    )

    axes_flat = np.ravel(np.atleast_1d(axes))
    x = third_distances / sigma
    order = np.argsort(x)
    x_plot = x[order]
    reactive_plot = np.asarray(reactive_by_epsilon[:, order], dtype=np.float64)
    tersoff_plot = np.asarray(tersoff_by_epsilon[:, order], dtype=np.float64)
    combined = np.concatenate([reactive_plot.ravel(), tersoff_plot.ravel()])
    y_min = float(np.nanmin(combined))
    y_max = float(np.nanmax(combined))
    y_span = y_max - y_min
    if y_span <= 0.0:
        y_span = max(abs(y_max), 1.0)
    y_pad = 0.08 * y_span
    y_limits = (min(0.0, y_min - y_pad), y_max + y_pad)
    x_limits = (float(np.min(x_plot)), float(np.max(x_plot)))
    x_ticks = np.linspace(x_limits[0], x_limits[1], 5)

    for idx, (axis, epsilon) in enumerate(zip(axes_flat, epsilons, strict=True)):
        axis.plot(
            x_plot,
            reactive_plot[idx],
            color=REACTIVE_COLOR,
            linewidth=1.5,
            marker="o",
            markersize=3.2,
            label="ReactiveLJ",
            zorder=3,
        )
        axis.plot(
            x_plot,
            tersoff_plot[idx],
            color=TERSOFF_COLOR,
            linewidth=1.5,
            marker="s",
            markersize=3.0,
            label="L-O Tersoff",
            zorder=3,
        )
        axis.format(
            xlabel=r"$r_\mathrm{AC}/\sigma$" if idx == (n_panels - 1) else "",
            ylabel=r"$\Delta U_\mathrm{min}$",
            xlim=x_limits,
            ylim=y_limits,
            xspineloc="both",
            yspineloc="both",
            xtickloc="both",
            ytickloc="both",
            tickdir="in",
            grid=False,
        )
        axis.set_xticks(x_ticks)
        if idx == (n_panels - 1):
            axis.set_xticklabels([f"{tick:g}" for tick in x_ticks])
        else:
            axis.set_xticklabels([])
        axis.tick_params(axis="both", labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(10)
        axis.yaxis.label.set_rotation(90)
        axis.yaxis.label.set_horizontalalignment("center")
        axis.yaxis.label.set_verticalalignment("bottom")
        axis.set_title(rf"$\varepsilon_\mathrm{{RLJ}}={epsilon:g}$", fontsize=10)
        if idx == 0:
            axis.legend(fontsize=8, frameon=True, loc="best")

    for axis in axes_flat[n_panels:]:
        axis.set_axis_off()

    fig.savefig(path)
    uplt.close(fig)


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
    reference_index = reference_distance_index(third_distances)
    reference_distance = float(third_distances[reference_index])
    print(
        "Weakening objective: "
        "Delta U_min(C)=min[U(C)]-min[U(C_ref)] "
        f"with C_ref/sigma={reference_distance / cfg.sigma:g}",
        flush=True,
    )

    rows: list[dict[str, float]] = []
    reactive_weakening_by_epsilon: list[np.ndarray] = []
    tersoff_weakening_by_epsilon: list[np.ndarray] = []

    for epsilon in args.epsilons:
        epsilon = float(epsilon)
        target_curves = reactive_curve_set(
            r_grid,
            epsilon=epsilon,
            cfg=cfg,
            third_distances=third_distances,
        )
        target_weakening, target_minima = minimum_point_weakening(
            target_curves,
            reference_index=reference_index,
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
                target_weakening=target_weakening,
                third_distances=third_distances,
                reference_index=reference_index,
                r_grid=r_grid,
                cfg=cfg,
            )

            if fit_loss < best_loss:
                best_loss = fit_loss
                best_vec = fit_vec
                best_curve = fit_curve

        assert best_vec is not None and best_curve is not None

        best_weakening, best_minima = minimum_point_weakening(
            best_curve,
            reference_index=reference_index,
        )
        weakening_rmse = compute_weakening_rmse(
            predicted=best_weakening,
            target=target_weakening,
            reference_index=reference_index,
        )
        fit_params = vector_to_params(best_vec)

        row = {
            "reactive_epsilon": epsilon,
            "loss": best_loss,
            "weakening_rmse": weakening_rmse,
            "target_reference_third_bead_distance": reference_distance,
            "target_reference_min_energy": float(target_minima[reference_index]),
            "fit_reference_min_energy": float(best_minima[reference_index]),
            "target_reactive_r_cut": cfg.reactive_r_cut,
            "target_weakening_inner": cfg.weakening_inner,
            "target_weakening_outer": cfg.weakening_outer,
            "r_cut": fit_params["r_cut"],
            "third_bead_min": float(np.min(third_distances)),
            "third_bead_max": float(np.max(third_distances)),
            "n_third_bead_distances": int(third_distances.size),
        }
        row.update(fit_params)
        rows.append(row)

        weakening_csv = os.path.join(
            output_dir,
            f"weakening_fit_eps_{epsilon:g}.csv",
        )
        write_weakening_csv(
            path=weakening_csv,
            third_distances=third_distances,
            sigma=cfg.sigma,
            reactive_weakening=target_weakening,
            tersoff_weakening=best_weakening,
            reactive_minima=target_minima,
            tersoff_minima=best_minima,
        )
        reactive_weakening_by_epsilon.append(target_weakening)
        tersoff_weakening_by_epsilon.append(best_weakening)

        print(
            f"epsilon={epsilon:g} fit_loss={best_loss:.6e} "
            f"weakening_rmse={weakening_rmse:.6e} "
            f"A1={fit_params['A1']:.4g} A2={fit_params['A2']:.4g} "
            f"weakening_csv={weakening_csv}",
            flush=True,
        )

    reactive_weakening_array = np.asarray(reactive_weakening_by_epsilon, dtype=np.float64)
    tersoff_weakening_array = np.asarray(tersoff_weakening_by_epsilon, dtype=np.float64)
    average_weakening_csv = os.path.join(output_dir, "average_weakening.csv")
    write_average_weakening_csv(
        path=average_weakening_csv,
        third_distances=third_distances,
        sigma=cfg.sigma,
        reactive_by_epsilon=reactive_weakening_array,
        tersoff_by_epsilon=tersoff_weakening_array,
    )
    average_weakening_plot = os.path.join(
        plot_dir,
        "average_weakening_vs_third_bead_distance.svg",
    )
    make_per_epsilon_weakening_plot(
        path=average_weakening_plot,
        epsilons=[float(epsilon) for epsilon in args.epsilons],
        third_distances=third_distances,
        sigma=cfg.sigma,
        reactive_by_epsilon=reactive_weakening_array,
        tersoff_by_epsilon=tersoff_weakening_array,
    )

    write_fit_csv(params_csv, rows)
    print(f"Wrote fitted parameters: {params_csv}", flush=True)
    print(f"Wrote weakening table: {average_weakening_csv}", flush=True)
    print(f"Wrote weakening plot: {average_weakening_plot}", flush=True)


if __name__ == "__main__":
    main()
