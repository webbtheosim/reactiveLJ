"""Shared analysis utilities for ReactiveLJ data processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class CorrelationAccumulator:
    """Accumulate correlations using a rolling buffer of sets."""
    max_lag: int

    def __post_init__(self) -> None:
        self.numerators = np.zeros(self.max_lag, dtype=np.float64)
        self.denominators = np.zeros(self.max_lag, dtype=np.float64)
        self._buffer: List[set] = []

    def update(self, current_set: set) -> None:
        # Compare with previous sets in reverse (lag 1 = most recent).
        for lag, prev_set in enumerate(reversed(self._buffer), start=1):
            if lag > self.max_lag:
                break
            idx = lag - 1
            if prev_set:
                self.numerators[idx] += len(prev_set & current_set)
                self.denominators[idx] += len(prev_set)

        # Push current set into the buffer
        self._buffer.append(current_set)
        if len(self._buffer) > self.max_lag:
            self._buffer.pop(0)

    def correlation(self) -> np.ndarray:
        corr = np.zeros_like(self.numerators)
        nonzero = self.denominators > 0
        corr[nonzero] = self.numerators[nonzero] / self.denominators[nonzero]
        return corr


def compute_r_thresh(sigma: float = 1.0) -> float:
    """Inflection point of the LJ potential in the two-body limit."""
    return sigma * (26.0 / 7.0) ** (1.0 / 6.0)


def minimum_image(dx: np.ndarray, box_length: float) -> np.ndarray:
    """Apply minimum image convention for a cubic box."""
    return dx - box_length * np.round(dx / box_length)


def reactive_weight(distance: float, inner: float, outer: float) -> float:
    """Coordination weight w(r) for ReactiveLJ (1 -> 0 cosine taper)."""
    if distance >= outer:
        return 0.0
    if distance <= inner:
        return 1.0
    angle = np.pi * (distance - inner) / (outer - inner)
    return 0.5 * (1.0 + np.cos(angle))


def build_cell_list(
    positions: np.ndarray, box_length: float, cutoff: float
) -> Tuple[List[List[int]], int]:
    """Build a simple cubic cell list for neighbor searching."""
    n_cells = max(1, int(box_length / cutoff))
    cell_size = box_length / n_cells

    frac = (positions + 0.5 * box_length) / cell_size
    coords = np.floor(frac).astype(np.int32) % n_cells
    flat = coords[:, 0] + n_cells * (coords[:, 1] + n_cells * coords[:, 2])

    cell_particles: List[List[int]] = [[] for _ in range(n_cells ** 3)]
    for idx, cell in enumerate(flat):
        cell_particles[cell].append(idx)

    return cell_particles, n_cells


def iter_neighbor_cells(cell_index: int, n_cells: int) -> Iterable[int]:
    """Yield neighbor cell indices (including self) for a given cell index."""
    cx = cell_index % n_cells
    cy = (cell_index // n_cells) % n_cells
    cz = cell_index // (n_cells * n_cells)

    for dx in (-1, 0, 1):
        nx = (cx + dx) % n_cells
        for dy in (-1, 0, 1):
            ny = (cy + dy) % n_cells
            for dz in (-1, 0, 1):
                nz = (cz + dz) % n_cells
                yield nx + n_cells * (ny + n_cells * nz)


def find_sticker_neighbor_pairs(
    positions: np.ndarray,
    sticker_ids: np.ndarray,
    box_length: float,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local sticker-index pairs and distances within ``cutoff``."""
    sticker_ids_arr = np.asarray(sticker_ids, dtype=np.int32)
    empty_idx = np.empty((0,), dtype=np.int32)
    empty_dist = np.empty((0,), dtype=np.float64)
    if sticker_ids_arr.size < 2 or cutoff <= 0.0:
        return empty_idx, empty_idx.copy(), empty_dist

    sticker_positions = positions[sticker_ids_arr]
    cell_particles, n_cells = build_cell_list(sticker_positions, box_length, cutoff)
    cutoff_sq = cutoff * cutoff

    pair_i: List[int] = []
    pair_j: List[int] = []
    pair_dist: List[float] = []

    for cell_index, particle_list in enumerate(cell_particles):
        if not particle_list:
            continue

        for neighbor_cell in iter_neighbor_cells(cell_index, n_cells):
            if neighbor_cell < cell_index:
                continue
            neighbor_list = cell_particles[neighbor_cell]
            if not neighbor_list:
                continue

            for i_idx in particle_list:
                for j_idx in neighbor_list:
                    if neighbor_cell == cell_index and j_idx <= i_idx:
                        continue

                    dx = sticker_positions[i_idx] - sticker_positions[j_idx]
                    dx = minimum_image(dx, box_length)
                    dist_sq = float(np.dot(dx, dx))
                    if dist_sq < cutoff_sq:
                        pair_i.append(i_idx)
                        pair_j.append(j_idx)
                        pair_dist.append(np.sqrt(dist_sq))

    if not pair_i:
        return empty_idx, empty_idx.copy(), empty_dist

    return (
        np.asarray(pair_i, dtype=np.int32),
        np.asarray(pair_j, dtype=np.int32),
        np.asarray(pair_dist, dtype=np.float64),
    )


def find_sticker_bonds(
    positions: np.ndarray,
    sticker_ids: np.ndarray,
    box_length: float,
    cutoff: float,
) -> set:
    """Identify sticker-sticker bonds based on a distance threshold."""
    pair_i, pair_j, _ = find_sticker_neighbor_pairs(
        positions, sticker_ids, box_length, cutoff
    )
    bonds: set = set()
    sticker_ids_arr = np.asarray(sticker_ids, dtype=np.int32)
    for i_idx, j_idx in zip(pair_i, pair_j):
        i_global = int(sticker_ids_arr[i_idx])
        j_global = int(sticker_ids_arr[j_idx])
        bonds.add(
            (i_global, j_global) if i_global < j_global else (j_global, i_global)
        )

    return bonds


class UnionFind:
    """Union-find data structure for cluster analysis."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]

    def cluster_sizes(self) -> np.ndarray:
        roots = np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int32)
        unique, counts = np.unique(roots, return_counts=True)
        return counts


def autocorr_fft(
    series: np.ndarray,
    subtract_mean: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Compute autocorrelation using FFT.

    When normalize is True, output is normalized to C(0)=1.
    When normalize is False, output is the unbiased autocovariance.
    """
    x = np.asarray(series, dtype=np.float64)
    if subtract_mean:
        x = x - np.mean(x)
    n = len(x)
    if n == 0:
        return np.array([])

    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = x

    fft = np.fft.rfft(padded)
    acf = np.fft.irfft(fft * np.conjugate(fft))[:n]

    # Unbiased normalization
    norm = np.arange(n, 0, -1, dtype=np.float64)
    acf = acf / norm

    if normalize and acf[0] != 0:
        acf = acf / acf[0]
    return acf


def multitau_autocovariance(
    series: np.ndarray, p: int = 16, m: int = 2, S: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute multi-tau autocovariance in batch on a logarithmic lag grid.

    The returned lag grid matches ``MultiTauCorrelator.result()`` for the same
    ``p``, ``m``, and ``S`` while avoiding per-sample Python updates.
    """
    if p % m != 0:
        raise ValueError("p must be divisible by m")

    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be 1D")
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    p_m = p // m
    lags: List[float] = []
    cov: List[float] = []

    level_data = x.copy()
    lag_scale = 1

    for level in range(S):
        n_level = level_data.size
        if n_level == 0:
            break

        j_start = 0 if level == 0 else p_m
        j_stop = min(p, n_level)
        for j in range(j_start, j_stop):
            span = n_level - j
            cov_ij = float(np.dot(level_data[:span], level_data[j:]) / span)
            lags.append(float(j * lag_scale))
            cov.append(cov_ij)

        if level == S - 1:
            break

        n_next = n_level // m
        if n_next == 0:
            break
        trimmed = level_data[: n_next * m]
        level_data = np.mean(trimmed.reshape(n_next, m), axis=1)
        lag_scale *= m

    return np.asarray(lags, dtype=np.float64), np.asarray(cov, dtype=np.float64)


class MultiTauCorrelator:
    """Multi-tau correlator for stress autocorrelation (Ramirez et al. 2010).

    Logarithmic time binning gives good statistics at all lag times.
    Streaming interface: feed samples one at a time via ``add()``.
    """

    def __init__(self, p: int = 16, m: int = 2, S: int = 40) -> None:
        if p % m != 0:
            raise ValueError("p must be divisible by m")
        self.p = p
        self.m = m
        self.S = S
        self.p_m = p // m
        self._sentinel = -1.0e30

        self.D = np.full((S, p), self._sentinel, dtype=np.float64)
        self.C = np.zeros((S, p), dtype=np.float64)
        self.N = np.zeros((S, p), dtype=np.int64)
        self.A = np.zeros(S, dtype=np.float64)
        self.M = np.zeros(S, dtype=np.int64)
        self.n_samples = 0

    def add(self, value: float) -> None:
        self.n_samples += 1
        self._add_level(value, 0)

    def _add_level(self, w: float, k: int) -> None:
        if k >= self.S:
            return
        p = self.p
        D = self.D
        C = self.C
        N = self.N

        # Shift register
        D[k, 1:] = D[k, :-1]
        D[k, 0] = w

        # Accumulate correlation products
        if k == 0:
            for j in range(p):
                if D[k, j] > self._sentinel:
                    C[k, j] += D[k, 0] * D[k, j]
                    N[k, j] += 1
        else:
            for j in range(self.p_m, p):
                if D[k, j] > self._sentinel:
                    C[k, j] += D[k, 0] * D[k, j]
                    N[k, j] += 1

        # Block average and propagate to next level
        self.A[k] += w
        self.M[k] += 1
        if self.M[k] == self.m:
            self._add_level(self.A[k] / self.m, k + 1)
            self.A[k] = 0.0
            self.M[k] = 0

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lag_indices, autocovariance) on a logarithmic grid.

        lag_indices are in units of the sampling interval.
        autocovariance is unnormalized: <x(0) x(t)>.
        """
        lags: List[int] = []
        corr: List[float] = []

        for j in range(self.p):
            if self.N[0, j] > 0:
                lags.append(j)
                corr.append(self.C[0, j] / self.N[0, j])

        for s in range(1, self.S):
            for j in range(self.p_m, self.p):
                if self.N[s, j] > 0:
                    lags.append(j * self.m**s)
                    corr.append(self.C[s, j] / self.N[s, j])

        return np.asarray(lags, dtype=np.float64), np.asarray(corr, dtype=np.float64)


def _exp_decay(time: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-time / tau)


def fit_exponential(
    time: np.ndarray,
    corr: np.ndarray,
    min_corr: float = 0.1,
    maxfev: int = 100_000,
) -> float:
    """Fit corr ~ exp(-t/tau) with robust fallbacks for fast decays."""
    n = min(len(time), len(corr))
    if n < 2:
        return float("nan")

    t = np.asarray(time[:n], dtype=np.float64)
    c = np.asarray(corr[:n], dtype=np.float64)
    finite_positive = np.isfinite(t) & np.isfinite(c) & (c > 0.0)
    if np.count_nonzero(finite_positive) < 2:
        return float("nan")

    threshold_candidates = [min_corr, 0.05, 0.02, 0.01, 0.005, 0.001]
    for threshold in threshold_candidates:
        mask = finite_positive & (c > threshold)
        if np.count_nonzero(mask) < 2:
            continue

        t_fit = t[mask]
        c_fit = c[mask]

        # Seed tau from a log-linear slope when possible.
        slope, _ = np.polyfit(t_fit, np.log(c_fit), 1)
        tau0 = -1.0 / slope if np.isfinite(slope) and slope < 0.0 else max(t_fit[0], 1.0)
        tau0 = max(tau0, 1e-12)

        try:
            params, _ = curve_fit(
                _exp_decay,
                t_fit,
                c_fit,
                p0=(tau0,),
                bounds=(1e-12, np.inf),
                maxfev=maxfev,
            )
            tau = float(params[0])
            if np.isfinite(tau) and tau > 0.0:
                return tau
        except (RuntimeError, ValueError):
            pass

        # Fallback to log-linear fit if nonlinear fit did not converge.
        if np.isfinite(slope) and slope < 0.0:
            return -1.0 / slope

    return float("nan")


def fit_plateau_exponential(
    time: np.ndarray,
    corr: np.ndarray,
    plateau_fraction: float = 0.2,
) -> Tuple[float, float]:
    """Fit corr ~ A + (1-A) exp(-t/tau) with A from the long-time plateau."""
    finite = np.isfinite(corr)
    if not np.any(finite):
        return float("nan"), float("nan")

    n_total = int(np.count_nonzero(finite))
    n_plateau = max(3, int(n_total * plateau_fraction))
    plateau_slice = corr[finite][-n_plateau:]
    plateau = float(np.median(plateau_slice))

    if not np.isfinite(plateau) or plateau >= 1.0:
        return plateau, float("nan")

    denom = 1.0 - plateau
    if denom <= 0.0:
        return plateau, float("nan")

    scaled = (corr - plateau) / denom
    mask = (scaled > 0.0) & np.isfinite(scaled)
    if np.count_nonzero(mask) < 2:
        return plateau, float("nan")

    slope, _ = np.polyfit(time[mask], np.log(scaled[mask]), 1)
    if not np.isfinite(slope) or slope >= 0.0:
        return plateau, float("nan")

    tau = -1.0 / slope
    return plateau, tau
