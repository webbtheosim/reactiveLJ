"""Shared analysis utilities for ReactiveLJ data processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import freud
import numba
import numpy as np
from scipy.optimize import curve_fit


@numba.njit(cache=False)
def _intersection_size_sorted(a: np.ndarray, b: np.ndarray) -> int:
    """Count |a ∩ b| exactly for sorted unique int64 arrays."""
    i = 0
    j = 0
    count = 0
    while i < a.size and j < b.size:
        a_i = a[i]
        b_j = b[j]
        if a_i == b_j:
            count += 1
            i += 1
            j += 1
        elif a_i < b_j:
            i += 1
        else:
            j += 1
    return count


@dataclass
class MultiTauSetCorrelationAccumulator:
    """Accumulate exact set overlaps on a logarithmic multi-tau lag grid."""
    max_lag: int
    p: int = 16
    m: int = 2
    S: int = 40

    def __post_init__(self) -> None:
        self.lag_indices = multitau_positive_lag_indices(
            self.max_lag,
            p=self.p,
            m=self.m,
            S=self.S,
        )
        self.numerators = np.zeros(self.lag_indices.size, dtype=np.float64)
        self.denominators = np.zeros(self.lag_indices.size, dtype=np.float64)
        self._buffer: List[np.ndarray | None] = [None] * max(self.max_lag, 1)
        self._cursor = 0
        self._filled = 0

    def update(self, current_ids: np.ndarray) -> None:
        current = np.asarray(current_ids, dtype=np.int64)
        for idx, lag in enumerate(self.lag_indices):
            if lag > self._filled:
                break
            prev_ids = self._buffer[(self._cursor - lag) % len(self._buffer)]
            if prev_ids is None:
                continue
            if prev_ids.size > 0:
                self.numerators[idx] += _intersection_size_sorted(prev_ids, current)
                self.denominators[idx] += prev_ids.size

        self._buffer[self._cursor] = current
        self._cursor = (self._cursor + 1) % len(self._buffer)
        self._filled = min(self._filled + 1, self.max_lag)

    def correlation(self) -> np.ndarray:
        corr = np.zeros_like(self.numerators)
        nonzero = self.denominators > 0
        corr[nonzero] = self.numerators[nonzero] / self.denominators[nonzero]
        return corr

    def valid_length(self) -> int:
        populated = self.denominators > 0
        if not np.any(populated):
            return 0
        return int(np.max(np.flatnonzero(populated))) + 1


def multitau_positive_lag_indices(
    max_lag: int,
    p: int = 16,
    m: int = 2,
    S: int = 40,
) -> np.ndarray:
    """Return strictly positive lag indices on a multi-tau grid."""
    if max_lag < 1:
        return np.empty((0,), dtype=np.int64)
    if p % m != 0:
        raise ValueError("p must be divisible by m")

    p_m = p // m
    lag_scale = 1
    lags: List[int] = []
    for level in range(S):
        j_start = 1 if level == 0 else p_m
        j_stop = min(p, max_lag // lag_scale + 1)
        if j_start >= j_stop:
            break
        for j in range(j_start, j_stop):
            lags.append(j * lag_scale)
        lag_scale *= m
        if lag_scale > max_lag:
            break
    return np.asarray(lags, dtype=np.int64)


def compute_r_thresh(sigma: float = 1.0) -> float:
    """Inflection point of the LJ potential in the two-body limit."""
    return sigma * (26.0 / 7.0) ** (1.0 / 6.0)


def reactive_weight(distance: float, inner: float, outer: float) -> float:
    """Coordination weight w(r) for ReactiveLJ (1 -> 0 cosine taper)."""
    if distance >= outer:
        return 0.0
    if distance <= inner:
        return 1.0
    angle = np.pi * (distance - inner) / (outer - inner)
    return 0.5 * (1.0 + np.cos(angle))


def find_sticker_neighbor_pairs(
    positions: np.ndarray,
    box_length: float,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local sticker-index pairs and distances within ``cutoff``."""
    empty_idx = np.empty((0,), dtype=np.int32)
    empty_dist = np.empty((0,), dtype=np.float64)
    positions_arr = np.asarray(positions, dtype=np.float32)
    if positions_arr.shape[0] < 2 or cutoff <= 0.0:
        return empty_idx, empty_idx.copy(), empty_dist

    query = freud.locality.AABBQuery(freud.box.Box.cube(box_length), positions_arr)
    nlist = query.query(
        positions_arr,
        dict(mode="ball", r_max=float(cutoff), exclude_ii=True),
    ).toNeighborList()

    mask = nlist.query_point_indices < nlist.point_indices
    if not np.any(mask):
        return empty_idx, empty_idx.copy(), empty_dist

    return (
        np.asarray(nlist.query_point_indices[mask], dtype=np.int32),
        np.asarray(nlist.point_indices[mask], dtype=np.int32),
        np.asarray(nlist.distances[mask], dtype=np.float64),
    )


def find_sticker_bonds(
    positions: np.ndarray,
    box_length: float,
    cutoff: float,
) -> set:
    """Identify sticker-sticker bonds based on a distance threshold."""
    pair_i, pair_j, _ = find_sticker_neighbor_pairs(positions, box_length, cutoff)
    bonds: set = set()
    for i_idx, j_idx in zip(pair_i, pair_j):
        i_global = int(i_idx)
        j_global = int(j_idx)
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
        root_mask = self.parent == np.arange(self.parent.size, dtype=np.int32)
        return self.size[root_mask]


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


def _fit_exponential_positive_points(
    time: np.ndarray,
    corr: np.ndarray,
    maxfev: int = 100_000,
) -> float:
    """Fit corr ~ exp(-t/tau) using all finite positive points provided."""
    n = min(len(time), len(corr))
    if n < 2:
        return float("nan")

    t = np.asarray(time[:n], dtype=np.float64)
    c = np.asarray(corr[:n], dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(c) & (c > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    t_fit = t[mask]
    c_fit = c[mask]
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

    if np.isfinite(slope) and slope < 0.0:
        return -1.0 / slope
    return float("nan")


def extract_semilog_linear_region(
    time: np.ndarray,
    corr: np.ndarray,
    min_points: int = 4,
    max_log_deviation: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the longest initial prefix that stays approximately linear on a semilog plot."""
    n = min(len(time), len(corr))
    if n < 2:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    t = np.asarray(time[:n], dtype=np.float64)
    c = np.asarray(corr[:n], dtype=np.float64)
    finite_positive = np.isfinite(t) & np.isfinite(c) & (c > 0.0)
    if np.count_nonzero(finite_positive) < 2:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    t = t[finite_positive]
    c = c[finite_positive]
    if t.size <= 2:
        return t, c

    fit_count = min(max(2, int(min_points)), t.size)
    log_c = np.log(c)
    prefix_stop = fit_count
    for stop in range(fit_count, t.size + 1):
        slope, intercept = np.polyfit(t[:stop], log_c[:stop], 1)
        fitted_log_c = slope * t[:stop] + intercept
        if np.max(np.abs(log_c[:stop] - fitted_log_c)) <= max_log_deviation:
            prefix_stop = stop
        else:
            break

    return t[:prefix_stop], c[:prefix_stop]


def fit_exponential(
    time: np.ndarray,
    corr: np.ndarray,
    min_corr: float | None = 0.1,
    maxfev: int = 100_000,
) -> float:
    """Fit corr ~ exp(-t/tau), optionally excluding low-correlation points."""
    n = min(len(time), len(corr))
    if n < 2:
        return float("nan")

    t = np.asarray(time[:n], dtype=np.float64)
    c = np.asarray(corr[:n], dtype=np.float64)
    finite_positive = np.isfinite(t) & np.isfinite(c) & (c > 0.0)
    if np.count_nonzero(finite_positive) < 2:
        return float("nan")

    if min_corr is None or min_corr <= 0.0:
        return _fit_exponential_positive_points(t, c, maxfev=maxfev)

    threshold_candidates = [min_corr, 0.05, 0.02, 0.01, 0.005, 0.001]
    for threshold in threshold_candidates:
        mask = finite_positive & (c > threshold)
        if np.count_nonzero(mask) < 2:
            continue

        tau = _fit_exponential_positive_points(t[mask], c[mask], maxfev=maxfev)
        if np.isfinite(tau):
            return tau

    return float("nan")


def fit_exponential_semilog_linear_region(
    time: np.ndarray,
    corr: np.ndarray,
    min_points: int = 4,
    max_log_deviation: float = 0.15,
    maxfev: int = 100_000,
) -> float:
    """Fit an exponential using only the initial semilog-linear portion of the correlation."""
    t_fit, c_fit = extract_semilog_linear_region(
        time,
        corr,
        min_points=min_points,
        max_log_deviation=max_log_deviation,
    )
    if t_fit.size < 2:
        return float("nan")
    return fit_exponential(t_fit, c_fit, min_corr=None, maxfev=maxfev)


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
