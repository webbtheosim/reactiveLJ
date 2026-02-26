"""Analyze ReactiveLJ trajectories (Block 2 metrics).

This script loops over GSD files produced by data_generation, computes the
analysis metrics described in agents.md, and averages results over replicates
for each ReactiveLJ attraction strength.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import gsd.hoomd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports resolve when running from repo root.
sys.path.append(os.path.dirname(__file__))

from analysis_utils import (
    CorrelationAccumulator,
    UnionFind,
    autocorr_fft,
    compute_r_thresh,
    find_sticker_bonds,
    fit_exponential,
    fit_plateau_exponential,
)


_BOND_TAU_FIT_WINDOWS = {
    3.0: (0.0, 400.0),
    6.0: (0.0, 600.0),
    9.0: (0.0, 1500.0),
    12.0: (0.0, 2500.0),
    15.0: (0.0, 2500.0),
    18.0: (0.0, 2500.0),
}

_OPEN_CORR_FIT_WINDOWS = {
    9.0: (0.0, 200.0),
    12.0: (0.0, 1000.0),
    15.0: (0.0, 2500.0),
    18.0: (0.0, 2500.0),
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ReactiveLJ trajectories.")
    parser.add_argument(
        "--input-root",
        default="../data_generation/outputs",
        help="Root directory containing eps_*/rep_*/trajectory.gsd",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for analysis results.",
    )
    parser.add_argument(
        "--analysis-stride",
        type=int,
        default=1,
        help="Stride for frames in analysis (1 = use every frame).",
    )
    parser.add_argument(
        "--max-lag-frames",
        type=int,
        default=25,
        help="Maximum lag (in frames) used for correlation functions.",
    )
    parser.add_argument(
        "--msd-sample",
        type=int,
        default=2000,
        help="Number of particles to sample for MSD (0 means all particles).",
    )
    parser.add_argument(
        "--msd-max-lag-frames",
        type=int,
        default=50,
        help="Maximum lag (in frames) used for MSD calculation.",
    )
    return parser.parse_args()


def discover_runs(input_root: str) -> List[Tuple[str, str]]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ"}
    runs = []
    for root, dirs, files in os.walk(input_root, topdown=True):
        # Keep test trajectories out of production analysis sweeps.
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        if "trajectory.gsd" in files:
            gsd_path = os.path.join(root, "trajectory.gsd")
            metadata_path = os.path.join(root, "metadata.json")
            runs.append((gsd_path, metadata_path))
    return runs


def get_bond_tau_fit_window(epsilon: float) -> Tuple[float, float] | None:
    for key, window in _BOND_TAU_FIT_WINDOWS.items():
        if abs(epsilon - key) < 1e-8:
            return window
    return None


def get_open_corr_fit_window(epsilon: float) -> Tuple[float, float] | None:
    for key, window in _OPEN_CORR_FIT_WINDOWS.items():
        if abs(epsilon - key) < 1e-8:
            return window
    return None


def should_compute_open_corr(epsilon: float) -> bool:
    return epsilon > 6.0


def fit_exponential_with_time_window(
    time: np.ndarray,
    corr: np.ndarray,
    fit_window: Tuple[float, float] | None,
) -> float:
    if fit_window is None:
        return fit_exponential(time, corr)
    t_min, t_max = fit_window
    mask = np.isfinite(time) & (time >= t_min) & (time <= t_max)
    if np.count_nonzero(mask) < 2:
        return fit_exponential(time, corr)
    tau = fit_exponential(time[mask], corr[mask])
    if np.isfinite(tau):
        return tau
    return fit_exponential(time, corr)


def find_pressure_key(frame) -> str | None:
    if not hasattr(frame, "log"):
        return None
    for key in frame.log.keys():
        if "pressure_tensor" in key:
            return key
    return None


def analyze_replicate(
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    msd_sample: int,
    msd_max_lag_frames: int,
    progress_label: str | None = None,
) -> Dict:
    with gsd.hoomd.open(gsd_path, "r") as traj:
        n_frames = len(traj)
        if n_frames == 0:
            raise RuntimeError(f"No frames found in {gsd_path}")
        n_analyzed = (n_frames + analysis_stride - 1) // analysis_stride
        progress_interval = max(1, n_analyzed // 10)

        first = traj[0]
        type_names = first.particles.types
        if "sticky" not in type_names:
            raise RuntimeError("Sticker type 'sticky' not found in trajectory.")
        sticker_type = type_names.index("sticky")

        n_polymer_particles = int(metadata.get("n_particles", first.particles.N))
        if n_polymer_particles <= 0:
            raise RuntimeError(
                f"metadata n_particles must be positive, got {n_polymer_particles}"
            )
        if n_polymer_particles > first.particles.N:
            raise RuntimeError(
                "metadata n_particles exceeds trajectory particle count "
                f"({n_polymer_particles} > {first.particles.N})"
            )

        typeid = first.particles.typeid[:n_polymer_particles]
        sticker_ids = np.where(typeid == sticker_type)[0]
        chain_length = int(metadata.get("chain_length", 1))
        n_chains = int(metadata.get("n_chains", n_polymer_particles // chain_length))
        # Chains are laid out sequentially in the snapshot: tag -> chain id.
        chain_ids = np.arange(n_polymer_particles, dtype=np.int32) // chain_length
        n_stickers = len(sticker_ids)
        sticker_idx_map = np.full(n_polymer_particles, -1, dtype=np.int32)
        sticker_idx_map[sticker_ids] = np.arange(n_stickers, dtype=np.int32)

        box_length = float(first.configuration.box[0])
        r_thresh = compute_r_thresh(metadata.get("reactive_sigma", 1.0))
        reactive_sigma = float(metadata.get("reactive_sigma", 1.0))
        r_cut = metadata.get("reactive_r_cut")
        if r_cut is None:
            r_cut = 1.5 * reactive_sigma
        weakening_inner = metadata.get("weakening_inner")
        if weakening_inner is None:
            weakening_inner = 1.3 * reactive_sigma
        weakening_outer = metadata.get("weakening_outer")
        if weakening_outer is None:
            weakening_outer = 1.5 * reactive_sigma

        dt = metadata.get("dt", 0.005)
        frame_steps = metadata.get("frame_steps", 10_000)
        frame_dt = dt * frame_steps * analysis_stride
        reactive_epsilon = float(metadata.get("reactive_epsilon", float("nan")))
        compute_open_corr = should_compute_open_corr(reactive_epsilon)

        # Correlation accumulators
        bond_corr = CorrelationAccumulator(max_lag_frames)
        open_corr = CorrelationAccumulator(max_lag_frames) if compute_open_corr else None

        # Per-frame time series
        p_open_series: List[float] = []
        p_series: List[float] = []
        epsilon_series: List[float] = []
        frac_bond0_series: List[float] = []
        frac_bond1_series: List[float] = []
        frac_bond_gt1_series: List[float] = []
        cexc_series: List[float] = []

        # Cluster distribution accumulators
        cluster_hist = np.zeros(n_chains + 1, dtype=np.float64)
        cluster_count_total = 0
        largest_cluster_fraction_sum = 0.0
        mean_cluster_size_sum = 0.0
        cluster_frames = 0

        # Bond stats
        intra_bond_total = 0
        inter_bond_total = 0
        bond_frames = 0

        # Exchange rates
        rate_assoc_sum = 0.0
        rate_dissoc_sum = 0.0
        rate_count = 0

        # Pressure tensor series for G(t)
        pressure_key = find_pressure_key(first)
        pressure_series: List[Tuple[float, float, float]] = []

        # MSD sampling
        n_particles = n_polymer_particles
        if msd_sample == 0 or msd_sample >= n_particles:
            sample_ids = np.arange(n_particles)
        else:
            rng = np.random.default_rng(12345)
            sample_ids = rng.choice(n_particles, size=msd_sample, replace=False)

        msd_positions: List[np.ndarray] = []

        prev_bonds: set | None = None
        prev_open: set | None = None
        prev_partners: Dict[int, set] | None = None

        for analyzed_idx, frame_idx in enumerate(
            range(0, n_frames, analysis_stride), start=1
        ):
            frame = traj[frame_idx]
            positions = frame.particles.position[:n_polymer_particles]
            images = getattr(frame.particles, "image", None)
            if images is not None:
                images = images[:n_polymer_particles]

            # Build bond network for stickers
            bonds = find_sticker_bonds(positions, sticker_ids, box_length, r_thresh)

            # Mean C_exc for reactive pairs in this frame
            if n_stickers > 1:
                sticker_positions = positions[sticker_ids]
                cexc_series.append(
                    compute_cexc_mean(
                        sticker_positions,
                        box_length,
                        float(r_cut),
                        float(weakening_inner),
                        float(weakening_outer),
                    )
                )

            bonded_stickers = set()
            for i, j in bonds:
                bonded_stickers.add(i)
                bonded_stickers.add(j)

            open_stickers = set(sticker_ids.tolist()) - bonded_stickers

            p_open = len(open_stickers) / n_stickers
            p = 1.0 - p_open
            p_open_series.append(p_open)
            p_series.append(p)

            f = metadata.get("stickers_per_chain", 4)
            p_c = 1.0 / (f - 1)
            epsilon_val = (p - p_c) / p_c
            epsilon_series.append(epsilon_val)

            # Sticker degree fractions
            if n_stickers > 0:
                degrees = np.zeros(n_stickers, dtype=np.int32)
                for i, j in bonds:
                    i_idx = sticker_idx_map[i]
                    j_idx = sticker_idx_map[j]
                    if i_idx >= 0:
                        degrees[i_idx] += 1
                    if j_idx >= 0:
                        degrees[j_idx] += 1
                count0 = int(np.sum(degrees == 0))
                count1 = int(np.sum(degrees == 1))
                count_gt1 = n_stickers - count0 - count1
                frac_bond0_series.append(count0 / n_stickers)
                frac_bond1_series.append(count1 / n_stickers)
                frac_bond_gt1_series.append(count_gt1 / n_stickers)

            # Cluster sizes based on chain connectivity
            uf = UnionFind(n_chains)
            for i, j in bonds:
                chain_i = int(chain_ids[i])
                chain_j = int(chain_ids[j])
                if chain_i != chain_j:
                    uf.union(chain_i, chain_j)

            sizes = uf.cluster_sizes()
            for size in sizes:
                cluster_hist[size] += 1
            cluster_count_total += len(sizes)
            largest_cluster_fraction_sum += float(np.max(sizes)) / n_chains
            mean_cluster_size_sum += float(np.mean(sizes))
            cluster_frames += 1

            # Intra vs inter bonds
            intra = 0
            inter = 0
            for i, j in bonds:
                if chain_ids[i] == chain_ids[j]:
                    intra += 1
                else:
                    inter += 1
            intra_bond_total += intra
            inter_bond_total += inter
            bond_frames += 1

            # Correlation accumulators
            bond_corr.update(bonds)
            if open_corr is not None:
                open_corr.update(open_stickers)

            # Exchange rates
            if (
                prev_bonds is not None
                and prev_open is not None
                and prev_partners is not None
            ):
                new_bonds = bonds - prev_bonds
                n_m = len(prev_open)
                if n_m > 0:
                    assoc = 0
                    dissoc = 0
                    for i, j in new_bonds:
                        i_prev = prev_partners.get(i, set())
                        j_prev = prev_partners.get(j, set())
                        if (i_prev and (j not in i_prev)) or (
                            j_prev and (i not in j_prev)
                        ):
                            assoc += 1
                        else:
                            dissoc += 1
                    rate_assoc_sum += assoc / (n_m * frame_dt)
                    rate_dissoc_sum += dissoc / (n_m * frame_dt)
                    rate_count += 1

            # Save partner mapping for next frame
            partner_map: Dict[int, set] = defaultdict(set)
            for i, j in bonds:
                partner_map[i].add(j)
                partner_map[j].add(i)

            prev_bonds = bonds
            prev_open = open_stickers
            prev_partners = partner_map

            # Pressure tensor for G(t)
            if pressure_key is not None and hasattr(frame, "log"):
                pressure_val = frame.log.get(pressure_key, None)
                if pressure_val is not None:
                    pressure_arr = np.squeeze(pressure_val)
                    if pressure_arr.shape[-1] >= 6:
                        xy = float(pressure_arr[1])
                        xz = float(pressure_arr[2])
                        yz = float(pressure_arr[4])
                        pressure_series.append((xy, xz, yz))

            # MSD sampling
            if images is None:
                unwrapped = positions[sample_ids]
            else:
                unwrapped = positions[sample_ids] + images[sample_ids] * box_length
            msd_positions.append(unwrapped.astype(np.float64))

            if progress_label is not None and (
                analyzed_idx % progress_interval == 0 or analyzed_idx == n_analyzed
            ):
                progress_pct = 100.0 * analyzed_idx / n_analyzed
                log(
                    f"{progress_label}: frame progress {analyzed_idx}/{n_analyzed} "
                    f"({progress_pct:.1f}%)"
                )

        # Build correlation functions
        cs = bond_corr.correlation()
        cb = open_corr.correlation() if open_corr is not None else None

        cs_time = np.arange(1, len(cs) + 1, dtype=np.float64) * frame_dt
        cb_time = (
            np.arange(1, len(cb) + 1, dtype=np.float64) * frame_dt
            if cb is not None
            else None
        )

        bond_fit_window = get_bond_tau_fit_window(reactive_epsilon)
        tau_s = fit_exponential_with_time_window(cs_time, cs, bond_fit_window)
        tau_b = (
            fit_exponential(cb_time, cb)
            if (cb_time is not None and cb is not None)
            else float("nan")
        )

        # Connectivity autocorrelation
        p_arr = np.array(p_series, dtype=np.float64)
        cp_full = autocorr_fft(p_arr, subtract_mean=True)
        cp = cp_full[1 : max_lag_frames + 1]
        cp_time = np.arange(1, len(cp) + 1, dtype=np.float64) * frame_dt
        tau_c = fit_exponential(cp_time, cp)

        # Stress autocorrelation for G(t)
        G_t = None
        G_time = None
        if pressure_series:
            pressure_arr = np.array(pressure_series, dtype=np.float64)
            g_components = []
            for idx in range(pressure_arr.shape[1]):
                acf = autocorr_fft(pressure_arr[:, idx], subtract_mean=True)
                g_components.append(acf)
            g_components = np.mean(np.vstack(g_components), axis=0)

            # Convert to modulus (Green-Kubo)
            volume = box_length**3
            kT = metadata.get("temperature", 1.0)
            G_full = volume / kT * g_components
            G_t = G_full[1 : max_lag_frames + 1]
            G_time = np.arange(1, len(G_t) + 1, dtype=np.float64) * frame_dt

        # MSD
        msd_time = None
        msd = None
        if msd_positions:
            pos = np.stack(msd_positions, axis=0)
            max_lag = min(msd_max_lag_frames, pos.shape[0] - 1)
            msd_vals = []
            for lag in range(1, max_lag + 1):
                diff = pos[lag:] - pos[:-lag]
                msd_vals.append(np.mean(np.sum(diff * diff, axis=-1)))
            msd = np.array(msd_vals, dtype=np.float64)
            msd_time = np.arange(1, len(msd) + 1, dtype=np.float64) * frame_dt

        # Summaries
        result = {
            "p_open_mean": (
                float(np.mean(p_open_series)) if p_open_series else float("nan")
            ),
            "p_mean": float(np.mean(p_series)) if p_series else float("nan"),
            "epsilon_mean": (
                float(np.mean(epsilon_series)) if epsilon_series else float("nan")
            ),
            "intra_bonds_mean": intra_bond_total / max(bond_frames, 1),
            "inter_bonds_mean": inter_bond_total / max(bond_frames, 1),
            "intra_inter_ratio": (
                intra_bond_total / inter_bond_total
                if inter_bond_total > 0
                else float("nan")
            ),
            "mean_cluster_size": mean_cluster_size_sum / max(cluster_frames, 1),
            "largest_cluster_fraction": largest_cluster_fraction_sum
            / max(cluster_frames, 1),
            "rate_assoc": rate_assoc_sum / max(rate_count, 1),
            "rate_dissoc": rate_dissoc_sum / max(rate_count, 1),
            "tau_s": tau_s,
            "tau_b": tau_b,
            "tau_c": tau_c,
            "stickers_per_chain": float(metadata.get("stickers_per_chain", 4)),
            "p_c": (
                float(1.0 / (float(metadata.get("stickers_per_chain", 4)) - 1.0))
                if float(metadata.get("stickers_per_chain", 4)) > 1.0
                else float("nan")
            ),
            "cluster_hist": cluster_hist,
            "cluster_count_total": cluster_count_total,
            "cs_time": cs_time,
            "cs": cs,
            "cb_time": cb_time,
            "cb": cb,
            "cp_time": cp_time,
            "cp": cp,
            "G_time": G_time,
            "G_t": G_t,
            "msd_time": msd_time,
            "msd": msd,
            "frac_bond0_series": np.array(frac_bond0_series, dtype=np.float64),
            "frac_bond1_series": np.array(frac_bond1_series, dtype=np.float64),
            "frac_bond_gt1_series": np.array(frac_bond_gt1_series, dtype=np.float64),
            "cexc_series": np.array(cexc_series, dtype=np.float64),
        }

        return result


def mean_and_stderr(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, stderr


def write_properties_csv(path: str, properties: Dict[str, Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("property,mean,stderr\n")
        for name, stats in properties.items():
            handle.write(f"{name},{stats['mean']},{stats['stderr']}\n")


def write_timeseries(
    path: str, time: np.ndarray, mean: np.ndarray, stderr: np.ndarray
) -> None:
    header = "time,mean,stderr\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for t, m, s in zip(time, mean, stderr):
            handle.write(f"{t:.6e},{m:.6e},{s:.6e}\n")


def write_autocorr_fit_plot(
    path: str,
    time: np.ndarray,
    values: np.ndarray,
    title: str,
    y_label: str,
    fit_window: Tuple[float, float] | None = None,
) -> None:
    """Plot median + IQR of replicate autocorrelations and an exponential fit."""
    median = np.median(values, axis=0)
    q1 = np.percentile(values, 25.0, axis=0)
    q3 = np.percentile(values, 75.0, axis=0)

    tau_fit = fit_exponential_with_time_window(time, median, fit_window)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, median, color="#2b2b2b", lw=2.0, label="Median")

    if np.isfinite(tau_fit):
        fit_curve = np.exp(-time / tau_fit)
        ax.plot(
            time,
            fit_curve,
            color="#e77500",
            lw=2.0,
            label=f"Exponential fit (tau={tau_fit:.3g})",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_plateau_fit_plot(
    path: str,
    time: np.ndarray,
    values: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """Plot median + IQR of replicate autocorrelations and a plateau+exp fit."""
    median = np.median(values, axis=0)
    q1 = np.percentile(values, 25.0, axis=0)
    q3 = np.percentile(values, 75.0, axis=0)

    plateau, tau_fit = fit_plateau_exponential(time, median)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, median, color="#2b2b2b", lw=2.0, label="Median")

    if np.isfinite(plateau) and np.isfinite(tau_fit):
        fit_curve = plateau + (1.0 - plateau) * np.exp(-time / tau_fit)
        ax.plot(
            time,
            fit_curve,
            color="#e77500",
            lw=2.0,
            label=f"Plateau+exp fit (A={plateau:.3g}, tau={tau_fit:.3g})",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compute_cexc_mean(
    sticker_positions: np.ndarray,
    box_length: float,
    r_cut: float,
    weakening_inner: float,
    weakening_outer: float,
    smooth_eps: float = 1e-6,
) -> float:
    n_stickers = sticker_positions.shape[0]
    if n_stickers < 2:
        return float("nan")

    r_max = max(r_cut, weakening_outer)
    # cKDTree with boxsize expects positions in [0, boxsize).
    wrapped = np.mod(sticker_positions + 0.5 * box_length, box_length)
    tree = cKDTree(wrapped, boxsize=box_length)
    pairs = tree.sparse_distance_matrix(tree, r_max, output_type="ndarray")
    if pairs.size == 0:
        return float("nan")

    if pairs.dtype.fields is not None:
        names = pairs.dtype.names or ()
        if "i" in names:
            i_idx = pairs["i"].astype(np.int64)
            j_idx = pairs["j"].astype(np.int64)
        else:
            i_idx = pairs["row"].astype(np.int64)
            j_idx = pairs["col"].astype(np.int64)
        if "v" in names:
            dist = pairs["v"].astype(np.float64)
        elif "d" in names:
            dist = pairs["d"].astype(np.float64)
        else:
            dist = pairs["dist"].astype(np.float64)
    else:
        i_idx = pairs[:, 0].astype(np.int64)
        j_idx = pairs[:, 1].astype(np.int64)
        dist = pairs[:, 2].astype(np.float64)

    mask_valid = i_idx != j_idx
    if not np.all(mask_valid):
        i_idx = i_idx[mask_valid]
        j_idx = j_idx[mask_valid]
        dist = dist[mask_valid]

    if i_idx.size == 0:
        return float("nan")

    coordination = np.zeros(n_stickers, dtype=np.float64)
    w_ij = np.zeros_like(dist)
    mask_inner = dist <= weakening_inner
    mask_outer = (dist > weakening_inner) & (dist < weakening_outer)
    if np.any(mask_inner):
        w_ij[mask_inner] = 1.0
    if np.any(mask_outer):
        angle = np.pi * (dist[mask_outer] - weakening_inner) / (
            weakening_outer - weakening_inner
        )
        w_ij[mask_outer] = 0.5 * (1.0 + np.cos(angle))

    if np.any(w_ij):
        np.add.at(coordination, i_idx, w_ij)
        np.add.at(coordination, j_idx, w_ij)

    mask_cut = dist < r_cut
    if not np.any(mask_cut):
        return float("nan")

    i_cut = i_idx[mask_cut]
    j_cut = j_idx[mask_cut]
    w_cut = w_ij[mask_cut]
    raw = (coordination[i_cut] - w_cut) + (coordination[j_cut] - w_cut)
    cexc_vals = 0.5 * (raw + np.sqrt(raw * raw + smooth_eps * smooth_eps))
    return float(np.mean(cexc_vals))


def analyze_replicate_job(
    epsilon: float,
    gsd_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    msd_sample: int,
    msd_max_lag_frames: int,
    rep_label: str,
    rel_path: str,
) -> Tuple[float, Dict]:
    log(f"{rep_label}: start ({rel_path})")
    result = analyze_replicate(
        gsd_path,
        metadata,
        analysis_stride,
        max_lag_frames,
        msd_sample,
        msd_max_lag_frames,
        progress_label=rep_label,
    )
    log(f"{rep_label}: done")
    return epsilon, result


def truncate_lag(
    time: np.ndarray | None, values: np.ndarray | None, max_lag: int
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if time is None or values is None:
        return None, None
    n = min(max_lag, len(time))
    return time[:n], values[:, :n]


def truncate_time_window(
    time: np.ndarray | None,
    values: np.ndarray | None,
    fit_window: Tuple[float, float] | None,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if time is None or values is None:
        return None, None
    if fit_window is None:
        return time, values
    t_min, t_max = fit_window
    mask = np.isfinite(time) & (time >= t_min) & (time <= t_max)
    if np.count_nonzero(mask) < 2:
        return time, values
    return time[mask], values[:, mask]


def write_fraction_violin_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    parts = ax.violinplot(data, positions=epsilon_values, widths=0.6, showextrema=False)
    for body in parts.get("bodies", []):
        body.set_facecolor("#9e9e9e")
        body.set_edgecolor("#6f6f6f")
        body.set_alpha(0.5)

    for eps, values in zip(epsilon_values, data):
        if values.size == 0:
            continue
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color="#2b2b2b", lw=2.0)
        ax.scatter([eps], [med], color="#2b2b2b", s=18, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_cexc_vs_epsilon_plot(
    path: str, epsilon_values: List[float], data: List[np.ndarray]
) -> None:
    medians = []
    q1s = []
    q3s = []
    for values in data:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            medians.append(float("nan"))
            q1s.append(float("nan"))
            q3s.append(float("nan"))
            continue
        medians.append(float(np.median(finite)))
        q1s.append(float(np.percentile(finite, 25.0)))
        q3s.append(float(np.percentile(finite, 75.0)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(epsilon_values, medians, color="#2b2b2b", marker="o", lw=2.0)
    ax.fill_between(epsilon_values, q1s, q3s, color="#9e9e9e", alpha=0.35)
    ax.set_title("Mean C_exc vs epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Mean C_exc (reactive pairs)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data: List[np.ndarray],
    title: str,
    y_label: str,
    log_transform: bool = False,
) -> None:
    violin_data: List[np.ndarray] = []
    positions: List[float] = []
    for eps, values in zip(epsilon_values, data):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if log_transform:
            arr = arr[arr > 0.0]
            if arr.size:
                arr = np.log(arr)
        if arr.size == 0:
            continue
        positions.append(float(eps))
        violin_data.append(arr)

    if not violin_data:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    parts = ax.violinplot(violin_data, positions=positions, widths=0.6, showextrema=False)
    for body in parts.get("bodies", []):
        body.set_facecolor("#9e9e9e")
        body.set_edgecolor("#6f6f6f")
        body.set_alpha(0.5)

    for eps, values in zip(positions, violin_data):
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color="#2b2b2b", lw=2.0)
        ax.scatter([eps], [med], color="#2b2b2b", s=18, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_dual_scalar_violin_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    data_left: List[np.ndarray],
    data_right: List[np.ndarray],
    title: str,
    y_label: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
) -> None:
    left_positions: List[float] = []
    left_violin_data: List[np.ndarray] = []
    left_medians: List[float] = []
    right_positions: List[float] = []
    right_violin_data: List[np.ndarray] = []
    right_medians: List[float] = []
    offset = 0.16
    width = 0.28

    for eps, left_vals, right_vals in zip(epsilon_values, data_left, data_right):
        left_arr = np.asarray(left_vals, dtype=np.float64)
        left_arr = left_arr[np.isfinite(left_arr)]
        right_arr = np.asarray(right_vals, dtype=np.float64)
        right_arr = right_arr[np.isfinite(right_arr)]

        if left_arr.size:
            left_positions.append(float(eps) - offset)
            left_violin_data.append(left_arr)
            left_medians.append(float(np.median(left_arr)))
        if right_arr.size:
            right_positions.append(float(eps) + offset)
            right_violin_data.append(right_arr)
            right_medians.append(float(np.median(right_arr)))

    if not left_violin_data and not right_violin_data:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    if left_violin_data:
        left_parts = ax.violinplot(
            left_violin_data, positions=left_positions, widths=width, showextrema=False
        )
        for body in left_parts.get("bodies", []):
            body.set_facecolor(left_color)
            body.set_edgecolor(left_color)
            body.set_alpha(0.35)
        for x, values in zip(left_positions, left_violin_data):
            q1 = float(np.percentile(values, 25.0))
            q3 = float(np.percentile(values, 75.0))
            med = float(np.median(values))
            ax.vlines(x, q1, q3, color=left_color, lw=2.0)
            ax.scatter([x], [med], color=left_color, s=18, zorder=3)
        ax.plot(left_positions, left_medians, color=left_color, lw=1.8, alpha=0.9, zorder=2)

    if right_violin_data:
        right_parts = ax.violinplot(
            right_violin_data, positions=right_positions, widths=width, showextrema=False
        )
        for body in right_parts.get("bodies", []):
            body.set_facecolor(right_color)
            body.set_edgecolor(right_color)
            body.set_alpha(0.35)
        for x, values in zip(right_positions, right_violin_data):
            q1 = float(np.percentile(values, 25.0))
            q3 = float(np.percentile(values, 75.0))
            med = float(np.median(values))
            ax.vlines(x, q1, q3, color=right_color, lw=2.0)
            ax.scatter([x], [med], color=right_color, s=18, zorder=3)
        ax.plot(
            right_positions,
            right_medians,
            color=right_color,
            lw=1.8,
            alpha=0.9,
            zorder=2,
        )

    legend_handles = [
        matplotlib.patches.Patch(
            facecolor=left_color, edgecolor=left_color, alpha=0.35, label=left_label
        ),
        matplotlib.patches.Patch(
            facecolor=right_color, edgecolor=right_color, alpha=0.35, label=right_label
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False)

    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_label)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilon_values])
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_log_tau_vs_epsilon_plot(
    path: str, epsilon_values: List[float], tau_data: List[np.ndarray]
) -> None:
    write_scalar_violin_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        data=tau_data,
        title="ln(Bond Correlation Decay Tau) vs epsilon",
        y_label="ln(bond persistence time)",
        log_transform=True,
    )


def write_tau_vs_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    tau_data: List[np.ndarray],
    title: str = "Bond Correlation Decay Tau vs epsilon",
    y_label: str = "bond persistence time",
) -> None:
    write_scalar_violin_vs_epsilon_plot(
        path=path,
        epsilon_values=epsilon_values,
        data=tau_data,
        title=title,
        y_label=y_label,
        log_transform=False,
    )


def main() -> None:
    args = parse_args()
    log(f"Scanning trajectories under {args.input_root}")
    runs = discover_runs(args.input_root)
    if not runs:
        raise RuntimeError(f"No trajectories found under {args.input_root}")
    log(f"Discovered {len(runs)} trajectory/metadata pairs")

    # Group runs by epsilon
    grouped: Dict[float, List[Tuple[str, Dict]]] = defaultdict(list)
    for gsd_path, metadata_path in runs:
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"Missing metadata.json for {gsd_path}")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        epsilon = float(metadata["reactive_epsilon"])
        grouped[epsilon].append((gsd_path, metadata))
    log(f"Grouped runs into {len(grouped)} epsilon values")

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    summary_json: Dict[str, Dict] = {}
    epsilon_values: List[float] = []
    frac_bond0_data: List[np.ndarray] = []
    frac_bond1_data: List[np.ndarray] = []
    frac_bond_gt1_data: List[np.ndarray] = []
    cexc_data: List[np.ndarray] = []
    msd_distribution_data: List[np.ndarray] = []
    tau_s_data: List[np.ndarray] = []
    tau_b_data: List[np.ndarray] = []
    scalar_violin_data: Dict[str, List[np.ndarray]] = {
        "p_open_mean": [],
        "p_mean": [],
        "epsilon_mean": [],
        "intra_inter_ratio": [],
        "tau_c": [],
        "rate_assoc": [],
        "rate_dissoc": [],
        "stickers_per_chain": [],
        "p_c": [],
    }

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is None:
        raise RuntimeError(
            "SLURM_CPUS_PER_TASK is not set. Run under Slurm with --cpus-per-task "
            "or export SLURM_CPUS_PER_TASK."
        )
    try:
        n_jobs = int(slurm_cpus)
    except ValueError as exc:
        raise RuntimeError(
            f"SLURM_CPUS_PER_TASK must be an integer, got '{slurm_cpus}'."
        ) from exc
    if n_jobs < 1:
        raise RuntimeError(
            f"SLURM_CPUS_PER_TASK must be >= 1, got '{slurm_cpus}'."
        )

    sorted_groups = sorted(grouped.items())
    jobs = []
    for eps_idx, (epsilon, entries) in enumerate(sorted_groups, start=1):
        log(
            f"Queueing epsilon group {eps_idx}/{len(sorted_groups)}: "
            f"eps={epsilon:g}, replicates={len(entries)}"
        )
        for rep_idx, (gsd_path, metadata) in enumerate(entries, start=1):
            rel_path = os.path.relpath(gsd_path, args.input_root)
            rep_label = f"eps={epsilon:g} rep={rep_idx}/{len(entries)}"
            jobs.append(
                delayed(analyze_replicate_job)(
                    epsilon,
                    gsd_path,
                    metadata,
                    args.analysis_stride,
                    args.max_lag_frames,
                    args.msd_sample,
                    args.msd_max_lag_frames,
                    rep_label,
                    rel_path,
                )
            )

    log(f"Starting parallel analysis with {n_jobs} workers on {len(jobs)} runs")
    results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    log(f"Completed parallel analysis for {len(results)} runs")

    grouped_results: Dict[float, List[Dict]] = defaultdict(list)
    for epsilon, result in results:
        grouped_results[epsilon].append(result)

    sorted_results = sorted(grouped_results.items())
    for eps_idx, (epsilon, replicate_results) in enumerate(sorted_results, start=1):
        log(
            f"Aggregating epsilon group {eps_idx}/{len(sorted_results)}: "
            f"eps={epsilon:g}, replicates={len(replicate_results)}"
        )
        epsilon_values.append(epsilon)

        # Aggregate scalar metrics
        scalar_keys = [
            "p_open_mean",
            "p_mean",
            "epsilon_mean",
            "intra_bonds_mean",
            "inter_bonds_mean",
            "intra_inter_ratio",
            "mean_cluster_size",
            "largest_cluster_fraction",
            "rate_assoc",
            "rate_dissoc",
            "tau_s",
            "tau_b",
            "tau_c",
        ]

        scalar_summary = {}
        for key in scalar_keys:
            vals = [res[key] for res in replicate_results]
            mean, stderr = mean_and_stderr(vals)
            scalar_summary[key] = {"mean": mean, "stderr": stderr}
        tau_s_data.append(
            np.asarray([float(res.get("tau_s", float("nan"))) for res in replicate_results], dtype=np.float64)
        )
        tau_b_data.append(
            np.asarray([float(res.get("tau_b", float("nan"))) for res in replicate_results], dtype=np.float64)
        )
        for key in scalar_violin_data:
            scalar_violin_data[key].append(
                np.asarray(
                    [float(res.get(key, float("nan"))) for res in replicate_results],
                    dtype=np.float64,
                )
            )

        f_mean, f_stderr = mean_and_stderr(
            [float(res.get("stickers_per_chain", float("nan"))) for res in replicate_results]
        )
        pc_mean, pc_stderr = mean_and_stderr(
            [float(res.get("p_c", float("nan"))) for res in replicate_results]
        )
        epsilon_properties = {
            "p_open": scalar_summary["p_open_mean"],
            "bonding_probability_p": scalar_summary["p_mean"],
            "gelation_epsilon": scalar_summary["epsilon_mean"],
            "stickers_per_chain_f": {"mean": f_mean, "stderr": f_stderr},
            "gel_point_p_c": {"mean": pc_mean, "stderr": pc_stderr},
            "intra_to_inter_bond_ratio": scalar_summary["intra_inter_ratio"],
            "bond_persistence_time_tau_s": scalar_summary["tau_s"],
            "open_sticker_persistence_time_tau_b": scalar_summary["tau_b"],
            "fluctuation_relaxation_time_tau_c": scalar_summary["tau_c"],
            "associative_exchange_rate_R_a": scalar_summary["rate_assoc"],
            "passive_dimerization_rate_R_d": scalar_summary["rate_dissoc"],
        }

        # Cluster distribution
        cluster_hists = [res["cluster_hist"] for res in replicate_results]
        cluster_counts = [res["cluster_count_total"] for res in replicate_results]
        max_len = max(len(hist) for hist in cluster_hists)
        padded = []
        for hist in cluster_hists:
            pad = np.zeros(max_len, dtype=np.float64)
            pad[: len(hist)] = hist
            padded.append(pad)
        cluster_p = [pad / max(count, 1) for pad, count in zip(padded, cluster_counts)]
        cluster_mean = np.mean(np.vstack(cluster_p), axis=0)
        cluster_stderr = np.std(np.vstack(cluster_p), axis=0, ddof=1) / np.sqrt(
            len(cluster_p)
        )

        # Time series averages
        def aggregate_timeseries(key_time: str, key_val: str):
            series = [
                (res[key_time], res[key_val])
                for res in replicate_results
                if res[key_val] is not None
            ]
            if not series:
                return None, None, None, None
            lengths = [len(t) for t, _ in series]
            if len(set(lengths)) != 1:
                raise RuntimeError(
                    f"Replicate series length mismatch for eps={epsilon:g}, "
                    f"series={key_val}: lengths={lengths}"
                )
            val_lengths = [len(v) for _, v in series]
            if len(set(val_lengths)) != 1:
                raise RuntimeError(
                    f"Replicate value length mismatch for eps={epsilon:g}, "
                    f"series={key_val}: lengths={val_lengths}"
                )
            n = lengths[0]
            time = series[0][0][:n]
            values = np.stack([v[:n] for _, v in series], axis=0)
            mean = np.mean(values, axis=0)
            stderr = (
                np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                if values.shape[0] > 1
                else np.zeros_like(mean)
            )
            return time, values, mean, stderr

        cs_time, cs_values, cs_mean, cs_stderr = aggregate_timeseries("cs_time", "cs")
        cb_time, cb_values, cb_mean, cb_stderr = aggregate_timeseries("cb_time", "cb")
        cp_time, cp_values, cp_mean, cp_stderr = aggregate_timeseries("cp_time", "cp")
        G_time, _, G_mean, G_stderr = aggregate_timeseries("G_time", "G_t")
        msd_time, _, msd_mean, msd_stderr = aggregate_timeseries("msd_time", "msd")

        # Fraction of sticker degrees across all frames/replicates
        frac_bond0_all = np.concatenate(
            [res["frac_bond0_series"] for res in replicate_results]
        )
        frac_bond1_all = np.concatenate(
            [res["frac_bond1_series"] for res in replicate_results]
        )
        frac_bond_gt1_all = np.concatenate(
            [res["frac_bond_gt1_series"] for res in replicate_results]
        )
        cexc_all = np.concatenate([res["cexc_series"] for res in replicate_results])
        frac_bond0_data.append(frac_bond0_all)
        frac_bond1_data.append(frac_bond1_all)
        frac_bond_gt1_data.append(frac_bond_gt1_all)
        cexc_data.append(cexc_all)
        msd_samples = [res["msd"] for res in replicate_results if res["msd"] is not None]
        if msd_samples:
            msd_distribution_data.append(
                np.concatenate([np.asarray(sample, dtype=np.float64) for sample in msd_samples])
            )
        else:
            msd_distribution_data.append(np.array([], dtype=np.float64))

        # Output per-epsilon directory
        eps_dir = os.path.join(args.output_dir, f"eps_{epsilon:g}")
        os.makedirs(eps_dir, exist_ok=True)
        log(f"Writing outputs for eps={epsilon:g} to {eps_dir}")

        # Cluster distribution CSV
        with open(
            os.path.join(eps_dir, "cluster_distribution.csv"), "w", encoding="utf-8"
        ) as handle:
            handle.write("cluster_size,mean,stderr\n")
            for size, (m, s) in enumerate(zip(cluster_mean, cluster_stderr)):
                handle.write(f"{size},{m:.6e},{s:.6e}\n")

        if cs_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "bond_correlation.csv"),
                cs_time,
                cs_mean,
                cs_stderr,
            )
        if cb_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "open_correlation.csv"),
                cb_time,
                cb_mean,
                cb_stderr,
            )
        if cp_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "connectivity_correlation.csv"),
                cp_time,
                cp_mean,
                cp_stderr,
            )
        if cs_time is not None and cs_values is not None:
            bond_fit_window = get_bond_tau_fit_window(epsilon)
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "bond_correlation_fit.png"),
                cs_time,
                cs_values,
                title=f"Bond Correlation Decay (eps={epsilon:g})",
                y_label="C_s(t)",
                fit_window=bond_fit_window,
            )
        cb_time_fit, cb_values_fit = truncate_lag(
            cb_time, cb_values, args.max_lag_frames
        )
        open_fit_window = get_open_corr_fit_window(epsilon)
        cb_time_fit, cb_values_fit = truncate_time_window(
            cb_time_fit, cb_values_fit, open_fit_window
        )
        if cb_time_fit is not None and cb_values_fit is not None:
            write_plateau_fit_plot(
                os.path.join(eps_dir, "open_correlation_fit.png"),
                cb_time_fit,
                cb_values_fit,
                title=f"Open-Sticker Correlation Decay (eps={epsilon:g})",
                y_label="Open-sticker persistence time",
            )
        cp_time_fit, cp_values_fit = truncate_lag(
            cp_time, cp_values, args.max_lag_frames
        )
        if cp_time_fit is not None and cp_values_fit is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "connectivity_correlation_fit.png"),
                cp_time_fit,
                cp_values_fit,
                title=f"Connectivity Correlation Decay (eps={epsilon:g})",
                y_label="C_p(t)",
            )
        if G_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "stress_modulus.csv"), G_time, G_mean, G_stderr
            )
        if msd_time is not None:
            write_timeseries(
                os.path.join(eps_dir, "msd.csv"), msd_time, msd_mean, msd_stderr
            )
        with open(os.path.join(eps_dir, "properties.json"), "w", encoding="utf-8") as handle:
            json.dump(epsilon_properties, handle, indent=2)
        write_properties_csv(os.path.join(eps_dir, "properties.csv"), epsilon_properties)

        summary_rows.append(
            {
                "epsilon": epsilon,
                **{f"{k}_mean": v["mean"] for k, v in scalar_summary.items()},
                **{f"{k}_stderr": v["stderr"] for k, v in scalar_summary.items()},
            }
        )
        summary_json[f"{epsilon:g}"] = scalar_summary
        log(f"Finished epsilon group eps={epsilon:g}")

    # Write summary JSON and CSV
    with open(
        os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary_json, handle, indent=2)
    log(f"Wrote {os.path.join(args.output_dir, 'summary.json')}")

    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(
            os.path.join(args.output_dir, "summary.csv"), "w", encoding="utf-8"
        ) as handle:
            handle.write(",".join(keys) + "\n")
            for row in summary_rows:
                handle.write(",".join(str(row[k]) for k in keys) + "\n")
        log(f"Wrote {os.path.join(args.output_dir, 'summary.csv')}")

    if epsilon_values:
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond0_violin.png"),
            epsilon_values,
            frac_bond0_data,
            title="Sticker Fraction with 0 Bonds",
            y_label="Fraction of stickers",
        )
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond1_violin.png"),
            epsilon_values,
            frac_bond1_data,
            title="Sticker Fraction with 1 Bond",
            y_label="Fraction of stickers",
        )
        write_fraction_violin_plot(
            os.path.join(args.output_dir, "sticker_fraction_bond_gt1_violin.png"),
            epsilon_values,
            frac_bond_gt1_data,
            title="Sticker Fraction with >1 Bonds",
            y_label="Fraction of stickers",
        )
        write_cexc_vs_epsilon_plot(
            os.path.join(args.output_dir, "cexc_vs_epsilon.png"),
            epsilon_values,
            cexc_data,
        )
        write_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "monomer_msd_violin_vs_epsilon.png"),
            epsilon_values,
            msd_distribution_data,
            title="Monomer MSD Distribution vs epsilon",
            y_label="monomer MSD",
        )
        write_log_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, "ln_bond_tau_vs_epsilon.png"),
            epsilon_values,
            tau_s_data,
        )
        write_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, "bond_tau_vs_epsilon.png"),
            epsilon_values,
            tau_s_data,
        )
        write_tau_vs_epsilon_plot(
            os.path.join(args.output_dir, "open_tau_vs_epsilon.png"),
            epsilon_values,
            tau_b_data,
            title="Open-Sticker Correlation Decay Tau vs epsilon",
            y_label="open sticker persistence time",
        )
        scalar_violin_specs = [
            (
                "p_open_mean",
                "p_open_vs_epsilon.png",
                "Open Sticker Fraction vs epsilon",
                "p_open",
            ),
            (
                "p_mean",
                "bonding_probability_vs_epsilon.png",
                "Bonding Probability vs epsilon",
                "p",
            ),
            (
                "epsilon_mean",
                "gelation_epsilon_vs_epsilon.png",
                "Degree of Gelation vs epsilon",
                "gelation epsilon",
            ),
            (
                "intra_inter_ratio",
                "intra_to_inter_bond_ratio_vs_epsilon.png",
                "Intra/Inter Bond Ratio vs epsilon",
                "intra/inter bond ratio",
            ),
            (
                "tau_c",
                "fluctuation_relaxation_tau_c_vs_epsilon.png",
                "Fluctuation Relaxation Time vs epsilon",
                "tau_c",
            ),
            (
                "stickers_per_chain",
                "stickers_per_chain_vs_epsilon.png",
                "Stickers per Chain vs epsilon",
                "f",
            ),
        ]
        for key, filename, title, y_label in scalar_violin_specs:
            write_scalar_violin_vs_epsilon_plot(
                os.path.join(args.output_dir, filename),
                epsilon_values,
                scalar_violin_data[key],
                title=title,
                y_label=y_label,
            )
        write_dual_scalar_violin_vs_epsilon_plot(
            os.path.join(args.output_dir, "exchange_rate_comparison_vs_epsilon.png"),
            epsilon_values,
            scalar_violin_data["rate_assoc"],
            scalar_violin_data["rate_dissoc"],
            title="Associative Exchange vs Passive Dimerization Rates",
            y_label="rate (1/time)",
            left_label="Associative exchange rate (R_a)",
            right_label="Passive dimerization rate (R_d)",
            left_color="#e77500",
            right_color="#121212",
        )

    log("Analysis complete")


if __name__ == "__main__":
    main()
