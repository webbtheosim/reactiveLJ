"""Analyze ReactiveLJ trajectories (Block 2 metrics).

This script loops over GSD files produced by data_generation, computes the
analysis metrics described in agents.md, and averages results over replicates
for each ReactiveLJ attraction strength.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import argparse
import hashlib
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import gsd.hoomd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports resolve when running from repo root.
sys.path.append(os.path.dirname(__file__))

FALLBACK_TAU_R0 = 4041.0
MAX_ANALYSIS_LAG_TAU_R0 = 100.0
MAX_ANALYSIS_LAG_TIME = FALLBACK_TAU_R0 * MAX_ANALYSIS_LAG_TAU_R0

from analysis_utils import (
    CorrelationAccumulator,
    UnionFind,
    autocorr_fft,
    compute_cexc_mean_from_neighbor_pairs,
    compute_r_thresh,
    compute_sticker_neighbor_pairs,
    extract_semilog_linear_region,
    fit_exponential,
    fit_exponential_semilog_linear_region,
    multitau_autocovariance,
)


REPLICATE_CACHE_VERSION = 2


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


def file_signature(path: str) -> Dict[str, int | str] | None:
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_replicate_cache_key(
    gsd_path: str,
    metadata_path: str,
    analysis_stride: int,
    max_lag_frames: int,
    msd_sample: int,
    msd_max_lag_frames: int,
) -> str:
    virial_log_path = os.path.join(os.path.dirname(gsd_path), "virial_tensor_log.gsd")
    payload = {
        "cache_version": REPLICATE_CACHE_VERSION,
        "sources": {
            "trajectory": file_signature(gsd_path),
            "metadata": file_signature(metadata_path),
            "virial_log": file_signature(virial_log_path),
        },
        "analysis_args": {
            "analysis_stride": int(analysis_stride),
            "max_lag_frames": int(max_lag_frames),
            "msd_sample": int(msd_sample),
            "msd_max_lag_frames": int(msd_max_lag_frames),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_replicate_result(cache_path: str) -> Dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def save_cached_replicate_result(cache_path: str, result: Dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def should_compute_open_corr(epsilon: float) -> bool:
    return epsilon > 6.0


def find_virial_key(frame) -> str | None:
    if not hasattr(frame, "log"):
        return None
    for key in frame.log.keys():
        if "virial_tensor" in key:
            return key
    return None


def parse_virial_tensor_components(
    virial_val,
) -> np.ndarray | None:
    virial_arr = np.asarray(np.squeeze(virial_val), dtype=np.float64)
    if virial_arr.ndim == 0 or virial_arr.shape[-1] < 6:
        return None
    # HOOMD tensor ordering is [xx, xy, xz, yy, yz, zz].
    return virial_arr[:6]


def load_virial_series_from_gsd(
    virial_gsd_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the polymer-network configurational virial log from GSD."""
    with gsd.hoomd.open(virial_gsd_path, "r") as virial_traj:
        n_frames = len(virial_traj)
        if n_frames == 0:
            return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
        virial_key = find_virial_key(virial_traj[0])
        if virial_key is None:
            return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
        virial_samples = np.empty((n_frames, 6), dtype=np.float64)
        virial_steps = np.empty((n_frames,), dtype=np.int64)
        count = 0
        for frame in virial_traj:
            if not hasattr(frame, "log"):
                continue
            virial_val = frame.log.get(virial_key, None)
            if virial_val is None:
                continue
            parsed = parse_virial_tensor_components(virial_val)
            if parsed is None:
                continue
            virial_samples[count] = parsed
            virial_steps[count] = int(frame.configuration.step)
            count += 1
    if count == 0:
        return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return virial_samples[:count], virial_steps[:count]


def infer_sample_dt(
    sample_steps: np.ndarray, dt: float, fallback_step_stride: float
) -> float:
    if sample_steps.size >= 2:
        diffs = np.diff(sample_steps)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size > 0:
            return dt * float(np.median(positive_diffs))
    return dt * float(fallback_step_stride)


def compute_stress_autocovariance_multitau(
    tensor_arr: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Compute the polymer-network stress autocovariance on a multi-tau lag grid."""
    if tensor_arr.ndim != 2 or tensor_arr.shape[0] <= 1 or tensor_arr.shape[1] < 6:
        return None, None

    xx = tensor_arr[:, 0]
    xy = tensor_arr[:, 1]
    xz = tensor_arr[:, 2]
    yy = tensor_arr[:, 3]
    yz = tensor_arr[:, 4]
    zz = tensor_arr[:, 5]

    weighted_series = (
        (1.0 / 5.0, (xy, xz, yz)),
        (1.0 / 30.0, (xx - yy, xx - zz, yy - zz)),
    )

    g_lags = None
    g_cov = None
    for weight, series_group in weighted_series:
        for series in series_group:
            centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
            lags_i, cov_i = multitau_autocovariance(centered)
            if g_lags is None:
                g_lags = lags_i
                g_cov = np.zeros_like(cov_i, dtype=np.float64)
            elif not np.array_equal(lags_i, g_lags):
                raise RuntimeError(
                    "Multi-tau lag grids do not match across stress components."
                )
            g_cov += weight * cov_i

    return g_lags, g_cov


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
        sticker_chain_ids = chain_ids[sticker_ids]
        bond_code_scale = np.int64(max(n_stickers, 1))

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
        r_cut = float(r_cut)
        weakening_inner = float(weakening_inner)
        weakening_outer = float(weakening_outer)
        max_pair_cutoff = max(float(r_thresh), r_cut, weakening_outer)

        dt = float(metadata.get("dt", 0.005))
        frame_steps = int(metadata.get("frame_steps", 10_000))
        frame_dt = dt * frame_steps * analysis_stride
        reactive_epsilon = float(metadata.get("reactive_epsilon", float("nan")))
        compute_open_corr = should_compute_open_corr(reactive_epsilon)
        stickers_per_chain = float(metadata.get("stickers_per_chain", 4))
        p_c = (
            float(1.0 / (stickers_per_chain - 1.0))
            if stickers_per_chain > 1.0
            else float("nan")
        )

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
        largest_network_timestep_series: List[float] = []
        largest_network_size_series: List[float] = []

        # Bond stats
        intra_bond_total = 0
        inter_bond_total = 0
        bond_frames = 0

        # Exchange rates
        rate_assoc_sum = 0.0
        rate_dissoc_sum = 0.0
        rate_count = 0

        # MSD sampling
        n_particles = n_polymer_particles
        if msd_sample == 0 or msd_sample >= n_particles:
            sample_ids = np.arange(n_particles)
        else:
            rng = np.random.default_rng(12345)
            sample_ids = rng.choice(n_particles, size=msd_sample, replace=False)

        msd_positions: List[np.ndarray] = []

        prev_bonds: set[int] | None = None
        prev_open_count: int | None = None
        prev_bonded_mask: np.ndarray | None = None

        for analyzed_idx, frame_idx in enumerate(
            range(0, n_frames, analysis_stride), start=1
        ):
            frame = traj[frame_idx]
            positions = frame.particles.position[:n_polymer_particles]
            images = getattr(frame.particles, "image", None)
            if images is not None:
                images = images[:n_polymer_particles]

            sticker_positions = positions[sticker_ids]
            pair_i, pair_j, pair_dist = compute_sticker_neighbor_pairs(
                sticker_positions, box_length, max_pair_cutoff
            )
            cexc_series.append(
                compute_cexc_mean_from_neighbor_pairs(
                    pair_i,
                    pair_j,
                    pair_dist,
                    n_stickers,
                    r_cut,
                    weakening_inner,
                    weakening_outer,
                )
            )

            degrees = np.zeros(n_stickers, dtype=np.int32)
            chain_pair_codes = np.empty((0,), dtype=np.int64)
            intra = 0
            inter = 0
            bonds: set[int] = set()
            if pair_i.size > 0:
                bond_mask = pair_dist < r_thresh
                bond_i = pair_i[bond_mask]
                bond_j = pair_j[bond_mask]
                if bond_i.size > 0:
                    np.add.at(degrees, bond_i, 1)
                    np.add.at(degrees, bond_j, 1)
                    bond_codes = (
                        bond_i.astype(np.int64, copy=False) * bond_code_scale
                        + bond_j.astype(np.int64, copy=False)
                    )
                    bonds = set(bond_codes.tolist())

                    chain_i = sticker_chain_ids[bond_i]
                    chain_j = sticker_chain_ids[bond_j]
                    same_chain = chain_i == chain_j
                    intra = int(np.count_nonzero(same_chain))
                    inter = int(bond_i.size - intra)

                    inter_mask = ~same_chain
                    if np.any(inter_mask):
                        chain_low = np.minimum(chain_i[inter_mask], chain_j[inter_mask]).astype(
                            np.int64, copy=False
                        )
                        chain_high = np.maximum(
                            chain_i[inter_mask], chain_j[inter_mask]
                        ).astype(np.int64, copy=False)
                        chain_pair_codes = np.unique(chain_low * n_chains + chain_high)

            open_idx = np.flatnonzero(degrees == 0)
            open_stickers = set(open_idx.tolist())

            p_open = open_idx.size / n_stickers
            p = 1.0 - p_open
            p_open_series.append(p_open)
            p_series.append(p)

            epsilon_val = (p - p_c) / p_c
            epsilon_series.append(epsilon_val)

            # Sticker degree fractions
            count0 = int(open_idx.size)
            count1 = int(np.count_nonzero(degrees == 1))
            count_gt1 = n_stickers - count0 - count1
            frac_bond0_series.append(count0 / n_stickers)
            frac_bond1_series.append(count1 / n_stickers)
            frac_bond_gt1_series.append(count_gt1 / n_stickers)

            # Cluster sizes based on chain connectivity
            if chain_pair_codes.size == 0:
                sizes = np.ones(n_chains, dtype=np.int32)
            else:
                uf = UnionFind(n_chains)
                chain_src = (chain_pair_codes // n_chains).astype(np.int32, copy=False)
                chain_dst = (chain_pair_codes % n_chains).astype(np.int32, copy=False)
                for chain_a, chain_b in zip(chain_src.tolist(), chain_dst.tolist()):
                    uf.union(chain_a, chain_b)
                sizes = uf.cluster_sizes()
            for size in sizes:
                cluster_hist[size] += 1
            cluster_count_total += len(sizes)
            largest_network_size = float(np.max(sizes))
            largest_cluster_fraction_sum += largest_network_size / n_chains
            mean_cluster_size_sum += float(np.mean(sizes))
            cluster_frames += 1
            frame_step = getattr(frame.configuration, "step", None)
            if frame_step is None:
                frame_step = int(frame_idx * frame_steps)
            largest_network_timestep_series.append(float(frame_step))
            largest_network_size_series.append(largest_network_size)

            # Intra vs inter bonds
            intra_bond_total += intra
            inter_bond_total += inter
            bond_frames += 1

            # Correlation accumulators
            bond_corr.update(bonds)
            if open_corr is not None:
                open_corr.update(open_stickers)

            # Exchange rates
            if prev_bonds is not None and prev_open_count is not None and prev_bonded_mask is not None:
                new_bonds = bonds - prev_bonds
                if prev_open_count > 0 and new_bonds:
                    new_codes = np.fromiter(
                        new_bonds, dtype=np.int64, count=len(new_bonds)
                    )
                    new_i = (new_codes // bond_code_scale).astype(np.int64, copy=False)
                    new_j = (new_codes % bond_code_scale).astype(np.int64, copy=False)
                    assoc_mask = prev_bonded_mask[new_i] | prev_bonded_mask[new_j]
                    assoc = int(np.count_nonzero(assoc_mask))
                    dissoc = int(new_codes.size - assoc)
                    rate_assoc_sum += assoc / (prev_open_count * frame_dt)
                    rate_dissoc_sum += dissoc / (prev_open_count * frame_dt)
                    rate_count += 1

            prev_bonds = bonds
            prev_open_count = count0
            prev_bonded_mask = degrees > 0

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

        tau_s = fit_exponential_semilog_linear_region(cs_time, cs)
        tau_b = (
            fit_exponential_semilog_linear_region(cb_time, cb)
            if (cb_time is not None and cb is not None)
            else float("nan")
        )

        # Connectivity autocorrelation
        p_arr = np.array(p_series, dtype=np.float64)
        cp_full = autocorr_fft(p_arr, subtract_mean=True)
        cp = cp_full[1 : max_lag_frames + 1]
        cp_time = np.arange(1, len(cp) + 1, dtype=np.float64) * frame_dt
        tau_c = fit_exponential(cp_time, cp)

        G_t = None
        G_time = None
        virial_source = None

        virial_arr = np.empty((0, 6), dtype=np.float64)
        virial_steps = np.empty((0,), dtype=np.int64)
        virial_log_path = os.path.join(
            os.path.dirname(gsd_path), "virial_tensor_log.gsd"
        )
        if os.path.exists(virial_log_path):
            virial_arr, virial_steps = load_virial_series_from_gsd(virial_log_path)
            if virial_arr.size > 0:
                virial_source = "virial_tensor_log.gsd"

        if virial_arr.shape[0] > 1:
            g_lags, g_cov = compute_stress_autocovariance_multitau(virial_arr)
            if g_lags is None or g_cov is None:
                g_lags = np.empty((0,), dtype=np.float64)
                g_cov = np.empty((0,), dtype=np.float64)

            skip = 1 if len(g_lags) > 1 and g_lags[0] == 0 else 0
            g_lags = g_lags[skip:]
            g_cov = g_cov[skip:]

            virial_dt = infer_sample_dt(
                virial_steps,
                dt,
                float(metadata.get("virial_log_steps", frame_steps)),
            )
            g_time = g_lags * virial_dt

            if virial_steps.size >= 2:
                runtime = float(np.max(virial_steps) - np.min(virial_steps)) * dt
            else:
                runtime = float(max(virial_arr.shape[0] - 1, 0)) * virial_dt
            max_g_time = min(0.2 * runtime, MAX_ANALYSIS_LAG_TIME)
            if np.isfinite(max_g_time) and max_g_time > 0.0:
                lag_mask = g_time <= max_g_time
                if np.any(lag_mask):
                    g_time = g_time[lag_mask]
                    g_cov = g_cov[lag_mask]

            volume = box_length**3
            kT = float(metadata.get("temperature", 1.0))
            # This Green-Kubo estimate uses the polymer-network configurational
            # stress log, not the total solution stress.
            G_t = (volume / kT) * g_cov
            G_time = g_time

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
            "largest_network_timestep": np.array(
                largest_network_timestep_series, dtype=np.float64
            ),
            "largest_network_size": np.array(
                largest_network_size_series, dtype=np.float64
            ),
            "cs_time": cs_time,
            "cs": cs,
            "cb_time": cb_time,
            "cb": cb,
            "cp_time": cp_time,
            "cp": cp,
            "G_time": G_time,
            "G_t": G_t,
            "virial_source": virial_source,
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
    use_semilog_linear_region: bool = False,
) -> None:
    """Plot median + IQR of replicate autocorrelations and an exponential fit."""
    median = np.median(values, axis=0)
    q1 = np.percentile(values, 25.0, axis=0)
    q3 = np.percentile(values, 75.0, axis=0)

    if use_semilog_linear_region:
        tau_fit = fit_exponential_semilog_linear_region(time, median)
        fit_time, _ = extract_semilog_linear_region(time, median)
    else:
        tau_fit = fit_exponential(time, median)
        fit_time = time

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(time, q1, q3, color="#9e9e9e", alpha=0.35, label="IQR")
    ax.plot(time, median, color="#2b2b2b", lw=2.0, label="Median")

    if np.isfinite(tau_fit) and fit_time.size > 0:
        fit_curve = np.exp(-fit_time / tau_fit)
        ax.plot(
            fit_time,
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


def analyze_replicate_job(
    epsilon: float,
    gsd_path: str,
    metadata_path: str,
    metadata: Dict,
    analysis_stride: int,
    max_lag_frames: int,
    msd_sample: int,
    msd_max_lag_frames: int,
    rep_label: str,
    rel_path: str,
    cache_root: str,
) -> Tuple[float, Dict]:
    cache_key = build_replicate_cache_key(
        gsd_path,
        metadata_path,
        analysis_stride,
        max_lag_frames,
        msd_sample,
        msd_max_lag_frames,
    )
    cache_path = os.path.join(cache_root, f"{cache_key}.pkl")
    cached = load_cached_replicate_result(cache_path)
    if cached is not None:
        log(f"{rep_label}: cache hit ({rel_path})")
        return epsilon, cached

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
    save_cached_replicate_result(cache_path, result)
    log(f"{rep_label}: done")
    return epsilon, result


def truncate_lag(
    time: np.ndarray | None, values: np.ndarray | None, max_lag: int
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if time is None or values is None:
        return None, None
    n = min(max_lag, len(time))
    return time[:n], values[:, :n]


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

    median_positions: List[float] = []
    median_values: List[float] = []
    for eps, values in zip(epsilon_values, data):
        if values.size == 0:
            continue
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color="#2b2b2b", lw=2.0)
        ax.scatter([eps], [med], color="#2b2b2b", s=18, zorder=3)
        median_positions.append(float(eps))
        median_values.append(med)

    if median_positions:
        ax.plot(
            median_positions,
            median_values,
            color="#2b2b2b",
            lw=1.8,
            alpha=0.9,
            zorder=2,
        )

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

    medians: List[float] = []
    for eps, values in zip(positions, violin_data):
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        med = float(np.median(values))
        ax.vlines(eps, q1, q3, color="#2b2b2b", lw=2.0)
        ax.scatter([eps], [med], color="#2b2b2b", s=18, zorder=3)
        medians.append(med)

    if positions:
        ax.plot(
            positions,
            medians,
            color="#2b2b2b",
            lw=1.8,
            alpha=0.9,
            zorder=2,
        )

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
    del left_label, right_label
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


def write_largest_network_vs_timestep_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    timestep_data: List[np.ndarray],
    size_mean_data: List[np.ndarray],
) -> None:
    series = []
    for eps, tvals, svals in zip(epsilon_values, timestep_data, size_mean_data):
        if tvals is None or svals is None:
            continue
        if len(tvals) == 0 or len(svals) == 0:
            continue
        series.append((float(eps), np.asarray(tvals), np.asarray(svals)))
    if not series:
        return

    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    denom = max(len(series) - 1, 1)
    for idx, (eps, tvals, svals) in enumerate(series):
        color = cmap(idx / denom)
        ax.plot(
            tvals,
            svals,
            color=color,
            lw=2.0,
            label=f"epsilon={eps:g}",
        )

    ax.set_title("Largest Network Size vs Timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Largest network size (chains)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_stress_modulus_by_epsilon_plot(
    path: str,
    epsilon_values: List[float],
    g_time_by_eps: Dict[float, np.ndarray],
    g_values_by_eps: Dict[float, np.ndarray],
    tau_r0: float,
) -> None:
    if not np.isfinite(tau_r0) or tau_r0 <= 0.0:
        tau_r0 = FALLBACK_TAU_R0

    series = []
    for eps in epsilon_values:
        lag_time = g_time_by_eps.get(eps)
        values = g_values_by_eps.get(eps)
        if lag_time is None or values is None:
            continue
        if len(lag_time) == 0 or values.size == 0:
            continue
        if values.ndim != 2 or values.shape[1] != len(lag_time):
            continue
        median = np.nanmedian(values, axis=0)
        stderr = np.zeros_like(median)
        for lag_idx in range(values.shape[1]):
            finite = values[np.isfinite(values[:, lag_idx]), lag_idx]
            if finite.size > 1:
                stderr[lag_idx] = float(np.std(finite, ddof=1) / np.sqrt(finite.size))
        series.append((eps, lag_time, median, stderr))

    if not series:
        return

    cmap = plt.get_cmap("plasma", len(series))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    max_plot_x = 0.0
    for idx, (eps, lag_time, median, stderr) in enumerate(series):
        color = cmap(idx)
        stop_mask = (~np.isfinite(median)) | (np.abs(median) <= stderr)
        stop_idx = np.flatnonzero(stop_mask)
        end_idx = int(stop_idx[0]) if stop_idx.size else int(median.size)
        if end_idx <= 0:
            continue

        x = lag_time[:end_idx] / tau_r0
        y = median[:end_idx]
        finite_positive = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if not np.any(finite_positive):
            continue

        ax.plot(
            x[finite_positive],
            y[finite_positive],
            color=color,
            lw=2.0,
            label=f"eps={eps:g}",
        )
        max_plot_x = max(max_plot_x, float(np.max(x[finite_positive])))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Stress Modulus vs Time")
    ax.set_xlabel(r"$t / \tau_R^0$")
    ax.set_ylabel(r"$G(t)$")
    if max_plot_x > 0.0:
        ax.set_xlim(right=min(MAX_ANALYSIS_LAG_TAU_R0, max_plot_x))
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log(f"Scanning trajectories under {args.input_root}")
    runs = discover_runs(args.input_root)
    if not runs:
        raise RuntimeError(f"No trajectories found under {args.input_root}")
    log(f"Discovered {len(runs)} trajectory/metadata pairs")

    tau_r0_reference = FALLBACK_TAU_R0

    # Group runs by epsilon
    grouped: Dict[float, List[Tuple[str, str, Dict]]] = defaultdict(list)
    for gsd_path, metadata_path in runs:
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"Missing metadata.json for {gsd_path}")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        epsilon = float(metadata["reactive_epsilon"])
        tau_r0 = float(metadata.get("tau_R0", float("nan")))
        if np.isfinite(tau_r0) and tau_r0 > 0.0:
            tau_r0_reference = tau_r0
        grouped[epsilon].append((gsd_path, metadata_path, metadata))
    log(f"Grouped runs into {len(grouped)} epsilon values")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_root = os.path.join(args.output_dir, ".replicate_cache")
    os.makedirs(cache_root, exist_ok=True)
    log(f"Using replicate cache directory {cache_root}")

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
    g_time_by_eps: Dict[float, np.ndarray] = {}
    g_values_by_eps: Dict[float, np.ndarray] = {}
    largest_network_timestep_data: List[np.ndarray] = []
    largest_network_size_mean_data: List[np.ndarray] = []
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
        for rep_idx, (gsd_path, metadata_path, metadata) in enumerate(entries, start=1):
            rel_path = os.path.relpath(gsd_path, args.input_root)
            rep_label = f"eps={epsilon:g} rep={rep_idx}/{len(entries)}"
            jobs.append(
                delayed(analyze_replicate_job)(
                    epsilon,
                    gsd_path,
                    metadata_path,
                    metadata,
                    args.analysis_stride,
                    args.max_lag_frames,
                    args.msd_sample,
                    args.msd_max_lag_frames,
                    rep_label,
                    rel_path,
                    cache_root,
                )
            )

    log(f"Starting parallel analysis with {n_jobs} workers on {len(jobs)} runs")
    # Keep parallelism at the replicate level; avoid nested worker pools inside
    # per-replicate analysis to prevent oversubscription under Slurm.
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
        cluster_values = np.vstack(cluster_p)
        cluster_mean = np.mean(cluster_values, axis=0)
        if len(cluster_p) > 1:
            cluster_stderr = np.std(cluster_values, axis=0, ddof=1) / np.sqrt(
                len(cluster_p)
            )
        else:
            cluster_stderr = np.zeros_like(cluster_mean)

        # Time series averages
        def aggregate_timeseries(key_time: str, key_val: str):
            series = [
                (res[key_time], res[key_val])
                for res in replicate_results
                if res[key_val] is not None
            ]
            if not series:
                return None, None, None, None
            n = min(min(len(t) for t, _ in series), min(len(v) for _, v in series))
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
        G_time, G_values, G_mean, G_stderr = aggregate_timeseries("G_time", "G_t")
        msd_time, _, msd_mean, msd_stderr = aggregate_timeseries("msd_time", "msd")
        (
            largest_network_timestep,
            _,
            largest_network_size_mean,
            largest_network_size_stderr,
        ) = aggregate_timeseries("largest_network_timestep", "largest_network_size")

        if G_time is not None and G_values is not None:
            g_time_by_eps[epsilon] = G_time
            g_values_by_eps[epsilon] = G_values

        largest_network_timestep_data.append(
            largest_network_timestep
            if largest_network_timestep is not None
            else np.array([], dtype=np.float64)
        )
        largest_network_size_mean_data.append(
            largest_network_size_mean
            if largest_network_size_mean is not None
            else np.array([], dtype=np.float64)
        )

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
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "bond_correlation_fit.png"),
                cs_time,
                cs_values,
                title=f"Bond Correlation Decay (eps={epsilon:g})",
                y_label="C_s(t)",
                use_semilog_linear_region=True,
            )
        cb_time_fit, cb_values_fit = truncate_lag(
            cb_time, cb_values, args.max_lag_frames
        )
        if cb_time_fit is not None and cb_values_fit is not None:
            write_autocorr_fit_plot(
                os.path.join(eps_dir, "open_correlation_fit.png"),
                cb_time_fit,
                cb_values_fit,
                title=f"Open-Sticker Correlation Decay (eps={epsilon:g})",
                y_label="Open-sticker persistence time",
                use_semilog_linear_region=True,
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
        if largest_network_timestep is not None and largest_network_size_mean is not None:
            write_timeseries(
                os.path.join(eps_dir, "largest_network_size.csv"),
                largest_network_timestep,
                largest_network_size_mean,
                largest_network_size_stderr,
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
        write_stress_modulus_by_epsilon_plot(
            os.path.join(args.output_dir, "stress_modulus_vs_epsilon_median_iqr.png"),
            epsilon_values,
            g_time_by_eps,
            g_values_by_eps,
            tau_r0_reference,
        )
        write_largest_network_vs_timestep_by_epsilon_plot(
            os.path.join(
                args.output_dir, "largest_network_size_vs_timestep_by_epsilon.png"
            ),
            epsilon_values,
            largest_network_timestep_data,
            largest_network_size_mean_data,
        )

    log("Analysis complete")


if __name__ == "__main__":
    main()
