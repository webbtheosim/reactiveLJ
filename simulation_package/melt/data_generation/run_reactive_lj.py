"""Run a single ReactiveLJ simulation replicate.

This script builds a KG melt (random-walk with rejection sampling), performs an
unsticky melt equilibration, assigns sticker identities, turns on the ReactiveLJ
interaction, equilibrates
again, and finally runs production while writing trajectory frames to GSD.

All key parameters are configurable via CLI flags. Keep this file as the
single source of truth for the Block 1 protocol, and adjust values as needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

import hoomd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
if SIM_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, SIM_PACKAGE_DIR)

try:
    import numba
except Exception as exc:  # pragma: no cover - required dependency
    raise RuntimeError("numba is required to run ReactiveLJ data generation.") from exc


@dataclass
class SimulationConfig:
    # System size and thermodynamic state
    n_chains: int = 4000
    chain_length: int = 40
    density: float = 0.85
    temperature: float = 1.0

    # KG bonded parameters
    fene_k: float = 30.0
    fene_r0: float = 1.5
    fene_epsilon: float = 1.0
    fene_sigma: float = 1.0
    k_bend: float = 1.5

    # Sticker placement
    stickers_per_chain: int = 4
    # Evenly spaced along the backbone; segment/offset are deprecated.
    segment_length: int = 10  # deprecated (unused)
    sticker_offset_in_segment: int = 4  # deprecated (unused)

    # Integrator
    dt: float = 0.005
    tau_T: float = 100.0
    zero_momentum_period: int = 100

    # Nonbonded baseline (WCA)
    lj_epsilon: float = 1.0
    lj_sigma: float = 1.0

    # ReactiveLJ parameters
    reactive_sigma: float = 1.0
    reactive_epsilon: float = 3.0
    reactive_r_cut: float | None = None
    weakening_inner: float | None = None
    weakening_outer: float | None = None
    weakening_exponent: float = 4.0
    smooth_elbow: bool = True
    smooth_kappa: float = 0.05
    smooth_beta: float = 1.0

    # Run lengths (steps)
    # NOTE: defaults are short for quick setup validation; increase for production.
    unsticky_equil_steps: int = 100_000
    reactive_equil_steps: int = 1_000_000
    production_steps: int = 1_000_000
    frame_steps: int = 10_000

    # Angle table
    angle_table_width: int = 1000

    # Neighbor list
    nlist_buffer: float = 0.4

    # Random-walk initialization controls.
    init_min_dist: float = 0.80
    init_bond_length: float = 0.97
    max_chain_attempts: int = 1000
    max_bead_attempts: int = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ReactiveLJ replicate of the KG melt."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="ReactiveLJ attraction strength (epsilon).",
    )
    parser.add_argument(
        "--replicate", type=int, required=True, help="Replicate index (1-based)."
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory for simulation outputs.",
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu", help="HOOMD device backend."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. Defaults to a deterministic value.",
    )
    parser.add_argument(
        "--init-min-dist",
        type=float,
        default=None,
        help=(
            "Minimum allowed non-bonded bead spacing during initialization "
            "(default 0.8)."
        ),
    )
    parser.add_argument(
        "--init-bond-length",
        type=float,
        default=None,
        help="Bond length for initial random walk (default 0.97).",
    )
    # Allow overrides for a few key run parameters
    parser.add_argument(
        "--frame-steps",
        type=int,
        default=None,
        help="GSD frame spacing in steps (default 10_000).",
    )
    parser.add_argument(
        "--unsticky-equil-steps",
        type=int,
        default=None,
        help="Equilibration steps for unsticky melt.",
    )
    parser.add_argument(
        "--reactive-equil-steps",
        type=int,
        default=None,
        help="Equilibration steps after enabling ReactiveLJ.",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=None,
        help="Production run length in steps.",
    )
    parser.add_argument(
        "--weakening-exponent",
        type=float,
        default=None,
        help="ReactiveLJ weakening exponent (default uses script config).",
    )
    return parser.parse_args()


def compute_box_length(n_particles: int, density: float) -> float:
    volume = n_particles / density
    return volume ** (1.0 / 3.0)


@numba.njit(cache=True)
def _seed_rng(seed: int) -> None:
    np.random.seed(seed)


@numba.njit(cache=True)
def _minimum_image(dx: float, box_length: float) -> float:
    return dx - box_length * np.float32(np.rint(dx / box_length))


@numba.njit(cache=True)
def _cell_index(
    pos: np.ndarray, box_length: float, cell_size: float, n_cells: int
) -> int:
    x = int(np.floor((pos[0] + 0.5 * box_length) / cell_size)) % n_cells
    y = int(np.floor((pos[1] + 0.5 * box_length) / cell_size)) % n_cells
    z = int(np.floor((pos[2] + 0.5 * box_length) / cell_size)) % n_cells
    return x + n_cells * (y + n_cells * z)


@numba.njit(cache=True)
def _is_valid_position_numba(
    pos: np.ndarray,
    local_positions: np.ndarray,
    n_local: int,
    bonded_index: int,
    global_positions: np.ndarray,
    head: np.ndarray,
    next_idx: np.ndarray,
    box_length: float,
    min_dist_sq: float,
    cell_size: float,
    n_cells: int,
) -> bool:
    for i in range(n_local):
        if bonded_index >= 0 and i == bonded_index:
            continue
        dx0 = _minimum_image(pos[0] - local_positions[i, 0], box_length)
        dx1 = _minimum_image(pos[1] - local_positions[i, 1], box_length)
        dx2 = _minimum_image(pos[2] - local_positions[i, 2], box_length)
        if dx0 * dx0 + dx1 * dx1 + dx2 * dx2 < min_dist_sq:
            return False

    cell_idx = _cell_index(pos, box_length, cell_size, n_cells)
    cx = cell_idx % n_cells
    cy = (cell_idx // n_cells) % n_cells
    cz = cell_idx // (n_cells * n_cells)
    for dx in (-1, 0, 1):
        nx = (cx + dx) % n_cells
        for dy in (-1, 0, 1):
            ny = (cy + dy) % n_cells
            for dz in (-1, 0, 1):
                nz = (cz + dz) % n_cells
                neighbor = nx + n_cells * (ny + n_cells * nz)
                idx = head[neighbor]
                while idx != -1:
                    dx0 = _minimum_image(pos[0] - global_positions[idx, 0], box_length)
                    dx1 = _minimum_image(pos[1] - global_positions[idx, 1], box_length)
                    dx2 = _minimum_image(pos[2] - global_positions[idx, 2], box_length)
                    if dx0 * dx0 + dx1 * dx1 + dx2 * dx2 < min_dist_sq:
                        return False
                    idx = next_idx[idx]
    return True


@numba.njit(cache=True)
def _random_unit_vector() -> np.ndarray:
    vec = np.empty(3, dtype=np.float32)
    vec[0] = np.random.normal()
    vec[1] = np.random.normal()
    vec[2] = np.random.normal()
    norm = np.float32(np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]))
    vec[0] /= norm
    vec[1] /= norm
    vec[2] /= norm
    return vec


@numba.njit(cache=True)
def _place_chain_numba(
    positions: np.ndarray,
    head: np.ndarray,
    next_idx: np.ndarray,
    start_index: int,
    chain_length: int,
    box_length: float,
    min_dist_sq: float,
    bond_length: float,
    cell_size: float,
    n_cells: int,
    max_chain_attempts: int,
    max_bead_attempts: int,
) -> tuple[bool, int]:
    attempts_used = 0
    half_box = np.float32(0.5) * box_length
    local_positions = np.empty((chain_length, 3), dtype=np.float32)
    for _ in range(max_chain_attempts):
        attempts_used += 1
        n_local = 0

        # First bead
        placed = False
        for _ in range(max_bead_attempts):
            pos = np.empty(3, dtype=np.float32)
            for k in range(3):
                pos[k] = (np.random.random() - 0.5) * box_length
            if _is_valid_position_numba(
                pos,
                local_positions,
                n_local,
                -1,
                positions,
                head,
                next_idx,
                box_length,
                min_dist_sq,
                cell_size,
                n_cells,
            ):
                local_positions[0, :] = pos
                n_local = 1
                placed = True
                break
        if not placed:
            continue

        prev = local_positions[0, :]
        for bead_idx in range(1, chain_length):
            placed = False
            for _ in range(max_bead_attempts):
                direction = _random_unit_vector()
                pos = prev + bond_length * direction
                pos = (pos + half_box) % box_length - half_box
                if _is_valid_position_numba(
                    pos,
                    local_positions,
                    n_local,
                    bead_idx - 1,
                    positions,
                    head,
                    next_idx,
                    box_length,
                    min_dist_sq,
                    cell_size,
                    n_cells,
                ):
                    local_positions[bead_idx, :] = pos
                    n_local += 1
                    prev = pos
                    placed = True
                    break
            if not placed:
                break

        if n_local == chain_length:
            # Commit chain
            for i in range(chain_length):
                idx = start_index + i
                positions[idx, :] = local_positions[i, :]
                cell = _cell_index(
                    local_positions[i, :], box_length, cell_size, n_cells
                )
                next_idx[idx] = head[cell]
                head[cell] = idx
            return True, attempts_used

    return False, attempts_used


def _build_random_walk_positions_numba(
    cfg: SimulationConfig,
    box_length: float,
    seed: int,
) -> np.ndarray:
    """Generate chain positions using Numba-accelerated rejection sampling."""
    if cfg.init_min_dist <= 0:
        raise ValueError("init_min_dist must be positive.")

    n_particles = cfg.n_chains * cfg.chain_length
    positions = np.zeros((n_particles, 3), dtype=np.float32)
    box_length_f = np.float32(box_length)
    cell_size = np.float32(cfg.init_min_dist)
    n_cells = max(1, int(box_length_f / cell_size))
    head = np.full(n_cells**3, -1, dtype=np.int32)
    next_idx = np.full(n_particles, -1, dtype=np.int32)
    min_dist_sq = np.float32(cfg.init_min_dist * cfg.init_min_dist)
    bond_length = np.float32(cfg.init_bond_length)

    _seed_rng(seed)

    placed_chains = 0
    total_chain_attempts = 0
    report_every = max(1, int(np.ceil(0.05 * cfg.n_chains)))

    for chain in range(cfg.n_chains):
        start_index = chain * cfg.chain_length
        success, attempts_used = _place_chain_numba(
            positions,
            head,
            next_idx,
            start_index,
            cfg.chain_length,
            box_length_f,
            min_dist_sq,
            bond_length,
            cell_size,
            n_cells,
            cfg.max_chain_attempts,
            cfg.max_bead_attempts,
        )
        if not success:
            raise RuntimeError(
                f"Failed to place chain {chain} after {cfg.max_chain_attempts} attempts."
            )

        placed_chains += 1
        total_chain_attempts += attempts_used
        if placed_chains % report_every == 0 or placed_chains == cfg.n_chains:
            fraction_accepted = placed_chains / total_chain_attempts
            print(
                "Initialization: placed "
                f"{placed_chains}/{cfg.n_chains} chains "
                f"(fraction_accepted={fraction_accepted:.4f})",
                flush=True,
            )

    print("Initialization: chain placement complete.", flush=True)
    return positions


def build_random_walk_positions(
    cfg: SimulationConfig,
    box_length: float,
    seed: int,
) -> np.ndarray:
    return _build_random_walk_positions_numba(cfg, box_length, seed)


def build_snapshot(
    cfg: SimulationConfig,
    seed: int,
    box_length: float,
) -> hoomd.Snapshot:
    n_particles = cfg.n_chains * cfg.chain_length

    positions = build_random_walk_positions(cfg, box_length, seed)

    # Bonds and angles for linear chains
    bonds = []
    angles = []
    for chain in range(cfg.n_chains):
        start = chain * cfg.chain_length
        end = start + cfg.chain_length
        for i in range(start, end - 1):
            bonds.append([i, i + 1])
        for i in range(start, end - 2):
            angles.append([i, i + 1, i + 2])

    bonds = np.array(bonds, dtype=np.int32)
    angles = np.array(angles, dtype=np.int32)

    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [box_length, box_length, box_length, 0, 0, 0]

        snap.particles.N = n_particles
        snap.particles.types = ["backbone", "sticky"]
        snap.particles.position[:] = positions
        snap.particles.typeid[:] = np.zeros(
            n_particles, dtype=np.int32
        )  # all backbone initially
        snap.particles.mass[:] = np.ones(n_particles, dtype=np.float32)
        snap.bonds.N = len(bonds)
        snap.bonds.types = ["FENE"]
        snap.bonds.typeid[:] = np.zeros(len(bonds), dtype=np.int32)
        snap.bonds.group[:] = bonds

        snap.angles.N = len(angles)
        snap.angles.types = ["bend"]
        snap.angles.typeid[:] = np.zeros(len(angles), dtype=np.int32)
        snap.angles.group[:] = angles

    return snap


def sticker_indices(cfg: SimulationConfig) -> np.ndarray:
    """Deterministically select evenly spaced sticker bead tags for each chain."""
    if cfg.stickers_per_chain <= 0:
        return np.array([], dtype=np.int32)

    segment = cfg.chain_length / cfg.stickers_per_chain
    offsets = np.rint((np.arange(cfg.stickers_per_chain) + 0.5) * segment).astype(
        np.int64
    )
    offsets = np.clip(offsets, 0, cfg.chain_length - 1)
    if np.unique(offsets).size != offsets.size:
        offsets = np.rint(
            np.linspace(0, cfg.chain_length - 1, cfg.stickers_per_chain + 2)[1:-1]
        ).astype(np.int64)
        offsets = np.clip(offsets, 0, cfg.chain_length - 1)
    if np.unique(offsets).size != offsets.size:
        raise RuntimeError(
            "sticker_indices could not generate unique offsets; adjust stickers_per_chain."
        )
    if np.any(np.diff(np.sort(offsets)) <= 1):
        raise RuntimeError(
            "sticker_indices produced adjacent stickers; adjust stickers_per_chain."
        )

    indices = []
    for chain in range(cfg.n_chains):
        chain_start = chain * cfg.chain_length
        indices.extend(chain_start + offsets)
    return np.array(indices, dtype=np.int32)


def set_stickers(sim: hoomd.Simulation, cfg: SimulationConfig) -> None:
    """Promote selected beads to the sticker type."""
    sticker_ids = sticker_indices(cfg)
    if sim.device.communicator.num_ranks != 1:
        raise RuntimeError("set_stickers requires a single-rank simulation.")
    with sim.state.cpu_local_snapshot as snap:
        tags = np.asarray(snap.particles.tag, dtype=np.int64)
        if tags.size == 0:
            raise RuntimeError("set_stickers found no particles in snapshot.")
        tag_to_index = np.full(tags.size, -1, dtype=np.int64)
        tag_to_index[tags] = np.arange(tags.size, dtype=np.int64)
        sticker_indices_local = tag_to_index[sticker_ids]
        if np.any(sticker_indices_local < 0):
            missing = sticker_ids[sticker_indices_local < 0]
            raise RuntimeError(
                f"set_stickers could not map tags to indices (missing tags: {missing[:8]})"
            )
        snap.particles.typeid[sticker_indices_local] = 1  # type "sticky"


def validate_stickers(sim: hoomd.Simulation, cfg: SimulationConfig) -> None:
    """Validate sticker assignment by tag and ensure no S-S bonded pairs."""
    sticker_ids = sticker_indices(cfg)
    if sim.device.communicator.num_ranks != 1:
        raise RuntimeError("validate_stickers requires a single-rank simulation.")

    expected = cfg.n_chains * cfg.stickers_per_chain
    if sticker_ids.size != expected:
        raise RuntimeError(
            f"validate_stickers expected {expected} stickers, got {sticker_ids.size}"
        )
    if np.unique(sticker_ids).size != sticker_ids.size:
        raise RuntimeError("validate_stickers found duplicate sticker tags.")

    with sim.state.cpu_local_snapshot as snap:
        tags = np.asarray(snap.particles.tag, dtype=np.int64)
        if tags.size == 0:
            raise RuntimeError("validate_stickers found no particles in snapshot.")
        tag_to_index = np.full(tags.size, -1, dtype=np.int64)
        tag_to_index[tags] = np.arange(tags.size, dtype=np.int64)
        sticker_indices_local = tag_to_index[sticker_ids]
        if np.any(sticker_indices_local < 0):
            missing = sticker_ids[sticker_indices_local < 0]
            raise RuntimeError(
                f"validate_stickers could not map tags to indices (missing tags: {missing[:8]})"
            )

        typeid = snap.particles.typeid
        if not np.all(typeid[sticker_indices_local] == 1):
            bad = sticker_ids[typeid[sticker_indices_local] != 1]
            raise RuntimeError(
                f"validate_stickers found non-sticker type for tags: {bad[:8]}"
            )

        if np.count_nonzero(typeid == 1) != expected:
            raise RuntimeError(
                f"validate_stickers expected {expected} S particles, "
                f"got {np.count_nonzero(typeid == 1)}"
            )

        bonds = np.asarray(snap.bonds.group, dtype=np.int64)
        if bonds.size:
            # bonds are stored as particle tags; map to local indices before checking
            bond_indices = tag_to_index[bonds]
            if np.any(bond_indices < 0):
                bad = bonds[bond_indices < 0]
                raise RuntimeError(
                    f"validate_stickers found bond tags not in snapshot: {bad[:5].tolist()}"
                )
            s_mask = typeid == 1
            s_bonds = s_mask[bond_indices[:, 0]] & s_mask[bond_indices[:, 1]]
            if np.any(s_bonds):
                bad_pairs = bonds[s_bonds][:5]
                raise RuntimeError(
                    "validate_stickers found S-S bonded pairs; sample tags: "
                    f"{bad_pairs.tolist()}"
                )


def report_min_ss_distance(sim: hoomd.Simulation) -> None:
    """Report the minimum S-S distance using a chunked minimum-image search."""
    if sim.device.communicator.num_ranks != 1:
        raise RuntimeError("report_min_ss_distance requires a single-rank simulation.")
    snap = sim.state.get_snapshot()
    if snap is None:
        print("Sticker diagnostics: snapshot unavailable.", flush=True)
        return
    pos = np.asarray(snap.particles.position, dtype=np.float64)
    typeid = np.asarray(snap.particles.typeid, dtype=np.int32)
    raw_tags = getattr(snap.particles, "tag", None)
    if raw_tags is None or len(raw_tags) == 0:
        tags = np.arange(pos.shape[0], dtype=np.int64)
    else:
        tags = np.asarray(raw_tags, dtype=np.int64)
    box = np.asarray(snap.configuration.box, dtype=np.float64)

    if pos.size == 0:
        print("Sticker diagnostics: no particles in snapshot.", flush=True)
        return

    s_mask = typeid == 1
    s_pos = pos[s_mask]
    s_tags = tags[s_mask]
    n_s = s_pos.shape[0]
    if n_s < 2:
        print(f"Sticker diagnostics: S count={n_s}, min_S_S=nan", flush=True)
        return

    Lx, Ly, Lz = box[0], box[1], box[2]
    if not (box[3] == 0.0 and box[4] == 0.0 and box[5] == 0.0):
        print(
            "Sticker diagnostics: triclinic box detected; min S-S distance is approximate.",
            flush=True,
        )

    min_dist2 = np.inf
    chunk = 512
    for i0 in range(0, n_s, chunk):
        i1 = min(n_s, i0 + chunk)
        a = s_pos[i0:i1]
        b = s_pos[i0:]
        d = a[:, None, :] - b[None, :, :]
        if Lx > 0:
            d[..., 0] -= Lx * np.rint(d[..., 0] / Lx)
        if Ly > 0:
            d[..., 1] -= Ly * np.rint(d[..., 1] / Ly)
        if Lz > 0:
            d[..., 2] -= Lz * np.rint(d[..., 2] / Lz)
        dist2 = np.einsum("ijk,ijk->ij", d, d)
        # mask self-distances in this block
        for k in range(i1 - i0):
            dist2[k, k] = np.inf
        local_min = dist2.min()
        if local_min < min_dist2:
            min_dist2 = local_min

    min_dist = float(np.sqrt(min_dist2)) if np.isfinite(min_dist2) else float("nan")
    print(
        f"Sticker diagnostics: S count={n_s}, min_S_S={min_dist:.6f}",
        flush=True,
    )


def make_angle_table(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate U(theta) = k_bend * (1 + cos(theta)) and its torque."""
    theta = np.linspace(0, np.pi, cfg.angle_table_width)
    # Straight chains (theta = pi) are the relaxed state.
    U = cfg.k_bend * (1.0 + np.cos(theta))
    tau = cfg.k_bend * np.sin(theta)
    return U.astype(np.float64), tau.astype(np.float64)


def write_metadata(
    path: str,
    cfg: SimulationConfig,
    epsilon: float,
    replicate: int,
    seed: int,
    target_box_length: float | None = None,
    initial_box_length: float | None = None,
) -> None:
    data = asdict(cfg)
    data.update(
        {
            "reactive_epsilon": epsilon,
            "replicate": replicate,
            "seed": seed,
            "n_particles": cfg.n_chains * cfg.chain_length,
        }
    )
    if target_box_length is not None:
        data["target_box_length"] = target_box_length
    if initial_box_length is not None:
        data["initial_box_length"] = initial_box_length
    data["init_min_dist"] = cfg.init_min_dist
    data["init_bond_length"] = cfg.init_bond_length
    data["max_chain_attempts"] = cfg.max_chain_attempts
    data["max_bead_attempts"] = cfg.max_bead_attempts
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def build_integrator(
    cfg: SimulationConfig, nlist: hoomd.md.nlist.NeighborList
) -> hoomd.md.Integrator:
    pair = hoomd.md.pair.LJ(nlist=nlist)
    wca_cut = 2 ** (1.0 / 6.0)

    pair.params[("backbone", "backbone")] = dict(
        epsilon=cfg.lj_epsilon, sigma=cfg.lj_sigma
    )
    pair.r_cut[("backbone", "backbone")] = wca_cut

    pair.params[("backbone", "sticky")] = dict(
        epsilon=cfg.lj_epsilon, sigma=cfg.lj_sigma
    )
    pair.r_cut[("backbone", "sticky")] = wca_cut

    # Keep sticker-sticker repulsion exclusively in ReactiveLJ to avoid double counting.
    pair.params[("sticky", "sticky")] = dict(epsilon=0.0, sigma=cfg.lj_sigma)
    pair.r_cut[("sticky", "sticky")] = wca_cut

    fene = hoomd.md.bond.FENEWCA()
    fene.params["FENE"] = dict(
        k=cfg.fene_k,
        r0=cfg.fene_r0,
        epsilon=cfg.fene_epsilon,
        sigma=cfg.fene_sigma,
        delta=0.0,
    )

    angle = hoomd.md.angle.Table(width=cfg.angle_table_width)
    U, tau = make_angle_table(cfg)
    angle.params["bend"] = dict(U=U, tau=tau)

    gamma = 1.0 / cfg.tau_T
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(),
        kT=cfg.temperature,
        default_gamma=gamma,
    )

    integrator = hoomd.md.Integrator(
        dt=cfg.dt,
        methods=[langevin],
        forces=[pair, fene, angle],
    )
    return integrator


def add_reactive_lj(
    integrator: hoomd.md.Integrator,
    nlist: hoomd.md.nlist.NeighborList,
    cfg: SimulationConfig,
    epsilon: float | None = None,
) -> hoomd.md.many_body.ReactiveLJ:
    if epsilon is None:
        epsilon = cfg.reactive_epsilon
    reactive = hoomd.md.many_body.ReactiveLJ(
        nlist=nlist,
        reactive_type="sticky",
        sigma=cfg.reactive_sigma,
        epsilon=epsilon,
        r_cut=cfg.reactive_r_cut,
        weakening_inner=cfg.weakening_inner,
        weakening_outer=cfg.weakening_outer,
        weakening_exponent=cfg.weakening_exponent,
        smooth_elbow=cfg.smooth_elbow,
        smooth_kappa=cfg.smooth_kappa,
        smooth_beta=cfg.smooth_beta,
    )
    integrator.forces.append(reactive)
    return reactive


def main() -> None:
    args = parse_args()

    cfg = SimulationConfig()
    cfg.reactive_epsilon = args.epsilon
    if args.weakening_exponent is not None:
        cfg.weakening_exponent = args.weakening_exponent

    if args.frame_steps is not None:
        cfg.frame_steps = args.frame_steps
    if args.init_min_dist is not None:
        cfg.init_min_dist = args.init_min_dist
    if args.init_bond_length is not None:
        cfg.init_bond_length = args.init_bond_length
    if args.unsticky_equil_steps is not None:
        cfg.unsticky_equil_steps = args.unsticky_equil_steps
    if args.reactive_equil_steps is not None:
        cfg.reactive_equil_steps = args.reactive_equil_steps
    if args.production_steps is not None:
        cfg.production_steps = args.production_steps

    seed = args.seed
    if seed is None:
        seed = int(10_000 * args.epsilon + args.replicate)

    output_dir = os.path.join(
        args.output_root, f"eps_{args.epsilon:g}", f"rep_{args.replicate:03d}"
    )
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.json")
    n_particles = cfg.n_chains * cfg.chain_length
    target_box_length = compute_box_length(n_particles, cfg.density)
    initial_box_length = target_box_length
    write_metadata(
        metadata_path,
        cfg,
        args.epsilon,
        args.replicate,
        seed,
        target_box_length=target_box_length,
        initial_box_length=initial_box_length,
    )

    device = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=seed)

    print(
        "HOOMD build:",
        f"source_dir={hoomd.version.source_dir}",
        f"git_sha1={hoomd.version.git_sha1}",
        f"compile_date={hoomd.version.compile_date}",
        f"gpu_enabled={hoomd.version.gpu_enabled}",
        f"gpu_platform={hoomd.version.gpu_platform}",
        flush=True,
    )
    print(f"many_body_py={hoomd.md.many_body.__file__}", flush=True)
    print(
        "ReactiveLJForceComputeGPU_present="
        f"{hasattr(hoomd.md._md, 'ReactiveLJForceComputeGPU')}",
        flush=True,
    )
    print(f"Device={sim.device}", flush=True)

    # Build the initial melt snapshot with random-walk + rejection sampling
    snapshot = build_snapshot(cfg, seed, initial_box_length)
    sim.create_state_from_snapshot(snapshot)

    # Integrator and updaters
    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    reactive_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(cfg, pair_nlist)
    sim.operations.integrator = integrator

    zero_momentum = hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(cfg.zero_momentum_period)
    )
    sim.operations.updaters.append(zero_momentum)

    # Thermalize velocities at the target temperature
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=cfg.temperature)

    # --- Unsticky equilibration stage (KG melt) ---
    if cfg.unsticky_equil_steps > 0:
        print(
            f"Stage=unsticky_equil start steps={cfg.unsticky_equil_steps}",
            flush=True,
        )
        sim.run(cfg.unsticky_equil_steps)
        print("Stage=unsticky_equil done", flush=True)

    # --- Enable stickers and ReactiveLJ ---
    print("Stage=enable_reactive start", flush=True)
    set_stickers(sim, cfg)
    validate_stickers(sim, cfg)
    report_min_ss_distance(sim)
    print("Stage=enable_reactive done", flush=True)

    # --- Reactive equilibration stage ---
    reactive = None
    if cfg.reactive_equil_steps > 0:
        print(
            f"Stage=reactive_equil start steps={cfg.reactive_equil_steps}",
            flush=True,
        )
        # ReactiveLJ requires epsilon > 0 (positive_real); start with a tiny value.
        start_eps = 1.0e-4
        end_eps = cfg.reactive_epsilon
        total_steps = cfg.reactive_equil_steps
        ramp_step = 1000
        n_segments = (total_steps + ramp_step - 1) // ramp_step
        reactive = None
        report_interval = max(1, n_segments // 20)
        ramp_dt = 1.0e-5
        integrator.dt = ramp_dt

        for segment in range(n_segments):
            if n_segments == 1:
                frac = 1.0
            else:
                frac = segment / (n_segments - 1)

            epsilon = start_eps + (end_eps - start_eps) * frac
            steps_this_segment = min(ramp_step, total_steps - segment * ramp_step)
            if reactive is not None:
                integrator.forces.remove(reactive)
            reactive = add_reactive_lj(integrator, reactive_nlist, cfg, epsilon=epsilon)

            if segment % report_interval == 0 or segment == n_segments - 1:
                progress_pct = 100.0 * (segment + 1) / n_segments
                print(
                    f"Stage=reactive_equil progress={progress_pct:.1f}% "
                    f"segment {segment + 1}/{n_segments} epsilon={epsilon:g} "
                    f"steps={steps_this_segment}",
                    flush=True,
                )
            sim.run(steps_this_segment)

        print("Stage=reactive_equil epsilon_ramp done", flush=True)

        # After the epsilon ramp, gradually restore the production timestep.
        dt_ramp_steps = 100_000
        dt_ramp_step = 1000
        dt_start = ramp_dt
        dt_end = cfg.dt
        n_dt_segments = (dt_ramp_steps + dt_ramp_step - 1) // dt_ramp_step
        dt_report_interval = max(1, n_dt_segments // 20)

        print(f"Stage=reactive_equil dt_ramp start steps={dt_ramp_steps}", flush=True)
        for segment in range(n_dt_segments):
            if n_dt_segments == 1:
                frac = 1.0
            else:
                frac = segment / (n_dt_segments - 1)

            dt_value = dt_start + (dt_end - dt_start) * frac
            integrator.dt = dt_value
            steps_this_segment = min(dt_ramp_step, dt_ramp_steps - segment * dt_ramp_step)

            if segment % dt_report_interval == 0 or segment == n_dt_segments - 1:
                progress_pct = 100.0 * (segment + 1) / n_dt_segments
                print(
                    f"Stage=reactive_equil dt_ramp progress={progress_pct:.1f}% "
                    f"segment {segment + 1}/{n_dt_segments} dt={dt_value:g} "
                    f"steps={steps_this_segment}",
                    flush=True,
                )
            sim.run(steps_this_segment)

        integrator.dt = dt_end
        print("Stage=reactive_equil dt_ramp done", flush=True)
        print("Stage=reactive_equil done", flush=True)
    else:
        reactive = add_reactive_lj(integrator, reactive_nlist, cfg)

    # --- Production stage ---
    print(
        f"Stage=production start steps={cfg.production_steps}",
        flush=True,
    )
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(thermo, quantities=["pressure_tensor"])
    logger.add(sim, quantities=["timestep"])

    gsd_path = os.path.join(output_dir, "trajectory.gsd")
    gsd_writer = hoomd.write.GSD(
        filename=gsd_path,
        trigger=hoomd.trigger.Periodic(cfg.frame_steps),
        mode="wb",
        filter=hoomd.filter.All(),
        # Keep trajectories compact: write only positions, images, and type ids.
        dynamic=["particles/position", "particles/image", "particles/typeid"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    production_start = time.perf_counter()
    sim.run(cfg.production_steps)
    print("Stage=production done", flush=True)
    production_elapsed = time.perf_counter() - production_start

    # Wallclock runtime reporting for production only.
    print(f"Production_runtime_seconds={production_elapsed:.2f}")
    print(f"Production_runtime_hours={production_elapsed / 3600.0:.3f}")


if __name__ == "__main__":
    main()
