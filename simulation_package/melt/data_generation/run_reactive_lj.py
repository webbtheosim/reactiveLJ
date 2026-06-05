"""Run a single ReactiveLJ simulation replicate.

This script builds a KG melt (random-walk with rejection sampling), performs an
unsticky melt equilibration, assigns sticker identities, turns on the ReactiveLJ
interaction, performs a short force ramp followed by a real sticky burn-in, and
finally runs production while writing a sticker-only structural trajectory, a
separate sampled MSD trajectory, and a high-frequency virial-tensor log.
Production can checkpoint and resume across multiple jobs.

When ``reactive_epsilon <= 0``, the workflow automatically falls back to pure
WCA sticker-sticker interactions (no ReactiveLJ force). In that mode, sticky
beads are dynamically identical to backbone beads in nonbonded interactions.

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

import cupy as cp
import gsd.hoomd
import numba
import numpy as np

import hoomd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
if SIM_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, SIM_PACKAGE_DIR)

DEFAULT_TAU_R0 = 4041.0
DEFAULT_TARGET_PRODUCTION_TAU_R0 = 10.0
DEFAULT_VIRIAL_LOG_STEPS = 1000
DEFAULT_REACTIVE_DT_RAMP_STEPS = 100_000
DEFAULT_FRAME_STEPS = 100_000
DEFAULT_MSD_PARTICLES = 2000
DEFAULT_MSD_SAMPLE_SEED = 12345
DEFAULT_PRODUCTION_CHUNK_STEPS = 1_000_000
DEFAULT_WALLTIME_SAFETY_BUFFER_SECONDS = 60.0
EXIT_REQUEUE_REQUIRED = 3


class VirialTensorLogger(hoomd.custom.Action):
    """Log the configurational virial tensor without the kinetic contribution."""

    def __init__(self, sim: hoomd.Simulation):
        self._sim = sim
        if not isinstance(sim.device, hoomd.device.GPU):
            raise RuntimeError(
                "VirialTensorLogger requires hoomd.device.GPU because it reads "
                "gpu_local_force_arrays."
            )
        if sim.device.communicator.num_ranks != 1:
            raise RuntimeError(
                "VirialTensorLogger only supports single-rank runs when using "
                "gpu_local_force_arrays."
            )

    @hoomd.logging.log(category="sequence")
    def virial_tensor(self):
        integrator = self._sim.operations.integrator
        if integrator is None:
            raise RuntimeError("Integrator must be attached before logging virials.")

        total = cp.zeros(6, dtype=cp.float64)
        for force in integrator.forces:
            # Reduce the per-particle virials on device and transfer back only
            # the final 6-component tensor for GSD logging.
            with force.gpu_local_force_arrays as arrays:
                virials = arrays.virial
                if virials is not None:
                    total += cp.sum(cp.asarray(virials), axis=0, dtype=cp.float64)
            additional = force.additional_virial
            if additional is not None:
                total += cp.asarray(additional, dtype=cp.float64).reshape(6)

        return cp.asnumpy(total / float(self._sim.state.box.volume))

    def act(self, timestep):
        return None


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
    # Production steps are derived from the requested tau_R^0 target in main().
    unsticky_equil_steps: int = 100_000
    reactive_equil_steps: int = 1_000_000
    sticky_burnin_steps: int = 10_000_000
    production_steps: int = 0
    frame_steps: int = DEFAULT_FRAME_STEPS
    virial_log_steps: int = DEFAULT_VIRIAL_LOG_STEPS
    msd_particles: int = DEFAULT_MSD_PARTICLES

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
    parser.add_argument(
        "--n-chains",
        type=int,
        default=None,
        help="Number of polymer chains in the melt (default 4000).",
    )
    # Allow overrides for a few key run parameters
    parser.add_argument(
        "--frame-steps",
        type=int,
        default=None,
        help=f"GSD frame spacing in steps (default {DEFAULT_FRAME_STEPS:_d}).",
    )
    parser.add_argument(
        "--virial-log-steps",
        type=int,
        default=None,
        help=(
            "Virial-tensor log spacing in steps "
            f"(default {DEFAULT_VIRIAL_LOG_STEPS})."
        ),
    )
    parser.add_argument(
        "--msd-particles",
        type=int,
        default=None,
        help=(
            "Number of monomers to record in the sampled MSD trajectory "
            f"(default {DEFAULT_MSD_PARTICLES})."
        ),
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
        "--sticky-burnin-steps",
        type=int,
        default=None,
        help=(
            "Full-timestep sticky burn-in steps after the ReactiveLJ ramp "
            "(default 10,000,000 for epsilon > 0)."
        ),
    )
    parser.add_argument(
        "--production-runtime-tau-r0",
        type=float,
        default=None,
        help=(
            f"Production run length expressed in units of the unsticky Rouse time "
            f"tau_R^0={DEFAULT_TAU_R0:g} tau_LJ "
            f"(default {DEFAULT_TARGET_PRODUCTION_TAU_R0:g})."
        ),
    )
    parser.add_argument(
        "--weakening-exponent",
        type=float,
        default=None,
        help="ReactiveLJ weakening exponent (default uses script config).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.gsd in the output directory when available.",
    )
    parser.add_argument(
        "--walltime-limit-seconds",
        type=float,
        default=None,
        help=(
            "Soft walltime cap for the whole job in seconds. When set, production "
            "runs in chunks and exits with a requeue code after checkpointing "
            "before the cap is exceeded."
        ),
    )
    parser.add_argument(
        "--walltime-safety-buffer-seconds",
        type=float,
        default=DEFAULT_WALLTIME_SAFETY_BUFFER_SECONDS,
        help=(
            "Additional safety margin in seconds used when deciding whether "
            "another production chunk can fit before the walltime cap."
        ),
    )
    parser.add_argument(
        "--production-chunk-steps",
        type=int,
        default=DEFAULT_PRODUCTION_CHUNK_STEPS,
        help=(
            "Maximum number of MD steps to run per production chunk before "
            "checkpointing and walltime checks."
        ),
    )
    return parser.parse_args()


def compute_box_length(n_particles: int, density: float) -> float:
    volume = n_particles / density
    return volume ** (1.0 / 3.0)


def production_steps_for_tau_r0(runtime_tau_r0: float, dt: float) -> int:
    """Convert a target runtime in tau_R^0 into MD integration steps."""
    if runtime_tau_r0 <= 0.0:
        raise ValueError("production runtime in tau_R^0 must be positive")
    return int(np.ceil(runtime_tau_r0 * DEFAULT_TAU_R0 / dt))


def reactive_dt_ramp_steps(cfg: SimulationConfig, reactive_lj_enabled: bool) -> int:
    if reactive_lj_enabled and cfg.reactive_equil_steps > 0:
        return DEFAULT_REACTIVE_DT_RAMP_STEPS
    return 0


def total_reactive_stage_steps(
    cfg: SimulationConfig, reactive_lj_enabled: bool
) -> int:
    return cfg.reactive_equil_steps + reactive_dt_ramp_steps(cfg, reactive_lj_enabled)


def sticky_burnin_stage_steps(
    cfg: SimulationConfig, reactive_lj_enabled: bool
) -> int:
    if reactive_lj_enabled and cfg.sticky_burnin_steps > 0:
        return cfg.sticky_burnin_steps
    return 0


def pre_production_steps(cfg: SimulationConfig, reactive_lj_enabled: bool) -> int:
    return (
        cfg.unsticky_equil_steps
        + total_reactive_stage_steps(cfg, reactive_lj_enabled)
        + sticky_burnin_stage_steps(cfg, reactive_lj_enabled)
    )


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


def sample_msd_particle_tags(cfg: SimulationConfig) -> np.ndarray:
    """Select the deterministic particle subset used for MSD logging."""
    n_particles = cfg.n_chains * cfg.chain_length
    if cfg.msd_particles <= 0:
        raise ValueError("msd_particles must be positive.")
    if cfg.msd_particles >= n_particles:
        return np.arange(n_particles, dtype=np.int32)

    rng = np.random.default_rng(DEFAULT_MSD_SAMPLE_SEED)
    sample = rng.choice(n_particles, size=cfg.msd_particles, replace=False)
    return np.sort(sample.astype(np.int32, copy=False))


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
    reactive_lj_enabled: bool,
    target_box_length: float | None = None,
    initial_box_length: float | None = None,
    extra_metadata: dict | None = None,
) -> None:
    data = asdict(cfg)
    data.update(
        {
            "reactive_epsilon": epsilon,
            "reactive_lj_enabled": reactive_lj_enabled,
            "replicate": replicate,
            "seed": seed,
            "n_particles": cfg.n_chains * cfg.chain_length,
        }
    )
    if target_box_length is not None:
        data["target_box_length"] = target_box_length
    if initial_box_length is not None:
        data["initial_box_length"] = initial_box_length
    if extra_metadata:
        data.update(extra_metadata)
    data["init_min_dist"] = cfg.init_min_dist
    data["init_bond_length"] = cfg.init_bond_length
    data["max_chain_attempts"] = cfg.max_chain_attempts
    data["max_bead_attempts"] = cfg.max_bead_attempts
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def read_metadata(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _matches_float(
    metadata: dict, key: str, expected: float, mismatches: list[str]
) -> None:
    value = metadata.get(key)
    if value is None or not np.isclose(float(value), float(expected)):
        mismatches.append(f"{key}={value!r} (expected {expected!r})")


def _matches_int(metadata: dict, key: str, expected: int, mismatches: list[str]) -> None:
    value = metadata.get(key)
    if value is None or int(value) != int(expected):
        mismatches.append(f"{key}={value!r} (expected {expected!r})")


def validate_existing_metadata(
    metadata: dict,
    cfg: SimulationConfig,
    *,
    epsilon: float,
    replicate: int,
    seed: int,
    reactive_lj_enabled: bool,
) -> None:
    mismatches: list[str] = []
    _matches_float(metadata, "reactive_epsilon", epsilon, mismatches)
    _matches_int(metadata, "replicate", replicate, mismatches)
    _matches_int(metadata, "seed", seed, mismatches)
    _matches_int(metadata, "n_chains", cfg.n_chains, mismatches)
    _matches_int(metadata, "chain_length", cfg.chain_length, mismatches)
    _matches_int(
        metadata, "n_particles", cfg.n_chains * cfg.chain_length, mismatches
    )
    _matches_float(metadata, "density", cfg.density, mismatches)
    _matches_float(metadata, "temperature", cfg.temperature, mismatches)
    _matches_float(metadata, "dt", cfg.dt, mismatches)
    _matches_int(metadata, "frame_steps", cfg.frame_steps, mismatches)
    _matches_int(
        metadata, "trajectory_frame_steps", cfg.frame_steps, mismatches
    )
    _matches_int(metadata, "virial_log_steps", cfg.virial_log_steps, mismatches)
    _matches_int(metadata, "msd_particles", cfg.msd_particles, mismatches)
    _matches_int(
        metadata, "unsticky_equil_steps", cfg.unsticky_equil_steps, mismatches
    )
    _matches_int(
        metadata, "reactive_equil_steps", cfg.reactive_equil_steps, mismatches
    )
    _matches_int(
        metadata, "sticky_burnin_steps", cfg.sticky_burnin_steps, mismatches
    )
    _matches_int(metadata, "production_steps", cfg.production_steps, mismatches)
    _matches_float(
        metadata,
        "weakening_exponent",
        cfg.weakening_exponent,
        mismatches,
    )
    if bool(metadata.get("reactive_lj_enabled")) != bool(reactive_lj_enabled):
        mismatches.append(
            "reactive_lj_enabled="
            f"{metadata.get('reactive_lj_enabled')!r} "
            f"(expected {reactive_lj_enabled!r})"
        )
    if mismatches:
        raise RuntimeError(
            "Existing metadata does not match the requested run configuration: "
            + "; ".join(mismatches)
        )


def write_checkpoint(path: str, sim: hoomd.Simulation) -> None:
    hoomd.write.GSD.write(state=sim.state, filename=path, mode="wb")


def flush_output_writers(sim: hoomd.Simulation) -> None:
    for writer in sim.operations.writers:
        flush = getattr(writer, "flush", None)
        if flush is not None:
            flush()


def prune_gsd_after_step(path: str, max_step: int) -> int:
    """Remove frames later than max_step from a GSD file by rewriting it."""
    if not os.path.exists(path):
        return 0

    needs_prune = False
    with gsd.hoomd.open(path, "r") as traj:
        for frame in traj:
            if int(frame.configuration.step) > max_step:
                needs_prune = True
                break
    if not needs_prune:
        return 0

    tmp_path = f"{path}.prune_tmp"
    dropped = 0
    try:
        with gsd.hoomd.open(path, "r") as source, gsd.hoomd.open(
            tmp_path,
            "w",
        ) as pruned:
            for frame in source:
                if int(frame.configuration.step) <= max_step:
                    pruned.append(frame)
                else:
                    dropped += 1
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return dropped


def build_integrator(
    cfg: SimulationConfig,
    nlist: hoomd.md.nlist.NeighborList,
    reactive_lj_enabled: bool,
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

    if reactive_lj_enabled:
        # Keep sticker-sticker repulsion exclusively in ReactiveLJ to avoid
        # double counting.
        pair.params[("sticky", "sticky")] = dict(epsilon=0.0, sigma=cfg.lj_sigma)
    else:
        # epsilon <= 0 fallback: sticky beads use the same WCA as backbone beads.
        pair.params[("sticky", "sticky")] = dict(
            epsilon=cfg.lj_epsilon, sigma=cfg.lj_sigma
        )
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


def main() -> int:
    job_start = time.perf_counter()
    args = parse_args()

    cfg = SimulationConfig()
    cfg.reactive_epsilon = args.epsilon
    reactive_lj_enabled = cfg.reactive_epsilon > 0.0
    if args.n_chains is not None:
        if args.n_chains <= 0:
            raise ValueError("--n-chains must be positive")
        cfg.n_chains = args.n_chains
    if args.weakening_exponent is not None:
        cfg.weakening_exponent = args.weakening_exponent

    if args.frame_steps is not None:
        cfg.frame_steps = args.frame_steps
    if args.virial_log_steps is not None:
        cfg.virial_log_steps = args.virial_log_steps
    if args.msd_particles is not None:
        cfg.msd_particles = args.msd_particles
    if args.init_min_dist is not None:
        cfg.init_min_dist = args.init_min_dist
    if args.init_bond_length is not None:
        cfg.init_bond_length = args.init_bond_length
    if args.unsticky_equil_steps is not None:
        cfg.unsticky_equil_steps = args.unsticky_equil_steps
    if args.reactive_equil_steps is not None:
        cfg.reactive_equil_steps = args.reactive_equil_steps
    if args.sticky_burnin_steps is not None:
        cfg.sticky_burnin_steps = args.sticky_burnin_steps
    if args.production_chunk_steps <= 0:
        raise ValueError("--production-chunk-steps must be positive")
    if args.walltime_limit_seconds is not None and args.walltime_limit_seconds <= 0.0:
        raise ValueError("--walltime-limit-seconds must be positive when set")
    if args.walltime_safety_buffer_seconds < 0.0:
        raise ValueError("--walltime-safety-buffer-seconds must be non-negative")

    target_production_tau_r0 = DEFAULT_TARGET_PRODUCTION_TAU_R0
    if args.production_runtime_tau_r0 is not None:
        target_production_tau_r0 = float(args.production_runtime_tau_r0)
    cfg.production_steps = production_steps_for_tau_r0(
        target_production_tau_r0, cfg.dt
    )
    production_runtime_tau_r0 = cfg.production_steps * cfg.dt / DEFAULT_TAU_R0
    reactive_stage_steps = total_reactive_stage_steps(cfg, reactive_lj_enabled)
    sticky_burnin_steps = sticky_burnin_stage_steps(cfg, reactive_lj_enabled)
    pre_production_total_steps = pre_production_steps(cfg, reactive_lj_enabled)
    pre_production_runtime_tau_r0 = pre_production_total_steps * cfg.dt / DEFAULT_TAU_R0
    sticky_burnin_runtime_tau_r0 = sticky_burnin_steps * cfg.dt / DEFAULT_TAU_R0
    production_target_step = pre_production_total_steps + cfg.production_steps

    seed = args.seed
    if seed is None:
        seed = int(10_000 * args.epsilon + args.replicate)

    output_dir = os.path.join(
        args.output_root, f"eps_{args.epsilon:g}", f"rep_{args.replicate:03d}"
    )
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.json")
    checkpoint_path = os.path.join(output_dir, "checkpoint.gsd")
    gsd_path = os.path.join(output_dir, "trajectory.gsd")
    msd_gsd_path = os.path.join(output_dir, "msd_trajectory.gsd")
    virial_gsd_path = os.path.join(output_dir, "virial_tensor_log.gsd")

    n_particles = cfg.n_chains * cfg.chain_length
    sticker_tags = sticker_indices(cfg)
    msd_particle_tags = sample_msd_particle_tags(cfg)
    target_box_length = compute_box_length(n_particles, cfg.density)
    initial_box_length = target_box_length

    existing_metadata = read_metadata(metadata_path)
    if existing_metadata is not None:
        validate_existing_metadata(
            existing_metadata,
            cfg,
            epsilon=args.epsilon,
            replicate=args.replicate,
            seed=seed,
            reactive_lj_enabled=reactive_lj_enabled,
        )

    prior_cumulative_production_walltime_seconds = 0.0
    if existing_metadata is not None:
        prior_cumulative_production_walltime_seconds = float(
            existing_metadata.get("cumulative_production_walltime_seconds", 0.0)
        )

    output_files = (gsd_path, msd_gsd_path, virial_gsd_path)
    checkpoint_exists = os.path.exists(checkpoint_path)
    existing_outputs = any(os.path.exists(path) for path in output_files)
    existing_status = (
        str(existing_metadata.get("run_status", "")) if existing_metadata else ""
    )
    existing_completed_steps = (
        int(existing_metadata.get("production_completed_steps", 0))
        if existing_metadata is not None
        else 0
    )
    if existing_status == "complete" and existing_completed_steps >= cfg.production_steps:
        print(
            "Stage=resume info=run_already_complete "
            f"completed_steps={existing_completed_steps}",
            flush=True,
        )
        return 0

    if args.resume:
        if checkpoint_exists and existing_metadata is None:
            raise RuntimeError(
                f"Found {checkpoint_path} but metadata.json is missing; cannot resume safely."
            )
        resume = checkpoint_exists
        if not resume and (existing_metadata is not None or existing_outputs):
            raise RuntimeError(
                "Resume requested but checkpoint.gsd is missing while partial outputs "
                "already exist. Clear or archive the output directory before rerunning."
            )
    else:
        resume = False
        if checkpoint_exists or existing_metadata is not None or existing_outputs:
            raise RuntimeError(
                "Output directory already contains data. Use --resume to continue "
                "from a checkpoint, or clear/archive the existing outputs."
            )

    metadata_base = {
        "tau_R0": DEFAULT_TAU_R0,
        "production_runtime_tau_r0": production_runtime_tau_r0,
        "reactive_dt_ramp_steps": reactive_dt_ramp_steps(cfg, reactive_lj_enabled),
        "reactive_equil_total_steps": reactive_stage_steps,
        "sticky_burnin_total_steps": sticky_burnin_steps,
        "sticky_burnin_runtime_tau_r0": sticky_burnin_runtime_tau_r0,
        "pre_production_steps": pre_production_total_steps,
        "pre_production_runtime_tau_r0": pre_production_runtime_tau_r0,
        "production_target_final_timestep": production_target_step,
        "trajectory_particle_subset": "sticky_only",
        "trajectory_particle_count": int(sticker_tags.size),
        "trajectory_frame_steps": cfg.frame_steps,
        "structural_trajectory_file": "trajectory.gsd",
        "msd_particle_count": int(msd_particle_tags.size),
        "msd_particle_tags": msd_particle_tags.tolist(),
        "msd_frame_steps": cfg.frame_steps,
        "msd_trajectory_file": "msd_trajectory.gsd",
        "msd_sample_seed": DEFAULT_MSD_SAMPLE_SEED,
        "virial_log_file": "virial_tensor_log.gsd",
        "checkpoint_file": "checkpoint.gsd",
        "resume_requested": bool(args.resume),
        "job_walltime_limit_seconds": (
            float(args.walltime_limit_seconds)
            if args.walltime_limit_seconds is not None
            else None
        ),
        "job_walltime_safety_buffer_seconds": float(
            args.walltime_safety_buffer_seconds
        ),
        "production_chunk_steps": int(args.production_chunk_steps),
    }

    def update_run_metadata(
        run_status: str,
        checkpoint_timestep: int | None,
        cumulative_production_walltime_seconds: float,
        note: str | None = None,
    ) -> None:
        if checkpoint_timestep is None:
            production_completed_steps = 0
        else:
            production_completed_steps = max(
                0, int(checkpoint_timestep) - pre_production_total_steps
            )
        production_completed_steps = min(cfg.production_steps, production_completed_steps)
        production_remaining_steps = max(
            0, cfg.production_steps - production_completed_steps
        )
        extra_metadata = dict(metadata_base)
        extra_metadata.update(
            {
                "run_status": run_status,
                "checkpoint_timestep": (
                    int(checkpoint_timestep)
                    if checkpoint_timestep is not None
                    else None
                ),
                "checkpoint_exists": os.path.exists(checkpoint_path),
                "production_completed_steps": int(production_completed_steps),
                "production_remaining_steps": int(production_remaining_steps),
                "production_completed_tau_r0": (
                    production_completed_steps * cfg.dt / DEFAULT_TAU_R0
                ),
                "production_remaining_tau_r0": (
                    production_remaining_steps * cfg.dt / DEFAULT_TAU_R0
                ),
                "production_fraction_complete": (
                    production_completed_steps / cfg.production_steps
                ),
                "cumulative_production_walltime_seconds": float(
                    cumulative_production_walltime_seconds
                ),
                "cumulative_production_walltime_hours": float(
                    cumulative_production_walltime_seconds / 3600.0
                ),
            }
        )
        if note is not None:
            extra_metadata["run_note"] = note
        write_metadata(
            metadata_path,
            cfg,
            args.epsilon,
            args.replicate,
            seed,
            reactive_lj_enabled=reactive_lj_enabled,
            target_box_length=target_box_length,
            initial_box_length=initial_box_length,
            extra_metadata=extra_metadata,
        )

    update_run_metadata(
        run_status="resuming" if resume else "initializing",
        checkpoint_timestep=(
            int(existing_metadata.get("checkpoint_timestep"))
            if resume
            and existing_metadata is not None
            and existing_metadata.get("checkpoint_timestep") is not None
            else None
        ),
        cumulative_production_walltime_seconds=prior_cumulative_production_walltime_seconds,
        note="Starting from checkpoint." if resume else "Fresh run starting.",
    )

    device = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=seed)

    if resume:
        print(f"Stage=resume start checkpoint={checkpoint_path}", flush=True)
        sim.create_state_from_gsd(filename=checkpoint_path)
        print(f"Stage=resume done timestep={sim.timestep}", flush=True)
    else:
        snapshot = build_snapshot(cfg, seed, initial_box_length)
        sim.create_state_from_snapshot(snapshot)

    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    reactive_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(
        cfg,
        pair_nlist,
        reactive_lj_enabled=reactive_lj_enabled,
    )
    sim.operations.integrator = integrator

    zero_momentum = hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(cfg.zero_momentum_period)
    )
    sim.operations.updaters.append(zero_momentum)

    if resume:
        if reactive_lj_enabled:
            add_reactive_lj(integrator, reactive_nlist, cfg)
        if sim.timestep < pre_production_total_steps:
            raise RuntimeError(
                "Checkpoint predates completion of pre-production. This workflow "
                "only supports resuming from production checkpoints."
            )
        checkpoint_step = int(sim.timestep)
        for output_path in output_files:
            dropped_frames = prune_gsd_after_step(output_path, checkpoint_step)
            if dropped_frames > 0:
                print(
                    "Stage=resume pruned_output "
                    f"file={os.path.basename(output_path)} "
                    f"checkpoint_timestep={checkpoint_step} "
                    f"dropped_frames={dropped_frames}",
                    flush=True,
                )
    else:
        sim.state.thermalize_particle_momenta(
            filter=hoomd.filter.All(), kT=cfg.temperature
        )

        if cfg.unsticky_equil_steps > 0:
            print(
                f"Stage=unsticky_equil start steps={cfg.unsticky_equil_steps}",
                flush=True,
            )
            sim.run(cfg.unsticky_equil_steps)
            print("Stage=unsticky_equil done", flush=True)

        print("Stage=enable_reactive start", flush=True)
        set_stickers(sim, cfg)
        validate_stickers(sim, cfg)
        if not reactive_lj_enabled:
            print(
                "Stage=enable_reactive info=ReactiveLJ disabled (epsilon<=0); "
                "using WCA-only sticky-sticky interactions.",
                flush=True,
            )
        print("Stage=enable_reactive done", flush=True)

        if reactive_lj_enabled and cfg.reactive_equil_steps > 0:
            print(
                f"Stage=reactive_equil start steps={cfg.reactive_equil_steps}",
                flush=True,
            )
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
                reactive = add_reactive_lj(
                    integrator, reactive_nlist, cfg, epsilon=epsilon
                )

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

            dt_ramp_steps = DEFAULT_REACTIVE_DT_RAMP_STEPS
            dt_ramp_step = 1000
            dt_start = ramp_dt
            dt_end = cfg.dt
            n_dt_segments = (dt_ramp_steps + dt_ramp_step - 1) // dt_ramp_step
            dt_report_interval = max(1, n_dt_segments // 20)

            print(
                f"Stage=reactive_equil dt_ramp start steps={dt_ramp_steps}",
                flush=True,
            )
            for segment in range(n_dt_segments):
                if n_dt_segments == 1:
                    frac = 1.0
                else:
                    frac = segment / (n_dt_segments - 1)

                dt_value = dt_start + (dt_end - dt_start) * frac
                integrator.dt = dt_value
                steps_this_segment = min(
                    dt_ramp_step, dt_ramp_steps - segment * dt_ramp_step
                )

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
        elif reactive_lj_enabled:
            add_reactive_lj(integrator, reactive_nlist, cfg)
        elif cfg.reactive_equil_steps > 0:
            print(
                "Stage=reactive_equil start "
                f"steps={cfg.reactive_equil_steps} mode=WCA_fallback",
                flush=True,
            )
            sim.run(cfg.reactive_equil_steps)
            print("Stage=reactive_equil done mode=WCA_fallback", flush=True)

        if reactive_lj_enabled and cfg.sticky_burnin_steps > 0:
            print(
                f"Stage=sticky_burnin start steps={cfg.sticky_burnin_steps} "
                f"runtime_tau_R0={cfg.sticky_burnin_steps * cfg.dt / DEFAULT_TAU_R0:.3f}",
                flush=True,
            )
            sim.run(cfg.sticky_burnin_steps)
            print("Stage=sticky_burnin done", flush=True)

        write_checkpoint(checkpoint_path, sim)
        update_run_metadata(
            run_status="checkpointed",
            checkpoint_timestep=int(sim.timestep),
            cumulative_production_walltime_seconds=prior_cumulative_production_walltime_seconds,
            note="Pre-production, including sticky burn-in, complete; initial checkpoint written.",
        )

    initial_production_completed_steps = max(0, int(sim.timestep) - pre_production_total_steps)
    if initial_production_completed_steps > 0:
        missing_outputs = [
            os.path.basename(path) for path in output_files if not os.path.exists(path)
        ]
        if missing_outputs:
            raise RuntimeError(
                "Cannot resume production because output files are missing: "
                + ", ".join(missing_outputs)
            )

    if sim.timestep >= production_target_step:
        update_run_metadata(
            run_status="complete",
            checkpoint_timestep=int(sim.timestep),
            cumulative_production_walltime_seconds=prior_cumulative_production_walltime_seconds,
            note="Run reached the target timestep before entering production.",
        )
        print("Stage=production done info=already_at_target", flush=True)
        return 0

    print(
        f"Stage=production start steps={cfg.production_steps} "
        f"runtime_tau_R0={production_runtime_tau_r0:.3f} "
        f"completed_steps={initial_production_completed_steps} "
        f"remaining_steps={production_target_step - int(sim.timestep)} "
        f"trajectory_subset=sticky_only "
        f"msd_frame_dt_tau_LJ={cfg.dt * cfg.frame_steps:.1f} "
        f"msd_particles={msd_particle_tags.size}",
        flush=True,
    )

    sim.always_compute_pressure = False

    virial_tensor_logger = VirialTensorLogger(sim)
    virial_logger = hoomd.logging.Logger()
    virial_logger.add(virial_tensor_logger, quantities=["virial_tensor"])

    append_mode = "ab" if initial_production_completed_steps > 0 else "wb"
    trajectory_filter = hoomd.filter.Tags(sticker_tags.tolist())
    trajectory_dynamic = ["particles/position"]

    gsd_writer = hoomd.write.GSD(
        filename=gsd_path,
        trigger=hoomd.trigger.Periodic(cfg.frame_steps),
        mode=append_mode,
        filter=trajectory_filter,
        dynamic=trajectory_dynamic,
    )
    sim.operations.writers.append(gsd_writer)

    msd_writer = hoomd.write.GSD(
        filename=msd_gsd_path,
        trigger=hoomd.trigger.Periodic(cfg.frame_steps),
        mode=append_mode,
        filter=hoomd.filter.Tags(msd_particle_tags.tolist()),
        dynamic=["particles/position", "particles/image"],
    )
    sim.operations.writers.append(msd_writer)

    virial_writer = hoomd.write.GSD(
        filename=virial_gsd_path,
        trigger=hoomd.trigger.Periodic(cfg.virial_log_steps),
        mode=append_mode,
        filter=hoomd.filter.Null(),
        dynamic=[],
        logger=virial_logger,
    )
    sim.operations.writers.append(virial_writer)

    production_elapsed = 0.0
    last_chunk_elapsed = 0.0
    chunk_index = 0

    while sim.timestep < production_target_step:
        current_timestep = int(sim.timestep)
        remaining_production_steps = production_target_step - current_timestep
        elapsed_job_seconds = time.perf_counter() - job_start
        if args.walltime_limit_seconds is not None:
            remaining_walltime_seconds = (
                args.walltime_limit_seconds - elapsed_job_seconds
            )
            required_margin_seconds = (
                args.walltime_safety_buffer_seconds + last_chunk_elapsed
            )
            if remaining_walltime_seconds <= required_margin_seconds:
                flush_output_writers(sim)
                write_checkpoint(checkpoint_path, sim)
                cumulative_walltime_seconds = (
                    prior_cumulative_production_walltime_seconds + production_elapsed
                )
                update_run_metadata(
                    run_status="checkpointed",
                    checkpoint_timestep=current_timestep,
                    cumulative_production_walltime_seconds=cumulative_walltime_seconds,
                    note=(
                        "Stopped before walltime cap; requeue required to finish "
                        f"{remaining_production_steps} remaining production steps."
                    ),
                )
                print(
                    "Stage=production checkpoint "
                    f"remaining_steps={remaining_production_steps} "
                    f"remaining_walltime_seconds={remaining_walltime_seconds:.1f} "
                    f"required_margin_seconds={required_margin_seconds:.1f}",
                    flush=True,
                )
                print(f"Production_runtime_seconds={production_elapsed:.2f}")
                print(f"Production_runtime_hours={production_elapsed / 3600.0:.3f}")
                return EXIT_REQUEUE_REQUIRED

        steps_this_chunk = min(args.production_chunk_steps, remaining_production_steps)
        chunk_index += 1
        chunk_start = time.perf_counter()
        sim.run(steps_this_chunk)
        chunk_elapsed = time.perf_counter() - chunk_start
        production_elapsed += chunk_elapsed
        last_chunk_elapsed = chunk_elapsed

        flush_output_writers(sim)
        write_checkpoint(checkpoint_path, sim)
        current_timestep = int(sim.timestep)
        current_completed_steps = current_timestep - pre_production_total_steps
        current_remaining_steps = max(0, production_target_step - current_timestep)
        cumulative_walltime_seconds = (
            prior_cumulative_production_walltime_seconds + production_elapsed
        )
        update_run_metadata(
            run_status=(
                "complete"
                if current_timestep >= production_target_step
                else "checkpointed"
            ),
            checkpoint_timestep=current_timestep,
            cumulative_production_walltime_seconds=cumulative_walltime_seconds,
            note=(
                None
                if current_timestep >= production_target_step
                else f"Checkpoint after production chunk {chunk_index}."
            ),
        )
        print(
            f"Stage=production progress={100.0 * current_completed_steps / cfg.production_steps:.2f}% "
            f"chunk={chunk_index} steps={steps_this_chunk} "
            f"chunk_seconds={chunk_elapsed:.2f} "
            f"completed_steps={current_completed_steps} "
            f"remaining_steps={current_remaining_steps}",
            flush=True,
        )

    print("Stage=production done", flush=True)
    print(f"Production_runtime_seconds={production_elapsed:.2f}")
    print(f"Production_runtime_hours={production_elapsed / 3600.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
