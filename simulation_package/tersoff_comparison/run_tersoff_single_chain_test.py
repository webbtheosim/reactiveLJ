#!/usr/bin/env python3
"""Run a single-chain Tersoff test in a large dilute box.

This script mirrors the ReactiveLJ single-chain test workflow, but replaces the
ReactiveLJ sticker interaction with fitted Tersoff parameters selected by the
requested ReactiveLJ epsilon label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

import hoomd


@dataclass
class TestConfig:
    # System setup
    chain_length: int = 20
    box_length: float = 20.0
    stickers_per_chain: int = 4
    temperature: float = 1.0

    # KG bonded parameters
    fene_k: float = 30.0
    fene_r0: float = 1.5
    fene_epsilon: float = 1.0
    fene_sigma: float = 1.0
    k_bend: float = 1.0

    # Integrator
    dt: float = 0.005
    tau_T: float = 0.1
    zero_momentum_period: int = 100

    # Nonbonded baseline (WCA)
    lj_epsilon: float = 1.0
    lj_sigma: float = 1.0

    # Tersoff label (used to select fitted row)
    reactive_epsilon: float = 9.0

    # Runtime
    run_steps: int = 1_000_000
    frame_steps: int = 10_000

    # Neighbor list
    nlist_buffer: float = 0.4

    # Initial chain generation
    init_bond_length: float = 0.97
    init_min_dist: float = 0.80
    max_bead_attempts: int = 5000

    # Angle table
    angle_table_width: int = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-chain Tersoff test.")
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="ReactiveLJ epsilon label used to select fitted Tersoff parameters.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Replicate index (default 1).",
    )
    parser.add_argument(
        "--tersoff-params-csv",
        default=os.path.join(os.path.dirname(__file__), "outputs", "tersoff_fitted_params.csv"),
        help="CSV produced by find_tersoff_params.py.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/TEST_SINGLE_CHAIN",
        help="Root directory for test outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="HOOMD device backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. Defaults to deterministic value from epsilon+replicate.",
    )
    parser.add_argument(
        "--run-steps",
        type=int,
        default=None,
        help="Total simulation steps (default 1_000_000).",
    )
    parser.add_argument(
        "--frame-steps",
        type=int,
        default=None,
        help="GSD frame spacing in steps (default 10_000).",
    )
    return parser.parse_args()


def sticker_indices(cfg: TestConfig) -> np.ndarray:
    if cfg.stickers_per_chain <= 0:
        return np.array([], dtype=np.int32)

    segment = cfg.chain_length / cfg.stickers_per_chain
    offsets = np.rint((np.arange(cfg.stickers_per_chain) + 0.5) * segment).astype(
        np.int64
    )
    offsets = np.clip(offsets, 0, cfg.chain_length - 1)
    if np.unique(offsets).size != offsets.size:
        raise RuntimeError(
            "sticker_indices could not generate unique offsets; "
            "reduce stickers_per_chain."
        )
    return offsets.astype(np.int32)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return random_unit_vector(rng)
    return vec / norm


def build_single_chain_positions(cfg: TestConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = np.zeros((cfg.chain_length, 3), dtype=np.float32)
    half_box = 0.5 * cfg.box_length
    min_dist_sq = cfg.init_min_dist * cfg.init_min_dist

    positions[0] = rng.uniform(-half_box, half_box, size=3).astype(np.float32)

    for bead_idx in range(1, cfg.chain_length):
        placed = False
        for _ in range(cfg.max_bead_attempts):
            direction = random_unit_vector(rng)
            candidate = positions[bead_idx - 1] + cfg.init_bond_length * direction
            candidate = (candidate + half_box) % cfg.box_length - half_box

            if bead_idx > 1:
                prev = positions[:bead_idx]
                dx = candidate - prev
                dx -= cfg.box_length * np.rint(dx / cfg.box_length)
                dist_sq = np.einsum("ij,ij->i", dx, dx)
                mask = np.arange(bead_idx) != (bead_idx - 1)
                if np.any(dist_sq[mask] < min_dist_sq):
                    continue

            positions[bead_idx] = candidate.astype(np.float32)
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Failed to place bead {bead_idx} after {cfg.max_bead_attempts} attempts."
            )

    return positions


def make_angle_table(cfg: TestConfig) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, np.pi, cfg.angle_table_width)
    U = cfg.k_bend * (1.0 + np.cos(theta))
    tau = cfg.k_bend * np.sin(theta)
    return U.astype(np.float64), tau.astype(np.float64)


def build_snapshot(cfg: TestConfig, seed: int) -> hoomd.Snapshot:
    positions = build_single_chain_positions(cfg, seed)

    bonds = np.array([[i, i + 1] for i in range(cfg.chain_length - 1)], dtype=np.int32)
    angles = np.array(
        [[i, i + 1, i + 2] for i in range(cfg.chain_length - 2)], dtype=np.int32
    )

    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [cfg.box_length, cfg.box_length, cfg.box_length, 0, 0, 0]
        snap.particles.N = cfg.chain_length
        snap.particles.types = ["backbone", "sticky"]
        snap.particles.position[:] = positions
        snap.particles.typeid[:] = np.zeros(cfg.chain_length, dtype=np.int32)
        snap.particles.mass[:] = np.ones(cfg.chain_length, dtype=np.float32)

        s_idx = sticker_indices(cfg)
        if s_idx.size:
            snap.particles.typeid[s_idx] = 1

        snap.bonds.N = len(bonds)
        snap.bonds.types = ["FENE"]
        snap.bonds.typeid[:] = np.zeros(len(bonds), dtype=np.int32)
        snap.bonds.group[:] = bonds

        snap.angles.N = len(angles)
        snap.angles.types = ["bend"]
        snap.angles.typeid[:] = np.zeros(len(angles), dtype=np.int32)
        snap.angles.group[:] = angles

    return snap


def build_integrator(
    cfg: TestConfig, nlist_pair: hoomd.md.nlist.NeighborList
) -> hoomd.md.Integrator:
    pair = hoomd.md.pair.LJ(nlist=nlist_pair)
    wca_cut = 2 ** (1.0 / 6.0)

    pair.params[("backbone", "backbone")] = dict(
        epsilon=cfg.lj_epsilon, sigma=cfg.lj_sigma
    )
    pair.r_cut[("backbone", "backbone")] = wca_cut

    pair.params[("backbone", "sticky")] = dict(epsilon=cfg.lj_epsilon, sigma=cfg.lj_sigma)
    pair.r_cut[("backbone", "sticky")] = wca_cut

    # Sticky-sticky interaction is provided by Tersoff below.
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

    return hoomd.md.Integrator(
        dt=cfg.dt,
        methods=[langevin],
        forces=[pair, fene, angle],
    )


def _find_row_for_epsilon(params_csv: str, epsilon: float) -> dict[str, float]:
    if not os.path.exists(params_csv):
        raise FileNotFoundError(
            f"Tersoff parameter CSV not found: {params_csv}. "
            "Run find_tersoff_params.py first."
        )

    rows: list[dict[str, str]] = []
    with open(params_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows found in Tersoff parameter CSV: {params_csv}")

    best = min(rows, key=lambda row: abs(float(row["reactive_epsilon"]) - epsilon))
    if abs(float(best["reactive_epsilon"]) - epsilon) > 1e-8:
        raise RuntimeError(
            f"No exact fitted parameters found for epsilon={epsilon:g} in {params_csv}"
        )

    parsed: dict[str, float] = {}
    for key, value in best.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _as_tersoff_params(row: dict[str, float]) -> dict:
    return {
        "magnitudes": (row["A1"], row["A2"]),
        "exp_factors": (row["lambda1"], row["lambda2"]),
        "lambda3": row["lambda3"],
        "dimer_r": row["dimer_r"],
        "cutoff_thickness": row["cutoff_thickness"],
        "alpha": row["alpha"],
        "n": row["n"],
        "gamma": row["gamma"],
        "c": row["c"],
        "d": row["d"],
        "m": row["m"],
    }


def add_tersoff(
    integrator: hoomd.md.Integrator,
    nlist_tersoff: hoomd.md.nlist.NeighborList,
    params_row: dict[str, float],
) -> hoomd.md.many_body.Tersoff:
    r_cut = float(params_row.get("r_cut", 1.5))
    noninteractive_r_cut = 1.0e-6
    tersoff = hoomd.md.many_body.Tersoff(nlist=nlist_tersoff, default_r_cut=r_cut)

    sticky_params = _as_tersoff_params(params_row)
    noninteractive_params = dict(sticky_params)
    noninteractive_params["magnitudes"] = (0.0, 0.0)

    for pair in (("backbone", "backbone"), ("backbone", "sticky")):
        tersoff.params[pair] = noninteractive_params
        tersoff.r_cut[pair] = noninteractive_r_cut

    tersoff.params[("sticky", "sticky")] = sticky_params
    tersoff.r_cut[("sticky", "sticky")] = r_cut
    integrator.forces.append(tersoff)
    return tersoff


def write_metadata(
    path: str,
    cfg: TestConfig,
    replicate: int,
    seed: int,
    params_row: dict[str, float],
    params_csv: str,
) -> None:
    data = asdict(cfg)
    data.update(
        {
            "n_chains": 1,
            "replicate": replicate,
            "seed": seed,
            "n_particles": cfg.chain_length,
            "test_name": "single_chain_tersoff",
            "interaction_model": "tersoff",
            "tersoff_params_csv": os.path.abspath(params_csv),
            "tersoff_params": {
                "magnitudes": [params_row["A1"], params_row["A2"]],
                "exp_factors": [params_row["lambda1"], params_row["lambda2"]],
                "lambda3": params_row["lambda3"],
                "dimer_r": params_row["dimer_r"],
                "cutoff_thickness": params_row["cutoff_thickness"],
                "alpha": params_row["alpha"],
                "n": params_row["n"],
                "gamma": params_row["gamma"],
                "c": params_row["c"],
                "d": params_row["d"],
                "m": params_row["m"],
                "r_cut": params_row.get("r_cut", 1.5),
            },
        }
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    args = parse_args()

    cfg = TestConfig()
    cfg.reactive_epsilon = args.epsilon
    if args.run_steps is not None:
        cfg.run_steps = args.run_steps
    if args.frame_steps is not None:
        cfg.frame_steps = args.frame_steps

    params_row = _find_row_for_epsilon(args.tersoff_params_csv, args.epsilon)

    seed = args.seed
    if seed is None:
        seed = int(30_000 * args.epsilon + args.replicate)

    output_dir = os.path.join(args.output_root, f"eps_{args.epsilon:g}")
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    write_metadata(
        metadata_path,
        cfg,
        replicate=args.replicate,
        seed=seed,
        params_row=params_row,
        params_csv=args.tersoff_params_csv,
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
    print(f"Device={sim.device}", flush=True)
    print("Force_mode=Tersoff", flush=True)
    print(
        f"Requested_epsilon={args.epsilon:g} Replicate={args.replicate} "
        f"A1={params_row['A1']:.6g} A2={params_row['A2']:.6g}",
        flush=True,
    )

    snapshot = build_snapshot(cfg, seed)
    sim.create_state_from_snapshot(snapshot)

    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    tersoff_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(cfg, pair_nlist)
    add_tersoff(integrator, tersoff_nlist, params_row)
    sim.operations.integrator = integrator

    zero_momentum = hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(cfg.zero_momentum_period)
    )
    sim.operations.updaters.append(zero_momentum)

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
        dynamic=["particles/position", "particles/image", "particles/typeid"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=cfg.temperature)

    print(
        "Stage=single_chain_test start",
        f"steps={cfg.run_steps}",
        f"epsilon={cfg.reactive_epsilon:g}",
        flush=True,
    )
    start = time.perf_counter()
    sim.run(cfg.run_steps)
    elapsed = time.perf_counter() - start
    print("Stage=single_chain_test done", flush=True)
    print(f"Runtime_seconds={elapsed:.2f}", flush=True)
    print(f"Runtime_hours={elapsed / 3600.0:.3f}", flush=True)


if __name__ == "__main__":
    main()
