#!/usr/bin/env python3
"""Run the small-system ReactiveLJ clumping sweep over weakening exponent p."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hoomd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_package.melt.data_generation.run_reactive_lj import (
    DEFAULT_REACTIVE_DT_RAMP_STEPS,
    SimulationConfig,
    add_reactive_lj,
    build_integrator,
    build_snapshot,
    compute_box_length,
    write_metadata,
)


FIXED_REACTIVE_EPSILON = 18.0
DEFAULT_N_CHAINS = 100
DEFAULT_CHAIN_LENGTH = 9
DEFAULT_STICKER_OFFSETS = (2, 6)
DEFAULT_UNSTICKY_EQUIL_STEPS = 100_000
DEFAULT_REACTIVE_EQUIL_STEPS = 1_000_000
DEFAULT_POST_ACTIVATION_EQUIL_STEPS = 1_000_000
DEFAULT_PRODUCTION_STEPS = 1_000_000
DEFAULT_FRAME_STEPS = 10_000


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate the small-melt ReactiveLJ clumping test data for a single "
            "weakening exponent p."
        )
    )
    parser.add_argument(
        "--p",
        type=int,
        required=True,
        help="ReactiveLJ weakening exponent p.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Replicate index (default: 1).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "outputs",
        help="Root directory for generated trajectories.",
    )
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="HOOMD execution backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed override.",
    )
    return parser.parse_args()


def custom_sticker_tags(cfg: SimulationConfig) -> np.ndarray:
    offsets = np.asarray(DEFAULT_STICKER_OFFSETS, dtype=np.int32)
    if cfg.chain_length <= int(offsets.max()):
        raise ValueError(
            "chain_length is too short for the requested sticker offsets "
            f"{tuple(int(value) for value in offsets)}"
        )
    tags = np.empty(cfg.n_chains * offsets.size, dtype=np.int32)
    for chain_idx in range(cfg.n_chains):
        start = chain_idx * cfg.chain_length
        begin = chain_idx * offsets.size
        tags[begin : begin + offsets.size] = start + offsets
    return tags


def set_custom_stickers(sim: hoomd.Simulation, sticker_tags: np.ndarray) -> None:
    if sim.device.communicator.num_ranks != 1:
        raise RuntimeError("set_custom_stickers requires a single-rank simulation.")
    with sim.state.cpu_local_snapshot as snap:
        tags = np.asarray(snap.particles.tag, dtype=np.int64)
        if tags.size == 0:
            raise RuntimeError("No particles found while assigning sticker tags.")
        tag_to_index = np.full(tags.size, -1, dtype=np.int64)
        tag_to_index[tags] = np.arange(tags.size, dtype=np.int64)
        local_indices = tag_to_index[np.asarray(sticker_tags, dtype=np.int64)]
        if np.any(local_indices < 0):
            missing = sticker_tags[local_indices < 0]
            raise RuntimeError(
                "Could not map sticker tags to local indices; missing tags: "
                f"{missing[:8].tolist()}"
            )
        snap.particles.typeid[local_indices] = 1


def validate_custom_stickers(
    sim: hoomd.Simulation,
    cfg: SimulationConfig,
    sticker_tags: np.ndarray,
) -> None:
    expected = cfg.n_chains * len(DEFAULT_STICKER_OFFSETS)
    if sticker_tags.size != expected:
        raise RuntimeError(
            f"Expected {expected} sticker tags, found {sticker_tags.size}."
        )
    if np.unique(sticker_tags).size != sticker_tags.size:
        raise RuntimeError("Duplicate sticker tags were generated.")

    if sim.device.communicator.num_ranks != 1:
        raise RuntimeError("validate_custom_stickers requires a single-rank simulation.")

    with sim.state.cpu_local_snapshot as snap:
        tags = np.asarray(snap.particles.tag, dtype=np.int64)
        if tags.size == 0:
            raise RuntimeError("No particles found while validating sticker tags.")
        tag_to_index = np.full(tags.size, -1, dtype=np.int64)
        tag_to_index[tags] = np.arange(tags.size, dtype=np.int64)
        local_indices = tag_to_index[np.asarray(sticker_tags, dtype=np.int64)]
        if np.any(local_indices < 0):
            missing = sticker_tags[local_indices < 0]
            raise RuntimeError(
                "Sticker validation could not map tags to local indices; missing "
                f"tags: {missing[:8].tolist()}"
            )

        typeid = np.asarray(snap.particles.typeid, dtype=np.int32)
        if not np.all(typeid[local_indices] == 1):
            bad = sticker_tags[typeid[local_indices] != 1]
            raise RuntimeError(
                "Expected all configured sticker tags to be type 'sticky'; bad "
                f"tags: {bad[:8].tolist()}"
            )

        if int(np.count_nonzero(typeid == 1)) != expected:
            raise RuntimeError(
                "Unexpected number of sticky particles after assignment: "
                f"{int(np.count_nonzero(typeid == 1))} != {expected}"
            )

        bonds = np.asarray(snap.bonds.group, dtype=np.int64)
        if bonds.size == 0:
            return
        bond_indices = tag_to_index[bonds]
        if np.any(bond_indices < 0):
            bad = bonds[bond_indices < 0]
            raise RuntimeError(
                "Bond validation found particle tags that do not exist in the "
                f"snapshot: {bad[:5].tolist()}"
            )
        sticky_mask = typeid == 1
        sticky_bonds = sticky_mask[bond_indices[:, 0]] & sticky_mask[bond_indices[:, 1]]
        if np.any(sticky_bonds):
            bad_pairs = bonds[sticky_bonds][:5]
            raise RuntimeError(
                "Found directly bonded sticky-sticky covalent pairs; sample tags: "
                f"{bad_pairs.tolist()}"
            )


def resolve_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return int(args.seed)
    return 180_000 + 1_000 * int(args.p) + int(args.replicate)


def run_reactive_activation(
    sim: hoomd.Simulation,
    integrator: hoomd.md.Integrator,
    reactive_nlist: hoomd.md.nlist.NeighborList,
    cfg: SimulationConfig,
) -> None:
    start_eps = 1.0e-4
    end_eps = cfg.reactive_epsilon
    ramp_step = 1_000
    total_steps = cfg.reactive_equil_steps
    n_segments = (total_steps + ramp_step - 1) // ramp_step
    report_interval = max(1, n_segments // 20)
    ramp_dt = 1.0e-5
    integrator.dt = ramp_dt

    reactive_force = None
    print(
        f"Stage=reactive_equil start steps={cfg.reactive_equil_steps} "
        f"epsilon_final={cfg.reactive_epsilon:g}",
        flush=True,
    )
    for segment in range(n_segments):
        frac = 1.0 if n_segments == 1 else segment / (n_segments - 1)
        epsilon = start_eps + (end_eps - start_eps) * frac
        steps_this_segment = min(ramp_step, total_steps - segment * ramp_step)
        if reactive_force is not None:
            integrator.forces.remove(reactive_force)
        reactive_force = add_reactive_lj(
            integrator,
            reactive_nlist,
            cfg,
            epsilon=epsilon,
        )
        if segment % report_interval == 0 or segment == n_segments - 1:
            progress_pct = 100.0 * (segment + 1) / n_segments
            print(
                f"Stage=reactive_equil epsilon_ramp progress={progress_pct:.1f}% "
                f"segment={segment + 1}/{n_segments} epsilon={epsilon:g}",
                flush=True,
            )
        sim.run(steps_this_segment)

    dt_ramp_step = 1_000
    dt_start = ramp_dt
    dt_end = cfg.dt
    n_dt_segments = (
        DEFAULT_REACTIVE_DT_RAMP_STEPS + dt_ramp_step - 1
    ) // dt_ramp_step
    dt_report_interval = max(1, n_dt_segments // 20)

    print(
        f"Stage=reactive_equil dt_ramp start steps={DEFAULT_REACTIVE_DT_RAMP_STEPS}",
        flush=True,
    )
    for segment in range(n_dt_segments):
        frac = 1.0 if n_dt_segments == 1 else segment / (n_dt_segments - 1)
        dt_value = dt_start + (dt_end - dt_start) * frac
        integrator.dt = dt_value
        steps_this_segment = min(
            dt_ramp_step,
            DEFAULT_REACTIVE_DT_RAMP_STEPS - segment * dt_ramp_step,
        )
        if segment % dt_report_interval == 0 or segment == n_dt_segments - 1:
            progress_pct = 100.0 * (segment + 1) / n_dt_segments
            print(
                f"Stage=reactive_equil dt_ramp progress={progress_pct:.1f}% "
                f"segment={segment + 1}/{n_dt_segments} dt={dt_value:g}",
                flush=True,
            )
        sim.run(steps_this_segment)

    integrator.dt = dt_end
    print("Stage=reactive_equil done", flush=True)


def write_run_metadata(
    metadata_path: Path,
    cfg: SimulationConfig,
    args: argparse.Namespace,
    seed: int,
    box_length: float,
    sticker_tags: np.ndarray,
) -> None:
    extra_metadata = {
        "clumping_test_p": int(args.p),
        "trajectory_particle_subset": "sticky_only",
        "trajectory_particle_count": int(sticker_tags.size),
        "trajectory_frame_steps": int(cfg.frame_steps),
        "trajectory_file": "trajectory.gsd",
        "post_activation_equil_steps": int(DEFAULT_POST_ACTIVATION_EQUIL_STEPS),
        "warmup_protocol": {
            "unsticky_equil_steps": int(cfg.unsticky_equil_steps),
            "reactive_equil_steps": int(cfg.reactive_equil_steps),
            "reactive_dt_ramp_steps": int(DEFAULT_REACTIVE_DT_RAMP_STEPS),
            "sticky_burnin_steps": 0,
        },
        "custom_sticker_offsets_zero_indexed": [
            int(offset) for offset in DEFAULT_STICKER_OFFSETS
        ],
        "run_status": "complete",
    }
    write_metadata(
        str(metadata_path),
        cfg,
        epsilon=cfg.reactive_epsilon,
        replicate=int(args.replicate),
        seed=seed,
        reactive_lj_enabled=True,
        target_box_length=float(box_length),
        initial_box_length=float(box_length),
        extra_metadata=extra_metadata,
    )


def main() -> int:
    args = parse_args()
    if args.p < 0:
        raise ValueError("--p must be non-negative.")
    if args.replicate <= 0:
        raise ValueError("--replicate must be positive.")

    cfg = SimulationConfig()
    cfg.n_chains = DEFAULT_N_CHAINS
    cfg.chain_length = DEFAULT_CHAIN_LENGTH
    cfg.stickers_per_chain = len(DEFAULT_STICKER_OFFSETS)
    cfg.reactive_epsilon = FIXED_REACTIVE_EPSILON
    cfg.weakening_exponent = float(args.p)
    cfg.unsticky_equil_steps = DEFAULT_UNSTICKY_EQUIL_STEPS
    cfg.reactive_equil_steps = DEFAULT_REACTIVE_EQUIL_STEPS
    cfg.sticky_burnin_steps = 0
    cfg.production_steps = DEFAULT_PRODUCTION_STEPS
    cfg.frame_steps = DEFAULT_FRAME_STEPS
    cfg.msd_particles = min(cfg.msd_particles, cfg.n_chains * cfg.chain_length)

    seed = resolve_seed(args)

    output_dir = args.output_root / f"p_{args.p}" / f"rep_{args.replicate:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    trajectory_path = output_dir / "trajectory.gsd"

    if trajectory_path.exists():
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if str(metadata.get("run_status", "")).lower() == "complete":
                print(
                    f"Stage=skip info=complete_existing_run output_dir={output_dir}",
                    flush=True,
                )
                return 0
        raise RuntimeError(
            f"Output trajectory already exists at {trajectory_path}; "
            "clear/archive the directory before rerunning."
        )

    n_particles = cfg.n_chains * cfg.chain_length
    box_length = compute_box_length(n_particles, cfg.density)
    sticker_tags = custom_sticker_tags(cfg)

    device = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=seed)
    snapshot = build_snapshot(cfg, seed, box_length)
    sim.create_state_from_snapshot(snapshot)

    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    reactive_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(cfg, pair_nlist, reactive_lj_enabled=True)
    sim.operations.integrator = integrator

    zero_momentum = hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(cfg.zero_momentum_period)
    )
    sim.operations.updaters.append(zero_momentum)

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=cfg.temperature)

    print(
        "Stage=initialization done "
        f"n_chains={cfg.n_chains} chain_length={cfg.chain_length} "
        f"stickers_per_chain={cfg.stickers_per_chain} p={args.p}",
        flush=True,
    )

    print(
        f"Stage=unsticky_equil start steps={cfg.unsticky_equil_steps}",
        flush=True,
    )
    sim.run(cfg.unsticky_equil_steps)
    print("Stage=unsticky_equil done", flush=True)

    print("Stage=enable_reactive start", flush=True)
    set_custom_stickers(sim, sticker_tags)
    validate_custom_stickers(sim, cfg, sticker_tags)
    print("Stage=enable_reactive done", flush=True)

    run_reactive_activation(sim, integrator, reactive_nlist, cfg)

    print(
        f"Stage=post_activation_equil start steps={DEFAULT_POST_ACTIVATION_EQUIL_STEPS}",
        flush=True,
    )
    sim.run(DEFAULT_POST_ACTIVATION_EQUIL_STEPS)
    print("Stage=post_activation_equil done", flush=True)

    trajectory_writer = hoomd.write.GSD(
        filename=str(trajectory_path),
        trigger=hoomd.trigger.Periodic(cfg.frame_steps),
        mode="wb",
        filter=hoomd.filter.Tags(sticker_tags.tolist()),
        dynamic=["particles/position"],
    )
    sim.operations.writers.append(trajectory_writer)

    print(
        f"Stage=production start steps={cfg.production_steps} "
        f"frame_steps={cfg.frame_steps} trajectory_subset=sticky_only",
        flush=True,
    )
    sim.run(cfg.production_steps)
    print("Stage=production done", flush=True)

    write_run_metadata(metadata_path, cfg, args, seed, box_length, sticker_tags)
    print(f"Wrote outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
