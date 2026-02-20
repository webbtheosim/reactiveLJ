#!/usr/bin/env python3
"""Run one ReactiveLJ replicate for energy-conservation validation.

This script mirrors the standard ReactiveLJ generation workflow but switches the
production stage to NVE and logs energy terms to the output trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import hoomd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
if SIM_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, SIM_PACKAGE_DIR)

from data_generation.run_reactive_lj import (
    SimulationConfig,
    add_reactive_lj,
    build_integrator,
    build_snapshot,
    compute_box_length,
    report_min_ss_distance,
    set_stickers,
    validate_stickers,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one ReactiveLJ replicate with NVE production and logged energies "
            "for conservation checks."
        )
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="ReactiveLJ attraction strength (epsilon).",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        required=True,
        help="Replicate index (1-based).",
    )
    parser.add_argument(
        "--output-root",
        default="../energy_conservation",
        help="Root directory for simulation outputs.",
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
        "--dt",
        type=float,
        default=None,
        help="Integrator timestep override for all stages.",
    )
    parser.add_argument(
        "--weakening-exponent",
        type=float,
        default=None,
        help="ReactiveLJ weakening exponent (default uses script config).",
    )
    return parser.parse_args()


def augment_metadata_for_energy_validation(path: str) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata["validation_mode"] = "energy_conservation"
    metadata["production_ensemble"] = "NVE"
    metadata["logged_energy_quantities"] = [
        "potential_energy",
        "kinetic_energy",
        "total_energy",
    ]

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


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
    if args.dt is not None:
        cfg.dt = args.dt

    seed = args.seed
    if seed is None:
        seed = int(30_000 * args.epsilon + args.replicate)

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
    augment_metadata_for_energy_validation(metadata_path)

    device = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=seed)

    print(f"Requested_epsilon={args.epsilon:g}", flush=True)
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

    snapshot = build_snapshot(cfg, seed, initial_box_length)
    sim.create_state_from_snapshot(snapshot)

    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    reactive_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(cfg, pair_nlist)
    sim.operations.integrator = integrator

    zero_momentum = hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(cfg.zero_momentum_period)
    )
    sim.operations.updaters.append(zero_momentum)

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=cfg.temperature)

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
    report_min_ss_distance(sim)
    print("Stage=enable_reactive done", flush=True)

    reactive = None
    if cfg.reactive_equil_steps > 0:
        print(
            f"Stage=reactive_equil start steps={cfg.reactive_equil_steps}",
            flush=True,
        )

        start_eps = 1.0e-4
        end_eps = cfg.reactive_epsilon
        total_steps = cfg.reactive_equil_steps
        ramp_step = 1000
        n_segments = (total_steps + ramp_step - 1) // ramp_step
        report_interval = max(1, n_segments // 20)
        ramp_dt = 1.0e-5
        integrator.dt = ramp_dt

        for segment in range(n_segments):
            frac = 1.0 if n_segments == 1 else segment / (n_segments - 1)
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

        dt_ramp_steps = 100_000
        dt_ramp_step = 1000
        dt_start = ramp_dt
        dt_end = cfg.dt
        n_dt_segments = (dt_ramp_steps + dt_ramp_step - 1) // dt_ramp_step
        dt_report_interval = max(1, n_dt_segments // 20)

        print(f"Stage=reactive_equil dt_ramp start steps={dt_ramp_steps}", flush=True)
        for segment in range(n_dt_segments):
            frac = 1.0 if n_dt_segments == 1 else segment / (n_dt_segments - 1)
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

    print("Stage=production switch_to_nve", flush=True)
    try:
        sim.operations.updaters.remove(zero_momentum)
    except ValueError:
        pass
    integrator.methods.clear()
    integrator.methods.append(hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All()))
    integrator.dt = cfg.dt

    print(
        f"Stage=production start steps={cfg.production_steps} ensemble=NVE",
        flush=True,
    )
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=["timestep"])
    logger.add(thermo, quantities=["potential_energy", "kinetic_energy"])
    logger[("energy", "total_energy")] = (
        lambda: thermo.potential_energy + thermo.kinetic_energy,
        "scalar",
    )

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

    production_start = time.perf_counter()
    sim.run(cfg.production_steps)
    gsd_writer.flush()
    print("Stage=production done", flush=True)
    production_elapsed = time.perf_counter() - production_start

    print(f"Production_runtime_seconds={production_elapsed:.2f}")
    print(f"Production_runtime_hours={production_elapsed / 3600.0:.3f}")


if __name__ == "__main__":
    main()
