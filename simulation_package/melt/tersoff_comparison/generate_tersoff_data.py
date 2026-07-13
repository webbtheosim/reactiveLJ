#!/usr/bin/env python3
"""Run one KG melt replicate using fitted Liu/O'Connor Tersoff interactions."""

from __future__ import annotations

import argparse
import csv
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
    build_integrator,
    build_snapshot,
    compute_box_length,
    set_stickers,
    validate_stickers,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Liu/O'Connor Tersoff replicate corresponding to a "
            "ReactiveLJ epsilon value."
        )
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="ReactiveLJ epsilon label used to select fitted Tersoff parameters.",
    )
    parser.add_argument(
        "--replicate", type=int, required=True, help="Replicate index (1-based)."
    )
    parser.add_argument(
        "--tersoff-params-csv",
        default=os.path.join(
            os.path.dirname(__file__), "outputs", "tersoff_fitted_params.csv"
        ),
        help="CSV produced by find_tersoff_params.py.",
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
        help="Minimum allowed non-bonded spacing during initialization.",
    )
    parser.add_argument(
        "--init-bond-length",
        type=float,
        default=None,
        help="Bond length for initial random walk.",
    )
    parser.add_argument(
        "--frame-steps",
        type=int,
        default=None,
        help="GSD frame spacing in steps.",
    )
    parser.add_argument(
        "--unsticky-equil-steps",
        type=int,
        default=None,
        help="Equilibration steps for unsticky melt.",
    )
    parser.add_argument(
        "--tersoff-equil-steps",
        "--reactive-equil-steps",
        dest="tersoff_equil_steps",
        type=int,
        default=None,
        help="Equilibration steps after enabling Tersoff interactions.",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=None,
        help="Production run length in steps.",
    )
    return parser.parse_args()


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


def _validate_liu_o_connor_subset(row: dict[str, float]) -> None:
    if abs(row.get("lambda3", 0.0)) > 1.0e-12:
        raise ValueError("optimized Liu/O'Connor Tersoff requires lambda3=0")
    if abs(row.get("c", 0.0)) > 1.0e-12:
        raise ValueError("optimized Liu/O'Connor Tersoff requires c=0 so g(theta)=1")


def _add_liu_o_connor_tersoff(
    integrator: hoomd.md.Integrator,
    nlist: hoomd.md.nlist.NeighborList,
    row: dict[str, float],
    initial_scale: float,
) -> hoomd.md.many_body.LiuOConnorTersoff:
    _validate_liu_o_connor_subset(row)
    force = hoomd.md.many_body.LiuOConnorTersoff(
        nlist=nlist,
        sticky_type="sticky",
        A1=initial_scale * row["A1"],
        A2=initial_scale * row["A2"],
        lambda1=row["lambda1"],
        lambda2=row["lambda2"],
        dimer_r=row["dimer_r"],
        cutoff_thickness=row["cutoff_thickness"],
        r_cut=float(row.get("r_cut", 1.5)),
        alpha=row["alpha"],
        n=row["n"],
        gamma=row["gamma"],
    )
    integrator.forces.append(force)
    return force


def _augment_metadata(
    path: str,
    params_row: dict[str, float],
    params_csv: str,
) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata["interaction_model"] = "liu_o_connor_tersoff"
    metadata["tersoff_params_csv"] = os.path.abspath(params_csv)
    metadata["tersoff_params"] = {
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
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()

    cfg = SimulationConfig()
    if args.frame_steps is not None:
        cfg.frame_steps = args.frame_steps
    if args.init_min_dist is not None:
        cfg.init_min_dist = args.init_min_dist
    if args.init_bond_length is not None:
        cfg.init_bond_length = args.init_bond_length
    if args.unsticky_equil_steps is not None:
        cfg.unsticky_equil_steps = args.unsticky_equil_steps
    if args.tersoff_equil_steps is not None:
        cfg.reactive_equil_steps = args.tersoff_equil_steps
    if args.production_steps is not None:
        cfg.production_steps = args.production_steps

    seed = args.seed
    if seed is None:
        seed = int(20_000 * args.epsilon + args.replicate)

    output_dir = os.path.join(
        args.output_root, f"eps_{args.epsilon:g}", f"rep_{args.replicate:03d}"
    )
    os.makedirs(output_dir, exist_ok=True)

    params_row = _find_row_for_epsilon(args.tersoff_params_csv, args.epsilon)

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
        reactive_lj_enabled=False,
        target_box_length=target_box_length,
        initial_box_length=initial_box_length,
    )
    _augment_metadata(
        metadata_path,
        params_row,
        args.tersoff_params_csv,
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
        "LiuOConnorTersoffForceComputeGPU_present="
        f"{hasattr(hoomd.md._md, 'LiuOConnorTersoffForceComputeGPU')}",
        flush=True,
    )
    print("Interaction_model=liu_o_connor_tersoff", flush=True)
    print(f"Device={sim.device}", flush=True)
    print(
        f"Requested_epsilon={args.epsilon:g} Replicate={args.replicate}",
        flush=True,
    )

    snapshot = build_snapshot(cfg, seed, initial_box_length)
    sim.create_state_from_snapshot(snapshot)

    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    tersoff_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(cfg, pair_nlist, reactive_lj_enabled=True)
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

    print("Stage=enable_tersoff start", flush=True)
    set_stickers(sim, cfg)
    validate_stickers(sim, cfg)
    print("Stage=enable_tersoff done", flush=True)

    tersoff = None

    if cfg.reactive_equil_steps > 0:
        print(
            f"Stage=tersoff_equil start steps={cfg.reactive_equil_steps}",
            flush=True,
        )

        total_steps = cfg.reactive_equil_steps
        ramp_step = 1000
        n_segments = (total_steps + ramp_step - 1) // ramp_step
        report_interval = max(1, n_segments // 20)

        ramp_dt = 1.0e-5
        integrator.dt = ramp_dt

        for segment in range(n_segments):
            frac = 1.0 if n_segments == 1 else segment / (n_segments - 1)
            scale = 1.0e-4 + (1.0 - 1.0e-4) * frac
            steps_this_segment = min(ramp_step, total_steps - segment * ramp_step)

            if tersoff is not None:
                integrator.forces.remove(tersoff)
            tersoff = _add_liu_o_connor_tersoff(
                integrator=integrator,
                nlist=tersoff_nlist,
                row=params_row,
                initial_scale=scale,
            )

            if segment % report_interval == 0 or segment == n_segments - 1:
                mags = (scale * params_row["A1"], scale * params_row["A2"])
                progress_pct = 100.0 * (segment + 1) / n_segments
                print(
                    f"Stage=tersoff_equil progress={progress_pct:.1f}% "
                    f"segment {segment + 1}/{n_segments} "
                    f"A1={mags[0]:g} A2={mags[1]:g} steps={steps_this_segment}",
                    flush=True,
                )
            sim.run(steps_this_segment)

        print("Stage=tersoff_equil ramp done", flush=True)

        dt_ramp_steps = 100_000
        dt_ramp_step = 1000
        dt_start = ramp_dt
        dt_end = cfg.dt
        n_dt_segments = (dt_ramp_steps + dt_ramp_step - 1) // dt_ramp_step
        dt_report_interval = max(1, n_dt_segments // 20)

        print(f"Stage=tersoff_equil dt_ramp start steps={dt_ramp_steps}", flush=True)
        for segment in range(n_dt_segments):
            frac = 1.0 if n_dt_segments == 1 else segment / (n_dt_segments - 1)
            dt_value = dt_start + (dt_end - dt_start) * frac
            integrator.dt = dt_value
            steps_this_segment = min(dt_ramp_step, dt_ramp_steps - segment * dt_ramp_step)

            if segment % dt_report_interval == 0 or segment == n_dt_segments - 1:
                progress_pct = 100.0 * (segment + 1) / n_dt_segments
                print(
                    f"Stage=tersoff_equil dt_ramp progress={progress_pct:.1f}% "
                    f"segment {segment + 1}/{n_dt_segments} dt={dt_value:g} "
                    f"steps={steps_this_segment}",
                    flush=True,
                )
            sim.run(steps_this_segment)

        integrator.dt = dt_end
        print("Stage=tersoff_equil dt_ramp done", flush=True)
        print("Stage=tersoff_equil done", flush=True)
    else:
        tersoff = _add_liu_o_connor_tersoff(
            integrator=integrator,
            nlist=tersoff_nlist,
            row=params_row,
            initial_scale=1.0,
        )

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
        dynamic=["particles/position", "particles/image", "particles/typeid"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    production_start = time.perf_counter()
    sim.run(cfg.production_steps)
    print("Stage=production done", flush=True)

    production_elapsed = time.perf_counter() - production_start
    print(f"Production_runtime_seconds={production_elapsed:.2f}")
    print(f"Production_runtime_hours={production_elapsed / 3600.0:.3f}")


if __name__ == "__main__":
    main()
