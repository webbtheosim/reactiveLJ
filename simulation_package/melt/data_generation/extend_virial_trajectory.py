#!/usr/bin/env python3
"""Extend clean ReactiveLJ checkpoints with dense virial-tensor sampling.

This script starts from full ``checkpoint.gsd`` states in ``outputs_clean`` and
runs a short continuation using the same production force-field setup as
``run_reactive_lj.py``. It writes only ``virial_tensor_log.gsd`` plus metadata
under a mirrored output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import List

import hoomd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_reactive_lj import (
    DEFAULT_TAU_R0,
    SimulationConfig,
    VirialTensorLogger,
    add_reactive_lj,
    build_integrator,
    flush_output_writers,
    production_steps_for_tau_r0,
    validate_stickers,
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Extend existing ReactiveLJ clean-output checkpoints with frequent "
            "virial-tensor logging."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=data_dir / "outputs_clean",
        help="Root containing eps_*/rep_*/checkpoint.gsd and metadata.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_dir / "outputs_virial_extended",
        help="Mirrored output root for dense virial logs.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help=(
            "Optional index into the sorted discovered run list. Intended for "
            "SLURM array jobs."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional epsilon filter.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=None,
        help="Optional replicate filter.",
    )
    parser.add_argument(
        "--weakening-exponent",
        type=float,
        default=None,
        help="Optional weakening-exponent filter.",
    )
    parser.add_argument(
        "--extension-runtime-tau-r0",
        type=float,
        default=10.0,
        help="Extension length in tau_R^0 units.",
    )
    parser.add_argument(
        "--virial-log-steps",
        type=int,
        default=100,
        help="Dense virial sampling interval in MD steps.",
    )
    parser.add_argument(
        "--device",
        choices=["gpu"],
        default="gpu",
        help="HOOMD device backend. GPU is required for the virial logger.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing extension virial log for the selected run.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional limit on discovered runs before task-index selection.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def value_matches(value: float | None, selected: float | None) -> bool:
    if selected is None:
        return True
    if value is None:
        return False
    return abs(float(value) - float(selected)) <= 1.0e-12


def discover_run_dirs(
    input_root: Path,
    epsilon: float | None,
    replicate: int | None,
    weakening_exponent: float | None,
) -> List[Path]:
    excluded_dirs = {"TEST", "TEST_CPU_REACTIVELJ", "archived"}
    run_dirs: List[Path] = []
    for metadata_path in input_root.rglob("metadata.json"):
        if any(part in excluded_dirs for part in metadata_path.parts):
            continue
        run_dir = metadata_path.parent
        checkpoint_path = run_dir / "checkpoint.gsd"
        if not checkpoint_path.is_file():
            continue
        metadata = load_metadata(metadata_path)
        run_epsilon = metadata.get("reactive_epsilon")
        run_replicate = metadata.get("replicate")
        run_weakening_exponent = metadata.get("weakening_exponent")
        if not value_matches(run_epsilon, epsilon):
            continue
        if replicate is not None and int(run_replicate) != int(replicate):
            continue
        if not value_matches(run_weakening_exponent, weakening_exponent):
            continue
        run_dirs.append(run_dir)

    return sorted(
        run_dirs,
        key=lambda path: (
            float(load_metadata(path / "metadata.json").get("reactive_epsilon", 0.0)),
            float(load_metadata(path / "metadata.json").get("weakening_exponent", 4.0)),
            int(load_metadata(path / "metadata.json").get("replicate", 0)),
            str(path),
        ),
    )


def config_from_metadata(
    metadata: dict, extension_steps: int, virial_log_steps: int
) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg_fields = {field.name for field in fields(SimulationConfig)}
    for key, value in metadata.items():
        if key in cfg_fields:
            setattr(cfg, key, value)
    cfg.reactive_epsilon = float(metadata.get("reactive_epsilon", cfg.reactive_epsilon))
    cfg.production_steps = int(extension_steps)
    cfg.virial_log_steps = int(virial_log_steps)
    return cfg


def select_run_dirs(args: argparse.Namespace) -> List[Path]:
    run_dirs = discover_run_dirs(
        args.input_root,
        epsilon=args.epsilon,
        replicate=args.replicate,
        weakening_exponent=args.weakening_exponent,
    )
    if args.max_runs > 0:
        run_dirs = run_dirs[: args.max_runs]
    if args.task_index is not None:
        if args.task_index < 0:
            raise RuntimeError("--task-index must be non-negative.")
        if args.task_index >= len(run_dirs):
            log(
                f"Task index {args.task_index} is outside discovered run count "
                f"{len(run_dirs)}; exiting."
            )
            return []
        run_dirs = [run_dirs[args.task_index]]
    return run_dirs


def relative_output_dir(input_root: Path, output_root: Path, run_dir: Path) -> Path:
    try:
        rel = run_dir.relative_to(input_root)
    except ValueError:
        rel = Path(run_dir.name)
    return output_root / rel


def write_extension_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_extension(
    run_dir: Path,
    input_root: Path,
    output_root: Path,
    extension_runtime_tau_r0: float,
    virial_log_steps: int,
    overwrite: bool,
) -> None:
    source_metadata_path = run_dir / "metadata.json"
    checkpoint_path = run_dir / "checkpoint.gsd"
    metadata = load_metadata(source_metadata_path)

    extension_steps = production_steps_for_tau_r0(
        float(extension_runtime_tau_r0),
        float(metadata.get("dt", 0.005)),
    )
    cfg = config_from_metadata(metadata, extension_steps, virial_log_steps)
    reactive_lj_enabled = bool(
        metadata.get("reactive_lj_enabled", cfg.reactive_epsilon > 0.0)
    )
    seed = int(metadata.get("seed", 1))

    output_dir = relative_output_dir(input_root, output_root, run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    virial_path = output_dir / "virial_tensor_log.gsd"
    output_metadata_path = output_dir / "metadata.json"

    if virial_path.exists():
        existing_metadata = (
            load_metadata(output_metadata_path) if output_metadata_path.exists() else {}
        )
        if existing_metadata.get("run_status") == "complete" and not overwrite:
            log(f"Skipping completed extension output: {virial_path}")
            return
        if not overwrite:
            raise RuntimeError(
                f"{virial_path} already exists. Use --overwrite to regenerate it."
            )

    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=seed)
    log(f"Loading checkpoint {checkpoint_path}")
    sim.create_state_from_gsd(filename=str(checkpoint_path))

    pair_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    reactive_nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    integrator = build_integrator(cfg, pair_nlist, reactive_lj_enabled)
    sim.operations.integrator = integrator
    if reactive_lj_enabled:
        add_reactive_lj(integrator, reactive_nlist, cfg)

    validate_stickers(sim, cfg)
    zero_momentum = hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(int(cfg.zero_momentum_period))
    )
    sim.operations.updaters.append(zero_momentum)

    sim.always_compute_pressure = False
    virial_tensor_logger = VirialTensorLogger(sim)
    virial_logger = hoomd.logging.Logger()
    virial_logger.add(virial_tensor_logger, quantities=["virial_tensor"])
    virial_writer = hoomd.write.GSD(
        filename=str(virial_path),
        trigger=hoomd.trigger.Periodic(int(cfg.virial_log_steps)),
        mode="wb",
        filter=hoomd.filter.Null(),
        dynamic=[],
        logger=virial_logger,
    )
    sim.operations.writers.append(virial_writer)

    start_timestep = int(sim.timestep)
    target_timestep = start_timestep + int(extension_steps)
    metadata_payload = asdict(cfg)
    metadata_payload.update(
        {
            "run_status": "running",
            "extension_run": True,
            "source_run_dir": str(run_dir.resolve()),
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_metadata": str(source_metadata_path.resolve()),
            "source_timestep": start_timestep,
            "extension_start_timestep": start_timestep,
            "extension_steps": int(extension_steps),
            "extension_runtime_tau_r0": float(extension_steps)
            * float(cfg.dt)
            / DEFAULT_TAU_R0,
            "extension_target_final_timestep": target_timestep,
            "tau_R0": DEFAULT_TAU_R0,
            "reactive_epsilon": float(
                metadata.get("reactive_epsilon", cfg.reactive_epsilon)
            ),
            "reactive_lj_enabled": reactive_lj_enabled,
            "replicate": int(metadata.get("replicate", 0)),
            "seed": seed,
            "n_particles": int(
                metadata.get("n_particles", cfg.n_chains * cfg.chain_length)
            ),
            "target_box_length": metadata.get("target_box_length"),
            "initial_box_length": metadata.get("initial_box_length"),
            "trajectory_particle_subset": "none",
            "virial_log_file": "virial_tensor_log.gsd",
            "position_frames_written": False,
        }
    )
    write_extension_metadata(output_metadata_path, metadata_payload)

    log(
        f"Extending eps={metadata_payload['reactive_epsilon']:g} "
        f"rep={metadata_payload['replicate']} "
        f"steps={extension_steps} "
        f"runtime_tau_R0={metadata_payload['extension_runtime_tau_r0']:.3f} "
        f"virial_log_steps={cfg.virial_log_steps}"
    )
    start_wall = time.perf_counter()
    sim.run(int(extension_steps))
    flush_output_writers(sim)
    walltime = time.perf_counter() - start_wall

    metadata_payload.update(
        {
            "run_status": "complete",
            "extension_final_timestep": int(sim.timestep),
            "extension_walltime_seconds": walltime,
        }
    )
    write_extension_metadata(output_metadata_path, metadata_payload)
    log(f"Wrote dense virial log to {virial_path}")


def main() -> int:
    args = parse_args()
    if args.extension_runtime_tau_r0 <= 0.0:
        raise RuntimeError("--extension-runtime-tau-r0 must be positive.")
    if args.virial_log_steps <= 0:
        raise RuntimeError("--virial-log-steps must be positive.")

    run_dirs = select_run_dirs(args)
    if not run_dirs:
        return 0

    log(f"Selected {len(run_dirs)} run(s) for virial extension")
    for run_dir in run_dirs:
        run_extension(
            run_dir=run_dir,
            input_root=args.input_root,
            output_root=args.output_root,
            extension_runtime_tau_r0=float(args.extension_runtime_tau_r0),
            virial_log_steps=int(args.virial_log_steps),
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
