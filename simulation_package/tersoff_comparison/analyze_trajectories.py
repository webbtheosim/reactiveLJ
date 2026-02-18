#!/usr/bin/env python3
"""Run Block 2 trajectory analysis on Tersoff data using the ReactiveLJ pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any

import gsd.hoomd
import numpy as np


LJ_INFLECTION_FACTOR = (26.0 / 7.0) ** (1.0 / 6.0)
INFLECTION_R_MIN = 1.0
INFLECTION_R_MAX = 1.5
INFLECTION_GRID_POINTS = 4096
MIN_ANALYSIS_FRAMES = 100


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _get_flag_value(args: list[str], flag: str) -> str | None:
    for idx, arg in enumerate(args):
        if arg == flag:
            if idx + 1 >= len(args):
                return None
            return args[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _strip_flag(args: list[str], flag: str) -> list[str]:
    stripped: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == flag:
            idx += 2
            continue
        if arg.startswith(f"{flag}="):
            idx += 1
            continue
        stripped.append(arg)
        idx += 1
    return stripped


def _gsd_frame_count(path: str) -> int | None:
    try:
        with gsd.hoomd.open(path, "r") as traj:
            return len(traj)
    except Exception:
        return None


def _discover_runs(input_root: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    skipped_short = 0
    skipped_unreadable = 0
    for root, _, files in os.walk(input_root):
        if "trajectory.gsd" in files and "metadata.json" in files:
            gsd_path = os.path.join(root, "trajectory.gsd")
            metadata_path = os.path.join(root, "metadata.json")
            frame_count = _gsd_frame_count(gsd_path)
            if frame_count is None:
                skipped_unreadable += 1
                continue
            if frame_count < MIN_ANALYSIS_FRAMES:
                skipped_short += 1
                continue
            runs.append((gsd_path, metadata_path))

    if skipped_short > 0 or skipped_unreadable > 0:
        print(
            "[run_discovery] "
            f"Skipped {skipped_short} short trajectories (<{MIN_ANALYSIS_FRAMES} frames) "
            f"and {skipped_unreadable} unreadable trajectories.",
            flush=True,
        )
    return sorted(runs)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return data


def _read_tersoff_param(
    params: dict[str, Any],
    key: str,
    list_key: str | None = None,
    list_idx: int | None = None,
) -> float | None:
    if key in params and params[key] is not None:
        return float(params[key])
    if list_key is not None and list_idx is not None:
        values = params.get(list_key)
        if isinstance(values, (list, tuple)) and len(values) > list_idx:
            value = values[list_idx]
            if value is not None:
                return float(value)
    return None


def _extract_tersoff_curve_params(metadata: dict[str, Any], metadata_path: str) -> dict[str, float]:
    params = metadata.get("tersoff_params")
    if not isinstance(params, dict):
        raise RuntimeError(f"Missing 'tersoff_params' in {metadata_path}")

    a1 = _read_tersoff_param(params, "A1", "magnitudes", 0)
    a2 = _read_tersoff_param(params, "A2", "magnitudes", 1)
    lambda1 = _read_tersoff_param(params, "lambda1", "exp_factors", 0)
    lambda2 = _read_tersoff_param(params, "lambda2", "exp_factors", 1)
    r_d = _read_tersoff_param(params, "r_D") or _read_tersoff_param(params, "dimer_r")
    r_cut = _read_tersoff_param(params, "r_cut")
    r_ct = _read_tersoff_param(params, "r_CT") or _read_tersoff_param(params, "cutoff_thickness")
    alpha = _read_tersoff_param(params, "alpha")

    missing = []
    if a1 is None:
        missing.append("A1")
    if a2 is None:
        missing.append("A2")
    if lambda1 is None:
        missing.append("lambda1")
    if lambda2 is None:
        missing.append("lambda2")
    if r_d is None:
        missing.append("r_D/dimer_r")
    if r_cut is None:
        missing.append("r_cut")
    if alpha is None:
        missing.append("alpha")
    if missing:
        raise RuntimeError(
            f"Missing Tersoff parameters in {metadata_path}: {', '.join(missing)}"
        )

    if r_ct is None:
        r_ct = r_cut - 1.3 * r_d
    if r_ct <= 0.0:
        raise RuntimeError(f"Invalid Tersoff cutoff_thickness in {metadata_path}: r_CT={r_ct}")

    return {
        "A1": float(a1),
        "A2": float(a2),
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "r_D": float(r_d),
        "r_cut": float(r_cut),
        "r_CT": float(r_ct),
        "alpha": float(alpha),
    }


def _tersoff_cutoff(distance: np.ndarray, r_cut: float, r_ct: float, alpha: float) -> np.ndarray:
    values = np.ones_like(distance, dtype=np.float64)
    values[distance >= r_cut] = 0.0
    transition = (distance > (r_cut - r_ct)) & (distance < r_cut)
    if np.any(transition):
        x = (distance[transition] - (r_cut - r_ct)) / r_ct
        x3 = x * x * x
        values[transition] = np.exp(-alpha * x3 / (x3 - 1.0))
    return values


def _zero_coord_tersoff_curve(r_values: np.ndarray, params: dict[str, float]) -> np.ndarray:
    fc = _tersoff_cutoff(r_values, params["r_cut"], params["r_CT"], params["alpha"])
    f_rep = params["A1"] * np.exp(params["lambda1"] * (params["r_D"] - r_values))
    f_att = params["A2"] * np.exp(params["lambda2"] * (params["r_D"] - r_values))
    return fc * (f_rep + f_att)


def _find_inflection_between_1p0_1p5(params: dict[str, float]) -> float:
    r_values = np.linspace(INFLECTION_R_MIN, INFLECTION_R_MAX, INFLECTION_GRID_POINTS)
    energies = _zero_coord_tersoff_curve(r_values, params)
    if not np.all(np.isfinite(energies)):
        raise RuntimeError("Encountered non-finite values in zero-coordination Tersoff curve")

    second_derivative = np.gradient(np.gradient(energies, r_values), r_values)
    candidates: list[float] = []
    for idx in range(len(r_values) - 1):
        y0 = second_derivative[idx]
        y1 = second_derivative[idx + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            candidates.append(float(r_values[idx]))
            continue
        if y0 * y1 < 0.0:
            frac = abs(y0) / (abs(y0) + abs(y1))
            x0 = r_values[idx]
            x1 = r_values[idx + 1]
            candidates.append(float(x0 + (x1 - x0) * frac))

    if not candidates:
        raise RuntimeError("No inflection point found between r=1.0 and r=1.5")

    r_at_min = float(r_values[int(np.argmin(energies))])
    candidates_after_min = [
        value for value in candidates if (value > r_at_min + 1e-6 and value < INFLECTION_R_MAX - 1e-6)
    ]
    if candidates_after_min:
        return float(min(candidates_after_min))

    interior_candidates = [value for value in candidates if value < INFLECTION_R_MAX - 1e-6]
    if interior_candidates:
        return float(interior_candidates[0])

    return float(candidates[0])


def _compute_cutoff_by_epsilon(runs: list[tuple[str, str]]) -> dict[float, float]:
    values_by_eps: dict[float, list[float]] = defaultdict(list)
    for _, metadata_path in runs:
        metadata = _read_json(metadata_path)
        epsilon_value = metadata.get("reactive_epsilon")
        if epsilon_value is None:
            continue
        epsilon = float(epsilon_value)
        tersoff_curve_params = _extract_tersoff_curve_params(metadata, metadata_path)
        cutoff = _find_inflection_between_1p0_1p5(tersoff_curve_params)
        values_by_eps[epsilon].append(cutoff)

    if not values_by_eps:
        raise RuntimeError("No epsilon groups with usable Tersoff parameters were found")

    cutoff_by_eps: dict[float, float] = {}
    for epsilon, values in sorted(values_by_eps.items()):
        arr = np.asarray(values, dtype=np.float64)
        cutoff = float(np.median(arr))
        cutoff_by_eps[epsilon] = cutoff
        spread = float(np.max(arr) - np.min(arr))
        sigma_equiv = cutoff / LJ_INFLECTION_FACTOR
        print(
            f"[Tersoff bond cutoff] eps={epsilon:g} r_inflect={cutoff:.6f} "
            f"sigma_equiv={sigma_equiv:.6f} spread={spread:.3e}",
            flush=True,
        )
    return cutoff_by_eps


def _symlink_or_copy(src: str, dst: str) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_input_tree_with_cutoffs(
    input_root: str,
    runs: list[tuple[str, str]],
    cutoff_by_eps: dict[float, float],
    staged_root: str,
) -> None:
    os.makedirs(staged_root, exist_ok=True)
    for gsd_path, metadata_path in runs:
        metadata = _read_json(metadata_path)
        epsilon_value = metadata.get("reactive_epsilon")
        if epsilon_value is None:
            raise RuntimeError(f"Missing reactive_epsilon in {metadata_path}")
        epsilon = float(epsilon_value)
        if epsilon not in cutoff_by_eps:
            raise RuntimeError(f"No computed cutoff for epsilon={epsilon:g}")

        bond_cutoff = cutoff_by_eps[epsilon]
        metadata["analysis_bond_cutoff"] = float(bond_cutoff)
        metadata["analysis_bond_rule"] = "tersoff_zero_coord_inflection_r_1p0_to_1p5"
        metadata["reactive_sigma"] = float(bond_cutoff / LJ_INFLECTION_FACTOR)

        rel_dir = os.path.relpath(os.path.dirname(gsd_path), input_root)
        staged_run_dir = os.path.join(staged_root, rel_dir)
        os.makedirs(staged_run_dir, exist_ok=True)

        staged_gsd_path = os.path.join(staged_run_dir, "trajectory.gsd")
        if os.path.lexists(staged_gsd_path):
            os.remove(staged_gsd_path)
        _symlink_or_copy(os.path.abspath(gsd_path), staged_gsd_path)

        staged_metadata_path = os.path.join(staged_run_dir, "metadata.json")
        with open(staged_metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_package_dir = os.path.dirname(script_dir)

    reactive_analysis_script = os.path.join(
        sim_package_dir, "analysis", "analyze_trajectories.py"
    )
    default_input_root = os.path.join(script_dir, "outputs")
    default_output_dir = os.path.join(script_dir, "analysis", "results")

    user_args = sys.argv[1:]
    if any(arg in {"-h", "--help"} for arg in user_args):
        raise SystemExit(subprocess.call([sys.executable, reactive_analysis_script, *user_args]))

    input_root_arg = _get_flag_value(user_args, "--input-root")
    input_root = (
        os.path.abspath(input_root_arg)
        if input_root_arg is not None
        else os.path.abspath(default_input_root)
    )

    passthrough_args = _strip_flag(user_args, "--input-root")
    if not _has_flag(passthrough_args, "--output-dir"):
        passthrough_args.extend(["--output-dir", default_output_dir])

    runs = _discover_runs(input_root)
    if not runs:
        raise RuntimeError(f"No runs discovered in input root: {input_root}")

    cutoff_by_eps = _compute_cutoff_by_epsilon(runs)

    with tempfile.TemporaryDirectory(prefix="tersoff_analysis_") as temp_dir:
        staged_input_root = os.path.join(temp_dir, "inputs")
        _stage_input_tree_with_cutoffs(
            input_root=input_root,
            runs=runs,
            cutoff_by_eps=cutoff_by_eps,
            staged_root=staged_input_root,
        )
        argv = [
            sys.executable,
            reactive_analysis_script,
            "--input-root",
            staged_input_root,
        ]
        argv.extend(passthrough_args)
        raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
