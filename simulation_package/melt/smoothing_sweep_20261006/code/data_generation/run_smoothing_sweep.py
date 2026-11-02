"""Run one manifest entry from the melt smoothing sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import load_manifest
import subprocess
import sys


def command_for_task(manifest: dict, task_id: int) -> list[str]:
    if not 0 <= task_id < len(manifest["tasks"]):
        raise ValueError(f"Task {task_id} is outside the manifest")
    task = manifest["tasks"][task_id]
    if task["task_id"] != task_id:
        raise ValueError("Manifest task IDs must match their array indices")
    case = manifest["cases"][task["case_index"]]
    command = [sys.executable, "-u", manifest["runner_snapshot"]]
    options = {
        "epsilon": task["epsilon"],
        "replicate": task["replicate"],
        "seed": task["seed"],
        "smooth-kappa": case["smooth_kappa"],
        "smooth-beta": case["smooth_beta"],
        "output-root": str(Path(manifest["output_root"]) / case["case_id"]),
        **manifest["simulation_options"],
    }
    for name, value in options.items():
        command.extend([f"--{name}", str(value)])
    command.append("--resume")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    manifest = load_manifest(args.manifest)
    command = command_for_task(manifest, task_id)
    print(json.dumps({"task": manifest["tasks"][task_id], "command": command}), flush=True)
    if args.print_command:
        return 0

    import hoomd

    expected_build = manifest["hoomd_build"]["git_sha1"]
    if hoomd.version.git_sha1 != expected_build:
        raise RuntimeError(
            f"HOOMD build changed: {hoomd.version.git_sha1} != {expected_build}"
        )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
