"""Submit the prepared sweep once, followed by dependent CPU analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from campaign import load_manifest
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-scheduler", action="store_true")
    args = parser.parse_args()
    campaign = Path(__file__).resolve().parent
    manifest_path = campaign / "manifest.json"
    manifest = load_manifest(manifest_path)
    (campaign / "logs").mkdir(exist_ok=True)
    receipt = campaign / "submission.json"
    if len(manifest["tasks"]) != 252 or manifest["max_concurrent_gpus"] != 11:
        raise RuntimeError("Unexpected simulation count or GPU limit")
    for entry in manifest["source_snapshots"]:
        digest = hashlib.sha256(Path(entry["snapshot"]).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Source snapshot changed: {entry['snapshot']}")
    export = f"ALL,SWEEP_MANIFEST={manifest_path}"
    code = campaign / "code"
    common = ["sbatch", "--parsable", f"--chdir={campaign}", f"--export={export}"]
    generation = common + [f"--output={campaign}/logs/generation_%A_%a.out",
                           str(code / "generate_smoothing_sweep.slurm")]
    if args.dry_run:
        print(json.dumps({"generation": generation, "dependent_analysis_array": "0-8",
                          "dependent_combination_job": True}, indent=2))
        return
    if args.validate_scheduler:
        result = subprocess.run(generation[:1] + ["--test-only"] + generation[1:],
                                check=False, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        raise SystemExit(result.returncode)

    # Save each accepted job immediately so a later submission failure cannot
    # lead to a duplicate GPU campaign on retry.
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get("status") == "submitted":
            raise RuntimeError(f"Campaign already submitted: {receipt}")
    else:
        record = {"manifest": str(manifest_path), "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                  "simulation_count": 252, "gpu_concurrency_limit": 11, "jobs": {}}
        fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")

    def submit(name: str, command: list[str]) -> str:
        if name in record["jobs"]:
            return record["jobs"][name]["job_id"]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            record.setdefault("failed_attempts", []).append({
                "name": name, "command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "time_utc": datetime.now(timezone.utc).isoformat(),
            })
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", flush=True)
            raise SystemExit(result.returncode)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected scheduler response: {result.stdout!r}")
        record["jobs"][name] = {"job_id": job_id, "command": command}
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Submitted {name}: {job_id}", flush=True)
        if result.stderr.strip():
            print(result.stderr.strip(), flush=True)
        return job_id

    generation_id = submit("generation", generation)
    analysis_id = submit("analysis", common + [f"--dependency=afterok:{generation_id}",
        f"--output={campaign}/logs/analysis_%A_%a.out", str(code / "run_smoothing_sweep_analysis.slurm")])
    submit("combined_report", common + [f"--dependency=afterok:{analysis_id}",
        f"--output={campaign}/logs/report_%j.out", str(code / "combine_smoothing_sweep_results.slurm")])
    record["status"] = "submitted"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    manifest["status"] = "submitted"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
