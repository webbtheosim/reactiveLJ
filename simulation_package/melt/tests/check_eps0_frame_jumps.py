#!/usr/bin/env python3
"""Check eps=0 GSD frame-step continuity in raw and cleaned melt outputs.

The checker reads only the ``configuration/step`` chunks from each GSD frame.
It reports non-uniform step deltas, backward/duplicate resume overlaps, and
forward gaps. It is intended to be run under Slurm with multiple worker
processes because the virial logs contain many small step chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import gsd.fl


DEFAULT_FILES = (
    "trajectory.gsd",
    "msd_trajectory.gsd",
    "virial_tensor_log.gsd",
)


@dataclass(frozen=True)
class WorkItem:
    root: str
    root_label: str
    relative_run_dir: str
    file_name: str
    metadata_path: str
    gsd_path: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    melt_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Parallel frame-step continuity check for eps=0 GSD files under "
            "data_generation/outputs and data_generation/outputs_clean."
        )
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[
            melt_dir / "data_generation" / "outputs",
            melt_dir / "data_generation" / "outputs_clean",
        ],
        help="Output roots to inspect.",
    )
    parser.add_argument(
        "--epsilon-dir",
        default="eps_0",
        help="Epsilon directory to inspect under each root.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help="GSD filenames to inspect in each replicate directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="Maximum anomalous step examples to print per file.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV summary path. Defaults to tests/frame_jump_report_eps0.csv.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("none", "nonmonotonic", "nonuniform"),
        default="nonuniform",
        help=(
            "Exit nonzero on selected anomaly class. nonuniform includes "
            "forward gaps, missing frames, short deltas, and off-grid frames."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-file completion messages.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_delta_from_metadata(metadata: dict[str, Any], file_name: str) -> int | None:
    frame_steps = metadata.get("frame_steps")
    if file_name == "trajectory.gsd":
        value = metadata.get("trajectory_frame_steps", frame_steps)
    elif file_name == "msd_trajectory.gsd":
        value = metadata.get("msd_frame_steps", frame_steps)
    elif file_name == "virial_tensor_log.gsd":
        value = metadata.get("virial_log_steps", frame_steps)
    else:
        value = None
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def infer_expected_delta(
    metadata: dict[str, Any],
    file_name: str,
    steps: np.ndarray,
) -> tuple[int | None, str]:
    metadata_delta = expected_delta_from_metadata(metadata, file_name)
    if metadata_delta is not None:
        return metadata_delta, "metadata"

    if steps.size < 2:
        return None, "unavailable"
    diffs = np.diff(steps)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return None, "unavailable"
    return int(round(float(np.median(positive_diffs)))), "median_positive_diff"


def read_steps(path: Path) -> np.ndarray:
    with gsd.fl.open(str(path), "r") as handle:
        steps = np.empty(handle.nframes, dtype=np.int64)
        for frame_index in range(handle.nframes):
            chunk = handle.read_chunk(
                frame=frame_index,
                name="configuration/step",
            )
            steps[frame_index] = int(np.asarray(chunk).reshape(-1)[0])
    return steps


def parse_replicate(relative_run_dir: str) -> int:
    match = re.search(r"(?:^|/)rep_(\d+)(?:$|/)", relative_run_dir)
    return int(match.group(1)) if match is not None else -1


def format_step_examples(
    steps: np.ndarray,
    indices: np.ndarray,
    max_examples: int,
) -> list[dict[str, int]]:
    examples: list[dict[str, int]] = []
    diffs = np.diff(steps)
    for index in indices[:max_examples]:
        idx = int(index)
        examples.append(
            {
                "frame_index": idx,
                "step_before": int(steps[idx]),
                "step_after": int(steps[idx + 1]),
                "step_delta": int(diffs[idx]),
            }
        )
    return examples


def inspect_work_item(item: WorkItem, max_examples: int) -> dict[str, Any]:
    started = time.time()
    path = Path(item.gsd_path)
    try:
        metadata = load_metadata(Path(item.metadata_path))
        steps = read_steps(path)
        expected_delta, expected_source = infer_expected_delta(
            metadata,
            item.file_name,
            steps,
        )
        report: dict[str, Any] = {
            "root_label": item.root_label,
            "relative_run_dir": item.relative_run_dir,
            "replicate": parse_replicate(item.relative_run_dir),
            "file_name": item.file_name,
            "path": str(path),
            "exists": True,
            "error": "",
            "nframes": int(steps.size),
            "first_step": int(steps[0]) if steps.size else None,
            "last_step": int(steps[-1]) if steps.size else None,
            "expected_step_delta": expected_delta,
            "expected_delta_source": expected_source,
            "median_positive_diff": None,
            "min_diff": None,
            "max_diff": None,
            "bad_diff_count": 0,
            "nonincreasing_count": 0,
            "short_delta_count": 0,
            "forward_gap_count": 0,
            "missing_expected_frames": 0,
            "off_grid_diff_count": 0,
            "off_grid_frame_count": 0,
            "examples": [],
            "elapsed_seconds": 0.0,
            "status": "OK",
        }

        if steps.size >= 2:
            diffs = np.diff(steps)
            positive = diffs[diffs > 0]
            if positive.size:
                report["median_positive_diff"] = float(np.median(positive))
            report["min_diff"] = int(np.min(diffs))
            report["max_diff"] = int(np.max(diffs))
            report["nonincreasing_count"] = int(np.sum(diffs <= 0))

            if expected_delta is not None and expected_delta > 0:
                bad = np.flatnonzero(diffs != expected_delta)
                forward = np.flatnonzero(diffs > expected_delta)
                short = np.flatnonzero((diffs > 0) & (diffs < expected_delta))
                off_grid_diffs = np.flatnonzero(diffs % expected_delta != 0)
                report["bad_diff_count"] = int(bad.size)
                report["forward_gap_count"] = int(forward.size)
                report["short_delta_count"] = int(short.size)
                report["off_grid_diff_count"] = int(off_grid_diffs.size)
                if forward.size:
                    missing = (diffs[forward] // expected_delta) - 1
                    report["missing_expected_frames"] = int(np.sum(missing))
                origin = int(steps[0])
                off_grid_frames = np.flatnonzero((steps - origin) % expected_delta != 0)
                report["off_grid_frame_count"] = int(off_grid_frames.size)
                report["examples"] = format_step_examples(steps, bad, max_examples)

        if report["error"]:
            report["status"] = "ERROR"
        elif report["nonincreasing_count"] > 0:
            report["status"] = "ERROR"
        elif (
            report["bad_diff_count"] > 0
            or report["off_grid_frame_count"] > 0
            or report["nframes"] == 0
        ):
            report["status"] = "WARN"
        report["elapsed_seconds"] = round(time.time() - started, 3)
        return report
    except Exception as exc:  # pragma: no cover - exercised on corrupt GSDs
        return {
            "root_label": item.root_label,
            "relative_run_dir": item.relative_run_dir,
            "replicate": parse_replicate(item.relative_run_dir),
            "file_name": item.file_name,
            "path": str(path),
            "exists": path.exists(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "nframes": None,
            "first_step": None,
            "last_step": None,
            "expected_step_delta": None,
            "expected_delta_source": None,
            "median_positive_diff": None,
            "min_diff": None,
            "max_diff": None,
            "bad_diff_count": None,
            "nonincreasing_count": None,
            "short_delta_count": None,
            "forward_gap_count": None,
            "missing_expected_frames": None,
            "off_grid_diff_count": None,
            "off_grid_frame_count": None,
            "examples": [],
            "elapsed_seconds": round(time.time() - started, 3),
            "status": "ERROR",
        }


def discover_work_items(
    roots: list[Path],
    epsilon_dir: str,
    file_names: list[str],
) -> tuple[list[WorkItem], list[dict[str, Any]]]:
    items: list[WorkItem] = []
    missing: list[dict[str, Any]] = []
    for root in roots:
        resolved_root = root.resolve()
        eps_dir = resolved_root / epsilon_dir
        if not eps_dir.exists():
            missing.append(
                {
                    "root_label": resolved_root.name,
                    "relative_run_dir": epsilon_dir,
                    "file_name": "",
                    "path": str(eps_dir),
                    "status": "ERROR",
                    "error": "epsilon directory does not exist",
                }
            )
            continue

        replicate_dirs = sorted(
            (path for path in eps_dir.glob("rep_*") if path.is_dir()),
            key=lambda path: parse_replicate(str(path.relative_to(resolved_root))),
        )
        if not replicate_dirs:
            missing.append(
                {
                    "root_label": resolved_root.name,
                    "relative_run_dir": epsilon_dir,
                    "file_name": "",
                    "path": str(eps_dir),
                    "status": "ERROR",
                    "error": "no replicate directories found",
                }
            )
            continue

        for run_dir in replicate_dirs:
            relative_run_dir = str(run_dir.relative_to(resolved_root))
            metadata_path = run_dir / "metadata.json"
            for file_name in file_names:
                gsd_path = run_dir / file_name
                if not gsd_path.exists():
                    missing.append(
                        {
                            "root_label": resolved_root.name,
                            "relative_run_dir": relative_run_dir,
                            "replicate": parse_replicate(relative_run_dir),
                            "file_name": file_name,
                            "path": str(gsd_path),
                            "exists": False,
                            "status": "ERROR",
                            "error": "file does not exist",
                        }
                    )
                    continue
                items.append(
                    WorkItem(
                        root=str(resolved_root),
                        root_label=resolved_root.name,
                        relative_run_dir=relative_run_dir,
                        file_name=file_name,
                        metadata_path=str(metadata_path),
                        gsd_path=str(gsd_path),
                    )
                )
    return items, missing


def result_sort_key(
    result: dict[str, Any],
    root_order: dict[str, int],
    file_order: dict[str, int],
) -> tuple[int, int, int, str]:
    return (
        root_order.get(str(result.get("root_label", "")), 999),
        int(result.get("replicate") or 0),
        file_order.get(str(result.get("file_name", "")), 999),
        str(result.get("path", "")),
    )


def bool_fail(result: dict[str, Any], fail_on: str) -> bool:
    if result.get("status") == "ERROR" or result.get("error"):
        return True
    if fail_on == "none":
        return False
    nonincreasing = int(result.get("nonincreasing_count") or 0) > 0
    if fail_on == "nonmonotonic":
        return nonincreasing
    return (
        nonincreasing
        or int(result.get("bad_diff_count") or 0) > 0
        or int(result.get("off_grid_frame_count") or 0) > 0
    )


def print_report(results: list[dict[str, Any]], fail_on: str) -> None:
    total_files = len(results)
    error_files = sum(1 for item in results if item.get("status") == "ERROR")
    warn_files = sum(1 for item in results if item.get("status") == "WARN")
    ok_files = sum(1 for item in results if item.get("status") == "OK")
    nonuniform_files = sum(
        1
        for item in results
        if int(item.get("bad_diff_count") or 0) > 0
        or int(item.get("off_grid_frame_count") or 0) > 0
    )
    forward_gap_files = sum(
        1 for item in results if int(item.get("forward_gap_count") or 0) > 0
    )
    missing_frames = sum(int(item.get("missing_expected_frames") or 0) for item in results)
    failed = any(bool_fail(item, fail_on) for item in results)

    print("\n=== EPS=0 GSD FRAME-STEP CONTINUITY SUMMARY ===", flush=True)
    print(f"files_checked: {total_files}", flush=True)
    print(f"files_ok: {ok_files}", flush=True)
    print(f"files_warn: {warn_files}", flush=True)
    print(f"files_error: {error_files}", flush=True)
    print(f"files_with_nonuniform_step_grid: {nonuniform_files}", flush=True)
    print(f"files_with_forward_gaps: {forward_gap_files}", flush=True)
    print(f"missing_expected_frames_total: {missing_frames}", flush=True)
    print(f"fail_on: {fail_on}", flush=True)
    print(f"RESULT: {'FAIL' if failed else 'PASS'}", flush=True)

    print("\n=== PER-FILE SUMMARY ===", flush=True)
    header = (
        "status root run file nframes first last expected "
        "bad noninc forward_gaps missing offgrid_frames elapsed_s"
    )
    print(header, flush=True)
    for item in results:
        print(
            " ".join(
                [
                    str(item.get("status", "")),
                    str(item.get("root_label", "")),
                    str(item.get("relative_run_dir", "")),
                    str(item.get("file_name", "")),
                    str(item.get("nframes", "")),
                    str(item.get("first_step", "")),
                    str(item.get("last_step", "")),
                    str(item.get("expected_step_delta", "")),
                    str(item.get("bad_diff_count", "")),
                    str(item.get("nonincreasing_count", "")),
                    str(item.get("forward_gap_count", "")),
                    str(item.get("missing_expected_frames", "")),
                    str(item.get("off_grid_frame_count", "")),
                    str(item.get("elapsed_seconds", "")),
                ]
            ),
            flush=True,
        )

    anomalous = [
        item
        for item in results
        if item.get("status") != "OK"
        or int(item.get("bad_diff_count") or 0) > 0
        or item.get("error")
    ]
    if anomalous:
        print("\n=== ANOMALY DETAILS ===", flush=True)
    for item in anomalous:
        print(
            (
                f"{item.get('status')} {item.get('root_label')}/"
                f"{item.get('relative_run_dir')}/{item.get('file_name')}"
            ),
            flush=True,
        )
        if item.get("error"):
            print(f"  error: {item.get('error')}", flush=True)
        print(
            (
                f"  expected_delta={item.get('expected_step_delta')} "
                f"source={item.get('expected_delta_source')} "
                f"nframes={item.get('nframes')} first={item.get('first_step')} "
                f"last={item.get('last_step')}"
            ),
            flush=True,
        )
        print(
            (
                f"  bad_diff_count={item.get('bad_diff_count')} "
                f"nonincreasing_count={item.get('nonincreasing_count')} "
                f"short_delta_count={item.get('short_delta_count')} "
                f"forward_gap_count={item.get('forward_gap_count')} "
                f"missing_expected_frames={item.get('missing_expected_frames')} "
                f"off_grid_diff_count={item.get('off_grid_diff_count')} "
                f"off_grid_frame_count={item.get('off_grid_frame_count')}"
            ),
            flush=True,
        )
        examples = item.get("examples") or []
        for example in examples:
            print(
                (
                    "  example: "
                    f"frame_index={example['frame_index']} "
                    f"{example['step_before']} -> {example['step_after']} "
                    f"delta={example['step_delta']}"
                ),
                flush=True,
            )


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "root_label",
        "relative_run_dir",
        "replicate",
        "file_name",
        "path",
        "exists",
        "error",
        "nframes",
        "first_step",
        "last_step",
        "expected_step_delta",
        "expected_delta_source",
        "median_positive_diff",
        "min_diff",
        "max_diff",
        "bad_diff_count",
        "nonincreasing_count",
        "short_delta_count",
        "forward_gap_count",
        "missing_expected_frames",
        "off_grid_diff_count",
        "off_grid_frame_count",
        "elapsed_seconds",
        "examples_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {field: item.get(field, "") for field in fieldnames}
            row["examples_json"] = json.dumps(item.get("examples") or [])
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    start = time.time()
    workers = max(1, int(args.workers))
    roots = [path.resolve() for path in args.roots]
    csv_out = args.csv_out
    if csv_out is None:
        csv_out = Path(__file__).resolve().parent / "frame_jump_report_eps0.csv"

    print("=== EPS=0 GSD FRAME-STEP CONTINUITY CHECK ===", flush=True)
    print(f"pid: {os.getpid()}", flush=True)
    print(f"roots: {', '.join(str(path) for path in roots)}", flush=True)
    print(f"epsilon_dir: {args.epsilon_dir}", flush=True)
    print(f"files: {', '.join(args.files)}", flush=True)
    print(f"workers: {workers}", flush=True)
    print(f"csv_out: {csv_out}", flush=True)

    items, missing = discover_work_items(roots, args.epsilon_dir, list(args.files))
    print(f"discovered_existing_files: {len(items)}", flush=True)
    print(f"missing_or_discovery_errors: {len(missing)}", flush=True)
    if not items and missing:
        results = missing
        print_report(results, args.fail_on)
        write_csv(csv_out, results)
        return 1

    results: list[dict[str, Any]] = list(missing)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(inspect_work_item, item, int(args.max_examples))
            for item in items
        ]
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            if not args.no_progress:
                print(
                    (
                        f"[progress] {completed}/{len(futures)} "
                        f"{result.get('status')} "
                        f"{result.get('root_label', '')}/"
                        f"{result.get('relative_run_dir', '')}/"
                        f"{result.get('file_name', '')}"
                    ),
                    flush=True,
                )

    root_order = {path.name: idx for idx, path in enumerate(roots)}
    file_order = {name: idx for idx, name in enumerate(args.files)}
    results.sort(key=lambda item: result_sort_key(item, root_order, file_order))

    print_report(results, args.fail_on)
    write_csv(csv_out, results)
    print(f"\nwrote_csv: {csv_out}", flush=True)
    print(f"elapsed_seconds_total: {time.time() - start:.3f}", flush=True)

    return 1 if any(bool_fail(item, args.fail_on) for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
