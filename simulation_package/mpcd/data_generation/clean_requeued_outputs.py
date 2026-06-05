#!/usr/bin/env python3
"""Clean MPCD output GSDs affected by checkpoint/requeue artifacts.

The cleaner follows the same policy as the melt cleaner:

* backward timestep jumps are repaired by keeping the appended resumed branch
  and dropping the previously written suffix whose steps are >= the resumed
  frame;
* forward gaps can be excluded, preserved for segment-aware analyses, or
  removed from the sampled grid by downsampling.

The archived MPCD outputs contain trajectory.gsd and virial_tensor_log.gsd, so
those are the default target files. Newer MPCD outputs may also contain
msd_trajectory.gsd; include it with --target-files when cleaning that tree.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gsd.fl
import gsd.hoomd
import numpy as np


KNOWN_GSDS = (
    "trajectory.gsd",
    "msd_trajectory.gsd",
    "virial_tensor_log.gsd",
)
DEFAULT_TARGET_GSDS = (
    "trajectory.gsd",
    "virial_tensor_log.gsd",
)


@dataclass
class GapExample:
    frame_index: int
    step_before: int
    step_after: int
    step_delta: int


@dataclass
class FileDecision:
    filename: str
    expected_step_delta: int
    output_step_delta: int
    input_frames: int
    output_frames: int
    dropped_overlap_frames: int
    forward_gap_count: int
    missing_frame_count: int
    max_missing_in_one_gap: int
    forward_gap_examples: List[GapExample]
    downsample_factor: int | None
    downsample_phase: int | None


@dataclass
class RunDecision:
    relative_dir: str
    included: bool
    reason: str
    files: List[FileDecision]


@dataclass(frozen=True)
class CleanTask:
    run_index: int
    run_count: int
    run_dir: Path
    source_root: Path
    clean_root: Path
    apply: bool
    forward_gap_policy: str
    downsample_factor: int
    downsample_files: Tuple[str, ...]
    target_filenames: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite MPCD output GSDs to remove backward resume overlaps. "
            "Forward gaps can be excluded, kept, or removed by downsampling."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=data_dir / "outputs",
        help="Existing output tree to clean.",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=data_dir / "outputs_clean",
        help="Clean output tree to write.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=data_dir / "outputs_raw_bad_overlaps",
        help="Destination name for the original output tree during --swap.",
    )
    parser.add_argument(
        "--target-files",
        nargs="+",
        default=list(DEFAULT_TARGET_GSDS),
        help=(
            "GSD filenames to inspect and rewrite. Default: trajectory.gsd "
            "virial_tensor_log.gsd. Add msd_trajectory.gsd for newer outputs."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the clean output tree. Without this flag, only scan and report.",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help=(
            "After a successful --apply run, rename source-root to raw-root and "
            "clean-root to source-root."
        ),
    )
    parser.add_argument(
        "--allow-empty-swap",
        action="store_true",
        help="Allow --swap even when every run is excluded.",
    )
    parser.add_argument(
        "--forward-gap-policy",
        choices=("exclude", "segment", "downsample"),
        default="segment",
        help=(
            "How to handle forward gaps after backward-overlap pruning. "
            "'exclude' omits affected runs, 'segment' keeps them for analyses "
            "that split at gaps, and 'downsample' coarsens selected files onto "
            "a uniform grid that avoids missing fine frames."
        ),
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=20,
        help="Coarse-grid factor used when --forward-gap-policy=downsample.",
    )
    parser.add_argument(
        "--downsample-files",
        nargs="+",
        default=["virial_tensor_log.gsd"],
        help=(
            "GSD filenames to downsample in downsample mode. Files with forward "
            "gaps are also downsampled even if not listed."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional limit for testing. 0 processes every discovered run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, default_workers),
        help=(
            "Number of run directories to process concurrently. Defaults to "
            "SLURM_CPUS_PER_TASK when available, otherwise 1."
        ),
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_step_deltas(metadata: Dict) -> Dict[str, int]:
    frame_steps = int(metadata["frame_steps"])
    return {
        "trajectory.gsd": int(metadata.get("trajectory_frame_steps", frame_steps)),
        "msd_trajectory.gsd": int(metadata.get("msd_frame_steps", frame_steps)),
        "virial_tensor_log.gsd": int(metadata.get("virial_log_steps", frame_steps)),
    }


def discover_run_dirs(source_root: Path, target_filenames: Iterable[str]) -> List[Path]:
    target_names = tuple(target_filenames)
    run_dirs: List[Path] = []
    for metadata_path in source_root.rglob("metadata.json"):
        run_dir = metadata_path.parent
        if all((run_dir / filename).is_file() for filename in target_names):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


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


def compute_keep_indices(steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    keep_indices: List[int] = []
    keep_steps: List[int] = []
    dropped = 0
    for frame_index, raw_step in enumerate(steps):
        step = int(raw_step)
        while keep_steps and keep_steps[-1] >= step:
            keep_steps.pop()
            keep_indices.pop()
            dropped += 1
        keep_steps.append(step)
        keep_indices.append(frame_index)
    return (
        np.asarray(keep_indices, dtype=np.int64),
        np.asarray(keep_steps, dtype=np.int64),
        int(dropped),
    )


def find_forward_gaps(
    steps: np.ndarray,
    expected_delta: int,
    max_examples: int = 5,
) -> Tuple[int, int, int, List[GapExample]]:
    if steps.size <= 1:
        return 0, 0, 0, []
    diffs = np.diff(steps)
    gap_indices = np.flatnonzero(diffs > expected_delta)
    missing_per_gap = (diffs[gap_indices] // expected_delta) - 1
    missing_count = int(np.sum(missing_per_gap)) if missing_per_gap.size else 0
    max_missing = int(np.max(missing_per_gap)) if missing_per_gap.size else 0
    examples = [
        GapExample(
            frame_index=int(index),
            step_before=int(steps[index]),
            step_after=int(steps[index + 1]),
            step_delta=int(diffs[index]),
        )
        for index in gap_indices[:max_examples]
    ]
    return int(gap_indices.size), missing_count, max_missing, examples


def missing_fine_indices(steps: np.ndarray, expected_delta: int) -> np.ndarray:
    if steps.size <= 1:
        return np.empty(0, dtype=np.int64)
    origin = int(steps[0])
    offsets = steps - origin
    if np.any(offsets % expected_delta != 0):
        raise RuntimeError(
            "Cannot downsample because frames are not aligned to the expected "
            f"{expected_delta}-step grid."
        )

    fine_indices = offsets // expected_delta
    missing: List[np.ndarray] = []
    diffs = np.diff(fine_indices)
    for index in np.flatnonzero(diffs > 1):
        missing.append(
            np.arange(
                int(fine_indices[index]) + 1,
                int(fine_indices[index + 1]),
                dtype=np.int64,
            )
        )
    if not missing:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(missing)


def downsample_keep_indices(
    keep_indices: np.ndarray,
    kept_steps: np.ndarray,
    expected_delta: int,
    downsample_factor: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if downsample_factor <= 1:
        raise RuntimeError("--downsample-factor must be greater than 1.")
    if kept_steps.size == 0:
        return keep_indices, kept_steps, 0, expected_delta * downsample_factor

    origin = int(kept_steps[0])
    offsets = kept_steps - origin
    if np.any(offsets % expected_delta != 0):
        raise RuntimeError(
            "Cannot downsample because frames are not aligned to the expected "
            f"{expected_delta}-step grid."
        )

    fine_indices = offsets // expected_delta
    missing = missing_fine_indices(kept_steps, expected_delta)
    bad_phases = set((missing % downsample_factor).astype(int).tolist())
    allowed_phases = [
        phase
        for phase in range(downsample_factor)
        if phase not in bad_phases
    ]
    if not allowed_phases:
        raise RuntimeError(
            f"Downsample factor {downsample_factor} has no phase that avoids "
            "all missing fine-frame timesteps."
        )

    phase_counts = {
        phase: int(np.sum((fine_indices % downsample_factor) == phase))
        for phase in allowed_phases
    }
    if 0 in phase_counts:
        phase = 0
    else:
        phase = max(allowed_phases, key=lambda item: (phase_counts[item], -item))

    selected = (fine_indices % downsample_factor) == phase
    selected_indices = keep_indices[selected]
    selected_steps = kept_steps[selected]
    output_delta = int(expected_delta * downsample_factor)
    if selected_steps.size > 1 and np.any(np.diff(selected_steps) != output_delta):
        raise RuntimeError(
            f"Downsample factor {downsample_factor} phase {phase} did not "
            "produce a uniform timestep grid."
        )
    return selected_indices, selected_steps, int(phase), output_delta


def inspect_gsd(
    path: Path,
    expected_delta: int,
) -> Tuple[FileDecision, np.ndarray, np.ndarray]:
    steps = read_steps(path)
    keep_indices, kept_steps, dropped = compute_keep_indices(steps)
    gap_count, missing_count, max_missing, gap_examples = find_forward_gaps(
        kept_steps,
        expected_delta,
    )
    return (
        FileDecision(
            filename=path.name,
            expected_step_delta=int(expected_delta),
            output_step_delta=int(expected_delta),
            input_frames=int(steps.size),
            output_frames=int(keep_indices.size),
            dropped_overlap_frames=int(dropped),
            forward_gap_count=int(gap_count),
            missing_frame_count=int(missing_count),
            max_missing_in_one_gap=int(max_missing),
            forward_gap_examples=gap_examples,
            downsample_factor=None,
            downsample_phase=None,
        ),
        keep_indices,
        kept_steps,
    )


def copy_other_run_files(source_dir: Path, clean_dir: Path) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(source_dir.iterdir()):
        if item.name in KNOWN_GSDS or item.name == "metadata.json":
            continue
        destination = clean_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        elif item.is_file():
            shutil.copy2(item, destination)


def write_clean_metadata(
    source_dir: Path,
    clean_dir: Path,
    file_decisions: List[FileDecision],
    forward_gap_policy: str,
    target_filenames: Iterable[str],
) -> None:
    metadata = load_metadata(source_dir / "metadata.json")
    original_sampling = {
        "frame_steps": metadata.get("frame_steps"),
        "trajectory_frame_steps": metadata.get("trajectory_frame_steps"),
        "msd_frame_steps": metadata.get("msd_frame_steps"),
        "virial_log_steps": metadata.get("virial_log_steps"),
    }
    downsampled_files = []
    for decision in file_decisions:
        if decision.downsample_factor is None:
            continue
        downsampled_files.append(decision.filename)
        if decision.filename == "trajectory.gsd":
            metadata["trajectory_frame_steps"] = int(decision.output_step_delta)
        elif decision.filename == "msd_trajectory.gsd":
            metadata["msd_frame_steps"] = int(decision.output_step_delta)
        elif decision.filename == "virial_tensor_log.gsd":
            metadata["virial_log_steps"] = int(decision.output_step_delta)

    metadata["cleaning_forward_gap_policy"] = forward_gap_policy
    metadata["cleaning_timestamp"] = datetime.now().isoformat(timespec="seconds")
    metadata["cleaning_cleaned_files"] = list(target_filenames)
    metadata["cleaning_downsampled_files"] = downsampled_files
    metadata["cleaning_original_sampling_steps"] = original_sampling
    if downsampled_files:
        metadata["cleaning_note"] = (
            "Backward resume overlaps were pruned in favor of the appended resumed "
            "branch. Downsampled files were written on a uniform timestep grid."
        )
    else:
        metadata["cleaning_note"] = (
            "Backward resume overlaps were pruned in favor of the appended resumed "
            "branch. Forward gaps, if present, were preserved for segment-aware "
            "analysis."
        )
    with (clean_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_clean_gsd(source_path: Path, clean_path: Path, keep_indices: np.ndarray) -> None:
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(str(source_path), "r") as source:
        source_frame_count = len(source)
        if keep_indices.size == source_frame_count:
            shutil.copy2(source_path, clean_path)
            return

    if keep_indices.size == 0:
        with gsd.hoomd.open(str(clean_path), "w"):
            return

    with gsd.hoomd.open(str(source_path), "r") as source, gsd.hoomd.open(
        str(clean_path),
        "w",
    ) as clean:
        if keep_indices.size * 2 <= source_frame_count:
            for frame_index in keep_indices:
                clean.append(source[int(frame_index)])
            return

        keep_iter = iter(int(index) for index in keep_indices)
        next_keep = next(keep_iter, None)
        for frame_index, frame in enumerate(source):
            if next_keep is None:
                break
            if frame_index == next_keep:
                clean.append(frame)
                next_keep = next(keep_iter, None)


def clean_run(
    run_dir: Path,
    source_root: Path,
    clean_root: Path,
    apply: bool,
    forward_gap_policy: str,
    downsample_factor: int,
    downsample_files: set[str],
    target_filenames: Tuple[str, ...],
) -> RunDecision:
    relative_dir = run_dir.relative_to(source_root)
    metadata = load_metadata(run_dir / "metadata.json")
    expected = expected_step_deltas(metadata)

    file_decisions: List[FileDecision] = []
    keep_by_filename: Dict[str, np.ndarray] = {}
    kept_steps_by_filename: Dict[str, np.ndarray] = {}
    for filename in target_filenames:
        decision, keep_indices, kept_steps = inspect_gsd(
            run_dir / filename,
            expected[filename],
        )
        file_decisions.append(decision)
        keep_by_filename[filename] = keep_indices
        kept_steps_by_filename[filename] = kept_steps

    gap_files = [
        decision.filename
        for decision in file_decisions
        if decision.forward_gap_count > 0
    ]
    if gap_files and forward_gap_policy == "exclude":
        return RunDecision(
            relative_dir=str(relative_dir),
            included=False,
            reason="forward gaps in " + ", ".join(gap_files),
            files=file_decisions,
        )

    if forward_gap_policy == "downsample":
        gap_file_set = set(gap_files)
        for decision in file_decisions:
            if (
                decision.filename not in downsample_files
                and decision.filename not in gap_file_set
            ):
                continue
            try:
                keep_indices, kept_steps, phase, output_delta = downsample_keep_indices(
                    keep_by_filename[decision.filename],
                    kept_steps_by_filename[decision.filename],
                    expected[decision.filename],
                    downsample_factor,
                )
            except RuntimeError as exc:
                return RunDecision(
                    relative_dir=str(relative_dir),
                    included=False,
                    reason=f"downsample failed for {decision.filename}: {exc}",
                    files=file_decisions,
                )
            keep_by_filename[decision.filename] = keep_indices
            kept_steps_by_filename[decision.filename] = kept_steps
            decision.output_frames = int(keep_indices.size)
            decision.output_step_delta = int(output_delta)
            decision.downsample_factor = int(downsample_factor)
            decision.downsample_phase = int(phase)
            residual_gap_count, _missing, _max_missing, _examples = find_forward_gaps(
                kept_steps,
                output_delta,
            )
            if residual_gap_count > 0:
                return RunDecision(
                    relative_dir=str(relative_dir),
                    included=False,
                    reason=(
                        "downsampled grid still has forward gaps in "
                        f"{decision.filename}"
                    ),
                    files=file_decisions,
                )

    if apply:
        clean_dir = clean_root / relative_dir
        copy_other_run_files(run_dir, clean_dir)
        write_clean_metadata(
            run_dir,
            clean_dir,
            file_decisions,
            forward_gap_policy,
            target_filenames,
        )
        for filename in target_filenames:
            write_clean_gsd(
                run_dir / filename,
                clean_dir / filename,
                keep_by_filename[filename],
            )

    return RunDecision(
        relative_dir=str(relative_dir),
        included=True,
        reason=(
            "cleaned "
            + ", ".join(target_filenames)
            + f" with forward_gap_policy={forward_gap_policy}"
        ),
        files=file_decisions,
    )


def clean_run_task(task: CleanTask) -> Tuple[int, RunDecision]:
    decision = clean_run(
        task.run_dir,
        task.source_root,
        task.clean_root,
        task.apply,
        task.forward_gap_policy,
        task.downsample_factor,
        set(task.downsample_files),
        task.target_filenames,
    )
    return task.run_index, decision


def get_process_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def write_manifest(
    path: Path,
    decisions: Iterable[RunDecision],
    args: argparse.Namespace,
) -> None:
    decisions_list = list(decisions)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(args.source_root),
        "clean_root": str(args.clean_root),
        "raw_root": str(args.raw_root),
        "forward_gap_policy": str(args.forward_gap_policy),
        "downsample_factor": int(args.downsample_factor),
        "downsample_files": list(args.downsample_files),
        "target_files": list(args.target_files),
        "included_runs": sum(1 for decision in decisions_list if decision.included),
        "excluded_runs": sum(1 for decision in decisions_list if not decision.included),
        "runs": [asdict(decision) for decision in decisions_list],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def swap_output_roots(source_root: Path, clean_root: Path, raw_root: Path) -> None:
    if raw_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing raw root: {raw_root}")
    if not clean_root.exists():
        raise RuntimeError(f"Clean root does not exist: {clean_root}")
    source_root.rename(raw_root)
    try:
        clean_root.rename(source_root)
    except Exception:
        raw_root.rename(source_root)
        raise


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    clean_root = args.clean_root.resolve()
    raw_root = args.raw_root.resolve()
    args.source_root = source_root
    args.clean_root = clean_root
    args.raw_root = raw_root

    if args.swap and not args.apply:
        raise RuntimeError("--swap requires --apply.")
    unknown_targets = set(args.target_files) - set(KNOWN_GSDS)
    if unknown_targets:
        raise RuntimeError(
            "--target-files contains unknown GSD names: "
            + ", ".join(sorted(unknown_targets))
        )
    unknown_downsample_files = set(args.downsample_files) - set(KNOWN_GSDS)
    if unknown_downsample_files:
        raise RuntimeError(
            "--downsample-files contains unknown GSD names: "
            + ", ".join(sorted(unknown_downsample_files))
        )
    if args.downsample_factor <= 1:
        raise RuntimeError("--downsample-factor must be greater than 1.")
    if args.workers <= 0:
        raise RuntimeError("--workers must be greater than 0.")
    if not source_root.is_dir():
        raise RuntimeError(f"Source root does not exist: {source_root}")
    if args.apply and clean_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing clean root: {clean_root}")
    if args.apply and args.swap and raw_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing raw root: {raw_root}")

    target_filenames = tuple(args.target_files)
    run_dirs = discover_run_dirs(source_root, target_filenames)
    if args.max_runs > 0:
        run_dirs = run_dirs[: args.max_runs]
    if not run_dirs:
        raise RuntimeError(
            f"No complete run directories with {', '.join(target_filenames)} "
            f"found under {source_root}"
        )

    if args.apply:
        clean_root.mkdir(parents=True, exist_ok=False)

    worker_count = min(int(args.workers), len(run_dirs))
    tasks = [
        CleanTask(
            run_index=run_index,
            run_count=len(run_dirs),
            run_dir=run_dir,
            source_root=source_root,
            clean_root=clean_root,
            apply=args.apply,
            forward_gap_policy=args.forward_gap_policy,
            downsample_factor=int(args.downsample_factor),
            downsample_files=tuple(args.downsample_files),
            target_filenames=target_filenames,
        )
        for run_index, run_dir in enumerate(run_dirs, start=1)
    ]
    print(
        f"Processing {len(tasks)} run directories with {worker_count} worker(s)",
        flush=True,
    )

    decisions_by_index: List[RunDecision | None] = [None] * len(tasks)
    if worker_count == 1:
        for task in tasks:
            relative_dir = task.run_dir.relative_to(source_root)
            print(f"[{task.run_index}/{task.run_count}] scanning {relative_dir}", flush=True)
            run_index, decision = clean_run_task(task)
            decisions_by_index[run_index - 1] = decision
            status = "included" if decision.included else "excluded"
            print(f"  {status}: {decision.reason}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_process_context(),
        ) as executor:
            futures = {
                executor.submit(clean_run_task, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                relative_dir = task.run_dir.relative_to(source_root)
                try:
                    run_index, decision = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Worker failed while cleaning {relative_dir}"
                    ) from exc
                decisions_by_index[run_index - 1] = decision
                status = "included" if decision.included else "excluded"
                print(
                    f"[{run_index}/{task.run_count}] {relative_dir}: "
                    f"{status}: {decision.reason}",
                    flush=True,
                )

    decisions = [
        decision
        for decision in decisions_by_index
        if decision is not None
    ]

    manifest_path = (
        clean_root / "cleaning_manifest.json"
        if args.apply
        else source_root.parent / "outputs_clean.dry_run_manifest.json"
    )
    write_manifest(manifest_path, decisions, args)
    included = sum(1 for decision in decisions if decision.included)
    excluded = len(decisions) - included
    print(
        f"Wrote manifest to {manifest_path}; included={included}, excluded={excluded}",
        flush=True,
    )

    if args.apply and args.swap:
        if included == 0 and not args.allow_empty_swap:
            raise RuntimeError(
                "Refusing to swap an empty clean output tree. Every discovered "
                "run was excluded; pass --allow-empty-swap to override."
            )
        swap_output_roots(source_root, clean_root, raw_root)
        print(
            f"Renamed {source_root} -> {raw_root} and {clean_root} -> {source_root}",
            flush=True,
        )


if __name__ == "__main__":
    main()
