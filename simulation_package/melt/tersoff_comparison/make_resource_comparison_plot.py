#!/usr/bin/env python3
"""Build memory-use comparison bar plots from Slurm accounting data."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

import ultraplot as uplt


EPSILONS_DEFAULT = (6.0, 12.0, 15.0, 18.0)
N_REP_DEFAULT = 10
RESOURCE_FIELDS = (
    "JobID",
    "JobName",
    "State",
    "NodeList",
    "MaxRSS",
    "TRESUsageInMax",
)
ARRAY_TASK_RE = re.compile(r"^(?P<job_id>\d+)_(?P<task_id>\d+)\.batch$")
LOG_JOB_RE = re.compile(r"_(?P<job_id>\d+)_(?P<task_id>\d+)\.out$")
GPU_MEM_RE = re.compile(r"(?:^|,)gres/gpumem=(?P<value>[^,]+)")
GPU_TYPE_RE = re.compile(r"\bGres=.*?\bgpu:([^:,\s(]+)")


@dataclass(frozen=True)
class AccountingSample:
    model: str
    epsilon: float
    array_job_id: str
    array_task_id: int
    state: str
    node_list: str
    max_rss_mib: float | None
    gpu_mem_mib: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar plots comparing host RAM and GPU memory for "
            "ReactiveLJ and Liu/O'Connor Tersoff Slurm array jobs."
        )
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(EPSILONS_DEFAULT),
        help="Epsilon values in array-order mapping.",
    )
    parser.add_argument(
        "--n-rep",
        type=int,
        default=N_REP_DEFAULT,
        help="Replicates per epsilon in the Slurm array mapping.",
    )
    parser.add_argument(
        "--reactive-job-ids",
        nargs="+",
        default=None,
        help="ReactiveLJ Slurm array job IDs. Defaults to IDs inferred from logs.",
    )
    parser.add_argument(
        "--tersoff-job-ids",
        nargs="+",
        default=None,
        help="Tersoff Slurm array job IDs. Defaults to IDs inferred from logs.",
    )
    parser.add_argument(
        "--reactive-log-glob",
        default="logs/generate_reactive_lj_data_*.out",
        help="Glob used to infer ReactiveLJ job IDs when --reactive-job-ids is omitted.",
    )
    parser.add_argument(
        "--tersoff-log-glob",
        default="logs/generate_tersoff_data_*.out",
        help="Glob used to infer Tersoff job IDs when --tersoff-job-ids is omitted.",
    )
    parser.add_argument(
        "--output-path",
        default="plots/memory_bar_comparison.svg",
        help="Output plot path.",
    )
    parser.add_argument(
        "--samples-csv",
        default="outputs/resource_samples.csv",
        help="CSV dump of parsed Slurm accounting samples.",
    )
    parser.add_argument(
        "--gpu-label",
        default=None,
        help="GPU label to use in the title. Defaults to inference from Slurm node metadata.",
    )
    return parser.parse_args()


def _infer_job_ids(log_glob: str) -> list[str]:
    counts: Counter[str] = Counter()
    for path in glob.glob(log_glob):
        match = LOG_JOB_RE.search(os.path.basename(path))
        if match is not None:
            counts[match.group("job_id")] += 1
    return [job_id for job_id, _count in counts.most_common()]


def _run_sacct(job_ids: list[str]) -> list[dict[str, str]]:
    cmd = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--parsable2",
        "--noheader",
        f"--format={','.join(RESOURCE_FIELDS)}",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as err:
        detail = err.stderr.strip() or err.stdout.strip() or str(err)
        raise RuntimeError(f"sacct failed while reading Slurm accounting: {detail}") from err

    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) < len(RESOURCE_FIELDS):
            continue
        rows.append(dict(zip(RESOURCE_FIELDS, values, strict=False)))
    return rows


def _parse_memory_to_mib(value: str) -> float | None:
    value = value.strip()
    if not value or value == "0":
        return None

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value)
    if match is None:
        return None

    amount = float(match.group(1))
    unit = match.group(2)
    factors = {
        "": 1.0 / (1024.0 * 1024.0),
        "K": 1.0 / 1024.0,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
        "P": 1024.0 * 1024.0 * 1024.0,
    }
    return amount * factors[unit]


def _gpu_memory_mib(tres_usage: str) -> float | None:
    match = GPU_MEM_RE.search(tres_usage)
    if match is None:
        return None
    return _parse_memory_to_mib(match.group("value"))


def _format_gpu_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    known_types = {
        "h100": "NVIDIA H100",
        "a100": "NVIDIA A100",
        "v100": "NVIDIA V100",
        "p100": "NVIDIA P100",
        "l40": "NVIDIA L40",
        "l40s": "NVIDIA L40S",
    }
    if normalized in known_types:
        return known_types[normalized]
    if normalized.startswith("rtx"):
        return f"NVIDIA {raw_type.upper()}"
    return raw_type.upper()


def _expand_nodelist(nodelist: str) -> list[str]:
    if not nodelist:
        return []
    if "[" not in nodelist and "," not in nodelist:
        return [nodelist]

    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return [nodelist]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _infer_gpu_label(rows: list[dict[str, str]]) -> str | None:
    node_names: list[str] = []
    seen_nodes: set[str] = set()
    for row in rows:
        if row["State"] != "COMPLETED":
            continue
        for node_name in _expand_nodelist(row["NodeList"]):
            if node_name in seen_nodes:
                continue
            seen_nodes.add(node_name)
            node_names.append(node_name)

    gpu_types: set[str] = set()
    for node_name in node_names:
        result = subprocess.run(
            ["scontrol", "show", "node", node_name],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            continue
        match = GPU_TYPE_RE.search(result.stdout)
        if match is not None:
            gpu_types.add(_format_gpu_type(match.group(1)))

    if not gpu_types:
        return None
    if len(gpu_types) == 1:
        return next(iter(gpu_types))
    return "Mixed GPUs: " + ", ".join(sorted(gpu_types))


def _epsilon_from_task_id(task_id: int, epsilons: list[float], n_rep: int) -> float | None:
    eps_index = task_id // n_rep
    if eps_index < 0 or eps_index >= len(epsilons):
        return None
    return float(epsilons[eps_index])


def _samples_from_rows(
    rows: list[dict[str, str]],
    job_id_to_model: dict[str, str],
    epsilons: list[float],
    n_rep: int,
) -> list[AccountingSample]:
    samples: list[AccountingSample] = []
    for row in rows:
        match = ARRAY_TASK_RE.match(row["JobID"])
        if match is None:
            continue

        state = row["State"]
        if state != "COMPLETED":
            continue

        array_job_id = match.group("job_id")
        model = job_id_to_model.get(array_job_id)
        if model is None:
            continue

        task_id = int(match.group("task_id"))
        epsilon = _epsilon_from_task_id(task_id, epsilons=epsilons, n_rep=n_rep)
        if epsilon is None:
            continue

        samples.append(
            AccountingSample(
                model=model,
                epsilon=epsilon,
                array_job_id=array_job_id,
                array_task_id=task_id,
                state=state,
                node_list=row["NodeList"],
                max_rss_mib=_parse_memory_to_mib(row["MaxRSS"]),
                gpu_mem_mib=_gpu_memory_mib(row["TRESUsageInMax"]),
            )
        )
    return samples


def dump_samples_csv(path: str, samples: list[AccountingSample]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "epsilon",
                "array_job_id",
                "array_task_id",
                "state",
                "node_list",
                "max_rss_mib",
                "gpu_mem_mib",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.model,
                    sample.epsilon,
                    sample.array_job_id,
                    sample.array_task_id,
                    sample.state,
                    sample.node_list,
                    sample.max_rss_mib,
                    sample.gpu_mem_mib,
                ]
            )


def _summary_series(data: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    errors = []
    for values in data:
        if len(values) == 0:
            centers.append(np.nan)
            errors.append(np.nan)
        else:
            arr = np.asarray(values, dtype=np.float64)
            centers.append(float(np.median(arr)))
            if arr.size == 1:
                errors.append(0.0)
            else:
                q25, q75 = np.percentile(arr, [25, 75])
                errors.append(float((q75 - q25) / 2.0))
    return np.asarray(centers, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def _group_values(
    samples: list[AccountingSample],
    model: str,
    epsilons: list[float],
    field: str,
) -> list[list[float]]:
    values_by_epsilon: dict[float, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.model != model:
            continue
        value = getattr(sample, field)
        if value is not None:
            values_by_epsilon[sample.epsilon].append(float(value))
    return [values_by_epsilon.get(eps, []) for eps in epsilons]


def _plot_metric(
    ax: Any,
    samples: list[AccountingSample],
    epsilons: list[float],
    field: str,
    ylabel: str,
) -> None:
    bases = np.arange(len(epsilons), dtype=float)
    reactive_data = _group_values(samples, "ReactiveLJ", epsilons, field)
    tersoff_data = _group_values(samples, "Tersoff", epsilons, field)

    reactive_medians, reactive_errors = _summary_series(reactive_data)
    tersoff_medians, tersoff_errors = _summary_series(tersoff_data)

    width = 0.36
    ax.bar(
        bases - width / 2,
        reactive_medians,
        width=width,
        yerr=reactive_errors,
        color="#e77500",
        edgecolor="#8f4a00",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "capsize": 2.0},
        label="ReactiveLJ",
        zorder=3,
    )
    ax.bar(
        bases + width / 2,
        tersoff_medians,
        width=width,
        yerr=tersoff_errors,
        color="#121212",
        edgecolor="#121212",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "capsize": 2.0},
        label="Tersoff analog",
        zorder=3,
    )

    ax.set_xticks(bases)
    ax.set_xticklabels([f"{eps:g}" for eps in epsilons], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(r"ReactiveLJ $\varepsilon$", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.format(
        xspineloc="both",
        yspineloc="both",
        xtickloc="both",
        ytickloc="both",
        tickdir="in",
        grid=False,
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.yaxis.label.set_rotation(90)
    ax.yaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("bottom")


def plot_samples(
    path: str,
    samples: list[AccountingSample],
    epsilons: list[float],
    gpu_label: str | None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = uplt.subplots(nrows=1, ncols=2, figsize=(6.6, 2.0), dpi=600)

    _plot_metric(
        axes[0],
        samples=samples,
        epsilons=epsilons,
        field="max_rss_mib",
        ylabel="Peak Host Memory (MiB)",
    )
    _plot_metric(
        axes[1],
        samples=samples,
        epsilons=epsilons,
        field="gpu_mem_mib",
        ylabel="Peak GPU Memory (MiB)",
    )

    axes[0].set_title("Host RAM", fontsize=12)
    gpu_title = "GPU Memory"
    if gpu_label:
        gpu_title = f"{gpu_title} ({gpu_label})"
    axes[1].set_title(gpu_title, fontsize=12)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=8,
        frameon=True,
        loc="t",
        ncols=2,
    )
    fig.suptitle("Resource Usage Comparison", fontsize=12, y=1.12)
    fig.savefig(path, bbox_inches="tight")
    uplt.close(fig)


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    reactive_glob = os.path.abspath(os.path.join(script_dir, args.reactive_log_glob))
    tersoff_glob = os.path.abspath(os.path.join(script_dir, args.tersoff_log_glob))
    output_path = os.path.abspath(os.path.join(script_dir, args.output_path))
    samples_csv = os.path.abspath(os.path.join(script_dir, args.samples_csv))

    reactive_job_ids = args.reactive_job_ids or _infer_job_ids(reactive_glob)
    tersoff_job_ids = args.tersoff_job_ids or _infer_job_ids(tersoff_glob)
    if not reactive_job_ids:
        raise RuntimeError("No ReactiveLJ job IDs provided or inferred from logs.")
    if not tersoff_job_ids:
        raise RuntimeError("No Tersoff job IDs provided or inferred from logs.")

    job_id_to_model = {job_id: "ReactiveLJ" for job_id in reactive_job_ids}
    job_id_to_model.update({job_id: "Tersoff" for job_id in tersoff_job_ids})

    rows = _run_sacct(reactive_job_ids + tersoff_job_ids)
    samples = _samples_from_rows(
        rows=rows,
        job_id_to_model=job_id_to_model,
        epsilons=[float(eps) for eps in args.epsilons],
        n_rep=args.n_rep,
    )

    gpu_label = args.gpu_label or _infer_gpu_label(rows)

    dump_samples_csv(samples_csv, samples)
    plot_samples(output_path, samples, [float(eps) for eps in args.epsilons], gpu_label=gpu_label)

    reactive_count = sum(sample.model == "ReactiveLJ" for sample in samples)
    tersoff_count = sum(sample.model == "Tersoff" for sample in samples)
    print(f"ReactiveLJ job IDs: {', '.join(reactive_job_ids)}")
    print(f"Tersoff job IDs: {', '.join(tersoff_job_ids)}")
    print(f"GPU label: {gpu_label or 'not inferred'}")
    print(f"Parsed completed batch samples: ReactiveLJ={reactive_count}, Tersoff={tersoff_count}")
    print(f"Wrote sample table: {samples_csv}")
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
