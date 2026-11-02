"""Report gelation, apparent turnover and fitted persistence for a sweep cell.

Bond identification, endpoint correlations, turnover classification and the
initial semilog fit use the existing melt analysis definitions. Only new sweep
trajectories listed in the manifest are read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign import load_manifest

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/reactive_lj_sweep_plot_{os.getuid()}")

import gsd.hoomd
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import (
    MultiTauSetCorrelationAccumulator,
    compute_r_thresh,
    find_sticker_neighbor_pairs,
    fit_exponential_semilog_linear_region,
)

MAX_LAG_TAU_LJ = 4041.0 * 1000.0


def turnover_counts(
    current_bonds: np.ndarray,
    previous_bonds: np.ndarray,
    previous_open: np.ndarray,
) -> tuple[int, int]:
    """Classify new pairs from the preceding saved frame, as in the melt code."""
    new_bonds = np.setdiff1d(current_bonds, previous_bonds, assume_unique=True)
    n_stickers = len(previous_open)
    new_i, new_j = new_bonds // n_stickers, new_bonds % n_stickers
    associative = int(np.count_nonzero((~previous_open[new_i]) | (~previous_open[new_j])))
    return associative, int(len(new_bonds) - associative)


def analyze_replicate(task: dict, case: dict, manifest: dict) -> dict:
    replicate_root = (
        Path(manifest["output_root"]) / case["case_id"]
        / f"eps_{task['epsilon']:g}" / f"rep_{task['replicate']:03d}"
    )
    metadata = json.loads((replicate_root / "metadata.json").read_text())
    expected = {
        "reactive_epsilon": task["epsilon"], "replicate": task["replicate"],
        "seed": task["seed"], "smooth_kappa": case["smooth_kappa"],
        "smooth_beta": case["smooth_beta"],
        **manifest["baseline_configuration"],
    }
    for name, value in expected.items():
        actual = metadata.get(name)
        if value is None or isinstance(value, bool):
            matches = actual == value
        else:
            matches = actual is not None and np.isclose(actual, value, rtol=1e-10, atol=1e-12)
        if not matches:
            raise RuntimeError(f"{replicate_root}: {name}={actual!r}, expected {value!r}")
    if metadata.get("run_status") != "complete":
        raise RuntimeError(f"{replicate_root}: simulation is not complete")
    if metadata["production_completed_steps"] != manifest["simulation_options"]["production-steps"]:
        raise RuntimeError(f"{replicate_root}: incomplete production duration")

    frame_dt = metadata["dt"] * metadata["frame_steps"]
    r_thresh = compute_r_thresh(metadata["reactive_sigma"])
    pair_cutoff = max(
        r_thresh, metadata["reactive_r_cut"] or 1.5 * metadata["reactive_sigma"],
        metadata["weakening_outer"] or 1.5 * metadata["reactive_sigma"],
    )
    p_c = 1.0 / (metadata["stickers_per_chain"] - 1.0)
    max_lag = max(1, int(np.floor(MAX_LAG_TAU_LJ / frame_dt + 1e-12)))
    correlation = MultiTauSetCorrelationAccumulator(max_lag)
    gelation, open_fraction, steps = [], [], []
    previous_bonds = previous_open = None
    associative_total = dissociative_total = 0
    free_sticker_time = 0.0
    trajectory = replicate_root / "trajectory.gsd"
    with gsd.hoomd.open(str(trajectory), "r") as frames:
        if len(frames) < 2:
            raise RuntimeError(f"{trajectory}: at least two frames are required")
        for frame in frames:
            n_stickers = frame.particles.N
            if n_stickers != metadata["n_chains"] * metadata["stickers_per_chain"]:
                raise RuntimeError(f"{trajectory}: unexpected sticker count")
            steps.append(int(frame.configuration.step))
            if len(steps) > 1 and steps[-1] - steps[-2] != metadata["frame_steps"]:
                raise RuntimeError(f"{trajectory}: unexpected saved-frame spacing")
            pair_i, pair_j, pair_distance = find_sticker_neighbor_pairs(
                frame.particles.position, float(frame.configuration.box[0]), pair_cutoff,
            )
            bonded = pair_distance < r_thresh
            bond_i, bond_j = pair_i[bonded], pair_j[bonded]
            degrees = np.zeros(n_stickers, dtype=np.int32)
            np.add.at(degrees, bond_i, 1)
            np.add.at(degrees, bond_j, 1)
            is_open = degrees == 0
            bond_ids = (
                np.minimum(bond_i, bond_j).astype(np.int64) * n_stickers
                + np.maximum(bond_i, bond_j).astype(np.int64)
            )
            bond_ids.sort()
            p_open = float(np.mean(is_open))
            open_fraction.append(p_open)
            gelation.append((1.0 - p_open - p_c) / p_c)
            correlation.update(bond_ids)
            if previous_bonds is not None:
                assoc, dissoc = turnover_counts(bond_ids, previous_bonds, previous_open)
                associative_total += assoc
                dissociative_total += dissoc
                free_sticker_time += float(np.count_nonzero(previous_open)) * frame_dt
            previous_bonds, previous_open = bond_ids, is_open.copy()

    valid_length = correlation.valid_length()
    cs = correlation.correlation()[:valid_length]
    time = correlation.lag_indices[:valid_length].astype(float) * frame_dt
    tau_s = fit_exponential_semilog_linear_region(time, cs)
    row = {
        "case_id": case["case_id"], "smooth_kappa": case["smooth_kappa"],
        "smooth_beta": case["smooth_beta"], "epsilon": task["epsilon"],
        "replicate": task["replicate"], "seed": task["seed"],
        "n_frames": len(steps), "frame_dt_tau_LJ": frame_dt,
        "observed_span_tau_LJ": (steps[-1] - steps[0]) * metadata["dt"],
        "gelation_mean": float(np.mean(gelation)),
        "p_open_mean": float(np.mean(open_fraction)),
        "associative_events": associative_total, "dissociative_events": dissociative_total,
        "free_sticker_time": free_sticker_time,
        "rate_assoc": associative_total / free_sticker_time if free_sticker_time else float("nan"),
        "rate_dissoc": dissociative_total / free_sticker_time if free_sticker_time else float("nan"),
        "tau_s": tau_s, "trajectory_path": str(trajectory),
    }
    print(f"Completed {case['case_id']} eps={task['epsilon']} rep={task['replicate']}", flush=True)
    return {"row": row, "cs_time": time, "cs": cs}


def mean_stderr(values) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    error = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else float("nan")
    return float(np.mean(finite)), error, len(finite)


def bootstrap_persistence(time: np.ndarray, curves: np.ndarray, seed: int) -> tuple[float, float, int]:
    estimate = fit_exponential_semilog_linear_region(time, np.mean(curves, axis=0))
    rng = np.random.default_rng(seed)
    fits = [
        fit_exponential_semilog_linear_region(
            time, np.mean(curves[rng.integers(0, len(curves), size=len(curves))], axis=0),
        )
        for _ in range(250)
    ]
    finite = np.asarray(fits)[np.isfinite(fits)]
    error = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    return float(estimate), error, len(finite)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(fig, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=250)
    plt.close(fig)


def plot_cell(rows: list[dict], case: dict, output: Path, tau_r0: float) -> None:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10})
    x = np.arange(len(rows))
    labels = ["WCA" if row["epsilon"] == 0 else f"{row['epsilon']:g}" for row in rows]
    title = rf"$\kappa={case['smooth_kappa']:g},\ \beta={case['smooth_beta']:g}$"
    xlabel = r"Sticker strength, $\varepsilon_{\mathrm{RLJ}}/\varepsilon_0$"
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.bar(x, [r["gelation_mean"] for r in rows], yerr=[r["gelation_stderr"] for r in rows],
           color="#e77500", edgecolor="black", linewidth=0.5, capsize=2)
    ax.set(xticks=x, xticklabels=labels, xlabel=xlabel,
           ylabel=r"Degree of gelation, $\mathcal{E}$", title=title)
    ax.axhline(0, color="black", linewidth=0.4)
    save_plot(fig, output / "gelation_epsilon_vs_epsilon")

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for offset, key, label, color in [(-0.18, "assoc", "assoc.", "#e77500"),
                                       (0.18, "dissoc", "dissoc.", "#121212")]:
        values = np.asarray([r[f"rate_{key}_mean"] for r in rows]) * tau_r0
        errors = np.asarray([r[f"rate_{key}_stderr"] for r in rows]) * tau_r0
        ax.bar(x + offset, values, width=0.36, yerr=errors,
               label=label, color=color, edgecolor="black", linewidth=0.5, capsize=2)
    if any(r["rate_assoc_mean"] > 0 or r["rate_dissoc_mean"] > 0 for r in rows):
        ax.set_yscale("log")
    ax.set(xticks=x, xticklabels=labels, xlabel=xlabel,
           ylabel=r"Apparent turnover, $\nu_\alpha\tau_R^0$", title=title)
    ax.legend(frameon=False, fontsize=8)
    save_plot(fig, output / "exchange_rate_comparison_vs_epsilon")

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    tau = np.asarray([r["tau_s_fit_mean_correlation"] for r in rows])
    error = np.asarray([r["tau_s_fit_bootstrap_stderr"] for r in rows])
    good = np.isfinite(tau) & (tau > 0)
    ax.errorbar(x[good], tau[good] / tau_r0, yerr=error[good] / tau_r0,
                marker="o", color="#e77500", linewidth=1, capsize=2)
    if np.any(good):
        ax.set_yscale("log")
    ax.set(xticks=x, xticklabels=labels, xlabel=xlabel,
           ylabel=r"Fitted persistence, $\tau_s/\tau_R^0$", title=title)
    save_plot(fig, output / "bond_persistence_vs_epsilon")


def analyze_case(manifest: dict, case_index: int, workers: int) -> None:
    case = manifest["cases"][case_index]
    tasks = [t for t in manifest["tasks"] if t["case_index"] == case_index]
    output = Path(manifest["analysis_root"]) / case["case_id"]
    output.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=workers)(
        delayed(analyze_replicate)(task, case, manifest) for task in tasks
    )
    write_csv(output / "replicate_results.csv", [r["row"] for r in results])
    grouped = defaultdict(list)
    for result in results:
        grouped[result["row"]["epsilon"]].append(result)
    summaries = []
    for epsilon, group in sorted(grouped.items()):
        if len(group) != manifest["n_replicates"]:
            raise RuntimeError(f"Incomplete replicate set for epsilon={epsilon}")
        row = {
            "case_id": case["case_id"], "smooth_kappa": case["smooth_kappa"],
            "smooth_beta": case["smooth_beta"], "epsilon": epsilon,
            "n_replicates": len(group), "n_frames": sum(r["row"]["n_frames"] for r in group),
            "frame_dt_tau_LJ": group[0]["row"]["frame_dt_tau_LJ"],
            "observed_span_tau_LJ": group[0]["row"]["observed_span_tau_LJ"],
        }
        for metric in ["gelation_mean", "p_open_mean", "rate_assoc", "rate_dissoc", "tau_s"]:
            mean, error, count = mean_stderr([r["row"][metric] for r in group])
            name = metric.removesuffix("_mean")
            row.update({f"{name}_mean": mean, f"{name}_stderr": error, f"{name}_n_finite": count})
        time = group[0]["cs_time"]
        if not all(np.array_equal(time, r["cs_time"]) for r in group):
            raise RuntimeError(f"Correlation lag grids differ for epsilon={epsilon}")
        curves = np.stack([r["cs"] for r in group])
        fit, error, count = bootstrap_persistence(time, curves, 20261006 + int(epsilon))
        row.update({
            "tau_s_fit_mean_correlation": fit, "tau_s_fit_bootstrap_stderr": error,
            "tau_s_bootstrap_n_finite": count,
            "tau_s_fit_over_observed_span": fit / row["observed_span_tau_LJ"],
        })
        summaries.append(row)
        epsilon_output = output / f"eps_{epsilon:g}"
        epsilon_output.mkdir(exist_ok=True)
        curve_rows = [
            {"time": float(t), "mean": float(np.mean(curves[:, i])),
             "stderr": float(np.std(curves[:, i], ddof=1) / np.sqrt(len(curves)))}
            for i, t in enumerate(time)
        ]
        write_csv(epsilon_output / "bond_correlation.csv", curve_rows)
        write_csv(epsilon_output / "bond_correlation_replicates.csv", [
            {"time": float(t), **{f"rep_{g['row']['replicate']:03d}": float(g["cs"][i]) for g in group}}
            for i, t in enumerate(time)
        ])
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "gelation_summary.csv", [
        {k: row[k] for k in ["epsilon", "gelation_mean", "gelation_stderr", "n_replicates", "n_frames"]}
        for row in summaries
    ])
    write_csv(output / "bond_persistence_times.csv", [
        {k: row[k] for k in ["epsilon", "tau_s_fit_mean_correlation", "tau_s_fit_bootstrap_stderr",
                             "tau_s_mean", "tau_s_stderr", "observed_span_tau_LJ",
                             "tau_s_fit_over_observed_span", "n_replicates"]}
        for row in summaries
    ])
    plot_cell(summaries, case, output, manifest["tau_R0"])
    (output / "analysis_metadata.json").write_text(json.dumps({
        "case": case, "source_manifest": manifest["manifest_path"],
        "uncertainty": "Independent-replicate SEM; fitted mean-correlation persistence uses 250 replicate bootstrap resamples.",
        "persistence_definition": "Endpoint pair-state correlation, initial semilog-linear exponential fit; identical melt estimator.",
        "turnover_definition": "New saved-frame bonds classified using prior-frame open states; rates per prior open-sticker time, plotted per tau_R0.",
        "gelation_definition": "(1 - p_open - 1/(f-1))/(1/(f-1)), identical melt mean-field indicator.",
        "status": "complete",
    }, indent=2) + "\n")


def combine_cases(manifest: dict) -> None:
    rows = []
    for case in manifest["cases"]:
        path = Path(manifest["analysis_root"]) / case["case_id"]
        if json.loads((path / "analysis_metadata.json").read_text())["status"] != "complete":
            raise RuntimeError(f"Incomplete analysis: {path}")
        with (path / "summary.csv").open() as handle:
            rows.extend(csv.DictReader(handle))
    output = Path(manifest["analysis_root"])
    write_csv(output / "all_grid_summary.csv", rows)
    reference = {float(r["epsilon"]): r for r in rows
                 if float(r["smooth_kappa"]) == 0.05 and float(r["smooth_beta"]) == 1.0}
    comparisons = []
    for row in rows:
        ref = reference[float(row["epsilon"])]
        compared = {k: row[k] for k in ["case_id", "smooth_kappa", "smooth_beta", "epsilon"]}
        for metric in ["gelation_mean", "rate_assoc_mean", "rate_dissoc_mean", "tau_s_fit_mean_correlation"]:
            value, baseline = float(row[metric]), float(ref[metric])
            compared[f"{metric}_difference"] = value - baseline
            compared[f"{metric}_ratio"] = value / baseline if baseline else float("nan")
        comparisons.append(compared)
    write_csv(output / "comparison_to_default.csv", comparisons)
    (output / "completion.json").write_text(json.dumps({
        "status": "complete", "n_cases": len(manifest["cases"]), "n_simulations": len(manifest["tasks"]),
        "summary_rows": len(rows), "manifest": manifest["manifest_path"],
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.combine:
        combine_cases(manifest)
    else:
        index = args.case_index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= index < len(manifest["cases"]):
            parser.error("case index outside the manifest")
        if args.workers < 1:
            parser.error("workers must be positive")
        analyze_case(manifest, index, args.workers)


if __name__ == "__main__":
    main()
