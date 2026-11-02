# Revision studies

These add-ons contain the implementation, run settings, analysis code, and
numeric figure data for the ReactiveLJ manuscript revisions.

| Study | Contents |
| --- | --- |
| [ReactiveAH comparison](melt/reactive_ah_comparison_20261029) | Separate CPU/CUDA force extension; 24 matched melt runs; replicate caches and analysis |
| [Smoothing sweep](melt/smoothing_sweep_20261006) | 252 melt runs over nine smoothing settings; replicate summaries and pair correlations |
| [Three-body energetics](three_body_barriers) | Full three-distance exchange paths, saddle refinement, coordination curves, and Figure 2 data |

The force equations, frozen simulation runners, analysis estimators, and
numeric caches are the versions used for the revision. Release changes
resolve paths relative to each campaign, make environment selection portable,
and allow cached plotting without a locally compiled GPU extension.
The original study scripts and the repository's HOOMD submodule pin are unchanged.

## Replot the published data

From the repository root, in a Python environment with the analysis dependencies:

```bash
python -m pip install -r simulation_package/revision-analysis-requirements.txt
python simulation_package/melt/reactive_ah_comparison_20261029/code/analyze_comparison.py --plots-only
python simulation_package/melt/smoothing_sweep_20261006/plot_figures.py
python simulation_package/three_body_barriers/plot_three_body.py
```

HOOMD and a GPU are not required for these commands. The ReactiveAH caches
contain all 24 replicate arrays. The smoothing release includes all nine
case summaries, 252 replicate results, and replicate pair-correlation curves.
The three-body release includes all 225 barrier values and the convergence
checks. The study READMEs describe how to regenerate these data from simulations
or numerical calculations. Large GSD trajectories and shared initial melts
are not included; the supplied launchers generate new runs with the recorded
parameters and seeds. Cached summaries can be reproduced without those trajectories.

The manuscript figures use Arial at 10 pt. Install Arial locally, or set
`REACTIVELJ_FONT_DIR` to a directory containing your licensed font files.
Without Arial, plotting uses Matplotlib's DejaVu Sans. `REACTIVELJ_FONT` can
select another installed font. This changes typography only.

## Simulation dependency

The revision campaigns used custom HOOMD 6.0.0 at commit
`485771cd90b48b896f1fbb7cec35a8b14c9116ee`. The original fork is private;
the required source is provided in the public
[HOOMD source bundle](../source_dependencies/hoomd/README.md).
The source-only public release has commit ID
`b4c0ad708bfc4ce1ca6aa0b4692b5f8f53b0c395`; its runtime and build source
matches the recorded study version byte for byte. Launchers accept either
this public build or the original recorded build. From the repository root,
prepare a separate source checkout using public upstream HOOMD and the bundle:

```bash
bash source_dependencies/hoomd/checkout.sh revision ../hoomd-revision-source
```

Follow the main README's GPU build instructions inside `../hoomd-revision-source`.
Use a separate Python environment, a compatible CUDA toolkit and CuPy build,
and the recorded analysis dependencies. The study used CUDA 12.9 and A100 GPUs.
The ReactiveAH extension builds against that installed HOOMD package; it does
not replace or modify ReactiveLJ. The small source-license file accompanies
its HOOMD-derived implementation.

The Slurm files retain the original task counts and concurrency limits.
Activate your environment before submission and adjust cluster-specific
resource directives such as `--constraint=nomig` if needed. No jobs are
submitted by the cached plotting commands.
