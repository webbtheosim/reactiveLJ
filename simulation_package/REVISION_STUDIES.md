# Revision studies

These folders contain the code, run settings, and cached results for the
additional studies in our ReactiveLJ revisions.

| Study | Contents |
| --- | --- |
| [ReactiveAH comparison](melt/reactive_ah_comparison_20261029) | ReactiveAH CPU/CUDA implementation and 24 matched melt simulations |
| [Smoothing sweep](melt/smoothing_sweep_20261006) | 252 melt simulations across nine smoothing-parameter combinations |
| [Three-body energetics](three_body_barriers) | Exchange barriers, coordination curves, and data for Figure 2 |

## Regenerate the figures

From the repository root, install the analysis dependencies and run:

```bash
python -m pip install -r simulation_package/revision-analysis-requirements.txt
python simulation_package/melt/reactive_ah_comparison_20261029/code/analyze_comparison.py --plots-only
python simulation_package/melt/smoothing_sweep_20261006/plot_figures.py
python simulation_package/three_body_barriers/plot_three_body.py
```

These commands use the cached data and run on a CPU without HOOMD. The full
GSD trajectories and shared initial melts are not included. Each study's
README describes how to generate new data using the recorded settings and seeds.

The plots use 10 pt Arial when available and DejaVu Sans otherwise. Set
`REACTIVELJ_FONT_DIR` to your font directory, or use `REACTIVELJ_FONT` to
select another installed font.

## Run the simulations

The revision simulations used HOOMD 6.0.0 at commit
`485771cd90b48b896f1fbb7cec35a8b14c9116ee` of
[conservative-reactiveLJ](https://github.com/matthewchertok/conservative-reactiveLJ).
The run scripts check this version. To obtain it:

```bash
git clone https://github.com/matthewchertok/conservative-reactiveLJ.git hoomd-revision-source
git -C hoomd-revision-source checkout 485771cd90b48b896f1fbb7cec35a8b14c9116ee
git -C hoomd-revision-source submodule update --init --recursive
```

Use a separate Python environment and follow the
[HOOMD build instructions](../README.md#build-the-custom-hoomd-blue),
substituting `hoomd-revision-source` for `hoomd-blue`.
We used CUDA 12.9 and A100 GPUs. Install a compatible CuPy version and the
analysis dependencies above. ReactiveAH is built as a separate extension
against this HOOMD installation, as described in its study README.

Before submitting the Slurm scripts, activate your environment and adjust
the resource requests for your cluster, including settings such as
`--constraint=nomig`.
