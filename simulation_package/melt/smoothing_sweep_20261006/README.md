# Melt smoothing-parameter sweep

This campaign repeats the gelation-extension melt protocol at
kappa = 0.025, 0.05, 0.1; beta = 0.5, 1, 2; and epsilon = 0, 3, 6, 9, 12, 15, 18.
Four replicates per condition give 252 simulations. The original submission
limited the array to 11 GPUs at once; that limit is retained in the launcher.

All other parameters are fixed in `manifest.json`. The system has 4000 chains
of 40 beads with four stickers per chain, density 0.85, temperature 1,
p = 4, and full timestep 0.005. Each replicate has 100,000 unsticky steps,
1,000,000 activation steps, 100,000 timestep-restoration steps,
1,000,000 sticky burn-in steps, and 10,000,000 production steps.
Sticker frames are saved every 100,000 steps. The frozen generator is
`code/data_generation/run_reactive_lj.py`.

## Replot cached results

Install the [analysis dependencies](../../REVISION_STUDIES.md), then run:

```bash
python plot_figures.py
```

This uses `results/all_grid_summary.csv` to generate the full-grid gelation,
turnover, and persistence figures under `results`. Each case also includes
replicate summaries and the replicate pair-correlation curves. Means and
standard errors use four independent replicas. The fitted mean-correlation
persistence uses 250 replicate bootstrap resamples. Sampling-limit markers
identify extrapolated fits, not bounds. Gelation is the manuscript's
mean-field indicator; turnover uses saved-frame new bonds and prior
open-sticker exposure.

## Rerun the campaign

Build the exact GPU HOOMD revision in the environment instructions and activate
it before submission. Adjust Slurm resource directives for your cluster.
The manifest paths are relative to this folder and are resolved by the launchers.

```bash
python submit_jobs.py --dry-run
python submit_jobs.py
```

The submitter launches the 252-task GPU array, a dependent nine-task CPU
analysis array, and a final summary job. It saves job IDs to `submission.json`
and refuses to submit a completed submission again. Production outputs go to
`outputs`; the existing cached results are overwritten only when analysis runs.

Preview an individual task without launching a simulation:

```bash
python code/data_generation/run_smoothing_sweep.py --manifest manifest.json --task-id 0 --print-command
```

After trajectories complete, analyze a single case or recombine the case tables:

```bash
python code/analysis/analyze_smoothing_sweep.py --manifest manifest.json --case-index 0 --workers 4
python code/analysis/analyze_smoothing_sweep.py --manifest manifest.json --combine
python plot_figures.py
```
