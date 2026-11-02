# ReactiveAH and ReactiveLJ matched melt comparison

This add-on defines a separate coordination-dependent Ashbaugh–Hatch force
and compares it with ReactiveLJ at epsilon = 3, 12, and 18. Four matched
replicates per strength give 24 simulations. Each pair uses the same saved
unsticky melt and random seed.

## Force definition

Let `u(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6] - LJ(rc)`,
`rm = 2^(1/6) sigma`, and `D = epsilon + LJ(rc)`. Then

```text
U_AH(r, C) = u(r) + [1-W(C)] D     for r <= rm
           = W(C) u(r)            for rm < r < rc
           = 0                   for r >= rc.
```

`W = (1+C_exc)^(-p)`, `C_exc = [raw + sqrt(raw^2 + 10^(-12))]/2`,
and `raw = C_i + C_j - 2 w(r_ij)`. The cosine coordination window is
1.3–1.5 sigma, p = 4, and rc = 1.5 sigma. The CPU and CUDA implementations
include the derivatives of W, including the coordination-dependent shift
on the inner branch. ReactiveAH joins at the LJ minimum without Hermite
smoothing. ReactiveLJ retains kappa = 0.05 and beta = 1.

Both models have the same fixed-coordination minimum, well depth, and outer
attractive branch. At p = 0, both reduce to the same shifted LJ potential.
The retained coordination regularization gives isolated-pair W =
0.9999980000025; the inner-branch difference for that isolated pair is at
most about 1.36e-6 epsilon. Under crowding, ReactiveAH can make some compressed
pair configurations unfavorable relative to separated stickers.

The underlying pair form follows
[Ashbaugh and Hatch, JACS 2008](https://doi.org/10.1021/ja802124e).
The coordination dependence and its force derivatives are defined by this add-on.

## Protocol

| Setting | Value |
| --- | --- |
| Chains / beads per chain / stickers per chain | 4000 / 40 / 4 |
| Density / temperature | 0.85 / 1 |
| Backbone | FENE-WCA bonds, bending stiffness 1.5, WCA nonbonded interactions |
| Timestep | 0.005 after activation |
| Langevin drag | 0.01 |
| Shared unsticky equilibration | 100,000 steps |
| Attraction ramp | 1,000,000 steps at dt = 1e-5, in 1,000-step segments |
| Timestep restoration | 100,000 steps |
| Sticky burn-in | 1,000,000 steps |
| Production | 10,000,000 steps = 50,000 tau |
| Sticker output | Every 100,000 steps; 100 frames per replicate |

`manifest.json` records the complete settings and seeds. `code/reference`
contains the frozen gelation-extension runner and analysis helpers.
`code/matched_runner.py` adds model selection and shared-state initialization.

## Reproduce the cached analysis

See [environment instructions](../../REVISION_STUDIES.md). From this directory:

```bash
python code/analyze_comparison.py --plots-only
```

Results appear under `results/figures`, with numeric summaries under `results`.
The bond cutoff is `(26/7)^(1/6) sigma`. Gelation is
`(phi_pair - 1/(f-1))/(1/(f-1))`, with f = 4. Turnover counts new bonds in
successive frames, classified by previous sticker states and divided by
previous open-sticker exposure. Uncertainty uses the four independent
replicates. Pair persistence is an endpoint correlation; fits use the study's
semilog procedure and 500 replicate bootstrap resamples.

The fitted-time diagnostic is retained for reproducibility although it is
not a manuscript figure. Hollow down/up triangles mark estimates below the
500-tau saved interval or above the 49,500-tau observed span. These are
extrapolated estimates, not bounds. The full correlation curves are provided.
The three-sticker maps sum all pair energies and subtract the isolated AB
dimer plus a separated third sticker; they are not just the AB contribution.

## Build and run new simulations

First install the pinned GPU HOOMD build described in the environment instructions.
Activate that environment and make its CUDA compiler available. Then:

```bash
bash build.sh -DCMAKE_CUDA_ARCHITECTURES=80
```

The extension is written into `code/`; no installed HOOMD files are changed.
Pass another architecture for a different GPU. Additional CMake options can
be appended to `build.sh`. `PYTHON` can select the environment's executable.

From this directory, submit the original dependent stages:

```bash
export AH_COMPARISON_ROOT="$PWD"
mkdir -p logs
validation=$(sbatch --parsable --output=logs/validation_%j.out validate.slurm)
preparation=$(sbatch --parsable --dependency=afterok:${validation} --output=logs/prepare_%A_%a.out prepare.slurm)
workflow=$(sbatch --parsable --dependency=afterok:${preparation} --output=logs/workflow_%j.out workflow_check.slurm)
simulation=$(sbatch --parsable --dependency=afterok:${workflow} --output=logs/simulation_%A_%a.out simulate.slurm)
sbatch --dependency=afterok:${simulation} --output=logs/analysis_%j.out analyze.slurm
```

This sequence uses at most 24 GPUs concurrently. Force validation must pass
on the locally built binary before preparation and simulation. The workflow
check runs a small melt and tests restart and analysis behavior. The recorded
production validation is under `validation/published`; it is historical
evidence, not authorization to skip validation of a new build.

For a task preview, use `python code/run_comparison.py --task-id 0 --dry-run`.
For full analysis after new trajectories complete, use
`python code/analyze_comparison.py --workers 4`.
