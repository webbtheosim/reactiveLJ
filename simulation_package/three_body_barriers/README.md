# Three-body energetics and effective valence

These scripts calculate Figure 2 from the full smoothed ReactiveLJ energy.
They include all three pair interactions and the leave-one-out coordination
terms. Energies are normalized by the actual isolated-dimer well depth U,
including the coordination regularization.

## Plot the published data

With the [analysis dependencies](../REVISION_STUDIES.md) installed:

```bash
python plot_three_body.py
```

The script reads `figure_barriers.json` (225 points) and writes SVG, PDF, and
PNG versions of the figure. Panel (b) is evaluated analytically using
`f(z) = z (2z-1)^(-p)`. Here z is neighbors per sticker, not excess coordination.
The energy per sticker is `-U f(z)/2`; larger f means more favorable binding.

## Recompute the barriers

The independent energy evaluator needs NumPy, SciPy, and Numba. Only the
optional `validate` stage imports the custom HOOMD build.

```bash
python verify_barriers.py validate   # compare energies/forces against HOOMD
python verify_barriers.py screen
python verify_barriers.py refine
python verify_barriers.py fine
python refine_saddles.py
python generate_figure_data.py
python plot_three_body.py
```

Run the numerical stages on an allocated CPU node. The finest grid uses
roughly 87 million vertices and needs several GB of RAM; 8 GB was requested
in the original calculations. Screening and figure generation checkpoint
completed rows in JSON and skip existing rows. To recompute from scratch,
use a separate copy with `grid_results.json`, `figure_barriers.json`, and
`*_path.npz` removed. Keep or regenerate `saddle_results.json` before generating
the complete figure.

The grid spans the three pair distances from 0.95 to 1.5 sigma, augmented
with the exact LJ minimum. All combinations satisfy the triangle inequalities.
The short-distance boundary is repulsive above the dissociation ceiling;
separations beyond the cutoff do not change the energy. Connectivity of
sublevel sets identifies the lowest threshold joining the two dimer states.
The initial spacing is 0.01 sigma. Continuous paths and alternative grid
connectivity provide discretization checks. Positive barriers are refined
with stationary-point searches and an index-one Hessian check, or with finer
grids where a saddle does not describe the selected path adequately.

For the default coordination window and p = 2, 4, and 8, a spacing of
0.00125 sigma gives continuous-path barriers within 2e-5 U of the refined
values 0.635966, 0.887084, and 0.924577 U. The corresponding path arrays,
`grid_results.json`, `saddle_results.json`, and `verification_summary.json`
record these checks. `half_U_mapping.json` gives p = 1.532396 for the half-U
barrier. The published validation records document the original calculation;
rerunning validation produces a new report for the local build.

These are isolated-cluster potential-energy barriers. The scripts do not
calculate melt activation free energies or chemical rate constants.
