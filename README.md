# ReactiveLJ simulations

This repository contains the HOOMD-blue implementation of ReactiveLJ and the
simulation and analysis scripts used in our paper. We examine associative
polymers in single-chain, melt, and semidilute systems using the Kremer-Grest
model, with an MPCD solvent for the semidilute systems.

All quantities are in reduced Lennard-Jones units unless otherwise stated.

## Repository layout

| Path | Purpose |
| --- | --- |
| [hoomd-blue](hoomd-blue) | Git submodule containing the custom HOOMD-blue source |
| [simulation_package/melt](simulation_package/melt) | Melt generation, cleanup, analysis, and plotting |
| [simulation_package/mpcd](simulation_package/mpcd) | MPCD-solution generation, cleanup, analysis, and plotting |
| [simulation_package/single_chain](simulation_package/single_chain) | Single-chain generation and analysis |

The repository includes cached results for the revision studies. Full
simulation trajectories are stored separately.

## Revision studies

The [revision studies guide](simulation_package/REVISION_STUDIES.md) covers the
ReactiveAH comparison, smoothing-parameter sweep, and three-body calculations.
It includes instructions for running these studies and regenerating the
figures from cached data.

## Clone the repository

Clone the repository with its HOOMD-blue source and dependencies:

~~~bash
git clone --recurse-submodules https://github.com/webbtheosim/reactiveLJ.git
cd reactiveLJ
git submodule update --init --recursive
~~~

The HOOMD submodule points to a specific version of
[conservative-reactiveLJ](https://github.com/matthewchertok/conservative-reactiveLJ).
After pulling updates, synchronize it with:

~~~bash
git submodule update --init --recursive
~~~

## Software environment

The simulations use our custom HOOMD build, which provides
`hoomd.md.many_body.ReactiveLJ`. The dependencies are:

- generation: HOOMD-blue, NumPy, Numba, GSD, and CuPy;
- analysis: NumPy, SciPy, GSD, freud, joblib, Matplotlib, and UltraPlot; and
- build tools: CMake, Ninja, a C++17 compiler, Eigen, pybind11, and Python
  development headers.

All three generators import CuPy, including the single-chain CPU workflow.
Install a CuPy version compatible with your CUDA runtime.

### Build the custom HOOMD-blue

Activate your simulation environment and load the compiler and CUDA modules
for your cluster. Then build and install HOOMD in that environment:

~~~bash
cd hoomd-blue
git submodule update --init --recursive

cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(which python)" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DENABLE_GPU=ON \
  -DBUILD_MD=ON \
  -DBUILD_MPCD=ON

cmake --build build --parallel
cmake --install build
cd ..
~~~

For single-chain CPU runs, use `-DENABLE_GPU=OFF`. The melt and MPCD workflows
require a GPU for their virial logging. Additional build options are described
in [hoomd-blue/BUILDING.rst](hoomd-blue/BUILDING.rst).

Check the installed version and ReactiveLJ class with:

~~~bash
python -c "import hoomd; print(hoomd.__file__); print(hoomd.version.version); print(hoomd.md.many_body.ReactiveLJ)"
~~~

ReactiveLJ supports one MPI rank per simulation. The Slurm arrays run
independent replicates, each using one GPU for the melt and MPCD systems.

## ReactiveLJ implementation

### Python interface

ReactiveLJ is defined in
[hoomd-blue/hoomd/md/many_body.py](hoomd-blue/hoomd/md/many_body.py)
and added to the integrator as follows:

~~~python
nlist = hoomd.md.nlist.Cell(buffer=0.4)

reactive = hoomd.md.many_body.ReactiveLJ(
    nlist=nlist,
    reactive_type="sticky",
    sigma=1.0,
    epsilon=18.0,
    r_cut=None,
    weakening_inner=None,
    weakening_outer=None,
    weakening_exponent=4.0,
    smooth_elbow=True,
    smooth_kappa=0.05,
    smooth_beta=1.0,
)

integrator.forces.append(reactive)
~~~

| Parameter | Meaning | Default when omitted |
| --- | --- | --- |
| nlist | HOOMD neighbor list used by the force | required |
| reactive_type | Only this particle type participates | required |
| sigma | Reactive Lennard-Jones length scale | required and positive |
| epsilon | Reactive Lennard-Jones energy scale | required and positive |
| r_cut | Pair cutoff | 1.5 sigma |
| weakening_inner | Inner edge of the coordination taper | 1.3 sigma |
| weakening_outer | Outer edge of the coordination taper | 1.5 sigma |
| weakening_exponent | Crowding-weakening exponent p (nonnegative) | 4 in these workflows |
| smooth_elbow | Enable Hermite smoothing near the LJ elbow | true |
| smooth_kappa | Smoothing-window prefactor | 0.05 |
| smooth_beta | Exponent controlling closure of the smoothing window | 1 |

The coordination window satisfies `weakening_inner < weakening_outer <= r_cut`.
Scalar parameters can be changed after the force is attached; the neighbor
list and reactive particle type remain fixed.

### Implementation files

| File | Responsibility |
| --- | --- |
| [many_body.py](hoomd-blue/hoomd/md/many_body.py) | User-facing Python class, validation, defaults, CPU/GPU dispatch |
| [ReactiveLJForceCompute.h](hoomd-blue/hoomd/md/ReactiveLJForceCompute.h) | CPU class declaration, parameters, scratch arrays, neighbor-list ownership |
| [ReactiveLJForceCompute.cc](hoomd-blue/hoomd/md/ReactiveLJForceCompute.cc) | CPU force/energy/virial calculation and pybind11 export |
| [ReactiveLJForceComputeGPU.h](hoomd-blue/hoomd/md/ReactiveLJForceComputeGPU.h) | GPU class declaration and autotuner state |
| [ReactiveLJForceComputeGPU.cc](hoomd-blue/hoomd/md/ReactiveLJForceComputeGPU.cc) | GPU host wrapper, scratch setup, autotuning, and three kernel launches |
| [ReactiveLJForceComputeGPU.cuh](hoomd-blue/hoomd/md/ReactiveLJForceComputeGPU.cuh) | GPU parameter structure and kernel-launch declarations |
| [ReactiveLJForceComputeGPU.cu](hoomd-blue/hoomd/md/ReactiveLJForceComputeGPU.cu) | CUDA/HIP device functions and coordination, lambda, and force kernels |
| [CMakeLists.txt](hoomd-blue/hoomd/md/CMakeLists.txt) | Adds CPU and GPU sources/headers to the HOOMD MD build |
| [module-md.cc](hoomd-blue/hoomd/md/module-md.cc) | Registers CPU and GPU pybind11 classes in the compiled _md module |

Both implementations use three steps:

1. Compute a smooth local coordination for every reactive particle.
2. Accumulate the derivative of total energy with respect to coordination
   (the lambda field).
3. Evaluate the pair and coordination-dependent forces, energy, and optional
   virial tensor.

The GPU implementation uses three kernels. The CPU uses a half neighbor list,
while the GPU uses a full neighbor list.

The `LiuOConnorTersoffForceCompute` files in the same directory implement the
isotropic Tersoff model used for the computational comparison.

### Non-associating reference

For `epsilon <= 0`, the generators use WCA interactions for all bead pairs
and omit the ReactiveLJ force. Sticker labels are retained for analysis.

## Cluster setup

The Slurm scripts use our Princeton cluster settings. Update the environment,
resource requests, and output paths for your cluster before submitting runs.

### Python environment

The main submission scripts load the environment with:

~~~bash
module load anaconda3/2025.12
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'
~~~

Replace these with your module and environment, for example:

~~~bash
module load <your-anaconda-or-mamba-module>
conda activate /path/to/your/reactiveLJ-environment
~~~

The main scripts are:

- simulation_package/melt/generate_data.slurm
- simulation_package/melt/run_analysis.slurm
- simulation_package/mpcd/generate_data.slurm
- simulation_package/mpcd/run_analysis.slurm
- simulation_package/single_chain/generate_data.slurm
- simulation_package/single_chain/run_analysis.slurm

The plotting, cleaning, validation, and extension scripts use the same
environment. To find paths that need updating:

~~~bash
rg -n "/home/mc7345|/scratch/gpfs/WEBB/(mc7345|chertok)|matthewchertok" \
  simulation_package --glob "*.slurm" --glob "*.py"
~~~

For the virial-extension workflow, `REACTIVELJ_SUPPORT_SITE_PACKAGES` selects
additional Python packages when HOOMD and the analysis dependencies are in
separate environments. Its defaults are in
`simulation_package/melt/data_generation/extend_virial_trajectory.py` and
`simulation_package/melt/extend_simulations_frequent_virial_sampling.slurm`.

### Slurm settings

Adjust the `#SBATCH` settings for your allocation:

- partition/queue and account or allocation;
- GPU resource syntax (for example --gres=gpu:1);
- walltime, memory, CPUs, QoS, and site-specific constraints;
- job-array range and concurrency limit; and
- log output paths.

Melt and MPCD jobs use one GPU per array task. Single-chain jobs default to
`DEVICE=cpu`; for `DEVICE=gpu`, also request a GPU in the Slurm header.

The generator arrays assume five epsilon values and ten replicates:

~~~text
EPSILONS=(0 6 12 15 18)
N_REP=10
tasks = 5 * 10 = 50
array indices = 0-49
~~~

When changing `EPSILONS` or `N_REP`, set the upper `--array` index to
`number_of_epsilons * N_REP - 1`. The value after `%` limits the number of
concurrent tasks.

Runs checkpoint and use `scontrol requeue` near `WALLTIME_CAP_SECONDS`, which
should be shorter than the Slurm walltime. If your cluster does not allow
requeueing, restart from the checkpoint with `--resume`.

### Data locations

Outputs default to each system's `data_generation/outputs` directory. For
melt and single-chain jobs, set `OUTPUT_ROOT` to use another location:

~~~bash
OUTPUT_ROOT=/path/to/project/reactiveLJ/melt sbatch generate_data.slurm
~~~

For MPCD, change the wrapper's `--output-root` argument or pass it directly
to `run_reactive_lj.py`.

Update the cleaning and analysis paths to match. MPCD analysis accepts
`INPUT_ROOT` at submission; melt and single-chain analysis paths are set in
their `run_analysis.slurm` files.

### Submit from the system directory

Create `logs` before submission because Slurm opens its log files before
running the script:

~~~bash
cd simulation_package/melt
mkdir -p logs
sbatch generate_data.slurm
~~~

Use the corresponding directory for MPCD or single-chain runs.

## Simulation protocol and outputs

Each replicate starts from an equilibrated non-associating system. Stickers
are then assigned, ReactiveLJ is turned on for positive epsilon, and the
system is equilibrated further before production. Seeds are determined by
epsilon and replicate index unless `--seed` is supplied.

Production runs are checkpointed in chunks. The generator updates
`metadata.json` after each checkpoint and exits with status 3 near the
walltime limit so that the Slurm wrapper can requeue the job.

Replicate outputs are organized as:

~~~text
data_generation/outputs/
└── eps_<epsilon>/
    └── rep_<three-digit-replicate>/
        ├── metadata.json
        ├── checkpoint.gsd
        └── trajectory.gsd
~~~

Melt and MPCD runs also write `msd_trajectory.gsd` and `virial_tensor_log.gsd`.
Each `metadata.json` records the parameters, sampling intervals, completed
steps, and checkpoint status. Use `--resume` to continue an existing run.

## Melt workflow

### Model and defaults

The melt generator creates 4,000 chains of 40 beads at density 0.85 and
temperature 1, with four evenly spaced stickers per chain. It uses FENE bonds,
a bending stiffness of 1.5, WCA baseline interactions, dt=0.005, and
weakening exponent p=4.

The Slurm script runs epsilon = 0, 6, 12, 15, and 18 with ten replicates each.
Its main defaults are:

- FRAME_STEPS=200000;
- VIRIAL_LOG_STEPS=5000;
- UNSTICKY_EQ_STEPS=100000;
- REACTIVE_EQ_STEPS=1000000;
- STICKY_BURNIN_STEPS=10000000;
- PRODUCTION_TAU_R0=3000;
- MSD_PARTICLES=2000; and
- PRODUCTION_CHUNK_STEPS=1000000.

The main trajectory contains stickers only, the MSD trajectory contains a
fixed sample of monomers, and the virial log contains the configurational
virial tensor at higher frequency.

### Generate melt data

~~~bash
cd simulation_package/melt
mkdir -p logs
sbatch generate_data.slurm
~~~

Parameters can be set at submission, for example:

~~~bash
N_REP=2 N_CHAINS=100 FRAME_STEPS=10000 PRODUCTION_TAU_R0=1 \
  sbatch --array=0-9%2 generate_data.slurm
~~~

This example runs a small test with five epsilon values and two replicates
per value.

To run one replicate within a GPU allocation:

~~~bash
python data_generation/run_reactive_lj.py \
  --epsilon 18 \
  --replicate 1 \
  --n-chains 4000 \
  --output-root data_generation/outputs \
  --weakening-exponent 4 \
  --production-runtime-tau-r0 3000 \
  --resume
~~~

Additional options are listed by
`python data_generation/run_reactive_lj.py --help`.

### Clean restarted trajectories

A restarted run can append frames that overlap an earlier part of the
trajectory. The melt analysis reads `data_generation/outputs_clean`, so first
run the cleaner in a CPU allocation:

~~~bash
cd simulation_package/melt

python data_generation/clean_requeued_outputs.py \
  --source-root data_generation/outputs \
  --clean-root data_generation/outputs_clean \
  --raw-root data_generation/outputs_raw_bad_overlaps \
  --forward-gap-policy segment \
  --workers 16 \
  --apply
~~~

Omit `--apply` to preview the changes. The command preserves the raw data;
`--swap` exchanges the raw and cleaned directories.

The command above handles the standard `eps_*` output layout.
`clean_requeued_outputs.slurm` is configured for the `p_2` and `p_8` sweeps;
edit `TARGET_P_ROOTS` and its command if using it for another layout.

### Analyze melt data

After cleaning:

~~~bash
cd simulation_package/melt
mkdir -p logs
sbatch run_analysis.slurm
~~~

The analysis reads `data_generation/outputs_clean`, uses `p=4` by default,
and writes `analysis/results`. It calculates:

- sticker bonding, valence, persistence, brachiation, and exchange statistics;
- intra- versus intermolecular bonding;
- chain-cluster distributions and gelation measures;
- sampled monomer MSD; and
- stress relaxation from the virial log.

Results include `summary.json`, `summary.csv`, files for each epsilon, and
plots. To select analyses directly:

~~~bash
python analysis/analyze_trajectories.py \
  --input-root data_generation/outputs_clean \
  --output-dir analysis/results \
  --weakening-exponents 4 \
  --analyses bond_statistics cluster_distribution gelation_epsilon
~~~

Valid analysis selections are all, msd, stress_modulus, bond_statistics,
cluster_distribution, and gelation_epsilon.

Additional melt scripts include:

- run_full_fft_stress_relaxation.slurm: full FFT stress-relaxation analysis;
- run_msd_plot.slurm: MSD parameter-sweep plot;
- run_bond_persistence_plot.slurm: persistence time versus weakening exponent;
- run_brachiation_plot.slurm: brachiation time versus weakening exponent;
- make_exchange_rate_plot.slurm: exchange-rate plot from summary.csv; and
- run_tau_s_tau_b_vs_dump_interval.slurm: sampling-interval sensitivity.

## MPCD polymer-solution workflow

### Model and defaults

The MPCD generator uses 40-bead chains with four stickers each. The box is
based on the 4,000-chain melt volume and adjusted to fit the MPCD cell grid;
the number of chains is set by the polymer weight fraction. Defaults are:

- polymer weight fraction 0.02;
- MPCD number density 5;
- MPCD cell size 1;
- solvent mass 1 and monomer mass 5;
- collision period 20 MD steps;
- SRD angle 130 degrees; and
- weakening exponent p=4.

The epsilon values, replicate count, equilibration stages, production length,
sampling intervals, and job-array setup follow the melt workflow.

### Generate MPCD data

~~~bash
cd simulation_package/mpcd
mkdir -p logs
sbatch generate_data.slurm
~~~

Common overrides include POLYMER_WEIGHT_FRACTION, MPCD_NUMBER_DENSITY,
COLLISION_PERIOD, FRAME_STEPS, VIRIAL_LOG_STEPS, PRODUCTION_TAU_R0,
WEAKENING_EXPONENT, and N_REP.

For one replicate within a GPU allocation:

~~~bash
python data_generation/run_reactive_lj.py \
  --epsilon 18 \
  --replicate 1 \
  --output-root data_generation/outputs \
  --polymer-weight-fraction 0.02 \
  --mpcd-number-density 5 \
  --collision-period 20 \
  --production-runtime-tau-r0 3000 \
  --resume
~~~

### Clean and analyze MPCD data

Clean restarted MPCD trajectories with:

~~~bash
cd simulation_package/mpcd
mkdir -p logs
sbatch clean_requeued_outputs.slurm
~~~

This writes `data_generation/outputs_clean` and preserves the raw data.
Use `APPLY=0` to preview the changes; `SWAP=1` exchanges the raw and cleaned
directories.

Run the main analysis with:

~~~bash
sbatch run_analysis.slurm
~~~

For cleaned data stored elsewhere:

~~~bash
INPUT_ROOT=/path/to/mpcd/outputs_clean sbatch run_analysis.slurm
~~~

The analysis reads the trajectories and metadata, averages replicates by
epsilon, and writes `analysis/results`. Outputs include `summary.json`,
`summary.csv`, and properties, correlations, and plots for each epsilon.
MSD and stress analyses use `msd_trajectory.gsd` and `virial_tensor_log.gsd`
when present. Other analyses cover bonding, gelation, cluster sizes,
turnover, and sticker-state correlations.

Additional wrappers include run_full_fft_stress_relaxation.slurm,
run_bond_persistence_vs_time_lag.slurm, and
run_tau_s_tau_b_vs_dump_interval.slurm.

## Single-chain workflow

### Model and defaults

The single-chain generator uses one 400-bead chain with 40 evenly spaced
stickers in a cubic box of side length 100. It defaults to CPU execution
and the same five epsilon values with ten replicates. Its main defaults are:

- FRAME_STEPS=10000;
- UNSTICKY_EQ_STEPS=100000;
- REACTIVE_EQ_STEPS=1000000;
- PRE_PROD_EQ_STEPS=5000000;
- PROD_STEPS=100000000;
- CHAIN_LENGTH=400;
- STICKERS_PER_CHAIN=40;
- BOX_LENGTH=100;
- DEVICE=cpu; and
- WEAKENING_EXPONENT=4.

For single-chain runs, `trajectory.gsd` contains all beads and is used for
MSD analysis. There are no separate MSD or virial trajectories.

### Generate single-chain data

~~~bash
cd simulation_package/single_chain
mkdir -p logs
sbatch generate_data.slurm
~~~

For one replicate in an allocation:

~~~bash
python data_generation/run_reactive_lj.py \
  --epsilon 18 \
  --replicate 1 \
  --chain-length 400 \
  --stickers-per-chain 40 \
  --box-length 100 \
  --device cpu \
  --production-steps 100000000 \
  --output-root data_generation/outputs \
  --resume
~~~

### Analyze single-chain data

The single-chain analysis reads the raw outputs directly:

~~~bash
cd simulation_package/single_chain
mkdir -p logs
sbatch run_analysis.slurm
~~~

Set MAX_LAG_FRAMES before submission to change the correlation window. The
analysis aggregates replicates by epsilon and writes summary.json,
summary.csv, per-epsilon property/correlation files, and plots of:

- bonded and total pair counts;
- associative, dissociative, and partner-swap rates;
- sticker persistence and brachiation times;
- radius of gyration and end-to-end distance;
- sticker valence fractions; and
- loop/domain-size distributions.

run_tau_s_tau_b_vs_dump_interval.slurm performs the corresponding
sampling-interval sensitivity analysis.

## Monitoring and troubleshooting

Check job status and follow a log with:

~~~bash
squeue -u "$USER"
tail -f logs/generate_data_<job>_<task>.out
~~~

To check the installation and locate run metadata:

~~~bash
# Confirm the custom force is installed.
python -c "import hoomd; print(hoomd.__file__); print(hasattr(hoomd.md.many_body, 'ReactiveLJ'))"

# Inspect completion states without loading GSD trajectories.
find data_generation/outputs -name metadata.json -print

# List every remaining cluster-specific filesystem path.
rg -n "/home/|/scratch/|conda activate|module load" . \
  --glob "*.slurm" --glob "*.py" \
  --glob "!hoomd-blue/build*/**"
~~~

If a run or analysis stops, check the following:

- ReactiveLJ is missing: the job imported an upstream or different HOOMD
  installation. Check hoomd.__file__ and rebuild/install into the active
  environment.
- GPU-local force arrays are unavailable: melt/MPCD virial logging was run on
  CPU or with multiple ranks. Use one GPU and one rank.
- Output directory already contains data: use --resume with the same
  configuration, or archive the old replicate directory.
- Existing metadata does not match: an override changed between requeued jobs.
  Resume only with exactly matching physical and sampling parameters.
- SLURM cannot open the log file: create logs before sbatch and submit from the
  system directory.
- Array tasks are skipped or out of range: make #SBATCH --array consistent
  with EPSILONS and N_REP.
- Analysis finds no runs: verify the input root and the expected
  eps_*/rep_*/trajectory.gsd plus metadata.json layout.

Keep `metadata.json` with each replicate, along with the run settings and
repository versions (`git rev-parse HEAD` and `git -C hoomd-blue rev-parse HEAD`).
Retain the raw trajectories while checking cleaned data and analysis results.
If the saved-frame interval changes, repeat the sampling-interval analysis
before comparing sticker correlation times.
