# ReactiveLJ simulations

This repository contains a custom HOOMD-blue implementation of the
coordination-dependent ReactiveLJ potential and the simulation/analysis
workflows used for three associative-polymer systems:

- a dense Kremer-Grest polymer melt;
- a Kremer-Grest polymer solution with MPCD solvent; and
- one dilute Kremer-Grest chain.

All quantities are expressed in reduced Lennard-Jones units unless noted
otherwise.

## Repository layout

| Path | Purpose |
| --- | --- |
| [hoomd-blue](hoomd-blue) | Git submodule containing the custom HOOMD-blue source |
| [simulation_package/melt](simulation_package/melt) | Melt generation, cleanup, analysis, and plotting |
| [simulation_package/mpcd](simulation_package/mpcd) | MPCD-solution generation, cleanup, analysis, and plotting |
| [simulation_package/single_chain](simulation_package/single_chain) | Single-chain generation and analysis |

Generated trajectories and analysis results are intentionally excluded from
Git.

## Clone the repository

Clone with submodules so that the custom HOOMD-blue source and its nested
dependencies are present:

~~~bash
git clone --recurse-submodules https://github.com/webbtheosim/reactiveLJ.git
cd reactiveLJ
git submodule update --init --recursive
~~~

The parent repository records one exact commit of
https://github.com/matthewchertok/conservative-reactiveLJ. To update the
checkout to the recorded commit after pulling the parent repository, run:

~~~bash
git submodule update --init --recursive
~~~

## Software environment

The generator scripts require the custom HOOMD build in this repository.
Installing an upstream HOOMD release will not provide
hoomd.md.many_body.ReactiveLJ.

The workflows also import the following Python packages:

- generation: HOOMD-blue, NumPy, Numba, GSD, and CuPy;
- analysis: NumPy, SciPy, GSD, freud, joblib, Matplotlib, and UltraPlot; and
- build tools: CMake, Ninja, a C++17 compiler, Eigen, pybind11, and Python
  development headers.

CuPy is imported by the current generator modules, including the single-chain
module, so install a CuPy build compatible with the cluster CUDA runtime even
when the single-chain SLURM job uses the CPU device.

### Build the custom HOOMD-blue

Activate the environment that will run the simulations, load the compiler and
CUDA modules required by the local cluster, and build against that same Python
executable:

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

For a CPU-only build, use -DENABLE_GPU=OFF. The melt and MPCD production
scripts request GPUs and their custom virial logger requires a GPU, so a
CPU-only build is primarily useful for the single-chain workflow and testing.
See [hoomd-blue/BUILDING.rst](hoomd-blue/BUILDING.rst) for compiler, CUDA/HIP,
MPI, and architecture-specific CMake options.

Confirm that Python imports the intended build:

~~~bash
python -c "import hoomd; print(hoomd.__file__); print(hoomd.version.version); print(hoomd.md.many_body.ReactiveLJ)"
~~~

The final expression must print the ReactiveLJ class. Check the reported
hoomd.__file__ carefully when multiple HOOMD installations are available.

The current ReactiveLJ force supports only single-rank simulations. Do not
launch the generator with multiple MPI ranks or multiple GPUs for one
replicate. Parallelism is provided by the SLURM job arrays, with one
independent replicate per array task.

## ReactiveLJ implementation

### Python interface

The public interface is
hoomd.md.many_body.ReactiveLJ, implemented in
[hoomd-blue/hoomd/md/many_body.py](hoomd-blue/hoomd/md/many_body.py).
The three generator scripts all construct it in the same way:

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

The parameters are:

| Parameter | Meaning | Default when omitted |
| --- | --- | --- |
| nlist | HOOMD neighbor list used by the force | required |
| reactive_type | Only this particle type participates | required |
| sigma | Reactive Lennard-Jones length scale | required and positive |
| epsilon | Reactive Lennard-Jones energy scale | required and positive |
| r_cut | Pair cutoff | 1.5 sigma |
| weakening_inner | Inner edge of the coordination taper | 1.3 sigma |
| weakening_outer | Outer edge of the coordination taper | 1.5 sigma |
| weakening_exponent | Non-negative coordination attenuation exponent p | 4 in these workflows |
| smooth_elbow | Enable Hermite smoothing near the LJ elbow | true |
| smooth_kappa | Smoothing-window prefactor | 0.05 |
| smooth_beta | Exponent controlling closure of the smoothing window | 1 |

weakening_outer must be larger than weakening_inner and no larger than r_cut.
The neighbor list and reactive particle type cannot be changed after the force
is attached. Scalar parameters can be updated after attachment.

### Source-file map

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

The CPU and GPU paths implement the same three conceptual stages:

1. Compute a smooth local coordination for every reactive particle.
2. Accumulate the derivative of total energy with respect to coordination
   (the lambda field).
3. Evaluate the pair and coordination-dependent forces, energy, and optional
   virial tensor.

The GPU implementation performs these as three tuned kernels. The CPU
implementation uses half neighbor-list storage; GPU execution requires full
neighbor-list storage. Both implementations reject domain-decomposed
multi-rank simulations.

The LiuOConnorTersoffForceCompute files in the same directory implement the
separate isotropic Tersoff comparison model. They are not required by the
ReactiveLJ workflows described below.

### Epsilon-zero behavior

The ReactiveLJ Python class requires positive epsilon. Consequently, all three
generator scripts handle epsilon less than or equal to zero by not constructing
ReactiveLJ. Sticky-sticky interactions then use the same WCA repulsion as all
other bead pairs. Sticker type labels remain in the trajectory for analysis,
but the beads are dynamically non-associating.

## Adapt the workflows to another HPC cluster

Do this before submitting any job. The checked-in SLURM scripts contain
Princeton-specific environment settings.

### 1. Replace the Conda environment

Every primary entry point currently contains:

~~~bash
module load anaconda3/2025.12
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'
~~~

Replace both lines with the module and environment used on the target cluster.
For example:

~~~bash
module load <your-anaconda-or-mamba-module>
conda activate /path/to/your/reactiveLJ-environment
~~~

At minimum, change these six files:

- simulation_package/melt/generate_data.slurm
- simulation_package/melt/run_analysis.slurm
- simulation_package/mpcd/generate_data.slurm
- simulation_package/mpcd/run_analysis.slurm
- simulation_package/single_chain/generate_data.slurm
- simulation_package/single_chain/run_analysis.slurm

The auxiliary plotting, cleaning, validation, and extension SLURM scripts also
contain the same path. Find every remaining user-specific path with:

~~~bash
rg -n "/home/mc7345|/scratch/gpfs/WEBB/(mc7345|chertok)|matthewchertok" \
  simulation_package --glob "*.slurm" --glob "*.py"
~~~

Do not replace the URL in .gitmodules: that URL identifies the public
HOOMD-blue submodule rather than a filesystem path.

One Python source file has an additional fallback that must be changed if the
virial-extension workflow is used:

- simulation_package/melt/data_generation/extend_virial_trajectory.py:
  REACTIVELJ_SUPPORT_SITE_PACKAGES;
- simulation_package/melt/extend_simulations_frequent_virial_sampling.slurm:
  the default REACTIVELJ_SUPPORT_SITE_PACKAGES value.

Those values must point to a compatible site-packages directory only when a
split custom-HOOMD/support-package environment is actually needed.

### 2. Adjust scheduler directives

Review the #SBATCH header in every submitted script:

- partition/queue and account or allocation;
- GPU resource syntax (for example --gres=gpu:1);
- walltime, memory, CPUs, QoS, and site-specific constraints;
- job-array range and concurrency limit; and
- log output paths.

Melt and MPCD generation require one GPU per array task. The checked-in
single-chain generator defaults to DEVICE=cpu and does not request a GPU. If
setting DEVICE=gpu for that workflow, also add the cluster's GPU partition and
GPU resource directives.

The generator arrays assume five epsilon values and ten replicates:

~~~text
EPSILONS=(0 6 12 15 18)
N_REP=10
tasks = 5 * 10 = 50
array indices = 0-49
~~~

If EPSILONS or N_REP changes, update --array so that its upper bound is
number_of_epsilons * N_REP - 1. The value after % is only the maximum number of
concurrent tasks.

The generation scripts use scontrol requeue when they approach their soft
walltime cap. Confirm that the cluster permits user requeue operations. Keep
WALLTIME_CAP_SECONDS shorter than the requested #SBATCH walltime. If requeue is
unavailable, submit shorter runs manually with --resume or adapt the wrapper to
resubmit checkpointed tasks.

### 3. Choose data locations

The primary scripts derive code paths from their own location; no
Matthew-specific repository path is needed. By default, large outputs go under
each system's data_generation/outputs directory.

For melt and single-chain jobs, set OUTPUT_ROOT at submission time to place
data on project or scratch storage:

~~~bash
OUTPUT_ROOT=/path/to/project/reactiveLJ/melt sbatch generate_data.slurm
~~~

The MPCD wrapper currently passes
$WORK_DIR/data_generation/outputs explicitly. To relocate MPCD data, edit its
--output-root argument or run run_reactive_lj.py directly with a different
--output-root.

If outputs are moved, also change the input roots in the corresponding cleaner
and analysis wrappers. MPCD run_analysis.slurm accepts INPUT_ROOT as an
environment variable. Melt and single-chain run_analysis.slurm currently use
paths under their own WORK_DIR and must be edited when data lives elsewhere.

### 4. Create log directories before submission

SLURM opens the output file before the script body runs. Create each log
directory first and submit from the system directory:

~~~bash
cd simulation_package/melt
mkdir -p logs
sbatch generate_data.slurm
~~~

Repeat for mpcd and single_chain. Submitting from these directories also keeps
relative SLURM log paths in the expected location.

## Common simulation behavior

Each generator:

- creates one deterministic replicate from epsilon and replicate index unless
  --seed is supplied;
- equilibrates a non-associating system;
- assigns evenly spaced sticker types;
- ramps/enables ReactiveLJ only when epsilon is positive;
- performs an additional no-output equilibration/burn-in;
- writes production data in chunks;
- updates metadata.json after every checkpoint;
- exits with status 3 near the soft walltime cap; and
- is requeued by the SLURM wrapper until metadata reports completion.

For all systems, output directories follow:

~~~text
data_generation/outputs/
└── eps_<epsilon>/
    └── rep_<three-digit-replicate>/
        ├── metadata.json
        ├── checkpoint.gsd
        └── trajectory.gsd
~~~

Melt and MPCD runs additionally write msd_trajectory.gsd and
virial_tensor_log.gsd. metadata.json is the source of truth for physical
parameters, sampling intervals, run status, completed production steps, and
checkpoint information.

Do not start a non-resume run in a nonempty replicate directory. The scripts
deliberately stop rather than overwrite existing data.

## Melt workflow

### Model and defaults

The melt generator creates 4,000 chains of 40 beads at density 0.85 and
temperature 1, with four evenly spaced stickers per chain. It uses FENE bonds,
a bending stiffness of 1.5, WCA baseline interactions, dt=0.005, and
weakening exponent p=4.

The checked-in SLURM defaults sweep epsilon = 0, 6, 12, 15, and 18 with ten
replicates each. Important wrapper defaults include:

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

Override wrapper parameters through exported environment variables:

~~~bash
N_REP=2 N_CHAINS=100 FRAME_STEPS=10000 PRODUCTION_TAU_R0=1 \
  sbatch --array=0-9%2 generate_data.slurm
~~~

That example is a small workflow test, not a production state point. The array
still has ten tasks because five epsilon values times two replicates equals
ten.

To run one replicate interactively inside an allocated GPU job:

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

Use python data_generation/run_reactive_lj.py --help for every available
override.

### Clean checkpoint/requeue artifacts

Requeueing can append an overlapping trajectory suffix after restart. The
analysis code contains overlap-aware readers, but the standard melt analysis
wrapper intentionally reads data_generation/outputs_clean.

For the direct eps_* layout produced by generate_data.slurm, run the cleaner in
a CPU allocation with:

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

Omit --apply for a dry run. Do not use --swap until the manifest has been
reviewed; --swap renames the raw and clean trees.

Important: the checked-in melt clean_requeued_outputs.slurm currently loops
over p_2 and p_8 parameter-sweep subtrees. It is suitable for that layout but
not the direct eps_* layout unless TARGET_P_ROOTS/the command body is adapted.
The direct Python command above handles the standard generator output.

### Analyze melt data

After cleaning:

~~~bash
cd simulation_package/melt
mkdir -p logs
sbatch run_analysis.slurm
~~~

The wrapper reads data_generation/outputs_clean, analyzes p=4 by default, and
writes analysis/results. It computes the selected families of:

- sticker bonding, valence, persistence, brachiation, and exchange statistics;
- intra- versus intermolecular bonding;
- chain-cluster distributions and gelation measures;
- sampled monomer MSD; and
- stress relaxation from the virial log.

The main products are analysis/results/summary.json,
analysis/results/summary.csv, per-epsilon CSV/JSON files, and publication
plots. To run selected families directly:

~~~bash
python analysis/analyze_trajectories.py \
  --input-root data_generation/outputs_clean \
  --output-dir analysis/results \
  --weakening-exponents 4 \
  --analyses bond_statistics cluster_distribution gelation_epsilon
~~~

Valid analysis selections are all, msd, stress_modulus, bond_statistics,
cluster_distribution, and gelation_epsilon.

Specialized melt wrappers provide targeted or publication-specific reruns:

- run_full_fft_stress_relaxation.slurm: full FFT stress-relaxation analysis;
- run_msd_plot.slurm: MSD parameter-sweep plot;
- run_bond_persistence_plot.slurm: persistence time versus weakening exponent;
- run_brachiation_plot.slurm: brachiation time versus weakening exponent;
- make_exchange_rate_plot.slurm: exchange-rate plot from summary.csv; and
- run_tau_s_tau_b_vs_dump_interval.slurm: sampling-interval sensitivity.

Review each wrapper's input/output paths and Conda activation before use.

## MPCD polymer-solution workflow

### Model and defaults

The MPCD generator uses 40-bead chains with four stickers each in a box based
on the 4,000-chain melt reference volume. It derives the number of chains from
the requested polymer weight fraction and snaps the box length to the MPCD
cell grid. Default solution parameters include:

- polymer weight fraction 0.02;
- MPCD number density 5;
- MPCD cell size 1;
- solvent mass 1 and monomer mass 5;
- collision period 20 MD steps;
- SRD angle 130 degrees; and
- weakening exponent p=4.

The epsilon sweep, replicate count, equilibration stages, production target,
sampling intervals, checkpointing, and job-array mapping otherwise mirror the
melt wrapper.

### Generate MPCD data

~~~bash
cd simulation_package/mpcd
mkdir -p logs
sbatch generate_data.slurm
~~~

Common overrides include POLYMER_WEIGHT_FRACTION, MPCD_NUMBER_DENSITY,
COLLISION_PERIOD, FRAME_STEPS, VIRIAL_LOG_STEPS, PRODUCTION_TAU_R0,
WEAKENING_EXPONENT, and N_REP.

For one allocated-GPU replicate:

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

The MPCD cleaner is already configured for the standard output layout:

~~~bash
cd simulation_package/mpcd
mkdir -p logs
sbatch clean_requeued_outputs.slurm
~~~

It writes data_generation/outputs_clean and leaves the raw tree unchanged by
default. Set APPLY=0 for a dry run. Avoid SWAP=1 until the generated manifest
has been reviewed.

Run the main analysis with:

~~~bash
sbatch run_analysis.slurm
~~~

To analyze a relocated clean tree:

~~~bash
INPUT_ROOT=/path/to/mpcd/outputs_clean sbatch run_analysis.slurm
~~~

The analysis discovers trajectory.gsd and metadata.json recursively, uses
msd_trajectory.gsd and virial_tensor_log.gsd when present, aggregates
replicates by epsilon, and writes analysis/results. Outputs include
summary.json, summary.csv, per-epsilon properties and correlations, and plots
for bonding, gelation, cluster sizes, exchange rates, persistence/brachiation,
sampled-monomer diffusion/MSD, and cluster evolution.

Additional wrappers include run_full_fft_stress_relaxation.slurm,
run_bond_persistence_vs_time_lag.slurm, and
run_tau_s_tau_b_vs_dump_interval.slurm.

## Single-chain workflow

### Model and defaults

The single-chain generator uses one 400-bead chain with 40 evenly spaced
stickers in a cubic box of side length 100. The SLURM wrapper defaults to CPU
execution and sweeps the same five epsilon values with ten replicates. Its
important defaults are:

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

Unlike melt and MPCD, trajectory.gsd contains the entire chain, and no separate
MSD or virial GSD is written.

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

Check array status and logs with the scheduler tools available on the cluster:

~~~bash
squeue -u "$USER"
tail -f logs/generate_data_<job>_<task>.out
~~~

Useful checks include:

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

Common failures:

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

## Reproducibility notes

- Record the parent commit and the hoomd-blue submodule commit with
  git rev-parse HEAD and git -C hoomd-blue rev-parse HEAD.
- Preserve metadata.json with every replicate.
- Record environment exports and any edited #SBATCH directives.
- Keep raw output trees until cleaned-output manifests and analysis results
  have been validated.
- When changing frame spacing, rerun the sampling-interval sensitivity tools
  before comparing bond persistence or brachiation times.
