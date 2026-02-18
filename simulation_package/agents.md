Within this directory, please create a subfolder data_generation, which will handle the code associated with generating data as described in Block 1. For the ReactiveLJ potential between stickers, refer to hoomd-blue/hoomd/md/ReactiveLJForceCompute.cc and hoomd-blue/hoomd/md/many_body.py. Feel free to use as many scripts as appropriate to ensure that the setup code in block 1 is as readable as possible (ensure all code is cleanly documented with human-style comments). Also structure the block 1 code such that a single slurm script, generate_data.slurm, can run the pipeline and generate a specified number of replicates. We should request 12 hours of GPU time per simulation and allow the user to specify a number of replicates (default 10) per reactiveLJ attraction strength. Set up the runs as a job array with 12 jobs running in parallel. Print the runtime of each job so I know how much time is really needed. 16GB of memory should be plenty. Write all outputs to GSD files saving a frame every N simulation steps, default 10_000. Equilibration before production sampling should be at least 2x the polymer relaxation time, which isn't known in advance but to be safe let's equilibrate for 1e7 steps (document this clearly in the comments so I know where to adjust if needed). We do not need to save GSD frames during relaxation steps, but of course, we should be saving frames during production.

Next, create a subfolder analysis, which will handle the code associated with block 2. Again, feel free to split into as many scripts as appropriate. A single slurm script, run_analysis.slurm, should perform the analysis. No GPU is needed here, and 1 hour 2 minutes of runtime should be plenty. For these, we can loop through the GSD files and average over all replicates per reactiveLJ attraction strength in reporting the data. Again, document all steps clearly so I can review what you did and adjust as needed.

The goal of this effort is to investigate whether reactiveLJ phenomenologically captures associative polymer dynamics by comparing metrics across replicates of different reactiveLJ association strength.

% =========================
% Block 1: Simulation setup
% =========================

\subsection{Simulation model and protocol (HOOMD-blue)}
\paragraph{Polymer melt.}
We simulate a dense Kremer--Grest (KG) melt of linear bead--spring chains in reduced Lennard--Jones (LJ) units with
$\sigma=1$, $\epsilon=1$, and bead mass $m=1$.
The system contains $N_{\mathrm{chains}}=4000$ chains of length $N=40$ beads, for a total of
$N_{\mathrm{tot}} = 1.6\times 10^{5}$ beads.
The melt number density is $\rho=0.85\,\sigma^{-3}$ and the temperature is $k_BT=1.0\,\epsilon$.
Periodic boundary conditions are applied in all directions in a cubic simulation box of volume
$V=N_{\mathrm{tot}}/\rho$ and side length $L=V^{1/3}$.

\paragraph{Bonded interactions.}
Adjacent beads along each chain are connected by the standard KG FENE bonds.
Chain semiflexibility is introduced with a harmonic cosine bending potential
\begin{equation}
U_{\mathrm{bend}}(\theta)=k_{\theta}\bigl(1-\cos\theta\bigr),
\end{equation}
with bending stiffness $k_{\theta}=1.5\,\epsilon$.

\paragraph{Nonbonded interactions (baseline melt).}
All non-sticky (backbone) beads interact via a purely repulsive truncated-and-shifted Lennard--Jones potential (WCA form),
with cutoff $r_c=2^{1/6}\sigma$.
Sticky beads (``stickers'') interact with all non-sticky beads using the same WCA repulsion to prevent overlap and maintain melt-like packing.

\paragraph{Sticker placement.}
Each chain contains $f=4$ stickers, evenly spaced along the backbone.
Operationally, we partition each chain into four 10-bead segments and place one sticker on one of the two central beads
of each segment (yielding average spacing $N_s\approx 10$ beads between stickers).

\paragraph{Reactive sticker--sticker interaction (ReactiveLJ).}
Sticker--sticker pairs experience the ReactiveLJ potential described elsewhere in this work, parameterized by
$\sigma=1$ and a sticker attraction strength $\epsilon_{\mathrm{RLJ}}$.
We sweep $\epsilon_{\mathrm{RLJ}} \in \{3,6,9,12,15,18\}$ to probe increasing associative strength.
All other (non-sticker) interactions remain WCA-only.

\paragraph{Integrator, ensemble, and thermostat.}
Simulations are performed in the NVT ensemble using a Langevin thermostat with damping time
$\tau_T = 100\,\tau_{\mathrm{LJ}}$ and time step $\Delta t = 0.005\,\tau_{\mathrm{LJ}}$.
To suppress center-of-mass drift, we subtract the instantaneous center-of-mass velocity every 100 integration steps.

\paragraph{Run length and replicates.}
For each $\epsilon_{\mathrm{RLJ}}$ value, we run production trajectories of $2\times 10^{7}$ time steps.
We perform independent replicate simulations (different random seeds / initial conditions) to enable uncertainty estimates;
the default is $n_{\mathrm{rep}}=10$ unless increased for noisy observables (e.g., stress autocorrelations).
Initialization consists of generating an equilibrated unsticky KG melt at the target $(\rho,T,k_{\theta})$ state point,
then assigning sticker identities and enabling the ReactiveLJ sticker--sticker interaction followed by additional equilibration
before production sampling.

% =========================
% Block 2: Analysis protocol
% =========================

\subsection{Analysis metrics and measurement definitions}
All analyses use a consistent bond-identification rule for sticker pairs and are computed from stored trajectory frames.
Let $\Delta t_{\mathrm{frame}}$ denote the time spacing between saved frames used for bond-based analyses.
Because finite time resolution can miss rapid break/reform events, we choose $\Delta t_{\mathrm{frame}}$
to be sufficiently small (rule of thumb: $\Delta t_{\mathrm{frame}} \lesssim 0.5\,\tau_s$ after a pilot run).

\paragraph{Sticker bond identification.}
Two stickers are classified as \emph{bonded} at time $t$ if their separation $r_{ij}(t)$ is below a threshold $r_{\mathrm{thresh}}$.
We define $r_{\mathrm{thresh}}$ from the two-body sticker--sticker binding potential $V_{\mathrm{ss}}(r)$ as the inflection point,
\begin{equation}
\left.\frac{\partial^2 V_{\mathrm{ss}}(r)}{\partial r^2}\right|_{r=r_{\mathrm{thresh}}}=0,
\end{equation}
which corresponds to the location of maximal restoring force of the attractive interaction.
This rule is used consistently for clusters, $p_{\mathrm{open}}$, lifetimes, exchange rates, and intra/inter bond counts.

\paragraph{Cluster size distribution.}
Construct a graph whose nodes are polymer chains and where an edge between chains $A$ and $B$ exists if any sticker on $A$
is bonded to any sticker on $B$ at the observation time.
Connected components of this graph define clusters; the cluster size $M$ is the number of chains in a component.
We report the cluster size distribution $P(M)$ and summary statistics (e.g., mean cluster size, largest cluster fraction).

\paragraph{Fraction of open stickers and degree of gelation.}
Let $p_{\mathrm{open}}(t)$ be the fraction of stickers that are unpaired at time $t$.
Define the instantaneous bonding probability
\begin{equation}
p(t) = 1 - p_{\mathrm{open}}(t).
\end{equation}
With $f$ stickers per chain (here $f=4$), define the Flory--Stockmayer gel point
\begin{equation}
p_c=\frac{1}{f-1},
\end{equation}
and the mean-field degree of gelation
\begin{equation}
\epsilon(t)=\frac{p(t)-p_c}{p_c}.
\end{equation}
We report time averages $\langle p_{\mathrm{open}}\rangle$, $\langle p\rangle$, and $\langle \epsilon\rangle$.

\paragraph{Intramolecular vs intermolecular bond statistics.}
For each sticker--sticker bond, classify it as intramolecular if both stickers belong to the same chain, otherwise intermolecular.
Report the fraction (or ratio) of intra- to intermolecular bonds.

\paragraph{Monomer mean-squared displacement (MSD).}
Compute the average monomer MSD
\begin{equation}
g_1(t)=\frac{1}{N_{\mathrm{tot}}}\sum_{i=1}^{N_{\mathrm{tot}}}\left\langle
\left|\mathbf r_i(t_0+t)-\mathbf r_i(t_0)\right|^2\right\rangle_{t_0},
\end{equation}
averaged over time origins $t_0$ and all monomers.

\paragraph{Sticker bond lifetime correlation time $\tau_s$.}
For each sticker pair $i$ that forms a bond during the trajectory, define a Boolean bond indicator
$s_i(t)\in\{0,1\}$, where $s_i(t)=1$ if that bond is present at time $t$.
Compute the normalized bond autocorrelation function
\begin{equation}
C_s(\Delta t)=\frac{\left\langle s_i(t_0)\,s_i(t_0+\Delta t)\right\rangle_{i,t_0}}{\left\langle s_i(t_0)\right\rangle_{i,t_0}},
\end{equation}
and extract $\tau_s$ by fitting the long-time decay to
\begin{equation}
C_s(\Delta t)\sim e^{-\Delta t/\tau_s}.
\end{equation}

\paragraph{Brachiation time $\tau_b$.}
Define $b_i(t)=1-s_i(t)$ as the indicator that the sticker is unpaired (``free'').
Compute the analogous correlation function
\begin{equation}
C_b(\Delta t)=\frac{\left\langle b_i(t_0)\,b_i(t_0+\Delta t)\right\rangle_{i,t_0}}{\left\langle b_i(t_0)\right\rangle_{i,t_0}},
\end{equation}
and fit $C_b(\Delta t)\sim e^{-\Delta t/\tau_b}$ to obtain the characteristic time for a free sticker to find a partner.

\paragraph{Connectivity fluctuation relaxation time $\tau_c$.}
Using the connectivity time series $p(t)=1-p_{\mathrm{open}}(t)$, compute the normalized autocorrelation of fluctuations:
\begin{equation}
C_p(\Delta t)=\frac{\left\langle \bigl(p(t_0)-\langle p\rangle\bigr)\bigl(p(t_0+\Delta t)-\langle p\rangle\bigr)\right\rangle_{t_0}}
{\left\langle \bigl(p(t_0)-\langle p\rangle\bigr)^2\right\rangle_{t_0}},
\end{equation}
and extract $\tau_c$ from $C_p(\Delta t)\sim e^{-\Delta t/\tau_c}$.

\paragraph{Viscoelastic network relaxation modulus $G(t)$.}
Compute the equilibrium stress autocorrelation (Green--Kubo) modulus
\begin{equation}
G(\Delta t)=\frac{V}{k_BT}\left\langle \sigma_{\alpha\beta}(t_0+\Delta t)\,\sigma_{\alpha\beta}(t_0)\right\rangle_{t_0},
\qquad \alpha\neq\beta,
\end{equation}
averaging over the three off-diagonal shear components $\alpha\beta\in\{xy,xz,yz\}$ and time origins $t_0$.
(Implementation may use a multi-$\tau$ correlator for efficiency.)

\paragraph{Associative vs dissociative (passive dimerization) exchange rates.}
Let $\Delta t_{\mathrm{frame}}$ be the analysis frame spacing.
Identify \emph{newly formed} sticker--sticker bonds that appear at frame $t$ but were absent at frame $t-\Delta t_{\mathrm{frame}}$.
Classify each newly formed bond as:
(i) \emph{associative-type} if at least one sticker in that bond had a different partner at time $t-\Delta t_{\mathrm{frame}}$;
(ii) otherwise \emph{dissociative-type} (passive dimerization), meaning both stickers were unpaired at the previous frame.
Let $N_m(t-\Delta t_{\mathrm{frame}})$ be the number of unpaired stickers at the previous frame, and let
$N_a(t)$ and $N_d(t)$ be the counts of newly formed bonds (or newly bonded stickers, using a consistent counting convention)
classified as associative-type and dissociative-type over the interval.
Define event rates per unpaired population as
\begin{equation}
R_a(t)=\frac{N_a(t)}{N_m(t-\Delta t_{\mathrm{frame}})\,\Delta t_{\mathrm{frame}}},\qquad
R_d(t)=\frac{N_d(t)}{N_m(t-\Delta t_{\mathrm{frame}})\,\Delta t_{\mathrm{frame}}}.
\end{equation}
We report time-averaged rates $\langle R_a\rangle$ and $\langle R_d\rangle$ and note that finite time resolution
yields lower bounds on true event rates.