// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ReactiveAHForceCompute.h"

#include "hoomd/ParticleData.h"

#include <algorithm>
#include <stdexcept>

namespace hoomd
    {
namespace md
    {

ReactiveAHForceCompute::ReactiveAHForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                               std::shared_ptr<NeighborList> nlist,
                                               const std::string& reactive_type,
                                               Scalar sigma,
                                               Scalar epsilon,
                                               Scalar r_cut,
                                               Scalar weakening_inner,
                                               Scalar weakening_outer,
                                               Scalar weakening_exponent,
                                               bool smooth_elbow,
                                               Scalar smooth_kappa,
                                               Scalar smooth_beta)
    : ForceCompute(sysdef), m_nlist(std::move(nlist)), m_typpair_idx(m_pdata->getNTypes()),
      m_matrix_registered(false), m_reactive_typeid(0), m_sigma(Scalar(1.0)), m_epsilon(Scalar(1.0)),
      m_r_cut(Scalar(1.0)), m_r_cut_sq(Scalar(1.0)), m_neighbor_rcut(Scalar(1.0)),
      m_energy_shift(Scalar(0.0)), m_weakening_inner(Scalar(1.0)), m_weakening_outer(Scalar(1.0)),
      m_weakening_exponent(Scalar(1.0)), m_inverse_span(Scalar(1.0)), m_r_elbow(Scalar(1.0)),
      m_smooth_elbow(smooth_elbow), m_smooth_kappa(smooth_kappa), m_smooth_beta(smooth_beta),
      m_smooth_delta_tol(Scalar(0.0)), m_smooth_r_min(Scalar(0.0))
    {
    if (!m_nlist)
        {
        throw std::invalid_argument("ReactiveAHForceCompute requires a valid neighbor list.");
        }

    m_exec_conf->msg->notice(6) << "Constructing ReactiveAHForceCompute" << std::endl;

    if (m_sysdef->isDomainDecomposed())
        {
        throw std::runtime_error(
            "ReactiveAHForceCompute currently supports only single-rank simulations.");
        }

    // GPU neighbor lists require full storage; CPU runs can leverage half storage for symmetry.
    const bool gpu_exec = m_exec_conf && m_exec_conf->isCUDAEnabled();
    const auto storage_mode = gpu_exec ? NeighborList::full : NeighborList::half;
    m_nlist->setStorageMode(storage_mode);

    // Scratch: coordination C_i
    GPUArray<Scalar> coordination(m_pdata->getMaxN(), m_exec_conf);
    m_coordination.swap(coordination);

    // Scratch: lambda_i = dU_total / dC_i
    // NOTE: requires you to add `GPUArray<Scalar> m_lambda;` to the header.
    GPUArray<Scalar> lambda(m_pdata->getMaxN(), m_exec_conf);
    m_lambda.swap(lambda);

    m_r_cut_matrix = std::make_shared<GPUArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
        {
        ArrayHandle<Scalar> h_matrix(*m_r_cut_matrix,
                                     access_location::host,
                                     access_mode::overwrite);
        std::fill_n(h_matrix.data, m_typpair_idx.getNumElements(), Scalar(0.0));
        }
    m_nlist->addRCutMatrix(m_r_cut_matrix);
    m_matrix_registered = true;

    setReactiveType(reactive_type);
    setParameters(sigma,
                  epsilon,
                  r_cut,
                  weakening_inner,
                  weakening_outer,
                  weakening_exponent,
                  smooth_elbow,
                  smooth_kappa,
                  smooth_beta);
    }

ReactiveAHForceCompute::~ReactiveAHForceCompute()
    {
    if (m_matrix_registered && m_nlist)
        {
        m_nlist->removeRCutMatrix(m_r_cut_matrix);
        m_matrix_registered = false;
        }
    }

void ReactiveAHForceCompute::setParameters(Scalar sigma,
                                           Scalar epsilon,
                                           Scalar r_cut,
                                           Scalar weakening_inner,
                                           Scalar weakening_outer,
                                           Scalar weakening_exponent,
                                           bool smooth_elbow,
                                           Scalar smooth_kappa,
                                           Scalar smooth_beta)
    {
    if (sigma <= Scalar(0.0))
        throw std::invalid_argument("sigma must be positive");
    if (epsilon <= Scalar(0.0))
        throw std::invalid_argument("epsilon must be positive");
    if (r_cut <= Scalar(0.0))
        throw std::invalid_argument("r_cut must be positive");
    if (weakening_exponent < Scalar(0.0))
        throw std::invalid_argument("weakening_exponent must be non-negative");
    if (weakening_outer <= weakening_inner)
        throw std::invalid_argument(
            "weakening_outer must be greater than weakening_inner for a valid taper window");
    if (smooth_kappa < Scalar(0.0))
        throw std::invalid_argument("smooth_kappa must be non-negative");
    if (smooth_beta <= Scalar(0.0))
        throw std::invalid_argument("smooth_beta must be positive");

    m_sigma = sigma;
    m_epsilon = epsilon;
    m_r_cut = r_cut;
    m_r_cut_sq = r_cut * r_cut;
    m_weakening_inner = weakening_inner;
    m_weakening_outer = weakening_outer;
    m_weakening_exponent = weakening_exponent;
    m_inverse_span = Scalar(1.0) / (m_weakening_outer - m_weakening_inner);
    if (smooth_elbow) throw std::invalid_argument("ReactiveAH has no Hermite smoothing");
    if (r_cut <= sigma * Scalar(1.1224620483093729814))
        throw std::invalid_argument("ReactiveAH requires r_cut above the LJ minimum");
    if (weakening_outer > r_cut)
        throw std::invalid_argument("ReactiveAH requires weakening_outer <= r_cut");
    m_smooth_elbow = false;
    m_smooth_kappa = smooth_kappa;
    m_smooth_beta = smooth_beta;

    Scalar sigma_over_rcut = m_sigma / m_r_cut;
    Scalar sigma_over_rcut_6 = slow::pow(sigma_over_rcut, Scalar(6.0));
    m_energy_shift = Scalar(4.0) * m_epsilon
                     * (sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6);

    Scalar quadratic_rhs = sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6;
    Scalar discriminant = Scalar(1.0) + Scalar(4.0) * quadratic_rhs;
    if (discriminant <= Scalar(0.0))
        {
        throw std::invalid_argument(
            "Unable to determine the elbow location; adjust the cutoff radius.");
        }
    Scalar sr6_at_zero = Scalar(0.5) * (Scalar(1.0) + slow::sqrt(discriminant));
    Scalar sr_root = slow::pow(sr6_at_zero, Scalar(1.0 / 6.0));
    m_r_elbow = m_sigma / sr_root;

    m_smooth_delta_tol = Scalar(1e-6) * m_sigma;
    m_smooth_r_min = Scalar(1e-7) * m_sigma;

    m_neighbor_rcut = std::max(m_r_cut, m_weakening_outer);
    updateNeighborListCutoffs();
    }

pybind11::dict ReactiveAHForceCompute::getParameters() const
    {
    pybind11::dict params;
    params["sigma"] = m_sigma;
    params["epsilon"] = m_epsilon;
    params["r_cut"] = m_r_cut;
    params["weakening_inner"] = m_weakening_inner;
    params["weakening_outer"] = m_weakening_outer;
    params["weakening_exponent"] = m_weakening_exponent;
    params["reactive_type"] = m_reactive_type_name;
    params["smooth_elbow"] = m_smooth_elbow;
    params["smooth_kappa"] = m_smooth_kappa;
    params["smooth_beta"] = m_smooth_beta;
    return params;
    }

std::shared_ptr<NeighborList> ReactiveAHForceCompute::getNeighborList() const
    {
    return m_nlist;
    }

const std::string& ReactiveAHForceCompute::getReactiveType() const
    {
    return m_reactive_type_name;
    }

/*! \brief Smooth positive-part for C_exc and its derivative.
 *
 *  We replace C_exc = max(raw, 0) with a C1-smooth approximation:
 *    C_exc(raw) = 0.5 * (raw + sqrt(raw^2 + eps^2))
 *
 *  This removes the kink at raw=0 while keeping C_exc ~ 0 for negative raw.
 *  We also return the "gate" g = dC_exc/draw, which smoothly interpolates [0,1]:
 *    g(raw) = 0.5 * (1 + raw / sqrt(raw^2 + eps^2))
 *
 *  IMPORTANT: "raw" here is the leave-one-out crowding argument:
 *    raw = (C_i - w_ij) + (C_j - w_ij)
 *  In your model, raw has NO r_ij dependence because C_i and C_j include w_ij,
 *  and you subtract it back out. That is the whole point of the leave-one-out design.
 */
inline void ReactiveAHForceCompute::evaluateCexcSmooth(Scalar raw,
                                                       Scalar& C_exc,
                                                       Scalar& gate) const
    {
    // Smoothing scale in "coordination units" (w is dimensionless).
    // Keep this small compared to typical variations in raw.
    const Scalar eps = Scalar(1e-6);

    const Scalar s = slow::sqrt(raw * raw + eps * eps);
    C_exc = Scalar(0.5) * (raw + s);
    gate = Scalar(0.5) * (Scalar(1.0) + raw / s);
    }

/*! \brief Compute dF/dr for the shifted LJ force F(r) = 24 eps / r [2 (sigma/r)^12 - (sigma/r)^6].
 *
 *  evaluateForceMagnitude(r) returns F(r) (signed), despite the name.
 */
inline Scalar ReactiveAHForceCompute::evaluateForceDerivative(Scalar distance) const
    {
    const Scalar r = std::max(distance, m_smooth_r_min);
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar inv_r2 = inv_r * inv_r;

    const Scalar sr = m_sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;

    // dF/dr = 24 eps * inv_r^2 * ( -26 sr12 + 7 sr6 )
    return Scalar(24.0) * m_epsilon * inv_r2 * (Scalar(-26.0) * sr12 + Scalar(7.0) * sr6);
    }

// Analytic coordination derivative of the shifted Ashbaugh-Hatch energy.
Scalar ReactiveAHForceCompute::evaluate_dU_dCexc(Scalar r_in, Scalar C_exc) const
    {
    if (r_in >= m_r_cut) return Scalar(0.0);
    const Scalar W = slow::pow(Scalar(1.0) + C_exc, -m_weakening_exponent);
    const Scalar dW_dC = -m_weakening_exponent * W / (Scalar(1.0) + C_exc);
    const Scalar r_minimum = m_sigma * Scalar(1.1224620483093729814);
    const Scalar depth = m_epsilon + m_energy_shift;
    return ((r_in <= r_minimum) ? -depth : evaluateEnergy(r_in)) * dW_dC;
    }

/*! \brief Derivative w'(r) for the smooth coordination weight w(r).
 *
 *  w(r) =
 *    1,                                   r <= r_inner
 *    0.5 [1 + cos(pi*(r-r_inner)/(r_outer-r_inner))], r_inner < r < r_outer
 *    0,                                   r >= r_outer
 *
 *  This derivative is needed to convert lambda_i = dU/dC_i into a physical force.
 */
Scalar ReactiveAHForceCompute::evaluateWeightDerivative(Scalar distance) const
    {
    if (distance >= m_weakening_outer || distance <= m_weakening_inner)
        {
        return Scalar(0.0);
        }

    const Scalar pi = Scalar(3.141592653589793238462643383279502884L);
    const Scalar fraction = (distance - m_weakening_inner) * m_inverse_span;
    const Scalar angle = pi * fraction;

    // w'(r) = -0.5 * sin(angle) * (pi * m_inverse_span)
    return -Scalar(0.5) * slow::sin(angle) * (pi * m_inverse_span);
    }

void ReactiveAHForceCompute::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);
    ensureScratchSize();

    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                              access_location::host,
                              access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // Scratch arrays
    ArrayHandle<Scalar> h_coord(m_coordination, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_lambda(m_lambda, access_location::host, access_mode::overwrite);

    const unsigned int N = m_pdata->getN();
    const BoxDim box = m_pdata->getGlobalBox();
    const size_t virial_pitch = m_virial.getPitch();
    const bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    PDataFlags flags = m_pdata->getFlags();
    const bool compute_virial = flags[pdata_flag::pressure_tensor];

    m_force.zeroFill();
    m_virial.zeroFill();

    std::fill_n(h_coord.data, m_coordination.getNumElements(), Scalar(0.0));
    std::fill_n(h_lambda.data, m_lambda.getNumElements(), Scalar(0.0));

    unsigned int reactive_count = 0;
    const Scalar neighbor_cut_sq = m_neighbor_rcut * m_neighbor_rcut;

    // -------------------------------------------------------------------------
    // PASS 1: Compute local coordination C_i = sum_k w(r_ik) for reactive beads.
    //
    // This uses the neighbor list out to neighbor_rcut = max(r_cut, weakening_outer)
    // so that all contributors to crowding are included.
    // -------------------------------------------------------------------------
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const unsigned int type_i = __scalar_as_int(pos_i.w);
        if (type_i != m_reactive_typeid)
            {
            continue;
            }

        reactive_count++;

        const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];

        for (unsigned int k = 0; k < n_neigh; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            if (j >= N)
                {
                continue;
                }

            const Scalar4 pos_j = h_pos.data[j];
            const unsigned int type_j = __scalar_as_int(pos_j.w);
            if (type_j != m_reactive_typeid)
                {
                continue;
                }

            Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
            dx = box.minImage(dx);
            const Scalar rsq = dot(dx, dx);
            if (rsq >= neighbor_cut_sq)
                {
                continue;
                }

            const Scalar r = slow::sqrt(rsq);
            const Scalar weight = computeWeight(r);
            if (weight == Scalar(0.0))
                {
                continue;
                }

            h_coord.data[i] += weight;
            if (third_law)
                {
                // In half-list mode, we must manually add symmetric contribution
                h_coord.data[j] += weight;
                }
            }
        }

    if (reactive_count == 0)
        {
        return;
        }

    // -------------------------------------------------------------------------
    // PASS 2: Compute lambda_i = dU_total / dC_i (EAM-style adjoint).
    //
    // For each energy-bearing pair (r < r_cut), compute dU_ij/dC_exc and add to both endpoints:
    //   lambda_i += dU_ij/dC_exc
    //   lambda_j += dU_ij/dC_exc
    //
    // dU/dC_exc is fully analytic (uses both Ashbaugh-Hatch branches).
    //
    // We enforce "pair-once" processing with j > i, regardless of neighbor-list storage mode.
    // -------------------------------------------------------------------------
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const unsigned int type_i = __scalar_as_int(pos_i.w);
        if (type_i != m_reactive_typeid)
            {
            continue;
            }

        const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];
        const Scalar Ci = h_coord.data[i];

        for (unsigned int k = 0; k < n_neigh; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            if (j >= N || j <= i)
                {
                continue; // pair-once
                }

            const Scalar4 pos_j = h_pos.data[j];
            const unsigned int type_j = __scalar_as_int(pos_j.w);
            if (type_j != m_reactive_typeid)
                {
                continue;
                }

            Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
            dx = box.minImage(dx);
            const Scalar rsq = dot(dx, dx);

            // Energy exists only for r < r_cut (must match Pass 3 pair energy).
            if (rsq >= m_r_cut_sq)
                {
                continue;
                }

            const Scalar r = slow::sqrt(rsq);

            // leave-one-out subtraction uses w_ij when r < weakening_outer; otherwise w_ij = 0
            Scalar w_ij = Scalar(0.0);
            if (r < m_weakening_outer)
                {
                w_ij = computeWeight(r);
                }

            const Scalar Cj = h_coord.data[j];

            // Leave-one-out crowding argument (raw) and smooth positive-part C_exc.
            // With leave-one-out, raw has no r_ij dependence (C_i contains w_ij; we subtract it).
            const Scalar raw = (Ci - w_ij) + (Cj - w_ij);

            Scalar C_exc = Scalar(0.0);
            Scalar gate = Scalar(0.0); // gate = dC_exc/draw in [0,1]
            evaluateCexcSmooth(raw, C_exc, gate);

            // Fully analytic dU/dC_exc (uses both Ashbaugh-Hatch branches).
            const Scalar dU_dCexc = evaluate_dU_dCexc(r, C_exc);

            // lambda_i = dU_total/dC_i.
            // For this pair: dU/dC_i = (dU/dC_exc) * (dC_exc/draw) * (draw/dC_i).
            // Here draw/dC_i = 1, so the factor is just gate.
            const Scalar contrib = gate * dU_dCexc;

            h_lambda.data[i] += contrib;
            h_lambda.data[j] += contrib;
            }
        }

    // -------------------------------------------------------------------------
    // PASS 3: Compute forces, energies, and virials.
    //
    // Total force = pair force + coordination (many-body) force.
    //
    // Pair force is computed as -dU_ij/dr holding C_exc fixed (no "forgotten chain rule").
    //
    // Many-body coordination force uses lambda_i = dU/dC_i:
    //   F_coord comes from - sum_i lambda_i * grad(C_i)
    //
    // BUT your C_exc is "leave-one-out": it subtracts w_ij from both Ci and Cj.
    // That implies the *current i-j distance* should NOT change C_exc through w_ij.
    //
    // To enforce that (and avoid the phantom "self" force), when computing the i-j
    // coordination force we subtract this pair's contribution from (lambda_i + lambda_j):
    //   lambda_other_sum = (lambda_i + lambda_j) - 2 * gate * (dU_ij/dC_exc)
    //
    // Then:
    //   dU/dr via coordination = lambda_other_sum * w'(r)
    //   F_coord_mag = - dU/dr
    // -------------------------------------------------------------------------
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const unsigned int type_i = __scalar_as_int(pos_i.w);
        if (type_i != m_reactive_typeid)
            {
            continue;
            }

        Scalar3 fi = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        Scalar pei = Scalar(0.0);
        Scalar virial_xx = Scalar(0.0);
        Scalar virial_xy = Scalar(0.0);
        Scalar virial_xz = Scalar(0.0);
        Scalar virial_yy = Scalar(0.0);
        Scalar virial_yz = Scalar(0.0);
        Scalar virial_zz = Scalar(0.0);

        const Scalar coordination_i = h_coord.data[i];

        const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];

        for (unsigned int k = 0; k < n_neigh; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            if (j >= N)
                {
                continue;
                }

            const Scalar4 pos_j = h_pos.data[j];
            const unsigned int type_j = __scalar_as_int(pos_j.w);
            if (type_j != m_reactive_typeid)
                {
                continue;
                }

            Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
            dx = box.minImage(dx);
            const Scalar rsq = dot(dx, dx);

            // Need neighbors out to max(r_cut, weakening_outer) because coordination forces act
            // where w'(r) != 0 even if pair LJ energy is zero.
            if (rsq >= neighbor_cut_sq)
                {
                continue;
                }

            const Scalar small_r = Scalar(1e-7);
            const Scalar small_r_sq = small_r * small_r;

            Scalar r = slow::sqrt(rsq);
            if (r < small_r)
                {
                r = small_r;
                }

            // Initialize pair energy/force as zero; only nonzero when rsq < r_cut^2.
            Scalar pair_energy = Scalar(0.0);
            Scalar force_divr = Scalar(0.0); // pair-only contribution

            // Leave-one-out crowding for this pair (used for both pair and coordination terms).
            const Scalar weight = computeWeight(r);
            const Scalar coordination_j = h_coord.data[j];
            const Scalar raw = (coordination_i - weight) + (coordination_j - weight);

            Scalar C_exc = Scalar(0.0);
            Scalar gate = Scalar(0.0);
            evaluateCexcSmooth(raw, C_exc, gate);

            // -----------------------------
            // Pair term: only for r < r_cut
            // -----------------------------
            if (rsq < m_r_cut_sq)
                {
                // No clamp needed: C_exc >= 0
                const Scalar weakening
                    = slow::pow(Scalar(1.0) + C_exc, -m_weakening_exponent);

                const Scalar inv_r = Scalar(1.0) / r;
                const Scalar rsq_safe = std::max(rsq, small_r_sq);
                const Scalar r2inv = Scalar(1.0) / rsq_safe;
                const Scalar sr = m_sigma * inv_r;
                const Scalar sr2 = sr * sr;
                const Scalar sr6 = sr2 * sr2 * sr2;
                const Scalar sr12 = sr6 * sr6;

                const Scalar base_energy
                    = Scalar(4.0) * m_epsilon * (sr12 - sr6) - m_energy_shift;
                const bool inner = r <= m_sigma * Scalar(1.1224620483093729814);
                const Scalar depth = m_epsilon + m_energy_shift;
                pair_energy = inner ? base_energy + (Scalar(1.0) - weakening) * depth
                                    : weakening * base_energy;
                const Scalar force_scale = inner ? Scalar(1.0) : weakening;
                const Scalar base_force_divr
                    = Scalar(24.0) * m_epsilon * (Scalar(2.0) * sr12 - sr6) * r2inv;
                force_divr = force_scale * base_force_divr;


                }

            // -----------------------------
            // Coordination-force term (many-body).
            //
            // We convert lambda_i = dU/dC_i into forces via grad(C_i) = sum_k w'(r_ik) rhat.
            //
            // IMPORTANT (leave-one-out self-cancellation):
            // For pair (i,j), the leave-one-out raw crowding is:
            //   raw = (C_i - w_ij) + (C_j - w_ij)
            // Since C_i and C_j include w_ij, raw has NO dependence on r_ij.
            //
            // Therefore, the "self" contribution of this pair must be removed from
            // (lambda_i + lambda_j) when turning lambda into a force along r_ij.
            //
            // With smooth C_exc, this pair contributes:
            //   lambda_i += gate * dU/dC_exc
            //   lambda_j += gate * dU/dC_exc
            // so we subtract 2 * gate * (dU/dC_exc)_pair.
            // -----------------------------
            Scalar force_divr_coord = Scalar(0.0);

            if (r > m_weakening_inner && r < m_weakening_outer)
                {
                // Compute dU_ij/dC_exc for THIS pair, fully analytically.
                const Scalar dU_dCexc_pair = evaluate_dU_dCexc(r, C_exc);

                // Remove this pair's "self" contribution from the lambda sum.
                const Scalar lambda_sum = h_lambda.data[i] + h_lambda.data[j];
                const Scalar lambda_other_sum
                    = lambda_sum - Scalar(2.0) * gate * dU_dCexc_pair;

                // Convert to a physical force along r_ij via w'(r).
                const Scalar dw_dr = evaluateWeightDerivative(r);

                // dU/dr via coordination = lambda_other_sum * dw/dr  =>  F = -dU/dr
                const Scalar force_mag_coord = -(lambda_other_sum * dw_dr);
                force_divr_coord = force_mag_coord / r;
                }

            // Total force for this neighbor interaction
            const Scalar total_force_divr = force_divr + force_divr_coord;

            const Scalar3 fij = dx * total_force_divr;
            fi += fij;

            const Scalar half_pair_energy = Scalar(0.5) * pair_energy;
            pei += half_pair_energy;

            if (compute_virial)
                {
                const Scalar force_div2r = Scalar(0.5) * total_force_divr;
                virial_xx += force_div2r * dx.x * dx.x;
                virial_xy += force_div2r * dx.x * dx.y;
                virial_xz += force_div2r * dx.x * dx.z;
                virial_yy += force_div2r * dx.y * dx.y;
                virial_yz += force_div2r * dx.y * dx.z;
                virial_zz += force_div2r * dx.z * dx.z;
                }

            if (third_law)
                {
                const unsigned int mem_j = j;
                h_force.data[mem_j].x -= fij.x;
                h_force.data[mem_j].y -= fij.y;
                h_force.data[mem_j].z -= fij.z;
                h_force.data[mem_j].w += half_pair_energy;

                if (compute_virial)
                    {
                    const Scalar force_div2r = Scalar(0.5) * total_force_divr;
                    h_virial.data[0 * virial_pitch + mem_j] += force_div2r * dx.x * dx.x;
                    h_virial.data[1 * virial_pitch + mem_j] += force_div2r * dx.x * dx.y;
                    h_virial.data[2 * virial_pitch + mem_j] += force_div2r * dx.x * dx.z;
                    h_virial.data[3 * virial_pitch + mem_j] += force_div2r * dx.y * dx.y;
                    h_virial.data[4 * virial_pitch + mem_j] += force_div2r * dx.y * dx.z;
                    h_virial.data[5 * virial_pitch + mem_j] += force_div2r * dx.z * dx.z;
                    }
                }
            }

        const unsigned int mem_i = i;
        h_force.data[mem_i].x += fi.x;
        h_force.data[mem_i].y += fi.y;
        h_force.data[mem_i].z += fi.z;
        h_force.data[mem_i].w += pei;

        if (compute_virial)
            {
            h_virial.data[0 * virial_pitch + mem_i] += virial_xx;
            h_virial.data[1 * virial_pitch + mem_i] += virial_xy;
            h_virial.data[2 * virial_pitch + mem_i] += virial_xz;
            h_virial.data[3 * virial_pitch + mem_i] += virial_yy;
            h_virial.data[4 * virial_pitch + mem_i] += virial_yz;
            h_virial.data[5 * virial_pitch + mem_i] += virial_zz;
            }
        }
    }

void ReactiveAHForceCompute::setReactiveType(const std::string& reactive_type)
    {
    if (reactive_type.empty())
        {
        throw std::invalid_argument("reactive_type must be a valid particle type name");
        }
    const unsigned int type_id = m_pdata->getTypeByName(reactive_type);
    if (type_id >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("reactive particle type is not defined in the snapshot");
        }
    m_reactive_typeid = type_id;
    m_reactive_type_name = reactive_type;
    }

void ReactiveAHForceCompute::updateNeighborListCutoffs()
    {
    if (!m_matrix_registered)
        {
        return;
        }

    ArrayHandle<Scalar> h_r_cut(*m_r_cut_matrix, access_location::host, access_mode::overwrite);
    std::fill(h_r_cut.data, h_r_cut.data + m_typpair_idx.getNumElements(), Scalar(0.0));

    if (m_reactive_typeid < m_pdata->getNTypes())
        {
        const unsigned int idx = m_typpair_idx(m_reactive_typeid, m_reactive_typeid);
        h_r_cut.data[idx] = m_neighbor_rcut;
        }

    m_nlist->notifyRCutMatrixChange();
    }

void ReactiveAHForceCompute::ensureScratchSize()
    {
    const size_t max_n = m_pdata->getMaxN();
    if (m_coordination.getNumElements() < max_n)
        {
        m_coordination.resize(max_n);
        }
    if (m_lambda.getNumElements() < max_n)
        {
        m_lambda.resize(max_n);
        }
    }

Scalar ReactiveAHForceCompute::computeWeight(Scalar distance) const
    {
    if (distance >= m_weakening_outer)
        {
        return Scalar(0.0);
        }
    if (distance <= m_weakening_inner)
        {
        return Scalar(1.0);
        }

    const Scalar pi = Scalar(3.141592653589793238462643383279502884L);
    const Scalar fraction = (distance - m_weakening_inner) * m_inverse_span;
    const Scalar angle = pi * fraction;
    return Scalar(0.5) * (Scalar(1.0) + slow::cos(angle));
    }

Scalar ReactiveAHForceCompute::evaluateEnergy(Scalar distance) const
    {
    const Scalar r = std::max(distance, m_smooth_r_min);
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar sr = m_sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;
    return Scalar(4.0) * m_epsilon * (sr12 - sr6) - m_energy_shift;
    }

Scalar ReactiveAHForceCompute::evaluateForceMagnitude(Scalar distance) const
    {
    const Scalar r = std::max(distance, m_smooth_r_min);
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar sr = m_sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;
    return Scalar(24.0) * m_epsilon * inv_r * (Scalar(2.0) * sr12 - sr6);
    }

namespace detail
    {
void export_ReactiveAHForceCompute(pybind11::module& m)
    {
    pybind11::class_<ReactiveAHForceCompute,
                     ForceCompute,
                     std::shared_ptr<ReactiveAHForceCompute>>(m, "ReactiveAHForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            const std::string&,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar,
                            bool,
                            Scalar,
                            Scalar>(),
             pybind11::arg("sysdef"),
             pybind11::arg("nlist"),
             pybind11::arg("reactive_type"),
             pybind11::arg("sigma"),
             pybind11::arg("epsilon"),
             pybind11::arg("r_cut"),
             pybind11::arg("weakening_inner"),
             pybind11::arg("weakening_outer"),
             pybind11::arg("weakening_exponent"),
             pybind11::arg("smooth_elbow") = false,
             pybind11::arg("smooth_kappa") = Scalar(0.05),
             pybind11::arg("smooth_beta") = Scalar(1.0))
        .def("set_params",
             &ReactiveAHForceCompute::setParameters,
             pybind11::arg("sigma"),
             pybind11::arg("epsilon"),
             pybind11::arg("r_cut"),
             pybind11::arg("weakening_inner"),
             pybind11::arg("weakening_outer"),
             pybind11::arg("weakening_exponent"),
             pybind11::arg("smooth_elbow") = false,
             pybind11::arg("smooth_kappa") = Scalar(0.05),
             pybind11::arg("smooth_beta") = Scalar(1.0))
        .def("get_params", &ReactiveAHForceCompute::getParameters)
        .def_property_readonly("neighbor_list", &ReactiveAHForceCompute::getNeighborList)
        .def_property_readonly("reactive_type", &ReactiveAHForceCompute::getReactiveType);
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd
