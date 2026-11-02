// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BoxDim.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#ifndef __REACTIVE_AH_FORCE_COMPUTE_H__
#define __REACTIVE_AH_FORCE_COMPUTE_H__

namespace hoomd
    {
namespace md
    {

//! Reactive Lennard-Jones force compute with coordination-based weakening.
class PYBIND11_EXPORT ReactiveAHForceCompute : public ForceCompute
    {
    public:
    ReactiveAHForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<NeighborList> nlist,
                           const std::string& reactive_type,
                           Scalar sigma,
                           Scalar epsilon,
                           Scalar r_cut,
                           Scalar weakening_inner,
                           Scalar weakening_outer,
                           Scalar weakening_exponent,
                           bool smooth_elbow = false,
                           Scalar smooth_kappa = Scalar(0.05),
                           Scalar smooth_beta = Scalar(1.0));

    ~ReactiveAHForceCompute() override;

    void setParameters(Scalar sigma,
                       Scalar epsilon,
                       Scalar r_cut,
                       Scalar weakening_inner,
                       Scalar weakening_outer,
                       Scalar weakening_exponent,
                       bool smooth_elbow = false,
                       Scalar smooth_kappa = Scalar(0.05),
                       Scalar smooth_beta = Scalar(1.0));

    pybind11::dict getParameters() const;

    std::shared_ptr<NeighborList> getNeighborList() const;

    const std::string& getReactiveType() const;

    protected:
    void computeForces(uint64_t timestep) override;
    void ensureScratchSize();

    std::shared_ptr<NeighborList> m_nlist;
    Index2D m_typpair_idx;
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_matrix;
    GPUArray<Scalar> m_coordination;
    GPUArray<Scalar> m_lambda;
    bool m_matrix_registered;

    std::string m_reactive_type_name;
    unsigned int m_reactive_typeid;

    Scalar m_sigma;
    Scalar m_epsilon;
    Scalar m_r_cut;
    Scalar m_r_cut_sq;
    Scalar m_neighbor_rcut;
    Scalar m_energy_shift;
    Scalar m_weakening_inner;
    Scalar m_weakening_outer;
    Scalar m_weakening_exponent;
    Scalar m_inverse_span;
    Scalar m_r_elbow;
    bool m_smooth_elbow;
    Scalar m_smooth_kappa;
    Scalar m_smooth_beta;
    Scalar m_smooth_delta_tol;
    Scalar m_smooth_r_min;

    private:
    void setReactiveType(const std::string& reactive_type);
    void updateNeighborListCutoffs();
    Scalar computeWeight(Scalar distance) const;
    Scalar evaluateEnergy(Scalar distance) const;
    Scalar evaluateForceMagnitude(Scalar distance) const;
    inline void evaluateCexcSmooth(Scalar raw, Scalar& C_exc, Scalar& gate) const;
    inline Scalar evaluateForceDerivative(Scalar distance) const;
    Scalar evaluate_dU_dCexc(Scalar r_in, Scalar C_exc) const;
    Scalar evaluateWeightDerivative(Scalar distance) const;
    };

namespace detail
    {
void export_ReactiveAHForceCompute(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif
