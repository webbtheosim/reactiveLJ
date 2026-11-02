// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "ReactiveAHForceComputeGPU.h"

#include "hoomd/Autotuner.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace hoomd
    {
namespace md
    {

ReactiveAHForceComputeGPU::ReactiveAHForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
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
    : ReactiveAHForceCompute(sysdef,
                             nlist,
                             reactive_type,
                             sigma,
                             epsilon,
                             r_cut,
                             weakening_inner,
                             weakening_outer,
                             weakening_exponent,
                             smooth_elbow,
                             smooth_kappa,
                             smooth_beta)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a ReactiveAHForceComputeGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing ReactiveAHForceComputeGPU");
        }

    auto make_block_size_range
        = [this](hipError_t (*query_max_block_size)(unsigned int&),
                 const char* kernel_name) -> std::vector<unsigned int>
    {
        auto range = AutotunerBase::makeBlockSizeRange(m_exec_conf);
        unsigned int kernel_max_block_size = 0;
        const hipError_t err = query_max_block_size(kernel_max_block_size);
        if (err != hipSuccess || kernel_max_block_size == 0)
            {
            m_exec_conf->msg->warning()
                << "ReactiveAHForceComputeGPU: failed to query max block size for " << kernel_name
                << " kernel; using default autotuner range. Error: " << hipGetErrorString(err)
                << std::endl;
            return range;
            }

        range.erase(std::remove_if(range.begin(),
                                   range.end(),
                                   [kernel_max_block_size](unsigned int block_size)
                                   { return block_size > kernel_max_block_size; }),
                    range.end());

        if (range.empty())
            {
            range.push_back(kernel_max_block_size);
            }

        return range;
    };

    const auto coord_block_size_range
        = make_block_size_range(kernel::gpu_reactiveah_get_coordination_max_block_size,
                                "coordination");
    const auto lambda_block_size_range
        = make_block_size_range(kernel::gpu_reactiveah_get_lambda_max_block_size, "lambda");
    const auto force_block_size_range
        = make_block_size_range(kernel::gpu_reactiveah_get_force_max_block_size, "force");

    m_tuner_coord.reset(new Autotuner<1>({coord_block_size_range},
                                         m_exec_conf,
                                         "reactiveah_coord"));
    m_tuner_lambda.reset(new Autotuner<1>({lambda_block_size_range},
                                          m_exec_conf,
                                          "reactiveah_lambda"));
    m_tuner_force.reset(new Autotuner<1>({force_block_size_range},
                                         m_exec_conf,
                                         "reactiveah_force"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_coord, m_tuner_lambda, m_tuner_force});

    GPUArray<unsigned int> error_flag(1, m_exec_conf);
    m_error_flag.swap(error_flag);

#ifdef ENABLE_MPI
    m_tuner_coord->setSync(bool(m_pdata->getDomainDecomposition()));
    m_tuner_lambda->setSync(bool(m_pdata->getDomainDecomposition()));
    m_tuner_force->setSync(bool(m_pdata->getDomainDecomposition()));
#endif
    }

void ReactiveAHForceComputeGPU::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);
    ensureScratchSize();

    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << "ReactiveAHForceComputeGPU cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in ReactiveAHForceComputeGPU");
        }

    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                              access_location::device,
                              access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_coord(m_coordination, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_lambda(m_lambda, access_location::device, access_mode::readwrite);
    {
    ArrayHandle<unsigned int> d_error(m_error_flag,
                                      access_location::device,
                                      access_mode::readwrite);

    const unsigned int N = m_pdata->getN();
    const size_t virial_pitch = m_virial.getPitch();
    const BoxDim box = m_pdata->getBox();

    PDataFlags flags = m_pdata->getFlags();
    const bool compute_virial = flags[pdata_flag::pressure_tensor];

    m_force.zeroFill();
    m_virial.zeroFill();
    m_coordination.zeroFill();
    m_lambda.zeroFill();
    m_error_flag.zeroFill();

    kernel::reactive_ah_params params;
    params.sigma = m_sigma;
    params.epsilon = m_epsilon;
    params.r_cut = m_r_cut;
    params.r_cut_sq = m_r_cut_sq;
    params.neighbor_rcut_sq = m_neighbor_rcut * m_neighbor_rcut;
    params.energy_shift = m_energy_shift;
    params.weakening_inner = m_weakening_inner;
    params.weakening_outer = m_weakening_outer;
    params.weakening_exponent = m_weakening_exponent;
    params.inverse_span = m_inverse_span;
    params.r_elbow = m_r_elbow;
    params.smooth_elbow = m_smooth_elbow;
    params.smooth_kappa = m_smooth_kappa;
    params.smooth_beta = m_smooth_beta;
    params.smooth_delta_tol = m_smooth_delta_tol;
    params.smooth_r_min = m_smooth_r_min;

    m_exec_conf->setDevice();

    // Pass 1: compute per-particle coordination numbers.
    m_tuner_coord->begin();
    hipError_t err = kernel::gpu_reactiveah_compute_coordination(N,
                                                                 d_pos.data,
                                                                 d_n_neigh.data,
                                                                 d_nlist.data,
                                                                 d_head_list.data,
                                                                 d_coord.data,
                                                                 m_reactive_typeid,
                                                                 box,
                                                                 params,
                                                                 m_tuner_coord->getParam()[0]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("ReactiveAHForceComputeGPU coordination kernel "
                                             "launch failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_coord->end();

    // Pass 2: accumulate lambda on each endpoint (pair-once, CPU-equivalent semantics).
    m_tuner_lambda->begin();
    err = kernel::gpu_reactiveah_compute_lambda(N,
                                                d_pos.data,
                                                d_n_neigh.data,
                                                d_nlist.data,
                                                d_head_list.data,
                                                d_coord.data,
                                                d_lambda.data,
                                                d_error.data,
                                                m_reactive_typeid,
                                                box,
                                                params,
                                                m_tuner_lambda->getParam()[0]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("ReactiveAHForceComputeGPU lambda kernel launch "
                                             "failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_lambda->end();

    // Pass 3: evaluate forces and optionally the virial tensor.
    m_tuner_force->begin();
    err = kernel::gpu_reactiveah_compute_forces(N,
                                                d_pos.data,
                                                d_n_neigh.data,
                                                d_nlist.data,
                                                d_head_list.data,
                                                d_coord.data,
                                                d_lambda.data,
                                                d_force.data,
                                                d_virial.data,
                                                virial_pitch,
                                                d_error.data,
                                                m_reactive_typeid,
                                                box,
                                                params,
                                                compute_virial,
                                                m_tuner_force->getParam()[0]);
    if (err != hipSuccess)
        {
        throw std::runtime_error(std::string("ReactiveAHForceComputeGPU force kernel launch "
                                             "failed: ")
                                 + hipGetErrorString(err));
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }
    m_tuner_force->end();
    }

    ArrayHandle<unsigned int> h_error(m_error_flag,
                                      access_location::host,
                                      access_mode::read);
    if (h_error.data[0] != 0)
        {
        if (h_error.data[0] == 1u)
            {
            throw std::runtime_error(
                "ReactiveAHForceCompute: r_elbow - delta must be greater than r_min; decrease "
                "delta or decrease r_min.");
            }
        if (h_error.data[0] == 2u)
            {
            throw std::runtime_error(
                "ReactiveAHForceCompute: r_elbow + delta must be less than r_cut; decrease delta "
                "or increase r_cut.");
            }
        throw std::runtime_error(
            "ReactiveAHForceCompute: r_elbow - delta must be greater than r_min; decrease delta or "
            "decrease r_min.");
        }
    }

namespace detail
    {
void export_ReactiveAHForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<ReactiveAHForceComputeGPU,
                     ReactiveAHForceCompute,
                     std::shared_ptr<ReactiveAHForceComputeGPU>>(m,
                                                                 "ReactiveAHForceComputeGPU")
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
             pybind11::arg("smooth_beta") = Scalar(1.0));
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
