// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "ReactiveAHForceComputeGPU.cuh"

#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

#define DEVICE __device__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
namespace
    {
hipError_t getCachedMaxThreadsPerBlock(const void* kernel_ptr,
                                       std::unordered_map<int, unsigned int>& cache,
                                       std::mutex& cache_mutex,
                                       unsigned int& max_threads_per_block)
    {
    int device = -1;
    hipError_t err = hipGetDevice(&device);
    if (err != hipSuccess)
        {
        return err;
        }

    {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(device);
    if (it != cache.end())
        {
        max_threads_per_block = it->second;
        return hipSuccess;
        }
    }

    hipFuncAttributes attr;
    err = hipFuncGetAttributes(&attr, kernel_ptr);
    if (err != hipSuccess)
        {
        return err;
        }

    max_threads_per_block = static_cast<unsigned int>(attr.maxThreadsPerBlock);

    {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.emplace(device, max_threads_per_block);
    }

    return hipSuccess;
    }
    } // end namespace

#if HOOMD_LONGREAL_SIZE == 64
#if (__CUDA_ARCH__ < 600)
DEVICE inline double atomicAddScalar(double* address, double val)
    {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
    }
#else
DEVICE inline double atomicAddScalar(double* address, double val)
    {
    return atomicAdd(address, val);
    }
#endif
#else
#ifdef __HIP_PLATFORM_HCC__
DEVICE inline float atomicAddScalar(float* address, float val)
    {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val + __uint_as_float(assumed)));
        } while (assumed != old);
    return __uint_as_float(old);
    }
#else
DEVICE inline float atomicAddScalar(float* address, float val)
    {
    return atomicAdd(address, val);
    }
#endif
#endif

// Smoothly project raw excess coordination onto max(raw, 0). The gate term is
// dC_exc/draw and is reused in chain-rule terms later on.
DEVICE inline void evaluateCexcSmooth(Scalar raw, Scalar& C_exc, Scalar& gate)
    {
    const Scalar eps = Scalar(1e-6);
    const Scalar s = slow::sqrt(raw * raw + eps * eps);
    C_exc = Scalar(0.5) * (raw + s);
    gate = Scalar(0.5) * (Scalar(1.0) + raw / s);
    }

// Cosine window used for coordination weights between inner and outer radii.
DEVICE inline Scalar computeWeight(Scalar distance, const reactive_ah_params& params)
    {
    if (distance >= params.weakening_outer)
        {
        return Scalar(0.0);
        }
    if (distance <= params.weakening_inner)
        {
        return Scalar(1.0);
        }

    const Scalar pi = Scalar(3.141592653589793238462643383279502884L);
    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    const Scalar angle = pi * fraction;
    return Scalar(0.5) * (Scalar(1.0) + slow::cos(angle));
    }

// Derivative of the cosine coordination window with respect to distance.
DEVICE inline Scalar evaluateWeightDerivative(Scalar distance, const reactive_ah_params& params)
    {
    if (distance >= params.weakening_outer || distance <= params.weakening_inner)
        {
        return Scalar(0.0);
        }

    const Scalar pi = Scalar(3.141592653589793238462643383279502884L);
    const Scalar fraction = (distance - params.weakening_inner) * params.inverse_span;
    const Scalar angle = pi * fraction;
    return -Scalar(0.5) * slow::sin(angle) * (pi * params.inverse_span);
    }

// Shifted Lennard-Jones energy used by the ReactiveAH pair branch.
DEVICE inline Scalar evaluateEnergy(Scalar distance, const reactive_ah_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar sr = params.sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;
    return Scalar(4.0) * params.epsilon * (sr12 - sr6) - params.energy_shift;
    }

// Positive magnitude of the pair force, F(r), before multiplying by direction.
DEVICE inline Scalar evaluateForceMagnitude(Scalar distance, const reactive_ah_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar sr = params.sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;
    return Scalar(24.0) * params.epsilon * inv_r * (Scalar(2.0) * sr12 - sr6);
    }

// Radial derivative dF/dr used in the smoothing-region chain rule.
DEVICE inline Scalar evaluateForceDerivative(Scalar distance, const reactive_ah_params& params)
    {
    const Scalar r = (distance > params.smooth_r_min) ? distance : params.smooth_r_min;
    const Scalar inv_r = Scalar(1.0) / r;
    const Scalar inv_r2 = inv_r * inv_r;

    const Scalar sr = params.sigma * inv_r;
    const Scalar sr2 = sr * sr;
    const Scalar sr6 = sr2 * sr2 * sr2;
    const Scalar sr12 = sr6 * sr6;

    return Scalar(24.0) * params.epsilon * inv_r2
           * (Scalar(-26.0) * sr12 + Scalar(7.0) * sr6);
    }

// Compute dU/dC_exc for one directed pair. This drives the many-body lambda
// accumulation and, through lambda, the coordination force in pass 3.
DEVICE inline Scalar evaluate_dU_dCexc(Scalar r_in,
                                      Scalar C_exc,
                                      const reactive_ah_params& params,
                                      unsigned int* error_flag)
    {
    if (r_in >= params.r_cut) return Scalar(0.0);
    const Scalar W = slow::pow(Scalar(1.0) + C_exc, -params.weakening_exponent);
    const Scalar dW_dC = -params.weakening_exponent * W / (Scalar(1.0) + C_exc);
    const Scalar r_minimum = params.sigma * Scalar(1.1224620483093729814);
    const Scalar depth = params.epsilon + params.energy_shift;
    return ((r_in <= r_minimum) ? -depth : evaluateEnergy(r_in, params)) * dW_dC;
    }

__global__ void gpu_reactiveah_compute_coordination_kernel(const unsigned int N,
                                                           const Scalar4* d_pos,
                                                           const unsigned int* d_n_neigh,
                                                           const unsigned int* d_nlist,
                                                           const size_t* d_head_list,
                                                           Scalar* d_coord,
                                                           unsigned int reactive_typeid,
                                                           BoxDim box,
                                                           reactive_ah_params params)
    {
    // Pass 1: each thread computes one particle's scalar coordination.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        d_coord[idx] = Scalar(0.0);
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];

    Scalar coord = Scalar(0.0);

    for (unsigned int k = 0; k < n_neigh; ++k)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        const unsigned int type_j = __scalar_as_int(pos_j.w);
        if (type_j != reactive_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.neighbor_rcut_sq)
            {
            continue;
            }

        Scalar r = Scalar(0.0);
        if (rsq > Scalar(0.0))
            {
            const Scalar inv_r = fast::rsqrt(rsq);
            r = rsq * inv_r;
            }
        const Scalar weight = computeWeight(r, params);
        if (weight == Scalar(0.0))
            {
            continue;
            }

        coord += weight;
        }

    d_coord[idx] = coord;
    }

__global__ void gpu_reactiveah_compute_lambda_kernel(const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const unsigned int* d_n_neigh,
                                                     const unsigned int* d_nlist,
                                                     const size_t* d_head_list,
                                                     const Scalar* d_coord,
                                                     Scalar* d_lambda,
                                                     unsigned int* d_error_flag,
                                                     unsigned int reactive_typeid,
                                                     BoxDim box,
                                                     reactive_ah_params params)
    {
    // Pass 2: process each pair once (j > i). Accumulate i's contribution in a
    // thread-local register and atomically add once at the end; atomically
    // update j per pair. This preserves CPU-equivalent pair-once semantics.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        return;
        }

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];
    const Scalar Ci = __ldg(d_coord + idx);
    Scalar lambda_local = Scalar(0.0);

    for (unsigned int k = 0; k < n_neigh; ++k)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N || j <= idx)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        const unsigned int type_j = __scalar_as_int(pos_j.w);
        if (type_j != reactive_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.r_cut_sq)
            {
            continue;
            }

        Scalar r = Scalar(0.0);
        if (rsq > Scalar(0.0))
            {
            const Scalar inv_r = fast::rsqrt(rsq);
            r = rsq * inv_r;
            }
        Scalar w_ij = Scalar(0.0);
        if (r < params.weakening_outer)
            {
            w_ij = computeWeight(r, params);
            }

        const Scalar Cj = __ldg(d_coord + j);
        const Scalar raw = (Ci - w_ij) + (Cj - w_ij);

        Scalar C_exc = Scalar(0.0);
        Scalar gate = Scalar(0.0);
        evaluateCexcSmooth(raw, C_exc, gate);

        const Scalar dU_dCexc = evaluate_dU_dCexc(r, C_exc, params, d_error_flag);
        const Scalar contrib = gate * dU_dCexc;

        lambda_local += contrib;
        atomicAddScalar(d_lambda + j, contrib);
        }

    if (lambda_local != Scalar(0.0))
        {
        atomicAddScalar(d_lambda + idx, lambda_local);
        }
    }

template<bool compute_virial>
__global__ void gpu_reactiveah_compute_forces_kernel(const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const unsigned int* d_n_neigh,
                                                     const unsigned int* d_nlist,
                                                     const size_t* d_head_list,
                                                     const Scalar* d_coord,
                                                     const Scalar* d_lambda,
                                                     Scalar4* d_force,
                                                     Scalar* d_virial,
                                                     const size_t virial_pitch,
                                                     unsigned int* d_error_flag,
                                                     unsigned int reactive_typeid,
                                                     BoxDim box,
                                                     reactive_ah_params params)
    {
    // Pass 3: evaluate pair force, coordination force, and energy/virial.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        {
        return;
        }

    const Scalar4 pos_i = __ldg(d_pos + idx);
    const unsigned int type_i = __scalar_as_int(pos_i.w);
    if (type_i != reactive_typeid)
        {
        return;
        }

    const Scalar coordination_i = __ldg(d_coord + idx);
    const Scalar lambda_i = __ldg(d_lambda + idx);

    Scalar3 fi = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar pei = Scalar(0.0);
    Scalar virial_xx = Scalar(0.0);
    Scalar virial_xy = Scalar(0.0);
    Scalar virial_xz = Scalar(0.0);
    Scalar virial_yy = Scalar(0.0);
    Scalar virial_yz = Scalar(0.0);
    Scalar virial_zz = Scalar(0.0);

    const Scalar3 pi = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
    const size_t head = d_head_list[idx];
    const unsigned int n_neigh = d_n_neigh[idx];

    for (unsigned int k = 0; k < n_neigh; ++k)
        {
        const unsigned int j = d_nlist[head + k];
        if (j >= N)
            {
            continue;
            }

        const Scalar4 pos_j = __ldg(d_pos + j);
        const unsigned int type_j = __scalar_as_int(pos_j.w);
        if (type_j != reactive_typeid)
            {
            continue;
            }

        Scalar3 dx = pi - make_scalar3(pos_j.x, pos_j.y, pos_j.z);
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);
        if (rsq >= params.neighbor_rcut_sq)
            {
            continue;
            }

        const Scalar small_r = Scalar(1e-7);
        Scalar r = Scalar(0.0);
        if (rsq > Scalar(0.0))
            {
            const Scalar inv_r = fast::rsqrt(rsq);
            r = rsq * inv_r;
            }
        if (r < small_r)
            {
            r = small_r;
            }

        // Early-out gate: if neither pair nor coordination terms can contribute,
        // skip all expensive weakening / smoothing math.
        const bool in_pair = rsq < params.r_cut_sq;
        const bool in_coord_window = r > params.weakening_inner && r < params.weakening_outer;
        if (!in_pair && !in_coord_window)
            {
            continue;
            }

        Scalar pair_energy = Scalar(0.0);
        Scalar force_divr = Scalar(0.0);

        const Scalar weight = (r < params.weakening_outer) ? computeWeight(r, params) : Scalar(0.0);
        const Scalar coordination_j = __ldg(d_coord + j);
        const Scalar raw = (coordination_i - weight) + (coordination_j - weight);

        Scalar C_exc = Scalar(0.0);
        Scalar gate = Scalar(0.0);
        evaluateCexcSmooth(raw, C_exc, gate);

        if (in_pair)
            {
            const Scalar weakening
                = slow::pow(Scalar(1.0) + C_exc, -params.weakening_exponent);

            const Scalar base_energy = evaluateEnergy(r, params);
            const bool inner = r <= params.sigma * Scalar(1.1224620483093729814);
            const Scalar depth = params.epsilon + params.energy_shift;
            pair_energy = inner ? base_energy + (Scalar(1.0) - weakening) * depth
                                : weakening * base_energy;
            const Scalar force_scale = inner ? Scalar(1.0) : weakening;
            const Scalar base_force_divr = evaluateForceMagnitude(r, params) / r;
            force_divr = force_scale * base_force_divr;


            }

        Scalar force_divr_coord = Scalar(0.0);
        if (in_coord_window)
            {
            const Scalar dU_dCexc_pair = evaluate_dU_dCexc(r, C_exc, params, d_error_flag);
            const Scalar pair_self_contrib = gate * dU_dCexc_pair;

            const Scalar lambda_sum = lambda_i + __ldg(d_lambda + j);
            const Scalar lambda_other_sum
                = lambda_sum - Scalar(2.0) * pair_self_contrib;
            const Scalar dw_dr = evaluateWeightDerivative(r, params);
            const Scalar force_mag_coord = -(lambda_other_sum * dw_dr);
            force_divr_coord = force_mag_coord / r;
            }

        const Scalar total_force_divr = force_divr + force_divr_coord;

        fi.x += dx.x * total_force_divr;
        fi.y += dx.y * total_force_divr;
        fi.z += dx.z * total_force_divr;

        pei += Scalar(0.5) * pair_energy;

        if constexpr (compute_virial)
            {
            const Scalar force_div2r = Scalar(0.5) * total_force_divr;
            virial_xx += force_div2r * dx.x * dx.x;
            virial_xy += force_div2r * dx.x * dx.y;
            virial_xz += force_div2r * dx.x * dx.z;
            virial_yy += force_div2r * dx.y * dx.y;
            virial_yz += force_div2r * dx.y * dx.z;
            virial_zz += force_div2r * dx.z * dx.z;
            }
        }

    d_force[idx].x += fi.x;
    d_force[idx].y += fi.y;
    d_force[idx].z += fi.z;
    d_force[idx].w += pei;

    if constexpr (compute_virial)
        {
        d_virial[0 * virial_pitch + idx] += virial_xx;
        d_virial[1 * virial_pitch + idx] += virial_xy;
        d_virial[2 * virial_pitch + idx] += virial_xz;
        d_virial[3 * virial_pitch + idx] += virial_yy;
        d_virial[4 * virial_pitch + idx] += virial_yz;
        d_virial[5 * virial_pitch + idx] += virial_zz;
        }
    }

hipError_t gpu_reactiveah_get_coordination_max_block_size(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_reactiveah_compute_coordination_kernel,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

hipError_t gpu_reactiveah_get_lambda_max_block_size(unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache;
    return getCachedMaxThreadsPerBlock((const void*)gpu_reactiveah_compute_lambda_kernel,
                                       max_threads_cache,
                                       cache_mutex,
                                       max_block_size);
    }

namespace
    {
hipError_t gpu_reactiveah_get_force_variant_max_block_size(bool compute_virial,
                                                           unsigned int& max_block_size)
    {
    static std::mutex cache_mutex;
    static std::unordered_map<int, unsigned int> max_threads_cache_with_virial;
    static std::unordered_map<int, unsigned int> max_threads_cache_without_virial;
    if (compute_virial)
        {
        return getCachedMaxThreadsPerBlock((const void*)gpu_reactiveah_compute_forces_kernel<true>,
                                           max_threads_cache_with_virial,
                                           cache_mutex,
                                           max_block_size);
        }

    return getCachedMaxThreadsPerBlock((const void*)gpu_reactiveah_compute_forces_kernel<false>,
                                       max_threads_cache_without_virial,
                                       cache_mutex,
                                       max_block_size);
    }
    } // end namespace

hipError_t gpu_reactiveah_get_force_max_block_size(unsigned int& max_block_size)
    {
    unsigned int max_block_size_with_virial = 0;
    hipError_t err
        = gpu_reactiveah_get_force_variant_max_block_size(true, max_block_size_with_virial);
    if (err != hipSuccess)
        {
        return err;
        }

    unsigned int max_block_size_without_virial = 0;
    err = gpu_reactiveah_get_force_variant_max_block_size(false, max_block_size_without_virial);
    if (err != hipSuccess)
        {
        return err;
        }

    // Use the stricter limit so one autotuner range is valid for both variants.
    max_block_size = std::min(max_block_size_with_virial, max_block_size_without_virial);
    return hipSuccess;
    }

hipError_t gpu_reactiveah_compute_coordination(const unsigned int N,
                                               const Scalar4* d_pos,
                                               const unsigned int* d_n_neigh,
                                               const unsigned int* d_nlist,
                                               const size_t* d_head_list,
                                               Scalar* d_coord,
                                               unsigned int reactive_typeid,
                                               BoxDim box,
                                               reactive_ah_params params,
                                               unsigned int block_size)
    {
    unsigned int max_threads_per_block = 0;
    hipError_t cache_err = gpu_reactiveah_get_coordination_max_block_size(max_threads_per_block);
    if (cache_err != hipSuccess)
        {
        return cache_err;
        }
    const unsigned int run_block_size
        = std::min(block_size, max_threads_per_block);
    if (run_block_size == 0)
        {
        return hipErrorInvalidConfiguration;
        }

    dim3 grid((N + run_block_size - 1) / run_block_size, 1, 1);
    dim3 threads(run_block_size, 1, 1);
    hipLaunchKernelGGL((gpu_reactiveah_compute_coordination_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       N,
                       d_pos,
                       d_n_neigh,
                       d_nlist,
                       d_head_list,
                       d_coord,
                       reactive_typeid,
                       box,
                       params);
    return hipGetLastError();
    }

hipError_t gpu_reactiveah_compute_lambda(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         Scalar* d_lambda,
                                         unsigned int* d_error_flag,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_ah_params params,
                                         unsigned int block_size)
    {
    unsigned int max_threads_per_block = 0;
    hipError_t cache_err = gpu_reactiveah_get_lambda_max_block_size(max_threads_per_block);
    if (cache_err != hipSuccess)
        {
        return cache_err;
        }
    const unsigned int run_block_size
        = std::min(block_size, max_threads_per_block);
    if (run_block_size == 0)
        {
        return hipErrorInvalidConfiguration;
        }

    dim3 grid((N + run_block_size - 1) / run_block_size, 1, 1);
    dim3 threads(run_block_size, 1, 1);
    hipLaunchKernelGGL((gpu_reactiveah_compute_lambda_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       N,
                       d_pos,
                       d_n_neigh,
                       d_nlist,
                       d_head_list,
                       d_coord,
                       d_lambda,
                       d_error_flag,
                       reactive_typeid,
                       box,
                       params);
    return hipGetLastError();
    }

hipError_t gpu_reactiveah_compute_forces(const unsigned int N,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_n_neigh,
                                         const unsigned int* d_nlist,
                                         const size_t* d_head_list,
                                         const Scalar* d_coord,
                                         const Scalar* d_lambda,
                                         Scalar4* d_force,
                                         Scalar* d_virial,
                                         const size_t virial_pitch,
                                         unsigned int* d_error_flag,
                                         unsigned int reactive_typeid,
                                         BoxDim box,
                                         reactive_ah_params params,
                                         unsigned int compute_virial,
                                         unsigned int block_size)
    {
    const bool do_compute_virial = (compute_virial != 0);

    unsigned int max_threads_per_block = 0;
    hipError_t cache_err
        = gpu_reactiveah_get_force_variant_max_block_size(do_compute_virial, max_threads_per_block);
    if (cache_err != hipSuccess)
        {
        return cache_err;
        }
    const unsigned int run_block_size
        = std::min(block_size, max_threads_per_block);
    if (run_block_size == 0)
        {
        return hipErrorInvalidConfiguration;
        }

    dim3 grid((N + run_block_size - 1) / run_block_size, 1, 1);
    dim3 threads(run_block_size, 1, 1);
    if (do_compute_virial)
        {
        hipLaunchKernelGGL((gpu_reactiveah_compute_forces_kernel<true>),
                           grid,
                           threads,
                           0,
                           0,
                           N,
                           d_pos,
                           d_n_neigh,
                           d_nlist,
                           d_head_list,
                           d_coord,
                           d_lambda,
                           d_force,
                           d_virial,
                           virial_pitch,
                           d_error_flag,
                           reactive_typeid,
                           box,
                           params);
        }
    else
        {
        hipLaunchKernelGGL((gpu_reactiveah_compute_forces_kernel<false>),
                           grid,
                           threads,
                           0,
                           0,
                           N,
                           d_pos,
                           d_n_neigh,
                           d_nlist,
                           d_head_list,
                           d_coord,
                           d_lambda,
                           d_force,
                           d_virial,
                           virial_pitch,
                           d_error_flag,
                           reactive_typeid,
                           box,
                           params);
        }
    return hipGetLastError();
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
