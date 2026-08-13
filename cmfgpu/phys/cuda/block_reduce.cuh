// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Block reductions for the global scalar accumulators.  Same-address atomics
// serialize in L2, so folding inside the block turns O(num_catchments) atomics
// per counter into O(num_blocks) -- and the tree sum is far more accurate.
//
// Every thread must reach these reductions with an identity value.

#ifndef CMFGPU_BLOCK_REDUCE_CUH
#define CMFGPU_BLOCK_REDUCE_CUH

template <typename REAL>
__device__ __forceinline__ REAL cmf_shfl_down_sum(
    REAL value, unsigned active_mask, int lane, int offset)
{
    REAL other = __shfl_down_sync(active_mask, value, offset);
    const int source_lane = lane + offset;
    if (source_lane >= 32
        || ((active_mask & (1u << source_lane)) == 0u))
        return (REAL)0;
    return other;
}

__device__ __forceinline__ int cmf_shfl_down_max(
    int value, unsigned active_mask, int lane, int offset)
{
    int other = __shfl_down_sync(active_mask, value, offset);
    const int source_lane = lane + offset;
    if (source_lane >= 32
        || ((active_mask & (1u << source_lane)) == 0u))
        return 0;
    return other;
}

// Sum N per-thread values across the block; one thread issues one atomic each.
template <typename REAL, int N>
__device__ __forceinline__ void cmf_block_atomic_add(
    REAL (&value)[N], REAL* const (&destination)[N], int slot)
{
    __shared__ REAL partial[N][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const unsigned active_mask = __activemask();

#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value[i] += cmf_shfl_down_sum(
                value[i], active_mask, lane, offset);
        }
        if (lane == 0) partial[i][warp] = value[i];
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        REAL total = (lane < num_warps) ? partial[i][lane] : (REAL)0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += cmf_shfl_down_sum(total, active_mask, lane, offset);
        }
        // A block contributing nothing leaves the accumulator untouched.
        if (lane == 0 && total != (REAL)0) atomicAdd(destination[i] + slot, total);
    }
}

// Block maximum folded into ``destination`` with one atomic.
__device__ __forceinline__ void cmf_block_atomic_max(
    int value, int* destination)
{
    __shared__ int partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const unsigned active_mask = __activemask();

    for (int offset = 16; offset > 0; offset >>= 1) {
        value = max(
            value, cmf_shfl_down_max(value, active_mask, lane, offset));
    }
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (warp != 0) return;
    int total = (lane < num_warps) ? partial[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        total = max(
            total, cmf_shfl_down_max(total, active_mask, lane, offset));
    }
    if (lane == 0 && total > 0) atomicMax(destination, total);
}

#endif  // CMFGPU_BLOCK_REDUCE_CUH
