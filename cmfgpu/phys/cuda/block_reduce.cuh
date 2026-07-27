// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Block reductions for the global scalar accumulators.  Same-address atomics
// serialize in L2, so folding inside the block turns O(num_catchments) atomics
// per counter into O(num_blocks) -- and the tree sum is far more accurate.
//
// Every thread must reach these: they use __syncthreads and full-mask
// shuffles, so callers keep out-of-range lanes alive with an identity value.

#ifndef CMFGPU_BLOCK_REDUCE_CUH
#define CMFGPU_BLOCK_REDUCE_CUH

#define CMF_FULL_WARP_MASK 0xffffffffu

// Sum N per-thread values across the block; one thread issues one atomic each.
template <typename REAL, int N>
__device__ __forceinline__ void cmf_block_atomic_add(
    REAL (&value)[N], REAL* const (&destination)[N], int slot)
{
    __shared__ REAL partial[N][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value[i] += __shfl_down_sync(CMF_FULL_WARP_MASK, value[i], offset);
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
            total += __shfl_down_sync(CMF_FULL_WARP_MASK, total, offset);
        }
        // A block contributing nothing leaves the accumulator untouched.
        if (lane == 0 && total != (REAL)0) atomicAdd(destination[i] + slot, total);
    }
}

// Block maximum folded into ``destination`` with one atomic.  Non-contributing
// lanes pass 0, the identity here because every real contribution is >= 1.
__device__ __forceinline__ void cmf_block_atomic_max(
    int value, int* destination)
{
    __shared__ int partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value = max(value, __shfl_down_sync(CMF_FULL_WARP_MASK, value, offset));
    }
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (warp != 0) return;
    int total = (lane < num_warps) ? partial[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        total = max(total, __shfl_down_sync(CMF_FULL_WARP_MASK, total, offset));
    }
    // Blocks with no contributing cell must not disturb the accumulator.
    if (lane == 0 && total > 0) atomicMax(destination, total);
}

#endif  // CMFGPU_BLOCK_REDUCE_CUH
