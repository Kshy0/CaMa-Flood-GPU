// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Helpers for the generated-entry bodies, which receive a route's canonical
// values as one struct.

#ifndef CMFGPU_CANONICAL_CUH
#define CMFGPU_CANONICAL_CUH

// An optional buffer reaches the body only while its feature is on; otherwise
// the body gets nullptr, whatever pointer type the absent buffer declares.
template <bool ON, class P>
__device__ __forceinline__ auto cmf_optional(P pointer)
{
    if constexpr (ON) return pointer;
    else return nullptr;
}

// Reservoir operation closes a bifurcation path whose upstream or downstream cell is
// dam-related: a reservoir cell or a cell draining directly into one.
template <bool HAS_RESERVOIR, class A>
__device__ __forceinline__ bool cmf_bifurcation_closed(const A& a, long t)
{
    if constexpr (HAS_RESERVOIR) {
        return a.is_dam_related_ptr[a.bifurcation_catchment_idx_ptr[t]]
            || a.is_dam_related_ptr[a.bifurcation_downstream_idx_ptr[t]];
    } else {
        return false;
    }
}

// Water-balance inputs of the *_log stage specs.  The plain stage specs have
// none, so their stage bodies take these null ones instead.
struct CmfNoBalance {
    decltype(nullptr) is_levee_ptr, total_storage_pre_sum_ptr, total_storage_next_sum_ptr,
        total_storage_new_sum_ptr, total_inflow_sum_ptr, total_outflow_sum_ptr,
        total_storage_stage_sum_ptr, river_storage_sum_ptr, flood_storage_sum_ptr,
        flood_area_sum_ptr, total_inflow_error_sum_ptr, total_stage_error_sum_ptr,
        current_step_ptr;
};

template <bool LOG, class A>
__device__ __forceinline__ auto cmf_balance(const A& a)
{
    if constexpr (LOG) return a;
    else return CmfNoBalance{};
}

// Fast integer bit-test for zero and bit-exact FP64-vs-FP32 comparison.
// On consumer NVIDIA GPUs (1/64 FP64 rate), avoiding unnecessary FP64 ops
// on dry cells cuts compute_flood_stage / outflow / inflow latency dramatically
// while preserving 100% IEEE-754 bit-exactness.
template <typename STO>
__device__ __forceinline__ bool cmf_is_zero_sto(STO x)
{
    if constexpr (sizeof(STO) == sizeof(double)) {
        return ((unsigned long long)__double_as_longlong(x) << 1) == 0ull;
    } else {
        return x == (STO)0;
    }
}

template <typename STO>
__device__ __forceinline__ bool cmf_is_neg_sto(STO x)
{
    if constexpr (sizeof(STO) == sizeof(double)) {
        unsigned long long bits = (unsigned long long)__double_as_longlong(x);
        return (bits - 0x8000000000000001ull) < 0x7ff0000000000000ull;
    } else {
        return x < (STO)0;
    }
}

// Bit-exact a > (STO)b where a32 == (REAL)a: by monotonicity of IEEE
// round-to-nearest-even, a32 > b => a > b and a32 < b => a < b; the FP64
// comparison is only needed on exact ties (a32 == b).
template <typename REAL, typename STO>
__device__ __forceinline__ bool cmf_gt_sto(STO a, REAL a32, REAL b)
{
    if constexpr (sizeof(STO) == sizeof(double) && sizeof(REAL) == sizeof(float)) {
        if (a32 > b) return true;
        if (a32 < b) return false;
        return a > (STO)b;
    } else {
        return a > (STO)b;
    }
}

// Fast fractional-power helpers for positive depths (d > 1e-5).
// Under HYDROFORGE_FAST_MATH=1, NVRTC's --use_fast_math does not replace cbrtf
// (it stays a ~42-instruction polynomial); using __powf(d, 1/3) + __fdividef
// achieves <= 8 ULP at ~2x higher throughput.
template <typename REAL>
__device__ __forceinline__ REAL cmf_inv_d73(REAL d)
{
#if HYDROFORGE_FAST_MATH && !defined(__HIPCC_RTC__)
    if constexpr (sizeof(REAL) == sizeof(float)) {
        return __fdividef(1.0f, d * d * __powf(d, 1.0f / 3.0f));
    }
#endif
    return (REAL)1.0 / (d * d * cbrt(d));
}

template <typename REAL>
__device__ __forceinline__ REAL cmf_inv_d43(REAL d)
{
#if HYDROFORGE_FAST_MATH && !defined(__HIPCC_RTC__)
    if constexpr (sizeof(REAL) == sizeof(float)) {
        return __fdividef(1.0f, d * __powf(d, 1.0f / 3.0f));
    }
#endif
    return (REAL)1.0 / (d * cbrt(d));
}

template <typename REAL>
__device__ __forceinline__ REAL cmf_pow23(REAL d)
{
#if HYDROFORGE_FAST_MATH && !defined(__HIPCC_RTC__)
    if constexpr (sizeof(REAL) == sizeof(float)) {
        const float x = d * d;
        return x > 0.0f ? __powf(x, 1.0f / 3.0f) : 0.0f;
    }
#endif
    return cbrt(d * d);
}

// Scatter accumulation into storage-typed buffers.
template <class T>
__device__ __forceinline__ void cmf_atomic_add(T* address, T value)
{
    atomicAdd(address, value);
}

#if defined(__GFX12__)
// gfx12's native global_atomic_add_f32 is several times slower than a 32-bit
// compare-and-swap loop (RX 9070 XT, 15' downstream scatter: 54 us vs 8 us),
// so FP32 storage accumulates through CAS there.  Each add stays one atomic
// read-modify-write; only the order of concurrent adds may differ, as with
// atomicAdd.
__device__ __forceinline__ void cmf_atomic_add(float* address, float value)
{
    unsigned int* word = reinterpret_cast<unsigned int*>(address);
    unsigned int observed = *word;
    unsigned int expected;
    do {
        expected = observed;
        observed = atomicCAS(word, expected, __float_as_uint(__uint_as_float(expected) + value));
    } while (observed != expected);
}
#endif

#endif  // CMFGPU_CANONICAL_CUH
