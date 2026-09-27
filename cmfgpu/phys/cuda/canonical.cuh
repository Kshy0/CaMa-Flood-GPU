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

#endif  // CMFGPU_CANONICAL_CUH
