// CaMa-Flood-GPU — Shared CUDA kernel definitions
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
#pragma once

#include <cuda_runtime.h>
#include <math.h>

// ── Utility ──────────────────────────────────────────────────────────────

__host__ __device__ inline int cdiv(int n, int d) { return (n + d - 1) / d; }

// ── Compile-time constants (must be defined via -D flags) ────────────────

#ifndef NUM_FLOOD_LEVELS
#error "NUM_FLOOD_LEVELS must be defined at compile time (e.g. -DNUM_FLOOD_LEVELS=10)"
#endif

#ifndef NUM_BIF_LEVELS
#error "NUM_BIF_LEVELS must be defined at compile time (e.g. -DNUM_BIF_LEVELS=5)"
#endif

#ifndef HAS_BIFURCATION_CONST
#error "HAS_BIFURCATION_CONST must be defined (0 or 1)"
#endif

#ifndef HAS_RESERVOIR_CONST
#error "HAS_RESERVOIR_CONST must be defined (0 or 1)"
#endif

#ifndef CMF_GRAVITY
#define CMF_GRAVITY 9.80665f
#endif

#ifndef CMF_MIN_KINEMATIC_SLOPE
#define CMF_MIN_KINEMATIC_SLOPE 1.0e-5f
#endif

#ifndef CMF_BLOCK_SIZE
#define CMF_BLOCK_SIZE 256
#endif

// ── Mixed-precision storage type ─────────────────────────────────────────
//
// By default, storage / accumulation variables use double (float64).
// Compile with  -DCMF_STORAGE_FLOAT  to switch to single precision
// (float32) for faster throughput at the cost of reduced mass-conservation
// accuracy in long simulations.
//
// Affected variables: river_storage, flood_storage, protected_storage,
//   outgoing_storage, river_inflow, flood_inflow, global_bifurcation_outflow,
//   total_storage, reservoir_total_inflow.
// ─────────────────────────────────────────────────────────────────────────

#ifdef CMF_STORAGE_FLOAT
using storage_t = float;
#define STO_ZERO  0.0f
#define STO_CAST(x) static_cast<float>(x)
#else
using storage_t = double;
#define STO_ZERO  0.0
#define STO_CAST(x) static_cast<double>(x)
#endif

static __device__ __forceinline__ storage_t sto_max(storage_t a, storage_t b) {
#ifdef CMF_STORAGE_FLOAT
    return fmaxf(a, b);
#else
    return fmax(a, b);
#endif
}

static __device__ __forceinline__ storage_t sto_min(storage_t a, storage_t b) {
#ifdef CMF_STORAGE_FLOAT
    return fminf(a, b);
#else
    return fmin(a, b);
#endif
}
