// CaMa-Flood-GPU — Common definitions for HIP kernels (AMD ROCm)
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#pragma once

#include <hip/hip_runtime.h>

// ── Ceiling division ─────────────────────────────────────────────────────
__host__ __device__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ── Compile-time constant sanity checks ──────────────────────────────────
#ifndef NUM_FLOOD_LEVELS
  #error "NUM_FLOOD_LEVELS must be defined at compile time (-DNUM_FLOOD_LEVELS=N)"
#endif
#ifndef NUM_BIF_LEVELS
  #error "NUM_BIF_LEVELS must be defined at compile time (-DNUM_BIF_LEVELS=N)"
#endif
#ifndef HAS_BIFURCATION_CONST
  #error "HAS_BIFURCATION_CONST must be defined at compile time"
#endif
#ifndef HAS_RESERVOIR_CONST
  #error "HAS_RESERVOIR_CONST must be defined at compile time"
#endif
#ifndef CMF_GRAVITY
  #error "CMF_GRAVITY must be defined at compile time (-DCMF_GRAVITY=9.80665f)"
#endif
#ifndef CMF_MIN_KINEMATIC_SLOPE
  #error "CMF_MIN_KINEMATIC_SLOPE must be defined at compile time"
#endif

// ── Thread-block size ────────────────────────────────────────────────────
#ifndef CMF_BLOCK_SIZE
  #define CMF_BLOCK_SIZE 256
#endif

// ── Storage precision ────────────────────────────────────────────────────
//
// By default storage accumulators use float64 (double) for numerical
// stability.  Defining CMF_STORAGE_FLOAT at compile time switches to
// float32, which is faster but can lose sub-daily precision on very
// large catchments.
//
#ifdef CMF_STORAGE_FLOAT
  using storage_t = float;
  #define STO_CAST(x) (x)
  #define STO_ZERO 0.0f
  #define sto_max  fmaxf
  #define sto_min  fminf
#else
  using storage_t = double;
  #define STO_CAST(x) static_cast<double>(x)
  #define STO_ZERO 0.0
  #define sto_max(a,b) fmax((a),(b))
  #define sto_min(a,b) fmin((a),(b))
#endif
