template <typename REAL, typename STO>
__device__ __forceinline__ void k_bif_inflow_cell(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const REAL* __restrict__ limit_rate, REAL* __restrict__ outflow,
    STO* __restrict__ global_bif_outflow, int num_levels, long t)
{
    int ci = cat_idx[t];
    int di = dn_idx[t];
    REAL lr_c = __ldg(limit_rate + ci);
    REAL lr_d = __ldg(limit_rate + di);

    REAL raw_sum = (REAL)0;
    for (int lv = 0; lv < num_levels; ++lv) {
        long li = t * (long)num_levels + lv;
        REAL o = outflow[li];
        raw_sum += o;
        REAL upd = (o >= (REAL)0) ? o * lr_c : o * lr_d;
        outflow[li] = upd;
    }
    REAL net = (raw_sum >= (REAL)0) ? raw_sum * lr_c : raw_sum * lr_d;
    atomicAdd(global_bif_outflow + ci, (STO)net);
    atomicAdd(global_bif_outflow + di, (STO)(-net));
}

// Generated-entry body over the canonical values ``a`` of
// compute_bifurcation_inflow.
template <typename REAL, typename STO, class A>
__device__ __forceinline__ void bif_inflow(const A& a, long t)
{
    k_bif_inflow_cell<REAL, STO>(
        a.bifurcation_catchment_idx_ptr, a.bifurcation_downstream_idx_ptr, a.limit_rate_ptr,
        a.bifurcation_outflow_ptr, a.global_bifurcation_outflow_ptr,
        a.num_bifurcation_levels, t);
}

template <typename REAL>
__device__ __forceinline__ REAL pow73(REAL x) { return x * x * cbrt(x); }

template <typename REAL, typename STO>
__device__ __forceinline__ void k_bif_outflow_cell(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const REAL* __restrict__ manning, REAL* __restrict__ outflow,
    const REAL* __restrict__ width, const REAL* __restrict__ length,
    const REAL* __restrict__ elevation, REAL* __restrict__ cs_depth,
    const REAL* __restrict__ river_depth,
    const REAL* __restrict__ river_height,
    const REAL* __restrict__ catchment_elevation,
    const STO* __restrict__ river_storage,
    const STO* __restrict__ flood_storage,
    STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr, int num_levels, long t)
{
    REAL time_step = __ldg(time_step_ptr);

    int ci = cat_idx[t];
    int di = dn_idx[t];
    REAL blen = __ldg(length + t);

    // D2SFCELV = D2RIVDPH + D2RIVELV, with D2RIVELV = D2ELEVTN - D2RIVHGT.
    REAL wse_c = __ldg(river_depth + ci)
        + (__ldg(catchment_elevation + ci) - __ldg(river_height + ci));
    REAL wse_d = __ldg(river_depth + di)
        + (__ldg(catchment_elevation + di) - __ldg(river_height + di));
    REAL max_wse = fmax(wse_c, wse_d);

    REAL slope = (wse_c - wse_d) / blen;
    slope = fmin(fmax(slope, (REAL)-CMF_ROUTING_SLOPE_LIMIT), (REAL)CMF_ROUTING_SLOPE_LIMIT);

    REAL ts_c = (REAL)(river_storage[ci] + flood_storage[ci]);
    REAL ts_d = (REAL)(river_storage[di] + flood_storage[di]);

    // Carry the per-level updated flows in registers so each level is stored
    // exactly once, instead of storing then re-loading them for the limiter.
    // CaMa bifurcation maps use NPTHLEV <= 5; the generic path below keeps
    // correctness for any level count.
    constexpr int MAX_REGISTER_LEVELS = 8;
    REAL sum_out = (REAL)0;
    REAL upd[MAX_REGISTER_LEVELS];
    const bool fits = (num_levels <= MAX_REGISTER_LEVELS);

    if (fits) {
#pragma unroll
        for (int lv = 0; lv < MAX_REGISTER_LEVELS; ++lv) {
            REAL upd_o = (REAL)0;
            if (lv < num_levels) {
                long li = t * (long)num_levels + lv;
                REAL man = __ldg(manning + li);
                REAL csd = __ldg(cs_depth + li);
                REAL elv = __ldg(elevation + li);
                REAL upd_csd = fmax(max_wse - elv, (REAL)0);
                REAL sifd = fmax(sqrt(upd_csd * csd), sqrt(upd_csd * (REAL)0.01));
                if (sifd > (REAL)1e-5) {
                    REAL w = __ldg(width + li);
                    REAL o = outflow[li];
                    REAL unit_o = o / w;
                    REAL num = w * (unit_o + gravity * time_step * sifd * slope);
                    REAL den = (REAL)1 + gravity * time_step * (man * man)
                        * fabs(unit_o) * ((REAL)1 / pow73(sifd));
                    upd_o = num / den;
                }
                sum_out += upd_o;
                cs_depth[li] = upd_csd;
            }
            upd[lv] = upd_o;
        }
    } else {
        for (int lv = 0; lv < num_levels; ++lv) {
            long li = t * (long)num_levels + lv;
            REAL man = __ldg(manning + li);
            REAL csd = __ldg(cs_depth + li);
            REAL elv = __ldg(elevation + li);
            REAL upd_csd = fmax(max_wse - elv, (REAL)0);
            REAL sifd = fmax(sqrt(upd_csd * csd), sqrt(upd_csd * (REAL)0.01));
            bool flow_condition = sifd > (REAL)1e-5;
            REAL upd_o = (REAL)0;
            if (flow_condition) {
                REAL w = __ldg(width + li);
                REAL o = outflow[li];
                REAL unit_o = o / w;
                REAL num = w * (unit_o + gravity * time_step * sifd * slope);
                REAL den = (REAL)1 + gravity * time_step * (man * man)
                    * fabs(unit_o) * ((REAL)1 / pow73(sifd));
                upd_o = num / den;
            }
            sum_out += upd_o;
            cs_depth[li] = upd_csd;
            outflow[li] = upd_o;
        }
    }

    // v4.23 storage-change limiter; a path whose levels sum to zero is left as is.
    REAL limit_rate = (sum_out != (REAL)0)
        ? fmin((REAL)CMF_BACKFLOW_STORAGE_FRACTION * fmin(ts_c, ts_d) / (fabs(sum_out) * time_step), (REAL)1)
        : (REAL)1;
    sum_out *= limit_rate;
    if (fits) {
#pragma unroll
        for (int lv = 0; lv < MAX_REGISTER_LEVELS; ++lv) {
            if (lv < num_levels) {
                outflow[t * (long)num_levels + lv] = upd[lv] * limit_rate;
            }
        }
    } else {
        for (int lv = 0; lv < num_levels; ++lv) {
            long li = t * (long)num_levels + lv;
            outflow[li] = outflow[li] * limit_rate;
        }
    }

    REAL pos = fmax(sum_out, (REAL)0);
    REAL neg = fmin(sum_out, (REAL)0);
    // P2STOOUT flows, multiplied by the step in compute_inflow.
    atomicAdd(outgoing_storage + ci, (STO)pos);
    atomicAdd(outgoing_storage + di, (STO)(-neg));
}

// Generated-entry body over the canonical values ``a`` of
// compute_bifurcation_outflow.
template <typename REAL, typename STO, class A>
__device__ __forceinline__ void bif_outflow(const A& a, long t)
{
    k_bif_outflow_cell<REAL, STO>(
        a.bifurcation_catchment_idx_ptr, a.bifurcation_downstream_idx_ptr,
        a.bifurcation_manning_ptr, a.bifurcation_outflow_ptr, a.bifurcation_width_ptr,
        a.bifurcation_length_ptr, a.bifurcation_elevation_ptr,
        a.bifurcation_cross_section_depth_ptr, a.river_depth_ptr, a.river_height_ptr,
        a.catchment_elevation_ptr, a.river_storage_ptr, a.flood_storage_ptr,
        a.outgoing_storage_ptr, a.gravity, a.time_step_ptr, a.num_bifurcation_levels, t);
}

// Ensemble members offset their slices, then run the bodies above.
template <typename REAL, typename STO, class A>
__device__ __forceinline__ void bif_inflow_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    a.limit_rate_ptr += member_offset;
    a.global_bifurcation_outflow_ptr += member_offset;
    a.bifurcation_outflow_ptr += member * a.num_bifurcation_paths * a.num_bifurcation_levels;
    bif_inflow<REAL, STO>(a, t);
}

template <typename REAL, typename STO, class A>
__device__ __forceinline__ void bif_outflow_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    const long path_offset = member * a.num_bifurcation_paths;
    const long level_offset = path_offset * a.num_bifurcation_levels;
    a.bifurcation_outflow_ptr += level_offset;
    a.bifurcation_cross_section_depth_ptr += level_offset;
    a.river_depth_ptr += member_offset;
    a.river_storage_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    a.outgoing_storage_ptr += member_offset;
    if (a.batched_bifurcation_manning) a.bifurcation_manning_ptr += level_offset;
    if (a.batched_bifurcation_width) a.bifurcation_width_ptr += level_offset;
    if (a.batched_bifurcation_length) a.bifurcation_length_ptr += path_offset;
    if (a.batched_bifurcation_elevation) a.bifurcation_elevation_ptr += level_offset;
    if (a.batched_river_height) a.river_height_ptr += member_offset;
    if (a.batched_catchment_elevation) a.catchment_elevation_ptr += member_offset;
    bif_outflow<REAL, STO>(a, t);
}
