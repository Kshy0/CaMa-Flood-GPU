// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;
constant bool batched_river_length_flag [[function_constant(0)]];
constant bool batched_river_width_flag [[function_constant(1)]];
constant bool batched_river_height_flag [[function_constant(2)]];
constant bool batched_catchment_area_flag [[function_constant(3)]];
constant bool batched_levee_crown_height_flag [[function_constant(4)]];
constant bool batched_levee_fraction_flag [[function_constant(5)]];
constant bool batched_levee_base_height_flag [[function_constant(6)]];
constant bool batched_flood_depth_table_flag [[function_constant(7)]];

// log_sums layout  (row major, stride = log_buffer_size):
//   row  6 = total_storage_stage_sum
//   row  7 = total_stage_error_sum
//   row  8 = river_storage_sum
//   row  9 = flood_storage_sum
//   row 10 = flood_area_sum

static inline void atomic_add_float(device atomic_float* addr, float val) {
    atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

struct compute_levee_stage_args {
    device int* levee_catchment_idx_buf [[id(0)]];
    device float* river_storage_buf [[id(1)]];
    device float* flood_storage_buf [[id(2)]];
    device float* protected_storage_buf [[id(3)]];
    device float* river_depth_buf [[id(4)]];
    device float* flood_depth_buf [[id(5)]];
    device float* protected_depth_buf [[id(6)]];
    device float* river_height_buf [[id(7)]];
    device float* flood_depth_table_buf [[id(8)]];
    device float* catchment_area_buf [[id(9)]];
    device float* river_width_buf [[id(10)]];
    device float* river_length_buf [[id(11)]];
    device float* levee_base_height_buf [[id(12)]];
    device float* levee_crown_height_buf [[id(13)]];
    device float* levee_fraction_buf [[id(14)]];
    device float* flood_fraction_buf [[id(15)]];
    constant int* num_levees [[id(16)]];
};

kernel void compute_levee_stage(
    constant compute_levee_stage_args& args [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
)
{
    device int* levee_catchment_idx_buf = args.levee_catchment_idx_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* protected_storage_buf = args.protected_storage_buf;
    device float* river_depth_buf = args.river_depth_buf;
    device float* flood_depth_buf = args.flood_depth_buf;
    device float* protected_depth_buf = args.protected_depth_buf;
    device float* river_height_buf = args.river_height_buf;
    device float* flood_depth_table_buf = args.flood_depth_table_buf;
    device float* catchment_area_buf = args.catchment_area_buf;
    device float* river_width_buf = args.river_width_buf;
    device float* river_length_buf = args.river_length_buf;
    device float* levee_base_height_buf = args.levee_base_height_buf;
    device float* levee_crown_height_buf = args.levee_crown_height_buf;
    device float* levee_fraction_buf = args.levee_fraction_buf;
    device float* flood_fraction_buf = args.flood_fraction_buf;
    const int num_levees = *args.num_levees;
    const int NUM_FLOOD_LEVELS = __NUM_FLOOD_LEVELS__;

    if ((int)idx >= num_levees) return;

    int ci = levee_catchment_idx_buf[idx];

    float river_length  = river_length_buf[ci];
    float river_width   = river_width_buf[ci];
    float river_height  = river_height_buf[ci];
    float catchment_area = catchment_area_buf[ci];

    float levee_crown_height = levee_crown_height_buf[idx];
    float levee_fraction     = levee_fraction_buf[idx];
    float levee_base_height  = levee_base_height_buf[idx];

    float river_storage_curr = river_storage_buf[ci];
    float flood_storage_curr = flood_storage_buf[ci];
    float flood_depth_curr   = flood_depth_buf[ci];

    float total_storage = river_storage_curr + flood_storage_curr;

    float river_max_storage = river_length * river_width * river_height;
    if (total_storage <= river_max_storage) {
        river_storage_buf[ci]     = river_storage_curr;
        flood_storage_buf[ci]     = flood_storage_curr;
        protected_storage_buf[ci] = 0.0f;
        protected_depth_buf[ci]   = 0.0f;
        return;
    }

    float dwth_inc = (catchment_area / river_length) / (float)NUM_FLOOD_LEVELS;
    float levee_distance = levee_fraction * (catchment_area / river_length);

    // Table scan — find levee_base_storage & levee_fill_storage
    float s_curr = river_max_storage;
    float dhgt_pre = 0.0f;
    float dwth_pre = river_width;

    float levee_base_storage = river_max_storage;
    float levee_fill_storage = river_max_storage;
    int found_base = 0;
    int found_fill = 0;

    // Case 3 search B state
    int ilev = (int)(levee_fraction * (float)NUM_FLOOD_LEVELS);
    float dsto_fil_B = 0.0f;
    float dwth_fil_B = 0.0f;
    float ddph_fil_B = 0.0f;
    float gradient_B = 0.0f;
    int found_B = 0;

    for (int i = 0; i < NUM_FLOOD_LEVELS; i++) {
        float depth_val = flood_depth_table_buf[ci * NUM_FLOOD_LEVELS + i];
        float dhgt_seg = max(depth_val - dhgt_pre, 1e-6f);
        float dwth_mid = dwth_pre + 0.5f * dwth_inc;
        float dsto_seg = river_length * dwth_mid * dhgt_seg;
        float s_next   = s_curr + dsto_seg;
        float gradient = dhgt_seg / dwth_inc;

        // Check Base
        bool cond_base = (levee_base_height > dhgt_pre) && (levee_base_height <= depth_val);
        if (cond_base && !found_base) {
            float ratio_base = (levee_base_height - dhgt_pre) / dhgt_seg;
            float dsto_base_partial = river_length * (dwth_pre + 0.5f * ratio_base * dwth_inc) * (ratio_base * dhgt_seg);
            levee_base_storage = s_curr + dsto_base_partial;
            found_base = 1;
        }

        // Check Fill
        bool cond_fill = (levee_crown_height > dhgt_pre) && (levee_crown_height <= depth_val);
        if (cond_fill && !found_fill) {
            float ratio_fill = (levee_crown_height - dhgt_pre) / dhgt_seg;
            float dsto_fill_partial = river_length * (dwth_pre + 0.5f * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg);
            levee_fill_storage = s_curr + dsto_fill_partial;
            found_fill = 1;
        }

        // Case 3 Search B
        if (i >= ilev && !found_B) {
            float dhgt_dif_loop = levee_crown_height - levee_base_height;
            float s_top_loop = levee_base_storage + (levee_distance + river_width) * dhgt_dif_loop * river_length;
            float dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length;
            float threshold = s_next + dsto_add_wedge;

            if (total_storage < threshold) {
                // Found
                if (i == ilev) {
                    dsto_fil_B = s_top_loop;
                }
                gradient_B = gradient;
                found_B = 1;
            } else {
                // Not found yet — update lower bound
                dsto_fil_B = threshold;
                dwth_fil_B = dwth_inc * (float)(i + 1) - levee_distance;
                ddph_fil_B = depth_val - levee_base_height;
            }
        }

        s_curr = s_next;
        dhgt_pre = depth_val;
        dwth_pre += dwth_inc;
        if (found_base && found_fill && found_B) break;
    }

    // Handle out of bounds
    if (!found_base) {
        levee_base_storage = (levee_base_height > dhgt_pre)
            ? s_curr + river_length * dwth_pre * (levee_base_height - dhgt_pre)
            : river_max_storage;
    }
    if (!found_fill) {
        levee_fill_storage = (levee_crown_height > dhgt_pre)
            ? s_curr + river_length * dwth_pre * (levee_crown_height - dhgt_pre)
            : river_max_storage;
    }

    // Calculate s_top
    float dhgt_dif = levee_crown_height - levee_base_height;
    float s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length;

    // Determine Case
    bool is_case4 = (total_storage >= levee_fill_storage);
    bool is_case3 = !is_case4 && (total_storage >= s_top);
    bool is_case2 = !is_case4 && !is_case3 && (total_storage >= levee_base_storage);

    // Outputs
    float r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc;

    if (is_case2) {
        float dsto_add = total_storage - levee_base_storage;
        float dwth_add = levee_distance + river_width;
        f_dph = levee_base_height + dsto_add / dwth_add / river_length;
        r_sto = river_max_storage + river_length * river_width * f_dph;
        r_dph = r_sto / river_length / river_width;
        f_sto = max(total_storage - r_sto, 0.0f);
        p_sto = 0.0f;
        p_dph = 0.0f;
        f_frc = levee_fraction;
    } else if (is_case3) {
        // Search B results
        float dsto_add_B = total_storage - dsto_fil_B;
        float term_B = dwth_fil_B * dwth_fil_B + 2.0f * dsto_add_B / river_length / (gradient_B + 1e-9f);
        float dwth_add_B = -dwth_fil_B + sqrt(max(term_B, 0.0f));
        float ddph_add_B = dwth_add_B * gradient_B;

        float p_dph_B, f_frc_B;
        if (found_B) {
            p_dph_B = levee_base_height + ddph_fil_B + ddph_add_B;
            f_frc_B = (dwth_fil_B + levee_distance) / (dwth_inc * (float)NUM_FLOOD_LEVELS);
        } else {
            float ddph_add_extra = dsto_add_B / (dwth_fil_B * river_length + 1e-9f);
            p_dph_B = levee_base_height + ddph_fil_B + ddph_add_extra;
            f_frc_B = 1.0f;
        }

        f_dph = levee_crown_height;
        r_sto = river_max_storage + river_length * river_width * f_dph;
        r_dph = r_sto / river_length / river_width;
        f_sto = max(s_top - r_sto, 0.0f);
        p_sto = max(total_storage - r_sto - f_sto, 0.0f);
        p_dph = p_dph_B;
        f_frc = clamp(f_frc_B, 0.0f, 1.0f);
    } else if (is_case4) {
        f_dph = flood_depth_curr;
        r_sto = river_storage_curr;
        float dsto_add = (f_dph - levee_crown_height) * (levee_distance + river_width) * river_length;
        f_sto = max(s_top + dsto_add - r_sto, 0.0f);
        p_sto = max(total_storage - r_sto - f_sto, 0.0f);
        p_dph = f_dph;
        r_dph = river_depth_buf[ci];
        f_frc = flood_fraction_buf[ci];
    } else {
        // Default (below levee base) — keep current
        r_sto = river_storage_curr;
        f_sto = flood_storage_curr;
        p_sto = 0.0f;
        r_dph = river_depth_buf[ci];
        f_dph = flood_depth_curr;
        p_dph = 0.0f;
        f_frc = flood_fraction_buf[ci];
    }

    // Store results
    river_storage_buf[ci]     = r_sto;
    flood_storage_buf[ci]     = f_sto;
    protected_storage_buf[ci] = p_sto;
    river_depth_buf[ci]       = r_dph;
    flood_depth_buf[ci]       = f_dph;
    protected_depth_buf[ci]   = p_dph;
    flood_fraction_buf[ci]    = f_frc;
}


// =====================================================================
// Batched levee stage kernel — loop-based: grid=num_levees, loops trials
// =====================================================================
struct compute_levee_stage_batched_args {
    device int* levee_catchment_idx_buf [[id(0)]];
    device float* river_storage_buf [[id(1)]];
    device float* flood_storage_buf [[id(2)]];
    device float* protected_storage_buf [[id(3)]];
    device float* river_depth_buf [[id(4)]];
    device float* flood_depth_buf [[id(5)]];
    device float* protected_depth_buf [[id(6)]];
    device float* river_height_buf [[id(7)]];
    device float* flood_depth_table_buf [[id(8)]];
    device float* catchment_area_buf [[id(9)]];
    device float* river_width_buf [[id(10)]];
    device float* river_length_buf [[id(11)]];
    device float* levee_base_height_buf [[id(12)]];
    device float* levee_crown_height_buf [[id(13)]];
    device float* levee_fraction_buf [[id(14)]];
    device float* flood_fraction_buf [[id(15)]];
    constant int* num_levees [[id(16)]];
    constant int* num_catchments [[id(17)]];
    constant int* num_trials [[id(18)]];
};

kernel void compute_levee_stage_batched(
    constant compute_levee_stage_batched_args& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
)
{
    device int* levee_catchment_idx_buf = args.levee_catchment_idx_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* protected_storage_buf = args.protected_storage_buf;
    device float* river_depth_buf = args.river_depth_buf;
    device float* flood_depth_buf = args.flood_depth_buf;
    device float* protected_depth_buf = args.protected_depth_buf;
    device float* river_height_buf = args.river_height_buf;
    device float* flood_depth_table_buf = args.flood_depth_table_buf;
    device float* catchment_area_buf = args.catchment_area_buf;
    device float* river_width_buf = args.river_width_buf;
    device float* river_length_buf = args.river_length_buf;
    device float* levee_base_height_buf = args.levee_base_height_buf;
    device float* levee_crown_height_buf = args.levee_crown_height_buf;
    device float* levee_fraction_buf = args.levee_fraction_buf;
    device float* flood_fraction_buf = args.flood_fraction_buf;
    const int num_levees = *args.num_levees;
    const int num_catchments = *args.num_catchments;
    const int num_trials = *args.num_trials;
    const int NF = __NUM_FLOOD_LEVELS__;

    if ((int)gid >= num_levees) return;

    int li = (int)gid;  // levee index
    int ci = levee_catchment_idx_buf[li];

    // ---- Load shared (non-trial) params once ----
    float rl_s = batched_river_length_flag ? 0.0f : river_length_buf[ci];
    float rw_s = batched_river_width_flag  ? 0.0f : river_width_buf[ci];
    float rh_s = batched_river_height_flag ? 0.0f : river_height_buf[ci];
    float ca_s = batched_catchment_area_flag ? 0.0f : catchment_area_buf[ci];
    float lch_s = batched_levee_crown_height_flag ? 0.0f : levee_crown_height_buf[li];
    float lf_s  = batched_levee_fraction_flag ? 0.0f : levee_fraction_buf[li];
    float lbh_s = batched_levee_base_height_flag ? 0.0f : levee_base_height_buf[li];

    for (int t = 0; t < num_trials; t++) {
        int to_c = t * num_catchments;
        int to_l = t * num_levees;
        int ci_g = to_c + ci;

        float river_length  = batched_river_length_flag ? river_length_buf[ci_g] : rl_s;
        float river_width   = batched_river_width_flag  ? river_width_buf[ci_g]  : rw_s;
        float river_height  = batched_river_height_flag ? river_height_buf[ci_g] : rh_s;
        float catchment_area = batched_catchment_area_flag ? catchment_area_buf[ci_g] : ca_s;
        float levee_crown_height = batched_levee_crown_height_flag ? levee_crown_height_buf[to_l + li] : lch_s;
        float levee_fraction     = batched_levee_fraction_flag ? levee_fraction_buf[to_l + li] : lf_s;
        float levee_base_height  = batched_levee_base_height_flag ? levee_base_height_buf[to_l + li] : lbh_s;

        float river_storage_curr = river_storage_buf[ci_g];
        float flood_storage_curr = flood_storage_buf[ci_g];
        float flood_depth_curr   = flood_depth_buf[ci_g];

        float total_storage = river_storage_curr + flood_storage_curr;

        float river_max_storage = river_length * river_width * river_height;
        if (total_storage <= river_max_storage) {
            river_storage_buf[ci_g]     = river_storage_curr;
            flood_storage_buf[ci_g]     = flood_storage_curr;
            protected_storage_buf[ci_g] = 0.0f;
            protected_depth_buf[ci_g]   = 0.0f;
            continue;
        }

        float dwth_inc = (catchment_area / river_length) / (float)NF;
        float levee_distance = levee_fraction * (catchment_area / river_length);

        float s_curr = river_max_storage;
        float dhgt_pre = 0.0f;
        float dwth_pre = river_width;
        float levee_base_storage = river_max_storage;
        float levee_fill_storage = river_max_storage;
        int found_base = 0, found_fill = 0;

        int ilev = (int)(levee_fraction * (float)NF);
        float dsto_fil_B = 0.0f, dwth_fil_B = 0.0f, ddph_fil_B = 0.0f, gradient_B = 0.0f;
        int found_B = 0;

        int table_base = batched_flood_depth_table_flag ? (to_c * NF) : 0;

        for (int i = 0; i < NF; i++) {
            float depth_val = flood_depth_table_buf[table_base + ci * NF + i];
            float dhgt_seg = max(depth_val - dhgt_pre, 1e-6f);
            float dwth_mid = dwth_pre + 0.5f * dwth_inc;
            float dsto_seg = river_length * dwth_mid * dhgt_seg;
            float s_next   = s_curr + dsto_seg;
            float gradient = dhgt_seg / dwth_inc;

            bool cond_base = (levee_base_height > dhgt_pre) && (levee_base_height <= depth_val);
            if (cond_base && !found_base) {
                float ratio = (levee_base_height - dhgt_pre) / dhgt_seg;
                levee_base_storage = s_curr + river_length * (dwth_pre + 0.5f * ratio * dwth_inc) * (ratio * dhgt_seg);
                found_base = 1;
            }

            bool cond_fill = (levee_crown_height > dhgt_pre) && (levee_crown_height <= depth_val);
            if (cond_fill && !found_fill) {
                float ratio = (levee_crown_height - dhgt_pre) / dhgt_seg;
                levee_fill_storage = s_curr + river_length * (dwth_pre + 0.5f * ratio * dwth_inc) * (ratio * dhgt_seg);
                found_fill = 1;
            }

            if (i >= ilev && !found_B) {
                float dhgt_dif_loop = levee_crown_height - levee_base_height;
                float s_top_loop = levee_base_storage + (levee_distance + river_width) * dhgt_dif_loop * river_length;
                float dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length;
                float threshold = s_next + dsto_add_wedge;

                if (total_storage < threshold) {
                    if (i == ilev) dsto_fil_B = s_top_loop;
                    gradient_B = gradient;
                    found_B = 1;
                } else {
                    dsto_fil_B = threshold;
                    dwth_fil_B = dwth_inc * (float)(i + 1) - levee_distance;
                    ddph_fil_B = depth_val - levee_base_height;
                }
            }

            s_curr = s_next;
            dhgt_pre = depth_val;
            dwth_pre += dwth_inc;
            if (found_base && found_fill && found_B) break;
        }

        if (!found_base) {
            levee_base_storage = (levee_base_height > dhgt_pre)
                ? s_curr + river_length * dwth_pre * (levee_base_height - dhgt_pre) : river_max_storage;
        }
        if (!found_fill) {
            levee_fill_storage = (levee_crown_height > dhgt_pre)
                ? s_curr + river_length * dwth_pre * (levee_crown_height - dhgt_pre) : river_max_storage;
        }

        float dhgt_dif = levee_crown_height - levee_base_height;
        float s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length;

        bool is_case4 = (total_storage >= levee_fill_storage);
        bool is_case3 = !is_case4 && (total_storage >= s_top);
        bool is_case2 = !is_case4 && !is_case3 && (total_storage >= levee_base_storage);

        float r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc;

        if (is_case2) {
            float dsto_add = total_storage - levee_base_storage;
            float dwth_add = levee_distance + river_width;
            f_dph = levee_base_height + dsto_add / dwth_add / river_length;
            r_sto = river_max_storage + river_length * river_width * f_dph;
            r_dph = r_sto / river_length / river_width;
            f_sto = max(total_storage - r_sto, 0.0f);
            p_sto = 0.0f; p_dph = 0.0f;
            f_frc = levee_fraction;
        } else if (is_case3) {
            float dsto_add_B = total_storage - dsto_fil_B;
            float term_B = dwth_fil_B * dwth_fil_B + 2.0f * dsto_add_B / river_length / (gradient_B + 1e-9f);
            float dwth_add_B = -dwth_fil_B + sqrt(max(term_B, 0.0f));
            float ddph_add_B = dwth_add_B * gradient_B;

            float p_dph_B, f_frc_B;
            if (found_B) {
                p_dph_B = levee_base_height + ddph_fil_B + ddph_add_B;
                f_frc_B = (dwth_fil_B + levee_distance) / (dwth_inc * (float)NF);
            } else {
                float ddph_add_extra = dsto_add_B / (dwth_fil_B * river_length + 1e-9f);
                p_dph_B = levee_base_height + ddph_fil_B + ddph_add_extra;
                f_frc_B = 1.0f;
            }

            f_dph = levee_crown_height;
            r_sto = river_max_storage + river_length * river_width * f_dph;
            r_dph = r_sto / river_length / river_width;
            f_sto = max(s_top - r_sto, 0.0f);
            p_sto = max(total_storage - r_sto - f_sto, 0.0f);
            p_dph = p_dph_B;
            f_frc = clamp(f_frc_B, 0.0f, 1.0f);
        } else if (is_case4) {
            f_dph = flood_depth_curr;
            r_sto = river_storage_curr;
            float dsto_add = (f_dph - levee_crown_height) * (levee_distance + river_width) * river_length;
            f_sto = max(s_top + dsto_add - r_sto, 0.0f);
            p_sto = max(total_storage - r_sto - f_sto, 0.0f);
            p_dph = f_dph;
            r_dph = river_depth_buf[ci_g];
            f_frc = flood_fraction_buf[ci_g];
        } else {
            r_sto = river_storage_curr;
            f_sto = flood_storage_curr;
            p_sto = 0.0f;
            r_dph = river_depth_buf[ci_g];
            f_dph = flood_depth_curr;
            p_dph = 0.0f;
            f_frc = flood_fraction_buf[ci_g];
        }

        river_storage_buf[ci_g]     = r_sto;
        flood_storage_buf[ci_g]     = f_sto;
        protected_storage_buf[ci_g] = p_sto;
        river_depth_buf[ci_g]       = r_dph;
        flood_depth_buf[ci_g]       = f_dph;
        protected_depth_buf[ci_g]   = p_dph;
        flood_fraction_buf[ci_g]    = f_frc;
    }
}

struct compute_levee_stage_log_args {
    device int* levee_catchment_idx_buf [[id(0)]];
    device float* river_storage_buf [[id(1)]];
    device float* flood_storage_buf [[id(2)]];
    device float* protected_storage_buf [[id(3)]];
    device float* river_depth_buf [[id(4)]];
    device float* flood_depth_buf [[id(5)]];
    device float* protected_depth_buf [[id(6)]];
    device float* river_height_buf [[id(7)]];
    device float* flood_depth_table_buf [[id(8)]];
    device float* catchment_area_buf [[id(9)]];
    device float* river_width_buf [[id(10)]];
    device float* river_length_buf [[id(11)]];
    device float* levee_base_height_buf [[id(12)]];
    device float* levee_crown_height_buf [[id(13)]];
    device float* levee_fraction_buf [[id(14)]];
    device float* flood_fraction_buf [[id(15)]];
    // Packed log sums — (11, log_buffer_size) contiguous
    device atomic_float* log_sums_buf [[id(16)]];
    constant int* num_levees [[id(17)]];
    device int* current_step_buf [[id(18)]];
    constant int* log_buffer_size [[id(19)]];
};

kernel void compute_levee_stage_log(
    constant compute_levee_stage_log_args& args [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
)
{
    device int* levee_catchment_idx_buf = args.levee_catchment_idx_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* protected_storage_buf = args.protected_storage_buf;
    device float* river_depth_buf = args.river_depth_buf;
    device float* flood_depth_buf = args.flood_depth_buf;
    device float* protected_depth_buf = args.protected_depth_buf;
    device float* river_height_buf = args.river_height_buf;
    device float* flood_depth_table_buf = args.flood_depth_table_buf;
    device float* catchment_area_buf = args.catchment_area_buf;
    device float* river_width_buf = args.river_width_buf;
    device float* river_length_buf = args.river_length_buf;
    device float* levee_base_height_buf = args.levee_base_height_buf;
    device float* levee_crown_height_buf = args.levee_crown_height_buf;
    device float* levee_fraction_buf = args.levee_fraction_buf;
    device float* flood_fraction_buf = args.flood_fraction_buf;
    // Packed log sums — (11, log_buffer_size) contiguous
    device atomic_float* log_sums_buf = args.log_sums_buf;
    const int num_levees = *args.num_levees;
    device int* current_step_buf = args.current_step_buf;
    const int log_buffer_size = *args.log_buffer_size;
    const int NUM_FLOOD_LEVELS = __NUM_FLOOD_LEVELS__;

    if ((int)idx >= num_levees) return;

    int current_step = current_step_buf[0];
    int lbs = log_buffer_size;
    int ci = levee_catchment_idx_buf[idx];

    float river_length  = river_length_buf[ci];
    float river_width   = river_width_buf[ci];
    float river_height  = river_height_buf[ci];
    float catchment_area = catchment_area_buf[ci];

    float levee_crown_height = levee_crown_height_buf[idx];
    float levee_fraction     = levee_fraction_buf[idx];
    float levee_base_height  = levee_base_height_buf[idx];

    float river_storage_curr = river_storage_buf[ci];
    float flood_storage_curr = flood_storage_buf[ci];
    float flood_depth_curr   = flood_depth_buf[ci];

    float total_storage = river_storage_curr + flood_storage_curr;

    float river_max_storage = river_length * river_width * river_height;
    if (total_storage <= river_max_storage) {
        float total_storage_stage_new = river_storage_curr + flood_storage_curr;
        atomic_add_float(&log_sums_buf[6  * lbs + current_step], total_storage_stage_new * 1e-9f);
        atomic_add_float(&log_sums_buf[8  * lbs + current_step], river_storage_curr * 1e-9f);
        atomic_add_float(&log_sums_buf[9  * lbs + current_step], flood_storage_curr * 1e-9f);
        atomic_add_float(&log_sums_buf[10 * lbs + current_step], flood_fraction_buf[ci] * catchment_area * 1e-9f);

        river_storage_buf[ci]     = river_storage_curr;
        flood_storage_buf[ci]     = flood_storage_curr;
        protected_storage_buf[ci] = 0.0f;
        protected_depth_buf[ci]   = 0.0f;
        return;
    }

    float dwth_inc = (catchment_area / river_length) / (float)NUM_FLOOD_LEVELS;
    float levee_distance = levee_fraction * (catchment_area / river_length);

    // Table scan — find levee_base_storage & levee_fill_storage
    float s_curr = river_max_storage;
    float dhgt_pre = 0.0f;
    float dwth_pre = river_width;

    float levee_base_storage = river_max_storage;
    float levee_fill_storage = river_max_storage;
    int found_base = 0;
    int found_fill = 0;

    // Case 3 search B state
    int ilev = (int)(levee_fraction * (float)NUM_FLOOD_LEVELS);
    float dsto_fil_B = 0.0f;
    float dwth_fil_B = 0.0f;
    float ddph_fil_B = 0.0f;
    float gradient_B = 0.0f;
    int found_B = 0;

    for (int i = 0; i < NUM_FLOOD_LEVELS; i++) {
        float depth_val = flood_depth_table_buf[ci * NUM_FLOOD_LEVELS + i];
        float dhgt_seg = max(depth_val - dhgt_pre, 1e-6f);
        float dwth_mid = dwth_pre + 0.5f * dwth_inc;
        float dsto_seg = river_length * dwth_mid * dhgt_seg;
        float s_next   = s_curr + dsto_seg;
        float gradient = dhgt_seg / dwth_inc;

        // Check Base
        bool cond_base = (levee_base_height > dhgt_pre) && (levee_base_height <= depth_val);
        if (cond_base && !found_base) {
            float ratio_base = (levee_base_height - dhgt_pre) / dhgt_seg;
            float dsto_base_partial = river_length * (dwth_pre + 0.5f * ratio_base * dwth_inc) * (ratio_base * dhgt_seg);
            levee_base_storage = s_curr + dsto_base_partial;
            found_base = 1;
        }

        // Check Fill
        bool cond_fill = (levee_crown_height > dhgt_pre) && (levee_crown_height <= depth_val);
        if (cond_fill && !found_fill) {
            float ratio_fill = (levee_crown_height - dhgt_pre) / dhgt_seg;
            float dsto_fill_partial = river_length * (dwth_pre + 0.5f * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg);
            levee_fill_storage = s_curr + dsto_fill_partial;
            found_fill = 1;
        }

        // Case 3 Search B
        if (i >= ilev && !found_B) {
            float dhgt_dif_loop = levee_crown_height - levee_base_height;
            float s_top_loop = levee_base_storage + (levee_distance + river_width) * dhgt_dif_loop * river_length;
            float dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length;
            float threshold = s_next + dsto_add_wedge;

            if (total_storage < threshold) {
                if (i == ilev) {
                    dsto_fil_B = s_top_loop;
                }
                gradient_B = gradient;
                found_B = 1;
            } else {
                dsto_fil_B = threshold;
                dwth_fil_B = dwth_inc * (float)(i + 1) - levee_distance;
                ddph_fil_B = depth_val - levee_base_height;
            }
        }

        s_curr = s_next;
        dhgt_pre = depth_val;
        dwth_pre += dwth_inc;
        if (found_base && found_fill && found_B) break;
    }

    // Handle out of bounds
    if (!found_base) {
        levee_base_storage = (levee_base_height > dhgt_pre)
            ? s_curr + river_length * dwth_pre * (levee_base_height - dhgt_pre)
            : river_max_storage;
    }
    if (!found_fill) {
        levee_fill_storage = (levee_crown_height > dhgt_pre)
            ? s_curr + river_length * dwth_pre * (levee_crown_height - dhgt_pre)
            : river_max_storage;
    }

    // Calculate s_top
    float dhgt_dif = levee_crown_height - levee_base_height;
    float s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length;

    // Determine Case
    bool is_case4 = (total_storage >= levee_fill_storage);
    bool is_case3 = !is_case4 && (total_storage >= s_top);
    bool is_case2 = !is_case4 && !is_case3 && (total_storage >= levee_base_storage);

    // Outputs
    float r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc;

    if (is_case2) {
        float dsto_add = total_storage - levee_base_storage;
        float dwth_add = levee_distance + river_width;
        f_dph = levee_base_height + dsto_add / dwth_add / river_length;
        r_sto = river_max_storage + river_length * river_width * f_dph;
        r_dph = r_sto / river_length / river_width;
        f_sto = max(total_storage - r_sto, 0.0f);
        p_sto = 0.0f;
        p_dph = 0.0f;
        f_frc = levee_fraction;
    } else if (is_case3) {
        float dsto_add_B = total_storage - dsto_fil_B;
        float term_B = dwth_fil_B * dwth_fil_B + 2.0f * dsto_add_B / river_length / (gradient_B + 1e-9f);
        float dwth_add_B = -dwth_fil_B + sqrt(max(term_B, 0.0f));
        float ddph_add_B = dwth_add_B * gradient_B;

        float p_dph_B, f_frc_B;
        if (found_B) {
            p_dph_B = levee_base_height + ddph_fil_B + ddph_add_B;
            f_frc_B = (dwth_fil_B + levee_distance) / (dwth_inc * (float)NUM_FLOOD_LEVELS);
        } else {
            float ddph_add_extra = dsto_add_B / (dwth_fil_B * river_length + 1e-9f);
            p_dph_B = levee_base_height + ddph_fil_B + ddph_add_extra;
            f_frc_B = 1.0f;
        }

        f_dph = levee_crown_height;
        r_sto = river_max_storage + river_length * river_width * f_dph;
        r_dph = r_sto / river_length / river_width;
        f_sto = max(s_top - r_sto, 0.0f);
        p_sto = max(total_storage - r_sto - f_sto, 0.0f);
        p_dph = p_dph_B;
        f_frc = clamp(f_frc_B, 0.0f, 1.0f);
    } else if (is_case4) {
        f_dph = flood_depth_curr;
        r_sto = river_storage_curr;
        float dsto_add = (f_dph - levee_crown_height) * (levee_distance + river_width) * river_length;
        f_sto = max(s_top + dsto_add - r_sto, 0.0f);
        p_sto = max(total_storage - r_sto - f_sto, 0.0f);
        p_dph = f_dph;
        r_dph = river_depth_buf[ci];
        f_frc = flood_fraction_buf[ci];
    } else {
        r_sto = river_storage_curr;
        f_sto = flood_storage_curr;
        p_sto = 0.0f;
        r_dph = river_depth_buf[ci];
        f_dph = flood_depth_curr;
        p_dph = 0.0f;
        f_frc = flood_fraction_buf[ci];
    }

    // Log: write to packed log_sums
    float total_storage_stage_new = r_sto + f_sto + p_sto;
    atomic_add_float(&log_sums_buf[6  * lbs + current_step], total_storage_stage_new * 1e-9f);
    atomic_add_float(&log_sums_buf[8  * lbs + current_step], r_sto * 1e-9f);
    atomic_add_float(&log_sums_buf[9  * lbs + current_step], f_sto * 1e-9f);
    atomic_add_float(&log_sums_buf[10 * lbs + current_step], f_frc * catchment_area * 1e-9f);
    atomic_add_float(&log_sums_buf[7  * lbs + current_step], (total_storage_stage_new - total_storage) * 1e-9f);

    // Store results
    river_storage_buf[ci]     = r_sto;
    flood_storage_buf[ci]     = f_sto;
    protected_storage_buf[ci] = p_sto;
    river_depth_buf[ci]       = r_dph;
    flood_depth_buf[ci]       = f_dph;
    protected_depth_buf[ci]   = p_dph;
    flood_fraction_buf[ci]    = f_frc;
}
