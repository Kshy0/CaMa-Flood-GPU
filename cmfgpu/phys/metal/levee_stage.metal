// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

kernel void compute_levee_stage(
    device int*   levee_catchment_idx_buf   [[buffer(0)]],
    device float* river_storage_buf         [[buffer(1)]],
    device float* flood_storage_buf         [[buffer(2)]],
    device float* protected_storage_buf     [[buffer(3)]],
    device float* river_depth_buf           [[buffer(4)]],
    device float* flood_depth_buf           [[buffer(5)]],
    device float* protected_depth_buf       [[buffer(6)]],
    device float* river_height_buf          [[buffer(7)]],
    device float* flood_depth_table_buf     [[buffer(8)]],
    device float* catchment_area_buf        [[buffer(9)]],
    device float* river_width_buf           [[buffer(10)]],
    device float* river_length_buf          [[buffer(11)]],
    device float* levee_base_height_buf     [[buffer(12)]],
    device float* levee_crown_height_buf    [[buffer(13)]],
    device float* levee_fraction_buf        [[buffer(14)]],
    device float* flood_fraction_buf        [[buffer(15)]],
    constant int& num_levees                [[buffer(16)]],
    uint idx [[thread_position_in_grid]]
)
{
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
